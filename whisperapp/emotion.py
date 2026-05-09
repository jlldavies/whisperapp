"""Emotion analysis: run enabled emotion models and insert markers into result.

Inserts pseudo-segments (is_emotion_marker=True) into transcription results
for txt/srt/vtt output only. Pure function — never mutates the input dict.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisperapp.config import Config
    from whisperapp.emotion_registry import ModelManager

logger = logging.getLogger(__name__)


def _make_marker(start: float, end: float, text: str) -> dict:
    return {"text": text, "start": start, "end": end, "is_emotion_marker": True}


def _combine_classification_results(
    model_outputs: list[list[dict]],
) -> dict[int, tuple[str, float]]:
    """Combine outputs from multiple classification models.

    If all models agree on a label for a segment → use that label.
    If split → use the highest-confidence label across all models.

    Returns: {segment_idx: (label, confidence)}
    """
    # Collect all results by segment
    by_segment: dict[int, list[tuple[str, float]]] = {}
    for model_result in model_outputs:
        for r in model_result:
            idx = r["segment_idx"]
            label = r["label"]
            conf = r["confidence"]
            by_segment.setdefault(idx, []).append((label, conf))

    combined: dict[int, tuple[str, float]] = {}
    for idx, predictions in by_segment.items():
        labels = [p[0] for p in predictions]
        if len(set(labels)) == 1:
            # All agree — use mean confidence
            avg_conf = sum(p[1] for p in predictions) / len(predictions)
            combined[idx] = (labels[0], avg_conf)
        else:
            # Split — use highest confidence
            best = max(predictions, key=lambda p: p[1])
            combined[idx] = best

    return combined


def _dimensional_to_markers(
    dim_results: list[dict],
    arousal_threshold: float = 0.7,
    valence_threshold: float = 0.3,
) -> dict[int, list[str]]:
    """Convert dimensional model results to label contributions."""
    contributions: dict[int, list[str]] = {}
    for r in dim_results:
        idx = r["segment_idx"]
        labels = []
        if r.get("arousal", 0) > arousal_threshold:
            labels.append("intense")
        if r.get("valence", 1) < valence_threshold:
            labels.append("negative tone")
        if labels:
            contributions[idx] = labels
    return contributions


def analyse_emotions(
    audio_path: str,
    result: dict,
    config: "Config",
    manager: "ModelManager",
) -> dict:
    """Run enabled emotion models and insert markers into result.

    Returns a copy of result with emotion marker pseudo-segments inserted.
    The input result is never mutated.
    """
    if not config.emotion_enabled:
        return result

    model_ids: list[str] = list(config.emotion_model_ids or [])
    if not model_ids:
        return result

    segments = result.get("segments", [])
    if not segments:
        return result

    from whisperapp.emotion_registry import EMOTION_MODELS, _REGISTRY_BY_ID

    # Separate classification from dimensional models
    classification_outputs: list[list[dict]] = []
    dimensional_outputs: list[dict] = []

    for model_id in model_ids:
        status = manager.get_status(model_id)
        if not status.get("downloaded"):
            logger.info("Emotion model %s not downloaded — skipping", model_id)
            continue

        model_info = _REGISTRY_BY_ID.get(model_id)
        if not model_info:
            continue

        try:
            infer_results = manager.infer(model_id, audio_path, segments)
        except Exception as exc:
            logger.warning("Emotion inference failed for %s: %s", model_id, exc)
            continue

        if model_info["type"] == "dimensional":
            dimensional_outputs.extend(infer_results)
        else:
            classification_outputs.append(infer_results)

    # Combine classification results
    conf_threshold = config.emotion_confidence_threshold
    combined_class: dict[int, tuple[str, float]] = {}
    if classification_outputs:
        combined_class = _combine_classification_results(classification_outputs)
        # Suppress below confidence threshold
        combined_class = {
            idx: (label, conf)
            for idx, (label, conf) in combined_class.items()
            if conf >= conf_threshold
        }

    # Parse dimensional results
    dim_contributions: dict[int, list[str]] = {}
    if dimensional_outputs:
        dim_contributions = _dimensional_to_markers(dimensional_outputs)

    # Build set of segment indices that need markers
    all_indices: set[int] = set(combined_class.keys()) | set(dim_contributions.keys())

    if not all_indices and not config.emotion_combine_with_ai:
        return result

    # Optionally combine with AI
    if config.emotion_combine_with_ai:
        try:
            combined_class, dim_contributions = _ai_synthesise(
                segments, combined_class, dim_contributions, config
            )
            all_indices = set(combined_class.keys()) | set(dim_contributions.keys())
        except Exception as exc:
            logger.warning("AI synthesis failed — using model outputs directly: %s", exc)

    # Build marker map: segment_idx → list of marker texts
    marker_map: dict[int, list[str]] = {}
    for idx in all_indices:
        labels: list[str] = []
        if idx in combined_class:
            label, _ = combined_class[idx]
            labels.append(f"[{label}]")
        if idx in dim_contributions:
            for dl in dim_contributions[idx]:
                labels.append(f"[{dl}]")
        if labels:
            marker_map[idx] = labels

    if not marker_map:
        return result

    # Insert markers after their corresponding segments
    new_segments: list[dict] = []
    for i, seg in enumerate(segments):
        new_segments.append(seg)
        if i in marker_map:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            for label_text in marker_map[i]:
                new_segments.append(_make_marker(start, end, label_text))

    new_result = dict(result)
    new_result["segments"] = new_segments
    return new_result


def _ai_synthesise(
    segments: list[dict],
    combined_class: dict[int, tuple[str, float]],
    dim_contributions: dict[int, list[str]],
    config: "Config",
) -> tuple[dict[int, tuple[str, float]], dict[int, list[str]]]:
    """Use configured AI provider to synthesise conflicting model outputs."""
    try:
        from whisperapp.ai import make_provider
        ai = make_provider(config.ai_provider, config.ai_api_key,
                           config.ai_model, config.ai_base_url)
        if not ai.is_available():
            return combined_class, dim_contributions
    except Exception:
        return combined_class, dim_contributions

    # Build a prompt summarising model disagreements and let AI decide
    # This is a best-effort synthesis — we return original on any failure
    prompt_lines = ["You are helping synthesise emotion detection results from multiple models."]
    prompt_lines.append("For each segment below, suggest the best emotion label or confirm the existing one.")
    prompt_lines.append("Return only a JSON object mapping segment index (string) to label string.")
    prompt_lines.append("")

    for idx in sorted(set(combined_class.keys()) | set(dim_contributions.keys())):
        seg = segments[idx] if idx < len(segments) else {}
        text = seg.get("text", "").strip()
        class_label = combined_class.get(idx, ("",))[0]
        dim_labels = dim_contributions.get(idx, [])
        prompt_lines.append(f"Segment {idx}: \"{text}\"")
        if class_label:
            prompt_lines.append(f"  Classification models: {class_label}")
        if dim_labels:
            prompt_lines.append(f"  Dimensional model: {', '.join(dim_labels)}")

    try:
        # Use a minimal AI call — providers expose a generic chat method
        if hasattr(ai, "chat"):
            response = ai.chat("\n".join(prompt_lines))
        else:
            return combined_class, dim_contributions

        import json
        # Extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start < 0 or end <= start:
            return combined_class, dim_contributions

        mapping = json.loads(response[start:end])
        new_class = dict(combined_class)
        for k, v in mapping.items():
            try:
                idx = int(k)
                # Preserve confidence if available
                conf = combined_class.get(idx, (None, 0.9))[1]
                new_class[idx] = (str(v).lower(), conf)
            except (ValueError, TypeError):
                pass
        return new_class, dim_contributions

    except Exception as exc:
        logger.debug("AI synthesis parse failed: %s", exc)
        return combined_class, dim_contributions
