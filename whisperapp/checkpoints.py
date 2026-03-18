import json
import shutil
from pathlib import Path

STAGES = ["transcription", "alignment", "diarization", "speaker_review", "saving"]

class CheckpointManager:
    def __init__(self, output_path, job_id):
        self.job_id = job_id
        self.dir = Path(output_path) / ".whisperapp_partials" / job_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage) -> Path:
        return self.dir / f"{stage}.json"

    def save(self, stage: str, data: dict):
        self._path(stage).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, stage: str) -> dict:
        p = self._path(stage)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def has(self, stage: str) -> bool:
        return self._path(stage).exists()

    def last_completed_stage(self) -> str | None:
        for stage in reversed(STAGES):
            if self.has(stage):
                return stage
        return None

    def remaining_stages(self) -> list[str]:
        last = self.last_completed_stage()
        if last is None:
            return STAGES.copy()
        idx = STAGES.index(last)
        return STAGES[idx + 1:]

    def cleanup(self):
        if self.dir.exists():
            shutil.rmtree(self.dir)
