---
description: AI-driven speaker review for a WhisperApp job — read the transcript, propose names, confirm with the user, post the labels back.
---

You are driving the speaker review for a WhisperApp transcription job. The
WhisperApp daemon is running on `127.0.0.1:7861` and exposes:

  - `GET  /jobs?status=speaker_review`         — list jobs awaiting review
  - `GET  /jobs/<id>`                          — job metadata (file_path, output_path, model, etc.)
  - `GET  /jobs/<id>/segments?offset=&limit=`  — paged diarized transcript (each segment has start/end/speaker/text)
  - `GET  /jobs/<id>/speakers`                 — per-speaker representative snippets with audio timestamps
  - `GET  /jobs/<id>/audio`                    — the source audio file (Range-able; for verification only — usually unnecessary)
  - `POST /jobs/<id>/speakers`                 — body `{"names": {"SPEAKER_00": "Alice", ...}}` → finalises the job. Empty / missing names keep the SPEAKER_XX placeholder.

The CLI shortcut is `whisperapp transcript <id> --all` — same thing as paging
`/segments`, just easier to read.

## Argument

`$ARGUMENTS` is either:
  - a job UUID (or unique 8-char prefix), OR
  - empty — in which case **list the speaker_review jobs and ask the user which to pick**.

## Workflow

1. **Resolve the job.** If the user gave a prefix, expand it by hitting `/jobs?status=speaker_review` and matching. Confirm the file_name in one short line so the user knows which one we're on.

2. **Read the full transcript.** `whisperapp transcript <id> --all` → reason about who's speaking from conversational cues: names mentioned, role markers ("hi this is Alice"), turn-taking, topical cues, the meeting context, any prior wiki / notes the user has shared this session.

3. **Propose names.** For each `SPEAKER_XX`, give the user:
     - The proposed name
     - One-sentence justification grounded in actual lines from the transcript (quote the line, mention the timestamp)
     - Confidence: high / medium / low
   If a speaker is unidentifiable (e.g. an automated phone tree), say so and suggest a descriptive label like "Automated", "Receptionist", or leave it empty (which keeps SPEAKER_XX).

4. **Wait for the user's reply.** They will either:
     - Confirm the lot ("yes" / "go" / "post it")
     - Correct individual entries ("SPEAKER_01 is Bob, not Alice")
     - Ask for a snippet to be played back to verify a voice — fetch `/jobs/<id>/speakers` for the timestamps, tell them where to listen in the WhisperApp UI

5. **Post.** When the user confirms, POST `{"names": {...}}` to `/jobs/<id>/speakers`. Read back the daemon's response — a 200 means the job moves to `done` and the txt/srt/vtt/json get written. A 409 means someone else already finished it (just inform the user; nothing to do).

## Style

  - Lead with the file name and proposed names — don't preamble.
  - Keep justifications short; quote the line that gave it away.
  - If the transcript is very long, scan top-to-bottom and stop reasoning once you have high-confidence labels — don't analyse every utterance for its own sake.
  - Don't pollute the user's name cache with placeholders. Leave a speaker as "" rather than guessing if confidence is low.

## Notes

  - You do NOT need WhisperApp's internal `ai_provider` configured for any of this — you ARE the AI provider. Skip `whisperapp identify-speakers`.
  - Speaker names the user has used before live in their browser's localStorage (`wa-speaker-names`), not on the server, so you can't pre-load history. The user can hint via the original `$ARGUMENTS` if they want ("$ARGUMENTS — call with the Davies family" etc.).
