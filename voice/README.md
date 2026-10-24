# scrub_voice: speech to instrument code

Push-to-talk mic, local Whisper, then the phrase is matched against the shared instrument vocabulary in `config/tray.yaml`. Each result overwrites `last_command.json` at the repo root (gitignored).

```bash
cd voice
uv venv --python 3.12 && uv pip install -e ../arm -e ".[dev]"
.venv/bin/python -m scrub_voice.listen                          # Enter, speak, Enter. First run downloads the model.
.venv/bin/python -m scrub_voice.listen --text "hand me a scalpel"   # no mic
.venv/bin/python -m pytest
```

`instrument` is the instrument id from `tray.yaml`, the shared key for voice, arm and CV:

```json
{"ok": true, "heard": "hand me a scalpel", "instrument": "scalpel_11", "category": "scalpel",
 "slot": "A4", "orientation": null, "error": null, "ts": 1793000000.0}
```

On failure `ok` is false, `instrument` is null and `error` says why. Only the tool is parsed; suture, size and orientation aren't. If Whisper mishears, try `--model small.en`.
