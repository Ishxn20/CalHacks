# Training runbook (for an agent on the GPU box)

You are running the training pipeline for the instrument detector in `vision/`, then reporting results so a human can pull them. Do the steps in order. Do not redesign anything: the goal is a faithful run and an honest report.

## Context

- The surgical-scrub robot hears a tool name, then needs an image of that exact tool from a tray camera. This detector finds it.
- Classes = every instrument id in `config/tray.yaml` (16 today, order matters). Do not edit `config/tray.yaml`.
- Training data is synthetic: `scrub_vision/render_dataset.py` renders top-down scenes from the STLs in `vision/assets/stl/` (all tools one turquoise colour, shading is the depth cue). `check_data.py` validates it and builds `vision/data/clean/`.
- **Overfit on purpose.** There is no early stopping (`patience=0`) and no flips. Do not add either. Train all requested epochs.
- The hard part is telling similar tools apart (scalpel #10/#11/#15/#20, needle holders, forceps). The report must show that, not just mAP.

## Steps

All commands from `vision/` on the branch you were given.

1. **Environment** (Python 3.12, uv):
   ```bash
   uv venv --python 3.12 && uv pip install -e ../arm -e ".[dev]"
   .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   If CUDA is `False`, install the CUDA build of torch that matches this machine (check `nvidia-smi`; on aarch64 use NVIDIA's wheels), and re-check. **Do not train on CPU.** If you cannot get CUDA working, stop and report why in `NOTES.md`.
2. **Tests:** `.venv/bin/python -m pytest -q` must pass.
3. **Render data** (about 8000 train / 1000 val; uses all CPU cores, no GPU):
   ```bash
   .venv/bin/python -m scrub_vision.render_dataset --train 8000 --val 1000
   ```
4. **Validate:** `.venv/bin/python -m scrub_vision.check_data`. It must end with `clean dataset: ...`. Roughly 1% rejected is normal. If it exits with `too few clean examples` or a class mismatch, stop and report.
5. **Train.** Run it in the background with a log (`nohup ... > train.log 2>&1 &`) and read the log every few minutes instead of blocking:
   ```bash
   .venv/bin/python -m scrub_vision.train --model yolo11s.pt --imgsz 640 --batch 32 --epochs 100 --name gpu
   ```
   - Out of memory: halve `--batch` and note it in `NOTES.md`. Too slow or idle GPU: raise `--batch`/`--workers`.
   - If the process dies, `--resume --name gpu` continues from the last epoch.
   - Ultralytics downloads `yolo11s.pt` on first use, so it needs internet.
6. **Results are packaged automatically** at the end of `train`, into `vision/results/gpu/` (`results.json`, `best.pt`, `results.csv`, `args.yaml`, `confusion_matrix_normalized.png`, `results.png`). Check `results.json` exists.

## Report

Write `vision/results/gpu/NOTES.md` using exactly this template (plain text, no extra sections):

```
# Training report: gpu
Status: DONE | FAILED | PARTIAL
Commit: <git rev-parse HEAD>
Machine: <GPU name from nvidia-smi>, <CPU cores>, <RAM>
Commands run: <the exact train command, plus the render command and its counts>
Wall time: render <m>, train <h>
Epochs completed: <n of 100>
mAP50 / mAP50-95: <x> / <y>
Confusion check: PASS | FAIL (<list "true -> pred rate" for each failure, or "none">)
Weakest 3 classes (by mAP50): <class mAP50>, <class mAP50>, <class mAP50>
Scalpel group: <for scalpel_10/11/15/20, each one's most-confused class and rate>
Deviations: <batch changes, OOM, restarts, resumes, code or version changes, anything not in this runbook; "none" if none>
Problems: <errors, warnings you could not resolve, anything suspicious; "none" if none>
```

`results.json` is the source of truth for numbers. Copy them, do not retype from memory. If the run FAILED or is PARTIAL, still write `NOTES.md` and say exactly where it stopped and the last 20 lines of `train.log`.

## Push back

Do not push to `main` or to the branch you were given.

```bash
git checkout -b results/gpu
git add vision/results/gpu
git commit -m "results: detector training on GPU box"
git push -u origin results/gpu
```

Only `vision/results/gpu/` should be in that commit. Never commit `vision/data/`, `vision/runs/`, `.venv/`, or `*.log` (all gitignored). If you had to change any code to make the run work, do not commit it. Put the diff in `NOTES.md` under Deviations instead. `best.pt` is a few MB for `yolo11n` and about 20 MB for `yolo11s`, which is fine to commit.
