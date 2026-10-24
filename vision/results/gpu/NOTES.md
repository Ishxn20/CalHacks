# Training report: gpu
Status: DONE
Commit: 8b9b053ff4d7b47345799433542ce2ed9bc07d72 (plus uncommitted code changes, see Deviations)
Machine: NVIDIA GB200 (hpc-rack-2-2, aarch64, Slurm job 13445), 32 CPUs and 128 GB RAM allocated, 1 GPU
Commands run: render: python -m scrub_vision.render_dataset --train 8000 --val 1000 (9000 images, 48669 labeled tools; check_data kept 8956, rejected 44 "image too blurry"; clean split train 7959 / val 997). train: python -m scrub_vision.train --model yolo11s.pt --imgsz 640 --batch 32 --workers 16 --epochs 100 --name gpu
Wall time: render 34s, train 54m (job total 55m18s)
Epochs completed: 100 of 100
mAP50 / mAP50-95: 0.9948 / 0.9935
Confusion check: PASS (none)
Weakest 3 classes (by mAP50): scalpel_10 0.994, scalpel_11 0.994, scalpel_15 0.995 (all other classes are 0.995)
Scalpel group: scalpel_10 -> scalpel_15 0.6%; scalpel_11 -> scalpel_15 0.9%; scalpel_15 -> scalpel_11 1.0%; scalpel_20 -> none (0.0%)
Deviations: (1) Code changes, uncommitted, that are not in the runbook. render_dataset.py: apparent tool size no longer depends on camera height H (was scale = U/(H/450), now scale = U) and the range is narrowed from (0.9, 1.6) to (1.1, 1.3), so tool length can separate lookalike tools (DeBakey/Fine vascular/Resano forceps, Mayo-Hegar short/long). train.py: YOLO scale=0.1 added (default 0.5). find.py: default --weights now picks the first existing of runs/tools, runs/gpu, results/gpu. Diff:
  render_dataset.py  -def render_scene(pool, ids, rng, scale_range=(0.9, 1.6)):  +def render_scene(pool, ids, rng, scale_range=(1.1, 1.3)):
  render_dataset.py  -scale = rng.uniform(*scale_range) / (H / 450)  +scale = rng.uniform(*scale_range)
  train.py           -fliplr=0.0, flipud=0.0, patience=0,  +fliplr=0.0, flipud=0.0, scale=0.1, patience=0,
  find.py            +def default_weights() (see file); --weights default uses it
  (2) Environment: no uv on the cluster, used python3.12 -m venv + pip. torch 2.11.0+cu128 from download.pytorch.org (aarch64). Replaced opencv-python with opencv-python-headless because the compute nodes have no libGL. (3) Run through Slurm (train.sbatch, untracked), not directly. (4) Batch kept at 32; no OOM, no restarts, no resumes.
Problems: none that blocked the run. The GPU was mostly idle (about 17% utilization in one sample) because the main Python process is CPU-bound at batch 32; --batch 128 or more would likely be faster. mAP is saturated and measured on synthetic val renders from the same generator, so it says nothing yet about real photos of the printed tools. In results.json the "confusions" rows with rate 0.0 name needle_holder_castroviejo as the "most confused" class only because argmax of an all-zero row is index 0; read them as no confusion.
