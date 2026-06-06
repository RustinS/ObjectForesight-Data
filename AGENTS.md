# ObjectForesight — AGENTS.md

## Purpose

This file defines project-specific AI instructions for the **EPIC-Kitchens 3D manipulation data processing pipeline** in this repository. The pipeline transforms raw EPIC-Kitchens videos into 6D object pose trajectory data for downstream prediction tasks.

This is a **multi-stage data processing project** with several subprojects:
- `EgoHOS/` — Hand-object segmentation
- `sam2/` — Segment Anything v2
- `FoundationPose/` — 6D object pose estimation
- `SpaTrackerV2/` — 3D point tracking
- `diffusion-vas/` — Video affordance selection
- `trellis/` — 3D reconstruction
- `future_pose_pred/` — Downstream pose prediction model (see its own `AGENTS.md`)

---

## Tool calls and environment

- **VERY IMPORTANT!!** When using the bash tool, set a timeout of **AT LEAST 5 minutes**.
- For any command that involves Python, first set up the environment with the **correct conda env for the step**:

  ```bash
  source ~/.bashrc
  module restore
  conda activate <env_name>
  ```

### Conda environments by step

| Step(s) | Conda Environment | Notes |
|---------|-------------------|-------|
| 1 (split) | `fpose` | General video processing |
| 2 (egohos) | `egohos` | Requires mmsegmentation |
| 3 (filtering) | `fpose` | Lightweight filtering |
| 4 (sam) | `sam` | SAM2 segmentation |
| 5 (obj_filter) | `fpose` | Lightweight filtering |
| 6, 8 (trellis) | `trellis` | 3D mesh generation |
| 7 (vas) | `vas` | Diffusion-based affordance |
| 9 (spatracker) | `fpose` | 3D point tracking |
| 10 (fpose) | `fpose` | FoundationPose 6D tracking |
| future_pose_pred | `futurepos` | Downstream prediction model |

- Assume all commands are run from the **repo root** (`/gpfs/scrubbed/rustin/3dmanip`) unless explicitly stated otherwise.
- For GPU-heavy processing steps, be mindful of resource allocation and prefer running on compute nodes, not login nodes.

---

## Overall structure

### Pipeline scripts (run in order)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_split.py` | Split EPIC videos into action segments (by narration_id) |
| 2 | `step2_egohos.py` | Hand and object segmentation using EgoHOS |
| 3 | `step3_filtering.py` | Filter sequences based on quality criteria |
| 4 | `step4_sam.py` | Refine object masks with SAM2 |
| 5 | `step5_obj_filter.py` | Filter objects by area, consistency, etc. |
| 6 | `step6_trellis.py` | Generate 3D meshes with Trellis |
| 7 | `step7_vas.py` | Video affordance selection (diffusion-vas) |
| 8 | `step8_trellis.py` | Additional Trellis processing |
| 9 | `step9_spatracker.py` | 3D point tracking with SpaTracker |
| 10 | `step10_fpose.py` | 6D pose tracking with FoundationPose |

### Key directories

- `/gpfs/scrubbed/rustin/manip_data/` — Processed action segments and intermediate outputs
- `outputs/` — Logs, checkpoints, and artifacts
- `csv_shards/` — Sharded CSVs for distributed processing
- `EPIC_100.csv` — Main EPIC annotations file

### Sharding pattern

Most scripts support multi-node parallelism via:
```bash
python step<N>_<name>.py --num_shards N --shard_idx i
```
Sharding uses `stable_int_hash(video_id) % num_shards == shard_idx` for deterministic partitioning.

---

## Code conventions

- All Python code must be formatted with **ruff** with line width **175**. Always format after editing.
- Keep functions focused and composable:
  - Avoid monolithic functions that mix data I/O, model logic, and visualization.
  - Prefer small helpers wired together in the main entrypoints.
- Use `sys.path.append("<subproject>")` for subproject imports (e.g., `sys.path.append("FoundationPose")`).
- Keep imports clean:
  - Remove unused imports.
  - Group standard library, third-party, and local imports logically.

When modifying existing functionality:

- Prefer local, minimal diffs that preserve behavior unless explicitly refactoring.
- Clearly annotate any behavioral changes in comments or docstrings.
- Test on a small subset before running at scale.

---

## Subproject-specific notes

### EgoHOS (`EgoHOS/`)
- Requires `mmsegmentation` submodule
- Uses segmentation refinement for mask cleanup

### FoundationPose (`FoundationPose/`)
- Requires EGL backend: `os.environ["PYOPENGL_PLATFORM"] = "egl"`
- Needs CUDA architecture set: `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"`

### future_pose_pred (`future_pose_pred/`)
- **Has its own `AGENTS.md`** with specific instructions for training/eval
- Uses Hydra configs in `future_pose_pred/conf/`
- Activate with `conda activate futurepos` (different environment!)

---

## Miscellaneous best practices

- **Avoid destructive shell operations:**
  - Do not suggest broad `rm -rf` commands (especially on `/`, `~`, or `outputs/` without filters).
- **File paths:**
  - Use absolute paths in scripts; data lives under `/gpfs/scrubbed/rustin/`.
- **Video processing:**
  - Use `decord` for efficient video reading.
  - Fall back to OpenCV if decord fails on corrupted videos.
- **Logging:**
  - Use `utils.rprint` for timestamped console output.
- **Memory management:**
  - Scripts have memory budgeting (`--mem_per_worker_mb`); respect these limits.
- **Debugging:**
  - For shape mismatches, print tensor/array shapes at the failure point.
  - For NaNs, check data normalization and model inputs first.
- **Visualization:**
  - Keep plotting helpers in `utils.py` or dedicated viz modules.
  - Use `viser_viz.py` for 3D visualization.

---

## Example commands

```bash
# Step 1: Split videos (8 workers, shard 0 of 48)
python step1_split.py --workers 8 --num_shards 48 --shard_idx 0

# Step 2: EgoHOS segmentation
python step2_egohos.py --num_shards 48 --shard_idx 0

# Step 10: FoundationPose tracking
python step10_fpose.py --num_shards 48 --shard_idx 0
```

---

The goal of these instructions is to keep the pipeline consistent, safe, and reproducible while processing EPIC-Kitchens data at scale.
