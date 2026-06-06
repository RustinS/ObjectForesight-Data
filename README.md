# ObjectForesight — Data Curation Pipeline

**The 3D data-curation pipeline that turns raw [EPIC-KITCHENS-100](https://epic-kitchens.github.io) videos into per-object 6-DoF trajectory extractions.**

This is the data pipeline behind [**ObjectForesight**](https://arxiv.org/abs/2601.05237). It produces the extractions released as the [`raivn/ObjectForesight-EPIC`](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC) dataset, which trains the model in [**RustinS/ObjectForesight**](https://github.com/RustinS/ObjectForesight).

## Pipeline

Run `step1`→`step10` in order. Each step is **sharded** (`--num_shards N --shard_idx i`) so it parallelizes across many GPUs, and runs in the conda env shown below (set up in [Installation](#installation)).

| Step | Script | What it does | Component | Env |
|---|---|---|---|---|
| 1 | `step1_split.py` | Split EPIC videos into action clips (by `narration_id`) | ffmpeg/decord | `fpose` |
| 2 | `step2_egohos.py` | Hand & object segmentation | [EgoHOS](https://github.com/owenzlz/EgoHOS) | `egohos` |
| 3 | `step3_filtering.py` | Drop low-quality sequences | — | `fpose` |
| 4 | `step4_sam.py` | Propagate/refine object masks | [SAM 2](https://github.com/facebookresearch/sam2) | `sam` |
| 5 | `step5_obj_filter.py` | Verify object tracks (InternVL); write `moved_by_hand.txt` | [InternVL](https://github.com/OpenGVLab/InternVL) | `fpose` |
| 6 | `step6_frame_filter.py` | Filter crops/masks by visibility (InternVL yes/partial/no) | InternVL | `fpose` |
| 7 | `step7_vas.py` | Amodal mask completion | [Diffusion-VAS](diffusion-vas/) | `vas` |
| 8 | `step8_trellis.py` | Image-to-3D object mesh | [TRELLIS](https://github.com/microsoft/TRELLIS) | `trellis` |
| 9 | `step9_spatracker.py` | 3D point tracks + metric depth → `spatracker.npz` | [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) | `fpose` |
| 10 | `step10_fpose.py` | 6-DoF object pose tracking → `foundationpose10/` | [FoundationPose](https://github.com/NVlabs/FoundationPose) | `fpose` |

`utils.py` holds shared helpers used across the steps. Default roots are `./EPIC-KITCHENS` (source videos) and `./manip_data` (clips + per-object outputs); override per script via `--help`.

## Installation

Each stage wraps a separate third-party system, bundled in this repo under its own subdirectory with its original install files, weight downloaders, and license. The pipeline uses **five conda environments** — install the ones you need. A recent CUDA toolkit (11/12) and a GPU are required throughout.

**`egohos` — EgoHOS (step 2)**
```bash
cd EgoHOS
conda create -n egohos python=3.9 -y && conda activate egohos
pip install -r requirements.txt          # includes mmsegmentation
bash download_checkpoints.sh
cd ..
```

**`sam` — SAM 2 (step 4)**
```bash
cd sam2
conda create -n sam python=3.11 -y && conda activate sam
pip install -e .
# download the SAM 2.1 checkpoints — see sam2/INSTALL.md
cd ..
```

**`vas` — Diffusion-VAS (step 7)**
```bash
cd diffusion-vas
conda create -n vas python=3.10 -y && conda activate vas
pip install -r requirements.txt
# download the Diffusion-VAS checkpoints — see diffusion-vas/README.md
cd ..
```

**`trellis` — TRELLIS (step 8)**
```bash
cd trellis
. ./setup.sh --new-env --basic           # creates the 'trellis' env + installs deps (see ./setup.sh --help)
cd ..
```

**`fpose` — FoundationPose · SpaTrackerV2 · InternVL · video IO (steps 1, 3, 5, 6, 9, 10)**
```bash
cd FoundationPose
bash build_all_conda.sh                   # builds its conda env + CUDA extensions (see FoundationPose/readme.md)
conda activate fpose                      # use the env name the build script creates
pip install -r ../SpaTrackerV2/requirements.txt          # SpaTrackerV2 (step 9)
pip install transformers accelerate einops timm decord   # InternVL filtering (5–6) + clip splitting (1)
cd ..
```

Each component downloads its own model weights on setup or first use — follow the install/README inside its subdirectory.

## Running the pipeline

1. **Get EPIC-KITCHENS-100** (agree to its [terms](https://epic-kitchens.github.io)) and place the source videos under `./EPIC-KITCHENS`.
2. **Run the steps in order**, activating the matching env and sharding across jobs. Every script takes `--num_shards`/`--shard_idx` and `--help`:

```bash
conda activate fpose  && python step1_split.py    --video_root ./EPIC-KITCHENS --out_root ./manip_data
conda activate egohos && python step2_egohos.py   --data_root  ./manip_data --num_shards 64 --shard_idx 0
conda activate fpose  && python step3_filtering.py --data_root  ./manip_data
conda activate sam    && python step4_sam.py      --video_root ./manip_data --num_shards 64 --shard_idx 0
# steps 5–10 likewise (see each script's --help for its flags)
```

Shard a step across N jobs by launching it once per `--shard_idx` in `[0, N)`.

## Output

Per action clip (`<narration_id>/`): EgoHOS + SAM 2 + amodal masks, a TRELLIS mesh per object, SpaTrackerV2 depth & 3D tracks (`spatracker.npz`), and FoundationPose 6-DoF poses (`foundationpose10/`). The dataloader in [RustinS/ObjectForesight](https://github.com/RustinS/ObjectForesight) windows these into training trajectories; the cleaned, packaged result is [`raivn/ObjectForesight-EPIC`](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC).

## Acknowledgments

This pipeline builds on [EgoHOS](https://github.com/owenzlz/EgoHOS), [SAM 2](https://github.com/facebookresearch/sam2), [Diffusion-VAS](diffusion-vas/), [TRELLIS](https://github.com/microsoft/TRELLIS), [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2), [FoundationPose](https://github.com/NVlabs/FoundationPose), and [InternVL](https://github.com/OpenGVLab/InternVL) (the [pipeline table](#pipeline) shows which stage uses each), bundled here under their respective licenses. The bundled copies may include minor local modifications for pipeline integration.

## License

Pipeline code (`step*.py`, `utils.py`) is for non-commercial research use, consistent with EPIC-KITCHENS-100 (CC BY-NC 4.0). Each bundled component keeps its own license — see its subdirectory.

## Citation

```bibtex
@article{soraki2026objectforesight,
  title   = {ObjectForesight: Predicting Future 3D Object Trajectories from Human Videos},
  author  = {Soraki, Rustin and Bharadhwaj, Homanga and Farhadi, Ali and Mottaghi, Roozbeh},
  journal = {arXiv preprint arXiv:2601.05237},
  year    = {2026}
}
```
Please also cite EPIC-KITCHENS-100 and each component (EgoHOS, SAM 2, Diffusion-VAS, TRELLIS, SpaTrackerV2, FoundationPose, InternVL).
