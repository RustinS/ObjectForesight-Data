# ObjectForesight — Data Curation Pipeline

**The 3D data-curation pipeline that turns raw [EPIC-KITCHENS-100](https://epic-kitchens.github.io) videos into per-object 6-DoF trajectory extractions.**

This is the data pipeline behind [**ObjectForesight**](https://arxiv.org/abs/2601.05237). It produces the extractions released as the [`raivn/ObjectForesight-EPIC`](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC) dataset, which trains the model in [**RustinS/ObjectForesight**](https://github.com/RustinS/ObjectForesight).

> ⚠️ **Heavy, multi-environment pipeline.** Each stage wraps a separate third-party system (EgoHOS, SAM 2, Diffusion-VAS, TRELLIS, SpaTrackerV2, FoundationPose, InternVL), each with **its own environment, model weights, and license** (several **non-commercial**). This is *not* a one-command install — set up each component independently. The components are vendored here for convenience; see each subdirectory's upstream project for installation, weights, and license terms.

## Pipeline (run in order)

Each step is **sharded** (process a subset of clips per job) so the pipeline parallelizes across many GPUs.

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

`utils.py` holds shared helpers used across the steps. Default input/output roots are `./EPIC-KITCHENS` (source videos) and `./manip_data` (clips + per-object outputs); override per script via `--help`.

## Setup

1. **Get EPIC-KITCHENS-100** and agree to its [terms](https://epic-kitchens.github.io).
2. **Install each component** in its own conda env (versions/weights per upstream). Suggested envs (one per stage group):

   ```bash
   conda create -n egohos python=3.9   # then install EgoHOS + mmsegmentation
   conda create -n sam    python=3.11  # SAM 2
   conda create -n vas    python=3.10  # Diffusion-VAS
   conda create -n trellis python=3.10 # TRELLIS
   conda create -n fpose  python=3.9   # FoundationPose / SpaTrackerV2 / InternVL utilities
   ```

   Follow each component's README (in its subdirectory) to install it and download its checkpoints.
3. **Run the steps in order**, activating the matching env for each (see the table). Each script takes `--help` for sharding/path options, e.g.:

   ```bash
   conda activate egohos && python step2_egohos.py --shard 0 --num-shards 64 --data-root /path/to/clips
   ```

## Output

Per action clip (`<narration_id>/`): EgoHOS + SAM 2 + amodal masks, a TRELLIS mesh per object, SpaTrackerV2 depth & 3D tracks (`spatracker.npz`), and FoundationPose 6-DoF poses (`foundationpose10/`). The dataloader in [RustinS/ObjectForesight](https://github.com/RustinS/ObjectForesight) windows these into training trajectories. The cleaned, packaged result is [`raivn/ObjectForesight-EPIC`](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC).

## Built on

This pipeline is **built on** the following third-party systems, **bundled here for convenience** — see each subdirectory and its upstream project for installation, weights, and license terms: [EgoHOS](https://github.com/owenzlz/EgoHOS), [SAM 2](https://github.com/facebookresearch/sam2), [Diffusion-VAS](diffusion-vas/), [TRELLIS](https://github.com/microsoft/TRELLIS), [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2), [FoundationPose](https://github.com/NVlabs/FoundationPose), and [InternVL](https://github.com/OpenGVLab/InternVL). The step table above maps each component to the stage that uses it. The bundled copies may include local modifications made for pipeline integration — consult each subdirectory for specifics.

## License

Pipeline code is for **non-commercial research**. It is a derivative-processing pipeline over **EPIC-KITCHENS-100** (CC BY-NC 4.0). **Each vendored component retains its own license** (e.g. FoundationPose is NVIDIA source-available/non-commercial) — review and comply with each before use or redistribution.

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
