# Depth-Aware Rotoscoping Pipeline

Video in, VFX-ready images out. For every frame, one OpenEXR file with **R, G, B** (the picture),
**A** (a soft cut-out of one chosen person or object) and **Z** (how far away every pixel is),
for compositing in Nuke, Fusion or Blender.

---

## Quick start: 3 steps, no typing

You need **Windows 10/11**, **[Git](https://git-scm.com/download/win)**, internet, and about **10 GB** free.
An NVIDIA GPU (8 GB+) runs everything; without one, the demo uses the CPU parts only.

**1. Get the project.** Open PowerShell (Start → type `PowerShell`), paste this, press Enter:
```powershell
git clone https://github.com/heroforurth/Depth-Aware-Rotoscoping-Pipeline.git
cd Depth-Aware-Rotoscoping-Pipeline
```

**2. Double-click `setup.bat`.** It installs Python 3.11 if you don't have it, builds the project's
own environment (`.venv`), installs PyTorch (GPU or CPU build, detected automatically) and the
libraries, and downloads the AI models. 10–30 minutes, once. It ends with **SETUP DONE**.

**3. Double-click `run_demo.bat`.** With a GPU it runs the whole pipeline on a sample film shot
(about 3 minutes) and opens the demo video. Without a GPU it runs the CPU stages on a synthetic shot.

That's it. If something goes wrong, see [When something goes wrong](#when-something-goes-wrong).

> **Why is there no ready-made `.venv` in the repo?** An environment is about 5 GB (mostly PyTorch for
> the GPU) and only works on the computer that built it. `setup.bat` builds an identical one on yours,
> from the pinned versions in `requirements.txt`.

---

## Contents
1. [What it does](#what-it-does)
2. [Run it on your own video](#run-it-on-your-own-video)
3. [Re-run the results and compare](#re-run-the-results-and-compare)
4. [Data](#data)
5. [Commands (by hand)](#commands-by-hand)
6. [Team tasks](#team-tasks)
7. [Share your work with Git](#share-your-work-with-git)
8. [Editing the project](#editing-the-project)
9. [When something goes wrong](#when-something-goes-wrong)
10. [How it works](#how-it-works)
11. [Results](#results)
12. [For graders](#for-graders)
13. [Licences and credits](#licences-and-credits)
14. [Words you will see](#words-you-will-see)

---

## What it does

```
video ─► frames ─┬─► [A] SAM 2 (find the person) ─► MatAnyone (soft edges) ───► alpha ─┐
                 └─► [B] Video Depth Anything (depth) ─► Alignment (one scale) ─► depth ─┴─► Fusion ─► EXR writer ─► 0001.exr
                                                                                       (sharp edges,   (R,G,B,A,Z)
                                                                                        Z stats)
```

| Stage | Folder | Makes | In one sentence |
|---|---|---|---|
| extract | `common/` | `frames/` | cuts the video into pictures `0001.png`, `0002.png`, … |
| segment | `segmentation/` | `mask/` | SAM 2 finds the person from a few clicks and tracks them |
| matte | `matting/` | `alpha/` | MatAnyone makes the edges soft (hair, motion blur) |
| depth | `depth/` | `depth_raw/` | Video Depth Anything estimates how far every pixel is |
| align | `alignment/` | `depth/` | puts all frames on one depth scale, so nothing flickers |
| fuse | `fusion/` | `depth_final/`, `stats.json` | sharpens depth at the person's edge, computes depth stats |
| exr | `exr_writer/` | `exr/` | writes the final R, G, B, A, Z files |

Every output goes to `data/<shot>/<stage folder>/`. Each module folder has a short technical README
(settings, speed, known failures) for whoever works on that module.

---

## Run it on your own video

1. Put the video in `videos/` (make the folder). 4–5 seconds is enough. This folder is
   never uploaded to GitHub, so private clips stay private.
2. Copy the example settings: `cp configs/example_shot.yaml configs/my_shot.yaml`, open it
   (Notepad or VS Code) and change two lines:
   ```yaml
   shot: my_shot
   input_video: videos/my_video.mp4
   ```
3. Open PowerShell in the project folder and switch on the environment:
   ```powershell
   cd Depth-Aware-Rotoscoping-Pipeline
   .venv\Scripts\activate
   ```
   You see `(.venv)` at the start of the line. Do this every time you open a new PowerShell.
4. Cut the video into frames:
   ```powershell
   python run_pipeline.py --config configs/my_shot.yaml --stages extract
   ```
5. Choose the object by clicking on it:
   ```powershell
   python -m common.pick_points data/my_shot/frames/0001.png
   ```
   Left click = on the object (3–5 clicks: face, body, legs). **If the object has separate parts
   (two people), click on every part.** Right click = something that is NOT the object. `u` undoes,
   `q` finishes. Paste the printed `segmentation:` block into `configs/my_shot.yaml`.
6. Run the rest, and look at the result:
   ```powershell
   python run_pipeline.py --config configs/my_shot.yaml --stages segment matte depth align fuse exr
   python -m common.preview data/my_shot mask        # -> data/my_shot/_preview/mask.jpg
   python -m demo.make_demo data/my_shot             # -> data/my_shot/demo.mp4
   ```
   If the green mask misses a part, add a click there (step 5) and run step 6 again.

---

## Re-run the results and compare

The numbers in the report are kept in **`reference_results/`** (in git). When you re-run an
evaluation, your numbers go to **`results/`** (not in git), so everyone can re-run without
changing the reference. Then compare:

| Double-click / command | What it does | Download | Time (laptop GPU) |
|---|---|---|---|
| `reproduce_results.bat` | Bonn depth (test + validation) + DAVIS segmentation, then compares | ~3.3 GB | ~1 hour |
| `reproduce_results.bat full` (from PowerShell) | + Sintel depth + YouTubeMatte matting | ~13 GB | ~3 hours |
| `python scripts/compare_results.py` | compares whatever is in `results/` with `reference_results/` | — | seconds |

The comparison prints each metric as **MATCH** (within 2%; GPUs differ slightly) or **DIFFER**.
Speed tables are shown but not judged, because they depend on your computer.

---

## Data

The big data is not in git (one shot with all outputs is about 5 GB). Pick what you need:

| You work on | You need | How |
|---|---|---|
| anything, with an NVIDIA GPU | input videos | already here: `samples/videos/*.mp4` (3 film shots, 100 frames each) |
| alignment, fusion, EXR, evaluation, no big GPU | ready-made masks, alpha and depth | **dev packs** on the [Releases page](https://github.com/heroforurth/Depth-Aware-Rotoscoping-Pipeline/releases/tag/data-v1): download a zip, unzip it into Project Folder (it creates `data/<shot>/`) |
| quick code tests, any computer | a small fake shot with true depth | `python -m common.make_dummy_shot --shot dummy` |
| the original 16-bit camera frames, more shots, 4K | Tears of Steel raw footage | `python -m common.get_tears_of_steel --tos_shot 01_2a --start 100 --count 100` (add `--res 4k` for 4K) |
| evaluation datasets | Bonn, DAVIS, Sintel, YouTubeMatte | `python scripts/get_datasets.py bonn davis sintel youtubematte` |

Sample shots: `tos_01_2a` (outdoor bridge, target: the woman), `tos_04_2d` (indoor, overlapping
actors, target: the grey-haired man), `tos_08_3a` (green screen, hair and beard). Their click
points are already in `configs/`.

After unzipping a dev pack, the CPU stages run at once:
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages align fuse exr
```

---

## Commands (by hand)

Always first: `cd Depth-Aware-Rotoscoping-Pipeline` and `.venv\Scripts\activate`.

```powershell
# whole pipeline from a video (configs/*_from_mp4.yaml start from samples/videos)
python run_pipeline.py --config configs/tos_01_2a_from_mp4.yaml
# only some stages
python run_pipeline.py --config configs/tos_08_3a.yaml --stages align fuse exr
# change a setting without editing the file
python run_pipeline.py --config configs/tos_08_3a.yaml --stages depth --set depth.model=dav2 depth.out_subdir=depth_raw_dav2
# check that a shot's files follow the rules (must end with OK)
python -m common.validate data/tos_08_3a
# quick pictures of a stage: mask, alpha, depth_raw, depth, depth_final
python -m common.preview data/tos_08_3a alpha
# demo video read back from the EXR files
python -m demo.make_demo data/tos_08_3a
# EXR for Blender (Blender 4.x needs its own channel names to show depth)
python run_pipeline.py --config configs/tos_01_2a.yaml --stages exr --set exr.layout=blender
```

All settings, with comments: `configs/defaults.yaml`. A shot config only lists what differs.

---

## Team tasks

| # | Role | Folder | First task |
|---|---|---|---|
| 1 | Data, infra, pipeline | `common/`, `run_pipeline.py` | Record 3–5 own clips (moving person, hair, busy background; 1080p, 4–5 s), make a config for each, help teammates with setup |
| 2 | SAM 2 segmentation | `segmentation/` | Choose clicks for the own clips; find a clip where the object gets lost and fix it with a second prompt on a later frame |
| 3 | MatAnyone matting | `matting/` | On a hair-heavy clip compare `--set matting.max_size=720` with `-1` (full size) |
| 4 | Depth models | `depth/` | Run both models on the own clips (default, and `--set depth.model=dav2 depth.out_subdir=depth_raw_dav2`) |
| 5 | Alignment, flicker | `alignment/` | Try `--set align.mode=global` / `flow`; stretch goal: stop `flow` drifting on long shots |
| 6 | Fusion, refinement | `fusion/` | Try `fusion.refine.sigma_color` 10 / 20 / 40, compare `common.preview ... depth_final` |
| 7 | EXR, compositing check | `exr_writer/` | Open an EXR in Blender (`--set exr.layout=blender`) or Nuke, make a comp with something behind the actor |
| 8 | Evaluation, report, demo | `evaluation/`, `reference_results/` | Run `reproduce_results.bat`, compare, add own-clip results to the report and slides, rehearse the demo |

Project rules that must not change without the team leader: "Design decisions" in [CLAUDE.md](CLAUDE.md).
Presentation plan and Q&A practice: [docs/PRESENTATION_AND_QA.md](docs/PRESENTATION_AND_QA.md).

---

## Share your work with Git

Work on your **own branch**; never push to `main`.
```powershell
git checkout depth-aware-rotoscoping-pipeline
git pull                                         # get the newest code first
git checkout -b role6-fusion-yourname            # your branch (once)
# ... change things ...
git status                                       # see what changed
git add fusion/fuse.py configs/my_shot.yaml      # add the files you changed, by name
git commit -m "Fusion: try sigma_color 10/20/40 on own clips"
git push -u origin role6-fusion-yourname
```
GitHub then shows **Compare & pull request**: click it, describe the change, ask for a review.
`data/`, `results/`, `videos/`, `checkpoints/`, `.venv/` are ignored by git on purpose.

---

## Editing the project

**Before every commit:** `python scripts/self_test.py` (30 s, no GPU). It checks the configs, imports,
metrics and the CPU pipeline, and must end with **All good.**

**Where to change what:**

| I want to… | Edit | Notes |
|---|---|---|
| change a setting (filter strength, matting size, depth model, …) | `configs/defaults.yaml` | every setting has a comment; a shot config only overrides what differs |
| try a setting once, without editing files | `--set key=value` on `run_pipeline.py` | e.g. `--set fusion.refine.sigma_color=10` |
| add a new shot | copy `configs/example_shot.yaml` | clicks with `python -m common.pick_points` |
| change how a stage works | that stage's folder (`segmentation/`, `matting/`, `depth/`, `alignment/`, `fusion/`, `exr_writer/`) | each has one `run(cfg, paths)` function and a short README |
| add a new stage | new folder with `run(cfg, paths)`, then one line in `STAGES` in `run_pipeline.py` | read inputs and write outputs only through `common/io.py` |
| change file names, formats or folders | `common/io.py` (and `common/validate.py`) | this is the interface contract: every stage depends on it |
| add a metric | `evaluation/metrics.py` | then use it in an `evaluation/run_*.py` script |
| add or re-run an experiment | `evaluation/run_*.py` → writes `results/` | copy a table to `reference_results/` only when the team agrees it's the new reference |
| change setup or pinned versions | `requirements.txt`, `scripts/setup_env.ps1`, `scripts/setup_models.ps1` | don't upgrade PyTorch/CUDA without asking the team leader |

**Project rules:**
- Files between stages follow the interface contract (`data/<shot>/<stage>/0001.png|.npy|.exr`).
- Settings live in configs, never hard-coded.
- Never multiply Z by alpha.
- Depth is converted to Z only in the EXR writer.

All rules: "Design decisions" in [CLAUDE.md](CLAUDE.md).

**Folder map:**
```
run_pipeline.py        one command for the whole pipeline (lists the stages)
setup.bat  run_demo.bat  reproduce_results.bat     double-click entry points
configs/               settings: defaults.yaml + one file per shot
common/                shared code: file I/O (the contract), config, frames, validator, previews, tools
segmentation/  matting/  depth/  alignment/  fusion/  exr_writer/     one folder per stage
demo/                  demo video from the EXR files
evaluation/            metrics, benchmarks, error analysis
scripts/               setup, datasets, reproduce, compare, self-test, dev packs
reference_results/     the numbers in the report (in git)
docs/                  data & splits, presentation & Q&A
samples/videos/        3 sample film shots (in git)
-- not in git: data/ results/ datasets/ checkpoints/ third_party/ .venv/ tools/ videos/
```

---

## When something goes wrong

| You see | It means | Do this |
|---|---|---|
| `git is not recognized` | Git not installed | Install [Git](https://git-scm.com/download/win), open a new PowerShell |
| `python is not recognized` / setup can't find Python | Python missing | `setup.bat` installs it with winget; if that fails, install Python 3.11 from python.org, tick **Add python.exe to PATH** |
| `No module named ...` | environment not switched on | `cd Depth-Aware-Rotoscoping-Pipeline` then `.venv\Scripts\activate` |
| `WinError 4551 ... Application Control policy` | Windows Smart App Control blocks PyTorch | Windows Security → App & browser control → Smart App Control → **Off**, restart, run `setup.bat` again |
| `WinError 206 ... filename too long` | folder path too long | clone into your folder first |
| setup stops halfway (internet) | download interrupted | double-click `setup.bat` again: it continues |
| `CUDA out of memory` | GPU memory full | close games, browsers and other GPU apps; try `--set matting.max_size=540` |
| `no CUDA GPU found, running on CPU` | no NVIDIA GPU, or CPU install | fine for align / fuse / exr; the AI stages are very slow |
| older NVIDIA driver can't run CUDA 13 | driver too old | update the driver from nvidia.com, then run `setup.bat` again |
| `segmentation.prompts is empty` | no clicks in your config | do step 5 of [Run it on your own video](#run-it-on-your-own-video) |
| `no depth in ... run the depth (or align) stage first` | an earlier stage didn't run | run that stage, or unzip a dev pack |
| `common.validate` shows `ERROR` | files missing or wrong size | read the error line; re-run the stage that makes that folder |
| Google Drive refuses YouTubeMatte | Drive's download limit | open the link the script prints in a browser, save the file where it says, run again |

Still stuck: copy the **whole** error (last 20 lines) and the command you typed into the team chat.

---

## How it works

**Models** (all "Small" versions, pretrained, used without training):

| Stage | Model | Why this one |
|---|---|---|
| segmentation | SAM 2.1 Small | follows any object from a few clicks, using a memory of earlier frames |
| matting | MatAnyone | soft alpha from one first-frame mask, steady over time |
| depth | Video Depth Anything Small | looks at 32 frames together, so depth doesn't flicker |
| depth baseline | Depth Anything V2 Small | one frame at a time; kept only to show the difference |

**Our own steps:**
- **Alignment** fits a scale and shift per frame (robust least squares; `flow` mode follows pixels with optical flow), then one scale for the whole shot.
- **Edge refinement** is a joint bilateral filter guided by colour and alpha, only within 12 px of the matte edge.
- **The EXR writer** stores RGB and A as half float and Z as 32-bit float.

**Depth convention:**
- Inside the pipeline, depth is relative disparity: larger = closer.
- The EXR writer converts it once to Z = distance: larger = farther, from 1 (near) to 10 (far), in relative units, not metres.
- Z is the full depth map, **never** multiplied by alpha: use A to cut out the object.

**Long videos and 4K:**
- Depth runs in overlapping chunks of 120 frames, which are cross-faded.
- 4K frames are shrunk for the depth model and the depth is enlarged back.
- Matting runs at 720 px.

Measured on an RTX 5060 Laptop (8 GB):

| Stage | s/frame 1080p | s/frame 4K |
|---|---|---|
| SAM 2 | 0.14 | 0.22 |
| MatAnyone (720 px) | 0.18 | 0.38 |
| Video Depth Anything | 0.47 | 0.51 |
| Alignment | 0.03 | 0.17 |
| Fusion | 0.28 | 0.81 |
| EXR writer | 0.25 | 1.08 |
| **Whole pipeline** | **1.35** | **3.2** |

---

## Results

Full tables: [reference_results/SUMMARY.md](reference_results/SUMMARY.md). Main findings, all on held-out test data:

| Finding | Number | Test set |
|---|---|---|
| Linking frames is decisive for segmentation | J&F 0.229 (copy first mask) → **0.883** (SAM 2) | DAVIS 2017 val, 61 objects |
| Soft matting halves the error on people | MAD **2.72** vs 5.56 (hard mask); matches the published MatAnyone result | YouTubeMatte, 64 clips |
| Video depth is steadier | less flicker on **28 of 28** sequences | Bonn + Sintel |
| Video depth is more accurate on real people | AbsRel **0.055** vs 0.104 | Bonn |
| …but not on every film sequence | Sintel median AbsRel 0.281 vs 0.273 (about equal) | Sintel |
| Our flow alignment reduces flicker | −23% (Bonn), −35% (Sintel), −30% (validation) | Bonn, Sintel |
| Refinement puts the depth edge on the matte | edge sharpness 0.59 → **0.95** | our outdoor film shot |

Error analysis with pictures of good and failed cases: [reference_results/error_analysis.md](reference_results/error_analysis.md).

---

## For graders

| Rubric item | Evidence |
|---|---|
| Task difficulty: video analysis | every model links frames; gain measured against no-temporal baselines: [reference_results/SUMMARY.md §1](reference_results/SUMMARY.md) |
| Problem definition, data preparation, splits, leakage | [docs/DATA_AND_SPLITS.md](docs/DATA_AND_SPLITS.md) |
| Model development, ≥ 3 approaches compared fairly | 5 segmentation/matting approaches, 4 depth approaches, 4 alignment modes: [reference_results/SUMMARY.md §3](reference_results/SUMMARY.md) |
| Evaluation on held-out test sets, error analysis | [reference_results/](reference_results/), [error_analysis.md](reference_results/error_analysis.md) |
| Presentation and Q&A | [docs/PRESENTATION_AND_QA.md](docs/PRESENTATION_AND_QA.md) |
| Code, instructions, data sources, configuration | this README, `configs/`, `setup.bat`, `run_demo.bat`, `reproduce_results.bat` |

---

## Licences and credits

- **Models:** SAM 2.1, Video Depth Anything Small and Depth Anything V2 Small are Apache 2.0. **MatAnyone is NTU S-Lab License 1.0 (non-commercial).** The models are downloaded by `setup.bat`, not stored in this repo.
- **Footage:** *Tears of Steel*, CC-BY 3.0, **(CC) Blender Foundation | mango.blender.org**. The actors keep their portrait rights, so use it for technical work and demos, not advertising.
- **Benchmarks:** Bonn RGB-D Dynamic (cite ReFusion, IROS 2019), MPI Sintel, DAVIS 2017, YouTubeMatte: research use. Details in [docs/DATA_AND_SPLITS.md](docs/DATA_AND_SPLITS.md).

---

## Words you will see

| Word | Meaning |
|---|---|
| frame | one picture of the video |
| mask | black/white picture: white = the object (hard edge) |
| alpha / matte | like a mask, but grey at soft edges (hair, blur) |
| depth / Z | how far each pixel is; relative ("nearer/farther"), not metres |
| EXR | film-industry image file; many channels, exact numbers |
| config | the `.yaml` file in `configs/` with all settings for one shot |
| stage | one step of the pipeline (segment, matte, depth, …) |
| GPU / CUDA | NVIDIA graphics card / its software; the AI models need it to be fast |
| `.venv` | the project's own Python environment, built by `setup.bat` |
| branch / commit / push / pull request | your copy of the code / a saved change / upload / ask to merge it |
