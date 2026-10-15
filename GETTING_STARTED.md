# Start here

This guide takes you from "I just got the repo" to "I ran the pipeline and know my task".
Follow the steps in order. Every command is typed in **PowerShell** (press Start, type `PowerShell`, open it).
If something fails, look at [When something goes wrong](#when-something-goes-wrong) at the bottom.

**What this project does:** it takes a video and makes one EXR image file per frame with 5 channels:
the picture (R, G, B), a cut-out of one person (A, the "alpha matte") and how far away every pixel is (Z, "depth").
VFX artists use these files to put CG things *behind* or *in front of* the person.

---

## Step 1. Install the tools (once, about 10 minutes)

| Tool | Get it from | Important |
|---|---|---|
| **Python 3.11** | [python.org/downloads](https://www.python.org/downloads/) (choose a 3.11.x version) | On the first install screen, tick **"Add python.exe to PATH"** |
| **Git** | [git-scm.com/download/win](https://git-scm.com/download/win) | Default options are fine |
| **VS Code** (optional, to read and edit code) | [code.visualstudio.com](https://code.visualstudio.com/) | |

Check that both work. Close PowerShell, open it again, then type:
```powershell
python --version
git --version
```
You should see `Python 3.11.x` and `git version 2.x`. If you see "not recognized", reinstall and tick "Add to PATH".

**Do you have an NVIDIA graphics card (GPU)?** Type `nvidia-smi`. If it prints a table with your GPU name, you have one.
- **With an NVIDIA GPU** (8 GB or more): you can run every step.
- **Without one:** you can still do alignment, fusion, the EXR writer and evaluation, using the ready-made data in step 4.

---

## Step 2. Get the code (once)

Use a **short folder**. Long folder paths break the install on Windows.
```powershell
git clone https://github.com/heroforurth/Depth-Aware-Rotoscoping-Pipeline.git C:\work\roto
cd C:\work\roto
```
From now on, always start by going to this folder: `cd C:\work\roto`.

---

## Step 3. Install the project (once, 10–30 minutes, needs internet)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1        # if you HAVE an NVIDIA GPU
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -Cpu   # if you DON'T
```
This makes a private Python environment (`.venv`), installs all libraries and downloads the AI models.
At the end you should see something like:
```
torch 2.14.0+cu130 | GPU: NVIDIA GeForce RTX ...      (or: GPU: none (CPU only))
Done. Activate the environment with:  .venv\Scripts\activate
```

**Every time you open a new PowerShell**, switch on the environment first:
```powershell
cd C:\work\roto
.venv\Scripts\activate
```
You will see `(.venv)` at the start of the line. If it's not there, the commands below will fail.

---

## Step 4. Get data

The big data is not in git (one shot = about 5 GB). Pick one:

**A. You have a GPU:** nothing to download. Three sample videos are already in `samples/videos/`. Go to step 5.

**B. You have no GPU:** download a ready-made "dev pack" (about 1 GB):
1. Open the repo on GitHub → **Releases** (right side) → download e.g. `tos_08_3a_devpack.zip`.
2. Unzip it **into `C:\work\roto`**. You should then have the folder `C:\work\roto\data\tos_08_3a\frames\`.

**C. Just testing code:** make a small fake shot. It takes 2 seconds and needs no GPU:
```powershell
python -m common.make_dummy_shot --shot dummy
```

More options: [samples/README.md](samples/README.md).

---

## Step 5. Run it and look at the result

**With a GPU** (about 2 minutes for 100 frames):
```powershell
python run_pipeline.py --config configs/tos_01_2a_from_mp4.yaml
```

**Without a GPU** (after unzipping a dev pack, about 1 minute):
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages align fuse exr
```

**Fake shot:**
```powershell
python run_pipeline.py --config configs/dummy.yaml
```

The run prints one block per stage and ends with `Done in ...s`. Then check and look:
```powershell
python -m common.validate data/tos_01_2a_from_mp4         # must end with OK
python -m common.preview data/tos_01_2a_from_mp4 alpha    # makes data\...\_preview\alpha.jpg
python -m common.preview data/tos_01_2a_from_mp4 depth_final
python -m demo.make_demo data/tos_01_2a_from_mp4          # makes data\...\demo.mp4
```
Open the `.jpg` and `.mp4` files from File Explorer. The finished EXR files are in `data\<shot>\exr\`.

**What happens inside**, in order (each stage is one folder of code):

| Stage | Code folder | Makes | In one sentence |
|---|---|---|---|
| extract | `common/` | `frames/` | cuts the video into pictures `0001.png`, `0002.png`, ... |
| segment | `segmentation/` | `mask/` | SAM 2 finds the person from a few clicks |
| matte | `matting/` | `alpha/` | MatAnyone makes the edges soft (hair) |
| depth | `depth/` | `depth_raw/` | Video Depth Anything guesses how far every pixel is |
| align | `alignment/` | `depth/` | puts all frames on the same depth scale (no flicker) |
| fuse | `fusion/` | `depth_final/`, `stats.json` | sharpens depth at the person's edge, computes depth stats |
| exr | `exr_writer/` | `exr/` | writes the final R, G, B, A, Z files |

Each folder has a README that explains its stage in more detail.

---

## Step 6. Use your OWN video

1. Put your video file in `C:\work\roto\videos\` (make the folder). Keep it short: 4–5 seconds is enough.
   This folder is never uploaded to GitHub, so private clips stay private.
2. Copy the example config: `copy configs\example_shot.yaml configs\my_shot.yaml`
3. Open `configs\my_shot.yaml` (in VS Code or Notepad) and change two lines:
   ```yaml
   shot: my_shot
   input_video: videos/my_video.mp4
   ```
4. Cut the video into frames:
   ```powershell
   python run_pipeline.py --config configs/my_shot.yaml --stages extract
   ```
5. Choose the person by clicking on them:
   ```powershell
   python -m common.pick_points data/my_shot/frames/0001.png
   ```
   Left click = on the person (3–5 clicks: face, body, legs). Right click = on something that is NOT the person.
   Press `u` to undo, `q` when done. It prints a block starting with `segmentation:`.
   Paste it into `configs\my_shot.yaml`, replacing the old `segmentation:` block.
6. Run the rest:
   ```powershell
   python run_pipeline.py --config configs/my_shot.yaml --stages segment matte depth align fuse exr
   python -m common.preview data/my_shot mask
   ```
   If the green mask misses a part (a sleeve, an arm), add a click there (step 5) and run again.

---

## Step 7. Your task

Every role has a folder and a first task. Read your folder's README first.

| # | Role | Your folder | First task |
|---|---|---|---|
| 1 | Data, infra, pipeline | `common/`, `run_pipeline.py` | Record 3–5 own clips (moving person, hair, busy background; 1080p, 4–5 s). Make a config for each (step 6). Help teammates through steps 1–3 |
| 2 | SAM 2 segmentation | `segmentation/` | Choose click points for the team's own clips. Find a clip where the person gets lost, and fix it with a second prompt on a later frame |
| 3 | MatAnyone matting | `matting/` | On a hair-heavy clip, compare `--set matting.max_size=720` with `-1` (full size). Look at the hair in `common.preview ... alpha` |
| 4 | Depth models | `depth/` | Run both models on the own clips: the default, and `--stages depth --set depth.model=dav2 depth.out_subdir=depth_raw_dav2` |
| 5 | Alignment, flicker | `alignment/` | Try `--set align.mode=global` / `flow` on the own clips. Stretch goal: stop the slow drift of `flow` on long shots |
| 6 | Fusion, refinement | `fusion/` | Try `fusion.refine.sigma_color` 10 / 20 / 40 and compare `common.preview ... depth_final`. Note which looks best and why |
| 7 | EXR, compositing check | `exr_writer/` | Open an EXR in Blender (`--set exr.layout=blender`) or Nuke. Make a small comp that puts something behind the actor |
| 8 | Evaluation, report, demo | `evaluation/`, `results/` | Add the own-clip results to the report and slides. Rehearse the demo |

Results so far: [results/SUMMARY.md](results/SUMMARY.md). Project rules (do not change without the team leader):
the "Design decisions" in [CLAUDE.md](CLAUDE.md).

---

## Step 8. Save and share your work with Git

Work on **your own branch**, never directly on `main`:
```powershell
git checkout main
git pull                                   # get the newest code first
git checkout -b role6-fusion-yourname       # make your branch (once)
```
After you change something:
```powershell
git status                                 # see what changed
git add fusion/fuse.py configs/my_shot.yaml   # add the files you changed (by name)
git commit -m "Fusion: try sigma_color 10/20/40 on own clips"
git push -u origin role6-fusion-yourname
```
Then open the repo on GitHub. It shows **"Compare & pull request"**: click it, write what you changed, and ask the team leader to review.

- Never add `data/`, `videos/` or `checkpoints/` (they are too big; git ignores them anyway).
- `git pull` before you start each day, so you have your teammates' changes.

---

## When something goes wrong

| You see | It means | Do this |
|---|---|---|
| `python is not recognized` | Python not on PATH | Reinstall Python, tick "Add python.exe to PATH" |
| `No module named ...` | environment not switched on | `cd C:\work\roto` then `.venv\Scripts\activate` |
| `WinError 4551 ... Application Control policy` | Windows Smart App Control blocks PyTorch | Windows Security → App & browser control → Smart App Control → **Off**, then restart |
| `WinError 206 ... filename too long` | folder path too long | Clone into `C:\work\roto` (step 2) |
| `CUDA out of memory` | GPU memory full | Close games, browsers and other GPU apps. Try `--set matting.max_size=540` |
| `no CUDA GPU found, running on CPU` | no NVIDIA GPU, or CPU install | Fine for align / fuse / exr; the AI stages will be very slow |
| `segmentation.prompts is empty` | no click points in your config | Do step 6.5 |
| `no depth in ... run the depth (or align) stage first` | a stage before this one didn't run | Run that earlier stage, or unzip a dev pack |
| `python -m common.validate` shows `ERROR` | files are missing or the wrong size | Read the error line; re-run the stage that makes that folder |
| a command runs for a long time with no output | a big model is working | Wait. The progress bars show it's alive |

Still stuck? Copy the **whole** error message (the last 20 lines) and send it to the team chat, with the command you typed.

---

## Words you will see

| Word | Meaning |
|---|---|
| **frame** | one picture of the video |
| **mask** | black/white picture: white = the person (hard edge) |
| **alpha / matte** | like a mask, but with grey values at soft edges (hair, blur) |
| **depth / Z** | how far away each pixel is. Relative: "closer or farther", not metres |
| **EXR** | image file format used in film VFX; holds many channels and exact numbers |
| **config** | the `.yaml` file in `configs/` with all settings for one shot |
| **stage** | one step of the pipeline (segment, matte, depth, ...) |
| **GPU / CUDA** | NVIDIA graphics card / its software. The AI models need it to be fast |
| **.venv** | the project's private Python environment |
| **branch / commit / push / pull request** | your own copy of the code / a saved change / upload / ask to merge into `main` |
