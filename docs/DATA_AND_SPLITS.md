# Problem definition, data and splits

## Problem
**Input:** a video (1080p; 4K tested) and, on its first frame, a few clicks on one object.
**Output:** for every frame an EXR with R, G, B, **A** (soft matte of that object) and **Z** (depth of the whole frame).

This is **video analysis**: every model we rely on links frames together.

| Sub-task | Output per pixel | Model | How it uses other frames |
|---|---|---|---|
| Video object segmentation | object / not object | SAM 2.1 Small | memory bank of earlier frames + memory attention |
| Video matting | alpha 0–1 | MatAnyone | memory propagation, core/boundary fusion |
| Video depth (pixel regression) | relative depth | Video Depth Anything Small | temporal attention over 32-frame windows |
| Depth alignment (ours) | depth scale per frame | least squares + optical flow | fits each frame to the previous one moved by flow |

Each sub-task is scored separately with its own metrics; the gain from using other frames is
measured against a baseline that does not (copy the first mask; per-frame Depth Anything V2).

## Objectives (measurable)
1. Segmentation: J (IoU), Dice and boundary F above the no-temporal baseline, on held-out video.
2. Matting: lower MAD / MSE / dtSSD than a binary mask, on held-out video with true alpha.
3. Depth: lower AbsRel, higher δ1 and lower flicker (OPW) than per-frame depth, on held-out video.
4. Deliverable: valid R, G, B, A, Z EXR at 1080p and 4K on an 8 GB GPU.

## No training: what "split" means here
We train no network. All models are **pretrained** and used **zero-shot**. The risk of leakage is
therefore (a) a test set that the model saw in ITS training, and (b) us choosing settings by looking
at test results. We control both:

| Set | Used for | Data | Never used for |
|---|---|---|---|
| **Development** | building and debugging, choosing every setting | synthetic shot (`common/make_dummy_shot.py`, true depth); Tears of Steel shots 01_2a, 04_2d, 08_3a (no labels) | reported test numbers |
| **Validation** | confirming design choices made on development data | Bonn `moving_nonobstructing_box`, `balloon_tracking` (2 sequences, NOT in the depth test set) | reported test numbers |
| **Test (depth)** | final numbers only | MPI Sintel final pass, 23 sequences, 1,064 frames; Bonn `balloon2, crowd2, crowd3, person_tracking2, synchronous`, 110 frames each from frame 30 | choosing settings |
| **Test (segmentation)** | final numbers only | DAVIS 2017 **val**, 30 videos, every object scored separately | choosing settings |
| **Test (segmentation + matting)** | final numbers only | YouTubeMatte 512×288, 64 clips (32 people × motion/static), true alpha | choosing settings |

**Unit of splitting = whole sequence.** No frame of a test sequence appears in development or
validation, so neighbouring frames cannot leak across sets. Bonn validation and test sequences come
from the same lab and camera: they are different recordings, but not a different environment. We state
this as a limitation.

**Settings fixed before testing.** SAM 2 and Video Depth Anything run with their default settings.
MatAnyone on the benchmarks uses its authors' published low-resolution settings (erode = dilate = 4,
warm-up = 1); on our film shots it uses 720 px, chosen on the development shot 08_3a. Refinement
parameters and the default alignment mode (`global`) were chosen on development data.

**One honest exception.** The `flow` alignment mode was designed after the first `per_frame` mode
performed badly on Bonn test sequences. To make sure that design choice does not depend on test data,
we re-checked it on the two Bonn **validation** sequences, which no decision had used
(results: `results/depth_bonn_val.md`).

## Do the pretrained models already know our test sets?
| Model | Trained on (from its paper) | Our test sets | Overlap? |
|---|---|---|---|
| Video Depth Anything | synthetic video depth sets (Virtual KITTI 2, IRS, TartanAir, PointOdyssey, …) + unlabeled images | Sintel, Bonn | **No.** The paper uses both only for zero-shot evaluation |
| Depth Anything V2 | synthetic images + pseudo-labelled real images | Sintel, Bonn | No (not in its training list) |
| MatAnyone | VM800, D646, AIM; segmentation: COCO, SPD, YouTubeVIS | YouTubeMatte | **No.** YouTubeMatte is the authors' own test benchmark |
| SAM 2.1 | SA-1B, SA-V, internal data; some paper mixtures add open-source VOS sets incl. DAVIS | DAVIS 2017 val | **Possible.** We use the val split only, but cannot rule out that DAVIS data was in the 2.1 training mix. YouTubeMatte gives a second, clean segmentation test |

## Data sources and licences
| Data | Source | Licence / terms |
|---|---|---|
| Tears of Steel | [mango.blender.org](https://mango.blender.org/production/4-tb-original-4k-footage-available-as-cc-by), raw EXR via xiph.org | CC-BY 3.0, (CC) Blender Foundation; actors keep portrait rights |
| MPI Sintel depth | [sintel.is.tue.mpg.de/depth](https://sintel.is.tue.mpg.de/depth) | research use |
| Bonn RGB-D Dynamic | [ipb.uni-bonn.de](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) | cite ReFusion (IROS 2019) |
| DAVIS 2017 | [davischallenge.org](https://davischallenge.org/) | research use |
| YouTubeMatte | [MatAnyone repo](https://github.com/pq-yang/MatAnyone) | research use |

## Data preparation
- Video → numbered PNG frames (`common/extract_frames.py`); Tears of Steel linear EXR → sRGB PNG with one exposure for the whole shot (`common/get_tears_of_steel.py`).
- Sintel `.dpt` and Bonn 16-bit depth PNG (÷5000 = metres) → float32 `.npy`; Bonn RGB and depth paired by nearest timestamp (`evaluation/prepare_benchmarks.py`).
- DAVIS palette annotations → one binary mask per object; YouTubeMatte alpha PNG → float 0–1, mask = alpha > 0.5.
- Invalid depth (0, or beyond 70 m Sintel / 10 m Bonn) is excluded from depth metrics.
- Every stage's output is checked against the interface contract (`common/validate.py`).
