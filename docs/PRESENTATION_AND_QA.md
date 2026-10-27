# Presentation plan and Q&A practice

The rubric gives 3 points for presenting "in a logical order, within the time limit", each person
"explaining their own responsibilities" and "answering questions correctly". Practise with this sheet.

## Talk plan (fill in the time limit: [__] minutes)
The slide order follows the grading rubric, so the examiner can tick each item as you go.

| # | Slide | Rubric item it covers | Presenter (role) | Time (of 15 min) |
|---|---|---|---|---|
| 1 | Title | — | 1 Data/infra | 0:15 |
| 2 | How this talk is organised, and who presents | Presentation (order, roles) | 1 Data/infra | 0:30 |
| 3 | Problem: a matte says what, depth says where | Problem definition | 1 Data/infra | 0:50 |
| 4 | Task type: video analysis (frames linked) | Task difficulty (4 pts) | 8 Evaluation | 1:00 |
| 5 | Deliverable: R, G, B, A, Z EXR | Problem definition | 7 EXR | 0:40 |
| 6 | Pipeline overview | Model development | 1 Data/infra | 0:50 |
| 7 | Data and splits (dev / validation / test, leakage) | Data preparation | 1 Data/infra | 1:10 |
| 8 | Segmentation: SAM 2 + MatAnyone, examples | Model development | 2 SAM 2 + 3 Matting | 0:50 |
| 9 | Segmentation results vs ground truth (5 approaches) | Experiments + evaluation | 2 SAM 2 | 1:10 |
| 10 | Matting results (true alpha) | Experiments + evaluation | 3 Matting | 1:00 |
| 11 | Depth results (4 approaches, Bonn + Sintel) | Experiments + evaluation | 4 Depth | 1:10 |
| 12 | Alignment: the failure and the fix | Experiments | 5 Alignment | 0:50 |
| 13 | Edge refinement | Experiments | 6 Fusion | 0:40 |
| 14 | Error analysis: best and worst cases | Result analysis | 8 Evaluation | 1:10 |
| 15 | Speed and 4K | Evaluation | 7 EXR | 0:30 |
| 16 | Demo (play demo.mp4) | — | 7 EXR | 0:40 |
| 17 | Limitations and next steps | Result analysis | 8 Evaluation | 0:40 |
|  | **Total** | | | **~15:15** |

Fill the `[Name]` placeholders on slide 2 with who presents each part.

Rules: each person says their role in one sentence before their slide ("I worked on the alignment
module, which..."). Do not read the slide: say the one finding, point at the number, move on.
Rehearse twice with a timer; cut words, not slides.

## Q&A practice: likely questions and good answers
Answer in 2–3 sentences: the fact, the number, the reason.

### Task and design
**Q: Why is this "video analysis" and not just image segmentation per frame?**
A: All three models link frames: SAM 2 keeps a memory of earlier frames, MatAnyone propagates memory,
and Video Depth Anything uses temporal attention over 32 frames. We measured the benefit: copying the
first mask (no temporal model) scores far lower J than SAM 2 (see reference_results/seg_davis.md), and per-frame
depth flickers more than video depth on 23 of 23 Sintel and 5 of 5 Bonn sequences.

**Q: You didn't train any model. What did you develop?**
A: The system around the models: alignment (least squares + optical flow), edge-aware refinement,
alpha-aware statistics, the EXR writer, long-video chunking, 4K handling, and the full evaluation.
Training needs labelled video depth and mattes we do not have; zero-shot pretrained models plus our
own geometry steps were the realistic choice for 8 GB and two weeks.

**Q: Why not multiply depth by alpha, so Z is only the object?**
A: It sets the background to zero and gives wrong depth at soft edges. The demo shows why the full Z
matters: the banner hides behind the man and the railing, who are not in the matte at all.

**Q: Why Video Depth Anything and not a per-frame model?**
A: Per-frame depth has a different scale on every frame, which flickers. VDA flickered less on every
test sequence; on Bonn it also halved the error. On Sintel its typical (median) accuracy is similar.

### Data and evaluation
**Q: How did you split the data? Is there leakage?**
A: Whole sequences are the unit: development (our film shots, a synthetic shot), validation (2 Bonn
sequences) and test (Sintel, 5 Bonn sequences, DAVIS 2017 val, YouTubeMatte). Settings were chosen on
development data only. The models' papers list their training data; Sintel, Bonn and YouTubeMatte are
not in it. SAM 2 may have seen DAVIS data, so YouTubeMatte is our second, clean segmentation test.

**Q: Why AbsRel and δ1 for depth?**
A: They are the standard metrics for relative depth: AbsRel is the average relative error, δ1 the
share of pixels within 25% of the truth. Because the models give relative depth, one scale and shift per
sequence is fitted first, the standard protocol.

**Q: How do you measure flicker without ground truth?**
A: Optical-flow warping error (OPW): move frame t+1's depth onto frame t with optical flow and measure
the difference where the flow is reliable. Stable depth gives a small error.

**Q: Why IoU, Dice and boundary F for segmentation?**
A: IoU and Dice measure area overlap with the true mask; boundary F checks the outline, which matters
most for rotoscoping. They are the DAVIS benchmark standard.

**Q: Your average depth on Sintel is good, but the median isn't better. Why?**
A: A few hard sequences (smoke in ambush_2, thin bamboo, fast motion in temple_2) dominate the mean.
We show them in the error analysis. VDA's consistent win is temporal stability.

**Q: Is MatAnyone actually better than SAM 2's mask?**
A: On people, yes: on YouTubeMatte (true alpha, 64 clips) MAD 2.72 vs 5.56 and the lowest flicker
(dtSSD 1.61). Our 2.72 matches the published result, which also checks our evaluation code. On general
objects (DAVIS: animals, bikes) SAM 2's mask is better, because MatAnyone is trained on people.

**Q: How do you know the temporal models help? Give one number.**
A: On DAVIS 2017 val, copying the first-frame mask scores J&F 0.229; SAM 2 with its frame memory 0.883.
For depth, the video model flickers less than per-frame depth on all 28 test sequences.

### Failures and limits
**Q: Where does your pipeline fail?**
A: An object made of separate parts when clicks cover only one part (two people as one object: J 0.38;
with a full first mask the same clip scores 0.99), objects that are tiny in the first frame, thin lines,
smoke and fog (depth). See reference_results/error_analysis.md.

**Q: Your first alignment method failed. What happened?**
A: It fitted every frame to the shot's median depth, which assumes a static camera. On Bonn's moving
camera the error doubled (crowd2: 0.16 → 0.32). We replaced it with a flow-based fit and checked that
choice on separate validation sequences.

**Q: Is the depth in metres?**
A: No. It is relative; Z runs from 1 (near) to 10 (far) in relative units. Metric depth is future work.

**Q: Why 720 px for MatAnyone?**
A: At full HD it needs more than our 8 GB of GPU memory and slows from 0.18 to 8.6 s/frame. The alpha
changes by only 0.0015 on average, but hair edges get a little softer.

### Engineering
**Q: Does it run on 4K?** Yes: 3.2 s/frame on an 8 GB laptop GPU; frames are shrunk for the depth
model and the depth is upsampled, because the model sees about 518 px anyway.

**Q: Does the EXR really work in compositing software?** Yes: Blender 4.2 reads our Z exactly
(difference 0.0). Nuke uses the standard `Z` channel name; it was not tested on Nuke.

**Q: Can someone else run it?** Yes: clone, double-click `setup.bat`, double-click `run_demo.bat`.
`reproduce_results.bat` re-runs the evaluation and compares the numbers with ours; the README is a
step-by-step guide, and the data packs are on the GitHub release.
