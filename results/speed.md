## Speed (RTX 5060 Laptop, 8 GB)

| Stage | 1080p s/frame (1920×1012, mean of 3 shots) | 4K s/frame (4096×2160) | GPU peak 1080p / 4K |
|---|---|---|---|
| SAM 2.1 Small | 0.137 | 0.2166 | 0.7 GB / 0.7 GB |
| MatAnyone (720 px) | 0.176 | 0.3777 | 1.4 GB / 1.5 GB |
| Video Depth Anything Small | 0.473 | 0.506 | 7.3 GB / 7.3 GB |
| DA V2 Small (baseline) | 0.097 | — | 0.4 GB / — |
| Align (global) | 0.030 | 0.169 | 0.0 GB / 0.0 GB |
| Fuse (refine + stats) | 0.278 | 0.8099 | 0.2 GB / 0.3 GB |
| EXR write | 0.254 | 1.0767 | 0.0 GB / 0.0 GB |
| **Pipeline total** (without baseline) | **1.35** | **3.16** | |

GPU peak = torch.cuda.max_memory_allocated (Windows overhead not included). MatAnyone at full 1080p: 8.6 s/frame (VRAM overflow), alpha MAD vs 720 px only 0.0015.
