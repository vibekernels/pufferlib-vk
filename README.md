# pufferlib-vk

PufferLib Breakout experiments comparing state-based MLP-LSTM vs pixel-based CNN training on an RTX 5090.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pufferlib[atari] imageio[ffmpeg] Pillow ninja
```

## State-based training (MLP-LSTM)

Uses puffer_breakout's fast C simulation with a 148K-parameter MLP-LSTM policy trained on the 118-dim game state vector.

```bash
# Train (default 90M steps, ~43s on 5090)
python -m pufferlib.pufferl train puffer_breakout

# Train longer for perfect scores (500M steps, ~3.5 min)
python -m pufferlib.pufferl train puffer_breakout --train.total-timesteps 500000000

# Generate videos of trained agent
python make_videos.py
```

## Pixel-based training (CNN)

Wraps puffer_breakout with a GPU renderer that produces 84x84 grayscale frames, then trains a 1.7M-parameter NatureCNN policy via PPO.

```bash
# Train (500M steps, default 512 envs on 5090)
python breakout_pixels.py train

# Generate evaluation videos
python breakout_pixels.py eval
```

## Results

| | MLP-LSTM (state) | CNN (pixels) | CNN-LSTM (pixels) |
|---|---|---|---|
| Observation | 118-dim vector | 84x84x4 grayscale | 84x84x4 grayscale |
| Parameters | 148K | 1.7M | 612K |
| Training SPS | 2.3M | 100K | ~45K |
| Avg return (50M steps) | 864 (perfect) | ~350 | ~260 |
| Max achieved score | 864/864 (perfect) | ~350 | ~260 |

The state-based agent achieves perfect scores consistently in under 4 minutes. The CNN agent with frame stacking is the best pixel-based approach, reaching ~350 avg return. Adding an LSTM on top of the CNN (to replace frame stacking for temporal information) actually performs worse — frame stacking provides velocity information more efficiently than an LSTM for Breakout.

### Optimization progression (pixel CNN)

| Optimization | SPS |
|---|---|
| Baseline (CPU rendering + PyTorch normalize) | 52K |
| GPU-direct observations (eliminate CPU roundtrip) | 66K |
| Fused CUDA uint8→float normalize kernel | 75K |
| Tuned batch params (512 envs, 2mb, 3 epochs) | 100K |

The CNN forward/backward pass is the dominant bottleneck. Custom CUDA convolution kernels were tested but cuDNN outperforms them (Winograd + tensor cores). `torch.compile`, bf16, and fp16 autocast were all tested but hurt performance for this small CNN — the network is too small for tensor cores to help.

## Renderers

The pixel-based agent needs a renderer to convert game state → 84x84 grayscale frames. We have two:

| Renderer | Method | Renders/sec (256 envs) |
|---|---|---|
| `fast_renderer.py` | Precomputed brick masks + torch.mm | 219K |
| `cuda_renderer.py` | Custom CUDA kernel (1 thread/pixel) | 1.3M (numpy) / 31M (GPU tensor) |

The CUDA renderer is used by default. End-to-end training SPS is ~52K regardless of renderer choice — the CNN forward/backward pass is the bottleneck, not rendering.

## Files

- `breakout_pixels.py` — Pixel-observation PufferEnv wrapper, CNN policy, and PPO training loop
- `fused_cnn.py` — Fused CUDA uint8→float normalize kernel + NatureCNN and CNN-LSTM models
- `cuda_renderer.py` — CUDA kernel renderer compiled at runtime via `torch.utils.cpp_extension.load_inline`
- `fast_renderer.py` — Fallback GPU renderer using torch matmul (no CUDA compilation needed)
- `make_videos.py` — Video generation for the state-based trained agent
