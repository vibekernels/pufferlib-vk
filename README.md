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
# Train (50M steps, ~16 min on 5090)
python breakout_pixels.py train --total-timesteps 50000000 --num-envs 256

# Generate evaluation videos
python breakout_pixels.py eval
```

## Results

| | MLP-LSTM (state) | CNN (pixels) |
|---|---|---|
| Observation | 118-dim vector | 84x84x4 grayscale |
| Parameters | 148K | 1.7M |
| Training SPS | 2.3M | 52K |
| Time to 300 avg return | ~90s | ~6 min |
| Max achieved score | 864/864 (perfect) | ~350 |
| Time to perfect score | 3.5 min (500M steps) | Not achieved at 50M |

The state-based agent achieves perfect scores consistently in under 4 minutes. The CNN agent learns a solid policy from pixels, reaching ~350 avg return in 16 minutes. The speed gap is primarily due to CNN forward/backward pass cost.

## Renderers

The pixel-based agent needs a renderer to convert game state → 84x84 grayscale frames. We have two:

| Renderer | Method | Renders/sec (256 envs) |
|---|---|---|
| `fast_renderer.py` | Precomputed brick masks + torch.mm | 219K |
| `cuda_renderer.py` | Custom CUDA kernel (1 thread/pixel) | 1.3M (numpy) / 31M (GPU tensor) |

The CUDA renderer is used by default. End-to-end training SPS is ~52K regardless of renderer choice — the CNN forward/backward pass is the bottleneck, not rendering.

## Files

- `breakout_pixels.py` — Pixel-observation PufferEnv wrapper, CNN policy, and PPO training loop
- `cuda_renderer.py` — CUDA kernel renderer compiled at runtime via `torch.utils.cpp_extension.load_inline`
- `fast_renderer.py` — Fallback GPU renderer using torch matmul (no CUDA compilation needed)
- `make_videos.py` — Video generation for the state-based trained agent
