"""CUDA kernel renderer for puffer_breakout.

Each CUDA thread renders one pixel for one environment.
No Python loops, no CPU↔GPU copies — observations go in on GPU, frames come out on GPU.
"""
import torch
from torch.utils.cpp_extension import load_inline

# Game constants baked into the kernel
CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Game constants
#define GAME_WIDTH 576
#define GAME_HEIGHT 330
#define BRICK_ROWS 6
#define BRICK_COLS 18
#define BRICK_WIDTH 32
#define BRICK_HEIGHT 12
#define Y_OFFSET 50
#define BALL_WIDTH 32
#define BALL_HEIGHT 32
#define PADDLE_HEIGHT 8
#define NUM_BRICKS (BRICK_ROWS * BRICK_COLS)

// Grayscale values (precomputed from RGB)
#define BG_GRAY 18
#define PADDLE_GRAY 178
#define BALL_GRAY 188

// Brick grayscale values per row
__constant__ unsigned char BRICK_GRAY[6] = {82, 121, 215, 134, 192, 69};

__global__ void render_kernel(
    const float* __restrict__ obs,  // (N, obs_dim)
    unsigned char* __restrict__ frames,  // (N, out_h, out_w)
    int N, int obs_dim, int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * out_h * out_w;
    if (idx >= total_pixels) return;

    int env = idx / (out_h * out_w);
    int rem = idx % (out_h * out_w);
    int py = rem / out_w;  // pixel y in output
    int px = rem % out_w;  // pixel x in output

    // Map output pixel to game coordinates
    float gx = (float)px * GAME_WIDTH / out_w;
    float gy = (float)py * GAME_HEIGHT / out_h;

    // Read observation for this environment
    const float* o = obs + env * obs_dim;
    float paddle_x = o[0] * GAME_WIDTH;
    float paddle_y = o[1] * GAME_HEIGHT;
    float ball_x = o[2] * GAME_WIDTH;
    float ball_y = o[3] * GAME_HEIGHT;
    float paddle_w = o[9] * 62.0f;

    // Default: background
    unsigned char color = BG_GRAY;

    // Check bricks (first, so paddle/ball draw on top)
    // Which brick column/row does this pixel fall in?
    int brick_col = (int)(gx / BRICK_WIDTH);
    int brick_row = (int)((gy - Y_OFFSET) / BRICK_HEIGHT);

    if (brick_col >= 0 && brick_col < BRICK_COLS &&
        brick_row >= 0 && brick_row < BRICK_ROWS) {
        float brick_x0 = brick_col * BRICK_WIDTH;
        float brick_y0 = brick_row * BRICK_HEIGHT + Y_OFFSET;
        float brick_x1 = brick_x0 + BRICK_WIDTH;
        float brick_y1 = brick_y0 + BRICK_HEIGHT;

        if (gx >= brick_x0 && gx < brick_x1 && gy >= brick_y0 && gy < brick_y1) {
            int brick_idx = brick_row * BRICK_COLS + brick_col;
            float brick_state = o[10 + brick_idx];
            if (brick_state < 0.5f) {
                color = BRICK_GRAY[brick_row];
            }
        }
    }

    // Check paddle
    if (gx >= paddle_x && gx < paddle_x + paddle_w &&
        gy >= paddle_y && gy < paddle_y + PADDLE_HEIGHT) {
        color = PADDLE_GRAY;
    }

    // Check ball (draws on top of everything)
    if (gx >= ball_x && gx < ball_x + BALL_WIDTH &&
        gy >= ball_y && gy < ball_y + BALL_HEIGHT) {
        color = BALL_GRAY;
    }

    frames[idx] = color;
}

torch::Tensor render_cuda(torch::Tensor obs, int out_h, int out_w) {
    int N = obs.size(0);
    int obs_dim = obs.size(1);

    auto frames = torch::empty({N, out_h, out_w},
        torch::TensorOptions().dtype(torch::kUInt8).device(obs.device()));

    int total = N * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    render_kernel<<<blocks, threads>>>(
        obs.data_ptr<float>(),
        frames.data_ptr<unsigned char>(),
        N, obs_dim, out_h, out_w
    );

    return frames;
}
"""

CPP_SOURCE = """
torch::Tensor render_cuda(torch::Tensor obs, int out_h, int out_w);
"""

# Compile the CUDA extension
_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='cuda_renderer',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['render_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class CUDARenderer:
    """CUDA kernel renderer. One thread per pixel per env."""

    def __init__(self, out_h=84, out_w=84, device='cuda'):
        self.out_h = out_h
        self.out_w = out_w
        self.device = device
        self._mod = _get_module()

    @torch.no_grad()
    def render(self, state_obs_np):
        """Render batch of observations to grayscale frames.

        Args:
            state_obs_np: numpy array (N, obs_dim)
        Returns:
            numpy array (N, out_h, out_w) uint8
        """
        obs = torch.from_numpy(state_obs_np).float().to(self.device)
        frames = self._mod.render_cuda(obs, self.out_h, self.out_w)
        return frames.cpu().numpy()

    def render_tensor(self, obs_tensor):
        """Render from GPU tensor, return GPU tensor. Zero CPU involvement."""
        return self._mod.render_cuda(obs_tensor, self.out_h, self.out_w)
