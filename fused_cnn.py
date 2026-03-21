"""Fused CUDA normalize + cuDNN convolutions for Breakout NatureCNN.

Key optimization: fuse uint8→float32/255 normalization into a single CUDA kernel,
eliminating two separate GPU passes over the largest tensor (924MB per minibatch).
Convolutions use cuDNN (which vastly outperforms naive CUDA conv kernels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import pufferlib.pytorch

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused uint8 → float32 / 255.0 in one kernel pass.
// Saves two separate kernel launches and ~2GB bandwidth per minibatch.
__global__ void fused_normalize_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Use vectorized 4-byte load when possible
    output[idx] = (float)__ldg(&input[idx]) * (1.0f / 255.0f);
}

// Vectorized version: process 4 uint8 values per thread
__global__ void fused_normalize_vec4_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int total
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 >= total) {
        // Handle tail elements
        for (int i = idx; i < total && i < idx + 4; i++) {
            output[i] = (float)input[i] * (1.0f / 255.0f);
        }
        return;
    }

    // Load 4 bytes at once
    uchar4 vals = *reinterpret_cast<const uchar4*>(&input[idx]);
    output[idx]     = (float)vals.x * (1.0f / 255.0f);
    output[idx + 1] = (float)vals.y * (1.0f / 255.0f);
    output[idx + 2] = (float)vals.z * (1.0f / 255.0f);
    output[idx + 3] = (float)vals.w * (1.0f / 255.0f);
}

torch::Tensor fused_normalize(torch::Tensor input) {
    int total = input.numel();
    auto output = torch::empty_like(input, torch::TensorOptions()
        .dtype(torch::kFloat32).device(input.device()));

    if (total % 4 == 0) {
        int threads = 256;
        int vec_total = total / 4;
        int blocks = (vec_total + threads - 1) / threads;
        fused_normalize_vec4_kernel<<<blocks, threads>>>(
            input.data_ptr<unsigned char>(),
            output.data_ptr<float>(), total);
    } else {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        fused_normalize_kernel<<<blocks, threads>>>(
            input.data_ptr<unsigned char>(),
            output.data_ptr<float>(), total);
    }
    return output;
}
"""

CPP_SOURCE = """
torch::Tensor fused_normalize(torch::Tensor input);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='fused_normalize',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['fused_normalize'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class FusedNormalize(torch.autograd.Function):
    """uint8 → float32/255 in a single CUDA kernel. No backward needed (uint8 leaf)."""

    @staticmethod
    def forward(ctx, input_uint8):
        return _get_module().fused_normalize(input_uint8)

    @staticmethod
    def backward(ctx, grad_output):
        # uint8 input has no gradient
        return None


class FusedBreakoutCNN(nn.Module):
    """NatureCNN with fused uint8 normalization + cuDNN convolutions.

    The normalize step (uint8→float/255) is a single CUDA kernel instead of two
    separate ops. Convolutions use cuDNN via standard PyTorch nn.Conv2d.
    """

    def __init__(self, env, hidden_size=512, framestack=4, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # Eagerly compile
        _get_module()

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(64 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def encode_observations(self, observations):
        # Single CUDA kernel: uint8 → float32/255
        x = FusedNormalize.apply(observations)
        return self.network(x)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        return self.actor(hidden), self.value_fn(hidden)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)

    def decode_actions(self, hidden):
        return self.actor(hidden), self.value_fn(hidden)
