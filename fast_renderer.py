"""Fast GPU renderer for puffer_breakout.

Renders directly at 84x84 using precomputed brick masks.
No Python loops at render time — everything is batched torch ops.
"""
import torch
import numpy as np

# Game constants
BRICK_ROWS = 6
BRICK_COLS = 18
BRICK_WIDTH = 32
BRICK_HEIGHT = 12
Y_OFFSET = 50
BALL_WIDTH = 32
BALL_HEIGHT = 32
PADDLE_HEIGHT = 8
GAME_WIDTH = 576
GAME_HEIGHT = 330

# Grayscale values
BG = np.array([6, 24, 24])
PADDLE = np.array([0, 255, 255])
BALL = np.array([255, 200, 50])
BRICK_COLORS_RGB = np.array([
    [230, 41, 55], [255, 161, 0], [253, 249, 0],
    [0, 228, 48], [102, 191, 255], [0, 82, 172],
])

BG_GRAY = int(0.299 * BG[0] + 0.587 * BG[1] + 0.114 * BG[2])
PADDLE_GRAY = int(0.299 * PADDLE[0] + 0.587 * PADDLE[1] + 0.114 * PADDLE[2])
BALL_GRAY = int(0.299 * BALL[0] + 0.587 * BALL[1] + 0.114 * BALL[2])
NUM_BRICKS = BRICK_ROWS * BRICK_COLS


class FastRenderer:
    """Fully vectorized GPU renderer. No Python loops at render time."""

    def __init__(self, out_h=84, out_w=84, device='cuda'):
        self.device = device
        self.out_h = out_h
        self.out_w = out_w

        # Scale factors: game coords -> output coords
        self.sx = out_w / GAME_WIDTH
        self.sy = out_h / GAME_HEIGHT

        # Precompute brick masks at output resolution: (NUM_BRICKS, out_h, out_w)
        brick_masks = torch.zeros(NUM_BRICKS, out_h, out_w, dtype=torch.bool, device=device)
        brick_gray_vals = torch.zeros(NUM_BRICKS, dtype=torch.uint8, device=device)

        yy = torch.arange(out_h, device=device).float()
        xx = torch.arange(out_w, device=device).float()

        for row in range(BRICK_ROWS):
            c = BRICK_COLORS_RGB[row]
            gray = int(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            for col in range(BRICK_COLS):
                idx = row * BRICK_COLS + col
                # Brick bounds in game coords
                bx0 = col * BRICK_WIDTH
                bx1 = (col + 1) * BRICK_WIDTH
                by0 = row * BRICK_HEIGHT + Y_OFFSET
                by1 = (row + 1) * BRICK_HEIGHT + Y_OFFSET
                # Convert to output coords
                ox0, ox1 = bx0 * self.sx, bx1 * self.sx
                oy0, oy1 = by0 * self.sy, by1 * self.sy
                y_mask = (yy >= oy0) & (yy < oy1)
                x_mask = (xx >= ox0) & (xx < ox1)
                brick_masks[idx] = y_mask.unsqueeze(1) & x_mask.unsqueeze(0)
                brick_gray_vals[idx] = gray

        self.brick_masks = brick_masks          # (108, 84, 84)
        self.brick_gray_vals = brick_gray_vals  # (108,)

        # Precompute coordinate grids for paddle/ball rendering
        self.yy = yy  # (out_h,)
        self.xx = xx  # (out_w,)

        # Scaled sizes
        self.paddle_h_s = PADDLE_HEIGHT * self.sy
        self.ball_w_s = BALL_WIDTH * self.sx
        self.ball_h_s = BALL_HEIGHT * self.sy

    @torch.no_grad()
    def render(self, state_obs_np):
        """Render batch of state observations to grayscale frames.

        Args:
            state_obs_np: numpy array (N, obs_dim) - raw observations from C env
        Returns:
            numpy array (N, out_h, out_w) uint8
        """
        N = state_obs_np.shape[0]
        obs = torch.from_numpy(state_obs_np).float().to(self.device)

        # Start with background
        frames = torch.full((N, self.out_h, self.out_w), BG_GRAY,
                            dtype=torch.float32, device=self.device)

        # === BRICKS (fully vectorized, no Python loop) ===
        # brick_alive: (N, 108) - True if brick is alive
        brick_alive = obs[:, 10:10 + NUM_BRICKS] < 0.5  # (N, 108)

        # For each brick, its gray value broadcast across its mask pixels
        # brick_masks: (108, H, W), brick_gray_vals: (108,)
        # We want: for each env, sum over alive bricks of (gray_val * mask)
        # This is equivalent to a weighted sum where weights = alive * gray_val

        # Compute per-brick contribution: (108, H, W) * (108,) -> (108, H, W)
        # Each brick paints gray_val where its mask is True
        # We need to find, for each pixel, the gray value of the topmost alive brick
        # Since bricks don't overlap, we can just sum contributions

        # brick_contrib: (108, H, W) with gray values where mask is True, 0 elsewhere
        brick_contrib = self.brick_masks.float() * self.brick_gray_vals.float().view(NUM_BRICKS, 1, 1)
        # (108, H, W)

        # any_brick_mask: for each pixel, is any brick covering it? (108, H, W)
        # We expand alive to (N, 108, 1, 1) and masks to (1, 108, H, W)
        # alive_masks: (N, 108, H, W) - True where brick is alive AND covers pixel
        # This is too much memory for large N. Use matmul instead.

        # Reshape brick_masks to (108, H*W) and multiply by alive (N, 108)
        masks_flat = self.brick_masks.float().view(NUM_BRICKS, -1)  # (108, H*W)
        contrib_flat = brick_contrib.view(NUM_BRICKS, -1)  # (108, H*W)

        # For each env, sum of alive brick contributions
        # result: (N, H*W) = (N, 108) @ (108, H*W)
        brick_pixels = torch.mm(brick_alive.float(), contrib_flat)  # (N, H*W)
        any_brick = torch.mm(brick_alive.float(), masks_flat)  # (N, H*W)

        brick_pixels = brick_pixels.view(N, self.out_h, self.out_w)
        any_brick = any_brick.view(N, self.out_h, self.out_w)

        # Apply brick pixels where any brick is present
        has_brick = any_brick > 0.5
        frames[has_brick] = brick_pixels[has_brick]

        # === PADDLE (vectorized across all envs) ===
        paddle_x = obs[:, 0] * GAME_WIDTH * self.sx   # (N,)
        paddle_y = obs[:, 1] * GAME_HEIGHT * self.sy
        paddle_w = obs[:, 9] * 62 * self.sx

        # (N, 1) comparisons with (1, out_w) -> (N, out_w)
        px_mask = (self.xx.unsqueeze(0) >= paddle_x.unsqueeze(1)) & \
                  (self.xx.unsqueeze(0) < (paddle_x + paddle_w).unsqueeze(1))
        py_mask = (self.yy.unsqueeze(0) >= paddle_y.unsqueeze(1)) & \
                  (self.yy.unsqueeze(0) < (paddle_y + self.paddle_h_s).unsqueeze(1))
        # (N, out_h, out_w)
        paddle_mask = py_mask.unsqueeze(2) & px_mask.unsqueeze(1)
        frames[paddle_mask] = PADDLE_GRAY

        # === BALL (vectorized across all envs) ===
        ball_x = obs[:, 2] * GAME_WIDTH * self.sx
        ball_y = obs[:, 3] * GAME_HEIGHT * self.sy

        bx_mask = (self.xx.unsqueeze(0) >= ball_x.unsqueeze(1)) & \
                  (self.xx.unsqueeze(0) < (ball_x + self.ball_w_s).unsqueeze(1))
        by_mask = (self.yy.unsqueeze(0) >= ball_y.unsqueeze(1)) & \
                  (self.yy.unsqueeze(0) < (ball_y + self.ball_h_s).unsqueeze(1))
        ball_mask = by_mask.unsqueeze(2) & bx_mask.unsqueeze(1)
        frames[ball_mask] = BALL_GRAY

        return frames.to(torch.uint8).cpu().numpy()
