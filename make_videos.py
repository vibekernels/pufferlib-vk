"""Generate videos of trained puffer_breakout agent."""
import numpy as np
import torch
import imageio
import pufferlib
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.ocean
import pufferlib.ocean.torch

# Game constants from breakout.h
WIDTH = 576
HEIGHT = 330
PADDLE_HEIGHT = 8
BALL_WIDTH = 32
BALL_HEIGHT = 32
BRICK_WIDTH = 32
BRICK_HEIGHT = 12
BRICK_ROWS = 6
BRICK_COLS = 18
Y_OFFSET = 50

# Colors matching the C renderer
BG_COLOR = np.array([6, 24, 24], dtype=np.uint8)
PADDLE_COLOR = np.array([0, 255, 255], dtype=np.uint8)
BALL_COLOR = np.array([255, 200, 50], dtype=np.uint8)
BRICK_COLORS = [
    np.array([230, 41, 55], dtype=np.uint8),    # RED
    np.array([255, 161, 0], dtype=np.uint8),     # ORANGE
    np.array([253, 249, 0], dtype=np.uint8),     # YELLOW
    np.array([0, 228, 48], dtype=np.uint8),      # GREEN
    np.array([102, 191, 255], dtype=np.uint8),   # SKYBLUE
    np.array([0, 82, 172], dtype=np.uint8),       # BLUE
]
TEXT_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def obs_to_frame(obs, scale=2):
    """Render a frame from the observation vector.

    Obs layout (from breakout.h compute_observations):
      [0] paddle_x / width
      [1] paddle_y / height
      [2] ball_x / width
      [3] ball_y / height
      [4] ball_vx / 512
      [5] ball_vy / 512
      [6] balls_fired / 5
      [7] score / 864
      [8] num_balls / 5
      [9] paddle_width / (2*31)
      [10:] brick_states (1=destroyed)
    """
    frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

    paddle_x = obs[0] * WIDTH
    paddle_y = obs[1] * HEIGHT
    paddle_w = obs[9] * 62  # 2 * HALF_PADDLE_WIDTH
    ball_x = obs[2] * WIDTH
    ball_y = obs[3] * HEIGHT
    score = int(obs[7] * 864)
    num_balls = int(round(obs[8] * 5))

    # Draw bricks
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            idx = row * BRICK_COLS + col
            if obs[10 + idx] < 0.5:  # not destroyed
                bx = int(col * BRICK_WIDTH)
                by = int(row * BRICK_HEIGHT + Y_OFFSET)
                frame[by:by+BRICK_HEIGHT, bx:bx+BRICK_WIDTH] = BRICK_COLORS[row]
                # Add a thin border
                frame[by, bx:bx+BRICK_WIDTH] = BG_COLOR
                frame[by:by+BRICK_HEIGHT, bx] = BG_COLOR

    # Draw paddle
    px, py = int(paddle_x), int(paddle_y)
    pw = int(paddle_w)
    py1, py2 = max(0, py), min(HEIGHT, py + PADDLE_HEIGHT)
    px1, px2 = max(0, px), min(WIDTH, px + pw)
    frame[py1:py2, px1:px2] = PADDLE_COLOR

    # Draw ball
    bx, by = int(ball_x), int(ball_y)
    bx1, bx2 = max(0, bx), min(WIDTH, bx + BALL_WIDTH)
    by1, by2 = max(0, by), min(HEIGHT, by + BALL_HEIGHT)
    frame[by1:by2, bx1:bx2] = BALL_COLOR

    # Draw score text area
    frame[5:20, 5:200] = BG_COLOR  # clear area

    # Scale up
    if scale > 1:
        frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

    return frame


def run_episode(env, policy, device, agent_idx=0, max_steps=5000):
    """Run one episode and collect frames.

    We track agent_idx in the vectorized env and record until it terminates.
    """
    ob, _ = env.reset()
    num_agents = ob.shape[0]
    frames = []
    total_reward = 0

    state = dict(
        lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
        lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
    )

    episode_started = False
    for step in range(max_steps):
        obs_np = ob[agent_idx]
        frames.append(obs_to_frame(obs_np))

        with torch.no_grad():
            ob_tensor = torch.as_tensor(ob).float().to(device)
            logits, value = policy.forward_eval(ob_tensor, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(env.action_space.shape)

        ob, reward, term, trunc, info = env.step(action)
        total_reward += reward[agent_idx]

        # After the first step the episode has started; when our agent terminates, stop
        if step > 0 and (term[agent_idx] or trunc[agent_idx]):
            frames.append(obs_to_frame(ob[agent_idx]))
            break

    return frames, total_reward


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create environment
    make_env = pufferlib.ocean.env_creator('puffer_breakout')
    env = pufferlib.vector.make(
        make_env,
        env_kwargs=dict(num_envs=1024, frameskip=4),
        backend=pufferlib.vector.Serial,
        num_envs=1,
    )

    # Create policy with RNN (matching training config)
    policy = pufferlib.ocean.torch.Policy(env.driver_env, hidden_size=128)
    policy = pufferlib.ocean.torch.Recurrent(env.driver_env, policy, input_size=128, hidden_size=128)
    policy = policy.to(device)

    # Load trained weights
    import glob, os
    model_path = max(glob.glob("experiments/*.pt"), key=os.path.getctime)
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()

    # Generate videos for multiple episodes
    output_dir = '/workspace/pufferlib-vk/videos'
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(3):
        print(f"Running episode {ep+1}/3...")
        frames, reward = run_episode(env, policy, device)
        print(f"  Episode {ep+1}: {len(frames)} frames, reward={reward:.1f}")

        # Save as GIF
        gif_path = os.path.join(output_dir, f'breakout_ep{ep+1}.gif')
        imageio.mimsave(gif_path, frames, fps=15, loop=0)
        print(f"  Saved GIF: {gif_path}")

        # Save as MP4
        mp4_path = os.path.join(output_dir, f'breakout_ep{ep+1}.mp4')
        imageio.mimsave(mp4_path, frames, fps=15)
        print(f"  Saved MP4: {mp4_path}")

    env.close()
    print(f"\nAll videos saved to {output_dir}/")
    print("To view remotely, run: python -m http.server 8080 --directory /workspace/pufferlib-vk/videos")


if __name__ == '__main__':
    main()
