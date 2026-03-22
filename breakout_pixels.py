"""Pixel-observation wrapper for puffer_breakout + CNN training.

Keeps the fast C simulation but renders pixel frames from game state,
so the agent must learn from vision like a real game.
"""
import numpy as np
import gymnasium
import torch
import torch.nn as nn

import pufferlib
import pufferlib.pytorch
import pufferlib.vector
import pufferlib.models

# Game constants from breakout.h
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

# Colors
BG = np.array([6, 24, 24], dtype=np.uint8)
PADDLE = np.array([0, 255, 255], dtype=np.uint8)
BALL = np.array([255, 200, 50], dtype=np.uint8)
BRICK_COLORS = np.array([
    [230, 41, 55],    # RED
    [255, 161, 0],    # ORANGE
    [253, 249, 0],    # YELLOW
    [0, 228, 48],     # GREEN
    [102, 191, 255],  # SKYBLUE
    [0, 82, 172],     # BLUE
], dtype=np.uint8)


from cuda_renderer import CUDARenderer
from fused_cnn import FusedBreakoutCNN, BreakoutCNNLSTM

_RENDERER = None

def batch_render(state_obs, num_agents, out_h=84, out_w=84):
    """Batch render using CUDA kernel renderer."""
    global _RENDERER
    if _RENDERER is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _RENDERER = CUDARenderer(out_h, out_w, device)
    return _RENDERER.render(state_obs)

def batch_render_gpu(state_obs_np, out_h=84, out_w=84):
    """Render on GPU, return GPU tensor. Avoids CPU roundtrip."""
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = CUDARenderer(out_h, out_w, 'cuda')
    obs_gpu = torch.from_numpy(state_obs_np).float().cuda()
    return _RENDERER.render_tensor(obs_gpu)


class BreakoutPixels(pufferlib.PufferEnv):
    """Wraps puffer_breakout to provide pixel observations instead of state.

    Uses the fast C simulation internally but renders 84x84 grayscale frames
    for the agent to learn from.
    """
    def __init__(self, num_envs=1, render_mode=None, frameskip=4,
                 frame_h=84, frame_w=84, framestack=4,
                 buf=None, seed=0, **kwargs):
        from pufferlib.ocean.breakout.breakout import Breakout

        self.frame_h = frame_h
        self.frame_w = frame_w
        self.framestack = framestack
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = 128

        # Observation: stacked grayscale frames (framestack, H, W)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255,
            shape=(framestack, frame_h, frame_w),
            dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(3)

        super().__init__(buf)

        # Create the underlying state-based environment
        self._inner = Breakout(num_envs=num_envs, frameskip=frameskip, seed=seed)

        # Frame stack buffer: (num_envs, framestack, H, W)
        self._frame_stack = np.zeros(
            (num_envs, framestack, frame_h, frame_w), dtype=np.uint8)
        self._gpu_stack = None
        self.tick = 0

    def _render_gpu(self):
        """Render current state on GPU and update GPU frame stack."""
        frames_gpu = batch_render_gpu(self._inner.observations,
                                      self.frame_h, self.frame_w)
        self._gpu_stack[:, :-1] = self._gpu_stack[:, 1:].clone()
        self._gpu_stack[:, -1] = frames_gpu

    def reset(self, seed=0):
        self._inner.reset(seed)
        self._frame_stack[:] = 0

        if torch.cuda.is_available():
            frames_gpu = batch_render_gpu(self._inner.observations,
                                          self.frame_h, self.frame_w)
            self._gpu_stack = torch.zeros(
                self.num_agents, self.framestack, self.frame_h, self.frame_w,
                dtype=torch.uint8, device='cuda')
            for i in range(self.framestack):
                self._gpu_stack[:, i] = frames_gpu
            self.observations[:] = self._gpu_stack.cpu().numpy()
        else:
            self._gpu_stack = None
            frames = batch_render(self._inner.observations, self.num_agents,
                                  self.frame_h, self.frame_w)
            for i in range(self.framestack):
                self._frame_stack[:, i] = frames
            self.observations[:] = self._frame_stack

        self.tick = 0
        return self.observations, []

    def reset_gpu(self, seed=0):
        """Reset and return GPU tensor directly. No CPU copy."""
        self._inner.reset(seed)
        frames_gpu = batch_render_gpu(self._inner.observations,
                                      self.frame_h, self.frame_w)
        self._gpu_stack = torch.zeros(
            self.num_agents, self.framestack, self.frame_h, self.frame_w,
            dtype=torch.uint8, device='cuda')
        for i in range(self.framestack):
            self._gpu_stack[:, i] = frames_gpu
        self.tick = 0
        return self._gpu_stack

    def step(self, actions):
        self._inner.actions[:] = actions
        self._inner.step(actions)
        self.tick += 1

        self.rewards[:] = self._inner.rewards
        self.terminals[:] = self._inner.terminals
        self.truncations[:] = self._inner.truncations

        terminated = np.where(self.terminals | self.truncations)[0]
        if len(terminated) > 0:
            self._frame_stack[terminated] = 0
            if self._gpu_stack is not None:
                self._gpu_stack[terminated] = 0

        if self._gpu_stack is not None:
            self._render_gpu()
            self.observations[:] = self._gpu_stack.cpu().numpy()
        else:
            frames = batch_render(self._inner.observations, self.num_agents,
                                  self.frame_h, self.frame_w)
            self._frame_stack[:, :-1] = self._frame_stack[:, 1:]
            self._frame_stack[:, -1] = frames
            self.observations[:] = self._frame_stack

        info = []
        if self.tick % self.log_interval == 0:
            from pufferlib.ocean.breakout import binding
            info.append(binding.vec_log(self._inner.c_envs))

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def step_gpu(self, actions_np):
        """Step and return GPU tensor directly. No CPU observation copy."""
        self._inner.actions[:] = actions_np
        self._inner.step(actions_np)
        self.tick += 1

        self.rewards[:] = self._inner.rewards
        self.terminals[:] = self._inner.terminals
        self.truncations[:] = self._inner.truncations

        terminated = np.where(self.terminals | self.truncations)[0]
        if len(terminated) > 0:
            self._gpu_stack[terminated] = 0

        self._render_gpu()
        return self._gpu_stack, self.rewards, self.terminals, self.truncations

    def render(self):
        if self._gpu_stack is not None:
            return self._gpu_stack[0, -1].cpu().numpy()
        return self._frame_stack[0, -1]

    def close(self):
        self._inner.close()


# CNN Policy for pixel observations
class BreakoutCNN(nn.Module):
    """NatureCNN policy for 84x84 grayscale Breakout frames."""
    def __init__(self, env, hidden_size=512, framestack=4, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

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

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        return self.network(observations.float() / 255.0)

    def decode_actions(self, hidden):
        return self.actor(hidden), self.value_fn(hidden)


if __name__ == '__main__':
    import time
    import os
    import sys
    import glob
    import imageio
    from collections import deque

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--total-timesteps', type=int, default=500_000_000)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--num-steps', type=int, default=128)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-coef', type=float, default=0.1)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--anneal-lr', action='store_true', default=True)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--framestack', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    env = BreakoutPixels(num_envs=args.num_envs, framestack=args.framestack)
    print(f"Obs space: {env.single_observation_space.shape}")
    print(f"Action space: {env.single_action_space}")
    print(f"Num agents: {env.num_agents}")

    # Create policy — NatureCNN with fused uint8 normalization
    policy = FusedBreakoutCNN(env, hidden_size=512,
                               framestack=args.framestack).to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: FusedBreakoutCNN, {num_params:,} params")

    if args.mode == 'eval':
        model_path = args.model_path or max(
            glob.glob("experiments/breakout_cnn_*.pt"), key=os.path.getctime)
        print(f"Loading: {model_path}")
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()

        eval_env = BreakoutPixels(num_envs=1, framestack=args.framestack)
        os.makedirs('videos', exist_ok=True)

        for ep in range(3):
            ob, _ = eval_env.reset()
            frames = []
            total_reward = 0
            for step in range(10000):
                frame = eval_env.render()
                rgb = np.stack([frame, frame, frame], axis=-1)
                rgb = np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)
                frames.append(rgb)

                with torch.no_grad():
                    ob_t = torch.as_tensor(ob).to(device)
                    logits, _ = policy.forward_eval(ob_t)
                    action = logits.argmax(dim=-1).cpu().numpy()
                    action = action.reshape(eval_env.action_space.shape)

                ob, reward, term, trunc, info = eval_env.step(action)
                total_reward += reward.sum()
                if term.any() or trunc.any():
                    break

            print(f"  Episode {ep+1}: {len(frames)} frames, reward={total_reward:.0f}")
            imageio.mimsave(f'videos/breakout_cnn_ep{ep+1}.mp4', frames, fps=15)
            imageio.mimsave(f'videos/breakout_cnn_ep{ep+1}.gif', frames, fps=15, loop=0)

        eval_env.close()
        print("Videos saved to videos/")
        sys.exit(0)

    # === TRAINING (PPO) ===
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    # Storage — all on GPU, shape (num_steps, num_envs, ...)
    obs_buf = torch.zeros((args.num_steps, args.num_envs) +
                          env.single_observation_space.shape, dtype=torch.uint8, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Use GPU-direct path
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        next_obs = env.reset_gpu()
    else:
        next_obs, _ = env.reset()
        next_obs = torch.as_tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    episode_returns = deque(maxlen=100)
    env_ep_rewards = np.zeros(args.num_envs, dtype=np.float32)

    save_dir = 'experiments'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nTraining for {args.total_timesteps:,} steps ({num_iterations} iterations)")
    print(f"Batch: {args.num_envs} envs × {args.num_steps} steps, "
          f"{args.num_minibatches} minibatches ({minibatch_size} each)")
    print()

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # === ROLLOUT ===
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                logits, value = policy(next_obs)
                values_buf[step] = value.flatten()
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample()
                logprobs_buf[step] = probs.log_prob(action)
            actions_buf[step] = action

            actions_np = action.cpu().numpy()
            if use_gpu:
                next_obs, reward, term, trunc = env.step_gpu(actions_np)
            else:
                next_obs_np, reward, term, trunc, infos = env.step(actions_np)
                next_obs = torch.as_tensor(next_obs_np).to(device)

            next_done_np = np.logical_or(term, trunc)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device) if not isinstance(reward, torch.Tensor) else reward.to(device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            env_ep_rewards += reward
            done_envs = np.where(next_done_np)[0]
            for idx in done_envs:
                episode_returns.append(env_ep_rewards[idx])
                env_ep_rewards[idx] = 0.0

        # === GAE ===
        with torch.no_grad():
            next_value = policy(next_obs)[1].flatten()
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # === PPO UPDATE ===
        b_obs = obs_buf.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                logits, newvalue = policy(b_obs[mb_inds])
                newvalue = newvalue.view(-1)
                probs = torch.distributions.Categorical(logits=logits)
                newlogprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        elapsed = time.time() - start_time
        remaining = (num_iterations - iteration) / max(1, iteration) * elapsed

        if iteration % 10 == 0 or iteration == num_iterations:
            avg_return = np.mean(episode_returns) if episode_returns else 0
            print(f"Iter {iteration}/{num_iterations} | Steps: {global_step:,} | "
                  f"SPS: {sps:,} | Return: {avg_return:.1f} | "
                  f"Loss: {loss.item():.3f} | Entropy: {entropy_loss.item():.3f} | "
                  f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s")

        # Save checkpoint every 100 iterations
        if iteration % 100 == 0 or iteration == num_iterations:
            path = os.path.join(save_dir, f'breakout_cnn_{global_step}.pt')
            torch.save(policy.state_dict(), path)

    env.close()
    print(f"\nTraining complete! Final model: {path}")
    print(f"Total time: {time.time() - start_time:.1f}s")
