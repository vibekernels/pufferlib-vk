"""Benchmark training speed optimizations."""
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque

from breakout_pixels import BreakoutPixels
from fused_cnn import FusedBreakoutCNN

device = torch.device('cuda')
NUM_ITERS = 30  # iterations to benchmark


def bench(label, num_envs=128, num_steps=128, num_minibatches=4,
          update_epochs=4, use_autocast=False, use_compile=False,
          use_grad_scaler=False):
    env = BreakoutPixels(num_envs=num_envs, framestack=4)
    policy = FusedBreakoutCNN(env, hidden_size=512, framestack=4).to(device)

    if use_compile:
        policy = torch.compile(policy)

    optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4, eps=1e-5)
    scaler = torch.amp.GradScaler('cuda') if use_grad_scaler else None

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    obs_buf = torch.zeros((num_steps, num_envs, 4, 84, 84), dtype=torch.uint8, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)
    advantages = torch.zeros((num_steps, num_envs), device=device)

    next_obs = env.reset_gpu()
    next_done = torch.zeros(num_envs, device=device)
    global_step = 0

    # Warmup (2 iters for compile, 1 otherwise)
    warmup = 3 if use_compile else 1
    start_time = None

    for iteration in range(1, NUM_ITERS + warmup + 1):
        if iteration == warmup + 1:
            torch.cuda.synchronize()
            start_time = time.time()
            global_step = 0

        # === ROLLOUT ===
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                if use_autocast:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits, value = policy(next_obs)
                else:
                    logits, value = policy(next_obs)
                values_buf[step] = value.flatten()
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample()
                logprobs_buf[step] = probs.log_prob(action)
            actions_buf[step] = action

            actions_np = action.cpu().numpy()
            next_obs, reward, term, trunc = env.step_gpu(actions_np)
            next_done_np = np.logical_or(term, trunc)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

        # === GAE ===
        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    next_value = policy(next_obs)[1].flatten()
            else:
                next_value = policy(next_obs)[1].flatten()
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + 0.99 * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # === PPO UPDATE ===
        b_obs = obs_buf.reshape((-1, 4, 84, 84))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                if use_autocast:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits, newvalue = policy(b_obs[mb_inds])
                else:
                    logits, newvalue = policy(b_obs[mb_inds])
                newvalue = newvalue.view(-1)
                probs = torch.distributions.Categorical(logits=logits)
                newlogprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - 0.1, 1 + 0.1)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    sps = int(global_step / elapsed)
    print(f"{label:45s} | {sps:>7,} SPS | {elapsed:.1f}s for {global_step:,} steps")
    env.close()
    return sps


if __name__ == '__main__':
    results = {}

    # Baseline
    results['baseline_128'] = bench("Baseline (128 envs, 4mb, 4ep)")

    # Best combos
    results['256e_4mb_4ep'] = bench("256 envs, 4mb, 4ep", num_envs=256)
    results['256e_2mb_4ep'] = bench("256 envs, 2mb, 4ep", num_envs=256, num_minibatches=2)
    results['256e_2mb_3ep'] = bench("256 envs, 2mb, 3ep", num_envs=256, num_minibatches=2, update_epochs=3)
    results['256e_4mb_3ep'] = bench("256 envs, 4mb, 3ep", num_envs=256, update_epochs=3)
    results['512e_4mb_4ep'] = bench("512 envs, 4mb, 4ep", num_envs=512)
    results['512e_2mb_3ep'] = bench("512 envs, 2mb, 3ep", num_envs=512, num_minibatches=2, update_epochs=3)
    results['1024e_4mb_4ep'] = bench("1024 envs, 4mb, 4ep", num_envs=1024)

    print("\n=== SUMMARY ===")
    baseline = results['baseline_128']
    for name, sps in results.items():
        delta = (sps / baseline - 1) * 100
        print(f"  {name:35s}: {sps:>7,} SPS ({delta:+.1f}%)")
