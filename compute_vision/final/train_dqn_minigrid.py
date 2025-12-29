import os
import random
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper


# -----------------------
# Action subset wrapper (7 -> 3)
# -----------------------
class ActionSubsetWrapper(gym.ActionWrapper):
    """
    把原本 7 actions 壓成 3 actions：
    0 -> 原本 0 (left)
    1 -> 原本 1 (right)
    2 -> 原本 2 (forward)
    """
    def __init__(self, env, mapping=(0, 1, 2)):
        super().__init__(env)
        self.mapping = list(mapping)
        self.action_space = gym.spaces.Discrete(len(self.mapping))

    def action(self, act):
        return self.mapping[int(act)]

class ExplorationBonusWrapper(gym.Wrapper):
    """
    用「局部視野畫面」的像素，給新看見區域 bonus。
    注意：要放在 ResizeObsWrapper 之前，避免插值灰邊造成假 seen。
    """
    def __init__(self, env, bonus_per_pixel=0.01, black_thr=3):
        super().__init__(env)
        self.bonus_per_pixel = float(bonus_per_pixel)
        self.black_thr = int(black_thr)
        self.prev_seen = None

    def _seen_mask(self, obs: np.ndarray) -> np.ndarray:
        # 更保守：RGB 三通道加總 > thr
        # （比 mean 更不容易被少量雜訊影響）
        return (obs.astype(np.int32).sum(axis=2) > self.black_thr)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_seen = self._seen_mask(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        seen = self._seen_mask(obs)
        if self.prev_seen is None:
            self.prev_seen = seen

        newly_seen = np.logical_and(seen, np.logical_not(self.prev_seen))
        frac = newly_seen.sum() / max(1, newly_seen.size)
        bonus = self.bonus_per_pixel * frac

        self.prev_seen = np.logical_or(self.prev_seen, seen)

        reward = float(reward) + float(bonus)
        info = dict(info)
        info["explore_bonus"] = float(bonus)
        return obs, reward, terminated, truncated, info

class FrameStackWrapper(gym.ObservationWrapper):
    """
    把最近 k 個 obs 疊在 channel 維度：
    (H,W,3) -> (H,W,3*k)
    """
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        assert k >= 1
        self.k = int(k)
        self.frames = deque(maxlen=self.k)

        h, w, c = env.observation_space.shape
        assert c == 3, f"預期 (H,W,3)，但收到 {env.observation_space.shape}"
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c * self.k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self.observation(np.array(obs)), info

    def observation(self, obs):
        # obs 這裡不用，直接用 deque 裡的 frames
        return np.concatenate(list(self.frames), axis=2)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(obs), reward, terminated, truncated, info
# -----------------------
# Fixed-size obs wrapper
# -----------------------
class ResizeObsWrapper(gym.ObservationWrapper):
    """
    把 obs (H,W,3) resize 成固定大小 (target_h,target_w,3)
    - 不改動 env 本身，只改回傳 obs
    """
    def __init__(self, env, target_hw=(64, 64)):
        super().__init__(env)
        self.target_hw = tuple(target_hw)
        h, w = self.target_hw
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        )

    def observation(self, obs):
        x = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        x = x.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()  # (h,w,3)
        return x.cpu().numpy()


# -----------------------
# Config
# -----------------------
@dataclass
class DQNConfig:
    env_id: str = "MiniGrid-FourRooms-v0"
    seed: int = 0

    total_steps: int = 400_000
    learning_starts: int = 10_000
    buffer_size: int = 500_000
    batch_size: int = 64

    gamma: float = 0.99
    lr: float = 1e-4

    train_freq: int = 4
    target_update_freq: int = 2_000

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.6

    max_grad_norm: float = 10.0

    save_dir: str = "checkpoints_dqn"
    save_every: int = 20_000

    # eval + best
    eval_every: int = 10_000
    eval_episodes: int = 10

    # ★固定輸入尺寸
    obs_hw: Tuple[int, int] = (64, 64)

    # ★你 make_env() 用到的兩個參數（一定要有）
    max_episode_steps: int = 300   # 解決 len 固定 100（TimeLimit）
    frame_stack: int = 4           # 局部視野的簡單記憶

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(start: float, end: float, fraction: float, t: int, T: int) -> float:
    cutoff = int(T * fraction)
    if cutoff <= 0:
        return end
    if t >= cutoff:
        return end
    return start + (end - start) * (t / cutoff)


# -----------------------
# Reward shaping wrapper
# -----------------------
class DistanceShapingWrapper(gym.Wrapper):
    """
    shaping = k * (prev_dist - new_dist)
    - 接近目標：正 shaping
    - 遠離目標：負 shaping
    若抓不到 agent/goal 位置，shaping=0（不改 reward）
    """
    def __init__(self, env, k: float = 0.01, enable: bool = True):
        super().__init__(env)
        self.k = float(k)
        self.enable = bool(enable)
        self.prev_dist = None

    def _get_positions(self):
        u = self.env.unwrapped
        agent_pos = getattr(u, "agent_pos", None)
        goal_pos = getattr(u, "goal_pos", None)
        return agent_pos, goal_pos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_dist = None
        if self.enable:
            a, g = self._get_positions()
            if a is not None and g is not None:
                self.prev_dist = math.dist(a, g)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.enable:
            a, g = self._get_positions()
            if a is not None and g is not None:
                dist = math.dist(a, g)
                if self.prev_dist is not None:
                    reward = float(reward) + self.k * (self.prev_dist - dist)
                self.prev_dist = dist

        return obs, reward, terminated, truncated, info


# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.s = deque(maxlen=capacity)
        self.a = deque(maxlen=capacity)
        self.r = deque(maxlen=capacity)
        self.ns = deque(maxlen=capacity)
        self.d = deque(maxlen=capacity)

    def __len__(self):
        return len(self.s)

    def add(self, s, a, r, ns, d):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.ns.append(ns)
        self.d.append(d)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.s), size=batch_size)
        s = np.stack([self.s[i] for i in idx], axis=0)
        a = np.array([self.a[i] for i in idx], dtype=np.int64)
        r = np.array([self.r[i] for i in idx], dtype=np.float32)
        ns = np.stack([self.ns[i] for i in idx], axis=0)
        d = np.array([self.d[i] for i in idx], dtype=np.float32)
        return s, a, r, ns, d


# -----------------------
# Q Network (CNN)
# -----------------------
class QNet(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        h, w, c = obs_shape
        assert c % 3 == 0, f"預期 channel 是 3 的倍數（frame stack），但收到 {obs_shape}"

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            y = self.net(x)
            feat_dim = y.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)



# -----------------------
# Env builder
# -----------------------
def make_env(cfg: DQNConfig, for_eval: bool = False):
    env = gym.make(cfg.env_id, render_mode="rgb_array")

    # ★關鍵：改內層 MiniGrid 的 max_steps（不然永遠 100）
    if hasattr(env.unwrapped, "max_steps"):
        env.unwrapped.max_steps = int(cfg.max_episode_steps)

    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)

    if not for_eval:
        env = ExplorationBonusWrapper(env, bonus_per_pixel=0.02, black_thr=3)

    env = ResizeObsWrapper(env, target_hw=cfg.obs_hw)

    if cfg.frame_stack > 1:
        env = FrameStackWrapper(env, k=cfg.frame_stack)

    env = ActionSubsetWrapper(env, mapping=(0, 1, 2))
    env.reset(seed=cfg.seed)
    return env




# -----------------------
# Eval (greedy)  ★改：回傳 avg_return + success_rate
# -----------------------
@torch.no_grad()
def evaluate_policy(cfg: DQNConfig, q: QNet, n_episodes: int) -> Tuple[float, float]:
    env = make_env(cfg, for_eval=True)
    q.eval()

    returns = []
    success = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            o = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)
            action = int(torch.argmax(q(o), dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)

            # FourRooms 通常到 goal 才 terminated 且 reward>0
            if terminated and float(reward) > 0.0:
                success += 1

        returns.append(ep_ret)

    env.close()
    q.train()

    avg_ret = float(np.mean(returns)) if returns else 0.0
    sr = success / max(1, n_episodes)
    return avg_ret, sr


# -----------------------
# Train
# -----------------------
def main():
    cfg = DQNConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    env = make_env(cfg, for_eval=False)
    assert isinstance(env.action_space, gym.spaces.Discrete), "DQN 需要 Discrete 動作空間"

    obs, info = env.reset()
    obs_shape = obs.shape
    n_actions = env.action_space.n  # 這裡會是 3

    q = QNet(obs_shape, n_actions).to(cfg.device)
    q_tgt = QNet(obs_shape, n_actions).to(cfg.device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.buffer_size)

    episode_return = 0.0
    episode_len = 0
    ep = 0

    # ★改：best 以 success_rate 為主，平手再比 avg_return
    best_sr = -1.0
    best_avg = -1e18
    best_path = os.path.join(cfg.save_dir, "best.pt")

    def save_ckpt(path: str, step_idx: int, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "q": q.state_dict(),
            "cfg": cfg.__dict__,
            "step": step_idx,
        }
        if extra is not None:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[DQN] saved: {path}")

    step_idx = 0
    try:
        for step in range(cfg.total_steps):
            step_idx = step + 1
            eps = linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_fraction, step, cfg.total_steps)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    o = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)
                    action = int(torch.argmax(q(o), dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rb.add(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_return += float(reward)
            episode_len += 1

            if done:
                ep += 1
                print(
                    f"[DQN] step={step_idx:7d} | ep={ep:4d} | return={episode_return:.3f} | "
                    f"len={episode_len:4d} | eps={eps:.3f}"
                )
                obs, info = env.reset()
                episode_return = 0.0
                episode_len = 0

            # train
            if step >= cfg.learning_starts and step % cfg.train_freq == 0 and len(rb) >= cfg.batch_size:
                s, a, r, ns, d = rb.sample(cfg.batch_size)

                s_t = torch.from_numpy(s).to(cfg.device)
                a_t = torch.from_numpy(a).to(cfg.device)
                r_t = torch.from_numpy(r).to(cfg.device)
                ns_t = torch.from_numpy(ns).to(cfg.device)
                d_t = torch.from_numpy(d).to(cfg.device)

                q_sa = q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = q_tgt(ns_t).max(dim=1)[0]
                    target = r_t + cfg.gamma * (1.0 - d_t) * max_next_q

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), cfg.max_grad_norm)
                opt.step()

            # target update
            if step >= cfg.learning_starts and step % cfg.target_update_freq == 0:
                q_tgt.load_state_dict(q.state_dict())

            # periodic save
            if step_idx % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.save_dir, f"dqn_step_{step_idx}.pt")
                save_ckpt(ckpt_path, step_idx)

            # eval + save best (★改：看 success_rate)
            if step_idx % cfg.eval_every == 0:
                avg_ret, sr = evaluate_policy(cfg, q, cfg.eval_episodes)
                print(
                    f"[DQN] eval @ step={step_idx}: avg_return={avg_ret:.3f}, success_rate={sr:.2f} "
                    f"(best_sr={best_sr:.2f}, best_avg={best_avg:.3f})"
                )

                is_better = (sr > best_sr) or (sr == best_sr and avg_ret > best_avg)
                if is_better:
                    best_sr = sr
                    best_avg = avg_ret
                    save_ckpt(
                        best_path,
                        step_idx,
                        extra={"best_sr": best_sr, "best_avg": best_avg},
                    )

    except KeyboardInterrupt:
        print("\n[DQN] KeyboardInterrupt: saving latest + best check...")
        last_path = os.path.join(cfg.save_dir, "last.pt")
        save_ckpt(last_path, step_idx)

        avg_ret, sr = evaluate_policy(cfg, q, cfg.eval_episodes)
        print(
            f"[DQN] eval @ interrupt step={step_idx}: avg_return={avg_ret:.3f}, success_rate={sr:.2f} "
            f"(best_sr={best_sr:.2f}, best_avg={best_avg:.3f})"
        )

        is_better = (sr > best_sr) or (sr == best_sr and avg_ret > best_avg)
        if is_better:
            best_sr = sr
            best_avg = avg_ret
            save_ckpt(
                best_path,
                step_idx,
                extra={"best_sr": best_sr, "best_avg": best_avg},
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
