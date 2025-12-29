import time
import torch
import gymnasium as gym
from train_dqn_minigrid import QNet  # 確保跟訓練那份同一個 QNet

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

class ActionSubsetWrapper(gym.ActionWrapper):
    """
    7 actions -> 3 actions：
    0 -> left  (原本 0)
    1 -> right (原本 1)
    2 -> forward(原本 2)
    """
    def __init__(self, env, mapping=(0, 1, 2)):
        super().__init__(env)
        self.mapping = list(mapping)
        self.action_space = gym.spaces.Discrete(len(self.mapping))

    def action(self, act):
        return self.mapping[int(act)]


def make_env(env_id: str):
    env = gym.make(env_id, render_mode="rgb_array")  # ★給 RGBImgObsWrapper 正確影像來源
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = ActionSubsetWrapper(env, mapping=(0, 1, 2))
    return env


def main():
    env_id = "MiniGrid-FourRooms-v0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_id)
    obs, info = env.reset()

    # 這裡會印出 3（因為被 wrapper 壓縮了）
    print("wrapped action_space.n =", env.action_space.n)

    # 仍可查看原始 action enum（在 unwrapped 上）
    print("unwrapped.actions =", getattr(env.unwrapped, "actions", None))
    if getattr(env.unwrapped, "actions", None) is not None:
        for i, act in enumerate(env.unwrapped.actions):
            print(i, act)

    # --- 建模 + 載權重（n_actions=3）---
    obs_shape = obs.shape
    n_actions = env.action_space.n  # ★=3
    qnet = QNet(obs_shape, n_actions).to(device)
    qnet.eval()

    ckpt_path = "checkpoints_dqn/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    qnet.load_state_dict(ckpt["q"], strict=True)

    for t in range(2000):
        with torch.no_grad():
            o = torch.from_numpy(obs).unsqueeze(0).to(device)
            qvals = qnet(o)  # (1,3)
            action = int(torch.argmax(qvals, dim=1).item())  # 0/1/2

        # action 是 0/1/2；wrapper 會自動轉成原本 0/1/2 去 step
        obs, reward, terminated, truncated, info = env.step(action)

        u = env.unwrapped
        print(
            "action(sub)=", action,
            "-> raw=", ["left", "right", "forward"][action],
            "pos=", getattr(u, "agent_pos", None),
            "dir=", getattr(u, "agent_dir", None),
            "reward=", reward,
        )

        time.sleep(0.03)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
