import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from minigrid.manual_control import ManualControl

env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)

manual_control = ManualControl(env, seed=0)
manual_control.start()
