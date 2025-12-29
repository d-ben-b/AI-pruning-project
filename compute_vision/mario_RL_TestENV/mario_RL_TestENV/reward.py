import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# # simple actions_dim = 7
# SIMPLE_MOVEMENT = [
#     ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
# ]
# -----------------------------------------------------------------------------
# 獎勵函數
"""
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

"""
"""
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
"""

# ===============to do===============================請自定義獎勵函數 至少7個(包含提供的)
# 例子:用來獎勵玩家蒐集硬幣的行為
# =============== Rewards (至少 7 個) ===========================


def get_coin_reward(info, reward, prev_info):
    """鼓勵蒐集硬幣：每多 1 coin +10"""
    total_reward = reward
    total_reward += (info["coins"] - prev_info["coins"]) * 10
    return total_reward


def distance_y_offset_reward(info, reward, prev_info):
    """
    鼓勵跳躍/高度變化：|Δy| 給小獎勵
    注意：y_pos 的尺度可能很大，先做 clipping 避免爆掉
    """
    total_reward = reward
    dy = info["y_pos"] - prev_info["y_pos"]

    # 只要有高度變化就給一點 shaping（避免卡牆）
    total_reward += min(abs(dy) * 0.05, 2.0)  # 上限 2 分
    return total_reward


def distance_x_offset_reward(info, reward, prev_info):
    """
    鼓勵前進，懲罰停留/後退
    """
    total_reward = reward
    dx = info["x_pos"] - prev_info["x_pos"]

    if dx > 0:
        total_reward += min(dx * 0.1, 5.0)  # 前進給獎勵，上限 5
    elif dx == 0:
        total_reward -= 0.2  # 原地小懲罰
    else:
        total_reward -= min(abs(dx) * 0.2, 5.0)  # 後退更大懲罰，上限 5
    return total_reward


def monster_score_reward(info, reward, prev_info):
    """
    鼓勵分數提升（例如擊敗敵人、碰到分數物件）
    score ↑ → 獎勵
    """
    total_reward = reward
    dscore = info["score"] - prev_info["score"]

    # 分數提升通常跳很大，縮放一下避免爆
    if dscore > 0:
        total_reward += min(dscore * 0.01, 5.0)  # 上限 5
    return total_reward


def final_flag_reward(info, reward):
    """
    完成關卡大獎勵
    """
    total_reward = reward
    if info.get("flag_get", False):
        total_reward += 500.0
    return total_reward


def life_loss_penalty_reward(info, reward, prev_info):
    """
    生命減少大懲罰（死亡或受傷）
    """
    total_reward = reward
    dlife = info["life"] - prev_info["life"]
    if dlife < 0:
        total_reward -= 200.0 * abs(dlife)  # 掉 1 條命 -200（可調）
    return total_reward


def time_penalty_reward(info, reward, prev_info):
    """
    時間流逝小懲罰，鼓勵更快通關
    """
    total_reward = reward
    dtime = info["time"] - prev_info["time"]  # 通常每 step 會 -1 或 0
    if dtime < 0:
        total_reward -= 0.05 * abs(dtime)  # 每少 1 秒 -0.05
    return total_reward


def alive_step_reward(info, reward):
    """
    每一步小獎勵（讓 agent 有「活著就有微小正回饋」的 shaping）
    避免 reward 太稀疏導致學不動
    """
    total_reward = reward
    total_reward += 0.02
    return total_reward


# =============== reward aggregator（建議你這樣用）================


def compute_custom_reward(info, base_reward, prev_info):
    r = base_reward
    r = get_coin_reward(info, r, prev_info)
    r = distance_y_offset_reward(info, r, prev_info)
    r = distance_x_offset_reward(info, r, prev_info)
    r = monster_score_reward(info, r, prev_info)
    r = life_loss_penalty_reward(info, r, prev_info)
    r = time_penalty_reward(info, r, prev_info)
    r = alive_step_reward(info, r)
    r = final_flag_reward(info, r)
    return r


# ===============to do==========================================
