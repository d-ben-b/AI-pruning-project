# myStrategy.py
import numpy as np

# ======== MACD Parameters (可自行微調/固定) ========
FAST_PERIOD = 8
SLOW_PERIOD = 24
SIGNAL_PERIOD = 9

# Precompute alphas
ALPHA_FAST = 2.0 / (FAST_PERIOD + 1.0)
ALPHA_SLOW = 2.0 / (SLOW_PERIOD + 1.0)
ALPHA_SIG = 2.0 / (SIGNAL_PERIOD + 1.0)

# ======== Global incremental state (每日 O(1) 更新) ========
_fast_ema = None
_slow_ema = None
_signal_ema = None

_macd_buffer = []  # 用於初始化 signal_ema，長度至多 SIGNAL_PERIOD

_prev_macd = None
_prev_signal = None


def _reset_state():
    global _fast_ema, _slow_ema, _signal_ema
    global _macd_buffer, _prev_macd, _prev_signal
    _fast_ema = None
    _slow_ema = None
    _signal_ema = None
    _macd_buffer = []
    _prev_macd = None
    _prev_signal = None


def myStrategy(pastPriceVec, currentPrice):
    """
    必須回傳：
      1  = buy
      0  = hold
     -1  = sell
    規範：只能用歷史到當日之資料（本實作僅用 pastPriceVec + currentPrice）
    效能：每日 O(1)
    """
    global _fast_ema, _slow_ema, _signal_ema
    global _macd_buffer, _prev_macd, _prev_signal

    # 在每個新 backtest 的第一天重置狀態
    if len(pastPriceVec) == 0:
        _reset_state()

    price = float(currentPrice)

    # 若資料未滿足 EMA 起算天數，先觀望
    # 需要用「包含今天」的視窗初始化 EMA：用 past tail + current
    if _fast_ema is None:
        if len(pastPriceVec) + 1 < FAST_PERIOD:
            return 0
        # 初始化 fast EMA = 近 FAST_PERIOD 天含今日的 SMA
        if FAST_PERIOD == 1:
            _fast_ema = price
        else:
            tail = (
                pastPriceVec[-(FAST_PERIOD - 1) :] if FAST_PERIOD > 1 else np.array([])
            )
            _fast_ema = (np.sum(tail) + price) / FAST_PERIOD
        # 仍可能 slow EMA 尚未就緒，當天暫不下單
        # 之後會繼續往下執行，讓 slow 也嘗試初始化

    if _slow_ema is None:
        if len(pastPriceVec) + 1 < SLOW_PERIOD:
            # fast_ema 就緒但 slow 未就緒：先觀望
            # 同時把 fast_ema 用今天價更新（避免停滯）
            _fast_ema = ALPHA_FAST * price + (1.0 - ALPHA_FAST) * _fast_ema
            return 0
        # 初始化 slow EMA = 近 SLOW_PERIOD 天含今日的 SMA
        if SLOW_PERIOD == 1:
            _slow_ema = price
        else:
            tail = (
                pastPriceVec[-(SLOW_PERIOD - 1) :] if SLOW_PERIOD > 1 else np.array([])
            )
            _slow_ema = (np.sum(tail) + price) / SLOW_PERIOD

        # 既然今天才把 slow 初始化，今天就先觀望，下一天再正常更新
        return 0

    # ===== 正常日：以 O(1) 增量更新兩條 EMA =====
    _fast_ema = ALPHA_FAST * price + (1.0 - ALPHA_FAST) * _fast_ema
    _slow_ema = ALPHA_SLOW * price + (1.0 - ALPHA_SLOW) * _slow_ema

    macd = _fast_ema - _slow_ema

    # ===== Signal 線初始化／更新 =====
    if _signal_ema is None:
        # 尚未有 signal_ema：先收集 MACD 值到 buffer
        _macd_buffer.append(macd)
        if len(_macd_buffer) < SIGNAL_PERIOD:
            # 尚未湊滿：觀望
            _prev_macd = macd
            _prev_signal = None
            return 0
        # 湊滿後，用 SMA 初始化 signal_ema
        _signal_ema = float(np.mean(_macd_buffer[-SIGNAL_PERIOD:]))
        _prev_macd = macd
        _prev_signal = _signal_ema
        # 當天剛初始化完成，保守起見先不下單
        return 0
    else:
        # 已初始化：用 EMA 方式增量更新
        _signal_ema = ALPHA_SIG * macd + (1.0 - ALPHA_SIG) * _signal_ema

    action = 0

    # ===== 交易邏輯（簡潔穩健）=====
    # 1) 進場：MACD 上穿 Signal，且價格位於慢線之上（過濾震盪）
    # 2) 出場：MACD 下穿 Signal（或可加：價格跌回慢線下）
    if (_prev_signal is not None) and (_prev_macd is not None):
        crossed_up = (_prev_macd <= _prev_signal) and (macd > _signal_ema)
        crossed_down = (_prev_macd >= _prev_signal) and (macd < _signal_ema)

        if crossed_up and (price > _slow_ema):
            action = 1
        elif crossed_down:
            action = -1
        else:
            action = 0

    # 更新「前一日」值
    _prev_macd = macd
    _prev_signal = _signal_ema

    return int(action)
