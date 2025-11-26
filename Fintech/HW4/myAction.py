import numpy as np

def myAction01(priceMat, transFeeRate1, transFeeRate2):
    days, stocks = priceMat.shape
    rate1, rate2 = transFeeRate1, transFeeRate2

    dp = [0.0] * days
    parent = {}  # parent[t] = (buy_day, stock)

    # record the best "buy record" for each stock
    best_buy_day = [-1] * stocks
    best_buy_cost = [float('inf')] * stocks
    best_buy_shares = [0] * stocks

    for t in range(days):
        # case 1: do nothing
        dp[t] = dp[t - 1] if t > 0 else 0

        # update possible buy records
        for s in range(stocks):
            if t >= 3:  # buy only if t >= 3 so cooldown before sells is always valid
                cost = priceMat[t][s] * (1 + rate1)
                shares = dp[t - 3] / cost if dp[t - 3] > 0 else 1e6 / cost
                if cost < best_buy_cost[s]:
                    best_buy_cost[s] = cost
                    best_buy_day[s] = t
                    best_buy_shares[s] = shares

        # case 3: try sell any stock
        for s in range(stocks):
            if best_buy_day[s] != -1:
                if t >= best_buy_day[s] + 3:   # enforce cooldown
                    revenue = best_buy_shares[s] * priceMat[t][s] * (1 - rate2)
                    if revenue > dp[t]:
                        dp[t] = revenue
                        parent[t] = (best_buy_day[s], s)

    # reconstruct all trades
    trades = []
    curr_t = days - 1
    while curr_t >= 0:
        if curr_t in parent:
            bday, stk = parent[curr_t]
            trades.append((bday, curr_t, stk))
            curr_t = bday - 3   # comply with cooldown
        else:
            curr_t -= 1

    # convert to action matrix
    actions = []
    curr_cash = 1e6
    for bday, sday, stk in reversed(trades):
        buy_price = priceMat[bday][stk] * (1 + rate1)
        shares = curr_cash / buy_price
        actions.append([bday, -1, stk, curr_cash])
        curr_cash = shares * priceMat[sday][stk] * (1 - rate2)
        actions.append([sday, stk, -1, curr_cash])

    actions = np.array(actions, dtype=object)
    actions = actions[actions[:, 0].argsort()]  # sort by day
    return actions

def myAction02(priceMat, transFeeRate1, transFeeRate2, K=5):
    days, stocks = priceMat.shape
    rate1, rate2 = transFeeRate1, transFeeRate2

    dp = [[0.0] * (K + 1) for _ in range(days)]
    parent = {}  # parent[(t,k)] = (buy_day, stock)

    best_buy_day = [[-1] * stocks for _ in range(K + 1)]
    best_buy_cost = [[float('inf')] * stocks for _ in range(K + 1)]
    best_buy_shares = [[0] * stocks for _ in range(K + 1)]

    for t in range(days):
        for k in range(K + 1):

            # case 1: do nothing
            dp[t][k] = dp[t - 1][k] if t > 0 else 0

            # case 2: buy attempt (only for k>=0)
            if k < K:
                for s in range(stocks):
                    if t >= 3:
                        cost = priceMat[t][s] * (1 + rate1)
                        cash = dp[t - 3][k] if t >= 3 else 0
                        shares = cash / cost if cash > 0 else 1e6 / cost
                        if cost < best_buy_cost[k][s]:
                            best_buy_cost[k][s] = cost
                            best_buy_day[k][s] = t
                            best_buy_shares[k][s] = shares

            # case 3: sell (increase k)
            if k > 0:
                for s in range(stocks):
                    bday = best_buy_day[k-1][s]
                    if bday != -1 and t >= bday + 3:
                        revenue = best_buy_shares[k-1][s] * priceMat[t][s] * (1 - rate2)
                        if revenue > dp[t][k]:
                            dp[t][k] = revenue
                            parent[(t, k)] = (bday, s)

    # reconstruction
    trades = []
    t = days - 1
    k = K
    while k > 0 and t >= 0:
        if (t, k) in parent:
            bday, stk = parent[(t, k)]
            trades.append((bday, t, stk))
            k -= 1
            t = bday - 3
        else:
            t -= 1

    # convert to action matrix
    actions = []
    curr_cash = 1e6
    for bday, sday, stk in reversed(trades):
        buy_price = priceMat[bday][stk] * (1 + rate1)
        shares = curr_cash / buy_price
        actions.append([bday, -1, stk, curr_cash])
        curr_cash = shares * priceMat[sday][stk] * (1 - rate2)
        actions.append([sday, stk, -1, curr_cash])

    actions = np.array(actions, dtype=object)
    actions = actions[actions[:, 0].argsort()]
    return actions


def myAction03(priceMatHistory, priceMatFuture, position, actionHistory, rate1, rate2):

    # === Basic checks ===
    if priceMatHistory is None or len(priceMatHistory) == 0:
        return None

    days_hist, stocks = priceMatHistory.shape
    today_idx = days_hist - 1
    today = priceMatHistory[-1]

    if priceMatFuture is None or len(priceMatFuture) == 0:
        return None
    
    # future window
    future = priceMatFuture   # shape = (F, S)
    
    # === Cooldown check ===
    if len(actionHistory) > 0:
        last_action = actionHistory[-1]
        last_trade_day = int(last_action[0])
        if today_idx < last_trade_day + 3:
            return None

    # === Parse position ===
    cash = position[-1]
    holding_stock = -1
    holding_units = 0
    
    for i in range(stocks):
        if position[i] > 0:
            holding_stock = i
            holding_units = position[i]
            break

    # ============================================================
    #   Case 1: Currently holding stock → Evaluate selling
    # ============================================================
    if holding_stock != -1:
        curr_price = today[holding_stock]
        future_prices = future[:, holding_stock]

        # If future minimum is lower than today → sell now
        if np.min(future_prices) < curr_price:
            sell_value = holding_units * curr_price
            return [today_idx, holding_stock, -1, float(sell_value)]

        # Else do nothing
        return None


    # ============================================================
    #   Case 2: Holding cash → Evaluate which stock to buy
    # ============================================================
    best_stock = -1
    best_gain = -1e18

    for s in range(stocks):
        curr = today[s]
        fut = future[:, s]

        # Expected gain from best future peak
        peak = np.max(fut)
        gain = (peak - curr) / curr

        if gain > best_gain:
            best_gain = gain
            best_stock = s

    # set threshold based on combined fees + safety buffer
    threshold = rate1 + rate2 + 0.01

    # Only buy if gain after max future is meaningful
    if best_gain > threshold:
        return [today_idx, -1, best_stock, float(cash)]
    
    return None
