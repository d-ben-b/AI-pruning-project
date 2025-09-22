def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    left = -0.1
    right = 0.1
    mid = 0
    eps = 1e-15

    while right - left > eps:
        mid = (left + right) / 2
        NPV = calculate_NPV(mid, cashFlowVec, cashFlowPeriod, compoundPeriod)
        print(f"left: {left:.4f}, right: {right:.4f}, mid: {mid:.4f}, NPV: {NPV:.4f}")
        if NPV > 0:
            left = mid
        else:
            right = mid
    return mid


def calculate_NPV(r, cashFlowVec, cashFlowPeriod, compoundPeriod):
    NPV = 0.0
    for i, cf in enumerate(cashFlowVec):
        power = i * (compoundPeriod / cashFlowPeriod)
        NPV += cf / (1.0 + r / compoundPeriod) ** power
    return NPV
