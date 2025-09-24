def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    left = -0.1
    right = 0.1
    mid = 0
    eps = 1e-15

    while right - left > eps:
        mid = (left + right) / 2
        NPV = calculate_NPV(mid, cashFlowVec, cashFlowPeriod, compoundPeriod)
        if NPV > 0:
            left = mid
        else:
            right = mid
    return mid


def calculate_NPV(r, cashFlowVec, cashFlowPeriod, compoundPeriod):
    NPV = 0
    for i, cf in enumerate(cashFlowVec):
        # Rate per compounding period
        rate_per_period = r * compoundPeriod / 12

        # Number of compounding periods until this cash flow
        num_periods = i * cashFlowPeriod / compoundPeriod

        # Calculate present value of this cash flow
        NPV += cf / ((1 + rate_per_period) ** num_periods)
    return NPV


if __name__ == "__main__":

    irr = irrFind([-100, -100, -100, -100, -100, -100, 700], 6, 6)
    print("Estimated IRR:", irr)
