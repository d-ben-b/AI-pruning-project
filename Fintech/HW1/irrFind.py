import sys


def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    if not error_handle(cashFlowVec, cashFlowPeriod, compoundPeriod):
        return 0
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
    # print(f"{mid:.4f}")
    return mid


def error_handle(cashFlowVec, cashFlowPeriod, compoundPeriod):
    if len(cashFlowVec) == 0:
        # print("Error: cashFlowVec is empty.")
        return False
    if cashFlowPeriod <= 0 or compoundPeriod <= 0:
        # print("Error: cashFlowPeriod and compoundPeriod must be positive.")
        return False
    if cashFlowPeriod % compoundPeriod != 0:
        # print("Error: cashFlowPeriod must be a multiple of compoundPeriod.")
        return False
    return True


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


def main():
    for i, input_line in enumerate(sys.stdin.readlines()):
        input_numbers = [int(x) for x in input_line.strip().split()]
        cashFlowPeriod, compoundPeriod = input_numbers[-2:]
        cashFlowVec = input_numbers[:-2]
        irr = irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod)
        # print(f"{round(irr * 100, 4):.4f}")


def _cli():
    irr = irrFind([-100, -100, -100, -100, -100, -100, 700], 6, 1)
    # print("Estimated IRR:", irr)


if __name__ == "__main__":
    main()
