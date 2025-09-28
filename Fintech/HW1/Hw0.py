def fibonacci(n: int) -> int:
    """計算第 n 個費氏數 (0 <= n <= 90)"""

    if n == 0:

        return 0

    elif n == 1:

        return 1

    a, b = 0, 1

    for _ in range(2, n + 1):

        a, b = b, a + b

    return b


def main():

    try:

        n_str = input().strip()  # 讀取輸入並去掉多餘空白

        n = int(n_str)  # 嘗試轉換成整數

        if n < 0 or n > 90:  # 非負整數範圍檢查

            print(0)

        else:

            print(fibonacci(n))

    except ValueError:

        # 如果輸入不是整數，直接輸出 0

        print(0)


if __name__ == "__main__":

    main()
