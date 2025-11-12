#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    # _Gx = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    # _Gy = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h


#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""

    """ Your code here """
    twoG = G.double()
    result = twoG.double()
    return result


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""

    """ Your code here """
    fourG = G.double().double()
    result = fourG + G
    return result


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""

    """ Your code here """
    INF = callback_get_INFINITY()
    if n == 0:
        return INF, 0, 0
    if n == 1:
        return point, 0, 0

    # MSB -> LSB，初始化 R=point（不計一次加法）
    R = point
    doubles = 0
    additions = 0

    for i in range(n.bit_length() - 2, -1, -1):
        R = R.double()
        doubles += 1
        if (n >> i) & 1:
            R = R + point
            additions += 1

    return R, doubles, additions


def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add using NAF (digits in {-1,0,1})."""
    INF = callback_get_INFINITY()
    if n == 0:
        return INF, 0, 0
    if n == 1:
        return point, 0, 0

    # 產生 NAF（LSB -> MSB）
    k = n
    naf = []
    while k > 0:
        if k & 1:
            ui = 2 - (k & 3)  # 1 或 -1
            naf.append(ui)
            k -= ui
        else:
            naf.append(0)
        k //= 2  # ★ 每輪都右移一位

    # 準備 -point：以 Z=1（仿射）建立新 PointJacobi
    curve = point.curve()
    try:
        ord_n = point.order()
    except Exception:
        ord_n = None
    neg_point = point.__class__(curve, point.x(), (-point.y()) % curve.p(), 1, ord_n)

    # 吃掉最高位 +1：R=point（不計一次加法）
    R = point
    doubles = 0
    additions = 0

    # 自「次高位」往 LSB
    for idx in range(len(naf) - 2, -1, -1):
        R = R.double()
        doubles += 1
        ui = naf[idx]
        if ui == 1:
            R = R + point
            additions += 1
        elif ui == -1:
            R = R + neg_point
            additions += 1

    return R, doubles, additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(
    private_key, hashID, callback_getG, callback_get_n, callback_randint
):
    G = callback_getG()
    n = callback_get_n()
    z = int(hashID, 16) % n

    while True:
        k = callback_randint(1, n - 1)
        R = k * G  # ★ 直接用內建的標量乘法，避免 double_and_add 與 INFINITY
        r = R.x() % n
        if r == 0:
            continue
        kinv = pow(k, -1, n)
        s = (kinv * (z + r * private_key)) % n
        if s == 0:
            continue
        if s > n // 2:  # Low-s 正規化（常見規範）
            s = n - s
        return (r, s)


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(
    public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY
):
    """Verify the digital signature."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    INF = callback_get_INFINITY()
    r, s = signature

    # 基本檢查
    if not (isinstance(r, int) and isinstance(s, int)):
        return False
    if not (1 <= r < n and 1 <= s < n):
        return False

    z = int(hashID, 16) % n
    w = pow(s, -1, n)
    u1 = (z * w) % n
    u2 = (r * w) % n

    U1, _, _ = double_and_add(u1, G, callback_get_INFINITY)
    U2, _, _ = double_and_add(u2, public_key, callback_get_INFINITY)

    # X = U1 + U2（處理無限遠點）
    if U1 == INF:
        X = U2
    elif U2 == INF:
        X = U1
    else:
        X = U1 + U2

    if X == INF:
        return False
    v = X.x() % n
    return v == r
