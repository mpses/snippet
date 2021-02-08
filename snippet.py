# ----------- 入力 ----------- #
import sys
input = sys.stdin.readline
lambda: map(int, input().split())
lambda: map(int, open(0).read().split())
lambda: [[*map(int, o.split())] for o in open(0)]
# --------------------------- #



# 再帰制限変更
import sys
sys.setrecursionlimit(50000)


# タイマー (sec.)
import time
start = time.time()
time_limit = 1.5
while time.time() - start < time_limit:
    pass


INF = float("inf")  # 10**9


# リスト埋め込み, AtCoder なら 50000 要素くらい・圧縮率が高ければそれ以上も埋め込める
def encode_list(lst):
    import array, gzip, base64
    int32 = "l" if array.array("l").itemsize == 4 else "i"
    return base64.b64encode(gzip.compress(array.array(int32, lst)))

def decode_list(lst):
    import array, gzip, base64
    int32 = "l" if array.array("l").itemsize == 4 else "i"
    return array.array(int32, gzip.decompress(base64.b64decode(lst)))


class V:
    # 更新クラス
    def __init__(self, f, v = None):
        self.f = f
        self.v = v

    def __str__(self):
        return str(self.v)

    def update(self, n):
        if n is None:
            return

        if self.v is None:
            self.v = n
            return

        self.v = self.f(self.v, n)


# メモ化
from functools import lru_cache
@lru_cache(maxsize = 1000)
def memowise():
    pass


def compress(a: list):
    # 座標圧縮
    s = sorted(set(a))
    d = {v : i for i, v in enumerate(s)}
    return s, d, [d[i] for i in a]


def separate(a: list, length) -> zip:
    # リスト分割
    return zip(*[iter(a)]*length)


import numpy as np
s = s.T  # 転置
s = s[::-1, :].T  # 90度回転
s = s[::-1, ::-1]  # 180度回転
s = np.rot90(s, n).copy()  # 反時計回りに 90 * n 度 回転

lambda: [*map(list, zip(*a))]   # 転置
lambda: [*map(list, zip(*a[::-1]))]   # 90度右回転
lambda: [*map(list, zip(*a))][::-1]   # 90度左回転
lambda: [b[::-1] for b in a][::-1]    # 180度回転


# ----------- mod系 ----------- #
MOD = 10**9 + 7
MOD2 = 998244353


def egcd(a, b):
    # 拡張ユークリッド互除法
    # g = gcd(a, b) および ax + by = g の最小整数解を返す
    if a == 0:
        return b, 0, 1
    g, y, x = egcd(b % a, a)
    return g, x - (b // a) * y, y


# a の逆元 (mod m)
def modinv(a, m = MOD): # ver.3.8
    return pow(a, -1, m)

def modinv2(a, p = MOD):
    return pow(a, p - 2, p)

def modinv3(a, m = MOD):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist.')
    else:
        return x % m


from math import factorial as f
P = lambda n, r: f(n) // f(n - r)
C = lambda n, r: f(n) // (f(n - r) * f(r))
H = lambda n, r: f(n + r - 1) // (f(r) * f(n - 1))


from scipy.special import comb
comb(10**5, 100, exact = True) % MOD


class Factorial:
    def __init__(self, n, mod):
        # O(n [+ log mod])
        self.f = f =[0] * (n + 1)
        f[0] = b = 1
        for i in range(1, n + 1):
            f[i] = b = b * i % mod
        self.inv = inv = [0] * (n + 1)
        inv[n] = b = modinv(self.f[n], mod)
        for i in range(n, 0, -1):
            inv[i - 1] = b = b * i % mod
        self.mod = mod

    def __call__(self, n, k):
        return self.C(n, k)

    def factorial(self, i):
        return self.f[i]

    def ifactorial(self, i):
        return self.inv[i]

    def C(self, n, k):
        if not 0 <= k <= n: return 0
        return self.f[n] * self.inv[n - k] * self.inv[k] % self.mod

    def P(self, n, k):
        if not 0 <= k <= n: return 0
        return self.f[n] * self.inv[n - k] % self.mod

    def H(self, n, k):
        if (n == 0 and k > 0) or k < 0: return 0
        return self.f[n + k - 1] * self.inv[k] % self.mod * self.inv[n - 1] % self.mod
# ----------------------------- #



# ---------- 約数・素数 ---------- #
def factorize(n) -> list:
    # 素因数分解 O(√n)
    b, e = 2, 0
    fct = []
    while b * b <= n:
        while n % b == 0:
            n //= b
            e += 1
        if e:
            fct += (b, e),
        b += 1
        e = 0
    if n > 1:
        fct += (n, 1),
    return fct
# P_n, E_n = zip(*factorize(n))


def div(n) -> list:
    # 約数列挙 O(√n)
    if n == 0:
        return [0]
    i = 1
    lower_table, upper_table = [], []
    while i * i <= n:
        if n % i == 0:
            lower_table += i,
            upper_table += n // i,
        i += 1
    return lower_table + upper_table[::-1]


def is_prime(n):
    # 素数判定 O(√n)
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return n != 1


def era(n, option = False) -> list:
    # エラトステネスの篩 (Sieve of Eratosthenes) O(nloglogn)
    p = [1] * (n + 1)
    p[0] = p[1] = 0
    for x in range(2, int(n**.5) + 1):
        if p[x]:
            for y in range(x*2, n + 1, x):
                p[y] = 0
    return [e for e, q in enumerate(p) if q] if option else p


def prime_checker(n, option = False) -> list:
    # エラトステネスの篩 (Sieve of Eratosthenes) O(nloglogn)
    p = [False, True, False, False, False, True] * (n // 6 + 1)
    del p[n + 1:]
    p[1 : 4] = False, True, True
    for x in range(5, int(n**.5 + 1)):
        if p[x]:
            p[x * x :: 2*x] = [False] * ((n // x - x) // 2 + 1)
    return [e for e, q in enumerate(p) if q] if option else p


class Osa_k:
    # Osa_k法
    def __init__(self, n_max):
        # O(nloglogn)
        self.n_max = n_max
        self.min_factor = min_factor = list(range(n_max + 1))
        min_factor[2::2] = [2] * (n_max // 2)
        min_factor[3::6] = [3] * ((n_max + 3) // 6)
        for i in range(5, int(n_max ** .5) + 1, 2):
            if min_factor[i] == i:
                for j in range(i*i, n_max + 1, i):
                    if min_factor[j] == j:
                        min_factor[j] = i

    def __call__(self, n):
        # 素因数分解 O(logn)
        if not 1 <= n <= self.n_max: raise ValueError("Invaild Value!")
        min_factor = self.min_factor
        n_twoes = (n & -n).bit_length() - 1
        res = [2] * n_twoes
        n >>= n_twoes
        resappend = res.append
        while n > 1:
            p = min_factor[n]
            resappend(p)
            n //= p
        return res
    
    def d(self, n):
        # 約数個数 O(logn)
        min_factor = self.min_factor
        t = []
        tappend = t.append
        while n > 1:
            if t and t[-1][0] == min_factor[n]:
                t[-1][1] += 1
            else:
                tappend([min_factor[n], 1])
            n //= min_factor[n]
        res = 1
        for _, i in t:
            res *= i + 1
        return res

    def div(self, n):
        # 約数列挙 O(logn + d(n))?
        factors = self.__call__(n)
        res = {1, n}
        for i in range(1, len(factors)):
            for j in combinations(factors, i):
                tmp = 1
                for i in j:
                    tmp *= i
                res |= {tmp}
        return res


def fast_prime_factorization(n):
    # 素因数分解(ロー法) O(⁴√n polylog(n))
    from subprocess import Popen, PIPE
    return [*map(int, Popen(["factor", str(n)], stdout=PIPE).communicate()[0].split()[1:])]

def fast_prime_factorization_many(lst):
    # 素因数分解(ロー法) 複数
    from subprocess import Popen, PIPE
    res = Popen(["factor"] + list(map(str, lst)), stdout=PIPE).communicate()[0].split(b"\n")[:-1]
    return [[*map(int, r.split()[1:])] for r in res]


# 2015, Forisekらによる高速素因数分解
# https://github.com/Lgeu/snippet
import array
# import bz2
import gzip
import base64
int32 = "l" if array.array("l").itemsize == 4 else "i"  # 環境によって異なる

# bases = [2404423, 3027617, 3715179, 3264583, ...
# bases = array.array(int32, bases)
# bases = gzip.compress(bases)
# bases = base64.b64encode(bases)

bases = b'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'
bases = base64.b64decode(bases)
bases = gzip.decompress(bases)
bases = array.array(int32, bases)
assert len(bases) == 16384

def is_SPRP(n, a):
    if n == a: return True
    if n % a == 0: return False
    d = m1 = n-1
    s = (d & -d).bit_length() - 1
    d >>= s
    cur = pow(a, d, n)
    if cur == 1: return True
    for _ in range(s):
        if cur == m1: return True
        cur = cur * cur % n
    return False

def is_prime(x):
    # Fast Primality Testing for Integers That Fit into a Machine Word
    # http://ceur-ws.org/Vol-1326/020-Forisek.pdf
    if x in [2, 3, 5, 7]: return True
    if x%2==0 or x%3==0 or x%5==0 or x%7==0: return False
    if x<121: return x>1
    if not is_SPRP(x, 2): return False
    h = ((x >> 32) ^ x) * 0x45d9f3b3335b369 & 0xffffffffffffffff
    h = ((h >> 32) ^ h) * 0x3335b36945d9f3b & 0xffffffffffffffff
    h = ((h >> 32) ^ h)
    b = bases[h & 16383]
    return is_SPRP(x, b&4095) and is_SPRP(x, b>>12)
# ------------------------------ #



# ---------- UnionFind ---------- #
class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.parents[x] > self.parents[y]:
            x, y = y, x
        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())


class WeightedUnionFind:
    # 重みつき UnionFind
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n
        self.weight = [0] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            px = self.find(self.parents[x])
            self.weight[x] += self.weight[self.parents[x]]
            self.parents[x] = px
            return px

    def union(self, x, y, w):
        w += self.weight[x] - self.weight[y]
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.parents[x] > self.parents[y]:
            x, y, w = y, x, -w
        self.parents[x] += self.parents[y]
        self.parents[y] = x
        self.weight[y] = w
        return

    def weig(self, x):
        self.find(x)
        return self.weight[x]

    def diff(self, x, y):
        return self.weig(y) - self.weig(x)

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())


INF = 10**9
from biesct import*
class PPUnionFind:
    # 部分永続 UnionFind
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n
        self.time = [INF] * n
        self.number_time = [[0] for _ in [None] * n]
        self.number_dots = [[1] for _ in [None] * n]

    def find(self, x, t):
        while self.time[x] <= t:
            x = self.parents[x]
        return x

    def union(self, x, y, t):
        x = self.find(x, t)
        y = self.find(y, t)
        if x == y:
            return 0
        if self.parents[x] > self.parents[y]:
            x, y = y, x
        self.parents[x] += self.parents[y]
        self.parents[y] = x
        self.number_time[x] += [t]
        self.number_dots[x] += [-self.parents[x]]
        self.time[y] = t
        return t

    def size(self, x, t):
        x = self.find(x, t)
        return self.number_dots[x][bisect_left(self.number_time[x], t) - 1]

    def whensame(self, x, y, t = 0):
        # いつ同じになるか
        if x == y:
            return t
        if self.time[x] == self.time[y] == INF:
            return -1
        if self.time[x] > self.time[y]:
            x, y = y, x
        return self.whensame(self.parents[x], y, self.time[x])

    def same(x, y, t):
        return self.find(x, t) == self.find(y, t)
# ------------------------------- #



def base(n, k: "2 <= |k| <= 62"):
    # k進数
    if not 2 <= abs(k) <= 62:
        raise ValueError(f"Invaild Value k: {k}")
    if abs(k) > 10:
        from string import ascii_uppercase, ascii_lowercase
        charall = ascii_uppercase + ascii_lowercase
    def char(a):
        return charall[a - 10] if a > 10 else str(a)
    b = ""
    while n:
        b += char(n % abs(k))
        if k < 0:
            n = 0--n // k
        else:
            n = n // k
    return b[::-1]


from itertools import accumulate
class Imos:
    # imos法
    # itertool.accumulate 必須
    def __init__(self, n):
        self.B = [0] * n
        self.n = n

    def __call__(self, l, r, v = 1):
        l, r = max(l, 0), min(r, self.n - 1)
        self.B[l] += v
        if r + 1 != self.n:
            self.B[r + 1] -= v

    def out(self):
        *res, = accumulate(self.B)
        # self.__init__(self.n)
        return res


from bisect import*
from itertools import accumulate, chain
class ImosCompressed:
    # クエリ先読み座圧対応imos法
    # bisect, itertools.accumulate, itertools.chain.from_iterable, compress 必須
    def __init__(self, query: "[[l, r, v], ...]", rclosed = False):
        if rclosed:
            query = [[l, r + 1, v] for l, r, v in query]
        *code, = chain.from_iterable(query)
        self.s, code, _ = compress(code[::2] + code[1::2])
        n = len(code)
        B = [0] * n

        for l, r, v in query:
            B[code[l]] += v
            if code[r] != n:
                B[code[r]] -= v
        self.code = code
        *self.res, = accumulate(B)

    def get(self, x):
        return self.res[bisect(self.s, x) - 1]
 
    def sum(self, l, r):
        # Σ[l, r) O(n)
        s, res = self.s, self.res
        L, R = bisect(s, l) - 1, bisect(s, r) - 1
        ans = (s[L + 1] - l) * res[L]
        for i in range(L + 1, R):
            ans += (s[i + 1] - s[i]) * res[i]
        ans += (r - s[R]) * res[R]
        return ans
 
    def sumall(self):
        return self.sum(0, self.s[-1] + 1)



# ------------- BIT ------------- #
class Bit:
    # Binary Indexed Tree
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def __iter__(self):
        psum = 0
        for i in range(self.size):
            csum = self.sum(i + 1)
            yield csum - psum
            psum = csum
        raise StopIteration()

    def __str__(self):  # O(nlogn)
        return str(list(self))

    def sum(self, i):
        # Σ [0, i)
        if not (0 <= i <= self.size): raise ValueError("error!")
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def add(self, i, x):
        if not (0 <= i < self.size): raise ValueError("error!")
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def __getitem__(self, key):
        if not (0 <= key < self.size): raise IndexError("error!")
        return self.sum(key + 1) - self.sum(key)

    def __setitem__(self, key, value):
        if not (0 <= key < self.size): raise IndexError("error!")
        self.add(key, value - self[key])


class BitImos:　
    # Query: O(logn)
    def __init__(self, n):
        self.bit = Bit(n + 1)

    def add(self, l, r, x = 1):
        # [l, r) += x
        self.bit.add(l, x)
        self.bit.add(r, -x)

    def get(self, i):
        return self[i]

    def __getitem__(self, key):
        return self.bit.sum(key + 1)


class Bit2:
    # Query: O(logn)
    def __init__(self, n):
        self.bit0 = Bit(n)
        self.bit1 = Bit(n)

    def add(self, l, r, x = 1):
        # [l, r) += x
        self.bit0.add(l, -x * (l - 1))
        self.bit1.add(l, x)
        self.bit0.add(r, x * (r - 1))
        self.bit1.add(r, -x)

    def sum(self, l, r):
        # Σ [l,r)
        res = 0
        res += self.bit0.sum(r) + self.bit1.sum(r) * (r - 1)
        res -= self.bit0.sum(l) + self.bit1.sum(l) * (l - 1)
        return res


def mergecount(A: list):
    # 転倒数 O(max(A) + nlogn)
    bit = Bit(max(A) + 1)
    cnt = 0
    for i, a in enumerate(A):
        cnt += i - bit.sum(a + 1)
        bit.add(a, 1)
    return cnt


def mergecount_(A: list):
    # 転倒数(座圧) O(nlogn)
    bit = Bit(n + 1)
    cnt = 0
    for i, (h, d) in enumerate(zip(A, compress(A)[2])):
        cnt += h * (i - bit.sum(d + 1))
        bit.add(d, 1)
    return cnt


class Bit_:
    def __init__(self, a):
        if hasattr(a, "__iter__"):
            le = len(a)
            self.n = 1 << le.bit_length()
            self.values = values = [0] * (self.n + 1)
            values[1:le+1] = a[:]
            for i in range(1, self.n):
                values[i + (i & -i)] += values[i]
        elif isinstance(a, int):
            self.n = 1 << a.bit_length()
            self.values = [0] * (self.n + 1)
        else:
            raise TypeError

    def add(self, i, val):
        n, values = self.n, self.values
        while i <= n:
            values[i] += val
            i += i & -i

    def sum(self, i):
        # Σ(0, i]
        values = self.values
        res = 0
        while i > 0:
            res += values[i]
            i -= i & -i
        return res

    def bisect_left(self, v):
        # sum(i) >= v となる最小の i
        n, values = self.n, self.values
        if v > values[n]:
            return None
        i, step = 0, n >> 1
        while step:
            if values[i + step] < v:
                i += step
                v -= values[i]
            step >>= 1
        return i + 1
# ------------------------------- #


# --------- Segment Tree --------- #
class SegmentTree(object):
    __slots__ = ["elem_size", "tree", "default", "op", "real_size"]

    def __init__(self, a, default = float("inf"), op: "func" = min):
        self.default = default
        self.op = op
        if hasattr(a, "__iter__"):
            self.real_size = len(a)
            self.elem_size = elem_size = 1 << (self.real_size - 1).bit_length()
            self.tree = tree = [default] * (elem_size * 2)
            tree[elem_size : elem_size + self.real_size] = a
            for i in range(elem_size - 1, 0, -1):
                tree[i] = op(tree[i << 1], tree[(i << 1) + 1])
        elif isinstance(a, int):
            self.real_size = a
            self.elem_size = elem_size = 1 << (self.real_size - 1).bit_length()
            self.tree = [default] * (elem_size * 2)
        else:
            raise TypeError

    def get_value(self, x: int, y: int) -> int:   # [x, y)
        l, r = x + self.elem_size, y + self.elem_size
        tree, result, op = self.tree, self.default, self.op
        while l < r:
            if l & 1:
                result = op(tree[l], result)
                l += 1
            if r & 1:
                r -= 1
                result = op(tree[r], result)
            l, r = l >> 1, r >> 1
        return result

    def __setitem__(self, i: int, value: int) -> None:
        k = self.elem_size + i
        op, tree = self.op, self.tree
        tree[k] = value
        while k > 1:
            k >>= 1
            tree[k] = op(tree[k << 1], tree[(k << 1) + 1])

    def __getitem__(self, i):
        return self.tree[i + self.elem_size]

    def debug(self):
        print(self.tree[self.elem_size : self.elem_size + self.real_size])
# -------------------------------- #