
from pathlib import Path
from functools import reduce
from time import perf_counter


def read_values(filename: Path):
    C = []
    N = []
    with open(filename) as fin:
        while True:
            C_line = fin.readline()
            N_line = fin.readline()
            if not C_line or not N_line:
                break
            C.append(int(C_line.partition(' = ')[2], 16))
            N.append(int(N_line.partition(' = ')[2], 16))
    return C, N

def CRT(C, N):
    NN = reduce(lambda x, y: x * y, N)
    N_star = [NN // n_i for n_i in N]
    C_inv = [pow(n_s, -1, n_i,) for n_s, n_i in zip(N_star, N)]
    C_crt = sum(c_i * n_s * c_i_inv for c_i, n_s, c_i_inv in zip(C, N_star, C_inv)) % NN
    return C_crt, NN

def root_binary_search(a: int, n: int) -> int:
    low, high = 1, 1 << ((a.bit_length() + n - 1) // n + 1)  # ~2*root upper bound
    while low < high:
        mid = (low + high) // 2
        mid_n = mid ** n
        if mid_n == a:
            return mid
        if mid_n < a:
            low = mid + 1
        else:
            high = mid
    return low - 1


def root_newton(a: int, n: int) -> int:
    x = a
    while True:
        y = ((n - 1) * x + a // (x ** (n - 1))) // n
        if y >= x:
            return x
        x = y


C, N = read_values(Path('SE_18_1024.txt'))

t0 = perf_counter()
C_crt, NN = CRT(C, N)
root_bin = root_binary_search(C_crt, 5)
dt = perf_counter() - t0
print(f'Binary search root took {dt:.6f}s')
t0 = perf_counter()
C_crt, NN = CRT(C, N)
root_newt = root_newton(C_crt, 5)
dt = perf_counter() - t0
print(f'Newton root took {dt:.6f}s')

print(f"Plaintext = {hex(root_newt)}")
