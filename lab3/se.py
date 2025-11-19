# %%
C = []
N = []
with open ("SE_18_256.txt") as fin:
    while True:
        C_line = fin.readline()
        N_line = fin.readline()
        if not C_line or not N_line:
            break
        C.append(int(C_line.partition(' = ')[2], 16))
        N.append(int(N_line.partition(' = ')[2], 16))

print(f"{C=}")
print(f"{N=}")
    

# %%
from functools import reduce
NN = reduce(lambda x, y: x*y, N)

# %%
N_star = [NN // n_i for n_i in N]

# %%
C_inv = [pow(n_s, -1, n_i) for n_s, n_i in zip(N_star, N)]

# %%
C_crt = sum(c_i * n_s * c_i_inv for c_i, n_s, c_i_inv in zip(C, N_star, C_inv)) % NN

# %%
a = 0
b = C_crt
while True:
    mid = (a + b) // 2
    if pow(mid, 3, NN) < C_crt:
        a = mid
    elif pow(mid, 3, NN) > C_crt:
        b = mid
    elif pow(mid, 3, NN) == C_crt:
        print(f"Found exact root: {mid}")
        break
    

# %%
bytelen = (mid.bit_length() + 7) // 8
plaintext = mid.to_bytes(bytelen, 'big')

# %%
plaintext.split(b'\x00')[-1]

# %%
plaintext

# %%



