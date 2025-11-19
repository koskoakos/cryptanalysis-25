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
for n1 in N:
    for n2 in N:
        print(gcd(n1, n2))

# %%
gcd(N[0], N[1])

# %%



