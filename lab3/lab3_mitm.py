#%%
import time
from tqdm import tqdm

C = None
N = None

with open("mitm_2048_18.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        name, value = line.split(" = ")
        name = name.strip()
        value = value.strip()

        if name == "C":
            C = int(value, 16)
        elif name == "N":
            N = int(value, 16)

e = 65537
l = 20
#%%
def mod_inv(a:int, n:int):
    a = a % n
    r = [n, a]
    q = []
    i = 0

    j = r[i] % r[i+1]

    while j != 0 :
        j = r[i] % r[i+1]
        r.append(j)
        q.append(r[i] // r[i+1])
        i += 1

    if r[-2] != 1:
        return None
    else:
        v = [0, 1]

        for i in range(len(q)):
            v.append(v[i] - v[i+1] * q[i])

    if v[-2] < 0:
        return n + v[-2]
    else:
        return v[-2]

mod_inv(89, 144)
#%%
limit = 2**(l//2)+1
t0 = time.time()
X = [pow(T, e, N) for T in range(1, limit)]
S = [mod_inv(m, N) for m in X]

#%%
print(f"Generated tables in {time.time() - t0} seconds")
print(f"Starting meet-in-the-middle search")
for i, s_val in tqdm(enumerate(S)):
    if s_val is None:
        continue
    Cs = (C * s_val) % N

    if Cs in X:
        j = X.index(Cs)

        M = (i + 1) * (j + 1)
        break
t1 = time.time()
print(f"Time taken: {t1 - t0} seconds")

print(f"Brute force RSA")
t0 = time.time()
for i in tqdm(range(2**l)):
    if pow(i+1, e, N) == C:
        M = i + 1
        break
else:
    print("Bruteforce failed somehow")
    
print(f"Time for brute force: {time.time() - t0} seconds")
print(f"check: (M={hex(M)})**e mod N == C {pow(M, e, N) == C}")

#%%

bytelen = (M.bit_length() + 7) // 8
plaintext = M.to_bytes(bytelen, 'big')

plaintext.split(b'\x00')[-1]

plaintext
# %%
M
# %%
hex(M)
# %%
