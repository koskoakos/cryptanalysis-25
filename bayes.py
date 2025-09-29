import numpy as np
import pandas as pd

PROB_CSV = "prob_18.csv"
TABLE_CSV = "table_18.csv"

prob = pd.read_csv(PROB_CSV, header=None).values

P_M = prob[0]
P_K = prob[1]

assert(np.isclose(sum(P_M), 1))
assert(np.isclose(sum(P_K), 1))
print(f"{P_M.sum()=}")
print(f"{P_K.sum()=}")

E = pd.read_csv(TABLE_CSV, header=None).values
print(E[:2])

P_C = np.zeros(20, dtype=float)
P_joint = np.zeros((20, 20), dtype=float)

for key in range(20):
    for m in range(20):
        c = E[key, m]
        mk = P_M[m] * P_K[key]
        P_C[c] += mk
        P_joint[m, c] += mk

P_post = P_joint/P_C

d_B = np.argmax(P_post, axis=0)
print(f"{d_B=}")

loss_D = float(np.sum(P_C * (1.0 - P_post.max(axis=0))))
print(f"{loss_D=}")


