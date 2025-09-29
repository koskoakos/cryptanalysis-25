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

d_S = P_post.T.copy()

delta_S = np.zeros_like(d_S)
for c in range(20):
    if P_C[c] <= 0:
        continue
    row = d_S[c]
    c_max = row.max()
    winners = np.isclose(row, c_max)
    cnt = int(winners.sum())
    if cnt == 0:
        continue

    delta_S[c, :] = 0.0
    delta_S[c, winners] = 1.0 / cnt

print(delta_S)

expected_correct = (delta_S * P_post.T).sum(axis=1)
loss_S = float((P_C * (1.0 - expected_correct)).sum())

print(loss_S)