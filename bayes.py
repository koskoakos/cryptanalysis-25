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

