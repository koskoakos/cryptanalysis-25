# %%

import math, random, itertools
from collections import Counter
from nltk.util import ngrams
from pathlib import Path

random.seed(42)

UA_ALPHABET = list("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")
BIGRAM_ALPHABET = [f"{a}{b}" for a,b in itertools.product(UA_ALPHABET, UA_ALPHABET)]

def normalize(text: str) -> str:
    allowed = set(UA_ALPHABET)
    normalized = []
    for ch in text.lower():
        if ch in allowed:
            if not ch == 'ґ':
                normalized.append(ch)
            else:
                normalized.append('г')
        
    return ''.join(normalized)

corpus_path = Path("fiction.txt")
big_ua_text = corpus_path.read_text(encoding="utf-8")
corpus_clean = normalize(big_ua_text)
print("Corpus length:", len(corpus_clean))


def freq_ref_from_corpus(corpus_clean,l):
    total = max(1, len(corpus_clean) - l + 1)
    cnt = Counter(''.join(ng) for ng in ngrams(corpus_clean, l))
    return {g: c/total for g,c in cnt.items()}

FREQ_REF = {1: freq_ref_from_corpus(corpus_clean, 1), 2: freq_ref_from_corpus(corpus_clean, 2)}



# %%

def corpus_slice(corpus_text: str, length: int) -> str:
    if length > len(corpus_text):
        raise ValueError("length exceeds corpus size")
    start = random.randint(0, len(corpus_text) - length)
    return corpus_text[start:start+length]

def affine_cipher(text, alphabet, l=1, k=None):
    m = len(alphabet)
    M = m if l==1 else m*m
    if k is None:
        while True:
            a = random.randint(2, M-1)
            if math.gcd(a, M) == 1:
                break
        b = random.randint(0, M-1)
    else:
        a,b = k
    idx = {ch:i for i,ch in enumerate(alphabet)}
    if l==1:
        return ''.join(alphabet[(a*idx[ch]+b)%M] for ch in text)
    t = text
    if len(t) % 2: 
        t+=alphabet[0]
    out = []
    for i in range(0,len(t),2):
        X = idx[t[i]] * m + idx[t[i+1]]
        Y = (a*X+b) % M
        out.append(alphabet[Y//m]) 
        out.append(alphabet[Y%m])
    return ''.join(out)

def vigenere_cipher(text,key,alphabet):
    idx = {ch:i for i,ch in enumerate(alphabet)}
    m = len(alphabet)
    out= []
    j=0
    for ch in text:
        k=idx[key[j%len(key)]]
        out.append(alphabet[(idx[ch]+k)%m])
        j+=1
    return ''.join(out)

def ring_abc(alphabet,n=10):
    s0,s1=random.choice(alphabet),random.choice(alphabet)
    idx={ch:i for i,ch in enumerate(alphabet)}
    S=[s0,s1]
    for i in range(2,n):
        yi=(idx[S[i-1]]+idx[S[i-2]])%len(alphabet)
        S.append(alphabet[yi])
    return ''.join(S)

# %%

random_string=lambda k: ''.join(random.choices(UA_ALPHABET, k=k))
DISTORTION_METHODS={
    "Vigenere K1": lambda t, l:vigenere_cipher(t,random_string(1), UA_ALPHABET),
    "Vigenere K5": lambda t, l:vigenere_cipher(t,random_string(5), UA_ALPHABET),
    "Vigenere K10": lambda t, l:vigenere_cipher(t,random_string(10), UA_ALPHABET),
    "Affine uni": lambda t, l:affine_cipher(t, UA_ALPHABET),
    "Affine bigram": lambda t, l:affine_cipher(t, UA_ALPHABET, l=2),
}
RANDOM_METHODS={
    "Random uni":lambda t, l: ''.join(random.choices(UA_ALPHABET,k=l)),
    "Random bigram":lambda t, l: ''.join(random.choices(BIGRAM_ALPHABET,k=(l+1)//2)[:l]),
    "Ring uni":lambda t, l: ring_abc(UA_ALPHABET, l),
    "Ring bigram":lambda t, l: ring_abc(BIGRAM_ALPHABET, l)[:l],
}

def generate_dataset(corpus_text, L_values, N_counts, distortion_methods):
    data={}
    for L in L_values:
        N=N_counts[L]
        texts=[corpus_slice(corpus_text,L) for _ in range(N)]
        data[L]={}
        for name,func in distortion_methods.items():
            pairs=[]
            for t in texts:
                y=func(t, L)
                pairs.append((t,y))
            data[L][name]=pairs
    return data



L_values=[10, 100, 1000, 10000]
N_counts={10:10000, 
          100:10000,
          1000:10000,
          10000:1000}
distorted_texts = generate_dataset(corpus_clean, L_values, N_counts, DISTORTION_METHODS)
random_texts = generate_dataset(corpus_clean, L_values, N_counts, RANDOM_METHODS)
print("Datasets built.")


# %%

def counts_once(text, l):
    N = len(text) - l + 1
    if N <= 0:
        return Counter(), 0, 0.0
    cnt = Counter(text[i:i+l] for i in range(N))
    H = 0.0
    for c in cnt.values():
        p = c / N
        H -= p * math.log2(p)
    return cnt, N, H / l


# %%

def build_sets(freq_ref, h, mode="rare"):
    if mode=="rare":
        return [ng for ng,_ in sorted(freq_ref.items(), key=lambda kv:kv[1])[:h]]
    elif mode in {"top"}:
        return [ng for ng,_ in sorted(freq_ref.items(), key=lambda kv:kv[1], reverse=True)[:h]]


# Precompute for l=1,2
refs = {}
for l in (1,2):
    fr = freq_ref_from_corpus(corpus_clean, l)
    refs[l] = {
        "freq_ref": fr,
        "Aprh3": set(build_sets(fr, 3, "rare")),
        "Aprh10": set(build_sets(fr, 10, "rare")),
        "Top50": build_sets(fr, 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
    }


# %%

def c_1_0(cnt, Aprh):
    return any(ng in cnt for ng in Aprh)  # False = H0

def c_1_1(cnt, Aprh, k):
    return len(set(cnt.keys()) & Aprh) >= k

def c_1_2(cnt, N, Aprh, freq_ref):
    if N == 0: return True
    return any((cnt.get(ng,0)/N) > freq_ref.get(ng, 0.0) for ng in Aprh)

def c_1_3(cnt, N, Aprh, freq_ref):
    if N == 0: return True
    obs = sum(cnt.get(ng,0) for ng in Aprh) / N
    ref = sum(freq_ref.get(ng,0.0) for ng in Aprh)
    return obs > ref

def c_3_0(H_obs, H_lang, k_H):
    return abs(H_obs - H_lang) > k_H

def c_5_1(cnt, N, top_list, k_empt):
    fempt = sum(1 for ng in top_list if cnt.get(ng, 0) == 0)
    return fempt >= k_empt

def count_empty_bins(cnt, top_list):
    return sum(1 for ng in top_list if cnt.get(ng, 0) == 0)

def eval_bins(N):
    bins = build_sets(FREQ_REF[1], N, "top")
    for method, texts in distorted_texts[1000].items():
        
        print("Method:", method)
        empty_x_total = 0
        empty_y_total = 0
        for text in texts:
            cntX, NX, HX = counts_once(text[0], 1)
            cntY, NY, HY = counts_once(text[1], 1)
            empty_x_total += count_empty_bins(cntX, bins)
            empty_y_total += count_empty_bins(cntY, bins)
        print(f" Average empty bins X: {empty_x_total/len(texts):.2f}")
        print(f" Average empty bins Y: {empty_y_total/len(texts):.2f}")
              
eval_bins(32)


# %%

ORDER = ['1.0','1.1','1.2','1.3','3.0','5.1']

def evaluate_boolean_predictions(y_true_pairs, y_pred_pairs):
    TP = TN = FP = FN = 0
    n_H0 = n_H1 = 0
    for truths, preds in zip(y_true_pairs, y_pred_pairs):
        for t, p in zip(truths, preds):
            if t:
                n_H1 += 1
                if p:
                    TP += 1
                else:
                    FN += 1
            else:
                n_H0 += 1
                if p:
                    FP += 1
                else:
                    TN += 1

    alpha = FP / n_H0 if n_H0 else 0.0
    beta = FN / n_H1 if n_H1 else 0.0
    return {'alpha': alpha, 'beta': beta}

def eval_distorted(distorted_texts, refs, order=ORDER, limit=0):
    for L in sorted(distorted_texts.keys()):
        for l in (1,2):
            fr      = refs[l]["freq_ref"]
            Aprh1_0   = refs[l]["Aprh1_0"]
            Aprh1_1  = refs[l]["Aprh1_1"]
            Aprh1_2  = refs[l]["Aprh1_2"]
            Aprh1_3  = refs[l]["Aprh1_3"]
            Top50   = refs[l]["Top50"]
            H_lang  = refs[l]["H_lang"]
            k_11 = refs[l]["k_11"]
            k_empt = refs[l]['k_empt']
            k_H = refs[l]['k_H']
            for name, pairs in distorted_texts[L].items():
                y_true = []
                preds = {c: [] for c in order}
                HH = []
                limit_counter = 0
                for (x, y) in pairs:
                    if limit and limit_counter >= limit:
                        break
                    limit_counter += 1
                    cntX, NX, HX = counts_once(x, l)
                    cntY, NY, HY = counts_once(y, l)
                    y_true.append([False, True])
                    if '1.0' in order:
                        preds['1.0'].append([c_1_0(cntX, Aprh1_0), c_1_0(cntY, Aprh1_0)])
                    if '1.1' in order:
                        preds['1.1'].append([c_1_1(cntX, Aprh1_1, k_11), c_1_1(cntY, Aprh1_1, k_11)])
                    if '1.2' in order:
                        preds['1.2'].append([c_1_2(cntX, NX, Aprh1_2, fr), c_1_2(cntY, NY, Aprh1_2, fr)])
                    if '1.3' in order:
                        preds['1.3'].append([c_1_3(cntX, NX, Aprh1_3, fr), c_1_3(cntY, NY, Aprh1_3, fr)])
                    if '3.0' in order:
                        preds['3.0'].append([c_3_0(HX, H_lang, k_H), c_3_0(HY, H_lang, k_H)])
                    if '5.1' in order:
                        preds['5.1'].append([c_5_1(cntX, NX, Top50, k_empt), c_5_1(cntY, NY, Top50, k_empt)])
                    if '6.0' in order:
                        preds['6.0'].append([criterion_6(x, refs[l]['comp_threshold'], refs[l]['comp_method']), 
                                             criterion_6(y, refs[l]['comp_threshold'], refs[l]['comp_method'])])
                    HH.append((HX, HY))
                print(f"Distorted | L={L} | l={l} | {name}")
                for c in order:
                    m = evaluate_boolean_predictions(y_true, preds[c])
                    print(f"  {c}: α={m['alpha']:.3f} β={m['beta']:.3f}")
                if '3.0' in order:
                    print(f"Average entropy differences for X: {sum(abs(hx - H_lang) for hx, hy in HH)/len(HH):.4f}")
                    print(f"Average entropy differences for Y: {sum(abs(hy - H_lang) for hx, hy in HH)/len(HH):.4f}")
    return preds


# %%
dist_10 = {10: distorted_texts[10]}
dist_100 = {100: distorted_texts[100]}
dist_1000 = {1000: distorted_texts[1000]}
dist_10000 = {10000: distorted_texts[10000]}

rand_10 = {10: random_texts[10]}


refs_10 = {
    1: {
        "freq_ref": FREQ_REF[1],
        "Aprh1_0": set(build_sets(FREQ_REF[1], 4, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[1], 18, "rare")),
        "Aprh1_2": set(build_sets(FREQ_REF[1], 4, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[1], 4, "rare")),
        "k_11": 4,
        "Top50": build_sets(FREQ_REF[1], 10, "top"),
        "k_empt": 7,
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 1.31,
    
    },
    2: {
        "freq_ref": FREQ_REF[2],
        "Aprh1_0": set(build_sets(FREQ_REF[2], 256, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[2], 384, "rare")),  
        "Aprh1_2": set(build_sets(FREQ_REF[2], 256, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[2], 256, "rare")),
        "k_11": 3,
        "Top50": build_sets(FREQ_REF[2], 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 2.669,
        "k_empt": 49,
    }
}

d = eval_distorted(dist_10, refs_10, order=['1.0'])

# %%
d = eval_distorted(rand_10, refs_10, order=['1.0'])

# %%
d = eval_distorted(dist_10, refs_10, order=['1.1', '1.2', '1.3'])

# %%
d = eval_distorted(rand_10, refs_10, order=['1.1', '1.2', '1.3'])

# %%
d = eval_distorted(dist_10, refs_10, order=['3.0'])

# %%
d = eval_distorted(rand_10, refs_10, order=['3.0'])

# %%
d = eval_distorted(dist_10, refs_10, order=['5.1'])
d = eval_distorted(rand_10, refs_10, order=['5.1'])

# %%


# %%


# %%



# %%
refs_100 = {
    1: {
        "freq_ref": FREQ_REF[1],
        "Aprh1_0": set(build_sets(FREQ_REF[1], 1, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[1], 4, "rare")),
        "Aprh1_2": set(build_sets(FREQ_REF[1], 2, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[1], 2, "rare")),
        "k_11": 3,
        "Top50": build_sets(FREQ_REF[1], 10, "top"),
        "k_empt": 1,
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.4,
        
    },
    2: {
        "freq_ref": FREQ_REF[2],
        "Aprh1_0": set(build_sets(FREQ_REF[2], 160, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[2], 360, "rare")),
        "Aprh1_2": set(build_sets(FREQ_REF[2], 200, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[2], 200, "rare")),
        "k_11": 15,
        "Top50": build_sets(FREQ_REF[2], 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 1.069,
        "k_empt": 30,
    }
}
d = eval_distorted(dist_100, refs_100, order=['1.0', '1.1', '1.2', '1.3'])

# %%
rand_100 = {100: random_texts[100]}
d = eval_distorted(rand_100, refs_100, order=['1.0', '1.1', '1.2', '1.3'])

# %%

d = eval_distorted(dist_100, refs_100, order=['3.0'])

# %%

d = eval_distorted(rand_100, refs_100, order=['3.0'])

# %%
refs_100[1]["Top50"] = build_sets(FREQ_REF[1], 15, "top")
refs_100[1]["k_empt"] = 1
refs_100[2]["Top50"] = build_sets(FREQ_REF[2], 50, "top")
refs_100[2]["k_empt"] = 30
d = eval_distorted(dist_100, refs_100, order=['5.1'])
d = eval_distorted(rand_100, refs_100, order=['5.1'])


# %%
refs_100[2]["Top50"] = build_sets(FREQ_REF[2], 100, "top")
refs_100[2]["k_empt"] = 80
d = eval_distorted(dist_100, refs_100, order=['5.1'])
d = eval_distorted(rand_100, refs_100, order=['5.1'])

# %%
refs_100[2]["Top50"] = build_sets(FREQ_REF[2], 200, "top")
refs_100[2]["k_empt"] = 180
d = eval_distorted(dist_100, refs_100, order=['5.1'])
d = eval_distorted(rand_100, refs_100, order=['5.1'])

# %%


# %%


# %%


# %%
refs_1000 = {
    1: {
        "freq_ref": FREQ_REF[1],
        "Aprh1_0": set(build_sets(FREQ_REF[1], 1, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[1], 3, "rare")),
        "Aprh1_2": set(build_sets(FREQ_REF[1], 1, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[1], 15, "rare")),
        "k_11": 3,
        "Top50": build_sets(FREQ_REF[1], 10, "top"),
        "k_empt": 1,
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.4,
    },
    2: {
        "freq_ref": FREQ_REF[2],
        "Aprh1_0": set(build_sets(FREQ_REF[2], 10, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[2], 64, "rare")),  
        "Aprh1_2": set(build_sets(FREQ_REF[2], 10, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[2], 10, "rare")),
        "k_11": 8,
        "Top50": build_sets(FREQ_REF[2], 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.25,
        "k_empt": 70,
    }
}

rand_1000 = {1000: random_texts[1000]}
d = eval_distorted(dist_1000, refs_1000, order=['1.0', '1.1', '1.2', '1.3'])
d = eval_distorted(rand_1000, refs_1000, order=['1.0', '1.1', '1.2', '1.3'])

# %%
refs_1000 = {
    1: {
        "freq_ref": FREQ_REF[1],
        "Aprh1_0": set(build_sets(FREQ_REF[1], 1, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[1], 3, "rare")),
        "Aprh1_2": set(build_sets(FREQ_REF[1], 1, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[1], 2, "rare")),
        "k_11": 3,
        "Top50": build_sets(FREQ_REF[1], 10, "top"),
        "k_empt": 1,
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.4,
    },
    2: {
        "freq_ref": FREQ_REF[2],
        "Aprh1_0": set(build_sets(FREQ_REF[2], 192, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[2], 256, "rare")),  
        "Aprh1_2": set(build_sets(FREQ_REF[2], 128, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[2], 128, "rare")),
        "k_11": 2,
        "Top50": build_sets(FREQ_REF[2], 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.25,
        "k_empt": 70,
    }
}

d = eval_distorted(dist_1000, refs_1000, order=['3.0'])
d = eval_distorted(rand_1000, refs_1000, order=['3.0'])

# %%
refs_1000[1]["Top50"] = build_sets(FREQ_REF[1], 32, "top")
refs_1000[1]["k_empt"] = 1
refs_1000[2]["Top50"] = build_sets(FREQ_REF[2], 50, "top")
refs_1000[2]["k_empt"] = 12
d = eval_distorted(dist_1000, refs_1000, order=['5.1'])
d = eval_distorted(rand_1000, refs_1000, order=['5.1'])

# %%

refs_1000[2]["Top50"] = build_sets(FREQ_REF[2], 100, "top")
refs_1000[2]["k_empt"] = 20
d = eval_distorted(dist_1000, refs_1000, order=['5.1'])
d = eval_distorted(rand_1000, refs_1000, order=['5.1'])

# %%
refs_1000
rand_1000 = {1000: random_texts[1000]}
refs_1000[2]["Top50"] = build_sets(FREQ_REF[2], 150, "top")
refs_1000[2]["k_empt"] = 24
d = eval_distorted(dist_1000, refs_1000, order=['5.1'])
d = eval_distorted(rand_1000, refs_1000, order=['5.1'])

# %%
dist_10000 = {10000: distorted_texts[10000]}
refs_10000 = {
    1: {
        "freq_ref": FREQ_REF[1],
        "Aprh1_0": set(build_sets(FREQ_REF[1], 1, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[1], 10, "rare")),
        "Aprh1_2": set(build_sets(FREQ_REF[1], 2, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[1], 2, "rare")),
        "k_11": 9,
        "Top50": build_sets(FREQ_REF[1], 10, "top"),
        "k_empt": 1,
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.4,
        
    },
    2: {
        "freq_ref": FREQ_REF[2],
        "Aprh1_0": set(build_sets(FREQ_REF[2], 512, "rare")),
        "Aprh1_1": set(build_sets(FREQ_REF[2], 256, "rare")),  
        "Aprh1_2": set(build_sets(FREQ_REF[2], 128, "rare")),
        "Aprh1_3": set(build_sets(FREQ_REF[2], 128, "rare")),
        "k_11": 2,
        "Top50": build_sets(FREQ_REF[2], 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
        "k_H": 0.3,
        "k_empt": 70,
    }
}
rand_10000 = {10000: random_texts[10000]}

# %%
refs_10000[1]["Aprh1_0"] = set(build_sets(FREQ_REF[1], 1, "rare"))
refs_10000[2]["Aprh1_0"] = set(build_sets(FREQ_REF[2], 5, "rare"))
d = eval_distorted(dist_10000, refs_10000, order=['1.0'])
d = eval_distorted(rand_10000, refs_10000, order=['1.0'])

# %%
refs_10000[1]["Aprh1_1"] = set(build_sets(FREQ_REF[1], 3, "rare"))
refs_10000[1]["k_11"] = 3
refs_10000[2]["Aprh1_1"] = set(build_sets(FREQ_REF[2], 25, "rare"))
refs_10000[2]["k_11"] = 10
d = eval_distorted(dist_10000, refs_10000, order=['1.1'])
d = eval_distorted(rand_10000, refs_10000, order=['1.1'])

# %%
refs_10000[1]["Aprh1_2"] = set(build_sets(FREQ_REF[1], 1, "rare"))
refs_10000[2]["Aprh1_2"] = set(build_sets(FREQ_REF[2], 5, "rare"))
d = eval_distorted(dist_10000, refs_10000, order=['1.2'])
d = eval_distorted(rand_10000, refs_10000, order=['1.2'])

# %%
refs_10000[1]["Aprh1_3"] = set(build_sets(FREQ_REF[1], 1, "rare"))
refs_10000[2]["Aprh1_3"] = set(build_sets(FREQ_REF[2], 4, "rare"))
d = eval_distorted(dist_10000, refs_10000, order=['1.3'])
d = eval_distorted(rand_10000, refs_10000, order=['1.3'])

# %%
refs_10000[1]["k_H"] = 0.35
refs_10000[2]["k_H"] = 0.42

d = eval_distorted(dist_10000, refs_10000, order=['3.0'])
d = eval_distorted(rand_10000, refs_10000, order=['3.0'])

# %%
refs_10000[1]["Top50"] = build_sets(FREQ_REF[1], 32, "top")
refs_10000[1]["k_empt"] = 1
refs_10000[2]["Top50"] = build_sets(FREQ_REF[2], 50, "top")
refs_10000[2]["k_empt"] = 1
d = eval_distorted(dist_10000, refs_10000, order=['5.1'])
d = eval_distorted(rand_10000, refs_10000, order=['5.1'])

# %%

refs_10000[2]["Top50"] = build_sets(FREQ_REF[2], 100, "top")
refs_10000[2]["k_empt"] = 2
d = eval_distorted(dist_10000, refs_10000, order=['5.1'])
d = eval_distorted(rand_10000, refs_10000, order=['5.1'])

# %%

refs_10000[2]["Top50"] = build_sets(FREQ_REF[2], 150, "top")
refs_10000[2]["k_empt"] = 3
d = eval_distorted(dist_10000, refs_10000, order=['5.1'])
d = eval_distorted(rand_10000, refs_10000, order=['5.1'])

# %%
refs_10[2]["Top50"] = build_sets(FREQ_REF[2], 200, "top")
refs_10[2]["k_empt"] = 198
d = eval_distorted(dist_10, refs_10, order=['5.1'])
d = eval_distorted(rand_10, refs_10, order=['5.1'])

# %%
refs_10[2]["Top50"] = build_sets(FREQ_REF[2], 100, "top")
refs_10[2]["k_empt"] = 98
d = eval_distorted(dist_10, refs_10, order=['5.1'])
d = eval_distorted(rand_10, refs_10, order=['5.1'])

# %%


# %%
import zlib, random, math
from collections import Counter


def alphabet_of(text: str) -> str:
    return "".join(k for k, _ in Counter(text).most_common())

def sample_random_like(text: str, n: int) -> str:
    alpha = alphabet_of(text) or UA_ALPHABET
    return "".join(random.choice(alpha) for _ in range(len(text)))

def ratio_zlib(s: str, level: int = 1) -> float:
    b = s.encode("utf-8")
    if not b:
        return 1.0
    c = zlib.compress(b, level)
    return len(c) / len(b)


UPPER_UA = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
LOWER_UA = UA_ALPHABET

def _enc_letters(n: int) -> str:
    if n <= 0:
        return UPPER_UA[0]
    base = len(UPPER_UA)
    out = []
    while n > 0:
        n -= 1
        out.append(UPPER_UA[n % base])
        n //= base
    return "".join(reversed(out))

def compress_rle(s: str) -> str:
    if not s:
        return ""
    out = []
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        run_len = j - i
        out.append(s[i] + _enc_letters(run_len))
        i = j
    return "".join(out)

def ratio_rle(s: str) -> float:
    if not s:
        return 1.0
    c = compress_rle(s)
    return len(c) / len(s)


from collections import Counter
import random

LOW = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UP  = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"  # codeletters
SEP = "АЬ"     # header entry sep (letters-only, won’t appear in body)
KV  = "ОЬ"     # header key/value sep
END = "ЯЬ"   # end-of-header



s = """Ти знаєш, що ти — людина.
Ти знаєш про це чи ні?
Усмішка твоя — єдина,
Мука твоя — єдина,
Очі твої — одні.

Більше тебе не буде.
Завтра на цій землі
Інші ходитимуть люди,
Інші кохатимуть люди —
Добрі, ласкаві й злі.

Сьогодні усе для тебе —
Озера, гаї, степи.
І жити спішити треба,
Кохати спішити треба —
Гляди ж не проспи!

Бо ти на землі — людина,
І хочеш того чи ні —
Усмішка твоя — єдина,
Мука твоя — єдина,
Очі твої — одні."""

t = normalize(s) 

LOW = "абвгдеєжзиіїйклмнопрстуфхцчшщьюя"   
UP  = "АБВГДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ" 

SEP = "ЮЯ"   
KV  = "ЮЄ"     
END = "ЩЩЩЩ"
SPACE_AFTER_HEADER = " "


def _pair_counts(seq: list[str]) -> Counter:
    return Counter(a + b for a, b in zip(seq, seq[1:]))

def _apply_pair(seq: list[str], pair: str, code: str) -> list[str]:
    out = []
    i, n = 0, len(seq)
    while i < n:
        if i + 1 < n and (seq[i] + seq[i+1]) == pair:
            out.append(code)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out

def _header(mapping: list[tuple[str, str]]) -> str:
    return "".join(c + KV + p + SEP for p, c in mapping) + END + SPACE_AFTER_HEADER

def compress_bpe_letters(text: str, max_merges: int = 26, min_count: int = 3) -> str:
    s = normalize(text)
    if not s:
        return END
    seq = list(s)
    mapping: list[tuple[str, str]] = []

    code_iter = iter(UP) 
    for _ in range(max_merges):
        counts = _pair_counts(seq)
        if not counts:
            break
        pair, cnt = counts.most_common(1)[0]
        if cnt < min_count:
            break
        try:
            code = next(code_iter)
        except StopIteration:
            break
        seq = _apply_pair(seq, pair, code)
        mapping.append((pair, code))

    header = _header(mapping)
    body = "".join(seq)
    return header + body

def decompress_bpe_letters(blob: str) -> str:
    pos = blob.find(END)
    if pos == -1:
        return blob 
    hdr = blob[:pos]
    body = blob[pos + len(END):]
    if body.startswith(SPACE_AFTER_HEADER):
        body = body[1:]

    dec: dict[str, str] = {}
    i = 0
    while i < len(hdr):
        j = hdr.find(SEP, i)
        if j == -1:
            break
        entry = hdr[i:j]
        m = entry.find(KV)
        if m != -1:
            code = entry[:m] 
            pair = entry[m+len(KV):]  
            if code and pair:
                dec[code] = pair
        i = j + len(SEP)

    changed = True
    while changed:
        changed = False
        out = []
        for ch in body:
            if ch in dec:
                out.append(dec[ch])
                changed = True
            else:
                out.append(ch)
        body = "".join(out)

    return body

def ratio_bpe(s: str) -> float:
    s = normalize(s)
    c = compress_bpe_letters(s)
    return len(c) / len(s)


compression_methods = {
    'bpe': ratio_bpe,
    'rle': ratio_rle,
    'zlib': ratio_zlib,
}

def compression_criterion(
    text: str,
    method: Callable
):
    text = normalize(text)
    r_text = method(text)
    return r_text

def criterion_6(text: str, threshold: float = 0.9, method: str = 'zlib') -> bool:
    ratio = compression_criterion(text, compression_methods[method])
    return ratio > threshold  # True = H1 (not random), False = H0 (random)



sample_ua_text = """
Відколи Івана Дідуха запам'ятали в селі газдою, відтоді він мав усе лиш одного коня і малий візок із дубовим дишлем. Коня запрягав у підруку, сам себе в борозну; на коня мав ремінну шлею і нашильник, а на себе Іван накладав малу мотузяну шлею. Нашильника не потребував, бо лівою рукою спирав, може, ліпше, як нашильником.
То як тягнули снопи з поля або гній у поле, то однако і на коні, і на Івані жили виступали, однако їм обом під гору посторонки моцувалися, як струнви, і однако з гори волочилися по землі. Догори ліз кінь як по леду, а Івана як коли би хто буком по чолі тріснув, така велика жила напухала йому на чолі. Згори кінь виглядав, як би Іван його повісив на нашильнику за якусь велику провину, а ліва рука Івана обвивалася сітею синіх жил, як ланцюгом із синьої сталі.
Не раз ранком, іще перед сходом сонця, їхав Іван у поле пільною доріжкою. Шлеї не мав на собі, лише йшов із правого боку і тримав дишель як би під пахою. І кінь, і Іван держалися крепко, бо оба відпочали через ніч. То як їм траплялося сходити з горба, то бігли. Бігли вдолину і лишали за собою сліди коліс, копит і широчезних п'ят Іванових. Придорожнє зілля і бадилля гойдалося, вихолітувалося на всі боки за возом і скидало росу на ті сліди. Але часом серед найбільшого розгону на самій середині гори Іван починав налягати на ногу і спирав коня. Сідав коло дороги, брав ногу в руки і слинив, аби найти те місце, де бодяк забився.
— Та цу ногу сапов шкребчи, не ти її слинов промивай, — говорив Іван іспересердя.
— Діду Іване, а батюгов того борозного, най біжить, коли овес поїдає...
Відкрити аудіокнигу на YouTube
Це хтось так брав на сміх Івана, що видів його патороч зі свого поля. Але Іван здавна привик до таких сміхованців і спокійно тягнув бодяк дальше. Як не міг бодяка витягнути, то кулаком його вгонив далі в ногу і, встаючи, казав:
— Не біси, вігниєш та й сам віпадеш, а я не маю чєсу з тобою панькатися...
"""


# %%

refs_1000[1]['comp_threshold'] = 0.48
refs_1000[1]['comp_method'] = 'zlib'
refs_1000[2]['comp_threshold'] = 0.48
refs_1000[2]['comp_method'] = 'zlib'


d = eval_distorted(rand_1000, refs_1000, order=['6.0'], limit=500)

# %%
refs_1000[1]['comp_threshold'] = 1.95
refs_1000[1]['comp_method'] = 'rle'

d = eval_distorted(rand_1000, refs_1000, order=['6.0'], limit=500)

# %%
refs_1000[1]['comp_threshold'] = 0.99
refs_1000[1]['comp_method'] = 'bpe'

d = eval_distorted(rand_1000, refs_1000, order=['6.0'], limit=100)

# %%
compression_results = {}
print("Compression ratios on random texts:")
for L, texts in random_texts.items():
    compression_results[L] = {method: [] for method in compression_methods.keys()}
    for method, samples in texts.items():
        compression_results[L] = {method: [func(sample[1]) for sample in samples] 
                                      for method, func in compression_methods.items()}  
    for method, ratios in compression_results[L].items():
        print(f"L={L} | {method.upper()}: mean={sum(ratios)/len(ratios):.4f}")
   
    
print("Compression ratios on plain texts:")
for L in L_values:
    plain_texts = [corpus_slice(corpus_clean, L) for _ in range(N_counts[L])]
    ratios = {method: [func(text) for text in plain_texts] for method, func in compression_methods.items()}
    print(f"L={L} | {method.upper()}: mean={sum(ratios[method])/len(ratios[method]):.4f}")

# %%
compression_results = {}
print("Compression ratios on random texts:")

for L, texts in rand_10000.items():
    compression_results[L] = {method: [] for method in compression_methods.keys()}
    limit_count = 0
    for method, samples in texts.items():

        for comp_method, func in compression_methods.items():
            print(f"Processing L={L}, method={method}, comp_method={comp_method}")
            compression_results[L][comp_method] = [func(sample[1]) for sample in samples[:100]]
        break
    for method, ratios in compression_results[L].items():
        print(f"L={L} | {method.upper()}: mean={sum(ratios)/len(ratios):.4f}")
   
    
print("Compression ratios on plain texts:")
for L in [10000]:
    plain_texts = [corpus_slice(corpus_clean, L) for _ in range(N_counts[L])]
    comp_results = {method: [] for method in compression_methods.keys()}
    for comp_method, func in compression_methods.items():
        comp_results[comp_method] = [func(text) for text in plain_texts]
    for method, ratios in comp_results.items():
        print(f"L={L} | {method.upper()}: mean={sum(ratios)/len(ratios):.4f}")



# %%


# %%



