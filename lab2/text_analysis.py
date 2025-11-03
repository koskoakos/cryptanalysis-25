


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
        
    return ''.join(ch for ch in text.lower() if ch in allowed)

corpus_path = Path("fiction.txt")
big_ua_text = corpus_path.read_text(encoding="utf-8")
corpus_clean = normalize(big_ua_text)
print("Corpus length:", len(corpus_clean))


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
    t=text
    if len(t)%2: t+=alphabet[0]
    out=[]
    for i in range(0,len(t),2):
        X=idx[t[i]]*m+idx[t[i+1]]
        Y=(a*X+b)%M
        out.append(alphabet[Y//m]); out.append(alphabet[Y%m])
    return ''.join(out)

def vigenere_cipher(text,key,alphabet):
    idx={ch:i for i,ch in enumerate(alphabet)}
    m=len(alphabet)
    out=[]
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


random_string=lambda k: ''.join(random.choices(UA_ALPHABET,k=k))
DISTORTION_METHODS={
    "Vigenere K1":lambda t:vigenere_cipher(t,random_string(1),UA_ALPHABET),
    "Vigenere K5":lambda t:vigenere_cipher(t,random_string(5),UA_ALPHABET),
    "Vigenere K10":lambda t:vigenere_cipher(t,random_string(10),UA_ALPHABET),
    "Affine uni":lambda t:affine_cipher(t,UA_ALPHABET),
    "Affine bigram":lambda t:affine_cipher(t,UA_ALPHABET,l=2),
}
RANDOM_METHODS={
    "Random uni":lambda l: ''.join(random.choices(UA_ALPHABET,k=l)),
    "Random bigram":lambda l: ''.join(random.choices(BIGRAM_ALPHABET,k=(l+1)//2))[:l],
    "Ring uni":lambda l: ring_abc(UA_ALPHABET,l),
    "Ring bigram":lambda l: ring_abc(BIGRAM_ALPHABET,l)[:l],
}

def generate_dataset(corpus_text,L_values,N_counts):
    data={}
    for L in L_values:
        N=N_counts[L]
        texts=[corpus_slice(corpus_text,L) for _ in range(N)]
        data[L]={}
        for name,func in DISTORTION_METHODS.items():
            pairs=[]
            for t in texts:
                y=func(t)
                if len(y)>L: y=y[:L]
                elif len(y)<L: y+=UA_ALPHABET[0]*(L-len(y))
                pairs.append((t,y))
            data[L][name]=pairs
    return data

def generate_random(L_values,N_counts):
    data={}
    for L in L_values:
        N=N_counts[L]
        data[L]={}
        for name,func in RANDOM_METHODS.items():
            data[L][name]=[func(L) for _ in range(N)]
    return data

L_values=[10,100,1000,10000]
N_counts={10:10000,100:10000,1000:10000,10000:1000}
distorted_texts=generate_dataset(corpus_clean,L_values,N_counts)
random_texts=generate_random(L_values,N_counts)
print("Datasets built.")



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



def freq_ref_from_corpus(corpus_clean,l):
    total = max(1, len(corpus_clean) - l + 1)
    cnt = Counter(''.join(ng) for ng in ngrams(corpus_clean, l))
    return {g: c/total for g,c in cnt.items()}

def build_sets(freq_ref,h,mode="rare"):
    if mode=="rare":
        return [ng for ng,_ in sorted(freq_ref.items(), key=lambda kv:kv[1])[:h]]
    elif mode in {"top"}:
        return [ng for ng,_ in sorted(freq_ref.items(), key=lambda kv:kv[1], reverse=True)[:h]]


# Precompute
refs = {}
for l in (1,2):
    fr = freq_ref_from_corpus(corpus_clean, l)
    refs[l] = {
        "freq_ref": fr,
        "Aprh3": set(build_sets(fr, 3, "rare")),
        "Aprh10": set(build_sets(fr,10, "rare")),
        "Top50": build_sets(fr, 50, "top"),
        "H_lang": counts_once(corpus_clean, l)[2],
    }
print(f"Reference stats prepared for l=1,2 {refs}")


def c_1_0(cnt, N, Aprh3):  # 1.0
    return any(ng in cnt for ng in Aprh3)

def c_1_1(cnt, N, Aprh10, k=3):  # 1.1
    occurrences = sum(cnt.get(ng, 0) for ng in Aprh10)
    return occurrences >= k

def c_1_2(cnt, N, Aprh10, freq_ref):  # 1.2
    if N == 0: return True
    return any( (cnt.get(ng,0)/N) > freq_ref.get(ng, 0.0) for ng in Aprh10 )

def c_1_3(cnt, N, Aprh10, freq_ref):  # 1.3
    if N == 0: return True
    obs = sum(cnt.get(ng,0) for ng in Aprh10) / N
    ref = sum(freq_ref.get(ng,0.0) for ng in Aprh10)
    return obs > ref

def c_3_0(H_obs, H_lang, k_H=0.1):  
    return abs(H_obs - H_lang) > k_H

def c_5_1(cnt, N, top_list, k_empt=5):  # 5.1
    fempt = sum(1 for ng in top_list if cnt.get(ng, 0) == 0)
    return fempt >= k_empt


# %%

ORDER = ['1.0','1.1','1.2','1.3','3.0','5.1']

def evaluate_boolean_predictions(y_true,y_pred):
    TP=sum(t and p for t,p in zip(y_true,y_pred))
    TN=sum((not t) and (not p) for t,p in zip(y_true,y_pred))
    FP=sum((not t) and p for t,p in zip(y_true,y_pred))
    FN=sum(t and (not p) for t,p in zip(y_true,y_pred))
    n_H0=sum(not t for t in y_true)
    n_H1=sum(t for t in y_true)
    alpha=FP/n_H0 if n_H0 else 0.0
    beta=FN/n_H1 if n_H1 else 0.0
    return dict(alpha=alpha,beta=beta)

def run_eval(distorted_texts, random_texts):
    for L in sorted(distorted_texts.keys()):
        # Distorted: paired H0 vs H1
        for l in (1,2):
            fr      = refs[l]["freq_ref"]
            Aprh3   = refs[l]["Aprh3"]
            Aprh10  = refs[l]["Aprh10"]
            Top50   = refs[l]["Top50"]
            H_lang  = refs[l]["H_lang"]

            for name, pairs in distorted_texts[L].items():
                y_true=[]; preds={c:[] for c in ORDER}
                for (x,y) in pairs:
                    cntX, NX, HX = counts_once(x, l)
                    cntY, NY, HY = counts_once(y, l)
                    y_true.extend([False, True])
                    preds['1.0'].extend([c_1_0(cntX,NX,Aprh3), c_1_0(cntY,NY,Aprh3)])
                    preds['1.1'].extend([c_1_1(cntX,NX,Aprh10,3), c_1_1(cntY,NY,Aprh10,3)])
                    preds['1.2'].extend([c_1_2(cntX,NX,Aprh10,fr), c_1_2(cntY,NY,Aprh10,fr)])
                    preds['1.3'].extend([c_1_3(cntX,NX,Aprh10,fr), c_1_3(cntY,NY,Aprh10,fr)])
                    preds['3.0'].extend([c_3_0(HX,H_lang), c_3_0(HY,H_lang)])
                    preds['5.1'].extend([c_5_1(cntX,NX,Top50,5), c_5_1(cntY,NY,Top50,5)])
                print(f"Distorted | L={L} | l={l} | {name}")
                for c in ORDER:
                    m = evaluate_boolean_predictions(y_true, preds[c])
                    print(f"  {c}: α={m['alpha']:.3f} β={m['beta']:.3f}")

        # Random: ignore α
        for l in (1,2):
            fr      = refs[l]["freq_ref"]
            Aprh3   = refs[l]["Aprh3"]
            Aprh10  = refs[l]["Aprh10"]
            Top50   = refs[l]["Top50"]
            H_lang  = refs[l]["H_lang"]

            for name, samples in random_texts[L].items():
                originals=[corpus_slice(corpus_clean, L) for _ in range(len(samples))]
                y_true=[]; preds={c:[] for c in ORDER}
                for x,y in zip(originals, samples):
                    cntX, NX, HX = counts_once(x, l)
                    cntY, NY, HY = counts_once(y, l)
                    y_true.extend([False, True])
                    preds['1.0'].extend([c_1_0(cntX,NX,Aprh3), c_1_0(cntY,NY,Aprh3)])
                    preds['1.1'].extend([c_1_1(cntX,NX,Aprh10,3), c_1_1(cntY,NY,Aprh10,3)])
                    preds['1.2'].extend([c_1_2(cntX,NX,Aprh10,fr), c_1_2(cntY,NY,Aprh10,fr)])
                    preds['1.3'].extend([c_1_3(cntX,NX,Aprh10,fr), c_1_3(cntY,NY,Aprh10,fr)])
                    preds['3.0'].extend([c_3_0(HX,H_lang), c_3_0(HY,H_lang)])
                    preds['5.1'].extend([c_5_1(cntX,NX,Top50,5), c_5_1(cntY,NY,Top50,5)])
                print(f"Random | L={L} | l={l} | {name}")
                for c in ORDER:
                    m = evaluate_boolean_predictions(y_true, preds[c])
                    print(f"  {c}: α={m['alpha']:.3f} β={m['beta']:.3f}")




run_eval(distorted_texts, random_texts)

# %%



