# %%
import math, random, json, itertools, collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True, parents=True)
(TABLES := RESULTS_DIR / "tables").mkdir(exist_ok=True, parents=True)
(FIGS := RESULTS_DIR / "figs").mkdir(exist_ok=True, parents=True)


# %%
import itertools
UA_ALPHABET = list("абвгдеєжзиіїйклмнопрстуфхцчшщьюя")

def normalize(text: str) -> str:
    t = []
    for ch in text.lower():
        if ch in UA_ALPHABET:
            t.append(ch)
    return "".join(t)

IDX_ALPHABET = {ch: i for i, ch in enumerate(UA_ALPHABET)}
BIGRAM_ALPHABET = [f'{a}{b}' for a, b in itertools.product(UA_ALPHABET, UA_ALPHABET)]
IDX_BIGRAM = {bi: i for i, bi in enumerate(BIGRAM_ALPHABET)}

# %%

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

corpus_path = Path("/Users/ksokso/Downloads/fiction.txt")
big_ua_text = corpus_path.read_text(encoding="utf-8")

corpus_clean = normalize(big_ua_text)
print("Corpus length:", len(corpus_clean))
corpus_clean[:200]


# %%
def corpus_slice_generator(corpus_text: str, length: int) -> str:
    """Return a random contiguous slice of `length` characters from the corpus."""
    if length <= 0:
        raise ValueError('length must be positive')
    if length > len(corpus_text):
        raise ValueError('length is greater than corpus size')
    start_idx = random.randint(0, len(corpus_text) - length)
    end_idx = start_idx + length
    return corpus_text[start_idx:end_idx]


# %%
import nltk
from nltk.util import ngrams
from collections import Counter


unigrams = Counter(ngrams(corpus_clean, 1))
bigrams = Counter(ngrams(corpus_clean, 2))

print("Top unigrams:", unigrams.most_common(5))
print("Top bigrams:", bigrams.most_common(5))


# %%
def affine_cipher(text: str, alphabet: list[str], k: tuple[int, int] = None):
    """y = (a*x + b) mod m
    """
    m = len(alphabet)
    if k is None:
        while True:
            a = random.randint(2, m - 1)
            if math.gcd(a, m) == 1:
                break
        b = random.randint(0, m - 1)
    else:
        a, b = k 

    out = []
    for ch in text:
        x = IDX_ALPHABET[ch]
        y = (a * x + b) % m
        out.append(alphabet[y])
    return "".join(out), (a, b)


# %%
import collections
from typing import Dict, Set

def build_sets(freq_ref: Dict[str, float], h: int, mode: str = "rare") -> Set[str]:
    """Select a subset of n-grams based on reference frequencies."""
    if not freq_ref or h <= 0:
        return set()

    sorted_items = sorted(freq_ref.items(), key=lambda kv: kv[1])

    if mode == "rare":
        selected = sorted_items[:h]
    elif mode == "common":
        selected = sorted_items[::-1][:h]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {ngram for ngram, _ in selected}



# %%
def vigenere_cipher(text: str, key: str, alphabet: list[str]) -> str:
    m = len(alphabet)
    idx = {ch: i for i, ch in enumerate(alphabet)}

    key_shifts = [idx[ch] for ch in key if ch in idx]

    out = []
    for i, ch in enumerate(text):
        if ch in idx:
            k = key_shifts[i % len(key_shifts)]
            y = (idx[ch] + k) % m
            out.append(alphabet[y])
        else:
            out.append(ch)
    return "".join(out)


# %%
vigenere_cipher("абвгд", "а", "абвгд")

# %%
vigenere_cipher("абвгд", "аааб", "абвгд")

# %%
def affine_bigram_cipher(text: str, alphabet: list[str], k: tuple[int, int] = None):

    m = len(alphabet)
    idx = {ch: i for i, ch in enumerate(alphabet)}
    if len(text) % 2:
        text += alphabet[0]

    if k is None:
        while True:
            a = random.randint(2, m*m - 1)
            if math.gcd(a, m*m) == 1:
                break
        b = random.randint(0, m*m - 1)
    else:
        a, b = k

    out = []
    for i in range(0, len(text), 2):
        x1 = idx[text[i]]
        x2 = idx[text[i+1]]
        X = x1 * m + x2
        Y = (a * X + b) % (m * m)
        y1, y2 = divmod(Y, m)
        out.append(alphabet[y1])
        out.append(alphabet[y2])
    return "".join(out), (a, b)


# %%


# %%
import math
from typing import Dict, Set

def _prepare_forbidden_data(text: str, l: int, h: int, freq_ref: Dict[str, float]):
    
    A_prh = build_sets(freq_ref, h, mode="rare")
    
    cnt = Counter(ngrams(text, l))
    N = len(text) - l + 1
    
    freq_obs = {
        x: cnt.get(x, 0) / N
        for x in A_prh
    }
    print(f'{A_prh=}, {freq_obs=}')
    
    return A_prh, freq_obs
    
def criterion_1_0(text: str, l: int, h: int, freq_ref: Dict[str, float]) -> float:
    A_prh, _ = _prepare_forbidden_data(text, l, h, freq_ref)
    for i in range(len(text) - l + 1):
        if text[i:i+l] in A_prh:
            return 1.0 
            
    return 0.0 

def criterion_1_1(text: str, l: int, h: int, freq_ref: Dict[str, float]) -> float:
    A_prh, freq_obs = _prepare_forbidden_data(text, l, h, freq_ref)
    
    unique_forbidden_hits = sum(1 for freq in freq_obs.values() if freq > 0)
    
    return float(unique_forbidden_hits)


def criterion_1_2(text: str, l: int, h: int, freq_ref: Dict[str, float]) -> float:
    A_prh, freq_obs = _prepare_forbidden_data(text, l, h, freq_ref)
    
    return max(freq_obs.values()) if freq_obs else 0.0

def criterion_1_3(text: str, l: int, h: int, freq_ref: Dict[str, float]) -> float:
    A_prh, freq_obs = _prepare_forbidden_data(text, l, h, freq_ref)
    
    F_p = sum(freq_obs.values())
    
    return F_p

# %%


# %%
vigenere_key = lambda k: ''.join(random.choices(UA_ALPHABET, k=k))

DISTORTION_METHODS = [
    ("Vigenere_K1", lambda t: vigenere_cipher(t, vigenere_key(1), UA_ALPHABET)),
    ("Vigenere_K5", lambda t: vigenere_cipher(t, vigenere_key(5), UA_ALPHABET)),
    ("Vigenere_K10", lambda t: vigenere_cipher(t, vigenere_key(10), UA_ALPHABET)),
    
    ("Affine_Mono", lambda t: affine_cipher(t, UA_ALPHABET)),
    
    ("Affine_Bigram", lambda t: affine_bigram_cipher(t, UA_ALPHABET)),
]

def generate_dataset(
    corpus_text: str, 
    alphabet: List[str], 
    L_values: List[int], 
    N_counts: Dict[int, int]
) -> Dict[str, List[Tuple[str, str]]]:

    datasets = collections.defaultdict(list)
    
    for L in L_values:
        N = N_counts.get(L, 0)
        if N == 0:
            continue
            
        print(f"Generating L={L}, N={N}")
        texts_X = [corpus_slice_generator(corpus_text, L) for _ in range(N)]
        
        for name, cipher_func in DISTORTION_METHODS:
            print(f"  -> Scrambling: {name}")
            for text_X in texts_X:
                text_Y = cipher_func(text_X)
                datasets[name].append((text_X, text_Y))
                    
    return dict(datasets)

L_values = [10, 100, 1000, 10000]
N_counts = {10: 10000, 100: 10000, 1000: 10000, 10000: 1000}

all_datasets = generate_dataset(corpus_clean, UA_ALPHABET, L_values, N_counts)

# %%
def ring_abc(s0, s1, alphabet, n=10):
    idx = {ch: i for i, ch in enumerate(alphabet)}
    S = [s0, s1]
    for i in range(2, n):
        yi = (idx[S[i-1]] + idx[S[i-2]]) % len(alphabet)
        S.append(alphabet[yi])     
    return ''.join(S)

ring_abc('кк', 'уу', BIGRAM_ALPHABET)

# %%
t1 = all_datasets['Vigenere_K1'][15000]

# %%
criterion_1_0(t1[0], l=1, h=4, freq_ref=uni_freq)

# %%
t1

# %% [markdown]
# # 


