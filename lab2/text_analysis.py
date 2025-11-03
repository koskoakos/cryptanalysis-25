# %%
import math, random, json, itertools, collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import itertools
import nltk
from nltk.util import ngrams
from collections import Counter

random.seed(42)
np.random.seed(42)

RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True, parents=True)
(TABLES := RESULTS_DIR / "tables").mkdir(exist_ok=True, parents=True)
(FIGS := RESULTS_DIR / "figs").mkdir(exist_ok=True, parents=True)

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

def corpus_slice_generator(corpus_text: str, length: int) -> str:
    """Return a random contiguous slice of `length` characters from the corpus."""
    if length <= 0:
        raise ValueError('length must be positive')
    if length > len(corpus_text):
        raise ValueError('length is greater than corpus size')
    start_idx = random.randint(0, len(corpus_text) - length)
    end_idx = start_idx + length
    return corpus_text[start_idx:end_idx]


unigrams = Counter(ngrams(corpus_clean, 1))
bigrams = Counter(ngrams(corpus_clean, 2))

print("Top unigrams:", unigrams.most_common(5))
print("Top bigrams:", bigrams.most_common(5))

uni_freq = {''.join(ng): cnt/len(corpus_clean) for ng, cnt in unigrams.items()}
bi_freq = {''.join(ng): cnt/(len(corpus_clean)-1) for ng, cnt in bigrams.items()}

# %%
def affine_cipher(text: str, alphabet: list[str], l: int = 1, k: tuple[int, int] = None):
    m = len(alphabet)
    modulus = m if l == 1 else m * m

    if k is None:
        while True:
            a = random.randint(2, modulus - 1)
            if math.gcd(a, modulus) == 1:
                break
        b = random.randint(0, modulus - 1)
    else:
        a, b = k

    idx = {ch: i for i, ch in enumerate(alphabet)}
    working_text = text
    if l == 2 and len(working_text) % 2:
        working_text += alphabet[0]

    out = []
    if l == 1:
        for ch in working_text:
            x = idx[ch]
            y = (a * x + b) % modulus
            out.append(alphabet[y])
    else:
        for i in range(0, len(working_text), 2):
            x1 = idx[working_text[i]]
            x2 = idx[working_text[i + 1]]
            X = x1 * m + x2
            Y = (a * X + b) % modulus
            y1, y2 = divmod(Y, m)
            out.append(alphabet[y1])
            out.append(alphabet[y2])

    return "".join(out)

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

def ring_abc(alphabet, n=10):
    s0, s1 = random.choices(alphabet, k=2)
    idx = {ch: i for i, ch in enumerate(alphabet)}
    S = [s0, s1]
    for i in range(2, n):
        yi = (idx[S[i-1]] + idx[S[i-2]]) % len(alphabet)
        S.append(alphabet[yi])     
    return ''.join(S)

ring_abc(BIGRAM_ALPHABET)


# %%


def build_sets(freq_ref: Dict[str, float], h: int, mode: str = "rare") -> list[str]:
    """Return a list of n-grams selected by frequency."""
    if h <= 0 or not freq_ref:
        return []

    mode_key = mode.lower()
    if mode_key == "rare":
        sorted_items = sorted(freq_ref.items(), key=lambda kv: kv[1])
    elif mode_key in {"top", "common"}:
        sorted_items = sorted(freq_ref.items(), key=lambda kv: kv[1], reverse=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return [ngram for ngram, _ in sorted_items[:h]]

_FORBIDDEN_CACHE: Dict[tuple[int, int, str], tuple[str, ...]] = {}

def get_forbidden_set(freq_ref: Dict[str, float], h: int, mode: str = "rare") -> tuple[str, ...]:
    mode_key = mode.lower()
    key = (id(freq_ref), h, mode_key)
    if key not in _FORBIDDEN_CACHE:
        _FORBIDDEN_CACHE[key] = tuple(build_sets(freq_ref, h, mode=mode))
    return _FORBIDDEN_CACHE[key]

def compute_forbidden_frequencies(text: str, l: int, forbidden: tuple[str, ...]) -> Dict[str, float]:
    if not forbidden:
        return {}
    if len(text) < l:
        return {ng: 0.0 for ng in forbidden}

    cnt = Counter(ngrams(text, l))
    N = len(text) - l + 1
    return {
        ng: cnt.get(tuple(ng), 0) / N
        for ng in forbidden
    }

def criterion_1_0(text: str, l: int, h: int, freq_ref: Dict[str, float], *, forbidden: tuple[str, ...] | None = None) -> float:
    if forbidden is None:
        forbidden = get_forbidden_set(freq_ref, h)
    if not forbidden:
        return 0.0

    end = len(text) - l + 1
    if end <= 0:
        return 0.0

    for i in range(end):
        if text[i:i + l] in forbidden:
            return 1.0
    return 0.0

def criterion_1_1(text: str, l: int, h: int, freq_ref: Dict[str, float], *, forbidden: tuple[str, ...] | None = None, freq_obs: Dict[str, float] | None = None) -> float:
    if forbidden is None:
        forbidden = get_forbidden_set(freq_ref, h)
    if freq_obs is None:
        freq_obs = compute_forbidden_frequencies(text, l, forbidden)
    return float(sum(1 for freq in freq_obs.values() if freq > 0))

def criterion_1_2(text: str, l: int, h: int, freq_ref: Dict[str, float], *, forbidden: tuple[str, ...] | None = None, freq_obs: Dict[str, float] | None = None) -> float:
    if forbidden is None:
        forbidden = get_forbidden_set(freq_ref, h)
    if freq_obs is None:
        freq_obs = compute_forbidden_frequencies(text, l, forbidden)
    return float(max(freq_obs.values(), default=0.0))

def criterion_1_3(text: str, l: int, h: int, freq_ref: Dict[str, float], *, forbidden: tuple[str, ...] | None = None, freq_obs: Dict[str, float] | None = None) -> float:
    if forbidden is None:
        forbidden = get_forbidden_set(freq_ref, h)
    if freq_obs is None:
        freq_obs = compute_forbidden_frequencies(text, l, forbidden)
    return float(sum(freq_obs.values()))

def entropy_per_symbol(freq: Dict[str, float], l: int) -> float:
    total = sum(freq.values())
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy / l

def entropy_for_text(text: str, l: int) -> float:
    counts = Counter(ngrams(text, l))
    freqs = {''.join(ng): cnt / (len(text) - l + 1) for ng, cnt in counts.items()}
    return entropy_per_symbol(counts, l)

def criterion_3_0(text: str, l: int, freq_ref: Dict[str, float], k_h: float) -> dict:
    ref_entropy = entropy_per_symbol(freq_ref, l)
    obs_entropy = entropy_for_text(text, l)
    diff = abs(ref_entropy - obs_entropy)
    return diff > k_h

def index_of_coincidence(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total < 2:
        return 0.0
    numerator = sum(c * (c - 1) for c in counts.values())
    return numerator / (total * (total - 1))

def index_of_coincidence_for_text(text: str, l: int) -> float:
    counts = Counter(ngrams(text, l))
    return index_of_coincidence(counts)

def top_ngrams(freq_ref: Dict[str, float], j: int) -> list[str]:
    """Return the j most frequent l-grams from the reference distribution."""
    if j <= 0:
        return []
    sorted_items = sorted(freq_ref.items(), key=lambda kv: kv[1], reverse=True)
    return [ngram for ngram, _ in sorted_items[:j]]

def criterion_5_1(text: str, l: int, freq_ref: Dict[str, float], j: int, k_empt: int) -> dict:
    top = top_ngrams(freq_ref, j)
    counts = Counter(ngrams(text, l))
    if not top:
        return False
    if not counts:
        empty_boxes = len(top)
    else:
        empty_boxes = sum(1 for ng in top if counts.get(tuple(ng), 0) == 0)
    return empty_boxes >= k_empt



# %%
random_string = lambda k: ''.join(random.choices(UA_ALPHABET, k=k))

DISTORTION_METHODS = {"Vigenere K1": lambda t: vigenere_cipher(t, random_string(1), UA_ALPHABET),
    "Vigenere K5": lambda t: vigenere_cipher(t, random_string(5), UA_ALPHABET),
    "Vigenere K10": lambda t: vigenere_cipher(t, random_string(10), UA_ALPHABET),
    "Affine uni": lambda t: affine_cipher(t, UA_ALPHABET),
    "Affine bigram": lambda t: affine_cipher(t, UA_ALPHABET, l=2),
}

RANDOM_METHODS = {
    "Random uni": lambda l: random_string(l),
    "Random bigram": lambda l: ''.join(
        random.choices(BIGRAM_ALPHABET, k=(l + 1) // 2)
    )[:l],
    "Ring uni": lambda l: ring_abc(UA_ALPHABET, l),
    "Ring bigram": lambda l: ring_abc(BIGRAM_ALPHABET, l)
}

def generate_random(alphabet: List[str], 
                    L_values: List[int],
                    N_values: Dict[int, int]) -> Dict[str, List[str]]:
    datasets = collections.defaultdict(list)
    for L in L_values:
        N = N_values.get(L, 0)
        if N == 0:
            continue
        
        datasets[L] = collections.defaultdict(list)
            
        print(f"Generating random L={L}, N={N}")
        for name, random_func in RANDOM_METHODS.items():
            print(f"  -> Generating: {name}")
            for _ in range(N):
                text_Y = random_func(L)
                datasets[L][name].append(text_Y)
    return dict(datasets)


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
        
        datasets[L] = collections.defaultdict(list)
            
        print(f"Generating L={L}, N={N}")
        texts_X = [corpus_slice_generator(corpus_text, L) for _ in range(N)]
        
        for name, cipher_func in DISTORTION_METHODS.items():
            print(f"  -> Scrambling: {name}")
            for text_X in texts_X:
                text_Y = cipher_func(text_X)
                datasets[L][name].append((text_X, text_Y))
                    
    return dict(datasets)

L_values = [10, 100, 1000, 10000]
N_counts = {10: 10000, 100: 10000, 1000: 10000, 10000: 1000}

distorted_texts = generate_dataset(corpus_clean, UA_ALPHABET, L_values, N_counts)
random_texts = generate_random(UA_ALPHABET, L_values, N_counts)



# %%



# %%
t1 = distorted_texts[100]['Vigenere K1'][0]


# %%
criterion_1_0(t1[0], l=1, h=2, freq_ref=uni_freq)

# %%
texts_10 = distorted_texts[100]
a = 0
b = 0

l = 1
h = 3

for (x, y) in texts_10['Affine uni'][:10]:
    a += criterion_5_1(x, l=2, freq_ref=uni_freq, j=10, k_empt=2)
    b += (1 - criterion_5_1(y[0], l=2, freq_ref=uni_freq, j=10, k_empt=2))

print(a, b)



# %% [markdown]
# # 

# %%
hs = {10: 3,
     100: 1,
     1000: 1,
     10000: 1}

ORDER = ['1.0', '1.1', '1.2', '1.3', '3.0', '5.1']

for L in sorted(distorted_texts.keys()):
    if L not in hs:
        continue
    h = hs[L]
    for method, pairs in distorted_texts[L].items():
        print(f"L={L} | Method={method}")
        results = {name: {1: (0, 0), 2: (0, 0)} for name in ORDER}
        for l in (1, 2):
            freq_ref = uni_freq if l == 1 else bi_freq
            forbidden = get_forbidden_set(freq_ref, h)
            cache = []
            for x, y in pairs:
                freq_x = compute_forbidden_frequencies(x, l, forbidden)
                freq_y = compute_forbidden_frequencies(y, l, forbidden)
                cache.append((x, y, freq_x, freq_y))

            fp_10 = sum(int(criterion_1_0(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden)) for x, _, _, _ in cache)
            fn_10 = sum(1 - int(criterion_1_0(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden)) for _, y, _, _ in cache)
            results['1.0'][l] = (fp_10, fn_10)

            fp_11 = sum(int(criterion_1_1(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_x)) for x, _, freq_x, _ in cache)
            fn_11 = sum(1 - int(criterion_1_1(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_y)) for _, y, _, freq_y in cache)
            results['1.1'][l] = (fp_11, fn_11)

            fp_12 = sum(int(criterion_1_2(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_x)) for x, _, freq_x, _ in cache)
            fn_12 = sum(1 - int(criterion_1_2(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_y)) for _, y, _, freq_y in cache)
            results['1.2'][l] = (fp_12, fn_12)

            fp_13 = sum(int(criterion_1_3(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_x)) for x, _, freq_x, _ in cache)
            fn_13 = sum(1 - int(criterion_1_3(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_y)) for _, y, _, freq_y in cache)
            results['1.3'][l] = (fp_13, fn_13)

            fp_30 = sum(int(criterion_3_0(x, l=l, freq_ref=freq_ref, k_h=0.1)) for x, _, _, _ in cache)
            fn_30 = sum(1 - int(criterion_3_0(y, l=l, freq_ref=freq_ref, k_h=0.1)) for _, y, _, _ in cache)
            results['3.0'][l] = (fp_30, fn_30)

            fp_51 = sum(int(criterion_5_1(x, l=l, freq_ref=freq_ref, j=50, k_empt=5)) for x, _, _, _ in cache)
            fn_51 = sum(1 - int(criterion_5_1(y, l=l, freq_ref=freq_ref, j=50, k_empt=5)) for _, y, _, _ in cache)
            results['5.1'][l] = (fp_51, fn_51)

        for name in ORDER:
            fp1, fn1 = results[name][1]
            fp2, fn2 = results[name][2]
            print(f"  Criterion {name}: FP(l=1)={fp1}, FN(l=1)={fn1}, FP(l=2)={fp2}, FN(l=2)={fn2}")
        print()



# %%
import bz2
from statistics import mean

def bwt_compressed_size(text: str) -> int:
    data = text.encode('utf-8')
    return len(bz2.compress(data))

def compression_stats(texts, sample_size=None):
    if sample_size is not None and sample_size > 0:
        texts = texts[:sample_size]
    sizes = []
    ratios = []
    for text in texts:
        original_size = len(text.encode('utf-8'))
        if original_size == 0:
            continue
        compressed_size = bwt_compressed_size(text)
        sizes.append(compressed_size)
        ratios.append(compressed_size / original_size)
    count = len(sizes)
    avg_size = mean(sizes) if sizes else 0.0
    avg_ratio = mean(ratios) if ratios else 0.0
    return count, avg_size, avg_ratio



# %%
COMPRESSION_SAMPLE_SIZE = 200

for L in sorted(distorted_texts.keys()):
    print(f"Compression analysis | L={L}")
    for method, pairs in distorted_texts[L].items():
        sample_pairs = pairs[:COMPRESSION_SAMPLE_SIZE] if COMPRESSION_SAMPLE_SIZE else pairs
        originals = [x for x, _ in sample_pairs]
        distorted_versions = [y for _, y in sample_pairs]
        n_orig, avg_size_orig, avg_ratio_orig = compression_stats(originals, sample_size=None)
        n_dist, avg_size_dist, avg_ratio_dist = compression_stats(distorted_versions, sample_size=None)
        print(f"  Method={method} | samples={n_orig}")
        print(f"    Original  : avg_compressed_bytes={avg_size_orig:.2f}, avg_ratio={avg_ratio_orig:.3f}")
        print(f"    Distorted : avg_compressed_bytes={avg_size_dist:.2f}, avg_ratio={avg_ratio_dist:.3f}")
    print()

for L in sorted(random_texts.keys()):
    print(f"Compression analysis on random texts | L={L}")
    for method, texts in random_texts[L].items():
        sample_texts = texts[:COMPRESSION_SAMPLE_SIZE] if COMPRESSION_SAMPLE_SIZE else texts
        n_rand, avg_size_rand, avg_ratio_rand = compression_stats(sample_texts, sample_size=None)
        print(f"  Method={method} | samples={n_rand}")
        print(f"    Random : avg_compressed_bytes={avg_size_rand:.2f}, avg_ratio={avg_ratio_rand:.3f}")
    print()



# %%
ORDER = ['1.0', '1.1', '1.2', '1.3', '3.0', '5.1']

for L in sorted(random_texts.keys()):
    if L not in hs:
        continue
    h = hs[L]
    for method, texts in random_texts[L].items():
        print(f"Random texts | L={L} | Method={method}")
        results = {name: {1: 0, 2: 0} for name in ORDER}
        for l in (1, 2):
            freq_ref = uni_freq if l == 1 else bi_freq
            forbidden = get_forbidden_set(freq_ref, h)
            freq_cache = [compute_forbidden_frequencies(text, l, forbidden) for text in texts]

            results['1.0'][l] = sum(int(not bool(criterion_1_0(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden))) for text in texts)
            results['1.1'][l] = sum(int(not bool(criterion_1_1(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_obs))) for text, freq_obs in zip(texts, freq_cache))
            results['1.2'][l] = sum(int(not bool(criterion_1_2(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_obs))) for text, freq_obs in zip(texts, freq_cache))
            results['1.3'][l] = sum(int(not bool(criterion_1_3(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_obs))) for text, freq_obs in zip(texts, freq_cache))
            results['3.0'][l] = sum(int(not bool(criterion_3_0(text, l=l, freq_ref=freq_ref, k_h=0.1))) for text in texts)
            results['5.1'][l] = sum(int(not bool(criterion_5_1(text, l=l, freq_ref=freq_ref, j=50, k_empt=5))) for text in texts)

        for name in ORDER:
            fn1 = results[name][1]
            fn2 = results[name][2]
            print(f"  Criterion {name}: FN(l=1)={fn1}, FN(l=2)={fn2}")
        print()



# %%


# %%
hs = {10: 3,
     100: 1,
     1000: 1,
     10000: 1}

ORDER = ['1.0', '1.1', '1.2', '1.3', '3.0', '5.1']

for L in sorted(distorted_texts.keys()):
    if L not in hs:
        continue
    h = hs[L]
    for method, pairs in distorted_texts[L].items():
        print(f"L={L} | Method={method}")
        summaries = {name: {1: (0, 0), 2: (0, 0)} for name in ORDER}
        for l in (1, 2):
            freq_ref = uni_freq if l == 1 else bi_freq
            forbidden = get_forbidden_set(freq_ref, h)
            data = []
            for x, y in pairs:
                freq_x = compute_forbidden_frequencies(x, l, forbidden)
                freq_y = compute_forbidden_frequencies(y, l, forbidden)
                data.append((x, y, freq_x, freq_y))

            fp_10 = sum(int(criterion_1_0(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden)) for x, _, _, _ in data)
            fn_10 = sum(1 - int(criterion_1_0(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden)) for _, y, _, _ in data)
            summaries['1.0'][l] = (fp_10, fn_10)

            fp_11 = sum(int(criterion_1_1(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_x)) for x, _, freq_x, _ in data)
            fn_11 = sum(1 - int(criterion_1_1(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_y)) for _, y, _, freq_y in data)
            summaries['1.1'][l] = (fp_11, fn_11)

            fp_12 = sum(int(criterion_1_2(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_x)) for x, _, freq_x, _ in data)
            fn_12 = sum(1 - int(criterion_1_2(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_y)) for _, y, _, freq_y in data)
            summaries['1.2'][l] = (fp_12, fn_12)

            fp_13 = sum(int(criterion_1_3(x, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_x)) for x, _, freq_x, _ in data)
            fn_13 = sum(1 - int(criterion_1_3(y, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_y)) for _, y, _, freq_y in data)
            summaries['1.3'][l] = (fp_13, fn_13)

            fp_30 = sum(int(criterion_3_0(x, l=l, freq_ref=freq_ref, k_h=0.1)) for x, _, _, _ in data)
            fn_30 = sum(1 - int(criterion_3_0(y, l=l, freq_ref=freq_ref, k_h=0.1)) for _, y, _, _ in data)
            summaries['3.0'][l] = (fp_30, fn_30)

            fp_51 = sum(int(criterion_5_1(x, l=l, freq_ref=freq_ref, j=50, k_empt=5)) for x, _, _, _ in data)
            fn_51 = sum(1 - int(criterion_5_1(y, l=l, freq_ref=freq_ref, j=50, k_empt=5)) for _, y, _, _ in data)
            summaries['5.1'][l] = (fp_51, fn_51)

        for name in ORDER:
            fp1, fn1 = summaries[name][1]
            fp2, fn2 = summaries[name][2]
            print(f"  Criterion {name}: FP(l=1)={fp1}, FN(l=1)={fn1}, FP(l=2)={fp2}, FN(l=2)={fn2}")
        print()



# %%
for L in sorted(random_texts.keys()):
    if L not in hs:
        continue
    h = hs[L]
    for method, texts in random_texts[L].items():
        print(f"Random texts | L={L} | Method={method}")
        results = {name: {1: 0, 2: 0} for name in ORDER}
        for l in (1, 2):
            freq_ref = uni_freq if l == 1 else bi_freq
            forbidden = get_forbidden_set(freq_ref, h)
            freq_cache = [compute_forbidden_frequencies(text, l, forbidden) for text in texts]

            errors_10 = sum(int(not bool(criterion_1_0(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden))) for text in texts)
            errors_11 = sum(int(not bool(criterion_1_1(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_obs))) for text, freq_obs in zip(texts, freq_cache))
            errors_12 = sum(int(not bool(criterion_1_2(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_obs))) for text, freq_obs in zip(texts, freq_cache))
            errors_13 = sum(int(not bool(criterion_1_3(text, l=l, h=h, freq_ref=freq_ref, forbidden=forbidden, freq_obs=freq_obs))) for text, freq_obs in zip(texts, freq_cache))
            errors_30 = sum(int(not bool(criterion_3_0(text, l=l, freq_ref=freq_ref, k_h=0.1))) for text in texts)
            errors_51 = sum(int(not bool(criterion_5_1(text, l=l, freq_ref=freq_ref, j=50, k_empt=5))) for text in texts)

            results['1.0'][l] = errors_10
            results['1.1'][l] = errors_11
            results['1.2'][l] = errors_12
            results['1.3'][l] = errors_13
            results['3.0'][l] = errors_30
            results['5.1'][l] = errors_51

        for name in ORDER:
            fn1 = results[name][1]
            fn2 = results[name][2]
            print(f"  Criterion {name}: FN(l=1)={fn1}, FN(l=2)={fn2}")
        print()



# %%


# %%


# %%


# %%


# %%



