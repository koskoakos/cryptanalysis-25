import numpy as np
from multiprocessing import shared_memory, Process, Event
import os
from gmpy2 import mpz, powmod
import hashlib
from tqdm import tqdm

C = None
N = None

with open("mitm_2048_5618.txt", "r", encoding="utf-8") as f:
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
l = 56

limit = 2 ** (l // 2) + 1          
TOTAL_ENTRIES = limit
RSA_BYTE_LENGTH = 256              # 2048-bit modulus

NUM_PROCESSES = 8
HASH_LEN = 8                       
DTYPE = np.dtype([
    ('hash',  f'V{HASH_LEN}'),
    ('index', np.uint64),
])
ITEM_SIZE = DTYPE.itemsize
TOTAL_BYTES = TOTAL_ENTRIES * ITEM_SIZE

e = mpz(e)
N = mpz(N)
C = mpz(C)

hash_func = hashlib.blake2b

stop_event = Event()

SHM_NAME = 'mitm_T_table'


def worker_task(shm_name, start_index, end_index, num_process):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_table = np.ndarray((TOTAL_ENTRIES,), dtype=DTYPE, buffer=existing_shm.buf)

    powmod_ = powmod
    hash_col = shared_table['hash']
    idx_col = shared_table['index']

    T = mpz(start_index)

    for i in tqdm(range(start_index, end_index),
                  position=num_process):
        X = powmod_(T, e, N)
        T += 1

        x_bytes = X.to_bytes(RSA_BYTE_LENGTH, byteorder='big')
        h = hash_func(x_bytes, digest_size=HASH_LEN)
        digest = h.digest()  

        hash_col[i] = np.frombuffer(digest, dtype=f'V{HASH_LEN}')[0]
        idx_col[i] = i

    existing_shm.close()


def search_task(shm_name, start_index, end_index, num_process, N, e, C, stop):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_table = np.ndarray((TOTAL_ENTRIES,), dtype=DTYPE, buffer=existing_shm.buf)
    neg_e = -e
    hash_keys = shared_table['hash']

    for i in tqdm(range(start_index, end_index),
                  position=num_process):
        if stop.is_set():
            existing_shm.close()
            return
        S_param = mpz(i)
        S_inv_e = powmod(S_param, neg_e, N)
        Y = (C * S_inv_e) % N

        Y_bytes = Y.to_bytes(length=RSA_BYTE_LENGTH, byteorder='big')
        h = hash_func(Y_bytes, digest_size=HASH_LEN)
        Y_hash = h.digest()   # bytes

        Y_key = np.frombuffer(Y_hash, dtype=f'V{HASH_LEN}')[0]

        idx = np.searchsorted(hash_keys, Y_key)

        if idx < len(hash_keys) and hash_keys[idx] == Y_key:
            T_param = mpz(shared_table[idx]['index'])
            m = (T_param * S_param) % N

            print(f"\nMatch! Proc {num_process}: S={S_param}, T={T_param}")
            print(f"Plain text M: {hex(m)}")
            M_e = powmod(m, e, N)
            if (C == M_e):
                print(f"Verified {M_e=} == {C}")
                stop.set()
                existing_shm.close()
                return
            else:
                print(f"Oops. Enc(E) {M_e} != {C=}")

    print(f"Searched {num_process} (PID {os.getpid()}): "
          f"[{start_index}, {end_index - 1}].")
    existing_shm.close()


if __name__ == '__main__':
    chunk_size = limit // NUM_PROCESSES
    processes = []

    try:
        temp_shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        temp_shm.close()
        temp_shm.unlink()
        print(f"Force unlink previous shmem {SHM_NAME}.")
    except FileNotFoundError:
        pass
    except Exception as ex:
        raise

    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=TOTAL_BYTES)
        print(f"{shm.name=}")

        shared_table = np.ndarray((TOTAL_ENTRIES,), dtype=DTYPE, buffer=shm.buf)

        for i in range(NUM_PROCESSES):
            start = i * chunk_size or 1
            end = (i + 1) * chunk_size if i < NUM_PROCESSES - 1 else limit
            p = Process(target=worker_task, args=(shm.name, start, end, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        shared_table_sort = np.ndarray((limit,), dtype=DTYPE, buffer=shm.buf)
        
        shared_table_sort.sort(order='hash')  # in-place sort 'hash'

        processes = []
        for i in range(NUM_PROCESSES):
            start = i * chunk_size or 1
            end = (i + 1) * chunk_size if i < NUM_PROCESSES - 1 else limit

            p = Process(
                target=search_task,
                args=(shm.name, start, end, i, N, e, C, stop_event)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        shm.close()
        shm.unlink()
        print(f"Unlinked shared memory")

    except Exception as ex:
        print(f"{ex=}")
        try:
            shm.unlink()
        except Exception:
            pass
        raise
