import math
import random


def next_prime(n):
    """Return the smallest prime > n."""
    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        r = int(math.sqrt(x))
        for i in range(3, r + 1, 2):
            if x % i == 0:
                return False
        return True

    candidate = n + 1
    while not is_prime(candidate):
        candidate += 1
    return candidate


def generate_hash_funcs(num_hashes, p, seed=None):
    # Use the specific seed passed for reproducibility
    rng = random.Random(seed)
    return [(rng.randrange(1, p), rng.randrange(1, p)) for _ in range(num_hashes)]


def compute_minhash_signatures(binary_vectors, n_hashes, p=None, seed=42):
    """Return signatures and deterministic product order."""
    any_pid = next(iter(binary_vectors))
    r = len(binary_vectors[any_pid])

    if p is None:
        p = next_prime(r)

    # Pass the seed to the generator
    hashes = generate_hash_funcs(n_hashes, p, seed)
    signatures = {}
    order = []

    ones_index = {pid: [i for i, b in enumerate(vec) if b]
                  for pid, vec in binary_vectors.items()}

    for pid, ones in ones_index.items():
        order.append(pid)

        if not ones:
            signatures[pid] = [p] * n_hashes
            continue

        sig = []
        for a, b in hashes:
            sig.append(min((a + b * x) % p for x in ones))

        signatures[pid] = sig

    return signatures, order, n_hashes, p