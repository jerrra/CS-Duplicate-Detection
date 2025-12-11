import math
import hashlib
from itertools import combinations
from collections import defaultdict


def hash_band(band_tuple, salt):
    s = salt + ":" + ",".join(map(str, band_tuple))
    return hashlib.sha1(s.encode()).hexdigest()

def lsh_band_candidate_pairs(signatures, order, b, r):
    """Banding + bucket grouping. Ignores the last partial band to ensure uniform probability."""
    # We do NOT use n to force the loop. We stick strictly to b bands of size r.
    # Check if we have enough rows
    first_sig_len = len(signatures[order[0]])
    if b * r > first_sig_len:
        raise ValueError(f"Cannot form {b} bands of {r} rows from signature length {first_sig_len}")

    buckets = [defaultdict(list) for _ in range(b)]

    for pid in order:
        sig = signatures[pid]
        
        for bi in range(b):
            start = bi * r
            end = start + r
            
            # This band is strictly length r. 
            # If we run out of signature (shouldn't happen with the check above), we stop.
            band = tuple(sig[start:end])
            
            # Salt is crucial so bands don't collide across positions
            key = hash_band(band, f"band_{bi}")
            buckets[bi][key].append(pid)

    candidate_pairs = set()
    for band in buckets:
        for members in band.values():
            if len(members) > 1:
                for a, bpid in combinations(members, 2):
                    candidate_pairs.add(frozenset([a, bpid]))

    return buckets, candidate_pairs