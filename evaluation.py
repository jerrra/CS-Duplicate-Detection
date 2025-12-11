import random
import numpy as np
from typing import List, Dict, Iterable, Tuple
from lsh import lsh_band_candidate_pairs
from model_words import get_brand_from_product, extract_diagonal_inches

def filter_candidates_with_stats(candidate_pairs, products_data):
    """
    Filters candidates and gathers diagnostics on WHY they were removed.
    Returns: (valid_pairs_set, stats_dict)
    """
    # Import inside function to avoid circular import (clustering imports eval_clusters from here)
    from clustering import _brands_compatible, _inches_compatible
    
    valid_pairs = set()
    stats = {
        "total_raw": len(candidate_pairs),
        "same_shop": 0,
        "brand_mismatch": 0,
        "inch_mismatch": 0,
        "kept": 0
    }
    
    for pair in candidate_pairs:
        p_list = list(pair)
        if len(p_list) != 2: 
            continue
            
        pid1, pid2 = p_list[0], p_list[1]
        p1 = products_data[pid1]
        p2 = products_data[pid2]

        if p1.get('shop') == p2.get('shop'):
            stats["same_shop"] += 1
            continue

        if not _brands_compatible(p1, p2):
            stats["brand_mismatch"] += 1
            continue

        if not _inches_compatible(p1, p2):
            stats["inch_mismatch"] += 1
            continue
            
        valid_pairs.add(pair)
        
    stats["kept"] = len(valid_pairs)
    return valid_pairs, stats

def filter_candidates(candidate_pairs, products_data):
    valid, _ = filter_candidates_with_stats(candidate_pairs, products_data)
    return valid

def eval_clusters(clusters, true_pairs):
    predicted_pairs = set()
    for c in clusters:
        if len(c) >= 2:
            c_list = sorted(c)
            for i in range(len(c_list)):
                for j in range(i + 1, len(c_list)):
                    predicted_pairs.add(frozenset([c_list[i], c_list[j]]))
    
    TP = len(predicted_pairs & true_pairs)
    FP = len(predicted_pairs - true_pairs)
    FN = len(true_pairs - predicted_pairs)
    
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return TP, FP, FN, recall, precision, f1_val

def choose_br_from_threshold(n: int, t: float, allow_partial_last_band: bool = False) -> Tuple[int, int, float]:
    best = None
    for r in range(1, n + 1):
        b = n // r 
        if b == 0: continue
        est = (1 / b) ** (1 / r)
        diff = abs(est - t)
        if best is None or diff < best[0]:
            best = (diff, b, r, est)
    _, b, r, est = best
    return b, r, est

def lsh_for_subset(subset_order: List[str], signatures: Dict[str, List[int]], b: int, r: int):
    # dynamic import to avoid circular issues if any
    from lsh import lsh_band_candidate_pairs
    _, candidate_pairs = lsh_band_candidate_pairs(
                signatures,
                subset_order,
                b=b,
                r=r
    )
    return candidate_pairs

def build_true_pairs(uids_by_model: Dict[str, Iterable[str]], allowed_ids: Iterable[str]):
    tp = set()
    allowed = set(allowed_ids)
    for _, uids in uids_by_model.items():
        filtered = [u for u in uids if u in allowed]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                tp.add(frozenset([filtered[i], filtered[j]]))
    return tp

def eval_pairs(candidate_pairs, true_pairs):
    dup_found = len(candidate_pairs & true_pairs)
    total_dup = len(true_pairs)
    comps = len(candidate_pairs)
    pc = dup_found / total_dup if total_dup else 0.0
    pq = dup_found / comps if comps else 0.0
    f1 = 2 * pc * pq / (pc + pq) if (pc + pq) else 0.0
    return pc, pq, f1, comps

def filter_pairs_to_ids(pairs, allowed_ids):
    allowed = set(allowed_ids)
    filtered = set()
    for p in pairs:
        items = list(p)
        if len(items) == 2:
            if items[0] in allowed and items[1] in allowed:
                filtered.add(p)
    return filtered

def run_t_sweep(signatures, order, true_groups, products_data, n_hashes, seed, 
                t_grid=None, bootstrap_runs=1, bootstrap_frac=1.0, 
                msm_config=None, 
                feature_sets=None):
    from clustering import build_clusters_msm
    
    if t_grid is None:
        t_grid = [round(x, 2) for x in np.arange(0.05, 1.01, 0.05)]
    
    rng = random.Random(seed)
    
    all_ids = order
    results = []

    # Unpack MSM features
    title_mw_sets, kv_mw_sets = feature_sets if feature_sets else (None, None)

    for t in t_grid:
        b, r_rows, est = choose_br_from_threshold(n_hashes, t, allow_partial_last_band=True)
        
        pcs, pqs, f1_stars, final_f1s, fracs = [], [], [], [], []
        
        # Absolute Count Stats
        lsh_count_raw = []
        inch_pruned_counts = []
        msm_pair_counts = []
        
        inch_pruned_fracs = []
        msm_frac_cand = [] 
        msm_frac_total = [] 
        cluster_sizes_snapshot = []
        
        for _ in range(bootstrap_runs):
            k = max(2, int(len(all_ids) * bootstrap_frac))
            subset = rng.sample(all_ids, k)
            
            raw_cand = lsh_for_subset(subset, signatures, b=b, r=r_rows)
            cand, stats = filter_candidates_with_stats(raw_cand, products_data)
            
            # --- Stats Collection ---
            total_raw = stats["total_raw"]
            n_kept = stats["kept"]
            n_inch_removed = stats["inch_mismatch"]
            
            # Store Absolute Counts
            lsh_count_raw.append(total_raw)
            inch_pruned_counts.append(n_inch_removed)
            msm_pair_counts.append(n_kept)
            
            # Ratios
            inch_frac = n_inch_removed / total_raw if total_raw > 0 else 0.0
            inch_pruned_fracs.append(inch_frac)
            
            total_possible_pairs = k * (k - 1) // 2
            
            msm_frac_cand.append(n_kept / total_raw if total_raw > 0 else 0.0)
            msm_frac_total.append(n_kept / total_possible_pairs if total_possible_pairs > 0 else 0.0)
            # ------------------------

            true_pairs_subset = build_true_pairs(true_groups, subset)
            pc, pq, f1_star, _ = eval_pairs(cand, true_pairs_subset)
            
            f1_msm_val = f1_star 
            if msm_config and title_mw_sets:
                clusters = build_clusters_msm(
                    cand, 
                    products_data, 
                    title_mw_sets, 
                    kv_mw_sets,
                    sim_threshold=msm_config['threshold'],
                    weights=msm_config['weights'],
                    same_shop_penalty=True,
                    brand_must_match=True
                )
                _, _, _, _, _, f1_msm_val = eval_clusters(clusters, true_pairs_subset)
                sizes = sorted([len(c) for c in clusters], reverse=True)
                cluster_sizes_snapshot = sizes[:20] 

            frac = len(raw_cand) / total_possible_pairs if total_possible_pairs else 0.0
            
            pcs.append(pc)
            pqs.append(pq)
            f1_stars.append(f1_star)
            final_f1s.append(f1_msm_val)
            fracs.append(frac)
            
        results.append({
            "t": t, 
            "b": b, 
            "r": r_rows, 
            "est_t": est,
            "pc": np.mean(pcs), 
            "pq": np.mean(pqs),
            "f1_star": np.mean(f1_stars), 
            "f1": np.mean(final_f1s),     
            "frac": np.mean(fracs),
            "frac_inch_pruned": np.mean(inch_pruned_fracs),
            
            # Average Counts
            "n_lsh_cand": np.mean(lsh_count_raw), # New
            "n_inch_pruned": np.mean(inch_pruned_counts), # New
            "n_msm": np.mean(msm_pair_counts),
            
            "frac_msm_of_cand": np.mean(msm_frac_cand),
            "frac_msm_of_total": np.mean(msm_frac_total),
            "top_cluster_sizes": cluster_sizes_snapshot 
        })
    return results