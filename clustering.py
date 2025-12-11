from collections import defaultdict
from typing import Dict, Set, List
import re
from evaluation import eval_clusters
from model_words import extract_diagonal_inches, get_brand_from_product

# =============================================================================
# Similarity & Helper Functions
# =============================================================================

def jaccard_sets(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    return len(a & b) / len(a | b)

def qgrams(s: str, q: int = 3) -> Set[str]:
    if not isinstance(s, str): return set()
    s = s.lower()
    if len(s) < q: return {s} if s else set()
    return {s[i: i + q] for i in range(len(s) - q + 1)}

def qgram_similarity(a: str, b: str, q: int = 3) -> float:
    qa, qb = qgrams(a, q), qgrams(b, q)
    if not qa and not qb: return 0.0
    return len(qa & qb) / len(qa | qb)

_BRAND_STOPWORDS = {"electronics", "electronic", "corp", "corporation", "company", "co", "inc", "ltd", "limited", "intl", "international"}

def _normalize_brand_str(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    tokens = [t for t in s.split() if t and t not in _BRAND_STOPWORDS]
    return " ".join(tokens)

# --- STRICT BLOCKING HELPERS (Explicitly Exported) ---

def _brands_compatible(prod_a: dict, prod_b: dict) -> bool:
    """
    Checks if brands are compatible using the ROBUST extraction (Title + KV).
    """
    raw_a = get_brand_from_product(prod_a)
    raw_b = get_brand_from_product(prod_b)
    
    # If either is missing, we cannot rule them out -> Compatible
    if not raw_a or not raw_b: return True
    
    norm_a = _normalize_brand_str(raw_a)
    norm_b = _normalize_brand_str(raw_b)
    
    if not norm_a or not norm_b: return True
    if norm_a == norm_b: return True
    
    # Allow small fuzziness (e.g. Samsung vs Samsng)
    sim = qgram_similarity(norm_a, norm_b, q=2)
    return sim >= 0.8

def _inches_compatible(prod_a: dict, prod_b: dict) -> bool:
    """
    Returns False (Incompatible) if BOTH have inches and they differ by > 0.55.
    Otherwise returns True (Compatible/Conservative).
    """
    i1 = extract_diagonal_inches(prod_a)
    i2 = extract_diagonal_inches(prod_b)
    
    if i1 is not None and i2 is not None:
        if abs(i1 - i2) > 0.55:
            return False
    return True

# --- SIMILARITY CALCULATION ---

def msm_similarity(prod_a, prod_b, title_mw_a, title_mw_b, kv_mw_a, kv_mw_b, weights=(0.4, 0.3, 0.3), q=3):
    w_key, w_hsm, w_title = weights
    total_w = w_key + w_hsm + w_title
    if total_w > 0:
        w_key /= total_w
        w_hsm /= total_w
        w_title /= total_w

    kv_a = prod_a.get("featuresMap", {}) or {}
    kv_b = prod_b.get("featuresMap", {}) or {}

    def norm_key(k):
        return re.sub(r"[^a-z0-9]", "", k.lower())

    map_a = {norm_key(k): v for k, v in kv_a.items()}
    map_b = {norm_key(k): v for k, v in kv_b.items()}

    common_keys = set(map_a.keys()) & set(map_b.keys())
    
    kv_sims = []
    for k in common_keys:
        kv_sims.append(qgram_similarity(str(map_a[k]), str(map_b[k]), q=q))
    
    key_sim = sum(kv_sims) / len(kv_sims) if kv_sims else 0.0
    hsm_sim = jaccard_sets(kv_mw_a, kv_mw_b)
    title_sim = jaccard_sets(title_mw_a, title_mw_b)

    sim = w_key * key_sim + w_hsm * hsm_sim + w_title * title_sim
    return sim

# =============================================================================
# Adapted Single Linkage Clustering (No Size Caps + Inch Blocking)
# =============================================================================

def _can_merge(cluster_a: List[str], cluster_b: List[str], products: Dict, 
               candidate_pair_set: Set[frozenset], check_lsh_compliance: bool) -> bool:
    """
    Checks if two clusters can merge based on the "Distance Infinity" rule.
    """
    for pid1 in cluster_a:
        for pid2 in cluster_b:
            p1_obj = products[pid1]
            p2_obj = products[pid2]

            # 1. Shop Constraint
            if p1_obj.get('shop') == p2_obj.get('shop'):
                return False

            # 2. Brand Constraint
            if not _brands_compatible(p1_obj, p2_obj):
                return False

            # 3. Inch Constraint
            if not _inches_compatible(p1_obj, p2_obj):
                return False
            
            # 4. LSH Constraint (Quasi-clique)
            if check_lsh_compliance:
                if frozenset([pid1, pid2]) not in candidate_pair_set:
                    return False
    return True


def build_clusters_msm(candidate_pairs, products, title_mw_sets, kv_mw_sets, 
                       sim_threshold=0.5, weights=(0.4, 0.3, 0.3), 
                       same_shop_penalty=True, brand_must_match=True):
    
    # 1. Calculate Edges & Similarity
    edges = []
    relevant_products = set()
    valid_candidate_set = set()

    for pid1, pid2 in candidate_pairs:
        prod_a = products[pid1]
        prod_b = products[pid2]

        # Basic Pre-filtering
        if same_shop_penalty and prod_a.get('shop') == prod_b.get('shop'): continue
        if brand_must_match and not _brands_compatible(prod_a, prod_b): continue
        if not _inches_compatible(prod_a, prod_b): continue

        sim = msm_similarity(
            prod_a, prod_b,
            title_mw_sets[pid1], title_mw_sets[pid2],
            kv_mw_sets[pid1], kv_mw_sets[pid2],
            weights=weights,
        )

        if sim >= sim_threshold:
            filtered_pair = frozenset([pid1, pid2])
            edges.append((sim, pid1, pid2))
            valid_candidate_set.add(filtered_pair)
            relevant_products.add(pid1)
            relevant_products.add(pid2)

    edges.sort(key=lambda x: x[0], reverse=True)

    p_to_cid = {pid: pid for pid in relevant_products}
    clusters = {pid: [pid] for pid in relevant_products}

    for _, pid1, pid2 in edges:
        cid1 = p_to_cid[pid1]
        cid2 = p_to_cid[pid2]

        if cid1 == cid2:
            continue

        c1_members = clusters[cid1]
        c2_members = clusters[cid2]

        if _can_merge(c1_members, c2_members, products, valid_candidate_set, check_lsh_compliance=True):
            if len(c1_members) < len(c2_members):
                c1_members, c2_members = c2_members, c1_members
                cid1, cid2 = cid2, cid1
            
            clusters[cid1].extend(clusters[cid2])
            for p in clusters[cid2]:
                p_to_cid[p] = cid1
            del clusters[cid2]

    return list(clusters.values())


def grid_search_joint(candidate_pairs, flat, title_mw_sets, kv_mw_sets, true_pairs_all, 
                      weight_grid, threshold_grid, **kwargs):

    
    min_th = min(threshold_grid)
    valid_pairs_list = []
    
    for pid1, pid2 in candidate_pairs:
        p1, p2 = flat[pid1], flat[pid2]
        if p1.get('shop') == p2.get('shop'): continue
        if not _brands_compatible(p1, p2): continue
        if not _inches_compatible(p1, p2): continue
        valid_pairs_list.append((pid1, pid2))

    best_f1 = -1
    best_config = {}
    best_clusters = []

    for wt in weight_grid:
        weighted_edges = []
        for pid1, pid2 in valid_pairs_list:
            sim = msm_similarity(
                flat[pid1], flat[pid2],
                title_mw_sets[pid1], title_mw_sets[pid2],
                kv_mw_sets[pid1], kv_mw_sets[pid2],
                weights=wt
            )
            if sim >= min_th:
                weighted_edges.append((sim, pid1, pid2))
        
        weighted_edges.sort(key=lambda x: x[0], reverse=True)

        for th in threshold_grid:
            current_edges = [e for e in weighted_edges if e[0] >= th]
            current_valid_set = {frozenset([e[1], e[2]]) for e in current_edges}

            relevant_products = set()
            for _, p1, p2 in current_edges:
                relevant_products.add(p1)
                relevant_products.add(p2)
            
            p_to_cid = {pid: pid for pid in relevant_products}
            clusters = {pid: [pid] for pid in relevant_products}
            
            for _, pid1, pid2 in current_edges:
                cid1 = p_to_cid[pid1]
                cid2 = p_to_cid[pid2]

                if cid1 == cid2: continue

                c1_mem = clusters[cid1]
                c2_mem = clusters[cid2]

                if _can_merge(c1_mem, c2_mem, flat, current_valid_set, check_lsh_compliance=True):
                    if len(c1_mem) < len(c2_mem):
                        c1_mem, c2_mem = c2_mem, c1_mem
                        cid1, cid2 = cid2, cid1
                    
                    clusters[cid1].extend(clusters[cid2])
                    for p in clusters[cid2]:
                        p_to_cid[p] = cid1
                    del clusters[cid2]

            final_clusters = list(clusters.values())
            
            _, _, _, _, _, f1_val = eval_clusters(final_clusters, true_pairs_all)

            if f1_val > best_f1:
                best_f1 = f1_val
                best_config = {
                    'threshold': th,
                    'weights': wt,
                    'size': 'inf'
                }
                best_clusters = final_clusters

    return best_config, best_f1, best_clusters