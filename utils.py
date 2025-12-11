import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from model_words import extract_model_words, extract_title_model_words, extract_kv_model_words
import re

def extract_shop_from_url(url):
    if not url: return "unknown"
    try:
        clean_url = url.replace("http://", "").replace("https://", "")
        parts = clean_url.split('.')
        if len(parts) >= 2:
            return parts[1].lower().strip()
        else:
            return "unknown"
    except:
        return "unknown"

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    flat = {}
    true_groups = {} 
    
    for model_id, products in data.items():
        pids = []
        for i, prod in enumerate(products):
            pid = f"{model_id}_{i}"
            pids.append(pid)
            url_shop = extract_shop_from_url(prod.get('url', ''))
            prod['shop'] = url_shop
            flat[pid] = prod
        true_groups[model_id] = pids
        
    return flat, true_groups

def get_true_pairs(true_groups):
    true_pairs_all = set()
    for _, uids in true_groups.items():
        if len(uids) >= 2:
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    true_pairs_all.add(frozenset([uids[i], uids[j]]))
    return true_pairs_all

def build_binary_vectors(flat, stop_freq=0.10):
    all_mw_raw = {uid: extract_model_words(prod) for uid, prod in flat.items()}

    global_counts = defaultdict(int)
    total_products = len(flat)
    for mws in all_mw_raw.values():
        for w in mws:
            global_counts[w] += 1

    valid_mw = {w for w, count in global_counts.items() if count < (total_products * stop_freq)}
    pruned_mw = set(global_counts.keys()) - valid_mw
    
    n_raw = len(global_counts)
    n_valid = len(valid_mw)
    n_pruned = n_raw - n_valid

    model_word_sets = {}
    for uid, mws in all_mw_raw.items():
        model_word_sets[uid] = {w for w in mws if w in valid_mw}

    MW = sorted(list(valid_mw))
    idx = {mw: i for i, mw in enumerate(MW)}

    binary_vectors = {}
    for uid, mw_set in model_word_sets.items():
        vec = [0] * len(MW)
        for w in mw_set:
            vec[idx[w]] = 1
        binary_vectors[uid] = vec
        
    stats = {
        "raw_vocab_size": n_raw,
        "final_vocab_size": n_valid,
        "pruned_count": n_pruned,
        "pruned_words": pruned_mw 
    }

    return binary_vectors, stats

def get_msm_features(flat):
    title_mw_sets = {uid: extract_title_model_words(prod.get("title", "")) for uid, prod in flat.items()}
    kv_mw_sets = {uid: extract_kv_model_words(prod.get("featuresMap", {}), title_mw=title_mw_sets[uid]) for uid, prod in flat.items()}
    return title_mw_sets, kv_mw_sets

# --- PLOTTING FUNCTIONS ---

def plot_pruning_impact(flat, stop_count_val, output_dir="plots"):
    """
    Generates histograms comparing Jaccard Similarities (Side-by-Side).
    Visuals updated for large font sizes.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("   [Analysis] Calculating full pairwise Jaccard matrices for impact visualization...")

    def get_jaccard_distribution(stop_val):
        vecs, _ = build_binary_vectors(flat, stop_freq=stop_val)
        if not vecs: return np.array([])
        
        matrix = np.array(list(vecs.values()), dtype=bool) 
        intersection = matrix.astype(int) @ matrix.astype(int).T
        
        row_sums = intersection.diagonal()
        unions = row_sums[:, None] + row_sums[None, :] - intersection
        
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard_mat = intersection / unions.astype(float)
            jaccard_mat[unions == 0] = 0.0
        
        tri_indices = np.triu_indices(len(vecs), k=1)
        return jaccard_mat[tri_indices]

    # Calculate Data
    dist_raw = get_jaccard_distribution(stop_val=1.0) 
    dist_pruned = get_jaccard_distribution(stop_val=stop_count_val) 

    # --- PRINT PERCENTAGE OF ZERO PAIRS ---
    total_pairs = len(dist_raw)
    zeros_raw = np.sum(dist_raw == 0)
    zeros_pruned = np.sum(dist_pruned == 0)

    pct_raw = (zeros_raw / total_pairs) * 100 if total_pairs > 0 else 0
    pct_pruned = (zeros_pruned / total_pairs) * 100 if total_pairs > 0 else 0

    print(f"\n   --- Jaccard Similarity Analysis ---")
    print(f"   Total Pairs Checked: {total_pairs}")
    print(f"   Pairs with Jaccard = 0 (Before Pruning): {zeros_raw} ({pct_raw:.2f}%)")
    print(f"   Pairs with Jaccard = 0 (After 10% Rule): {zeros_pruned} ({pct_pruned:.2f}%)")
    print(f"   (The {pct_pruned:.2f}% of pairs are mathematically invisible to LSH because they share NO rare words)\n")

    # --- PLOT: SIDE-BY-SIDE (Independent Linear Scales) ---
    # Increased figsize slightly to accommodate large text
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    
    # Left: Raw
    ax1.hist(dist_raw, bins=50, range=(0.0, 1.0), color='red', alpha=0.7)
    ax1.set_title("Before Pruning (Raw Vectors)", fontsize=22)
    ax1.set_xlabel("Jaccard Similarity", fontsize=20)
    ax1.set_ylabel("Number of Pairs", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # Right: Pruned
    ax2.hist(dist_pruned, bins=50, range=(0.0, 1.0), color='blue', alpha=0.7)
    ax2.set_title(f"After {int(stop_count_val*100)}% Pruning", fontsize=22)
    ax2.set_xlabel("Jaccard Similarity", fontsize=20)
    ax2.set_ylabel("Number of Pairs", fontsize=20) 
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    # Force plain int notation
    ax2.ticklabel_format(style='plain', axis='y')

    plt.suptitle("Impact of Pruning (Independent Scales)", fontsize=26)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path2 = os.path.join(output_dir, "impact_side_by_side.png")
    plt.savefig(out_path2, bbox_inches="tight")
    plt.close()
    
    print(f"   [Analysis] Impact plots saved to: {out_path2}")

def plot_results(results, output_dir="plots"):
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    sorted_res = sorted(results, key=lambda r: r["frac"])
    frac = [r["frac"] for r in sorted_res]
    
    # 1. Standard Individual Plots (Requested to keep)
    # UPDATED TITLES (Removed 'Recall'/'Precision', Fixed 'F1 measure')
    plots_config = [
        ("pc", "Pair Completeness", "pair_completeness.png"),
        ("pq", "Pair Quality", "pair_quality.png"),
        ("f1_star", "F1* Measure", "f1_star_vs_fraction.png"),
        ("f1", "Final F1 measure", "f1_vs_fraction.png")
    ]

    saved_files = []

    for key, ylabel, filename in plots_config:
        if key in sorted_res[0]:
            vals = [r[key] for r in sorted_res]
            plt.figure(figsize=(10, 8)) # Slightly larger for big text
            plt.plot(frac, vals, marker="o", linestyle="-", linewidth=2.5, markersize=6)
            
            # INCREASED FONT SIZES
            plt.xlabel("Fraction of Comparisons", fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.title(f"{ylabel} vs Fraction of Comparisons", fontsize=24)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.6)
            
            out_path = os.path.join(output_dir, filename)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            saved_files.append(out_path)

    # 2. Combined 2x2 Figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Performance Metrics vs Fraction of Comparisons", fontsize=28)

    # Flatten axis array for easy iteration
    ax_list = axs.flatten()
    
    # Order: PC, PQ, F1*, F1
    metrics_to_plot = ["pc", "pq", "f1_star", "f1"]
    # UPDATED TITLES HERE TOO
    titles = [
        "Pair Completeness", 
        "Pair Quality", 
        "F1* Measure (LSH Only)", 
        "Final F1 measure"
    ]
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in sorted_res[0]:
            vals = [r[metric] for r in sorted_res]
            ax = ax_list[i]
            # Thicker lines
            ax.plot(frac, vals, marker="o", linestyle="-", linewidth=2.5, markersize=5)
            
            # LARGE FONTS
            ax.set_title(titles[i], fontsize=22)
            ax.set_xlabel("Fraction of Comparisons", fontsize=18)
            ax.set_ylabel("Value", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    out_2x2 = os.path.join(output_dir, "combined_metrics_2x2.png")
    plt.savefig(out_2x2, bbox_inches="tight")
    plt.close()
    saved_files.append(out_2x2)

    print(f"\nPlots saved to directory '{output_dir}':")
    for p in saved_files:
        print(f" - {os.path.basename(p)}")