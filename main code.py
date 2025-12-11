import numpy as np
import random
from collections import defaultdict

# Import your modules
from minhashing import compute_minhash_signatures
from evaluation import run_t_sweep, filter_candidates_with_stats, filter_pairs_to_ids
from lsh import lsh_band_candidate_pairs
from utils import load_data, get_true_pairs, build_binary_vectors, get_msm_features, plot_results, plot_pruning_impact
from clustering import grid_search_joint
from model_words import build_brand_cache
from evaluation import choose_br_from_threshold


path = "C:/Users/jeroe/Documents/5th year Master Econometrics/2. Computer Science/assignment/TVs-all-merged/TVs-all-merged.json"
seed = 10
total_bootstraps = 100
stop_freq_val = 0.1 

t_grid = [round(x, 2) for x in np.arange(0, 1.01, 0.05)]
msm_threshold_grid = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 
msm_weight_grid = []
for w1 in [0.2, 0.3, 0.4, 0.5]:
    for w2 in [0.2, 0.3, 0.4, 0.5]:
        w3 = round(1.0 - w1 - w2, 2)
        if 0 <= w3 <= 1:
            msm_weight_grid.append((w1, w2, w3))

# --- STEP 1: LOAD DATA ---
print("\n--- Load Dataset ---")
flat, true_groups = load_data(path)
all_pids = list(flat.keys())
n_total = len(all_pids)
print(f"Total Products: {n_total}")
true_pairs_all = get_true_pairs(true_groups)

# BUILD BRAND CACHE
build_brand_cache(flat.values())

# --- VISUALIZE IMPACT OF 10% RULE ---
print("\n--- Generating Impact Visualization (10% Rule) ---")
plot_pruning_impact(flat, stop_freq_val)

# --- STORAGE FOR AVERAGING RESULTS ---
bootstrap_results_by_t = defaultdict(list)
vocab_stats_log = []
all_pruned_sets = [] 
global_theoretical_ceiling = 0.0

print(f"\n--- Starting {total_bootstraps} Full Bootstrap Cycles ---")

for i in range(total_bootstraps):
    print(f"\n{'='*20} BOOTSTRAP RUN {i+1}/{total_bootstraps} {'='*20}")
    
    # 1. SETUP DETERMINISTIC RANDOMNESS FOR THIS RUN
    run_seed = seed + i 
    rng_boot = random.Random(run_seed)
    
    # 2. SPLIT DATA
    bootstrap_sample = rng_boot.choices(all_pids, k=n_total)
    
    train_keys = set(bootstrap_sample)
    test_keys = set(all_pids) - train_keys
    
    train_flat = {k: flat[k] for k in train_keys}
    test_flat = {k: flat[k] for k in test_keys}
    
    print(f"Split: {len(train_flat)} Train, {len(test_flat)} Test (Seed: {run_seed})")
    
    train_true_pairs = filter_pairs_to_ids(true_pairs_all, train_keys)
    
    print(">> Training Phase (Optimizing MSM Params over Grid of LSH Thresholds)...")
    
    train_vecs, train_stats = build_binary_vectors(train_flat, stop_freq=stop_freq_val)
    vocab_stats_log.append(train_stats)
    all_pruned_sets.append(train_stats['pruned_words'])

    # --- DIAGNOSTIC VERIFICATION (Run 1 Only) ---
    if i == 0:
        print("\n   [VERIFICATION] Checking Theoretical Max Fraction (Pairs sharing > 0 words)...")
        vec_list = list(train_vecs.values())
        mat = np.array(vec_list) 
        dot_products = mat @ mat.T
        n_prods = len(vec_list)
        total_pairs = n_prods * (n_prods - 1) // 2
        hits = np.sum(dot_products > 0)
        hits = (hits - n_prods) // 2
        # Store global theoretical theoretical ceiling for Funnel print
        global_theoretical_ceiling = hits / total_pairs 
        print(f"   Theoretical LSH Ceiling (Jaccard > 0): {global_theoretical_ceiling:.4f}")
        print(f"   (This explains why Fraction of Comparisons caps at ~{global_theoretical_ceiling:.2f})\n")
    # -------------------------------------------------------------

    vocab_size_train = len(next(iter(train_vecs.values())))
    n_hashes = int(vocab_size_train * 0.5)
    
    train_sigs, train_order, _, _ = compute_minhash_signatures(train_vecs, n_hashes=n_hashes, seed=run_seed)
    train_title_mw, train_kv_mw = get_msm_features(train_flat)

    best_config = None
    best_train_f1_overall = -1.0
    
    train_lsh_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for train_t in train_lsh_thresholds:
        b_train, r_train, _ = choose_br_from_threshold(n_hashes, train_t, allow_partial_last_band=True)
        _, train_raw_candidates = lsh_band_candidate_pairs(train_sigs, train_order, b=b_train, r=r_train)
        
        train_candidates_filtered, _ = filter_candidates_with_stats(train_raw_candidates, train_flat)
        
        if not train_candidates_filtered:
            continue

        curr_config, curr_f1, _ = grid_search_joint(
            train_candidates_filtered, train_flat, train_title_mw, train_kv_mw, train_true_pairs,
            weight_grid=msm_weight_grid, threshold_grid=msm_threshold_grid
        )
        
        if curr_f1 > best_train_f1_overall:
            best_train_f1_overall = curr_f1
            best_config = curr_config

    print(f"   [Train] Best MSM Found: Th={best_config['threshold']}, W={best_config['weights']} (F1: {best_train_f1_overall:.4f})")

    # 3. TESTING PHASE
    print(">> Testing Phase (Sweeping t for Plotting)...")
    
    test_vecs, _ = build_binary_vectors(test_flat, stop_freq=stop_freq_val)
    
    test_sigs, test_order, _, _ = compute_minhash_signatures(test_vecs, n_hashes=n_hashes, seed=run_seed)
    test_title_mw, test_kv_mw = get_msm_features(test_flat)

    test_sweep_results = run_t_sweep(
        signatures=test_sigs,
        order=test_order,
        true_groups=true_groups,
        products_data=test_flat,
        n_hashes=n_hashes,
        seed=run_seed, 
        t_grid=t_grid,
        bootstrap_runs=1,
        bootstrap_frac=1.0,
        msm_config=best_config,         
        feature_sets=(test_title_mw, test_kv_mw)
    )
    
    for row in test_sweep_results:
        bootstrap_results_by_t[row['t']].append(row)

print("\n" + "="*40)
print(f"CALCULATING AVERAGES OVER {total_bootstraps} BOOTSTRAPS")
print("="*40)

averaged_results = []

for t in t_grid:
    runs = bootstrap_results_by_t[t]
    if not runs: continue
    
    avg_row = {
        't': t,
        'b': runs[0]['b'],
        'r': runs[0]['r'],
        'frac': np.mean([r['frac'] for r in runs]),
        'pc': np.mean([r['pc'] for r in runs]),
        'pq': np.mean([r['pq'] for r in runs]),
        'f1_star': np.mean([r['f1_star'] for r in runs]),
        'f1': np.mean([r['f1'] for r in runs]),
        'frac_inch_pruned': np.mean([r['frac_inch_pruned'] for r in runs]),
        'frac_msm_of_cand': np.mean([r['frac_msm_of_cand'] for r in runs]),
        'frac_msm_of_total': np.mean([r['frac_msm_of_total'] for r in runs]),
        # Absolute Counts
        'n_lsh_cand': np.mean([r['n_lsh_cand'] for r in runs]),
        'n_inch_pruned': np.mean([r['n_inch_pruned'] for r in runs]),
        'n_msm': np.mean([r['n_msm'] for r in runs]),
    }
    averaged_results.append(avg_row)

final_plot_data = [row for row in averaged_results if row['pc'] > 0]

print("\nFinal Averaged Plot Data:")
print(f"{'t':<5} {'Frac':<8} {'PC':<8} {'PQ':<8} {'F1*':<8} {'Final F1':<10} {'MSM Pairs':<10} {'Surv cand post3 %':<20} {'Surv Tot post3%':<18} {'Inch cand Pruned %':<20}")

for row in final_plot_data:
    print(f"{row['t']:<5.2f} {row['frac']:<8.3f} {row['pc']:<8.3f} {row['pq']:<8.3f} {row['f1_star']:<8.3f} {row['f1']:<10.3f} {row['n_msm']:<10.1f} {row['frac_msm_of_cand']*100:<20.2f} {row['frac_msm_of_total']*100:<18.2f} {row['frac_inch_pruned']*100:<20.2f}")

# --- VOCAB STATS & PRUNED WORDS ---
print("\n" + "="*40)
print("VOCABULARY AND PRUNING STATISTICS (Cumulative)")
print("="*40)
if vocab_stats_log:
    avg_raw = np.mean([s['raw_vocab_size'] for s in vocab_stats_log])
    avg_pruned = np.mean([s['pruned_count'] for s in vocab_stats_log])
    print(f"Original Model Words (Avg):  {avg_raw:.1f}")
    print(f"Pruned ('Stop') Words (Avg): {avg_pruned:.1f} (Freq > {stop_freq_val})")

    if all_pruned_sets:
        all_intersection = set.intersection(*all_pruned_sets)
        print(f"\nINTERSECTION of Pruned Words (Across {len(all_pruned_sets)} Bootstraps) [Total: {len(all_intersection)}]:")
        print(sorted(list(all_intersection)))

# --- FUNNEL TABLE (TRUE VALUES) ---
print("\n" + "="*40)
print("EFFICIENCY FUNNEL (Averaged over Bootstraps)")
print("Based on Threshold with Best Final F1")
print("="*40)

if final_plot_data:
    best_row = max(final_plot_data, key=lambda x: x['f1'])
    
    total_pairs = n_total * (n_total - 1) / 2
    pairs_jaccard_pos = total_pairs * global_theoretical_ceiling
    
    # Use direct averages captured from run_t_sweep
    lsh_candidates = best_row['n_lsh_cand']
    inch_removed_count = best_row['n_inch_pruned']
    msm_pairs = best_row['n_msm']
    
    print(f"Selected Threshold t:     {best_row['t']}")
    print(f"Achieved F1 Score:        {best_row['f1']:.4f}")
    print("-" * 50)
    print(f"1. Total Possible Pairs:      {int(total_pairs):<10,}")
    print(f"2. Pairs w/ Jaccard > 0:      {int(pairs_jaccard_pos):<10,} (Theoretical Ceiling, ~{global_theoretical_ceiling*100:.1f}%)")
    print(f"3. LSH Candidates:            {int(lsh_candidates):<10,} (Evaluated)")
    print(f"4. Removed by Inch Blocking:  {int(inch_removed_count):<10,} (Subset of Candidates)")
    print(f"5. Final MSM Pairs:           {int(msm_pairs):<10,} (Input to Clusters)")
    print("-" * 50)
    print(f"Total Efficiency Gain:        {(1 - msm_pairs/total_pairs)*100:.2f}% reduction")
print("="*40)

plot_results(final_plot_data)
print("\nPlots generated.")