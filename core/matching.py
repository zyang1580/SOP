"""
Core Matching Engine (GPU Accelerated)
"""
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager

def _worker_with_queue(queue, treatment_hash_map, control_hash_map, treatment_group, control_group, 
                       nonexact_categorical_cols, numerical_cols, threshold_dict, batch_size, device, result_dict):
    local_indices = torch.full((len(treatment_group),), -1, dtype=torch.long, device=device)
    local_distances = torch.full((len(treatment_group),), float('inf'), dtype=torch.float32, device=device)

    while True:
        hash_key = queue.get()
        if hash_key is None: break
        if hash_key not in control_hash_map: continue

        t_data = treatment_group.iloc[treatment_hash_map[hash_key]]
        c_data = control_group.iloc[control_hash_map[hash_key]]

        for t_start in range(0, len(t_data), batch_size):
            t_batch = t_data.iloc[t_start:t_start + batch_size]
            t_num = torch.tensor(t_batch[[f"scaled_{col}" for col in numerical_cols]].values, dtype=torch.float32).to(device)
            t_orig = torch.tensor(t_batch[numerical_cols].values, dtype=torch.float32).to(device)

            for c_start in range(0, len(c_data), batch_size):
                c_batch = c_data.iloc[c_start:c_start + batch_size]
                c_num = torch.tensor(c_batch[[f"scaled_{col}" for col in numerical_cols]].values, dtype=torch.float32).to(device)
                c_orig = torch.tensor(c_batch[numerical_cols].values, dtype=torch.float32).to(device)

                distances = torch.norm(t_num[:, None, :] - c_num[None, :, :], dim=2, dtype=torch.float32)
                diff_matrix = torch.abs(t_orig[:, None, :] - c_orig[None, :, :])
                max_matrix = torch.max(torch.abs(t_orig[:, None, :]), torch.abs(c_orig[None, :, :])) + 1e-6
                rel_diff = diff_matrix / max_matrix

                zero_mask = (t_orig[:, None, :] == 0) & (c_orig[None, :, :] == 0)
                valid_mask = torch.all((rel_diff <= threshold_dict.get('default', 0.8)) | zero_mask, dim=2)
                
                for col_idx, col_name in enumerate(numerical_cols):
                    if col_name in threshold_dict and threshold_dict[col_name] != 1:
                        valid_mask &= (rel_diff[:, :, col_idx] <= threshold_dict[col_name])

                if not valid_mask.any(): continue

                valid_distances = distances.masked_fill(~valid_mask, float('inf'))
                min_distances, min_indices = valid_distances.min(dim=1)

                t_global = torch.tensor(t_batch.index, device=device)
                c_global = torch.tensor(c_batch.index, device=device)[min_indices]

                update_mask = min_distances < local_distances[t_global]
                local_distances[t_global[update_mask]] = min_distances[update_mask]
                local_indices[t_global[update_mask]] = c_global[update_mask]

    result_dict[mp.current_process().name] = (local_indices.cpu().numpy(), local_distances.cpu().numpy())

def run_parallel_matching(treatment_hash_map, control_hash_map, treatment_group, control_group, numerical_cols, threshold_dict):
    queue, manager = Queue(), Manager()
    result_dict = manager.dict()
    for k in treatment_hash_map.keys(): queue.put(k)
    for _ in range(4): queue.put(None)
    
    procs = []
    for i in range(4):
        device = f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
        p = Process(target=_worker_with_queue, args=(queue, treatment_hash_map, control_hash_map, treatment_group, control_group, [], numerical_cols, threshold_dict, 5000, device, result_dict))
        procs.append(p); p.start()
    for p in procs: p.join()

    g_idx = np.full(len(treatment_group), -1, dtype=np.int64)
    g_dist = np.full(len(treatment_group), np.inf, dtype=np.float32)
    for idx, dist in result_dict.values():
        mask = dist < g_dist
        g_dist[mask], g_idx[mask] = dist[mask], idx[mask]
    return g_idx, g_dist

def evaluate_balance(df, t_col, n_cols, p_treat, m_ctrl, prefix):
    import warnings; warnings.filterwarnings("ignore")
    results = []
    for c in n_cols:
        pt, pc = df[df[t_col]==1][c].dropna(), df[df[t_col]==0][c].dropna()
        tt, mc = p_treat[c], m_ctrl[c]
        v_pt, v_pc = np.var(pt, ddof=1), np.var(pc, ddof=1)
        v_tt, v_mc = np.var(tt, ddof=1), np.var(mc, ddof=1)
        results.append({
            'Feature': c,
            'Pre-Match SMD': (np.mean(pt)-np.mean(pc))/np.sqrt((v_pt+v_pc)/2) if v_pt+v_pc>0 else np.nan,
            'Pre-Match VR': v_pt/v_pc if v_pc>0 else np.nan,
            'Post-Match SMD': (np.mean(tt)-np.mean(mc))/np.sqrt((v_tt+v_mc)/2) if v_tt+v_mc>0 else np.nan,
            'Post-Match VR': v_tt/v_mc if v_mc>0 else np.nan
        })
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{prefix}_balance.csv", index=False)
    return res_df
