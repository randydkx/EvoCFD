import numpy as np
import pandas as pd
import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from model.utils import deep_learning
from model.stage_dataclass import dataclass as StageDataContainer
import re
from model.dlmodel.model.utils import set_gpu, set_seeds, mkdir
from sklearn.preprocessing import KBinsDiscretizer
from model.logger import Logger, print_to_console
import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


deep_learning_models = ['EvoCFD']

EXCLUDE_COLUMNS = ['dt', 'label']
STAGE_EXPORT_DIR = Path('assets/stage_exports')
SAMPLE_RANDOM_STATE = 42
RUN_LOG_ROOT = Path('Results')
DATASET_ALIASES = {
    'ieee': 'ieee',
    'perfraud': 'PerFraud',
    'merfraud': 'MerFraud',
}
DATASET_STAGE_SUMMARY_FILES = {
    'ieee': 'ieee_stage_summary.json',
    'perfraud': 'PerFraud_stage_summary.json',
    'merfraud': 'MerFraud_stage_summary.json',
}
DATASET_TEMPORAL_DT_MODES = {
    'ieee': 'elapsed_seconds',
}
DATASET_TEMPORAL_EMBEDDING_PERIODS = {}
DATASET_TEMPORAL_PERIOD_LABELS = {}


def _parse_metric_from_filename(filename: str):
    match = re.search(r'(recall|metric)([0-9]+(?:\.[0-9]+)?)', filename)
    if not match:
        return None
    tag = match.group(1)
    raw_value = float(match.group(2))
    value = raw_value if tag != 'recall' else raw_value / 100.0
    return {'tag': tag, 'value': value, 'raw_value': raw_value}


def _update_evaluation_summary(model_path: str, model_type: str):
    eval_root = os.path.join(model_path, 'evaluation_best', model_type)
    summary = {}
    if os.path.isdir(eval_root):
        for dataset_stage in sorted(os.listdir(eval_root)):
            stage_dir = os.path.join(eval_root, dataset_stage)
            if not os.path.isdir(stage_dir):
                continue
            best_file = None
            best_metric = float('-inf')
            best_payload = None
            for fname in os.listdir(stage_dir):
                payload = _parse_metric_from_filename(fname)
                if payload is None:
                    continue
                if payload['value'] > best_metric:
                    best_metric = payload['value']
                    best_file = fname
                    best_payload = payload
            if best_file is None or best_payload is None:
                continue
            dataset, stage = dataset_stage.split('-', 1) if '-' in dataset_stage else (dataset_stage, dataset_stage)
            stage_map = summary.setdefault(dataset, {})
            stage_map[stage] = {
                'metric_tag': best_payload['tag'],
                'metric_value_display': best_payload['raw_value'],
                'checkpoint': os.path.join('evaluation_best', model_type, dataset_stage, best_file)
            }
    os.makedirs(eval_root, exist_ok=True)
    summary_path = os.path.join(eval_root, 'summary.json')
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _init_logger(dataset: str, stage: str, model: str, dl_args) -> Logger:
    logger = Logger(logging_dir = dl_args.save_path, DEBUG=False)
    logger.set_logfile('log.txt')
    print_to_console(f'set logger file -> {dl_args.save_path}')
    return logger



def _canonical_dataset_key(dataset_key: str) -> str:
    raw_key = str(dataset_key or '').strip()
    return DATASET_ALIASES.get(raw_key.lower(), raw_key)


def _dataset_lookup_key(dataset_key: str) -> str:
    return _canonical_dataset_key(dataset_key).lower()


def _data_root() -> Path:
    return Path(os.environ.get('EVO_CFD_DATA_ROOT', '/data'))


def _dataset_file(dataset_key: str) -> Path:
    return _data_root() / f"fraud_corp_{_canonical_dataset_key(dataset_key)}_output_2024.csv"


def _summary_path(dataset_key: str) -> Path:
    lookup_key = _dataset_lookup_key(dataset_key)
    return STAGE_EXPORT_DIR / DATASET_STAGE_SUMMARY_FILES.get(
        lookup_key,
        f"{_canonical_dataset_key(dataset_key)}_stage_summary.json"
    )





def _resolve_temporal_dt_mode(dataset_key: str) -> str:
    lookup_key = _dataset_lookup_key(dataset_key)
    return DATASET_TEMPORAL_DT_MODES.get(lookup_key, 'calendar_date')


def _resolve_temporal_embedding_periods(dataset_key: str):
    lookup_key = _dataset_lookup_key(dataset_key)
    periods = DATASET_TEMPORAL_EMBEDDING_PERIODS.get(lookup_key)
    return list(periods) if periods is not None else None


def _resolve_temporal_period_labels(dataset_key: str):
    lookup_key = _dataset_lookup_key(dataset_key)
    labels = DATASET_TEMPORAL_PERIOD_LABELS.get(lookup_key)
    return list(labels) if labels is not None else ['year', 'month', 'week', 'day']




def _normalize_dataset_dt(df: pd.DataFrame, dataset_key: str, splits: dict):
    lookup_key = _dataset_lookup_key(dataset_key)
    if 'dt' not in df.columns:
        raise ValueError("Dataset must include a 'dt' column before dt normalization.")

    normalized = df.copy()
    normalized['dt'] = pd.to_numeric(normalized['dt'], errors='raise').astype(np.int64)
    processed_splits = dict(splits)

    if lookup_key == 'ieee':
        # Preserve the raw IEEE elapsed-seconds timeline so temporal features match
        # the legacy repo exactly. The old pipeline did not re-anchor dt to zero.
        test_start, test_stop = processed_splits['test']
        dt_processing = {
            'mode': 'elapsed_seconds',
            'origin_dt': 0,
            'raw_dt_range': [
                int(df['dt'].min()),
                int(df['dt'].max()),
            ],
            'normalized_dt_range': [
                int(normalized['dt'].min()),
                int(normalized['dt'].max()),
            ],
            'raw_test_range': [int(test_start), int(test_stop)],
            'normalized_test_range': list(processed_splits['test']),
        }
        return normalized, processed_splits, dt_processing

    dt_processing = {
        'mode': 'calendar_date',
        'origin_dt': None,
        'raw_dt_range': [int(normalized['dt'].min()), int(normalized['dt'].max())],
        'normalized_dt_range': [int(normalized['dt'].min()), int(normalized['dt'].max())],
        'raw_test_range': [int(processed_splits['test'][0]), int(processed_splits['test'][1])],
        'normalized_test_range': [int(processed_splits['test'][0]), int(processed_splits['test'][1])],
    }
    return normalized, processed_splits, dt_processing


def print_top_features(method_name, metric_name, metric_dict, top_k=30, log_file=None):
    sorted_feats = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    lines = [f"--- Top {top_k} Features by {metric_name} ({method_name}) ---"]
    for feat, val in sorted_feats:
        lines.append(f"{feat}: {val:.4f}")
    output = "\n".join(lines)
    print(output)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(output + "\n\n")

def log_global_stats(metric_name, values_dict, log_file):
    values = np.array(list(values_dict.values()))
    
    lines = []
    lines.append(f"\n[Metrics] {metric_name} Global Summary:")
    
    if len(values) > 0:
        g_max = np.max(values)
        g_mean = np.mean(values)
        g_med = np.median(values)
        lines.append(f"  Max   : {g_max:.6f}")
        lines.append(f"  Mean  : {g_mean:.6f}")
        lines.append(f"  Median: {g_med:.6f}")
        lines.append("-" * 60)

        lines.append(f"  [Scheme 3] L-p Norm Average:")
        p_params = [5, 10, 15, 20, 25, 30]
        for p in p_params:
            try:
                score_lp = np.power(np.mean(np.power(values, p)), 1/p)
                lines.append(f"    p={p:<2} : {score_lp:.6f}")
            except Exception:
                lines.append(f"    p={p:<2} : Overflow/Error")
        lines.append("-" * 60)

        lines.append(f"  [Scheme 4] Softmax Weighted Average:")
        t_params = [5, 10, 15, 20, 25, 30]
        for T in t_params:
            try:
                logits = values * T
                logits_safe = logits - np.max(logits)
                exp_vals = np.exp(logits_safe)
                weights = exp_vals / np.sum(exp_vals)
                score_softmax = np.sum(values * weights)
                lines.append(f"    T={T:<2} : {score_softmax:.6f}")
            except Exception as e:
                lines.append(f"    T={T:<2} : Error {e}")
        lines.append("-" * 60)
    else:
        lines.append("  No data calculated.")
    
    output_str = "\n".join(lines) + "\n"
    
    print(output_str)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(output_str)

def calculate_drift_with_clip(df_prev, df_curr, candidate_features=None, n_bins=100):
    cols_prev = set(df_prev.columns)
    cols_curr = set(df_curr.columns)
    
    if candidate_features is not None:
        shared_features = sorted(list(cols_prev & cols_curr & set(candidate_features)))
    else:
        shared_features = sorted(list(cols_prev & cols_curr))
    
    shared_features = [f for f in shared_features if f not in ['dt', 'label']]

    js_dict = {}
    l1_dict = {}
    
    print(f"\n[Metrics] Calculating JS & L1 (Strict History Basis) for {len(shared_features)} features...")
    
    for feat in tqdm(shared_features, desc="Calculating Drift"):
        try:
            prev_vals = pd.to_numeric(df_prev[feat], errors='coerce').dropna().values
            curr_vals = pd.to_numeric(df_curr[feat], errors='coerce').dropna().values
        except Exception:
            js_dict[feat] = 0.0
            l1_dict[feat] = 0.0
            continue
        
        if len(prev_vals) == 0 or len(curr_vals) == 0: 
            js_dict[feat] = 0.0
            l1_dict[feat] = 0.0
            continue
            
        prev_std = np.std(prev_vals)
        prev_mean = np.mean(prev_vals)
        
        if prev_std == 0:
            if np.std(curr_vals) == 0 and np.mean(curr_vals) == prev_mean:
                js_dict[feat] = 0.0
                l1_dict[feat] = 0.0
            else:
                js_dict[feat] = 1.0 
                l1_dict[feat] = 2.0
            continue

        try:
            n_u_prev = len(np.unique(prev_vals))
            current_n_bins = min(n_bins, max(2, n_u_prev))
            
            lower_bound = np.percentile(prev_vals, 1)
            upper_bound = np.percentile(prev_vals, 99)
            
            if upper_bound - lower_bound < 1e-6:
                lower_bound -= 1e-3
                upper_bound += 1e-3
                
            prev_clipped = np.clip(prev_vals, lower_bound, upper_bound)
            curr_clipped = np.clip(curr_vals, lower_bound, upper_bound)
            
            prev_hist, _ = np.histogram(prev_clipped, bins=current_n_bins, range=(lower_bound, upper_bound))
            curr_hist, _ = np.histogram(curr_clipped, bins=current_n_bins, range=(lower_bound, upper_bound))
            
            prev_sum = prev_hist.sum()
            curr_sum = curr_hist.sum()
            
            if prev_sum == 0 or curr_sum == 0:
                js_dict[feat] = 0.0
                l1_dict[feat] = 0.0
                continue
                
            p_prev = prev_hist / prev_sum
            p_curr = curr_hist / curr_sum

            m = 0.5 * (p_prev + p_curr)
            
            # KL(P || M) = sum(P * log2(P / M))
            kl_pm = np.sum(np.where(p_prev > 0, p_prev * np.log2(p_prev / m), 0))
            kl_qm = np.sum(np.where(p_curr > 0, p_curr * np.log2(p_curr / m), 0))

            # JS Divergence
            js_divergence = 0.5 * (kl_pm + kl_qm)
            
            js_dist = np.sqrt(np.maximum(js_divergence, 0.0))
            
            l1_dist = np.sum(np.abs(p_prev - p_curr))
            
            js_dict[feat] = js_dist
            l1_dict[feat] = l1_dist
            
        except Exception as e:
            print(f"Error calculating drift for {feat}: {e}")
            js_dict[feat] = 0.0
            l1_dict[feat] = 0.0
    
    return l1_dict, js_dict

def calculate_drift_quantile(df_prev, df_curr, candidate_features=None, n_bins=100):
    cols_prev = set(df_prev.columns)
    cols_curr = set(df_curr.columns)
    
    if candidate_features is not None:
        shared_features = sorted(list(cols_prev & cols_curr & set(candidate_features)))
    else:
        shared_features = sorted(list(cols_prev & cols_curr))
    
    shared_features = [f for f in shared_features if f not in ['dt', 'label']]

    js_dict = {}
    l1_dict = {}
    
    print(f"\n[Metrics] Calculating JS & L1 (Quantile + Dynamic Bins) for {len(shared_features)} features...")
    
    for feat in tqdm(shared_features, desc="Calculating Drift"):
        try:
            prev_vals = pd.to_numeric(df_prev[feat], errors='coerce').dropna().values
            curr_vals = pd.to_numeric(df_curr[feat], errors='coerce').dropna().values
        except Exception:
            js_dict[feat] = 0.0
            l1_dict[feat] = 0.0
            continue
            
        if len(prev_vals) == 0 or len(curr_vals) == 0: 
            js_dict[feat] = 0.0
            l1_dict[feat] = 0.0
            continue
            
        prev_std = np.std(prev_vals)
        prev_mean = np.mean(prev_vals)
        
        if prev_std == 0:
            if np.std(curr_vals) == 0 and np.mean(curr_vals) == prev_mean:
                js_dict[feat] = 0.0
                l1_dict[feat] = 0.0
            else:
                js_dict[feat] = 1.0
                l1_dict[feat] = 2.0
            continue

        try:
            n_u_prev = len(np.unique(prev_vals))
            current_n_bins = min(n_bins, max(2, n_u_prev))
            
            quantiles = np.linspace(0, 100, current_n_bins + 1)
            bin_edges = np.percentile(prev_vals, quantiles)
            
            bin_edges = np.unique(bin_edges)
            
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            if len(bin_edges) < 2:
                js_dict[feat] = 0.0 
                l1_dict[feat] = 0.0
                continue

            prev_hist, _ = np.histogram(prev_vals, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_vals, bins=bin_edges)
            
            prev_sum = prev_hist.sum()
            curr_sum = curr_hist.sum()
            
            if prev_sum == 0 or curr_sum == 0:
                js_dict[feat] = 0.0
                l1_dict[feat] = 0.0
                continue

            p_prev = prev_hist / prev_sum
            p_curr = curr_hist / curr_sum

            # M = (P + Q) / 2
            m = 0.5 * (p_prev + p_curr)
            
            # KL(P || M) = sum(P * log2(P / M))
            kl_pm = np.sum(np.where(p_prev > 0, p_prev * np.log2(p_prev / m), 0))
            kl_qm = np.sum(np.where(p_curr > 0, p_curr * np.log2(p_curr / m), 0))

            # JS Divergence
            js_divergence = 0.5 * (kl_pm + kl_qm)
            
            js_dist = np.sqrt(np.maximum(js_divergence, 0.0))
            
            l1_dist = np.sum(np.abs(p_prev - p_curr))
            
            js_dict[feat] = js_dist
            l1_dict[feat] = l1_dist#l1_dist#reverse_kl##l2_dist#
            
        except Exception as e:
            print(f"Error calculating drift for {feat}: {e}")
            js_dict[feat] = 0.0
            l1_dict[feat] = 0.0
    
    return l1_dict, js_dict


def analyze_data_drift(dataset, stage_data, log_dir='./drift_logs'):
    os.makedirs(log_dir, exist_ok=True)
    summary = stage_data.summary
    stages = stage_data.stage_datasets
    stage_infos = summary['stages']
    
    if len(stages) < 2:
        print("[Metrics] Not enough stages to compute drift.")
        return
        
    calc_pairs = []
    for i in range(len(stages) - 1):
        prev_name = stage_infos[i]['name']
        curr_name = stage_infos[i+1]['name']
        
        feats = list(set(stages[i].columns) & set(stages[i+1].columns))
        
        calc_pairs.append({
            'name': f"{prev_name}_to_{curr_name}",
            'prev_df': stages[i],
            'curr_df': stages[i+1],
            'feats': feats
        })
        
    full_lam_dict = {}
    
    for pair in calc_pairs:
        print(f"\n[Metrics] Computing Shift for pair: {dataset} - {pair['name']}")
        current_log_file = os.path.join(log_dir, f"{dataset}_{pair['name']}_features_log.txt")
        
        with open(current_log_file, 'w') as f: 
            f.write(f"Drift Log for {dataset} {pair['name']}\n")

        strategies = [
            (calculate_drift_quantile, "Quantile"),
            (calculate_drift_with_clip, "Clip")
        ]
        
        for func, method_name in strategies:
            print(f"\n>>> Running Strategy: {method_name} ...")
            
            res_l1_dict, res_js_dict = func(pair['prev_df'], pair['curr_df'], candidate_features=pair['feats'])

            print_top_features(method_name, "JS Divergence", res_js_dict, top_k=30, log_file=current_log_file)
            print_top_features(method_name, "L1 Distance", res_l1_dict, top_k=30, log_file=current_log_file)
            
            log_global_stats(f"{method_name} - JS Divergence", res_js_dict, current_log_file)
            log_global_stats(f"{method_name} - L1 Distance", res_l1_dict, current_log_file)

            if method_name == "Quantile":
                full_lam_dict.update(res_l1_dict)
                
    return full_lam_dict


def preprocess_data(X_train, X_test,strategy='quantile'):
    """
    Preprocesses the data with conditional handling for empty test set
    """
    exclude_cols = ['dt', 'label']
    feature_cols = [c for c in X_train.columns if c not in exclude_cols]

    unique_counts = X_train[feature_cols].nunique()
    num_feature_names = unique_counts[unique_counts >= 100].index.tolist()
    cat_feature_names = unique_counts[unique_counts < 100].index.tolist()

    kbin = KBinsDiscretizer(n_bins=100, strategy=strategy, encode='ordinal')
    X_num_train = X_train[num_feature_names]
    kbin.fit(X_num_train)
    
    X_num_binned_train = pd.DataFrame(kbin.transform(X_num_train), 
                                    columns=num_feature_names,
                                    index=X_train.index)
    
    X_num_binned_test = None
    if X_test is not None and not X_test.empty:  
        X_num_binned_test = pd.DataFrame(kbin.transform(X_test[num_feature_names]),
                                       columns=num_feature_names,
                                       index=X_test.index)
    
    X_processed_train = pd.concat([X_num_binned_train, X_train[cat_feature_names], X_train[exclude_cols]], axis=1)
    X_processed_test = pd.concat([X_num_binned_test, X_test[cat_feature_names], X_test[exclude_cols]], axis=1) if X_num_binned_test is not None else None
    
    X_processed_train = X_processed_train[X_train.columns]
    X_processed_test = X_processed_test[X_test.columns] if X_processed_test is not None else None
    
    return X_processed_train, X_processed_test


def _load_full_dataset(filename, chunksize=100000):
    df = pd.DataFrame()
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    with open(filename, 'r') as handle:
        total_rows = max(sum(1 for _ in handle) - 1, 1)
    chunk_list = []
    for chunk in tqdm(
        pd.read_csv(filename, chunksize=chunksize, low_memory=False),
        total=total_rows // chunksize + 1,
        desc="Loading Files"
    ):
        chunk_list.append(chunk)
    if chunk_list:
        df = pd.concat(chunk_list, ignore_index=True)
    return df


def _prepare_subset(df, logger):
    logger.debug(f"Preparing subset: rows={len(df):,}, cols={len(df.columns)}")
    subset = df.copy()
    subset = subset.dropna(axis=1, how='all')
    total_columns = len(subset.columns)
    
    for idx, column in enumerate(subset.columns, start=1):
        if column in EXCLUDE_COLUMNS:
            continue
        column_mean = subset[column].mean()
        if pd.isna(column_mean):
            column_mean = 0.0
        subset[column] = subset[column].fillna(column_mean)
        if idx % 50 == 0 or idx == total_columns:
            logger.debug(f"Filled NaNs for {idx}/{total_columns} columns in subset")
    logger.debug("Subset preparation complete")
    return subset



def _resolve_stage_identifier(stage, stage_count):
    if isinstance(stage, int):
        idx = stage
    else:
        stage_str = str(stage).strip().lower()
        if stage_str in {'m1', 'stage1', 'stage_1'}:
            idx = 1
        elif stage_str in {'m2', 'stage2', 'stage_2'}:
            idx = min(2, stage_count)
        elif stage_str in {'new', 'latest', 'stage_last', 'final'}:
            idx = stage_count
        else:
            digits = re.findall(r'\d+', stage_str)
            if digits:
                idx = int(digits[0])
            else:
                raise ValueError(f"Unrecognized stage identifier: {stage}")
    if idx < 1 or idx > stage_count:
        raise ValueError(f"Stage index {idx} is out of bounds for {stage_count} stages")
    return idx




def _read_summary(dataset_key: str):
    path = _summary_path(dataset_key)
    if not path.exists():
        return None
    with open(path, 'r') as handle:
        return json.load(handle)



def _sample_dataframe(df: pd.DataFrame, sample_ratio: float) -> pd.DataFrame:
    if sample_ratio >= 1.0:
        return df
    sample_size = max(1, int(len(df) * sample_ratio))
    sampled = df.sample(n=sample_size, random_state=SAMPLE_RANDOM_STATE)
    return sampled.sort_index(kind='stable').reset_index(drop=True)


def _ensure_stage_columns(df: pd.DataFrame, feature_union):
    required_columns = feature_union + ['dt', 'label']
    for column in required_columns:
        if column not in df.columns:
            if column in EXCLUDE_COLUMNS:
                raise ValueError(f"Column '{column}' missing from stage data.")
            df[column] = 0.0
    return df[required_columns].copy()


def _resolve_clean_dataset_file(dataset_key: str, summary: dict) -> Path:
    configured_path = _dataset_file(dataset_key)
    if configured_path.exists():
        return configured_path
    summary_source = summary.get('source_file')
    if summary_source:
        summary_path = Path(summary_source)
        if summary_path.exists():
            return summary_path
    return configured_path


def _load_base_frames_from_summary(dataset_key: str, summary: dict, sample_ratio: float, logger):
    source_file = _resolve_clean_dataset_file(dataset_key, summary)
    logger.debug(f"Loading raw data from {source_file}")
    df = _load_full_dataset(str(source_file))
    if 'dt' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must include 'dt' and 'label' columns.")

    df, _, _ = _normalize_dataset_dt(df, dataset_key, _summary_splits(summary))
    logger.debug(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    summary_ratio = float(summary.get('sample_ratio', 1.0))
    if abs(summary_ratio - sample_ratio) > 1e-9:
        raise ValueError(
            f"Requested sample_ratio={sample_ratio} does not match assets summary sample_ratio={summary_ratio}. "
            "Use the sample_ratio encoded in assets/stage_exports instead of regenerating assets."
        )
    if sample_ratio < 1.0:
        df = _sample_dataframe(df, sample_ratio)
        logger.debug(f"Sampled down to {len(df):,} rows ({sample_ratio:.2%} of original)")
    return df.sort_values('dt').reset_index(drop=True)


def _summary_splits(summary: dict):
    test_range = summary.get('test_range') or {}
    if 'start' not in test_range or 'end' not in test_range:
        raise ValueError("Assets summary missing test_range.start/end.")
    return {'test': [int(test_range['start']), int(test_range['end'])]}


def _validate_summary(dataset_key: str, summary: dict):
    if summary is None:
        raise FileNotFoundError(
            f"Missing assets summary for dataset '{dataset_key}': {_summary_path(dataset_key)}"
        )
    canonical_dataset = _canonical_dataset_key(dataset_key)
    if summary.get('dataset') != canonical_dataset:
        raise ValueError(
            f"Assets summary dataset mismatch: expected '{canonical_dataset}', got '{summary.get('dataset')}'."
        )
    stage_entries = summary.get('stages') or []
    if not stage_entries:
        raise ValueError("Assets summary does not contain stage metadata.")
    if 'task' not in summary:
        raise ValueError("Assets summary missing task metadata.")


def _materialize_from_summary(dataset_key: str, summary: dict, sample_ratio: float, logger):
    _validate_summary(dataset_key, summary)
    df = _load_base_frames_from_summary(dataset_key, summary, sample_ratio, logger)

    test_range = summary['test_range']
    test_start = int(test_range['start'])
    test_end = int(test_range['end'])
    test_raw = df[(df['dt'] >= test_start) & (df['dt'] <= test_end)].copy()
    if test_raw.empty:
        raise ValueError("Test data is empty for the range recorded in assets summary.")
    test_prepared = _prepare_subset(test_raw, logger)

    stage_train_sets = []
    stage_test_sets = []
    stage_positive_sets = []

    stage_entries = sorted(summary['stages'], key=lambda entry: entry.get('index', 0))
    for stage_meta in stage_entries:
        feature_set = stage_meta.get('feature_set') or []
        stage_name = stage_meta.get('name') or f"stage_{stage_meta.get('index', 0)}"
        dt_range = stage_meta.get('dt_range')
        if not feature_set:
            raise ValueError(f"Stage '{stage_name}' is missing feature_set in assets summary.")
        if not dt_range or len(dt_range) != 2:
            raise ValueError(f"Stage '{stage_name}' is missing dt_range in assets summary.")

        start_dt, end_dt = map(int, dt_range)
        stage_raw = df[(df['dt'] >= start_dt) & (df['dt'] <= end_dt)].copy()
        if stage_raw.empty:
            raise ValueError(f"Stage '{stage_name}' has no rows in assets-defined range {dt_range}.")

        stage_prepared = _prepare_subset(stage_raw, logger)
        stage_full = _ensure_stage_columns(stage_prepared, feature_set)
        test_full = _ensure_stage_columns(test_prepared.copy(), feature_set)
        train_processed, test_processed = preprocess_data(stage_full, test_full)
        stage_positive = train_processed[train_processed['label'] == 1].copy()

        stage_train_sets.append(train_processed)
        stage_test_sets.append(test_processed)
        stage_positive_sets.append(stage_positive)

    logger.debug(
        f"Materialized {len(stage_train_sets)} stage dataset(s) from raw data using read-only assets summary"
    )
    return stage_train_sets, stage_test_sets, stage_positive_sets


def get_all_stage_splits(dataset, sample_ratio=1.0, logger=None):
    dataset_key = _canonical_dataset_key(dataset)
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError("sample_ratio must be within (0, 1].")
    if logger:
        logger.debug(
            f"Preparing stage splits for '{dataset_key}' from read-only assets summary (sample_ratio={sample_ratio})"
        )
    summary = _read_summary(dataset_key)
    _validate_summary(dataset_key, summary)
    stage_train_sets, stage_test_sets, stage_positive_sets = _materialize_from_summary(
        dataset_key, summary, sample_ratio, logger
    )
    return StageDataContainer(stage_train_sets, stage_test_sets, stage_positive_sets, summary)


def split_dataset(dataset, stage, sample_ratio=1.0, logger=None):
    stage_data = get_all_stage_splits(
        dataset, sample_ratio=sample_ratio, logger=logger
    )
    summary = stage_data.summary
    stage_count = summary['stage_count']
    stage_index = _resolve_stage_identifier(stage, stage_count)
    if logger:
        logger.debug(
            f"Returning {len(stage_data.stage_datasets)} stage dataset(s) with per-stage test splits and positives; stage '{summary['stages'][stage_index - 1]['name']}' highlighted."
        )
    return stage_data


def sort_features(feature_list):
    return sorted(feature_list, key=lambda x: int(re.search(r'feature_(\d+)', x).group(1)))


def evaluate_model(args, dataset, model, stage_data, stage_index, logger, dl_context=None):
    summary = stage_data.summary
    if model != 'EvoCFD':
        raise ValueError(f"Only EvoCFD is available, got: {model}")
    if dl_context is None:
        raise ValueError("EvoCFD context must be prepared before evaluation.")
    dl_args, default_para = dl_context
    stage_name = summary['stages'][stage_index]['name']
    dl_args.stage = stage_name

    set_gpu(dl_args.gpu)
    mkdir(dl_args.save_path)

    set_seeds(dl_args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dl_args.config['training']['n_bins'] = dl_args.n_bins
    return deep_learning(dl_args, stage_data, stage_index, logger)


def main(args, dataset, model, stage, sample_ratio, dl_context=None):
    logger = _init_logger(dataset, stage, model, dl_args = dl_context[0])
    logger.debug(
        f"Run start -> dataset={dataset}, model={model}, stage={stage}, keep_ratio={keep_ratio}, sample_ratio={sample_ratio}"
    )
    stage_data = split_dataset(
        dataset, stage, sample_ratio=sample_ratio, logger=logger
    )
    summary = stage_data.summary
    logger.debug(
        f"Dataset split ready: {len(stage_data.stage_datasets)} stage chunk(s), {len(stage_data.test_sets)} per-stage test chunk(s), {len(stage_data.stage_positive_sets)} positive-only chunk(s)"
    )
    
    if getattr(args, 'calc_drift', False):
        logger.debug("(Data Drift Analysis)...")
        analyze_data_drift(dataset, stage_data)

    # Select the requested stage only if we actually run downstream evaluation.
    stage_index = _resolve_stage_identifier(stage, summary['stage_count'])
    selected_stage = stage_data.stage_datasets[stage_index - 1]
    selected_test = stage_data.test_sets[stage_index - 1]
    selected_positive = stage_data.stage_positive_sets[stage_index - 1]
    logger.debug(
        f"Stage {summary['stages'][stage_index - 1]['name']} rows={len(selected_stage):,}, features={len(selected_stage.columns)}"
    )
    logger.debug(
        f"Stage-matched test chunk rows={len(selected_test):,}, features={len(selected_test.columns)}"
    )
    logger.debug(
        f"Stage positives rows={len(selected_positive):,} (label==1)"
    )
    
    # breakpoint()

    best_recall = evaluate_model(args, dataset, model, stage_data, stage_index - 1, logger, dl_context=dl_context)
    
    method_stage_root = os.path.join(
        dl_context[0].model_path,
        dl_context[0].model_type,
        f"{dl_context[0].dataset}-{dl_context[0].stage}"
    )
    persist_best = bool(getattr(dl_context[0], 'persist_best_or_last', True))
    src_ckpt_name = f"best-val-{dl_context[0].seed}.pth" if persist_best else f"epoch-last-{dl_context[0].seed}.pth"
    best_model_path_from = os.path.join(method_stage_root, dl_context[0].run_tag, src_ckpt_name)
    if os.path.exists(best_model_path_from):
        method_eval_root = os.path.join(dl_context[0].model_path, 'evaluation_best', dl_context[0].model_type)
        best_model_stage_dir = os.path.join(method_eval_root, f"{dl_context[0].dataset}-{dl_context[0].stage}")
        best_model_filename = f"best-val-{dl_context[0].seed}-recall{best_recall * 100:.2f}.pth"
        best_model_path_to = os.path.join(best_model_stage_dir, best_model_filename)

        mkdir(best_model_stage_dir)
        should_copy = True
        if os.path.exists(best_model_path_to):
            try:
                dst_payload = torch.load(best_model_path_to, map_location='cpu')
                src_payload = torch.load(best_model_path_from, map_location='cpu')
                dst_has_extra_state = isinstance(dst_payload, dict) and any(
                    key in dst_payload for key in ('continual_state', 'extra_state', 'metadata')
                )
                src_has_extra_state = isinstance(src_payload, dict) and any(
                    key in src_payload for key in ('continual_state', 'extra_state', 'metadata')
                )
                if dst_has_extra_state and not src_has_extra_state:
                    should_copy = False
            except Exception:
                should_copy = True

        if should_copy:
            shutil.copy(best_model_path_from, best_model_path_to)

        logger.debug(
            f"{'Best' if persist_best else 'Last'} model saved to {best_model_path_to}"
            if should_copy else
            f"Retained richer evaluation checkpoint at {best_model_path_to}"
        )
    else:
        logger.debug(f"Model file '{src_ckpt_name}' not found at {method_stage_root}. Skipping model save.")
    if dl_context is not None:
        _update_evaluation_summary(dl_context[0].model_path, dl_context[0].model_type)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset Name")
    parser.add_argument('--model', type=str, required=True, choices=deep_learning_models,
                        help="Model Name. Only EvoCFD is available.")
    parser.add_argument('--stage', type=str, required=True,
                        help="Training Stage")
    parser.add_argument('--precision', type=str, required=False,
                        help="in [float, double], the precision of parameters")
    parser.add_argument('--keep_ratio', type=float, required=False,
                        help="Ratio")
    parser.add_argument('--apply_max_sv_clip', action='store_true')
    parser.add_argument('--clipping_ratio_alpha', type=float, default=0.01)
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help="Fraction of total rows to use (0<ratio<=1).")
    parser.add_argument('--exp', type=str, default=None,
                        help="Override experiment tag for deep models")
    parser.add_argument('--num_eigen', type=int, default=None,
                        help="Override number of eigen components for EvoCFD runs")
    parser.add_argument('--max_epoch', type=int, default=None,
                        help="Override deep training max epochs")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Override deep training batch size")
    parser.add_argument('--batch_size_prev', type=int, default=None,
                        help="Override prior stage batch size for warm-start logic")
    parser.add_argument('--normalization', type=str, default=None,
                        help="Override feature normalization strategy for deep models")
    parser.add_argument('--num_nan_policy', type=str, default=None,
                        help="Override numeric NaN imputation policy for deep models")
    parser.add_argument('--cat_nan_policy', type=str, default=None,
                        help="Override categorical NaN policy for deep models")
    parser.add_argument('--cat_policy', type=str, default=None,
                        help="Override categorical encoding policy for deep models")
    parser.add_argument('--num_policy', type=str, default=None,
                        help="Override numeric feature encoding policy for deep models")
    parser.add_argument('--n_bins', type=int, default=None,
                        help="Override discretization bins for histogram-based layers")
    parser.add_argument('--cat_min_frequency', type=float, default=None,
                        help="Override minimum category frequency for grouping")
    parser.add_argument('--n_trials', type=int, default=None,
                        help="Override number of hyperparameter trials for tuning")
    parser.add_argument('--seed_num', type=int, default=None,
                        help="Override number of random seeds for deep evaluation")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override the random seed for deep evaluation")
    parser.add_argument('--workers', type=int, default=None,
                        help="Override data loader worker count for deep training")
    parser.add_argument('--gpu', type=str, default=None,
                        help="Override GPU id list (comma-separated) for deep training")
    parser.add_argument('--evaluate_option', type=str, default=None,
                        help="Override evaluation selection strategy (e.g., best-val)")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Override deep model checkpoint root directory")
    parser.add_argument('--lr', type=float, default=None,
                        help='Override main learning rate for deep models.')
    parser.add_argument('--head_lr', type=float, default=None,
                        help='Override classifier head learning rate for deep models.')
    parser.add_argument('--bn_lr', type=float, default=None,
                        help='Override normalization layer learning rate for deep models.')
    parser.add_argument('--svd_lr', type=float, default=None,
                        help='Override SVD parameter-group learning rate for deep models.')
    parser.add_argument('--tokenizer_lr', type=float, default=None,
                        help='Override tokenizer learning rate for deep models.')
    parser.add_argument('--time_lr', type=float, default=None,
                        help='Override temporal embedding and time-tokenizer learning rate for deep models.')
    parser.add_argument('--eta_min', type=float, default=None,
                        help='Override cosine scheduler minimum learning rate for deep models.')
    parser.add_argument('--token_level_NSP', action='store_true',
                        help="Whether to use token-level Null Space Projection")
    parser.add_argument('--freeze_shared_tokenizer', action='store_true',
                        help='Freeze tokenizer parameters for features shared with earlier stages (deep models).')
    parser.add_argument('--only_use_shared_features', action='store_true',
                        help='Use only the shared-with-previous-stage feature tokenizer; exclude new features for the current stage (deep models).')
    parser.add_argument('--calc_drift', action='store_true',
                        help="feature shift analysis between stages")
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--evaluate_model_path', type=str, default=None)
    parser.add_argument(
        '--persist_best_or_last',
        type=int,
        default=1,
        choices=[0, 1],
        help='1: persist best checkpoint to evaluation_best; 0: persist last-epoch checkpoint to evaluation_best'
    )
    
    args = parser.parse_args()
    dataset = _canonical_dataset_key(args.dataset)
    model = args.model
    stage = args.stage
    keep_ratio = args.keep_ratio
    sample_ratio = args.sample_ratio
    dl_context = None
    if model in deep_learning_models:
        with open('model/dlmodel/configs/deep_configs.json', 'r') as file:
            deep_defaults = json.load(file)

        deep_args_payload = {
            'dataset': dataset,
            'stage': stage,
            'exp': args.exp if args.exp is not None else deep_defaults.get('exp', ""),
            'num_eigen': getattr(args, 'num_eigen', None) or deep_defaults.get('num_eigen', 100),
            'model_type': model,
            'max_epoch': getattr(args, 'max_epoch', None) or deep_defaults.get('max_epoch', 100),
            'batch_size': getattr(args, 'batch_size', None) or deep_defaults.get('batch_size', 1024),
            'batch_size_prev': getattr(args, 'batch_size_prev', None) or deep_defaults.get('batch_size_prev', deep_defaults.get('batch_size', 1024)),
            'normalization': getattr(args, 'normalization', None) or deep_defaults.get('normalization', 'standard'),
            'num_nan_policy': getattr(args, 'num_nan_policy', None) or deep_defaults.get('num_nan_policy', 'mean'),
            'cat_nan_policy': getattr(args, 'cat_nan_policy', None) or deep_defaults.get('cat_nan_policy', 'new'),
            'cat_policy': getattr(args, 'cat_policy', None) or deep_defaults.get('cat_policy', 'ordinal'),
            'num_policy': getattr(args, 'num_policy', None) or deep_defaults.get('num_policy', 'none'),
            'n_bins': getattr(args, 'n_bins', None) or deep_defaults.get('n_bins', 2),
            'cat_min_frequency': getattr(args, 'cat_min_frequency', None) or deep_defaults.get('cat_min_frequency', 0.0),
            'n_trials': getattr(args, 'n_trials', None) or deep_defaults.get('n_trials', 100),
            'seed_num': getattr(args, 'seed_num', None) or deep_defaults.get('seed_num', 3),
            'seed': getattr(args, 'seed', None) or deep_defaults.get('seed', 0),
            'workers': getattr(args, 'workers', None) or deep_defaults.get('workers', 0),
            'gpu': getattr(args, 'gpu', None) or deep_defaults.get('gpu', '0'),
            'evaluate_option': getattr(args, 'evaluate_option', None) or deep_defaults.get('evaluate_option', 'best-val'),
            'model_path': getattr(args, 'model_path', None) or deep_defaults.get('model_path', 'results_model'),
            'apply_max_sv_clip': args.apply_max_sv_clip,
            'clipping_ratio_alpha': args.clipping_ratio_alpha,
            'keep_ratio': args.keep_ratio,
            'precision': getattr(args, 'precision', None) or deep_defaults.get('precision', 'float'),
            'token_level_NSP': getattr(args, 'token_level_NSP', False),
            'freeze_shared_tokenizer': getattr(args, 'freeze_shared_tokenizer', False),
            'only_use_shared_features': getattr(args, 'only_use_shared_features', False),
            'persist_best_or_last': bool(getattr(args, 'persist_best_or_last', 1)),
            'train': getattr(args, "evaluation", False) == False,
            'evaluate_model_path': getattr(args, "evaluate_model_path", None),
        }
        
        dl_args = argparse.Namespace(**deep_args_payload)
        logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_tag = f"{dl_args.exp}/{logtime}-gpu{dl_args.gpu}" if dl_args.exp else f"{logtime}-gpu{dl_args.gpu}"
        dl_args.run_tag = run_tag
        dl_args.save_path = os.path.join(
            dl_args.model_path,
            dl_args.model_type,
            f"{dl_args.dataset}-{dl_args.stage}",
            dl_args.run_tag
        )
        
        config_default_path = os.path.join('model/dlmodel/configs/default_param.json')
        
        with open(config_default_path, 'r') as file:
            default_para = json.load(file)
        
        dl_args.config = default_para[dl_args.model_type]
        dl_args.temporal_dt_mode = _resolve_temporal_dt_mode(dl_args.dataset)
        temporal_cfg = dl_args.config.setdefault('model', {}).setdefault('temporal_embeddings', {})
        resolved_periods = _resolve_temporal_embedding_periods(dl_args.dataset)
        if resolved_periods is not None:
            temporal_cfg['periods'] = resolved_periods
        dl_args.temporal_period_labels = _resolve_temporal_period_labels(dl_args.dataset)
        training_cfg = dl_args.config.setdefault('training', {})
        if args.lr is not None:
            training_cfg['lr'] = float(args.lr)
        if args.head_lr is not None:
            training_cfg['head_lr'] = float(args.head_lr)
        if args.bn_lr is not None:
            training_cfg['bn_lr'] = float(args.bn_lr)
        if args.svd_lr is not None:
            training_cfg['svd_lr'] = float(args.svd_lr)
        if args.tokenizer_lr is not None:
            training_cfg['tokenizer_lr'] = float(args.tokenizer_lr)
        if args.time_lr is not None:
            training_cfg['time_lr'] = float(args.time_lr)
        if args.eta_min is not None:
            training_cfg['eta_min'] = float(args.eta_min)
        dl_context = (dl_args, default_para)

    main(args, dataset, model, stage, sample_ratio, dl_context=dl_context)
