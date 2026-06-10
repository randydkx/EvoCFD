import typing as ty
from copy import deepcopy
from pathlib import Path
import os
import re
import numpy as np
import sklearn.preprocessing
import torch
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import category_encoders
import datetime
# from model.stage_dataclass import dataclass as StageDataContainer


ArrayDict = ty.Dict[str, np.ndarray]

def raise_unknown(unknown_what: str, unknown_value: ty.Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def data_nan_process(
    N_data,
    C_data,
    num_nan_policy,
    cat_nan_policy,
    num_new_value=None,
    imputer=None,
    cat_new_value=None,
    dtype=np.float32,
):
    """
    Process the NaN values in the dataset.

    :param N_data: np.ndarray (Numerical features)
    :param C_data: np.ndarray (Categorical features)
    :param num_nan_policy: str ('mean', 'median')
    :param cat_nan_policy: str ('new', 'most_frequent')
    :param num_new_value: Optional[np.ndarray] - pre-calculated mean/median
    :param imputer: Optional[SimpleImputer] - pre-fitted imputer
    :param cat_new_value: Optional[str] - value for 'new' policy
    :return: Tuple
    """
    
    # --- Numerical Data Processing ---
    target_dtype = np.dtype(dtype)

    if N_data is None:
        N = None
    else:
        N = deepcopy(N_data)
        
        if N.ndim == 1:
            N = N.reshape(-1, 1)
        
        N = N.astype(target_dtype)
        
        num_nan_masks = np.isnan(N)
        
        if num_nan_masks.any():
            if num_new_value is None:
                if num_nan_policy == 'mean':
                    num_new_value = np.nanmean(N, axis=0)
                elif num_nan_policy == 'median':
                    num_new_value = np.nanmedian(N, axis=0)
                else:
                    raise ValueError(f"Unknown numerical NaN policy: {num_nan_policy}")
            
            num_nan_indices = np.where(num_nan_masks)
            N[num_nan_indices] = np.take(num_new_value, num_nan_indices[1])
            
    # --- Categorical Data Processing ---
    if C_data is None:
        C = None
    else:
        
        C = deepcopy(C_data)
        
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        
        C = C.astype(str)
            
        missing_identifiers = ['nan', 'NaN', '', 'None', 'nan']
        cat_nan_masks = np.isin(C, missing_identifiers)
        

        if cat_nan_masks.any():  
            if cat_nan_policy == 'new':
                if cat_new_value is None:
                    cat_new_value = '___null___'
                    imputer = None
                
                C[cat_nan_masks] = cat_new_value

            elif cat_nan_policy == 'most_frequent':
                
                
                C_obj = C.astype(object)
                C_obj[cat_nan_masks] = np.nan 
                
                if imputer is None:
                    cat_new_value = None
                    imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
                    imputer.fit(C_obj)
                
                C = imputer.transform(C_obj).astype(str)
                
            else:
                raise ValueError(f"Unknown categorical NaN policy: {cat_nan_policy}")
        
    result = (N, C, num_new_value, imputer, cat_new_value)
    return result

import torch
import numpy as np

def num_enc_process(N_data, num_policy, n_bins=2, y_train=None, encoder=None):
    """
    Process the numerical features in the dataset.
    
    :param N_data: np.ndarray (Shape: [n_samples, n_features])
    :param num_policy: str
    :param n_bins: int
    :param y_train: Optional[np.ndarray]
    :param encoder: Optional[EncodingClass]
    :return: Tuple[np.ndarray, Optional[EncodingClass]]
    """
    from ..lib.num_embeddings import compute_bins, PiecewiseLinearEncoding, UnaryEncoding, JohnsonEncoding, BinsEncoding

    if N_data is not None:
        if num_policy == 'none':
            return N_data, None

        N_tensor = torch.from_numpy(N_data)
        
        y_tensor = torch.from_numpy(y_train) if y_train is not None else None

        policy_map = {
            'Q_PLE': (PiecewiseLinearEncoding, False),
            'T_PLE': (PiecewiseLinearEncoding, True),
            'Q_Unary': (UnaryEncoding, False),
            'T_Unary': (UnaryEncoding, True),
            'Q_bins': (BinsEncoding, False),
            'T_bins': (BinsEncoding, True),
            'Q_Johnson': (JohnsonEncoding, False),
            'T_Johnson': (JohnsonEncoding, True),
        }

        if num_policy in policy_map:
            EncodingClass, use_tree = policy_map[num_policy]

            if encoder is None:
                if use_tree:
                    tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                    temp_data_dict = {'x': N_tensor} 
                    
                    bins_dict = compute_bins(
                        temp_data_dict, 
                        n_bins=n_bins, 
                        tree_kwargs=tree_kwargs, 
                        y=y_tensor, 
                        regression=False
                    )
                    bins = bins_dict['x']
                else:
                    temp_data_dict = {'x': N_tensor}
                    bins_dict = compute_bins(
                        temp_data_dict, 
                        n_bins=n_bins, 
                        tree_kwargs=None, 
                        y=None, 
                        regression=None
                    )
                    bins = bins_dict['x']

                encoder = EncodingClass(bins)

            encoded_tensor = encoder(N_tensor)
            
            N_data_processed = encoded_tensor.cpu().numpy()
            
            return N_data_processed, encoder

        else:
            return N_data, None

    else:
        return N_data, None

def cat_enc_process(N_data, C_data, cat_policy, y_train=None, ord_encoder=None, mode_values=None, cat_encoder=None):
    """
    Process the categorical features in the dataset.

    :param N_data: np.ndarray or None
    :param C_data: np.ndarray or None
    :param cat_policy: str
    :param y_train: Optional[np.ndarray]
    :param ord_encoder: Optional[OrdinalEncoder]
    :param mode_values: Optional[List[int]]
    :param cat_encoder: Optional[OneHotEncoder] or other encoders
    :return: Tuple
    """

    if C_data is not None:
        unknown_value = np.iinfo('int64').max - 3
        
        if ord_encoder is None:
            ord_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='use_encoded_value',  
                unknown_value=unknown_value, 
                dtype='int64', 
            ).fit(C_data)
        
        C_encoded = ord_encoder.transform(C_data)

        if mode_values is not None:
            for column_idx in range(C_encoded.shape[1]):
                mask = C_encoded[:, column_idx] == unknown_value
                if np.any(mask):
                    C_encoded[mask, column_idx] = mode_values[column_idx]
        else:
            mode_values = []
            for col_idx in range(C_encoded.shape[1]):
                col = C_encoded[:, col_idx]
                valid_mask = col != unknown_value
                if np.any(valid_mask):
                    vals, counts = np.unique(col[valid_mask], return_counts=True)
                    mode_val = vals[np.argmax(counts)]
                else:
                    mode_val = col[0] # Fallback
                mode_values.append(mode_val)
            
            for column_idx in range(C_encoded.shape[1]):
                mask = C_encoded[:, column_idx] == unknown_value
                if np.any(mask):
                    C_encoded[mask, column_idx] = mode_values[column_idx]

        C_data = C_encoded

        if cat_policy == 'indices':
            return N_data, C_data, ord_encoder, mode_values, cat_encoder
        
        elif cat_policy == 'ordinal':
            cat_encoder = ord_encoder
            
        elif cat_policy == 'ohe':
            if cat_encoder is None:
                cat_encoder = sklearn.preprocessing.OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype='float64'
                )
                cat_encoder.fit(C_data)
            C_data = cat_encoder.transform(C_data)
            
        elif cat_policy == 'binary':
            if cat_encoder is None:
                cat_encoder = category_encoders.BinaryEncoder()
                cat_encoder.fit(C_data.astype(str))
            C_data = cat_encoder.transform(C_data.astype(str)).values
            
        elif cat_policy == 'hash':
            if cat_encoder is None:
                cat_encoder = category_encoders.HashingEncoder()
                cat_encoder.fit(C_data.astype(str))
            C_data = cat_encoder.transform(C_data.astype(str)).values
            
        elif cat_policy == 'loo':
            if cat_encoder is None:
                cat_encoder = category_encoders.LeaveOneOutEncoder()
                cat_encoder.fit(C_data.astype(str), y_train)
            C_data = cat_encoder.transform(C_data.astype(str)).values
            
        elif cat_policy == 'target':
            if cat_encoder is None:
                cat_encoder = category_encoders.TargetEncoder()
                cat_encoder.fit(C_data.astype(str), y_train)
            C_data = cat_encoder.transform(C_data.astype(str)).values
            
        elif cat_policy == 'catboost':
            if cat_encoder is None:
                cat_encoder = category_encoders.CatBoostEncoder()
                cat_encoder.fit(C_data.astype(str), y_train)
            C_data = cat_encoder.transform(C_data.astype(str)).values
            
        elif cat_policy == 'tabr_ohe':
            if cat_encoder is None:
                cat_encoder = sklearn.preprocessing.OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype='float64'
                )
                cat_encoder.fit(C_data)
            C_data_ohe = cat_encoder.transform(C_data)
            return N_data, C_data_ohe, ord_encoder, mode_values, cat_encoder
            
        else:
            raise ValueError(f"Unknown categorical encoding policy: {cat_policy}")

        if N_data is None:
            final_data = C_data
            final_C = None
        else:
            final_data = np.hstack((N_data, C_data))
            final_C = None 

        return final_data, final_C, ord_encoder, mode_values, cat_encoder

    else:
        return N_data, C_data, None, None, None

def data_norm_process(N_data, normalization, seed, normalizer=None):
    """
    Process the normalization of the dataset.

    :param N_data: np.ndarray
    :param normalization: str
    :param seed: int
    :param normalizer: Optional[TransformerMixin]
    :return: Tuple[np.ndarray, Optional[TransformerMixin]]
    """
    if N_data is None or normalization == 'none':
        return N_data, None

    if N_data.ndim == 1:
        N_data = N_data.reshape(-1, 1)

    if normalizer is None:
        if normalization == 'standard':
            normalizer = sklearn.preprocessing.StandardScaler()
        elif normalization == 'minmax':
            normalizer = sklearn.preprocessing.MinMaxScaler()
        elif normalization == 'quantile':
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(N_data.shape[0] // 30, 1000), 10),
                random_state=seed
            )
        elif normalization == 'maxabs':
            normalizer = sklearn.preprocessing.MaxAbsScaler()
        elif normalization == 'power':
            normalizer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson')
        elif normalization == 'robust':
            normalizer = sklearn.preprocessing.RobustScaler()
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        # Match the old repo bit-for-bit: fitting on a copy changes the final
        # reduction order enough to affect StandardScaler statistics at 1e-11 scale.
        normalizer.fit(N_data.copy())

    N_data_processed = normalizer.transform(N_data)
    
    return N_data_processed, normalizer

def data_label_process(y_data, encoder=None):
    """
    Process the labels in the dataset.

    :param y_data: np.ndarray
    :param info: Optional[Dict[str, Any]]
    :param encoder: Optional[LabelEncoder]
    :return: Tuple[np.ndarray, Dict[str, Any], Optional[LabelEncoder]]
    """
    if y_data is None:
        return None, None

    y = deepcopy(y_data)

    if y.ndim > 1:
        y = y.ravel()

    if encoder is None:
        encoder = sklearn.preprocessing.LabelEncoder().fit(y)
    
    y_transformed = encoder.transform(y)
    
    return y_transformed, encoder

def data_dt_process(dt, mean=None, std=None):
    """
    Process datetime data.
    
    :param dt: np.ndarray (Either calendar dates like YYYYMMDD or elapsed seconds like IEEE)
    :param mean: float (Mean for normalization, calculate if None)
    :param std: float (Std for normalization, calculate if None)
    :return: Tuple[np.ndarray, float, float]
    """
    if dt is None:
        return None, mean, std

    dt_flat = dt.ravel()
    values_float = np.asarray(dt_flat, dtype=float)

    def _looks_like_calendar_date(values: np.ndarray) -> bool:
        if values.size == 0:
            return False
        int_values = values.astype(np.int64)
        min_value = int(int_values.min())
        max_value = int(int_values.max())
        if min_value < 19000101 or max_value > 21001231:
            return False
        for value in int_values[: min(len(int_values), 32)]:
            value_str = str(int(value))
            if len(value_str) != 8:
                return False
            try:
                datetime.date(int(value_str[:4]), int(value_str[4:6]), int(value_str[6:8]))
            except ValueError:
                return False
        return True

    if _looks_like_calendar_date(values_float):
        base = datetime.date(2023, 6, 1)
        values_str = [str(int(v)) for v in values_float]
        dates = [datetime.date(int(v[:4]), int(v[4:6]), int(v[6:8])) for v in values_str]
        trend = np.array([(d - base).days * 86400 for d in dates], dtype=float)
        year = np.array([d.year - 2023 for d in dates], dtype=float)
        month = np.array([d.month / 12 for d in dates], dtype=float)
        day = np.array([d.day / 31 for d in dates], dtype=float)
    else:
        anchor = datetime.datetime(2023, 6, 1)
        trend = values_float.astype(float)
        datetimes = [anchor + datetime.timedelta(seconds=float(v)) for v in trend]
        year = np.array([d.year - anchor.year for d in datetimes], dtype=float)
        month = np.array([d.month / 12 for d in datetimes], dtype=float)
        day = np.array([
            (d.hour * 3600 + d.minute * 60 + d.second) / 86400 for d in datetimes
        ], dtype=float)

    if mean is None:
        t_mean = np.mean(trend)
    else:
        t_mean = mean
        
    if std is None:
        t_std = np.std(trend)
        if t_std == 0: t_std = 1.0
    else:
        t_std = std

    trend_normalized = (trend - t_mean) / t_std

    result = np.stack([year, month, day, trend, trend_normalized], axis=1)
    
    return result, t_mean, t_std

from model.stage_dataclass import OneStageDataset


def data_loader_process(onestage_dataset: OneStageDataset, device, batch_size, is_train):
    
    loss_fn = F.cross_entropy
    loader_kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_shuffle = os.environ.get('EVOCFD_DEBUG_DISABLE_TRAIN_SHUFFLE') != '1'

    if is_train:
        train_loader = DataLoader(dataset=onestage_dataset, shuffle=train_shuffle, **loader_kwargs)
        val_loader = DataLoader(dataset=onestage_dataset, shuffle=False, **loader_kwargs)
        
        return train_loader, val_loader, loss_fn
    else:
        test_loader = DataLoader(dataset=onestage_dataset, shuffle=False, **loader_kwargs)
        
        return test_loader, loss_fn
    

def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    """
    Convert the numpy arrays to torch tensors.

    :param data: ArrayDict
    :return: Dict[str, torch.Tensor]
    """
    return {k: torch.as_tensor(v) for k, v in data.items()}

def get_categories(
    X_cat: ty.Optional[ty.Dict[str, torch.Tensor]]
) -> ty.Optional[ty.List[int]]:
    """
    Get the categories for each categorical feature.

    :param X_cat: Optional[Dict[str, torch.Tensor]]
    :return: Optional[List[int]]
    """
    return (
        None
        if X_cat is None
        else [
            len(set(X_cat['train'][:, i].tolist()))
            for i in range(X_cat['train'].shape[1])
        ]
    )
