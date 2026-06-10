from dataclasses import dataclass as dc_dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typing as ty
import re

ArrayDict = ty.Dict[str, np.ndarray]

from torch.utils.data import Dataset
import torch



@dc_dataclass
class dataclass:
    """Container for staged dataset artifacts and helpers to materialize tensors."""

    stage_datasets: List[pd.DataFrame]
    test_sets: List[Optional[pd.DataFrame]]
    stage_positive_sets: List[pd.DataFrame]
    summary: Dict
    
    def _build_component(
        self,
        stage_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Optional[Dict[str, np.ndarray]]:
        if not feature_cols:
            return None
        
        return stage_data[feature_cols].to_numpy(copy=True)

    def _build_stage_bundle(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        stage_meta: Dict = None,
        _type = 'train'
    ):
        if _type == 'train':
            stage_data = train_df
        elif _type == "test":
            stage_data = test_df
        elif _type == "val":
            stage_data = train_df
        
        numeric_cols = stage_meta.get('numeric_features', []) or []
        categorical_cols = stage_meta.get('categorical_features', []) or []
        N = self._build_component(stage_data, numeric_cols)
        C = self._build_component(stage_data, categorical_cols)
        dt = stage_data['dt'].to_numpy(copy=True)
        if dt is not None:
            dt = dt.reshape(-1, 1)
        y = stage_data['label'].to_numpy(copy=True)
    
        
        return OneStageDataset(N, C, dt, y, stage_meta)

    def obtain_stage_data(self, stage_index: int):
        stages = self.summary.get('stages', []) or []
        if stage_index < 0 or stage_index >= len(stages):
            return None, None, None, None
        stage_meta = stages[stage_index]
        
        train_val_data = self._build_stage_bundle(train_df = self.stage_datasets[stage_index], stage_meta = stage_meta, _type="train")
        
        test_data = self._build_stage_bundle(test_df = self.test_sets[stage_index], stage_meta = stage_meta, _type="test")
        
        train_val_data_pos = self._build_stage_bundle(train_df = self.stage_positive_sets[stage_index], stage_meta = stage_meta, _type="train")
        
        return train_val_data, test_data, train_val_data_pos, stage_meta

    def previous_stage_name(self, stag_index: int) -> Optional[str]:
        
        return self.stage_name(stag_index - 1)
    
    def stage_count(self) -> int:
        return len(self.summary.get('stages', []) or [])

    def stage_name(self, stage_index: int) -> Optional[str]:
        stages = self.summary.get('stages', []) or []
        if 0 <= stage_index < len(stages):
            return stages[stage_index].get('name')
        return None
    
    def overall_feature_names(self) -> List[str]:
        feature_names = set()
        stages = self.summary.get('stages', []) or []
        for stage_meta in stages:
            numeric_cols = stage_meta.get('numeric_features', []) or []
            categorical_cols = stage_meta.get('categorical_features', []) or []
            feature_names.update(numeric_cols)
            feature_names.update(categorical_cols)
        
        def _feature_sort_key(name: str):
            match = re.search(r'feature_(\d+)', name)
            if match:
                return (0, int(match.group(1)))
            return (1, name)

        return sorted(feature_names, key=_feature_sort_key)
    
    def d_numeric(self):
        stages = self.summary.get('stages', []) or []
        unique_cols = set()
        
        for stage_meta in stages:
            numeric_cols = stage_meta.get('numeric_features', []) or []
            unique_cols.update(numeric_cols)
            
        return len(unique_cols)
    
    def d_categorical(self):
        stages = self.summary.get('stages', []) or []
        unique_cols = set()
        
        for stage_meta in stages:
            categorical_cols = stage_meta.get('categorical_features', []) or []
            unique_cols.update(categorical_cols)
            
        return len(unique_cols)

    def num_feature_names_of_stage(self, stage_index: int) -> List[str]:
        
        stages = self.summary.get('stages', []) or []
        
        if 0 <= stage_index < len(stages):
            stage_meta = stages[stage_index]
            numeric_cols = stage_meta.get('numeric_features', []) or []
            feature_names = numeric_cols
            return feature_names
        return []
    
    def cat_feature_names_of_stage(self, stage_index: int) -> List[str]:
        
        stages = self.summary.get('stages', []) or []
        
        if 0 <= stage_index < len(stages):
            stage_meta = stages[stage_index]
            categorical_cols = stage_meta.get('categorical_features', []) or []
            feature_names = categorical_cols
            return feature_names
        return []




class OneStageDataset(Dataset):
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    dt: ArrayDict
    y: ArrayDict
    info: ty.Dict[str, ty.Any]
    
    def __init__(self, N, C, dt, y, info):
        self.N = N
        self.C = C
        self.dt = dt
        self.y = y
        self.info = info
        
        self.tensorization()
    
    def tensorization(self):
        self.N = torch.as_tensor(self.N) if self.N is not None else None
        self.C = torch.as_tensor(self.C) if self.C is not None else None
        self.dt = torch.as_tensor(self.dt) if self.dt is not None else None
        self.y = torch.as_tensor(self.y)
        
    def to_device(self, device):
        self.N, self.C, self.dt, self.y = (
            (None if x is None else x.to(device)) for x in (self.N, self.C, self.dt, self.y)
        )

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_cat_features(self) -> int:
        return self.info['n_cat_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: str) -> int:
        """
        Return the size of the dataset partition.

        Args:

        - part: str

        Returns: int
        """
        X = self.N if self.N is not None else self.C
        assert(X is not None)
        return len(X[part])
    
    def get_dim_in(self):
        return self.n_num_features + self.n_cat_features

    def get_categories(self):
        return (
            None
            if self.C is None
            else [
                len(set(self.C[:, i].cpu().tolist()))
                for i in range(self.C.shape[1])
            ]
        )

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        if self.N is not None and self.C is not None:
            data = (self.N[i], self.C[i])
        elif self.C is not None and self.N is None:
            data, label = self.C[i], self.y[i]
        else:
            data, label = self.N[i], self.y[i]
        label = self.y[i]
        if self.dt is not None:
            dt = self.dt[i]
            return data, dt, label
        else:
            return data, label
        