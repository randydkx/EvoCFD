import abc
import random
import torch
import numpy as np
import time
import os.path as osp
from tqdm import tqdm
import sklearn.metrics as skm
import re
import os
import shutil
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.stage_dataclass import dataclass as StageDataContainer
from model.dlmodel.model.models.FTT import Transformer
from copy import deepcopy

from model.dlmodel.model.utils import (
    Timer,
    Averager,
    set_seeds,
    get_device
)

from model.stage_dataclass import OneStageDataset

from ..lib.data import (
    data_nan_process,
    cat_enc_process,
    num_enc_process,
    data_norm_process,
    data_label_process,
    data_dt_process,
    data_loader_process,
    get_categories
)

def check_softmax(logits):
    """
    Check if the logits are already probabilities, and if not, convert them to probabilities.
    :param logits: np.ndarray of shape (N, C) with logits
    :return: np.ndarray of shape (N, C) with probabilities
    """
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits


def _to_numpy_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)



class Method(object, metaclass=abc.ABCMeta):
    def __init__(self, args, stage_data):
        """
        :param args: argparse object
        """
        self.args = args
        precision_value = str(getattr(self.args, 'precision', 'float') or 'float').strip().lower()
        if precision_value not in {'float', 'double'}:
            raise ValueError(f"Unsupported precision '{precision_value}'. Expected 'float' or 'double'.")
        self.args.precision = precision_value
        self._precision_mode = precision_value
        self._torch_precision_dtype = torch.float64 if precision_value == 'double' else torch.float32
        self._numpy_precision_dtype = np.float64 if precision_value == 'double' else np.float32
        self.stage_data: StageDataContainer = stage_data
        self.D = None
        self.train_step = 0
        self.val_count = 0
        self.continue_training = True
        self.fea_in = defaultdict(dict)
        
        self.model = None
        self.shared_numeric_features: List[str] = []
        
        self.timer = Timer()
        self.trlog = {}
        self.trlog['args'] = vars(args)
        self.trlog['train_loss'] = []
        self.trlog['best_epoch'] = 0
        self.trlog['best_res'] = 1.0
        self.trlog['val_recall'] = []
        self.requested_gpu_ids = self._parse_requested_gpu_ids(getattr(self.args, 'gpu', None))
        self.args.device = self._resolve_requested_device()
        self.config = self.args.config

        self.stage_index = None
        
        self.prev_model = None
        self.dataset = getattr(self.args, 'dataset', None)
        self.stage = getattr(self.args, 'stage', None)
        self.recall_history: List[float] = []
        # Control whether to persist the best model or the last epoch model
        # True -> persist best model (default). False -> persist last epoch model.
        if not hasattr(self.args, 'persist_best_or_last'):
            setattr(self.args, 'persist_best_or_last', True)

    def _parse_requested_gpu_ids(self, gpu_value) -> List[int]:
        if gpu_value is None:
            return []

        gpu_ids = []
        for token in str(gpu_value).split(','):
            token = token.strip()
            if not token:
                continue
            try:
                gpu_ids.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid GPU id '{token}' in gpu setting '{gpu_value}'.") from exc
        return gpu_ids

    def _resolve_requested_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device('cpu')

        if not self.requested_gpu_ids:
            return get_device()

        device_count = torch.cuda.device_count()
        invalid_gpu_ids = [gpu_id for gpu_id in self.requested_gpu_ids if gpu_id < 0 or gpu_id >= device_count]
        if invalid_gpu_ids:
            if len(self.requested_gpu_ids) == device_count:
                self.requested_gpu_ids = list(range(device_count))
            else:
                raise ValueError(
                    f"Requested GPU ids {invalid_gpu_ids} are unavailable; visible device count is {device_count}."
                )

        return torch.device(f'cuda:{self.requested_gpu_ids[0]}')

    def _wrap_model_for_requested_gpus(self):
        if self.model is None or not torch.cuda.is_available():
            return
        if len(self.requested_gpu_ids) <= 1:
            return

        logger = getattr(self, 'logger', None)
        if logger is not None:
            logger.debug(f"Using requested GPU ids {self.requested_gpu_ids}")

        self.model = torch.nn.DataParallel(
            self.model,
            device_ids=self.requested_gpu_ids,
            output_device=self.requested_gpu_ids[0],
        )

    def reset_stats_withconfig(self, config):
        """
        Reset the training statistics with a new configuration.
        :param config: dict, new configuration
        """
        set_seeds(self.args.seed)
        self.train_step = 0
        self.val_count = 0
        self.continue_training = True
        self.timer = Timer()
        # train statistics
        self.trlog = {}
        self.trlog['args'] = vars(self.args)
        self.trlog['train_loss'] = []
        self.trlog['best_epoch'] = 0
        self.trlog['best_res'] = 1.0
        self.trlog['val_recall'] = []
        self.recall_history = []
    
    def _to_numpy(self, arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr
            if torch.is_tensor(arr):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

    def _torch_dtype(self) -> torch.dtype:
        return self._torch_precision_dtype

    def _numpy_dtype(self):
        return self._numpy_precision_dtype

    def _cast_float_array(self, array):
        if array is None:
            return None
        np_array = np.asarray(array)
        if not np.issubdtype(np_array.dtype, np.floating):
            return np_array
        target_dtype = self._numpy_dtype()
        if np_array.dtype == target_dtype:
            return np_array
        return np_array.astype(target_dtype, copy=False)

    def _apply_precision_to_model(self):
        dtype = self._torch_dtype()

        def _move(module: Optional[torch.nn.Module]):
            if module is None:
                return None
            module.to(device=self.args.device, dtype=dtype)
            return module

        self.model = _move(self.model)
        self.prev_model = _move(self.prev_model)

    def _cast_float_tensor(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor):
            return tensor
        if not torch.is_floating_point(tensor):
            return tensor
        target_dtype = self._torch_dtype()
        if tensor.dtype == target_dtype:
            return tensor
        return tensor.to(dtype=target_dtype)

    def _apply_back(self, dataset, N, C, dt, y):
        if dataset is None:
            return None
        dataset.N = N
        dataset.C = C
        dataset.dt = dt
        dataset.y = y
        dataset.tensorization()
        dataset.N = self._cast_float_tensor(dataset.N)
        dataset.C = self._cast_float_tensor(dataset.C)
        dataset.dt = self._cast_float_tensor(dataset.dt)
        
        return dataset
    
    def data_preprocessing_train_and_test(
        self,
        train_dataset: StageDataContainer = None,
        test_dataset: StageDataContainer = None,
        batch_size: Optional[int] = None,
    ):
        """
        Preprocess the training and testing data.
        :param train_dataset: StageDataContainer, training dataset
        :param test_dataset: StageDataContainer, testing dataset
        """
        loaders = {
            'train_loader': None,
            'val_loader': None,
            'test_loader': None,
            'criterion': None
        }

        if batch_size is None:
            batch_size = self.args.batch_size

        processing_state = {}

        if train_dataset is not None:
            N = self._to_numpy(train_dataset.N)
            C = self._to_numpy(train_dataset.C)
            dt = self._to_numpy(train_dataset.dt)
            y = self._to_numpy(train_dataset.y)

            N, C, num_new_value, imputer, cat_new_value = data_nan_process(
                N,
                C,
                self.args.num_nan_policy,
                self.args.cat_nan_policy,
                dtype=self._numpy_dtype()
            )
            y, label_encoder = data_label_process(y)
            dt, dt_mean, dt_std = data_dt_process(dt)
            dt = self._cast_float_array(dt)
            N, num_encoder = num_enc_process(
                N,
                num_policy=self.args.num_policy,
                n_bins=self.args.config['training']['n_bins'],
                y_train=y
            )
            N = self._cast_float_array(N)
            N, C, ord_encoder, mode_values, cat_encoder = cat_enc_process(
                N, C, self.args.cat_policy, y
            )
            N = self._cast_float_array(N)
            C = self._cast_float_array(C)
            N, normalizer = data_norm_process(
                N, self.args.normalization, self.args.seed
            )
            N = self._cast_float_array(N)

            processing_state = {
                'num_new_value': num_new_value,
                'imputer': imputer,
                'cat_new_value': cat_new_value,
                'label_encoder': label_encoder,
                'dt_mean': dt_mean,
                'dt_std': dt_std,
                'num_encoder': num_encoder,
                'ord_encoder': ord_encoder,
                'mode_values': mode_values,
                'cat_encoder': cat_encoder,
                'normalizer': normalizer
            }

            self.N, self.C, self.dt, self.y = N, C, dt, y
            self.num_new_value = num_new_value
            self.imputer = imputer
            self.cat_new_value = cat_new_value
            self.label_encoder = label_encoder
            self.dt_mean = dt_mean
            self.dt_std = dt_std
            self.num_encoder = num_encoder
            self.ord_encoder = ord_encoder
            self.mode_values = mode_values
            self.cat_encoder = cat_encoder
            self.normalizer = normalizer

            self._apply_back(train_dataset, N, C, dt, y)

            loaders['train_loader'], loaders['val_loader'], loaders['criterion'] = data_loader_process(
                train_dataset,
                self.args.device,
                batch_size,
                is_train=True
            )

        if test_dataset is not None:
            if not processing_state:
                raise ValueError('Processing test data requires a fitted training dataset.')

            N_test = self._to_numpy(test_dataset.N)
            C_test = self._to_numpy(test_dataset.C)
            dt_test = self._to_numpy(test_dataset.dt)
            y_test = self._to_numpy(test_dataset.y)

            N_test, C_test, _, _, _ = data_nan_process(
                N_test,
                C_test,
                self.args.num_nan_policy,
                self.args.cat_nan_policy,
                num_new_value=processing_state['num_new_value'],
                imputer=processing_state['imputer'],
                cat_new_value=processing_state['cat_new_value'],
                dtype=self._numpy_dtype()
            )
            y_test, _ = data_label_process(
                y_test, encoder=processing_state['label_encoder']
            )
            dt_test, _, _ = data_dt_process(
                dt_test,
                processing_state['dt_mean'],
                processing_state['dt_std']
            )
            dt_test = self._cast_float_array(dt_test)
            N_test, _ = num_enc_process(
                N_test,
                num_policy=self.args.num_policy,
                n_bins=self.args.config['training']['n_bins'],
                y_train=y_test,
                encoder=processing_state['num_encoder']
            )
            N_test = self._cast_float_array(N_test)
            N_test, C_test, _, _, _ = cat_enc_process(
                N_test,
                C_test,
                self.args.cat_policy,
                y_test,
                ord_encoder=processing_state['ord_encoder'],
                mode_values=processing_state['mode_values'],
                cat_encoder=processing_state['cat_encoder']
            )
            N_test = self._cast_float_array(N_test)
            C_test = self._cast_float_array(C_test)
            N_test, _ = data_norm_process(
                N_test,
                self.args.normalization,
                self.args.seed,
                normalizer=processing_state['normalizer']
            )
            N_test = self._cast_float_array(N_test)

            self.N_test = N_test
            self._apply_back(test_dataset, N_test, C_test, dt_test, y_test)

            loaders['test_loader'], _ = data_loader_process(
                test_dataset,
                self.args.device,
                batch_size,
                is_train=False
            )

        return loaders

    def apply_current_process_to_dataset(self, to_dataset: StageDataContainer):
        if not hasattr(self, 'imputer') or not hasattr(self, 'normalizer'):
            raise ValueError('No processing state found. Please fit on training data first.')

        N = self._to_numpy(to_dataset.N)
        C = self._to_numpy(to_dataset.C)
        dt = self._to_numpy(to_dataset.dt)
        y = self._to_numpy(to_dataset.y)

        N, C, _, _, _ = data_nan_process(
            N,
            C,
            self.args.num_nan_policy,
            self.args.cat_nan_policy,
            num_new_value=self.num_new_value,
            imputer=self.imputer,
            cat_new_value=self.cat_new_value,
            dtype=self._numpy_dtype()
        )
        y, _ = data_label_process(y, encoder=self.label_encoder)
        dt, _, _ = data_dt_process(dt, self.dt_mean, self.dt_std)
        dt = self._cast_float_array(dt)
        N, _ = num_enc_process(
            N,
            num_policy=self.args.num_policy,
            n_bins=self.args.config['training']['n_bins'],
            y_train=y,
            encoder=self.num_encoder
        )
        N = self._cast_float_array(N)
        N, C, _, _, _ = cat_enc_process(
            N,
            C,
            self.args.cat_policy,
            y,
            ord_encoder=self.ord_encoder,
            mode_values=self.mode_values,
            cat_encoder=self.cat_encoder
        )
        N = self._cast_float_array(N)
        C = self._cast_float_array(C)
        N, _ = data_norm_process(
            N,
            self.args.normalization,
            self.args.seed,
            normalizer=self.normalizer
        )
        N = self._cast_float_array(N)

        self._apply_back(to_dataset, N, C, dt, y)
        
        loader, loader_val, _ = data_loader_process(
            to_dataset,
            self.args.device,
            self.args.batch_size,
            is_train=True
        )
        
        return loader, loader_val
    
    def obtain_prev_tokenizer_feature_names(self):
        """Return tokenizer feature order used in previous stages, plus overlap.

        NOTE: This function returns the union of numeric features from *all* previous
        stages (0..stage_index-1). The returned `shared_numeric_features` are features
        that are present in the current stage and also in that previous-union.
        """
        if self.stage_data is None or self.stage_index is None:
            return [], []

        stage_count = self.stage_data.stage_count()
        if stage_count == 0:
            return [], []

        ordered_union: List[str] = []
        for idx in range(0, self.stage_index):
            stage_features = self.stage_data.num_feature_names_of_stage(idx)
            if not stage_features:
                continue
            new_feature = self.sort_features([fea for fea in stage_features if fea not in ordered_union])
            ordered_union.extend(new_feature)

        current_numeric = list(self.D_meta.get('numeric_features', []) or [])
        shared_numeric_features = self.sort_features([fea for fea in current_numeric if fea in ordered_union])

        return ordered_union, shared_numeric_features

    def shared_numeric_with_previous_stage(self):
        """Compute shared numeric features between current stage and immediate previous stage."""
        if self.stage_data is None or self.stage_index is None:
            return []
        if self.stage_index <= 0:
            return []
        prev_features = self.stage_data.num_feature_names_of_stage(self.stage_index - 1) or []
        current_numeric = list(self.D_meta.get('numeric_features', []) or [])
        prev_set = set(prev_features)
        return self.sort_features([fea for fea in current_numeric if fea in prev_set])

    def shared_numeric_with_all_previous_stages(self):
        """Compute shared numeric features between current stage and the union of all previous stages."""
        if self.stage_data is None or self.stage_index is None:
            return []
        _, shared = self.obtain_prev_tokenizer_feature_names()
        return shared
    
    def data_format(self, stage_idx):
        
        if stage_idx < 0 or stage_idx >= self.stage_data.stage_count():
            return None, None, None, None, None, None
        
        D, D_test, D_pos, D_meta = self.stage_data.obtain_stage_data(stage_idx)
            
        loader_batch_size = self.args.batch_size
        if self.stage_index is not None and stage_idx < self.stage_index:
            loader_batch_size = getattr(self.args, 'batch_size_prev', None) or self.args.batch_size

        loaders_D = self.data_preprocessing_train_and_test(
            D,
            D_test,
            batch_size=loader_batch_size,
        )
            
        loaders_D_pos, _ = self.apply_current_process_to_dataset(D_pos)
        
        
        n_num, n_cat = D_meta['numeric_features'], D_meta['categorical_features']
        
        
        self.logger.debug(f'Stage {stage_idx} data format:')
        self.logger.debug(f'Number of numerical features: {n_num}')
        self.logger.debug(f'Number of categorical features: {n_cat}')
        self.logger.debug(f'Number of training samples: {len(D)}')
        self.logger.debug(f'Number of testing samples: {len(D_test)}')
        
        # self.logger.debug(f'D.N[0:5]: {D.N[0:5]}')
            
        return loaders_D['train_loader'], loaders_D['val_loader'], loaders_D['test_loader'], loaders_D['criterion'], loaders_D_pos, D_meta
    

    def init_model_optimizer(self):
        owner = self.model.module if hasattr(self.model, "module") else self.model
        for n, p in owner.named_parameters():
            print(f'{n}')
        print(list(p for n, p in owner.named_parameters() if 'head' in n))
        fea_params = [p for n, p in owner.named_parameters() if ('head' not in n and 'norm' not in n and 'tokenizer' not in n and "temporal_embeddings" not in n)]
        cls_params_all = list(p for n, p in owner.named_children() if 'head' in n)[0]
        cls_params = list(cls_params_all.parameters())
        norm_param = [p for n, p in owner.named_parameters() if 'norm' in n]
        tokenizer_params = [
            p for n, p in owner.named_parameters() if ('tokenizer' in n)
        ]

        print("fea_params names:")
        for name, param in owner.named_parameters():
            if ('head' not in name and 'norm' not in name and 'tokenizer' not in name and "temporal_embeddings" not in name):
                print(name)
                
        print("tokenizer_param names:")
        for name, param in owner.named_parameters():
            if ('tokenizer' in name):
                print(name)

        # print(cls_params)
        # print(norm_param)
        model_optimizer_arg = {'params': [{'params': fea_params, 'lr': self.config['training']['lr']},
                                          {'params': cls_params, 'weight_decay': 0.0,
                                              'lr': self.config['training']['head_lr']},
                                          {'params': norm_param, 'lr': self.config['training']['bn_lr']},
                                          {'params': tokenizer_params, 'lr': self.config['training']['tokenizer_lr']}],
                               'lr': self.config['training']['lr'],
                               'weight_decay': self.config['training']['model_weight_decay']}
        

        self.optimizer = optim.Adam(**model_optimizer_arg)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.max_epoch, eta_min=self.config['training']['eta_min'])

    def has_prev_data(self):
        
        return self.train_loader_prev_D is not None

    def on_training_preparation(self):
        """Hook for algorithm-specific setup that requires data loaders and the current model."""
        return

    def on_before_training_loop(self):
        """Hook executed after optimizers are created but before training epochs begin."""
        return

    def on_before_optimizer_step(self, tokenizer_owner, batch_index, batch):
        """Hook executed after ``loss.backward()`` but before ``optimizer.step()``."""
        return

    def _freeze_shared_tokenizer_grads(
        self,
        tokenizer_owner,
        *,
        tokenizer_attr: str,
        shared_features_attr: str = 'shared_numeric_features_all_prev',
    ) -> None:
        """Zero gradients for shared feature rows in a Tokenizer-like module.

        This implements partial freezing for feature-tokenizer models, where a single
        `weight` / `bias` parameter spans multiple features (so `requires_grad=False`
        cannot be applied per-feature).

        Parameters
        - tokenizer_owner: model (or model.module under DataParallel)
        - tokenizer_attr: attribute name on the owner, e.g. 'tokenizer' or 'sub_tokenizer'
        - shared_features_attr: attribute name on `self` holding the features to freeze
        """
        if not bool(getattr(self.args, 'freeze_shared_tokenizer', False)):
            return

        shared = getattr(self, shared_features_attr, None) or []
        if not shared:
            return

        tokenizer = getattr(tokenizer_owner, tokenizer_attr, None)
        if tokenizer is None:
            return

        # Tokenizer layout: if `cls` is True, weight rows are [CLS, feat0, feat1, ...]
        cls_offset = 1 if getattr(tokenizer, 'cls', True) else 0

        feature_map = getattr(tokenizer, 'feature_name_to_idx', None)
        if not isinstance(feature_map, dict):
            return

        w = getattr(tokenizer, 'weight', None)
        b = getattr(tokenizer, 'bias', None)

        for feat in shared:
            idx = feature_map.get(feat)
            if idx is None:
                continue
            weight_row = cls_offset + idx
            if w is not None and getattr(w, 'grad', None) is not None:
                if weight_row < w.grad.size(0):
                    w.grad[weight_row].zero_()
            if b is not None and getattr(b, 'grad', None) is not None:
                if idx < b.grad.size(0):
                    b.grad[idx].zero_()
                    
        self.logger.debug(f'shared feature shape = {len(shared)}')

    def _freeze_shared_tokenizer_grads_via_feature_names(
        self,
        *,
        tokenizer,
        shared_feature_names: List[str],
        param_attr_names: tuple = ('weight', 'bias'),
        feature_names_attr_candidates: tuple = ('feature_names', 'tokenizer_feature_names'),
        feature_name_to_idx_attr_candidates: tuple = ('feature_name_to_idx',),
        feature_name_to_idx: Optional[dict] = None,
    ) -> None:
        """Freeze (zero grad) tokenizer params for specific feature names.

        This is a more general helper than `_freeze_shared_tokenizer_grads`:
        - Supports Tokenizer-like modules with per-feature rows (e.g., weight/bias).
        - Supports per-feature ModuleList tokenizers, one module per feature.

        Freezing is gated by CLI flag `--freeze_shared_tokenizer`.
        """
        
        if not bool(getattr(self.args, 'freeze_shared_tokenizer', False)):
            return
        if tokenizer is None:
            return
        if not shared_feature_names:
            return
        

        # print(f'len(shared_feature_names) = {len(shared_feature_names)}')

        # Resolve feature->index mapping
        fmap = feature_name_to_idx
        if fmap is None:
            for cand in feature_name_to_idx_attr_candidates:
                maybe = getattr(tokenizer, cand, None)
                if isinstance(maybe, dict):
                    fmap = maybe
                    break

        # Resolve feature name list (optional, used as fallback)
        fnames = None
        for cand in feature_names_attr_candidates:
            maybe = getattr(tokenizer, cand, None)
            if isinstance(maybe, (list, tuple)):
                fnames = list(maybe)
                break

        def _feature_idx(name: str) -> Optional[int]:
            if isinstance(fmap, dict):
                return fmap.get(name)
            if fnames is not None:
                try:
                    return fnames.index(name)
                except ValueError:
                    return None
            return None

        # Case A: per-feature tokenizers implemented as ModuleList.
        if isinstance(tokenizer, nn.ModuleList):
            for feat in shared_feature_names:
                idx = _feature_idx(feat)
                if idx is None or idx < 0 or idx >= len(tokenizer):
                    continue
                module = tokenizer[idx]
                for p in module.parameters(recurse=True):
                    if getattr(p, 'grad', None) is not None:
                        p.grad.zero_()
            return

        # Case B: matrix-style Tokenizer parameters (weight/bias with feature rows)
        cls_offset = 1 if bool(getattr(tokenizer, 'cls', False)) else 0

        for feat in shared_feature_names:
            idx = _feature_idx(feat)
            if idx is None:
                continue

            for attr in param_attr_names:
                param = getattr(tokenizer, attr, None)
                if param is None:
                    continue
                grad = getattr(param, 'grad', None)
                if grad is None:
                    continue

                # Heuristic: if param has a CLS row, apply offset only when it fits.
                row = idx
                if grad.dim() >= 1:
                    if (cls_offset > 0) and (idx + cls_offset) < grad.size(0):
                        # If param likely includes CLS (e.g., FTT weight), use offset.
                        # For tokenizers without CLS, idx+1 may overflow and we keep idx.
                        # If both fit, prefer offset only when the tokenizer explicitly has `cls=True`.
                        if bool(getattr(tokenizer, 'cls', False)):
                            row = idx + cls_offset

                if grad.dim() == 1:
                    if 0 <= idx < grad.size(0):
                        grad[idx].zero_()
                else:
                    if 0 <= row < grad.size(0):
                        grad[row].zero_()

    # Backwards-compat alias (some code calls the singular name)
    def _freeze_shared_tokenizer_grads_via_feature_name(self, *args, **kwargs):
        return self._freeze_shared_tokenizer_grads_via_feature_names(*args, **kwargs)

    def _model_forward(self, X_num, X_cat, dt, go_sub_and_extra: bool = False):
        """Centralized forward hook so subclasses can inject custom arguments."""
        return self.model(
            X_num,
            X_cat,
            dt,
            go_sub_and_extra=go_sub_and_extra,
        )

    def sort_features(self, feature_list):
        if len(feature_list) == 0:
            return []
        return sorted(feature_list, key=lambda x: int(re.search(r'feature_(\d+)', x).group(1)))

    def _log_trainable_parameters(self, note="initial"):
        self.params_json = {p: n for n, p in self.model.named_parameters()}
        if self.logger is None:
            return
        self.logger.debug(f"Trainable parameters snapshot ({note}):")
        for name, param in self.model.named_parameters():
            self.logger.debug(f"{name:60} requires_grad={param.requires_grad}")
    
    def fit(self, stage_index, train = True, config = None, logger = None, sys_args = None):
        
        self.logger = logger
        self.stage_index = stage_index
        self.recall_history = []
        self.trlog['val_recall'] = []
        
        self.logger.debug(f'sys_args = {sys_args}')
        
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.sys_args = sys_args
        
        self.train_loader_D, _, self.test_loader_D, self.criterion, _, self.D_meta = self.data_format(stage_index)

        self.train_loader_prev_D, _, self.test_loader_prev_D, _, _,  self.prev_D_meta = self.data_format(stage_index - 1)
        
        self.logger.debug(f'Previous stage data loader: {self.train_loader_prev_D}')
        
        self.n_num_features, self.n_cat_features = len(self.D_meta['numeric_features']), len(self.D_meta['categorical_features'])

        self.shared_numeric_features = []
        if self.train_loader_prev_D is not None and self.prev_D_meta is not None:
            # curr_features = set(self.D_meta.get('numeric_features', []))
            # prev_features = set(self.prev_D_meta.get('numeric_features', []))
            # self.shared_numeric_features = self.sort_features(curr_features.intersection(prev_features))\
            self.tokenizer_feat_names_prev, self.shared_numeric_features = self.obtain_prev_tokenizer_feature_names()
            self.logger.debug(f'length of shared numeric features with previous stage: {len(self.shared_numeric_features)}')
            self.logger.debug(f'shared numeric features: {self.shared_numeric_features}')
            self.logger.debug(f'tokenizer_feat_names_prev: {self.tokenizer_feat_names_prev}')
        
        if config is not None:
            self.reset_stats_withconfig(config)

        
        self.construct_model(train = train)
        self._apply_precision_to_model()
        
        
        self.on_training_preparation()

        self.init_model_optimizer()
        self.logger.debug('Classifier Optimizer is reset!')

        self._log_trainable_parameters(note="model parameters required grad ... ")
        
        self.model.zero_grad()
        
        self.on_before_training_loop()
        
        if not train:
            time_cost = 1
            return time_cost, 0.0
        
        time_cost = 0
        best_recall = None
        
        for epoch in range(self.args.max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            best_recall, current_acc, current_recall, current_auc = self.validate(epoch)
            elapsed = time.time() - tic
            time_cost += elapsed
            logger.debug(f'Epoch: {epoch}, Time cost: {elapsed}\n')
            self.logger.info('Epoch {}/{}, Lr={:.7f}, Time cost={:.2f}s, Accuracy={:.4f}, AUC={:.4f}, Recall={:.5f}, Best val={:.5f} @ epoch {}'.format(
                    epoch + 1, self.args.max_epoch, self.optimizer.param_groups[0]['lr'], elapsed, current_acc, current_auc, current_recall, best_recall, self.trlog['best_epoch'] + 1)
                    )
            
            self.scheduler.step()
            if not self.continue_training:
                break
        
        checkpoint_model = self._build_checkpoint_model()
        self._save_unmerged_checkpoint('epoch-last-{}-no-merge.pth'.format(str(self.args.seed)))
        torch.save(
            dict(params=checkpoint_model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        del checkpoint_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._export_best_checkpoint(best_recall)
        self._plot_recall_curve()
        
        return time_cost, best_recall

    def merge_tokenizers(self):
        raise NotImplementedError

    def _capture_rng_state(self):
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda'])

    def _build_checkpoint_model(self):
        """Create a CPU copy of the current model with merged tokenizers for persistence."""
        rng_state = self._capture_rng_state()
        if self.args.stage == 'stage_1':
            merged_model = None
        else:
            merged_model = self.merge_tokenizers()
        if merged_model is None:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            merged_model = deepcopy(base_model)

        self._restore_rng_state(rng_state)

        merged_model.eval()
        merged_model = merged_model.to('cpu')
        return merged_model
    
    def construct_model(self, train=True):
        raise NotImplementedError

    def _current_stage_name(self) -> Optional[str]:
        if self.stage_data is not None and self.stage_index is not None:
            stage_name = self.stage_data.stage_name(self.stage_index)
            if stage_name:
                return stage_name
        return self.stage

    def _stage_storage_key(self, stage_name: Optional[str]) -> Optional[str]:
        if not stage_name or not self.dataset:
            return None
        return f"{self.dataset}-{stage_name}"

    def _method_eval_root(self) -> Optional[str]:
        model_path = getattr(self.args, 'model_path', None)
        model_type = getattr(self.args, 'model_type', None)
        if not model_path or not model_type:
            return None
        return os.path.join(model_path, 'evaluation_best', model_type)

    def _method_train_root(self) -> Optional[str]:
        model_path = getattr(self.args, 'model_path', None)
        model_type = getattr(self.args, 'model_type', None)
        if not model_path or not model_type:
            return None
        return os.path.join(model_path, model_type)

    def _stage_train_root(self, stage_name: Optional[str]) -> Optional[str]:
        stage_key = self._stage_storage_key(stage_name)
        train_root = self._method_train_root()
        if stage_key is None or train_root is None:
            return None
        return os.path.join(train_root, stage_key)

    def _extract_metric_from_filename(self, filename: str) -> Optional[float]:
        if not filename.endswith('.pth'):
            return None
        for tag in ('recall', 'metric'):
            match = re.search(rf"{tag}([0-9]+(?:\.[0-9]+)?)", filename)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None

    def _select_checkpoint_from_evaluation(self, stage_name: Optional[str]) -> Optional[str]:
        stage_key = self._stage_storage_key(stage_name)
        method_eval_root = self._method_eval_root()
        if stage_key is None or method_eval_root is None:
            return None
        eval_dir = os.path.join(method_eval_root, stage_key)
        if not os.path.isdir(eval_dir):
            return None
        best_path = None
        best_score = float('-inf')
        for fname in os.listdir(eval_dir):
            metric_value = self._extract_metric_from_filename(fname)
            if metric_value is None:
                continue
            if metric_value > best_score:
                best_score = metric_value
                best_path = os.path.join(eval_dir, fname)
        
        return best_path

    def _select_checkpoint_from_stage_dir(self, stage_name: Optional[str]) -> Optional[str]:
        stage_dir = self._stage_train_root(stage_name)
        if stage_dir is None:
            return None
        direct_ckpt = os.path.join(stage_dir, f"best-val-{self.args.seed}.pth")
        if os.path.isfile(direct_ckpt):
            return direct_ckpt
        if not os.path.isdir(stage_dir):
            return None
        candidates = []
        for entry in os.listdir(stage_dir):
            entry_path = os.path.join(stage_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            candidate = os.path.join(entry_path, f"best-val-{self.args.seed}.pth")
            if os.path.isfile(candidate):
                candidates.append(candidate)
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        
        return candidates[0]

    def _resolve_stage_checkpoint(self, stage_name: Optional[str]) -> Optional[str]:
        if stage_name is None:
            return None
        checkpoint = self._select_checkpoint_from_evaluation(stage_name)
        
        # if checkpoint is None:
        #     checkpoint = self._select_checkpoint_from_stage_dir(stage_name)
        
        logger = getattr(self, 'logger', None)
        if checkpoint and logger is not None:
            logger.debug(f"Using checkpoint '{checkpoint}' for stage '{stage_name}'")
        elif checkpoint is None and logger is not None:
            logger.debug(f"No checkpoint found for stage '{stage_name}'")
        return checkpoint

    def _export_best_checkpoint(self, best_metric: Optional[float]):
        if best_metric is None or not hasattr(self.args, 'save_path') or not hasattr(self.args, 'model_path'):
            return
        # Decide which source checkpoint to persist based on the hyperparameter
        persist_best = getattr(self.args, 'persist_best_or_last', True)
        best_src = os.path.join(self.args.save_path, f"best-val-{self.args.seed}.pth")
        last_src = os.path.join(self.args.save_path, f"epoch-last-{self.args.seed}.pth")
        if persist_best:
            src = best_src
        else:
            src = last_src
        if not os.path.isfile(src):
            return
        stage_name = self._current_stage_name()
        stage_key = self._stage_storage_key(stage_name)
        method_eval_root = self._method_eval_root()
        if stage_key is None or method_eval_root is None:
            return
        eval_dir = os.path.join(method_eval_root, stage_key)
        os.makedirs(eval_dir, exist_ok=True)
        metric_tag = 'recall'
        # Use provided best_metric for naming; if missing, use 0.0
        metric_value = float(best_metric * 100.0) if best_metric is not None else 0.0
        dest = os.path.join(eval_dir, f"best-val-{self.args.seed}-{metric_tag}{metric_value:.2f}.pth")
        shutil.copy(src, dest)
        logger = getattr(self, 'logger', None)
        if logger is not None:
            logger.debug(f"Best checkpoint copied to {dest}")

    def _save_unmerged_checkpoint(self, file_name: str):
        if self.model is None:
            return
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, file_name),
        )

    def _resolve_eval_checkpoint(self, base_name: str):
        path = self.args.evaluate_model_path if self.args.evaluate_model_path is not None else self.args.save_path
        no_merge_path = os.path.join(path, f"{base_name}-{self.args.seed}-no-merge.pth")
        merged_path = os.path.join(path, f"{base_name}-{self.args.seed}.pth")
        if os.path.isfile(no_merge_path):
            return no_merge_path, True
        return merged_path, False

    def _plot_recall_curve(self):
        if not self.recall_history or not hasattr(self.args, 'save_path'):
            return
        epochs = list(range(1, len(self.recall_history) + 1))
        ylabel = 'Recall'
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, self.recall_history, marker='o', label=ylabel)
        if self.trlog.get('best_epoch') is not None and self.trlog['best_epoch'] < len(epochs):
            best_epoch = self.trlog['best_epoch'] + 1
            best_value = self.recall_history[self.trlog['best_epoch']]
            plt.scatter([best_epoch], [best_value], color='red', label='Best')
            plt.axvline(best_epoch, color='red', linestyle='--', linewidth=0.8)
        title_stage = self._current_stage_name() or 'stage'
        plt.title(f"{title_stage} {ylabel} vs Epoch")
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend()
        plot_path = os.path.join(self.args.save_path, 'recall_curve.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger = getattr(self, 'logger', None)
        if logger is not None:
            logger.debug(f"Saved recall curve to {plot_path}")

    def _build_prediction_payload(self, predictions, labels):
        prediction_array = _to_numpy_array(predictions)
        label_array = _to_numpy_array(labels)

        if prediction_array.ndim != 2 or prediction_array.shape[1] < 2:
            raise ValueError(
                f"Expected prediction array with shape (N, C>=2), got {prediction_array.shape}."
            )

        probabilities = check_softmax(prediction_array)
        fps, recalls, thresholds = skm.roc_curve(label_array, probabilities[:, 1])
        over_budget = np.where(fps > 0.005)[0]
        if over_budget.size > 0:
            target_id = max(int(over_budget.min()) - 1, 0)
        else:
            target_id = max(len(thresholds) - 1, 0)

        threshold = float(thresholds[target_id])
        decisions = (probabilities[:, 1] > threshold).astype(np.int64)
        predicted_label = probabilities.argmax(axis=-1).astype(np.int64)
        correct = (decisions * label_array).astype(np.int64)

        return {
            'current_prediction': correct,
            'threshold_prediction': decisions,
            'predicted_label': predicted_label,
            'positive_probability': probabilities[:, 1],
            'probabilities': probabilities,
            'raw_prediction': prediction_array,
            'test_label': label_array,
            'threshold': np.asarray(threshold, dtype=np.float32),
            'fpr_at_threshold': np.asarray(float(fps[target_id]), dtype=np.float32),
            'recall_at_threshold': np.asarray(float(recalls[target_id]), dtype=np.float32),
        }

    def _save_prediction_artifact(self, file_path, predictions, labels, epoch=None):
        payload = self._build_prediction_payload(predictions, labels)
        if epoch is not None:
            payload['epoch'] = np.asarray(int(epoch) + 1, dtype=np.int64)
        np.savez_compressed(file_path, **payload)

    def _save_epoch_prediction(self, epoch, predictions, labels):
        save_path = getattr(self.args, 'save_path', None)
        if not save_path:
            return
        epoch_dir = os.path.join(save_path, 'epoch_predictions')
        os.makedirs(epoch_dir, exist_ok=True)
        file_path = os.path.join(epoch_dir, f'epoch_{int(epoch) + 1:04d}.npz')
        self._save_prediction_artifact(file_path, predictions, labels, epoch=epoch)

    def tokenizer_weight_and_bias_selection(self, tokenizer_weight, tokenizer_bias, tokenizer_names, used_feature_names):

        assert len(tokenizer_names) == len(tokenizer_weight) - 1
        
        indices = [tokenizer_names.index(name) for name in used_feature_names]
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        weight = nn.Parameter(
            torch.cat([
                tokenizer_weight[0:1],
                tokenizer_weight[1 + indices_tensor]
            ], dim=0)
        )
        
        bias = None
        if tokenizer_bias is not None:
            bias = nn.Parameter(tokenizer_bias[indices_tensor])

        return weight, bias

    def _normalize_checkpoint_state_dict(self, checkpoint, target_state_dict):
        target_keys = set(target_state_dict.keys())
        candidate_state_dicts = [checkpoint]

        if any(key.startswith('module.') for key in checkpoint.keys()):
            candidate_state_dicts.append(
                {
                    key.replace('module.', '', 1) if key.startswith('module.') else key: value
                    for key, value in checkpoint.items()
                }
            )
        else:
            candidate_state_dicts.append(
                {f'module.{key}': value for key, value in checkpoint.items()}
            )

        best_state_dict = checkpoint
        best_match_count = -1
        for candidate in candidate_state_dicts:
            match_count = len(target_keys.intersection(candidate.keys()))
            if match_count > best_match_count:
                best_state_dict = candidate
                best_match_count = match_count

        return best_state_dict, max(best_match_count, 0)
        
    def derive_model_from_checkpoint(self, checkpoint_path, template_model=None, used_feature_names=None):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = payload['params'] if isinstance(payload, dict) and 'params' in payload else payload

        if template_model is None:
            template_model = self.model

        model_copy = deepcopy(template_model)
        model_copy_owner = model_copy.module if hasattr(model_copy, 'module') else model_copy

        if self.args.stage != 'stage_1' and self.prev_model is not None and used_feature_names is not None:
            from ..models.FTT import Tokenizer

            if any(key.startswith('module.') for key in checkpoint.keys()) and not hasattr(model_copy, 'module'):
                checkpoint = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}
            
            tokenizer_owner = self.prev_model.module if hasattr(self.prev_model, 'module') else self.prev_model
            
            weight_key = 'module.tokenizer.weight' if 'module.tokenizer.weight' in checkpoint else 'tokenizer.weight'
            bias_key = 'module.tokenizer.bias' if 'module.tokenizer.bias' in checkpoint else 'tokenizer.bias'

            feature_names_tokenizer = tokenizer_owner.tokenizer.feature_names + self.new_numeric_features

            if weight_key in checkpoint and bias_key in checkpoint:
                checkpoint[weight_key], checkpoint[bias_key] = \
                    self.tokenizer_weight_and_bias_selection(
                        checkpoint[weight_key], 
                        checkpoint[bias_key],
                        feature_names_tokenizer, 
                        used_feature_names
                    )
            
            model_copy_owner.tokenizer = Tokenizer(
                    d_numerical=len(used_feature_names),
                    categories=[],
                    d_token=tokenizer_owner.tokenizer.d_token,
                    bias=True,
                    feature_names=used_feature_names,
                    cls_token=True,
                )
            model_copy_owner.sub_tokenizer = None
            model_copy_owner.extra_tokenizer = None
            
            model_copy_owner.update_feature_set(new_feature_names=None, shared_feature_names=None, tokenizer_feature_names=used_feature_names, current_stage_feature_names=used_feature_names)

        checkpoint, matched_key_count = self._normalize_checkpoint_state_dict(
            checkpoint,
            model_copy_owner.state_dict(),
        )
        if matched_key_count == 0:
            raise RuntimeError(
                f"No checkpoint keys matched model parameters while loading '{checkpoint_path}'."
            )

        missing_keys, unexpected_keys = model_copy_owner.load_state_dict(checkpoint, strict=False)

        if self.logger is not None:
            self.logger.debug(
                f'loaded {matched_key_count} checkpoint keys; '
                f'missing keys: {missing_keys}, unexpected keys: {unexpected_keys} ... '
                f'checkpoint_path={checkpoint_path} ... '
            )
        
        model_copy.to(device=self.args.device, dtype=self._torch_dtype())
        model_copy.eval()
        
        return model_copy

    def _specialize_prev_model_tokenizer(self, prev_model, stage_numeric_features):
        """Configure prev_model.tokenizer to match the numeric feature space of a prior stage."""
        if prev_model is None or not stage_numeric_features:
            return prev_model

        tokenizer_owner = prev_model.module if hasattr(prev_model, 'module') else prev_model
        tokenizer = getattr(tokenizer_owner, 'tokenizer', None)
        if tokenizer is None or not hasattr(tokenizer, 'feature_names'):
            return prev_model

        requested_features = list(stage_numeric_features)
        current_features = list(getattr(tokenizer, 'feature_names', []))
        if requested_features == current_features:
            return prev_model

        missing = [name for name in requested_features if name not in current_features]
        if missing:
            raise ValueError(
                f"Cannot specialize previous tokenizer; missing features: {missing}"
            )

        from ..models.FTT import Tokenizer

        weight, bias = self.tokenizer_weight_and_bias_selection(
            tokenizer.weight,
            tokenizer.bias,
            current_features,
            requested_features
        )

        new_tokenizer = Tokenizer(
            d_numerical=len(requested_features),
            categories=[],
            d_token=tokenizer.d_token,
            bias=tokenizer.bias is not None,
            feature_names=requested_features,
            cls_token=getattr(tokenizer, 'cls', True)
        )

        new_tokenizer.weight = nn.Parameter(weight.detach().clone())
        if new_tokenizer.bias is not None and bias is not None:
            new_tokenizer.bias = nn.Parameter(bias.detach().clone())

        tokenizer_owner.tokenizer = new_tokenizer
        tokenizer_owner.sub_tokenizer = None
        tokenizer_owner.extra_tokenizer = None
        tokenizer_owner.update_feature_set(
            new_feature_names=None,
            shared_feature_names=None,
            tokenizer_feature_names=requested_features,
            current_stage_feature_names=requested_features
        )

        logger = getattr(self, 'logger', None)
        if logger is not None:
            logger.debug(
                f"Specialized previous tokenizer to {len(requested_features)} numeric features."
            )

        return prev_model

    def calculate_and_log_gain_rate_etc(self, wrt_which_stage = None):
        
        prev_model = None
        
        if wrt_which_stage is None:
            wrt_which_stage = self.stage_index - 1
            test_loader_wrt_which_stage = self.test_loader_prev_D
            meta_wrt_which_stage = self.prev_D_meta
            prev_model = self.prev_model
        else:
            _, _, test_loader_wrt_which_stage, _, _, meta_wrt_which_stage= self.data_format(wrt_which_stage)
            
            prev_stage = self.stage_data.stage_name(wrt_which_stage)
            prev_ckpt = self._select_checkpoint_from_stage_dir(prev_stage)
            if prev_ckpt is None:
                raise FileNotFoundError(f"Previous stage checkpoint not found for stage '{prev_stage}'")
            prev_model = self.derive_model_from_checkpoint(prev_ckpt, prev_model, meta_wrt_which_stage['numeric_features'])
        
        if self.args.evaluate_model_path is not None:
            path = self.args.evaluate_model_path
        else:
            path = self.args.save_path

        current_best_path, use_stage_structure = self._resolve_eval_checkpoint('best-val')
        current_model = self.derive_model_from_checkpoint(
            current_best_path,
            self.model,
            None if use_stage_structure else self.D_meta['numeric_features'],
        )

        _, vres_curr, _, test_prediction_curr, test_label_curr = self.predict(
            self.test_loader_D,
            current_model,
            self.D_meta,
            go_sub_and_extra=(self.args.stage != 'stage_1') if use_stage_structure else False,
        )
        
        self.logger.info(f'best-val-model (stage {self.stage_index + 1})')
        self.logger.info(f'Accuracy: {vres_curr[0] * 100:.2f} ')
        self.logger.info(f'Recall: {vres_curr[1] * 100:.2f} ')
        self.logger.info(f'Avg_Precision: {vres_curr[2] * 100:.2f}, ')
        self.logger.info(f'F1: {vres_curr[3] : .4f} ')
        self.logger.info(f'LogLoss: {vres_curr[4] : .4f} ')
        self.logger.info(f'AUC: {vres_curr[5]: .3f} \n')
        
        self._save_prediction_artifact(
            os.path.join(path, "prediction_best_val.npz"),
            test_prediction_curr,
            test_label_curr,
        )
        
        current_best_path, use_stage_structure = self._resolve_eval_checkpoint('epoch-last')
        epoch_last_model = self.derive_model_from_checkpoint(
            current_best_path,
            self.model,
            None if use_stage_structure else self.D_meta['numeric_features'],
        )

        _, vres_curr, _, test_prediction_curr, test_label_curr = self.predict(
            self.test_loader_D,
            epoch_last_model,
            self.D_meta,
            go_sub_and_extra=(self.args.stage != 'stage_1') if use_stage_structure else False,
        )
        
        self.logger.info(f'epoch-last-model (stage {self.stage_index + 1})')
        self.logger.info(f'Accuracy: {vres_curr[0] * 100:.2f} ')
        self.logger.info(f'Recall: {vres_curr[1] * 100:.2f} ')
        self.logger.info(f'Avg_Precision: {vres_curr[2] * 100:.2f}, ')
        self.logger.info(f'F1: {vres_curr[3] : .4f} ')
        self.logger.info(f'LogLoss: {vres_curr[4] : .4f} ')
        self.logger.info(f'AUC: {vres_curr[5]: .3f} \n')
        
        self._save_prediction_artifact(
            os.path.join(path, "prediction_epoch_last.npz"),
            test_prediction_curr,
            test_label_curr,
        )
        
        # if self.has_prev_data():
        #     prev_model.to(device=self.args.device, dtype=self._torch_dtype())
        #     prev_model.eval()
        #     if isinstance(meta_wrt_which_stage, dict):
        #         prev_features = meta_wrt_which_stage.get('numeric_features')
        #     else:
        #         prev_features = getattr(meta_wrt_which_stage, 'numeric_features', None)
        #     self._specialize_prev_model_tokenizer(prev_model, prev_features)
        #     _, vres_prev, _, _, test_label_prev = self.predict(test_loader_wrt_which_stage, prev_model, meta_wrt_which_stage)
        #     self.logger.info(f'Previous Stage (stage {wrt_which_stage + 1})')
        #     self.logger.info(f'Accuracy: {vres_prev[0] * 100:.2f} ')
        #     self.logger.info(f'Recall: {vres_prev[1] * 100:.2f} ')
        #     self.logger.info(f'Avg_Precision: {vres_prev[2] * 100:.2f}, ')
        #     self.logger.info(f'F1: {vres_prev[3] : .4f} ')
        #     self.logger.info(f'LogLoss: {vres_prev[4] : .4f} ')
        #     self.logger.info(f'AUC: {vres_prev[5]: .3f} \n')
            
        #     assert torch.equal(test_label_prev, test_label_curr), "Test labels from previous and current stages do not match."
            
        #     label_sum = test_label_curr.sum().item()
            
        #     gain_rate = (vres_prev[-1] - vres_curr[-1] == 1.).sum() / label_sum if label_sum else 0.0
        #     reduce_rate = (vres_curr[-1] - vres_prev[-1] == 1.).sum() / label_sum if label_sum else 0.0
            
        #     self.logger.info(f'Gain rate: {gain_rate}, Reduce rate: {reduce_rate}')
        

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        :param epoch: int, the current epoch
        """
        self.model.train()
        
        if self.has_prev_data():
            self.prev_model.eval()
        tl = Averager()
        loaders = [self.train_loader_D]

        tokenizer_owner = self.model.module if hasattr(self.model, 'module') else self.model

        def _parse_debug_batch_count(env_name):
            raw_value = os.environ.get(env_name, '0')
            try:
                return max(int(raw_value), 0)
            except (TypeError, ValueError):
                return 0

        debug_trace_batches = _parse_debug_batch_count('EVOCFD_DEBUG_TRACE_BATCHES')
        debug_stop_after_batches = _parse_debug_batch_count('EVOCFD_DEBUG_STOP_AFTER_BATCHES')
        debug_reset_train_seed = os.environ.get('EVOCFD_DEBUG_RESET_TRAIN_SEED') == '1'

        if debug_reset_train_seed:
            set_seeds(int(getattr(self.args, 'seed', 0)))
            if self.logger is not None:
                self.logger.debug(
                    f'[EVOCFD_DEBUG][train] reset global seeds before epoch {epoch} with seed={int(getattr(self.args, "seed", 0))}'
                )

        def _tokenizer_stats(tokenizer, prefix):
            if tokenizer is None:
                return f'{prefix}=None'
            weight_grad = getattr(tokenizer.weight, 'grad', None)
            bias_grad = getattr(tokenizer.bias, 'grad', None) if getattr(tokenizer, 'bias', None) is not None else None
            weight_nonzero = int((weight_grad != 0).sum().item()) if weight_grad is not None else 0
            bias_nonzero = int((bias_grad != 0).sum().item()) if bias_grad is not None else 0
            return (
                f'{prefix}.weight_norm={float(tokenizer.weight.detach().norm().item()):.6f}, '
                f'{prefix}.weight_grad_norm={float(weight_grad.norm().item()):.6f}' if weight_grad is not None else f'{prefix}.weight_grad_norm=None'
            ) + (
                f', {prefix}.weight_grad_nonzero={weight_nonzero}, '
                f'{prefix}.bias_grad_norm={float(bias_grad.norm().item()):.6f}' if bias_grad is not None else f', {prefix}.weight_grad_nonzero={weight_nonzero}, {prefix}.bias_grad_norm=None'
            ) + f', {prefix}.bias_grad_nonzero={bias_nonzero}'

        def _snapshot_tokenizer(tokenizer):
            if tokenizer is None:
                return None
            snapshot = {'weight': tokenizer.weight.detach().clone()}
            if getattr(tokenizer, 'bias', None) is not None:
                snapshot['bias'] = tokenizer.bias.detach().clone()
            return snapshot

        def _delta_stats(tokenizer, snapshot, prefix):
            if tokenizer is None or snapshot is None:
                return f'{prefix}_delta=None'
            parts = [f'{prefix}.weight_delta_norm={float((tokenizer.weight.detach() - snapshot["weight"]).norm().item()):.6f}']
            if 'bias' in snapshot and getattr(tokenizer, 'bias', None) is not None:
                parts.append(f'{prefix}.bias_delta_norm={float((tokenizer.bias.detach() - snapshot["bias"]).norm().item()):.6f}')
            return ', '.join(parts)

        for i, batches in enumerate(zip(*loaders), 1):
            X_num_0, dt_0, y_0 = batches[0]

            if i == 1 and X_num_0.dim() > 1:
                expected_num = len(self.D_meta['numeric_features'])
                if X_num_0.size(1) != expected_num:
                    raise RuntimeError(
                        f"Numeric feature shape mismatch: batch has {X_num_0.size(1)} columns, expected {expected_num}."
                    )
            X_num_0 = X_num_0.to(self.args.device, non_blocking=True)
            if dt_0 is not None:
                dt_0 = dt_0.to(self.args.device, non_blocking=True)
            y_0 = y_0.to(self.args.device, non_blocking=True)
                    
            go_sub_and_extra = self.args.stage != 'stage_1'
            logits_0 = self._model_forward(
                X_num_0,
                None,
                dt_0,
                go_sub_and_extra=go_sub_and_extra,
            )
            
            loss = self.criterion(logits_0, y_0)

            debug_first_batch = os.environ.get('EVOCFD_DEBUG_FIRST_BATCH') == '1' and i == 1
            debug_batch = debug_first_batch or (debug_trace_batches > 0 and i <= debug_trace_batches)
            sub_snapshot = None
            extra_snapshot = None
            if debug_batch:
                sub_tokenizer = getattr(tokenizer_owner, 'sub_tokenizer', None)
                extra_tokenizer = getattr(tokenizer_owner, 'extra_tokenizer', None)
                sub_snapshot = _snapshot_tokenizer(sub_tokenizer)
                extra_snapshot = _snapshot_tokenizer(extra_tokenizer)
                print(
                    '[EVOCFD_DEBUG][new][train][pre_backward] '
                    f'epoch={epoch + 1} | batch={i} | input_shape={tuple(X_num_0.shape)} | '
                    f'dt_shape={tuple(dt_0.shape) if dt_0 is not None else None} | '
                    f'target_head={y_0[:5].detach().cpu().tolist()} | '
                    f'logits_head={logits_0[:5].detach().cpu().tolist()} | '
                    f'loss={float(loss.item()):.6f} | '
                    f'go_sub_and_extra={go_sub_and_extra}'
                )
            

            tl.add(loss.item())
            self.optimizer.zero_grad()
            
            loss.backward()

            if debug_batch:
                print(
                    '[EVOCFD_DEBUG][new][train][post_backward] '
                    + _tokenizer_stats(getattr(tokenizer_owner, 'sub_tokenizer', None), 'sub')
                    + ' | '
                    + _tokenizer_stats(getattr(tokenizer_owner, 'extra_tokenizer', None), 'extra')
                )
            
            self.on_before_optimizer_step(
                tokenizer_owner=tokenizer_owner,
                batch_index=i,
                batch=batches
            )

            if debug_batch:
                print(
                    '[EVOCFD_DEBUG][new][train][post_mask] '
                    + _tokenizer_stats(getattr(tokenizer_owner, 'sub_tokenizer', None), 'sub')
                    + ' | '
                    + _tokenizer_stats(getattr(tokenizer_owner, 'extra_tokenizer', None), 'extra')
                )

            self.optimizer.step()

            if debug_batch:
                print(
                    '[EVOCFD_DEBUG][new][train][post_step] '
                    + _delta_stats(getattr(tokenizer_owner, 'sub_tokenizer', None), sub_snapshot, 'sub')
                    + ' | '
                    + _delta_stats(getattr(tokenizer_owner, 'extra_tokenizer', None), extra_snapshot, 'extra')
                )
                should_stop_after_first = os.environ.get('EVOCFD_DEBUG_STOP_AFTER_FIRST_BATCH') == '1' and i == 1
                should_stop_after_n = debug_stop_after_batches > 0 and i >= debug_stop_after_batches
                if should_stop_after_first or should_stop_after_n:
                    if should_stop_after_first:
                        print('[EVOCFD_DEBUG][new][train] stopping after first batch by request')
                    else:
                        print(f'[EVOCFD_DEBUG][new][train] stopping after batch {i} by request')
                    break
            
            if (i-1) % 50 == 0 or i == len(self.train_loader_D):
                self.logger.debug('epoch {}/{}, train {}/{}, loss={:.4f}, lr={:.4g}'.format(
                    epoch + 1, self.args.max_epoch, i, len(self.train_loader_D), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    
    
    def validate(self, epoch):
        """
        Validate the model.
        :param epoch: int, the current epoch
        """
        self.logger.debug('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for _, (X, dt, y) in tqdm(enumerate(self.test_loader_D)): # modify the get_dataset() and data_loader_process() in ../lib/data
                X = X.to(self.args.device, non_blocking=True)
                if dt is not None:
                    dt = dt.to(self.args.device, non_blocking=True)
                y = y.to(self.args.device, non_blocking=True)
                
                go_sub_and_extra = self.args.stage != 'stage_1'
                pred = self._model_forward(
                    X,
                    None,
                    dt,
                    go_sub_and_extra=go_sub_and_extra,
                )
                
                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()   
        
        vres, metric_name = self.metric(test_logit, test_label)
        recall_value = float(vres[1])
        self._save_epoch_prediction(epoch, test_logit, test_label)
        self.recall_history.append(recall_value)
        self.trlog.setdefault('val_recall', []).append(recall_value)
        self.logger.debug('epoch {}/{}, val, loss={:.4f}, accuracy={:.4f}, recall={:.5f}, auc={:.4f}'.format(epoch, self.args.max_epoch, vl, vres[0], vres[1], vres[-2]))
        if np.greater_equal(vres[1], self.trlog['best_res']) or epoch == 0:
            self._save_unmerged_checkpoint('best-val-{}-no-merge.pth'.format(str(self.args.seed)))
            checkpoint_model = self._build_checkpoint_model()
            self.trlog['best_res'] = vres[1]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=checkpoint_model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            del checkpoint_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 30:
                self.continue_training = False
        
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))
        
        return self.trlog['best_res'], vres[0], vres[1], vres[-2]

    def predict(self, data_loader, used_model, meta, go_sub_and_extra = None):
        
        ## Evaluation Stage
        used_model.eval()
        model_device = torch.device(self.args.device)
        if go_sub_and_extra is None:
            go_sub_and_extra = self.args.stage != 'stage_1'
        
        test_logit, test_label = [], []
        with torch.no_grad():
            for _, (X, dt, y) in tqdm(enumerate(data_loader)):
                X = X.to(model_device, non_blocking=True)
                if dt is not None:
                    dt = dt.to(model_device, non_blocking=True)
                y = y.to(model_device, non_blocking=True)
                
                pred = used_model(
                    X,
                    None,
                    dt=dt,
                    ret_feature=False,
                    go_sub_and_extra=go_sub_and_extra,
                )
                
                test_logit.append(pred)
                test_label.append(y)
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        
        vl = self.criterion(test_logit, test_label).item()     
        
        vres, metric_name = self.metric(test_logit, test_label)
        
        return vl, vres, metric_name, test_logit.cpu(), test_label.cpu()

    def metric(self, predictions, labels):
        """
        Compute the evaluation metric.
        :param predictions: np.ndarray, predictions
        :param labels: np.ndarray, labels
        :param y_info: dict, information about the labels
        :return: tuple, (metric, metric_name)
        """
        if not isinstance(labels, np.ndarray):
            labels = labels.cpu().numpy()
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.cpu().numpy()
        
        
        # if not softmax, convert to probabilities
        predictions = check_softmax(predictions)
        accuracy = skm.accuracy_score(labels, predictions.argmax(axis=-1))
        # avg_recall = skm.balanced_accuracy_score(labels, predictions.argmax(axis=-1))
        fps, recalls, thresholds = skm.roc_curve(labels, predictions[:, 1])
        target_id = np.where(fps > 0.005)[0].min() - 1
        decisions = (predictions[:, 1] > thresholds[target_id]).astype(float)
        correct = decisions * labels
        recall = recalls[target_id]
        avg_precision = skm.precision_score(labels, predictions.argmax(axis=-1), average='macro')
        f1_score = skm.f1_score(labels, predictions.argmax(axis=-1), average='binary')
        log_loss = skm.log_loss(labels, predictions)
        auc = skm.roc_auc_score(labels, predictions[:, 1])
        
        return (accuracy, recall, avg_precision, f1_score, log_loss, auc, correct), ("Accuracy", "Recall", "Avg_Precision", "F1", "LogLoss", "AUC", "Correct")
        

    def cross_entropy(self, logits, labels, reduction='mean'):
        N, C = logits.shape
        assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
        log_logits = F.log_softmax(logits, dim=1)
        losses = -torch.sum(log_logits * labels, dim=1)  # (N)
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return torch.sum(losses) / logits.size(0)
        elif reduction == 'sum':
            return torch.sum(losses)
        else:
            raise AssertionError('reduction has to be none, mean or sum')
