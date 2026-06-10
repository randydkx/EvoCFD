from .base import Method
import torch
import torch.nn as nn
import os
import re
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from model.dlmodel.model.utils import AverageMeter

class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

def l2_sq_sum_per_row(x):
    # x: [B, D] -> [B]
    return (x * x).sum(dim=1)


class EvoCFDMethod(Method):

    def __init__(self, args, stage_data):
        super().__init__(args, stage_data)
        self.stage = args.stage
        self.dataset = args.dataset
        self.args.precision = 'double'
        self._precision_mode = 'double'
        self._torch_precision_dtype = torch.float64
        self._numpy_precision_dtype = np.float64
        self.stage_index = int(self.stage.split('_')[-1]) - 1
        self.keep_ratio = getattr(args, 'keep_ratio', None)
        self.clipping_ratio_alpha = getattr(args, 'clipping_ratio_alpha', None)
        self.apply_max_sv_clip = getattr(args, 'apply_max_sv_clip', False)
        self.required_NSP_modules_list = []
        self.required_NSP_modules_name = []
        self.stats = {}
        self.ub_p = 1 / 10
        self.ub_delta = 1 / 100
        self.ub_epsilon_ratio = 0

    @property
    def overall_featureset(self):
        return self.stage_data.overall_feature_names()

    def _checkpoint_candidates(self, root_dir, checkpoint_name):
        if root_dir is None or not os.path.isdir(root_dir):
            return []

        candidates = []
        seen = set()
        for current_root, _, files in os.walk(root_dir):
            if checkpoint_name not in files:
                continue
            candidate = os.path.join(current_root, checkpoint_name)
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates

    def _select_checkpoint_from_stage_dir(self, stage_name):
        stage_dir = self._stage_train_root(stage_name)
        if stage_dir is None or not os.path.isdir(stage_dir):
            return None

        preferred_root = None
        exp_name = getattr(self.args, 'exp', None)
        if exp_name:
            candidate_root = os.path.join(stage_dir, exp_name)
            if os.path.isdir(candidate_root):
                preferred_root = candidate_root

        checkpoint_names = [f"best-val-{self.args.seed}.pth", f"epoch-last-{self.args.seed}.pth"]
        if not getattr(self.args, 'persist_best_or_last', True):
            checkpoint_names.reverse()

        search_roots = []
        if preferred_root is not None:
            search_roots.append(preferred_root)
        search_roots.append(stage_dir)

        for root_dir in search_roots:
            for checkpoint_name in checkpoint_names:
                candidates = self._checkpoint_candidates(root_dir, checkpoint_name)
                if candidates:
                    return candidates[0]
        return None

    def _resolve_stage_checkpoint(self, stage_name):
        if stage_name is None:
            return None

        checkpoint = self._select_checkpoint_from_evaluation(stage_name)
        if checkpoint is None:
            checkpoint = self._select_checkpoint_from_stage_dir(stage_name)

        if checkpoint and self.logger is not None:
            self.logger.debug(f"Using checkpoint '{checkpoint}' for stage '{stage_name}'")
        elif checkpoint is None and self.logger is not None:
            self.logger.debug(f"No checkpoint found for stage '{stage_name}'")
        return checkpoint

    def _apply_precision_to_model(self):
        if self.model is not None:
            self.model = self.model.to(self.args.device)
            self.model.double()
        if self.prev_model is not None:
            self.prev_model = self.prev_model.to(torch.device('cpu'))

    def _consume_legacy_stage_construction_rng(self, model_config):
        if self.stage_index < 2:
            return

        from ..models.FTT import Transformer

        first_stage_features = list(self.stage_data.num_feature_names_of_stage(0) or [])
        if not first_stage_features:
            return

        Transformer(
            d_numerical=len(first_stage_features),
            categories=[],
            new_feature_names=None,
            shared_feature_names=None,
            tokenizer_feature_names=first_stage_features,
            d_out=2,
            logger=self.logger,
            **model_config
        )

    def construct_model(self, model_config = None, train=False):
        from ..models.FTT import Transformer, Tokenizer
        if model_config is None:
            model_config = self.args.config['model']

        curr_numeric_features = list(self.D_meta.get('numeric_features', []))
        shared_numeric = self.shared_numeric_features
        self.new_numeric_features = self.sort_features([f for f in curr_numeric_features if f not in shared_numeric])

        if self.args.stage == 'stage_1':
            self.model = Transformer(
                d_numerical=len(curr_numeric_features),
                categories=[],
                new_feature_names=None,
                shared_feature_names=None,
                tokenizer_feature_names=curr_numeric_features,
                d_out=2,
                logger = self.logger,
                **model_config
            ).to(self.args.device)
            
        else:
            prev_stage = self.stage_data.previous_stage_name(self.stage_index)
            prev_ckpt = self._resolve_stage_checkpoint(prev_stage)
            if prev_ckpt is None:
                raise FileNotFoundError(f"Unable to locate checkpoint for previous stage '{prev_stage}'.")

            prev_numeric_features = self.tokenizer_feat_names_prev
            self._consume_legacy_stage_construction_rng(model_config)
            self.prev_model = Transformer(
                d_numerical=len(prev_numeric_features),
                categories=[],
                new_feature_names=None,
                shared_feature_names=None,
                tokenizer_feature_names=prev_numeric_features,
                d_out=2,
                logger = self.logger,
                **model_config
            )

            ckpt = torch.load(prev_ckpt, map_location='cpu')['params']
            if any(k.startswith("module.") for k in ckpt.keys()):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            self.prev_model.load_state_dict(ckpt, strict=True)
            self.prev_model = self.prev_model.to(torch.device('cpu'))
            self.model = copy.deepcopy(self.prev_model).to(self.args.device)
            

        self.model = self.model.to(self.args.device)
        self._wrap_model_for_requested_gpus()

        self._initialize_stage_tokenizers()
        
        # torch.cuda.empty_cache()

    def _initialize_stage_tokenizers(self):
        owner = self.model.module if hasattr(self.model, 'module') else self.model
        owner.legacy_drop_first_extra_token = self.args.stage != 'stage_1'
        if self.args.stage == 'stage_1':
            owner.sub_tokenizer = None
            owner.extra_tokenizer = None
            owner.first_tokenizer = None
            owner.second_tokenizer = None
            owner.subsub_tokenizer = None
            return

        from ..models.FTT import Tokenizer
        token_cfg = self.args.config['model']
        owner.first_tokenizer = None
        owner.second_tokenizer = None
        owner.subsub_tokenizer = None

        if self.new_numeric_features:
            owner.extra_tokenizer = Tokenizer(
                d_numerical=len(self.new_numeric_features),
                categories=None,
                feature_names=self.new_numeric_features,
                d_token=token_cfg['d_token'],
                bias=token_cfg['token_bias'],
                cls_token=False,
            ).to(self.args.device)
        else:
            owner.extra_tokenizer = None

        if self.stage_index >= 2:
            first_stage_features = list(self.stage_data.num_feature_names_of_stage(0) or [])
            previous_stage_features = list(self.stage_data.num_feature_names_of_stage(self.stage_index - 1) or [])
            current_stage_features = list(self.D_meta.get('numeric_features', []) or [])
            first_shared_features = [name for name in current_stage_features if name in first_stage_features]

            if first_stage_features:
                owner.first_tokenizer = owner.tokenizer.extract_subtokenizer(first_stage_features).to(self.args.device)
            if previous_stage_features:
                owner.second_tokenizer = owner.tokenizer.extract_subtokenizer(previous_stage_features).to(self.args.device)

        if self.shared_numeric_features:
            owner.sub_tokenizer = owner.tokenizer.extract_subtokenizer(self.shared_numeric_features).to(self.args.device)
        else:
            owner.sub_tokenizer = None

        if self.stage_index >= 2 and first_shared_features:
            owner.subsub_tokenizer = owner.tokenizer.extract_subtokenizer(first_shared_features).to(self.args.device)

    def reset_model_tokenizer_status(self, new_numeric = None, shared_feature_names = None, tokenizer_feature_names = None, current_stage_feature_names = None):
        if self.args.stage == 'stage_1':
            return
        from ..models.FTT import Tokenizer
        new_numeric = list(new_numeric or [])
        shared_feature_names = list(shared_feature_names or [])
        
        owner = self.model.module if hasattr(self.model, 'module') else self.model
        prev_owner = self.prev_model.module if hasattr(self.prev_model, 'module') else self.prev_model
        existing_extra = getattr(owner, 'extra_tokenizer', None)
        needs_fresh_extra = (
            len(new_numeric) > 0 and (
                existing_extra is None
                or list(getattr(existing_extra, 'feature_names', [])) != new_numeric
            )
        )
        new_tokenizer = None
        if needs_fresh_extra:
            new_tokenizer = Tokenizer(
                d_numerical=len(new_numeric),
                categories=None,
                feature_names=new_numeric,
                d_token=self.args.config['model']['d_token'],
                bias=self.args.config['model']['token_bias'],
                cls_token=False,
            ).to(self.args.device)
        
        owner.update_feature_set(new_feature_names = new_numeric, shared_feature_names = shared_feature_names,\
            tokenizer_feature_names = None, current_stage_feature_names = current_stage_feature_names)

        existing_sub = getattr(owner, 'sub_tokenizer', None)
        sub_matches = (
            existing_sub is not None
            and list(getattr(existing_sub, 'feature_names', []) or []) == shared_feature_names
        )
        if not shared_feature_names:
            owner.sub_tokenizer = None
        elif not sub_matches:
            owner.sub_tokenizer = prev_owner.tokenizer.extract_subtokenizer(shared_feature_names).to(self.args.device)

        if not new_numeric:
            owner.extra_tokenizer = None
        elif owner.extra_tokenizer is None:
            owner.extra_tokenizer = new_tokenizer
        elif new_tokenizer is not None:
            owner.extra_tokenizer.load_state_dict(new_tokenizer.state_dict())
    
    def merge_tokenizers(self):
        if self.args.stage == 'stage_1':
            return None
        owner_copy = copy.deepcopy(self.model.module if hasattr(self.model, 'module') else self.model)
        if hasattr(owner_copy, '_merge'):
            owner_copy._merge()
        return owner_copy

    def init_model_optimizer(self):
        owner = self.model.module if hasattr(self.model, 'module') else self.model

        for name, param in owner.named_parameters():
            if 'time_tokenizer' in name or 'temporal_embeddings' in name:
                if self.args.stage == 'stage_1':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for n, _ in owner.named_children():
            self.logger.debug(f'{n}')
        self.logger.debug(list(p for n, p in owner.named_children() if 'head' in n))

        fea_params = []
        other_fea_params = []
        for name, param in owner.named_parameters():
            if not param.requires_grad:
                continue
                
            if any(key in name for key in ['head', 'norm', 'tokenizer', 'temporal_embeddings']):
                continue
                
            if 'W_out' in name:
                fea_params.append(param)
            else:
                other_fea_params.append(param)

        cls_params_all = list(p for n, p in owner.named_children() if 'head' in n)[0]
        cls_params = [p for p in cls_params_all.parameters() if p.requires_grad]
        
        norm_params = [p for n, p in owner.named_parameters() if 'norm' in n and p.requires_grad]
        
        if self.args.stage == 'stage_1':
            tokenizer_params = [
                p for n, p in owner.named_parameters()
                if 'tokenizer' in n and p.requires_grad
            ]
            time_params = [
                p for n, p in owner.named_parameters() 
                if 'temporal_embeddings' in n and p.requires_grad
            ]
        else:
            tokenizer_params = [
                p for n, p in owner.named_parameters()
                if ('sub_tokenizer' in n or 'extra_tokenizer' in n) and p.requires_grad
            ]
            time_params = [
                p for n, p in owner.named_parameters() 
                if ('temporal_embeddings' in n or 'time_tokenizer' in n) and p.requires_grad
            ]

        # ==========================================================
        # ==========================================================
        self.logger.debug("=== Optimizer Groups Verification ===")
        
        if self.args.stage == 'stage_1':
            time_names = [n for n, p in owner.named_parameters() if 'temporal_embeddings' in n and p.requires_grad]
            tokenizer_names = [n for n, p in owner.named_parameters() if 'tokenizer' in n and p.requires_grad]
        else:
            time_names = [n for n, p in owner.named_parameters() if ('temporal_embeddings' in n or 'time_tokenizer' in n) and p.requires_grad]
            tokenizer_names = [n for n, p in owner.named_parameters() if ('sub_tokenizer' in n or 'extra_tokenizer' in n) and p.requires_grad]

        self.logger.debug(f"[Time Params] (Count: {len(time_names)}): {time_names}")

        self.logger.debug(f"[Tokenizer Params] (Count: {len(tokenizer_names)}): {tokenizer_names}")
        
        norm_names = [n for n, p in owner.named_parameters() if 'norm' in n and p.requires_grad]
        self.logger.debug(f"[Norm Params] (Count: {len(norm_names)}): {norm_names}")

        cls_names = [n for n, p in cls_params_all.named_parameters() if p.requires_grad]
        self.logger.debug(f"[CLS Params] (Count: {len(cls_names)}): {cls_names}")
        
        fea_names = [n for n, p in owner.named_parameters() if 'W_out' in n and not any(key in n for key in ['head', 'norm', 'tokenizer', 'temporal_embeddings']) and p.requires_grad]
        self.logger.debug(f"[Fea SVD Params] (Count: {len(fea_names)}): {fea_names}")
        
        self.logger.debug(f"[Other Fea Params] Count: {len(other_fea_params)}")
        
        self.logger.debug("=====================================")

        # ==========================================================
        # ==========================================================
        optimizer_param_groups = [
            {
                'params': fea_params,
                'svd': True,
                'lr': self.config['training']['lr'],
                'thres': self.config['training']['svd_thres'],
                'num_eigen': self.config['training']['num_eigen'],
            },
            {
                'params': other_fea_params,
                'weight_decay': self.config['training']['model_weight_decay'],
                'lr': self.config['training']['lr'],
            },
            {
                'params': cls_params,
                'weight_decay': 0.0,
                'lr': self.config['training']['head_lr'],
            },
            {
                'params': norm_params,
                'lr': self.config['training']['bn_lr'],
            },
            {
                'params': tokenizer_params,
                'lr': self.config['training']['tokenizer_lr'],
                'name': 'tokenizer',
            },
            {
                'params': time_params,
                'lr': self.config['training'].get('time_lr', self.config['training']['time_lr']),
                'name': 'time',
            },
        ]

        optimizer_param_groups = [g for g in optimizer_param_groups if len(g['params']) > 0]        
        
        model_optimizer_arg = {
            'params': optimizer_param_groups,
            'lr': self.config['training']['lr'],
            'weight_decay': self.config['training']['model_weight_decay'],
        }

        model_optimizer_name = self.config['training'].get('model_optimizer', 'Adam')
        if model_optimizer_name in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['training']['momentum']
        elif model_optimizer_name in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif model_optimizer_name in ['amsgrad']:
            model_optimizer_arg['amsgrad'] = True
            self.config['training']['model_optimizer'] = 'Adam'

        self.lambda_deviation = self.config['training'].get('lambda_deviation', 0.0)
        self.lambda_distill = self.config['training'].get('lambda_distill', 0.0)

        if self.has_prev_data():
            model_optimizer_arg['apply_max_sv_clip'] = self.apply_max_sv_clip
            model_optimizer_arg['clipping_ratio_alpha'] = self.clipping_ratio_alpha
            from model.null_space_optimizer_global_clipping import Adam
            self.optimizer = Adam(**model_optimizer_arg)
            self.optimizer.logger = self.logger
            if self.logger is not None:
                self.logger.debug(f'self.optimizer.apply_max_sv_clip = {self.optimizer.apply_max_sv_clip}')
                self.logger.debug(f'self.optimizer.clipping_ratio_alpha = {self.optimizer.clipping_ratio_alpha}')
        else:
            self.optimizer = optim.Adam(**model_optimizer_arg)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.max_epoch,
            eta_min=1e-5,
        )


    def on_before_training_loop(self):
        if not self.has_prev_data():
            if self.logger is not None:
                self.logger.debug('no previous data ... ')
            return

        self.stats = {}
        with torch.no_grad():
            self.required_NSP_modules_list, self.required_NSP_modules_name = self.update_optim_transforms(self.train_loader_prev_D)
            if self.logger is not None:
                self.logger.debug('successfully reset the transforms ... ')

        if self.logger is not None:
            self.logger.debug(f'required_NSP_modules_name: {self.required_NSP_modules_name}')

        self.reset_model_tokenizer_status(
            new_numeric=self.new_numeric_features,
            shared_feature_names=self.shared_numeric_features,
            tokenizer_feature_names=None,
            current_stage_feature_names=self.D_meta['numeric_features'],
        )
        if self.logger is not None:
            self.logger.debug('reset model tokenizer, now can use sub and extra tokenizers ... ')

        self.register_upperbound_hooks()

    def on_before_optimizer_step(self, tokenizer_owner, batch_index, batch):
        if not self.has_prev_data():
            return

        sub_tokenizer = getattr(tokenizer_owner, 'sub_tokenizer', None)
        if (
            self.keep_ratio is not None
            and self.keep_ratio != 1.0
            and sub_tokenizer is not None
        ):
            self.LTH_param(
                sub_tokenizer,
                keep_ratio=self.keep_ratio
            )

        # self._freeze_shared_tokenizer_grads(
        #     tokenizer_owner,
        #     tokenizer_attr='sub_tokenizer',
        #     shared_features_attr='shared_numeric_features',
        # )

        self.optimizer.upperbound_stats = self.upperbound_calculation()

    def register_upperbound_hooks(self):
        handles = []
        if self.logger is not None:
            self.logger.debug("Registering upperbound hooks ...")
        for m, name in zip(self.required_NSP_modules_list, self.required_NSP_modules_name):
            if isinstance(m, nn.Linear):
                if self.logger is not None:
                    self.logger.debug(f"[upperbound] Registering hook for layer: {name}")
                h = m.register_forward_hook(self.make_upperbound_hook(name))
                handles.append(h)
        self._upperbound_handles = handles
        
        self.logger.debug(f"[upperbound] Registered {len(handles)} forward hooks.")
        return handles

    def make_upperbound_hook(self, name):
        def hook(module, fea_in, fea_out):
            return self.upperbound_update(name, module, fea_in, fea_out)
        return hook

    def _ensure_stat_entry(self, key):
        if key not in self.stats:
            self.stats[key] = {
                'sum_E_norm_Wx': 0.0,
                'sum_norm_E_x': 0.0,
                'sum_E_norm_x2': 0.0,
                'count_samples': 0,
                'count_batches': 0,
            }

    def update_cov(self, fea_in_, k, module_name='unknown'):
        if len(fea_in_.shape) == 3:
            cov = torch.mm(fea_in_.mean(1).transpose(0, 1), fea_in_.mean(1))
        else:
            cov = torch.mm(fea_in_.transpose(0, 1), fea_in_)

        if len(self.fea_in.get(k, [])) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] + cov

    def upperbound_update(self, name, module, fea_in, fea_out):
        x_in = fea_in[0].detach()
        key = name
        self._ensure_stat_entry(key)

        if x_in.dim() == 3:
            x_bar = x_in.mean(dim=1, keepdim=False)
        elif x_in.dim() == 2:
            x_bar = x_in
        else:
            x_bar = x_in.view(x_in.size(0), -1)
        B, _ = x_bar.shape

        Ex = x_bar.mean(dim=0)
        norm_Ex = Ex.norm(p=2).item()
        E_norm_x2 = l2_sq_sum_per_row(x_bar).mean().item()

        Wx_bar = fea_out.mean(dim=1)
        E_norm_Wx = Wx_bar.norm(p=2, dim=1).mean().item()

        self.stats[key]['sum_E_norm_Wx'] += E_norm_Wx * B
        self.stats[key]['sum_norm_E_x'] += norm_Ex
        self.stats[key]['sum_E_norm_x2'] += E_norm_x2 * B
        self.stats[key]['count_samples'] += B
        self.stats[key]['count_batches'] += 1

    def upperbound_calculation(self):
        tr_stats = {}
        for key, s in self.stats.items():
            cs = max(1, s['count_samples'])
            cb = max(1, s['count_batches'])

            E_norm_Wx = s['sum_E_norm_Wx'] / cs
            norm_E_x = s['sum_norm_E_x'] / cb
            E_norm_x2 = s['sum_E_norm_x2'] / cs
            tr_cov = max(0.0, E_norm_x2 - (norm_E_x ** 2))

            denom = norm_E_x + (tr_cov / max(self.ub_delta, 1e-12)) ** 0.5
            denom = max(denom, 1e-12)
            upperbound_1 = self.ub_p * E_norm_Wx / denom * (1 - self.ub_epsilon_ratio)

            tr_stats[key] = {
                'E[||W x_bar||_2]': E_norm_Wx,
                '||E[x_bar]||_2': norm_E_x,
                'E[||x_bar||_2^2]': E_norm_x2,
                'tr(Sigma)': tr_cov,
                'upperbound_1': upperbound_1,
                'samples(B-sum)': s['count_samples'],
                'batches': s['count_batches'],
            }

        return tr_stats

    @torch.no_grad()
    def compute_cov(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear):
            self.update_cov(torch.mean(fea_in[0], 1, False).detach(), module.weight)
        torch.cuda.empty_cache()

        return None

    def update_optim_transforms(self, prev_train_loader):
        self.model.eval()
        owner = self.model.module if hasattr(self.model, 'module') else self.model
        modules = [m for n, m in owner.named_modules() if hasattr(m, 'weight') and 'head' not in n and 'tokenizer' not in n and 'temporal_embeddings' not in n and 'norm' not in n]
        modules_names = [n for n, m in owner.named_modules() if hasattr(m, 'weight') and 'head' not in n and 'tokenizer' not in n and 'temporal_embeddings' not in n and 'norm' not in n]
        # modules = [m for n, m in owner.named_modules() if hasattr(m, 'weight') and 'head' not in n and 'tokenizer' not in n and 'temporal_embeddings' not in n and 'norm' not in n and 'linear1' not in n]
        # modules_names = [n for n, m in owner.named_modules() if hasattr(m, 'weight') and 'head' not in n and 'tokenizer' not in n and 'temporal_embeddings' not in n and 'norm' not in n and 'linear1' not in n]
        self.logger.debug(f"modules_names:{modules_names}")

        self.optimizer.NSP_module_name = modules_names

        handles = []
        for m, name in zip(modules, modules_names):
            if isinstance(m, nn.Linear):
                setattr(m, '_nsp_name', name)
                handles.append(m.register_forward_hook(hook=self.compute_cov))

        original_tokenizer = None
        prev_stage_feature_names = []
        if self.stage_data is not None and self.stage_index is not None and self.stage_index > 0:
            prev_stage_feature_names = self.stage_data.num_feature_names_of_stage(self.stage_index - 1)
        tokenizer_feature_names = list(getattr(owner.tokenizer, 'feature_names', []) or [])
        if prev_stage_feature_names and tokenizer_feature_names != prev_stage_feature_names:
            original_tokenizer = owner.tokenizer
            cached_prev_tokenizer = getattr(owner, 'second_tokenizer', None)
            if cached_prev_tokenizer is not None and list(getattr(cached_prev_tokenizer, 'feature_names', []) or []) == list(prev_stage_feature_names):
                owner.tokenizer = cached_prev_tokenizer
            else:
                owner.tokenizer = original_tokenizer.extract_subtokenizer(prev_stage_feature_names).to(self.args.device)

        torch.cuda.empty_cache()
        try:
            for i, (inputs, dt, target) in enumerate(prev_train_loader):
                inputs = inputs.to(self.args.device, non_blocking=True)
                if i % 100 == 0:
                    self.logger.debug(f'{i}/{len(prev_train_loader)} {inputs.size(1)}')
                self.model.forward(x_num=inputs, x_cat=None, dt=None, go_sub_and_extra=False)
        finally:
            if original_tokenizer is not None:
                owner.tokenizer = original_tokenizer
        # print(f'len of self.fea_in = {len(self.fea_in)}')

        for key, value in self.fea_in.items():
            value /= len(prev_train_loader.dataset)
            # print(f'key.shape = {key.shape}, value.shape = {value.shape}')

        self.optimizer.get_eigens(self.fea_in)
        self.optimizer.get_transforms()
        for h in handles:
            h.remove()

        self.fea_in.clear()
        torch.cuda.empty_cache()

        return modules, modules_names

    def LTH_param(self, tokenizer, keep_ratio=0.5, eps=1e-10):
        if tokenizer is None or not hasattr(tokenizer, 'weight'):
            return

        score_list = []
        scores = []

        weight = tokenizer.weight.data[1:]
        w_grad = tokenizer.weight.grad[1:] if tokenizer.weight.grad is not None else torch.zeros_like(weight)
        w_score = torch.abs(weight * w_grad)
        score_list.append(w_score.view(-1))
        scores.append({'type': 'weight', 'score': w_score})

        if tokenizer.bias is not None:
            bias = tokenizer.bias.data
            b_grad = tokenizer.bias.grad if tokenizer.bias.grad is not None else torch.zeros_like(bias)
            b_score = torch.abs(bias * b_grad)
            score_list.append(b_score.view(-1))
            scores.append({'type': 'bias', 'score': b_score})

        all_scores = torch.cat(score_list)
        num_to_prune = int(all_scores.numel() * keep_ratio)

        non_zero_scores = all_scores[all_scores > 0]
        if non_zero_scores.numel() == 0:
            return
        num_to_nonozero_prune = int(non_zero_scores.numel() * keep_ratio)
        num_non_zero_to_keep = min(num_to_prune, num_to_nonozero_prune)
        if num_non_zero_to_keep <= 0:
            return
        threshold, _ = torch.topk(non_zero_scores, num_non_zero_to_keep, sorted=True)
        cutoff = threshold[-1]

        for entry in scores:
            score = entry['score']
            param_type = entry['type']

            mask = (score >= cutoff).float()

            if param_type == 'weight':
                if tokenizer.weight.grad is not None:
                    mask_reshaped = mask.view_as(tokenizer.weight.grad[1:])
                    tokenizer.weight.grad[1:] *= mask_reshaped
            elif param_type == 'bias':
                if tokenizer.bias is not None and tokenizer.bias.grad is not None:
                    mask_reshaped = mask.view_as(tokenizer.bias.grad)
                    tokenizer.bias.grad *= mask_reshaped

    def collect_feature(self, data_loader) -> torch.Tensor:
        self.model.eval()
        all_features = []
        with torch.no_grad():
            for _, (X_num, Y_num) in tqdm(enumerate(data_loader)):
                X_num = X_num.to(self.args.device, non_blocking=True)
                feature = self.model(X_num, None, ret_feature=True).cpu()
                all_features.append(feature)
        return torch.cat(all_features, dim=0)

    def a_distance(self, source_feature: torch.Tensor, target_feature: torch.Tensor, progress=True, training_epochs=100):
        dtype = self._torch_dtype()
        source_label = torch.ones((source_feature.shape[0], 1), dtype=dtype, device=source_feature.device)
        target_label = torch.zeros((target_feature.shape[0], 1), dtype=dtype, device=target_feature.device)
        feature = torch.cat([source_feature, target_feature], dim=0)
        label = torch.cat([source_label, target_label], dim=0)

        dataset = TensorDataset(feature, label)
        length = len(dataset)
        train_size = int(0.8 * length)
        val_size = length - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=False)

        anet = ANet(feature.shape[1]).to(self.args.device, dtype=dtype)
        optimizer = torch.optim.SGD(anet.parameters(), lr=0.01)
        a_distance = 2.0
        for epoch in range(training_epochs):
            anet.train()
            for (x, label) in train_loader:
                x = x.to(self.args.device, dtype=dtype)
                label = label.to(self.args.device, dtype=dtype)
                anet.zero_grad()
                y = anet(x)
                loss = F.binary_cross_entropy(y, label)
                loss.backward()
                optimizer.step()

            anet.eval()
            meter = AverageMeter("accuracy", ":4.2f")
            with torch.no_grad():
                for (x, label) in val_loader:
                    x = x.to(self.args.device, dtype=dtype)
                    label = label.to(self.args.device, dtype=dtype)
                    y = anet(x)
                    acc = self.binary_accuracy(y, label)
                    meter.update(acc, x.shape[0])
            error = 1 - meter.avg / 100
            a_distance = 2 * (1 - 2 * error)
            if progress:
                self.logger.debug("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

        return a_distance

    def binary_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            batch_size = target.size(0)
            dtype = self._torch_dtype()
            pred = (output >= 0.5).to(dtype=dtype).t().view(-1)
            correct = pred.eq(target.view(-1)).to(dtype=dtype).sum()
            correct = correct * (100.0 / batch_size)
            return float(correct.item())
