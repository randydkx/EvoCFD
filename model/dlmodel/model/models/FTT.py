import math
import os
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from ..lib.temporal_embeddings import TemporalEmbeddings
# Source: https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py
def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def get_nonglu_activation_fn(name):
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )


def get_activation_fn(name):
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )

class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x

class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
        feature_names: ty.List[str],
        cls_token: bool = True,
    ) -> None:
        super().__init__()
        
        self.d_token = d_token
        if categories is None or len(categories) == 0:
            self.d_numerical = d_numerical
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.cls = cls_token
        self.feature_names = feature_names
        self.feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}
        input_dim = d_numerical + (1 if cls_token else 0)

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(input_dim, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def extract_subtokenizer(self, feature_names_subset: ty.List[str]) -> "Tokenizer":
        if not feature_names_subset:
            raise ValueError("feature_names_subset must not be empty when extracting subtokenizer")
        indices = [self.feature_name_to_idx[name] for name in feature_names_subset]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        new_tokenizer = Tokenizer(
            d_numerical=len(feature_names_subset),
            categories=None,
            d_token=self.weight.shape[1],
            feature_names=feature_names_subset,
            bias=self.bias is not None,
        )

        new_tokenizer.weight = nn.Parameter(
            torch.cat([
                self.weight[0:1],
                self.weight[1 + indices_tensor]
            ], dim=0)
        )

        if self.bias is not None:
            new_tokenizer.bias = nn.Parameter(self.bias[indices_tensor])

        return new_tokenizer

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        parts = []
        if self.cls:
            parts.append(torch.ones(len(x_num), 1, device=x_num.device))
        parts.append(x_num)
        x_num = torch.cat(parts, dim=1)
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            if self.cls:
                bias = torch.cat(
                    [
                        torch.zeros(1, self.bias.shape[1], device=x.device),
                        self.bias,
                    ],
                    dim=0
                )
            else:
                bias = self.bias
            x = x + bias[None]
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]] = None,
        token_bias: bool,
        new_feature_names: ty.List[str] = None,
        shared_feature_names: ty.Optional[ty.List[str]] = None,
        tokenizer_feature_names: ty.Optional[ty.List[str]] = None,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
        temporal_embeddings: ty.Optional[dict],
        # extra_tokenizer: ty.Optional[Tokenizer] = None,
        # sub_tokenizer: ty.Optional[Tokenizer] = None,
        logger = None
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        shared_feature_names = shared_feature_names or []
        new_feature_names = new_feature_names or []
        self.logger = logger
        
        self.extra_tokenizer: Tokenizer = None
        self.sub_tokenizer: Tokenizer = None
        # When enabled, the staged forward path (go_sub_and_extra=True) will ignore
        # the extra/new-feature tokenizer, while keeping tokenizer modules intact for
        # checkpoint compatibility.
        self.only_use_shared_features: bool = False
        
        self.update_feature_set(new_feature_names, shared_feature_names, tokenizer_feature_names, current_stage_feature_names=tokenizer_feature_names)

        self.temporal_embeddings = TemporalEmbeddings(**temporal_embeddings)
        temporal_dim = self.temporal_embeddings.out_dim
        
        self.tokenizer = Tokenizer(
                d_numerical=d_numerical,
                categories=categories,
                d_token=d_token,
                bias=token_bias,
                feature_names=self.tokenizer_feature_names,
            )

        self.time_tokenizer = Tokenizer(
            d_numerical=temporal_dim,
            categories=None,
            d_token=d_token,
            bias=token_bias,
            feature_names=[],
            cls_token=False
        )

        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _merge(self):
        if self.extra_tokenizer is None:
            return

        combined_features = list(self.tokenizer.feature_names) + list(self.extra_tokenizer.feature_names)
        new_weight = torch.cat([
            self.tokenizer.weight[0:1],
            self.tokenizer.weight[1:],
            self.extra_tokenizer.weight,
        ], dim=0)

        new_tokenizer = Tokenizer(
            d_numerical=len(combined_features),
            categories=None,
            d_token=self.tokenizer.weight.shape[1],
            feature_names=combined_features,
            bias=self.tokenizer.bias is not None,
        ).to(new_weight.device)

        new_tokenizer.weight = nn.Parameter(new_weight)
        if self.tokenizer.bias is not None:
            new_tokenizer.bias = nn.Parameter(
                torch.cat([self.tokenizer.bias, self.extra_tokenizer.bias], dim=0)
            )

        self.update_feature_set(
            None,
            None,
            tokenizer_feature_names=combined_features,
            current_stage_feature_names=combined_features,
        )

        self.tokenizer = new_tokenizer
        self.sub_tokenizer = None
        self.extra_tokenizer = None

    def safe_merge_tokenizers(self, prev_tokenizer, device):
        
        if self.sub_tokenizer is None or self.extra_tokenizer is None:
            return None
        
        self.logger.debug(f'merging tokenizers: prev_tokenizer with {len(prev_tokenizer.feature_names)} features, sub_tokenizer with {len(self.sub_tokenizer.feature_names)} features, extra_tokenizer with {len(self.extra_tokenizer.feature_names)} features.')

        prev_feature_names = list(prev_tokenizer.feature_names)
        merged_feature_names = prev_feature_names.copy()

        merged_weight = prev_tokenizer.weight.detach().clone().to(device)
        has_bias = prev_tokenizer.bias is not None
        merged_bias = (
            prev_tokenizer.bias.detach().clone().to(device)
            if has_bias and prev_tokenizer.bias is not None
            else None
        )

        if self.sub_tokenizer is not None:
            sub_weight = self.sub_tokenizer.weight.detach().to(device)
            sub_names = list(self.sub_tokenizer.feature_names)
            name_to_idx = {name: idx for idx, name in enumerate(prev_feature_names)}

            # refresh CLS token with the optimized version
            merged_weight[0:1] = sub_weight[0:1]

            if sub_names:
                sub_feature_weights = sub_weight[1:]
                for feat_name, feat_weight in zip(sub_names, sub_feature_weights):
                    prev_idx = name_to_idx.get(feat_name)
                    if prev_idx is None:
                        continue
                    merged_weight[1 + prev_idx] = feat_weight

            if has_bias and self.sub_tokenizer.bias is not None:
                sub_bias = self.sub_tokenizer.bias.detach().to(device)
                for feat_name, feat_bias in zip(sub_names, sub_bias):
                    prev_idx = name_to_idx.get(feat_name)
                    if prev_idx is None:
                        continue
                    merged_bias[prev_idx] = feat_bias

        if self.extra_tokenizer is not None:
            merged_feature_names.extend(self.extra_tokenizer.feature_names)
            extra_weight = self.extra_tokenizer.weight.detach().to(device)
            merged_weight = torch.cat([merged_weight, extra_weight], dim=0)
            if has_bias and self.extra_tokenizer.bias is not None:
                extra_bias = self.extra_tokenizer.bias.detach().to(device)
                merged_bias = torch.cat([merged_bias, extra_bias], dim=0)
        else:
            if not merged_feature_names:
                merged_feature_names = prev_feature_names

        merged = Tokenizer(
            d_numerical=len(merged_feature_names),
            categories=None,
            d_token=merged_weight.shape[1],
            feature_names=merged_feature_names,
            bias=has_bias
        ).to(device)

        merged.weight = nn.Parameter(merged_weight)
        if has_bias and merged_bias is not None:
            merged.bias = nn.Parameter(merged_bias)


        self.update_feature_set(None, None, tokenizer_feature_names=merged_feature_names, current_stage_feature_names=merged_feature_names)

        self.tokenizer = merged
        self.sub_tokenizer = None
        self.extra_tokenizer = None

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def update_feature_set(self, new_feature_names, shared_feature_names, tokenizer_feature_names = None, current_stage_feature_names = None):
        
        # if tokenizer_feature_names is None:
        #     tokenizer_feature_names = shared_feature_names + [name for name in new_feature_names if name not in shared_feature_names]
        
        # self.logger.debug(f'updating feature set: new_feature_names={new_feature_names}')
        # self.logger.debug(f'updating feature set: shared_feature_names={shared_feature_names}')
        # self.logger.debug(f'updating feature set: tokenizer_feature_names={tokenizer_feature_names}')
        # self.logger.debug(f'updating feature set: current_stage_feature_names={current_stage_feature_names}')
        
        self.new_feature_names = new_feature_names
        self.shared_feature_names = shared_feature_names
        self.tokenizer_feature_names = tokenizer_feature_names
        self.current_stage_feature_names = current_stage_feature_names
        
        if shared_feature_names is not None and len(shared_feature_names) > 0:
            self.shared_feature_indices = [self.current_stage_feature_names.index(name) for name in self.shared_feature_names]
        else:
            self.shared_feature_indices = None
        
        if new_feature_names is not None and len(new_feature_names) > 0:
            self.new_feature_indices = [self.current_stage_feature_names.index(name) for name in self.new_feature_names]
        else:
            self.new_feature_indices = None
    
    def forward(
        self,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor] = None,
        dt=None,
        ret_feature: bool = False,
        go_sub_and_extra = False,
    ) -> Tensor:
        token_chunks: ty.List[Tensor] = []
        x_shared = None
        x_new = None
        extra_tokens = None
        used_extra_tokens = None

        # breakpoint()
        
        # if not go_sub_and_extra:
        #     assert self.sub_tokenizer is None and self.extra_tokenizer is None
        
        if go_sub_and_extra:
            if self.sub_tokenizer is not None and self.shared_feature_indices is not None:
                x_shared = x_num[:, self.shared_feature_indices]
                token_chunks.append(self.sub_tokenizer(x_shared, None))

            # In shared-only mode, keep tokenizer modules unchanged for load/strict
            # state_dict compatibility, but do not use the extra tokenizer in forward.
            if not getattr(self, 'only_use_shared_features', False):
                if self.extra_tokenizer is not None and self.new_feature_indices is not None:
                    x_new = x_num[:, self.new_feature_indices]
                    extra_tokens = self.extra_tokenizer(x_new, None)
                    if getattr(self, 'legacy_drop_first_extra_token', False) and extra_tokens.size(1) > 0:
                        used_extra_tokens = extra_tokens[:, 1:]
                    elif token_chunks and getattr(self.extra_tokenizer, 'cls', True) and extra_tokens.size(1) > 0:
                        used_extra_tokens = extra_tokens[:, 1:]
                    else:
                        used_extra_tokens = extra_tokens
                    token_chunks.append(used_extra_tokens)

        if (
            os.environ.get('EVOCFD_DEBUG_STAGE2_FLOW') == '1'
            and go_sub_and_extra
            and not getattr(self, '_debug_stage2_forward_printed', False)
        ):
            token_shapes = [tuple(chunk.shape) for chunk in token_chunks]
            print(
                '[EVOCFD_DEBUG][new][forward] '
                f'current_stage_len={len(self.current_stage_feature_names or [])} | '
                f'tokenizer_len={len(self.tokenizer.feature_names)} | '
                f'shared_len={len(self.shared_feature_names or [])} | '
                f'new_len={len(self.new_feature_names or [])} | '
                f'shared_head={list((self.shared_feature_names or [])[:8])} | '
                f'new_head={list((self.new_feature_names or [])[:8])} | '
                f'shared_idx_head={list((self.shared_feature_indices or [])[:8]) if self.shared_feature_indices is not None else []} | '
                f'new_idx_head={list((self.new_feature_indices or [])[:8]) if self.new_feature_indices is not None else []} | '
                f'x_shared_head={x_shared[0, :min(5, x_shared.shape[1])].detach().cpu().tolist() if x_shared is not None else None} | '
                f'x_new_head={x_new[0, :min(5, x_new.shape[1])].detach().cpu().tolist() if x_new is not None else None} | '
                f'extra_raw_shape={tuple(extra_tokens.shape) if extra_tokens is not None else None} | '
                f'extra_used_shape={tuple(used_extra_tokens.shape) if used_extra_tokens is not None else None} | '
                f'token_chunk_shapes={token_shapes}'
            )
            self._debug_stage2_forward_printed = True
        
        # breakpoint()
        
        if go_sub_and_extra and len(token_chunks) > 0:
            x = torch.cat(token_chunks, dim=1)
        else:
            x = self.tokenizer(x_num, x_cat)

        # breakpoint()

        if dt is not None:
            idx = self.temporal_embeddings(dt[:, -2], dt[:, -1]).flatten(1)
            time_tokens = self.time_tokenizer(idx, None)
            x = torch.cat([x, time_tokens], dim=1)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        if ret_feature:
            return x
        x = self.head(x)
        
        return x.squeeze(-1)