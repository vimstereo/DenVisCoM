import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (single_head_full_attention, single_head_split_window_attention,
                        single_head_full_attention_1d, single_head_split_window_attention_1d)
from .utils import generate_shift_window_attn_mask, generate_shift_window_attn_mask_1d
from DenVisCor.nvim import mamba_vision_T
from pathlib import Path
class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # for stereo: 2d attn in self-attn, 1d attn in cross-attn
        is_self_attn = (query - key).abs().max() < 1e-6

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        if attn_type == 'swin' and attn_num_splits > 1:  # self, cross-attn: both swin 2d
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask,
                                                             )

        elif attn_type == 'self_swin2d_cross_1d':  # self-attn: swin 2d, cross-attn: full 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]

                else:
                    # cross attn 1d
                    message = single_head_full_attention_1d(query, key, value,
                                                            h=height,
                                                            w=width,
                                                            )

        elif attn_type == 'self_swin2d_cross_swin1d':  # self-attn: swin 2d, cross-attn: swin 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        # self attn shift window
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]
                else:
                    if attn_num_splits > 1:
                        assert shifted_window_attn_mask_1d is not None
                        # cross attn 1d shift
                        message = single_head_split_window_attention_1d(query, key, value,
                                                                        num_splits=attn_num_splits,
                                                                        with_shift=with_shift,
                                                                        h=height,
                                                                        w=width,
                                                                        attn_mask=shifted_window_attn_mask_1d,
                                                                        )
                    else:
                        message = single_head_full_attention_1d(query, key, value,
                                                                h=height,
                                                                w=width,
                                                                )

        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_type=attn_type,
                                with_shift=with_shift,
                                attn_num_splits=attn_num_splits,
                                )

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                                     attn_type=attn_type,
                                     with_shift=with_shift,
                                     attn_num_splits=attn_num_splits,
                                     )

        return source


class FeatureTransformer(nn.Module):
    def __init__(self,
                 attn,
                 interweaving,
                 depth,
                 heads,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
    ):
        super(FeatureTransformer, self).__init__()

        self.mamba_vision = mamba_vision_T(attn=attn,interweaving=interweaving,depth=depth,heads=heads)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self,feature0,feature1,attn_type='swin',attn_num_splits=None,**kwargs):
       
        concat0 = torch.cat((feature0, feature1), dim=0)

        concat0 = self.mamba_vision(concat0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)
        return feature0, feature1
