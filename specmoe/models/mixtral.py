# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modifications Copyright 2025

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
"""Inference-only Mixtral model."""

from typing import Iterable, Optional, Tuple, List, Union
import logging
logger = logging.getLogger(__name__)

import torch
from torch import nn
from transformers import MixtralConfig
import tqdm

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from specmoe.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, set_weight_attrs


from specmoe.backend.forward_batch_info import ForwardBatch, DecodePart
from specmoe.layers.FusedMoE import FusedMoE
from specmoe.layers.attention import RadixAttention



class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        pin_experts_when_load_weight: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            renormalize=True,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=add_prefix("experts", prefix),
            pin_experts_when_load_weight=pin_experts_when_load_weight,
        )

    def forward(self, hidden_states: torch.Tensor, experts_mapping: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits, experts_mapping)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(orig_shape)


class MixtralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        is_decode_gpu_attn: bool = False,
        micro_batch_id: int = -1,
        nano_batch_id: int = -1,
        gpu_attn_init_slot: int = -1, # init_forward_metadata中，当前nanobatch对应的slot编号
        gpu_attn_nano_batch_slice: Tuple[int, int] = None,
    ) -> torch.Tensor:
        if forward_batch.decode_part == DecodePart.ALL or forward_batch.decode_part == DecodePart.PREATTN:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # logger.debug(f"positions: {positions}")
            q, k = self.rotary_emb(positions, q, k)
            if forward_batch.decode_part == DecodePart.PREATTN:
                return torch.cat([q, k, v], dim=-1).contiguous()
            
        if forward_batch.decode_part == DecodePart.ALL:
            attn_output = self.attn(q, k, v, forward_batch, micro_batch_id=micro_batch_id)
            forward_batch.attn_event.record(torch.cuda.current_stream()) # for prefill stage
        if forward_batch.decode_part == DecodePart.CPU_ATTN: 
            if is_decode_gpu_attn: # TODO: 切出此nano batch的部分
                qkv = forward_batch.qkv_gpu.view(-1, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
                # 切分出GPU执行的部分
                nano_slice = slice(gpu_attn_nano_batch_slice[0]*forward_batch.gpu_attn_per_request_input_ids_size, gpu_attn_nano_batch_slice[1]*forward_batch.gpu_attn_per_request_input_ids_size)
                q = qkv[nano_slice, :self.num_heads, :].contiguous()
                k = qkv[nano_slice, self.num_heads : self.num_heads + self.num_kv_heads, :].contiguous()
                v = qkv[nano_slice, self.num_heads + self.num_kv_heads :, :].contiguous()
                attn_output = self.attn(q, k, v, forward_batch, is_decode_gpu_attn=is_decode_gpu_attn, micro_batch_id=micro_batch_id, nano_batch_id=nano_batch_id, gpu_attn_init_slot=gpu_attn_init_slot)
                return attn_output
            else:
                attn_output = self.attn(None, None, None, forward_batch, micro_batch_id=micro_batch_id) # QKV are in forward_batch.qkv_pin, so there is no need to pass them
                return attn_output
        if forward_batch.decode_part == DecodePart.ALL or forward_batch.decode_part == DecodePart.POSTATTN:
            if forward_batch.decode_part == DecodePart.POSTATTN:
                attn_output = hidden_states
            output, _ = self.o_proj(attn_output)
        return output

class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.block_sparse_moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            pin_experts_when_load_weight=config.pin_experts_when_load_weight,
            prefix=add_prefix("block_sparse_moe", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        is_decode_gpu_attn: bool = False,
        micro_batch_id: int = -1,
        nano_batch_id: int = -1,
        gpu_attn_init_slot: int = -1, # init_forward_metadata中，当前nanobatch对应的slot编号
        gpu_attn_nano_batch_slice: Tuple[int, int] = None,
    ) -> torch.Tensor:
        # logger.debug(f"layer {self.layer_id}, residual: {residual}")
        if forward_batch.decode_part == DecodePart.ALL or forward_batch.decode_part == DecodePart.PREATTN:
            if residual is None:
                assert self.layer_id == 0
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
        
        if forward_batch.decode_part == DecodePart.PREATTN:
            qkv = self.self_attn(positions, hidden_states, forward_batch)
            # if self.layer_id <= 3:
            #     print(f"layer {self.layer_id}, hidden_states before self_attn: {hidden_states[:, :4]}, (shape: {hidden_states.shape})")
            return qkv, residual
        elif forward_batch.decode_part == DecodePart.CPU_ATTN or is_decode_gpu_attn: # core attention
            attn_out = self.self_attn(positions, hidden_states, forward_batch, is_decode_gpu_attn=is_decode_gpu_attn, micro_batch_id=micro_batch_id, nano_batch_id=nano_batch_id, gpu_attn_init_slot=gpu_attn_init_slot, gpu_attn_nano_batch_slice=gpu_attn_nano_batch_slice)
            # if self.layer_id <= 3:
            #     print(f"layer {self.layer_id}, hidden_states after core attention: {attn_out.reshape(-1, 4096)[:, :4]}, (shape: {attn_out.shape})")
            return attn_out
        elif forward_batch.decode_part == DecodePart.ALL or forward_batch.decode_part == DecodePart.POSTATTN:
            # Self Attention
            # if self.layer_id <= 3:
            #     print(f"layer {self.layer_id}, hidden_states before attention: {hidden_states}")
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                micro_batch_id=micro_batch_id,
            )
            # if self.layer_id <= 0:
            #     print(f"layer {self.layer_id}, hidden_states after self_attn: {hidden_states.reshape(-1, 194, 4096)}, (shape: {hidden_states.shape})")
            # logger.debug(f"layer {self.layer_id}, hidden_states after attention: {hidden_states}")
            # Fully Connected
            # print(hidden_states.shape, residual.shape)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            # if self.layer_id <= 0:
            #     print(f"layer {self.layer_id}, hidden_states before moe: {hidden_states.reshape(-1, 194, 4096)}, (shape: {hidden_states.shape})")
            hidden_states = self.block_sparse_moe(hidden_states, forward_batch.experts_mapping)
            # logger.debug(f"experts_mapping: {forward_batch.experts_mapping}")
            # if self.layer_id <= 0:
            #     print(f"layer {self.layer_id}, hidden_states after moe: {hidden_states.reshape(-1, 194, 4096)}")
        
        return hidden_states, residual
        # # Self Attention
        # if residual is None:
        #     residual = hidden_states
        #     hidden_states = self.input_layernorm(hidden_states)
        # else:
        #     hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # hidden_states = self.self_attn(
        #     positions=positions,
        #     hidden_states=hidden_states,
        #     forward_batch=forward_batch,
        # )

        # # Fully Connected
        # hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # hidden_states = self.block_sparse_moe(hidden_states)
        # return hidden_states, residual


class MixtralModel(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MixtralDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: List[int] = None,
        is_decode_gpu_attn: bool = False,
        micro_batch_id: int = -1,
        nano_batch_id: int = -1,
        gpu_attn_init_slot: int = -1, # init_forward_metadata中，当前nanobatch对应的slot编号
        gpu_attn_nano_batch_slice: Tuple[int, int] = None,
    ) -> torch.Tensor:
        if cur_layers is None:
            cur_layers = range(len(self.layers))
        if hidden_states is None and input_ids is not None and forward_batch.decode_part != DecodePart.CPU_ATTN:
            assert 0 in cur_layers
            hidden_states = self.embed_tokens(input_ids)
        for i in cur_layers:
            layer = self.layers[i]
            if forward_batch.decode_part == DecodePart.ALL or forward_batch.decode_part == DecodePart.POSTATTN:
                hidden_states, residual = layer(positions, hidden_states,
                                                forward_batch,
                                                residual,
                                                micro_batch_id=micro_batch_id)
            elif forward_batch.decode_part == DecodePart.PREATTN:
                qkv, residual = layer(positions, hidden_states,
                                                forward_batch,
                                                residual)
                return qkv, residual
            elif forward_batch.decode_part == DecodePart.CPU_ATTN or is_decode_gpu_attn:
                attn_out = layer(None, None, forward_batch, None, is_decode_gpu_attn=is_decode_gpu_attn, micro_batch_id=micro_batch_id, nano_batch_id=nano_batch_id, gpu_attn_init_slot=gpu_attn_init_slot, gpu_attn_nano_batch_slice=gpu_attn_nano_batch_slice)
                return attn_out
        if cur_layers[-1] != len(self.layers) - 1:
            return hidden_states, residual
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states, _


class MixtralForCausalLMOff(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None, 
        cur_layers: List[int] = None,
        is_decode_gpu_attn: bool = False,
        micro_batch_id: int = -1,
        nano_batch_id: int = -1,
        gpu_attn_init_slot: int = -1, # init_forward_metadata中，当前nanobatch对应的slot编号
        gpu_attn_nano_batch_slice: Tuple[int, int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
        if forward_batch.decode_part == DecodePart.ALL or forward_batch.decode_part == DecodePart.POSTATTN: 
            if hidden_states is not None: 
                if len(hidden_states.shape) == 3:
                    hidden_states = hidden_states.view(hidden_states.shape[0], -1)
                elif len(hidden_states.shape) == 4: # 当speculative decoding时q_len>1，导致hidden_states的shape为[batch_size, q_len, num_heads, head_dim]，要变为[batch_size*q_len, num_heads*head_dim]
                    hidden_states = hidden_states.view(hidden_states.shape[0] * hidden_states.shape[1], -1)
                elif len(hidden_states.shape) == 2:
                    pass
                else:
                    assert False
            hidden_states, residual = self.model(input_ids, positions,
                                    forward_batch, input_embeds=input_embeds, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers, micro_batch_id=micro_batch_id)
        elif forward_batch.decode_part == DecodePart.PREATTN: # first norm + QKV calc + RoPE
            qkv, residual = self.model(input_ids, positions, forward_batch, input_embeds=input_embeds, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return qkv, residual
        elif forward_batch.decode_part == DecodePart.CPU_ATTN or is_decode_gpu_attn: # core attention 
            attn_out = self.model(input_ids, positions,
                                    forward_batch, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers, is_decode_gpu_attn=is_decode_gpu_attn, micro_batch_id=micro_batch_id, nano_batch_id=nano_batch_id, gpu_attn_init_slot=gpu_attn_init_slot, gpu_attn_nano_batch_slice=gpu_attn_nano_batch_slice)
            return attn_out
        
        if self.config.num_hidden_layers - 1 not in cur_layers: # not the last layer
            return hidden_states, residual
        else:
            forward_batch.return_logprob = False
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head.weight, forward_batch
            )
    
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        # print(f"expert_params_mapping: {expert_params_mapping}")

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in tqdm.tqdm(weights):
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    if name is None:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        
    def get_experts_cache(self, layer_id: int):
        return self.model.layers[layer_id].block_sparse_moe.experts.ws.data
    
    def link_gpu_experts_cache(self, expert_pool):
        for i in range(self.config.num_hidden_layers):
            set_weight_attrs(self.model.layers[i].block_sparse_moe.experts.ws, {"gpu_cache": expert_pool})
            # setattr(self.model.layers[i].block_sparse_moe.experts.ws, "gpu_cache", expert_pool)
    
    def del_experts_cache(self):
        for i in range(self.config.num_hidden_layers):
            # setattr(weight, key, value)
            if hasattr(self.model.layers[i].block_sparse_moe.experts.ws, "gpu_cache"):
                delattr(self.model.layers[i].block_sparse_moe.experts.ws, "gpu_cache")
            # self.model.layers[i].block_sparse_moe.experts.ws.gpu_cache = None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



EntryClass = MixtralForCausalLMOff
