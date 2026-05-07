import os
from typing import Optional, Union
import logging
import torch
from specmoe.utils.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_num_layers,
    get_hidden_size,
    get_num_attention_heads,
    get_num_kv_heads,
    get_intermediate_size,
    get_num_experts,
    get_topk,
    get_eos_token_id,
)
from specmoe.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(
        self,
        path: str,
        is_draft_model: bool = False,
        spec_algorithm: Optional[SpeculativeAlgorithm] = None,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        pin_experts_when_load_weight: bool = False,
    ) -> None:
        self.model_path = path
        self.is_draft_model = is_draft_model
        self.spec_algorithm = spec_algorithm
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.hf_config = get_config(self.model_path, trust_remote_code, revision, is_draft_model, spec_algorithm)
        self.hf_config.pin_experts_when_load_weight = pin_experts_when_load_weight
        # logger.info(f"hf_config: {self.hf_config}")

        # Unify the config keys for hf_config
        self.context_length = context_length if context_length is not None else get_context_length(self.hf_config)
        self.head_dim =  get_hidden_size(self.hf_config) // get_num_attention_heads(self.hf_config)
        self.num_attention_heads = get_num_attention_heads(self.hf_config)
        self.num_key_value_heads = get_num_kv_heads(self.hf_config)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = get_hidden_size(self.hf_config)
        self.num_hidden_layers = get_num_layers(self.hf_config)
        self.vocab_size = self.hf_config.vocab_size
        self.num_local_experts = get_num_experts(self.hf_config)
        self.intermediate_size = get_intermediate_size(self.hf_config)
        self.topk = get_topk(self.hf_config)
        self.eos_token_id = get_eos_token_id(self.hf_config)
        self.dtype = torch.float16

    def get_expert_total_size(self) -> float: # assume in float16, in GB
        return self.num_local_experts * self.num_hidden_layers * (self.hidden_size * self.intermediate_size * 3) * 2 / 1024 / 1024 / 1024 # in GB
    
    def get_per_token_per_layer_kv_size(self) -> float: # assume in float16, in GB
        return 2 * 2 * self.head_dim * self.num_key_value_heads / 1024 / 1024 / 1024 # in GB
    
    def get_all_size(self) -> float: # assume in float16, in GB
        # expert
        res = self.get_expert_total_size()
        # preattention, matmul for QKV
        res += self.num_hidden_layers * (self.hidden_size * self.num_attention_heads * self.head_dim * 2 + self.hidden_size * self.num_key_value_heads * self.head_dim) * 2 / 1024 / 1024 / 1024 # in GB
        # gate
        res += self.num_hidden_layers * (self.hidden_size * self.num_local_experts) * 2 / 1024 / 1024 / 1024 # in GB
        return res

