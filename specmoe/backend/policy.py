from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional
import logging
from math import ceil

from specmoe.utils.server_args import ServerArgs
from specmoe.utils.model_config import ModelConfig

logger = logging.getLogger(__name__)

class SpecPolicy(IntEnum):
    NONE = auto()
    SequentialGPUonly = auto()
    SequentialCPUonly = auto()
    SequentialCGCoop = auto()

    @staticmethod
    def from_string(name: str):
        name_map = {
            "SEQUENTIALGPUONLY": SpecPolicy.SequentialGPUonly,
            "SEQUENTIALCPUONLY": SpecPolicy.SequentialCPUonly,
            "SEQUENTIALCGCOOP": SpecPolicy.SequentialCGCoop,
            None: SpecPolicy.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
    
    def is_none(self):
        return self == SpecPolicy.NONE
@dataclass
class Policy:
    # system policy
    global_batch_size: int # 存储在CPU内存中的所有batch，尽可能大地占满CPU内存
    # In decode stage, batchsize = global_batch_size
    batchsize: int # the sum of all micro_batch_size (batchsize = micro_batch_num * micro_batch_size)
    micro_batch_size: int
    micro_batch_num: int

    weight_cache_ratio: float

    gpu_kv_pool_slot_num: int
    draft_gpu_kv_pool_slot_num: int

    gpu_attention_ratio: float

    gpu_attention_micro_batch_size: Optional[int] # only for decoding
    gpu_attention_nano_batch_size: Optional[int] # only for decoding

    spec_policy: SpecPolicy

    draft_model_placement: str = 'GPU'
    draft_gpu_execution_ratio: float = 0
    draft_gpu_req_num: int = 0
    
    stage: str = 'prefill'

    @classmethod
    def get_prefill_policy(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ):
        micro_batch_num = server_args.prefill_micro_batch_num
        micro_batch_size = server_args.prefill_micro_batch_size
        batchsize = micro_batch_num * micro_batch_size
        logger.info(f"get_prefill_policy: micro_batch_num: {micro_batch_num}, micro_batch_size: {micro_batch_size}, batchsize: {batchsize}")

        # gpu_kv_pool_slot_num = int(
        #     (
        #         server_args.gpu_hbm_size * server_args.mem_fraction_static
        #         - model_config.get_expert_total_size()
        #         * server_args.prefill_weight_cache_ratio
        #     ) 
        #     / (model_config.get_per_token_per_layer_kv_size() * 2) # GPU only cache 2 layers
        # )
        gpu_kv_pool_slot_num = micro_batch_size * server_args.max_seq_length
        logger.info(f"get_prefill_policy: gpu_kv_pool_slot_num in prefill stage: {gpu_kv_pool_slot_num}")
        logger.info(f"get_prefill_policy: expert total size: {model_config.get_expert_total_size()}")
        assert gpu_kv_pool_slot_num > 0, "GPU KV cache pool is not enough"
        return cls(
            stage='prefill',
            global_batch_size=server_args.global_batch_size,
            batchsize=batchsize,
            micro_batch_num=micro_batch_num,
            micro_batch_size=micro_batch_size,
            weight_cache_ratio=server_args.prefill_weight_cache_ratio,
            gpu_kv_pool_slot_num=gpu_kv_pool_slot_num,
            draft_gpu_kv_pool_slot_num=gpu_kv_pool_slot_num,
            gpu_attention_ratio = 1,
            gpu_attention_micro_batch_size=micro_batch_size,
            gpu_attention_nano_batch_size=None,
            spec_policy=SpecPolicy.NONE,
            draft_model_placement=server_args.draft_model_placement,
            draft_gpu_execution_ratio=server_args.draft_gpu_execution_ratio,
        )
    
    @classmethod
    def get_decode_policy(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ):
        micro_batch_num = server_args.decode_micro_batch_num
        micro_batch_size = server_args.decode_micro_batch_size
        batchsize = micro_batch_num * micro_batch_size
        assert micro_batch_size == (server_args.global_batch_size + micro_batch_num - 1) // micro_batch_num
        gpu_kv_pool_slot_num = int(2 * server_args.max_seq_length * server_args.decode_gpu_attention_nano_batch_size) + 1 # 防止是0
        draft_gpu_kv_pool_slot_num = int(server_args.max_seq_length * server_args.decode_micro_batch_size * server_args.draft_gpu_execution_ratio) + 1
        assert (server_args.global_batch_size + server_args.decode_micro_batch_num - 1) // server_args.decode_micro_batch_num == server_args.decode_micro_batch_size, "global_batch_size must be equal to decode_micro_batch_size * decode_micro_batch_num"
        assert server_args.decode_gpu_attention_nano_batch_size <= server_args.decode_gpu_attention_micro_batch_size, "decode_gpu_attention_nano_batch_size must be less than or equal to decode_gpu_attention_micro_batch_size"
        assert server_args.draft_gpu_execution_ratio >= server_args.decode_gpu_attention_ratio, "draft_gpu_execution_ratio must be greater than or equal to decode_gpu_attention_ratio"

        draft_gpu_req_num = int(server_args.global_batch_size * server_args.draft_gpu_execution_ratio)

        spec_policy = SpecPolicy.from_string(server_args.decode_spec_policy)
        # if server_args.speculative_algorithm is None:
        #     assert spec_policy.is_none()

        return cls(
            stage='decode',
            global_batch_size=server_args.global_batch_size,
            batchsize=batchsize,
            micro_batch_num=micro_batch_num,
            micro_batch_size=micro_batch_size,
            weight_cache_ratio=server_args.decode_weight_cache_ratio,
            gpu_kv_pool_slot_num=gpu_kv_pool_slot_num,
            draft_gpu_kv_pool_slot_num=draft_gpu_kv_pool_slot_num,
            gpu_attention_ratio=server_args.decode_gpu_attention_ratio,
            gpu_attention_micro_batch_size=server_args.decode_gpu_attention_micro_batch_size,
            gpu_attention_nano_batch_size=server_args.decode_gpu_attention_nano_batch_size,
            spec_policy=spec_policy, 
            draft_model_placement=server_args.draft_model_placement,
            draft_gpu_execution_ratio=server_args.draft_gpu_execution_ratio,
            draft_gpu_req_num=draft_gpu_req_num,
        )
    
    def clone(self):
        return Policy(
            stage=self.stage,
            global_batch_size=self.global_batch_size,
            batchsize=self.batchsize,
            micro_batch_num=self.micro_batch_num,
            micro_batch_size=self.micro_batch_size,
            weight_cache_ratio=self.weight_cache_ratio,
            gpu_kv_pool_slot_num=self.gpu_kv_pool_slot_num,
            draft_gpu_kv_pool_slot_num=self.draft_gpu_kv_pool_slot_num,
            gpu_attention_ratio=self.gpu_attention_ratio,
            gpu_attention_micro_batch_size=self.gpu_attention_micro_batch_size,
            gpu_attention_nano_batch_size=self.gpu_attention_nano_batch_size,
            spec_policy=self.spec_policy, 
            draft_model_placement=self.draft_model_placement,
            draft_gpu_execution_ratio=self.draft_gpu_execution_ratio,
            draft_gpu_req_num=self.draft_gpu_req_num,
        )
