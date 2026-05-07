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
"""ModelRunner runs the forward passes of the models."""

import dataclasses
import torch
from torch import nn
from typing import Optional, List, Tuple
import logging
import os
from functools import lru_cache
from pathlib import Path
import importlib
from time import sleep
import time

from sglang.srt.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.utils import get_available_gpu_memory, get_bool_env_var
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.dp_attention import initialize_dp_attention, get_attention_tp_group
from sglang.srt.configs.load_config import LoadConfig

import specmoe
from specmoe.speculative.spec_info import SpeculativeAlgorithm
from specmoe.utils.model_config import ModelConfig
from specmoe.utils.server_args import ServerArgs
from specmoe.backend.memory import ReqToTokenPool, TokenToKVPoolAllocatorGPU, TokenToKVPoolAllocatorCPU, MHATokenToKVPool
from specmoe.backend.forward_batch_info import ForwardBatch, ForwardMode
from specmoe.utils.data_info import SamplingBatchInfo, ModelWorkerBatch
from specmoe.backend.policy import Policy, SpecPolicy
from specmoe.backend.execution_engine import ExecutionEngine
from specmoe.layers.logits_processor import LogitsProcessorOutput

logger = logging.getLogger(__name__)

@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    for module_path in (Path(specmoe.__file__).parent / "models").glob("*.py"):
        module = importlib.import_module(f"specmoe.models.{module_path.stem}")
        if hasattr(module, "EntryClass"):
            model_arch_name_to_cls[module.EntryClass.__name__] = module.EntryClass
    return model_arch_name_to_cls


def get_model_cls_by_arch_name(model_arch_names, offload):
    model_arch_name_to_cls = import_model_classes()

    model_class = None
    for arch in model_arch_names:
        if offload:
            arch += "Off"
        if arch in model_arch_name_to_cls:
            model_class = model_arch_name_to_cls[arch]
            break
    else:
        raise ValueError(
            f"Unsupported architectures: {arch}. "
            f"Supported list: {list(model_arch_name_to_cls.keys())}"
        )
    return model_class


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
        offload: bool = True,
        gpu_req_to_token_pool: Optional[ReqToTokenPool] = None,
        gpu_token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocatorGPU] = None,
        cpu_req_to_token_pool: Optional[ReqToTokenPool] = None,
        cpu_token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocatorCPU] = None,
    ):
        logger.debug("My ModelRunner init")
        self.offload = offload
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.should_log = tp_rank == 0
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.use_mla_backend = False
        self.dtype = self.model_config.dtype
        self.gpu_req_to_token_pool = gpu_req_to_token_pool
        self.cpu_req_to_token_pool = cpu_req_to_token_pool
        self.gpu_token_to_kv_pool_allocator = gpu_token_to_kv_pool_allocator
        self.cpu_token_to_kv_pool_allocator = cpu_token_to_kv_pool_allocator

        # Model-specific adjustment
        # self.model_specific_adjustment()

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # min_per_gpu_memory = get_available_gpu_memory(
        #     self.device, self.gpu_id, distributed=self.tp_size > 1
        # )

        # If it is a draft model tp_group can be different.
        self.initialize(min_per_gpu_memory)

        if self.is_draft_worker:
            num_q_heads = self.model_config.num_attention_heads // 1
            n_kv_heads = self.model_config.num_key_value_heads // 1
            head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
            max_token_num = self.server_args.speculative_eagle_topk
            max_token_num = max(max_token_num, self.server_args.speculative_num_draft_tokens)
            if self.server_args.speculative_eagle_topk_gpu is not None:
                max_token_num = max(max_token_num, self.server_args.speculative_eagle_topk_gpu)
                max_token_num = max(max_token_num, self.server_args.speculative_num_draft_tokens_gpu)
            max_token_num = 20 # XXX: Forcely set to 20, for dynamic policy
            self.draft_qkv_pin = torch.empty(
                (self.server_args.decode_micro_batch_size * max_token_num, (num_q_heads + 2 * n_kv_heads) * head_dim),
                device='cpu',
                pin_memory=True,
            )
            self.draft_hidden_pin = torch.empty(
                (
                    self.server_args.decode_micro_batch_size * max_token_num,
                    num_q_heads,
                    head_dim,
                ),
                device="cpu",
                pin_memory=True,
            )
            logger.info(f"draft qkv pin size: {self.draft_qkv_pin.numel() * 2 / (1 << 30):.2f} GB")
            logger.info(f"draft hidden pin size: {self.draft_hidden_pin.numel() * 2 / (1 << 30):.2f} GB")


    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args

        # Load the model
        # self.sampler = Sampler()
        self.load_model()

        if self.device == "cuda":
            self.init_cublas()
        else:
            assert False
        
        if self.server_args.dont_output_eos:
            self.logit_bias = torch.zeros(self.model_config.vocab_size, device=self.device)
            self.logit_bias[self.model_config.eos_token_id] = -1e3
        else:
            self.logit_bias = None

        # auxiliary hidden capture mode. 
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            self.model.set_eagle3_layers_to_capture()
    
    def switch_to_decode(self, policy: Policy):
        begin_time = time.time()
        self.execution_engine.switch_to_decode(policy)
        # TODO: Delete GPU kv cache, because decode stage will not use GPU attention
        # self.gpu_token_to_kv_pool_allocator.clear()
        # self.gpu_token_to_kv_pool._clear_buffers()
        if self.server_args.target_cg_nano_kv_cache_slot != 0:
            self.gpu_token_to_kv_pool_allocator.adjust_capacity(1, self.server_args.target_cg_nano_kv_cache_slot) 
        else:
            self.gpu_token_to_kv_pool_allocator.adjust_capacity(1, self.server_args.max_gpu_attn_req_total_length_in_a_batch) # XXX: target model在decode阶段需要用到一层kv cache，因为发现GPUAttn计算和kv cache fetch根本overlap不起来，kernel launch的开销太大了，所以不如直接用一层，增大一次Attn的nano batchsize.
        # self.gpu_token_to_kv_pool_allocator.size = self.server_args.max_draft_gpu_req_total_length_in_a_batch # 有可能draft需要gpu kv cache，但是target不需要，要保留allocator的size
        end_time = time.time()
        logger.info(f"switch to decode stage, time cost: {end_time - begin_time:.2f}s. After switch, gpu available slots: {self.gpu_token_to_kv_pool_allocator.available_size()}")

    def load_model(self):
        """See also vllm/model_executor/model_loader.py::get_model"""
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.debug(
            f"Load weight begin. avail mem={before_avail_memory:.2f} GB"
        )

        # Select model class
        architectures = getattr(self.model_config.hf_config, "architectures", [])
        model_class = get_model_cls_by_arch_name(architectures, self.offload)
        logger.debug(f"Rank {self.tp_rank}: load weight begin.")

        # Load weights
        with set_default_torch_dtype(torch.float16):
            with torch.device("cuda"):
                model = model_class(
                    config=self.model_config.hf_config
                )
            loader = DefaultModelLoader(LoadConfig(load_format="auto"))
            model.load_weights(
                loader._get_all_weights(
                    self.model_config,
                    model
                )
            )
        self.model = model.eval()

        logger.debug(f"Rank {self.tp_rank}: load weight end.")

    def init_memory_pool(
        self,
        policy: Policy # assume prefill policy
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )
        assert policy.stage == "prefill"
        begin_time = time.time()
        self.cpu_max_total_num_tokens = self.profile_max_num_token_cpu()
        self.gpu_max_total_num_tokens = policy.gpu_kv_pool_slot_num

        logger.info(f"max total token numbers in GPU: {self.gpu_max_total_num_tokens}")
        logger.info(f"max total token numbers in CPU: {self.cpu_max_total_num_tokens}")

        max_num_reqs = policy.global_batch_size

        
        if self.gpu_req_to_token_pool is None:
            self.gpu_req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.server_args.max_seq_length + 4,
                device=self.device
            )
            self.cpu_req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.server_args.max_seq_length + 4,
                device='cpu'
            )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker
        
        if not hasattr(self.server_args, "allocate_cpu_slot"):
            self.server_args.allocate_cpu_slot = self.cpu_max_total_num_tokens

        if self.use_mla_backend:
            assert False, "DeepSeek not supported now."
        else:
            self.cpu_token_to_kv_pool = MHATokenToKVPool(
                self.server_args.allocate_cpu_slot,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_dim=self.model_config.head_dim,
                head_num=self.model_config.num_key_value_heads,
                layer_num=self.model_config.num_hidden_layers,
                device="cpu",
                is_draft_model=self.is_draft_worker,
            )
            self.gpu_token_to_kv_pool = MHATokenToKVPool(
                self.gpu_max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_dim=self.model_config.head_dim,
                head_num=self.model_config.num_key_value_heads,
                layer_num=1, # 对于target model和draft model，都只用一层就行
                device=self.device,
                is_draft_model=self.is_draft_worker,
            )

        if self.gpu_token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                self.cpu_token_to_kv_pool_allocator = TokenToKVPoolAllocatorCPU(
                    self.cpu_max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device="cpu",
                    kvcache=self.cpu_token_to_kv_pool,
                    max_output_length=self.server_args.max_output_length,
                    max_speculative_draft_tokens=self.server_args.max_speculative_draft_tokens,
                )
                self.gpu_token_to_kv_pool_allocator = TokenToKVPoolAllocatorGPU(
                    self.gpu_max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.gpu_token_to_kv_pool,
                )
            else:
                assert False, "page size > 1 not supported now."
        else:
            assert self.is_draft_worker

        logger.info(
            f"Memory pool end. Use time: {time.time() - begin_time:.2f} s. "
            f"avail gpu mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def profile_max_num_token_cpu(self):
        # available_gpu_memory = get_available_gpu_memory(
        #     self.device, self.gpu_id, distributed=self.tp_size > 1
        # )
        if self.use_mla_backend:
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.num_key_value_heads
                * self.model_config.head_dim
                * 2 # K 和 V
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        cpu_max_num_token = int(self.server_args.available_cpu_dram_for_kvcache * (1 << 30) // (cell_size * self.model_config.num_hidden_layers)) # CPU上存所有层KV cache
        return cpu_max_num_token

    def init_torch_distributed(self):
        logger.debug("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        if self.device == "cuda":
            backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)

        if not self.is_draft_worker:
            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size,
                rank=self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
            initialize_dp_attention(
                enable_dp_attention=self.server_args.enable_dp_attention,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
            )

        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        self.tp_group = get_tp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                    logger.warning(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )
                else:
                    raise ValueError(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )

        logger.debug(
            f"Init torch distributed ends. mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        from specmoe.layers.attention_backend.triton_backend import TritonAttnBackend
        # from specmoe.layers.attention_backend.cpu_native_backend import CPUNativeAttnBackend
        from specmoe.layers.attention_backend.cpu_backend import CPUAttnBackend
        self.gpu_attn_backend = TritonAttnBackend(self)
        # self.cpu_attn_backend = CPUNativeAttnBackend(self)
        self.cpu_attn_backend = CPUAttnBackend(self)

    def forward_prefill(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        '''
        返回值：List[LogitsProcessorOutput]
            LogitsProcessorOutput: 最后一层的logits输出, hidden_state_to_store
        '''
        for i, forward_batch in enumerate(forward_micro_batches):
            self.gpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=i)
        logger.debug(f"prefill begin")

        res = self.execution_engine.prefill_forward(forward_micro_batches, policy)
        # res = []
        # for mb in forward_micro_batches:
        #     res.append(LogitsProcessorOutput(hidden_states=None, hidden_states_cpu=torch.randn(mb.seq_lens.sum().item(), 4096, device='cpu', pin_memory=True), next_token_logits=torch.randn(len(mb.seq_lens), 32000, device='cuda')))

        logger.debug(f"prefill end")
        return res

    def forward_decode(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        pt = 0
        for i, forward_batch in enumerate(forward_micro_batches):
            self.cpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=i)
            if policy.gpu_attention_ratio != 0 and len(forward_batch.gpu_seq_lens) > 0:
                forward_batch.gpu_attn_init_slot = []
                nano_batch_num = (len(forward_batch.gpu_seq_lens) + forward_batch.policy.gpu_attention_nano_batch_size - 1) // forward_batch.policy.gpu_attention_nano_batch_size
                for j in range(nano_batch_num):
                    begin = j * forward_batch.policy.gpu_attention_nano_batch_size
                    end = min(begin + forward_batch.policy.gpu_attention_nano_batch_size, len(forward_batch.gpu_seq_lens))
                    self.gpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=pt, begin=begin, end=end)
                    forward_batch.gpu_attn_init_slot.append(pt)
                    pt += 1
                # print(f"gpu req lens: {forward_batch.gpu_seq_lens}")
        logger.debug("decode begin")
        res, performance_recoder = self.execution_engine.decode_forward(forward_micro_batches, policy)
        logger.debug("decode end")
        return res, performance_recoder
    
    def forward_verify_sequential_gpu_only(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        # 这个好像不能复用 execution_engine的pipeline，所以直接把execution engine的很多变量清理一下，比如pin memory之类的
        # 推理前先将所有kv cache搬运到GPU上，TODO: 需要在prepare阶段提前分配好GPU上的token slot，包括原先的KV cache以及新token生成的KV cache的slot
        pass
    
    def forward_verify_sequential_cpu_only(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        # 这个好像可以复用 execution_engine的pipeline
        # 需要改动CPU kernel，以使其支持custom mask以及qlen>1的speculative decoding
        # 需要改动pin memory等的初始化逻辑，以使其支持qlen>1的情况
        for i, forward_batch in enumerate(forward_micro_batches):
            self.cpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=i)
        logger.debug("sequential cpu only verify begin")
        res, performance_recoder = self.execution_engine.decode_forward(forward_micro_batches, policy)
        logger.debug("sequential cpu only verify end")
        return res, performance_recoder

    def forward_verify_sequential_cg_coop(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        # 需要大改pipeline，可能要丢弃掉execution engine，但是复用execution context，以支持CPU、GPU同时协作
        pt = 0
        for i, forward_batch in enumerate(forward_micro_batches):
            self.cpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=i)
            if policy.gpu_attention_ratio != 0 and len(forward_batch.gpu_seq_lens) > 0:
                forward_batch.gpu_attn_init_slot = []
                nano_batch_num = (len(forward_batch.gpu_seq_lens) + forward_batch.policy.gpu_attention_nano_batch_size - 1) // forward_batch.policy.gpu_attention_nano_batch_size
                for j in range(nano_batch_num):
                    begin = j * forward_batch.policy.gpu_attention_nano_batch_size
                    end = min(begin + forward_batch.policy.gpu_attention_nano_batch_size, len(forward_batch.gpu_seq_lens))
                    self.gpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=pt, begin=begin, end=end)
                    forward_batch.gpu_attn_init_slot.append(pt)
                    pt += 1
        logger.debug("sequential cpu-gpu cooperation verify begin")
        res, performance_recoder = self.execution_engine.decode_forward(forward_micro_batches, policy)
        logger.debug("sequential cpu-gpu cooperation verify end")
        return res, performance_recoder

    def forward(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        self.execution_engine.update_before_forward(forward_micro_batches)
        if policy.stage == 'prefill':
            res = self.forward_prefill(forward_micro_batches, policy)
            return res, None
        elif policy.stage == 'decode':
            if policy.spec_policy.is_none():
                res, performance_recoder = self.forward_decode(forward_micro_batches, policy)
            elif policy.spec_policy == SpecPolicy.SequentialGPUonly:
                res, performance_recoder = self.forward_verify_sequential_gpu_only(forward_micro_batches, policy)
            elif policy.spec_policy == SpecPolicy.SequentialCPUonly:
                res, performance_recoder = self.forward_verify_sequential_cpu_only(forward_micro_batches, policy)
            elif policy.spec_policy == SpecPolicy.SequentialCGCoop:
                res, performance_recoder = self.forward_verify_sequential_cg_coop(forward_micro_batches, policy)
            else:
                assert False, f"spec policy: {policy.spec_policy} not supported yet"
            return res, performance_recoder
        else:
            raise ValueError(f"Invalid policy stage: {policy.stage}")
        # if forward_micro_batches[0].forward_mode.is_decode():
        #     return self.forward_decode(forward_micro_batches)
        # elif forward_micro_batches[0].forward_mode.is_extend():
        #     return self.forward_extend(
        #         forward_batch
        #     )
        # elif forward_batch.forward_mode.is_idle():
        #     return self.forward_idle(forward_batch)
        # else:
        #     raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")
    
    def draft_forward_prefill(self, forward_micro_batches: List[ForwardBatch], policy: Policy) -> List[LogitsProcessorOutput]:
        '''
        返回值：List[LogitsProcessorOutput]
            LogitsProcessorOutput: 最后一层的logits输出, hidden_state_to_store
        '''
        for i, forward_batch in enumerate(forward_micro_batches):
            self.gpu_attn_backend.init_forward_metadata(forward_batch, micro_batch_id=i)
        logger.debug(f"draft model prefill begin")

        # res = self.execution_engine.prefill_forward(forward_micro_batches, policy)
        res = []
        for i, forward_batch in enumerate(forward_micro_batches):
            forward_batch.spec_info.hidden_states = forward_batch.spec_info.hidden_states.to('cuda')
            res.append(self.model.forward(input_ids=forward_batch.input_ids, positions=forward_batch.positions, forward_batch=forward_batch, micro_batch_id=i))
            # res[-1].hidden_states = res[-1].hidden_states.to('cpu', non_blocking=True)
            # res[-1].next_token_logits = res[-1].next_token_logits.to('cpu', non_blocking=True)
            del forward_batch.spec_info.hidden_states

            # XXX: need to offload draft model KV cache to CPU
            src_indices = forward_batch.gpu_out_cache_loc
            dst_indices = forward_batch.cpu_out_cache_loc
            src = self.gpu_token_to_kv_pool.get_layer_kv_cache(0, src_indices)
            # logger.debug(f"src: {src.shape}, dst_indices: {dst_indices}, src_indices: {src_indices}, seq_lens: {forward_batch.seq_lens}")
            self.cpu_token_to_kv_pool_allocator.offload_kv_cache_prefill(0, src, dst_indices, forward_batch.seq_lens, self.cpu_token_to_kv_pool.kv_buffer) # EAGLE only has one layer
            torch.cuda.synchronize()
            logger.debug(f"draft model KV cache is offloaded to CPU")

        logger.debug(f"draft model prefill end")
        return res
    def draft_forward_extend_after_verify(self, forward_micro_batches: List[ForwardBatch], policy: Policy, gpu_execution: bool = True) -> List[LogitsProcessorOutput]:
        # for forward_batch in forward_micro_batches:
        #     assert forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
        #     self.gpu_attn_backend.init_forward_metadata(forward_batch)
        if gpu_execution and policy.draft_gpu_req_num > 0:
            for i, forward_batch in enumerate(forward_micro_batches):
                self.gpu_attn_backend.init_forward_metadata(forward_batch) 
        elif not gpu_execution:
            for i, forward_batch in enumerate(forward_micro_batches):
                self.cpu_attn_backend.init_forward_metadata(forward_batch) # XXX: 因为draft model现在都是假设一个一个执行，所以不需要指定micro_batch_id，未来需要多个micro-batch重叠时，可能有用
        logger.debug(f"draft model extend after verify begin")
        res = []
        for forward_batch in forward_micro_batches:
            assert forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            if gpu_execution:
                pass
            else:
                q_num = forward_batch.input_ids.shape[0]
                forward_batch.qkv_pin = self.draft_qkv_pin[:q_num]
                forward_batch.hidden_pin = self.draft_hidden_pin[:q_num]
            res.append(self.model.forward(input_ids=forward_batch.input_ids, positions=forward_batch.positions.to('cuda'), forward_batch=forward_batch, gpu_execution=gpu_execution))
        logger.debug(f"draft model extend after verify end")
        return res

    def draft_forward_generate(self, forward_micro_batches: List[ForwardBatch], policy: Policy, process_cpu_requets: bool = True, gpu_execution: bool = True) -> List[LogitsProcessorOutput]:
        assert self.is_draft_worker
        # TODO: 以下好像是重复init forward metadata, 在eagle_worker里就Init过了
        # if gpu_execution:
        #     for i, forward_batch in enumerate(forward_micro_batches):
        #         self.gpu_attn_backend.init_forward_metadata(forward_batch) 
        # else:
        #     for i, forward_batch in enumerate(forward_micro_batches):
        #         self.cpu_attn_backend.init_forward_metadata(forward_batch) # XXX: 因为draft model现在都是假设一个一个执行，即len(forward_micro_batches) == 1，所以不需要指定micro_batch_id，未来需要多个micro-batch重叠时，可能有用
        # logger.debug(f"draft token generate begin")
        res = []
        for i, forward_batch in enumerate(forward_micro_batches):
            assert forward_batch.forward_mode == ForwardMode.DECODE
            if gpu_execution:
                pass
            else:
                q_num = forward_batch.input_ids.shape[0]
                forward_batch.qkv_pin = self.draft_qkv_pin[:q_num]
                forward_batch.hidden_pin = self.draft_hidden_pin[:q_num]
            res.append(self.model.forward(input_ids=forward_batch.input_ids, positions=forward_batch.positions.to('cuda'), forward_batch=forward_batch, process_cpu_requets=process_cpu_requets, gpu_execution=gpu_execution))
        # logger.debug(f"draft token generate end")
        return res


    def draft_forward(self, forward_micro_batches: List[ForwardBatch], policy: Policy, is_draft_generate: bool = False, process_cpu_requets: bool = True, gpu_execution: bool = True) -> List[LogitsProcessorOutput]:
        assert self.is_draft_worker
        if policy.stage == 'prefill':
            res = self.draft_forward_prefill(forward_micro_batches, policy)
        elif policy.stage == 'decode':
            if not is_draft_generate:
                res = self.draft_forward_extend_after_verify(forward_micro_batches, policy, gpu_execution)
            else:
                res = self.draft_forward_generate(forward_micro_batches, policy, process_cpu_requets, gpu_execution)
        else:
            raise ValueError(f"Invalid policy stage: {policy.stage}")
        return res

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # # Apply logit bias
        # if sampling_info.sampling_info_done:
        #     # Overlap mode: the function update_regex_vocab_mask was executed
        #     # in process_batch_result of the last batch.
        #     if sampling_info.grammars:
        #         sampling_info.sampling_info_done.wait()
        # else:
        #     # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
        #     sampling_info.update_regex_vocab_mask()
        # sampling_info.apply_logits_bias(logits_output.next_token_logits)
        return 

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ModelWorkerBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        # no need to preprocess logits
        # self._preprocess_logits(logits_output, forward_batch.sampling_info)

        next_token_ids, next_token_probs = forward_batch.sample(logits_output)

        # Sample the next tokens
        # next_token_ids = self.sampler(
        #     logits_output,
        #     forward_batch.sampling_info,
        #     return_logprob=False,
        #     top_logprobs_nums=[],
        #     token_ids_logprobs=[],
        # )
        return next_token_ids

    def init_execution_engine(self, policy: Policy, execution_engine: Optional[ExecutionEngine] = None):
        self.init_memory_pool(policy)
        self.init_attention_backend()
        if execution_engine is None: # Target Model case
            self.execution_engine = ExecutionEngine(
                model=self.model,
                model_config=self.model_config,
                server_args=self.server_args,
                policy=policy,
                cpu_req_to_token_pool=self.cpu_req_to_token_pool,
                gpu_req_to_token_pool=self.gpu_req_to_token_pool,
                cpu_token_to_kv_pool_allocator=self.cpu_token_to_kv_pool_allocator,
                gpu_token_to_kv_pool_allocator=self.gpu_token_to_kv_pool_allocator,
            )
            self.execution_engine.init_gpu_experts()
        else: # Draft Model case
            self.execution_engine = execution_engine
    
    def change_execution_engine_policy(self, policy: Policy):
        pass