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
"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Union
import logging

import torch
import triton
import triton.language as tl

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.utils import flatten_nested_list, get_compiler_backend

if TYPE_CHECKING:
    from specmoe.layers.attention_backend.base_attention_backend import AttentionBackend
    from specmoe.utils.data_info import ModelWorkerBatch
    from specmoe.speculative.spec_info import SpeculativeAlgorithm
    from specmoe.utils.data_info import SamplingBatchInfo
    from specmoe.backend.memory import ReqToTokenPool, KVCache
    from specmoe.backend.model_runner import ModelRunner
    from specmoe.speculative.speculative_utils import EagleDraftInput, EagleVerifyInput
    from specmoe.backend.policy import Policy

from specmoe.utils.server_args import ServerArgs
from specmoe.utils.model_config import ModelConfig

logger = logging.getLogger(__name__)

class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers wil be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_prefill(self):
        return self.is_extend()

    def is_extend(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.TARGET_VERIFY
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self):
        return self == ForwardMode.DRAFT_EXTEND

    def is_extend_or_draft_extend_or_mixed(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.MIXED
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
        )

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

class DecodePart(IntEnum):
    CPU_ATTN = auto()
    PREATTN = auto()
    POSTATTN = auto()
    ALL = auto()

class CaptureHiddenMode(IntEnum):
    NULL = auto()
    # Capture hidden states of all tokens.
    FULL = auto()
    # Capture a hidden state of the last token.
    LAST = auto()

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST

@dataclass
class MicroBatchPolicy:
    micro_batch_size: int
    # cpu_bachsize + gpu_batchsize == micro_batch_size
    cpu_bachsize: int
    gpu_batchsize: int

    cpu_draft_budget: int
    gpu_draft_budget: int

    cpu_reqs_id: List[int]
    gpu_reqs_id: List[int]
@dataclass
class ForwardMetaData:
    # system policy
    global_batch_size: int # 尽可能大地占满CPU内存
    batchsize: int # the sum of all micro_batch_size (batchsize = micro_batch_num * micro_batch_size)
    micro_batch_size: int
    micro_batch_num: int

    micro_batch_policies: List[MicroBatchPolicy]

    stage: str = 'prefill'

    @classmethod
    def get_prefill_policy(
        cls,
        server_args: ServerArgs,
        global_batch_size: int,
        model_config: ModelConfig,
    ):
        micro_batch_num = server_args.prefill_micro_batch_num
        micro_batch_size = server_args.prefill_micro_batch_size
        batchsize = micro_batch_num * micro_batch_size
        micro_batch_policies = []

        accumulate_micro_batch_size = 0
        for i in range(micro_batch_num - 1):
            micro_batchsize = global_batch_size // micro_batch_num
            micro_batch_policies.append(
                MicroBatchPolicy(
                    micro_batch_size=micro_batchsize,
                    cpu_bachsize=0,
                    gpu_batchsize=micro_batchsize,
                    cpu_draft_budget=0,
                    gpu_draft_budget=0,
                    cpu_reqs_id=[],
                    gpu_reqs_id=[i for i in range(micro_batchsize)],
                )
            )
            accumulate_micro_batch_size += micro_batchsize
        micro_batch_policies.append(
            MicroBatchPolicy(
                micro_batch_size=global_batch_size - accumulate_micro_batch_size,
                cpu_bachsize=0,
                gpu_batchsize=global_batch_size - accumulate_micro_batch_size,
                cpu_draft_budget=0,
                gpu_draft_budget=0,
                cpu_reqs_id=[],
                gpu_reqs_id=[i for i in range(global_batch_size - accumulate_micro_batch_size)],
            )
        )

        return cls(
            stage='prefill',
            global_batch_size=global_batch_size,
            batchsize=batchsize,
            micro_batch_num=micro_batch_num,
            micro_batch_size=micro_batch_size,
            micro_batch_policies=micro_batch_policies,
        )
    
    def refine_micro_batch_reqs(self, micro_batch_id: int, cpu_reqs_id: List[int], gpu_reqs_id: List[int]):
        self.micro_batch_policies[micro_batch_id].cpu_reqs_id = cpu_reqs_id
        self.micro_batch_policies[micro_batch_id].gpu_reqs_id = gpu_reqs_id

@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    decode_part: DecodePart
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    gpu_req_pool_indices: torch.Tensor
    cpu_req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    cpu_seq_lens: torch.Tensor
    gpu_seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    gpu_out_cache_loc: torch.Tensor
    cpu_out_cache_loc: torch.Tensor

    # The request ids, used in cpu attention
    cpu_batch_rids: List[int]
    gpu_batch_rids: List[int]

    # policy
    policy: Policy

    # The sum of all sequence lengths
    seq_lens_sum: int
    gpu_seq_lens_sum: int
    cpu_seq_lens_sum: int
    
    gpu_seq_lens_list: List[int] = None

    # Tensor for decoding
    hidden_states: torch.Tensor = None
    hidden_states_cpu: torch.Tensor = None
    residual: torch.Tensor = None
    residual_cpu: torch.Tensor = None
    qkv: torch.Tensor = None
    experts_mapping: torch.Tensor = None # expert mapping

    qkv_pin: torch.Tensor = None # use for cpu attention input
    hidden_pin: torch.Tensor = None # used for cpu attention output

    # sync
    attn_event: torch.cuda.Event = None

    # Position information
    positions: torch.Tensor = None
    cpu_positions: Optional[torch.Tensor] = None
    gpu_positions: Optional[torch.Tensor] = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    # extend_prefix_lens_cpu: Optional[List[int]] = None
    # extend_seq_lens_cpu: Optional[List[int]] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    gpu_req_to_token_pool: ReqToTokenPool = None
    cpu_req_to_token_pool: ReqToTokenPool = None
    gpu_token_to_kv_pool: KVCache = None
    cpu_token_to_kv_pool: KVCache = None
    gpu_attn_backend: AttentionBackend = None
    cpu_attn_backend: AttentionBackend = None

    # Speculative decoding
    spec_info: Optional[Union[EagleVerifyInput, EagleDraftInput]] = None
    spec_algorithm: SpeculativeAlgorithm = None
    capture_hidden_mode: CaptureHiddenMode = None
    draft_model_placement: str = 'GPU'
    
    # Other
    gpu_attn_input_ids_size: int = 0
    gpu_attn_per_request_input_ids_size: int = 0

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        device = model_runner.device
        # extend_input_logprob_token_ids_gpu = None
        # if batch.extend_input_logprob_token_ids is not None:
        #     extend_input_logprob_token_ids_gpu = (
        #         batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
        #     )
        if batch.forward_mode.is_extend():
            decode_part = DecodePart.ALL
        else:
            decode_part = DecodePart.PREATTN
        
        ret = cls(
            forward_mode=batch.forward_mode,
            decode_part=decode_part,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            gpu_req_pool_indices=batch.gpu_req_pool_indices,
            cpu_req_pool_indices=batch.cpu_req_pool_indices,
            seq_lens=batch.seq_lens,
            cpu_seq_lens=batch.cpu_seq_lens,
            gpu_seq_lens=batch.gpu_seq_lens,
            gpu_out_cache_loc=batch.gpu_out_cache_loc,
            cpu_out_cache_loc=batch.cpu_out_cache_loc,
            cpu_batch_rids = batch.cpu_reqs_rids,
            gpu_batch_rids = batch.gpu_reqs_rids,
            seq_lens_sum=batch.seq_lens_sum,
            gpu_seq_lens_sum=batch.gpu_seq_lens_sum,
            cpu_seq_lens_sum=batch.cpu_seq_lens_sum,
            policy=batch.policy,
            gpu_req_to_token_pool=model_runner.gpu_req_to_token_pool,
            cpu_req_to_token_pool=model_runner.cpu_req_to_token_pool,
            gpu_token_to_kv_pool=model_runner.gpu_token_to_kv_pool,
            cpu_token_to_kv_pool=model_runner.cpu_token_to_kv_pool,
            gpu_attn_backend=model_runner.gpu_attn_backend,
            cpu_attn_backend=model_runner.cpu_attn_backend,
            spec_algorithm=batch.spec_algorithm,
            spec_info=batch.spec_info,
            capture_hidden_mode=batch.capture_hidden_mode,
            draft_model_placement=model_runner.server_args.draft_model_placement,
        )
        
        if ret.forward_mode.is_idle():
            ret.positions = torch.empty((0,), device=device)
            return ret
        

        # Override the positions with spec_info
        if (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions
        
        # only target verify and normal decode need GPU Attention
        if ret.spec_info is not None and batch.forward_mode.is_target_verify(): # for target verify
            # 根据spec_info更新gpu_attn_input_ids_size
            ret.gpu_attn_input_ids_size = ret.spec_info.draft_token_num_gpu * len(ret.gpu_seq_lens) 
            ret.gpu_attn_per_request_input_ids_size = ret.spec_info.draft_token_num_gpu
        elif ret.spec_info is not None and batch.forward_mode.is_draft_extend(): # for draft extend
            pass # TODO: may need change
        elif ret.spec_info is not None and batch.forward_mode.is_decode(): # for draft generate
            pass
        elif batch.forward_mode.is_decode_or_idle(): # for normal decode
            # in normal decode, all request has exactly one query
            ret.gpu_attn_input_ids_size = len(ret.gpu_seq_lens)
            ret.gpu_attn_per_request_input_ids_size = 1
        elif batch.forward_mode == ForwardMode.EXTEND: # for extend, all request attention run in GPU
            ret.gpu_attn_input_ids_size = len(ret.input_ids)

        # Init position information
        if ret.forward_mode.is_decode():
            if ret.positions is None:
                ret.positions = clamp_position(batch.seq_lens).to('cuda')
                # logger.debug(f"seq_lens: {batch.seq_lens}")
                # logger.debug(f"positions: {ret.positions}")
        else:
            if isinstance(batch.extend_seq_lens, List):
                ret.extend_seq_lens = torch.tensor(
                    batch.extend_seq_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
            else:
                ret.extend_seq_lens = batch.extend_seq_lens.clone().detach().to(device, non_blocking=True)
            if isinstance(batch.extend_prefix_lens, List):
                ret.extend_prefix_lens = torch.tensor(
                    batch.extend_prefix_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
            else:
                ret.extend_prefix_lens = batch.extend_prefix_lens.clone().detach().to(device, non_blocking=True)

            if model_runner.server_args.attention_backend != "torch_native":
                ret.extend_num_tokens = batch.extend_num_tokens
                if batch.extend_num_tokens is None:
                    assert batch.forward_mode != ForwardMode.EXTEND
                    ret.extend_num_tokens = 1
                positions, ret.extend_start_loc = compute_position_triton(
                    ret.extend_prefix_lens,
                    ret.extend_seq_lens,
                    ret.extend_num_tokens,
                )
            else:
                assert False
                positions, ret.extend_start_loc = compute_position_torch(
                    ret.extend_prefix_lens, ret.extend_seq_lens
                )
            if ret.positions is None:
                ret.positions = positions
        return ret

def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.cat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc


@torch.compile(dynamic=True, backend=get_compiler_backend())
def clamp_position(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)

