import dataclasses
from typing import List, Optional, Union, Tuple, Set, TYPE_CHECKING
from enum import IntEnum, auto
import torch
import logging
import triton
import triton.language as tl
import math
import copy

from specmoe.utils.server_args import ServerArgs
from specmoe.utils.model_config import ModelConfig
from specmoe.backend.memory import ReqToTokenPool, TokenToKVPoolAllocatorGPU, TokenToKVPoolAllocatorCPU
from specmoe.backend.forward_batch_info import ForwardMode, CaptureHiddenMode
from specmoe.backend.policy import Policy

if TYPE_CHECKING:
    from specmoe.speculative.speculative_utils import EagleDraftInput, EagleVerifyInput
    from specmoe.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

global_server_args_dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "triton_attention_reduce_in_fp32": ServerArgs.triton_attention_reduce_in_fp32,
    "torchao_config": ServerArgs.torchao_config,
    "enable_nan_detection": ServerArgs.enable_nan_detection,
    "enable_dp_attention": ServerArgs.enable_dp_attention,
    "enable_ep_moe": ServerArgs.enable_ep_moe,
    "enable_deepep_moe": ServerArgs.enable_deepep_moe,
    "deepep_mode": ServerArgs.deepep_mode,
    "device": ServerArgs.device,
    "speculative_accept_threshold_single": ServerArgs.speculative_accept_threshold_single,
    "speculative_accept_threshold_acc": ServerArgs.speculative_accept_threshold_acc,
    "disable_radix_cache": ServerArgs.disable_radix_cache,
    "flashinfer_mla_disable_ragged": ServerArgs.flashinfer_mla_disable_ragged,
    "moe_dense_tp_size": ServerArgs.moe_dense_tp_size,
    "chunked_prefill_size": ServerArgs.chunked_prefill_size,
    "n_share_experts_fusion": ServerArgs.n_share_experts_fusion,
    "disable_chunked_prefix_cache": ServerArgs.disable_chunked_prefix_cache,
    "use_mla_backend": ServerArgs.use_mla_backend,
}

class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }

class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message="Unknown error", status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


bid = 0
SCHEDULE_BATCH_ID = 0
def next_schedule_batch_id():
    global SCHEDULE_BATCH_ID
    SCHEDULE_BATCH_ID += 1
    return SCHEDULE_BATCH_ID

@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    greedy: torch.Tensor

    # Device 
    device: str = "cuda"

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        for item in [
            "temperatures",
            "top_ks",
            "greedy",
        ]:
            value = getattr(self, item, None)
            setattr(self, item, value[keep_indices_device])
    
    @classmethod
    def from_schedule_batch(cls, batch: "ScheduleBatch", vocab_size: int):
        reqs = batch.reqs
        device = batch.device
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperatures for r in reqs],
                dtype=torch.float,
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ks = torch.tensor(
            [r.sampling_params.top_ks for r in reqs], dtype=torch.int32
        ).to(device, non_blocking=True)
        top_ps = torch.tensor(
            [r.sampling_params.top_ps for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)

        ret = cls(
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            greedy=all(r.sampling_params.top_ks <= 1 for r in reqs),
            device=device,
        )
        return ret
RID = 0
class Req():
    """The input and output status of a request."""
    # original_prompt: str = None
    # sampling_params: SamplingBatchInfo = None
    # original_input_id: Tuple[int] = None  
    # output_ids: List[int] = None
    # fill_ids: List[int] = None  

    def __init__(
        self,
        original_prompt: str,
        sampling_params: SamplingBatchInfo,
        max_new_tokens: int = 32,
        original_input_ids: List[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.original_prompt = original_prompt
        self.sampling_params = sampling_params
        self.max_new_tokens = max_new_tokens
        self.original_input_ids = original_input_ids
        self.output_ids = []
        self.fill_ids = None  # fill_ids = origin_input_ids + output_ids
        self.return_hidden_states = False # used in eagle 

        global RID
        RID += 1
        self.rid = RID

        # Speculative Decoding
        self.verified_but_not_forward_ids: List[int] = []
        self.spec_info: Optional[EagleDraftInput] = None # 用于存储prefill结束后的specinfo，因为prefill结束后会销毁schedule batch
        self.spec_verify_ct = 0        # The number of verification forward passes in the speculative decoding. This is used to compute the average acceptance length per request.

        # Memory pool info
        self.cpu_req_pool_idx: Optional[int] = None
        self.gpu_req_pool_idx: Optional[int] = None

        # Finish
        self.tokenizer = None
        self.finished_reason = None
        self.eos_token_id = eos_token_id

        self.return_logprob = False
    
    def __str__(self):
        return f"original_prompt: {self.original_prompt}, sampling_params: {self.sampling_params}, original_input_ids: {self.original_input_ids}, eos_token_id: {self.eos_token_id}"
    
    def finished(self):
        return self.finished_reason is not None
    
    def init_next_round_input(
        self,
    ):
        self.fill_ids = self.original_input_ids + self.output_ids

        # logger.debug(f"fill_ids length: {len(self.fill_ids)}, output_ids: {self.output_ids}")

        self.extend_input_len = len(self.fill_ids)
    
    def check_finished(self):
        if self.finished():
            return

        if len(self.output_ids) >= self.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]

        matched_eos = False

        # Check stop token ids
        matched_eos = last_token_id == self.eos_token_id
        if matched_eos:
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
            return
    
    def add_output_ids(self, output_ids: Union[List[int], int]):
        output_ids = output_ids.cpu().tolist()
        if isinstance(output_ids, int):
            self.output_ids.append(output_ids)
        else:
            self.output_ids.extend(output_ids)


@dataclasses.dataclass
class ScheduleBatch():
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    is_req_on_gpu: List[bool] = None
    cpu_reqs_rids: List[int] = None
    gpu_reqs_rids: List[int] = None
    gpu_req_to_token_pool: ReqToTokenPool = None
    gpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorGPU = None
    cpu_req_to_token_pool: ReqToTokenPool = None
    cpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU = None

    decode_schedule_batch_id: int = None

    # Policy
    policy: Policy = None

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    all_requests_finished: bool = False

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int64
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    cpu_req_pool_indices: torch.Tensor = None  # shape: [b], int64
    gpu_req_pool_indices: torch.Tensor = None  # shape: [b], int64
    seq_lens: torch.Tensor = None  # shape: [b], int64
    cpu_seq_lens: torch.Tensor = None  # shape: [b], int64
    gpu_seq_lens: torch.Tensor = None  # shape: [b], int64
    # The output locations of the KV cache
    cpu_out_cache_loc: torch.Tensor = None  # shape: [b], int64
    gpu_out_cache_loc: torch.Tensor = None  # shape: [b], int64
    output_ids: torch.Tensor = None  # shape: [b], int64

    # The sum of all sequence lengths
    seq_lens_sum: int = None
    cpu_seq_lens_sum: int = None
    gpu_seq_lens_sum: int = None

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: int = None

    # Device
    device: str = "cuda"

    # Speculative decoding
    spec_algorithm: "SpeculativeAlgorithm" = None
    spec_info: Optional[Union["EagleDraftInput", "EagleVerifyInput"]] = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        is_req_on_gpu: List[bool],
        gpu_req_to_token_pool: ReqToTokenPool,
        gpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorGPU,
        cpu_req_to_token_pool: ReqToTokenPool,
        cpu_token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU,
        model_config: ModelConfig,
        spec_algorithm: "SpeculativeAlgorithm",
        policy: Policy,
        split_for_draft: bool = False,
    ):
        if policy.gpu_attention_ratio != 0 and policy.stage == "decode" and not split_for_draft: # 如果gpu_attention_ratio != 0，则按照gpu_attention_ratio对reqs进行排序再划分，传入的is_req_on_gpu会被覆盖
            # 按照length从小到大对reqs进行排序
            reqs.sort(key=lambda x: len(x.fill_ids))
            # 按照gpu_attention_ratio对reqs进行划分
            gpu_req_num = policy.gpu_attention_micro_batch_size
            # 按照gpu_attention_ratio对is_req_on_gpu进行划分
            is_req_on_gpu = [True] * gpu_req_num + [False] * (len(reqs) - gpu_req_num)
        else:
            pass

        cpu_reqs_rids = [reqs[i].rid for i in range(len(reqs)) if not is_req_on_gpu[i]]
        gpu_reqs_rids = [reqs[i].rid for i in range(len(reqs)) if is_req_on_gpu[i]]
        
        if policy.stage == "decode":
            decode_schedule_batch_id = next_schedule_batch_id()
        else:
            decode_schedule_batch_id = None

        return cls(
            decode_schedule_batch_id=decode_schedule_batch_id,
            reqs=reqs,
            is_req_on_gpu=is_req_on_gpu,
            gpu_reqs_rids=gpu_reqs_rids,
            cpu_reqs_rids=cpu_reqs_rids,
            gpu_req_to_token_pool=gpu_req_to_token_pool,
            gpu_token_to_kv_pool_allocator=gpu_token_to_kv_pool_allocator,
            cpu_req_to_token_pool=cpu_req_to_token_pool,
            cpu_token_to_kv_pool_allocator=cpu_token_to_kv_pool_allocator,
            model_config=model_config,
            device=gpu_req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            policy=policy.clone(),
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def get_model_worker_batch(self) -> "ModelWorkerBatch":
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            if self.forward_mode == ForwardMode.TARGET_VERIFY: # XXX: 这个extend_seq_lens和extend_prefix_lens还有extend_num_tokens在speculative decoding中好像没用，因为也不需要算position
                extend_seq_lens = [1] * len(self.reqs)
                extend_prefix_lens = [0] * len(self.reqs)
                self.extend_num_tokens = len(self.reqs)
            if self.forward_mode == ForwardMode.DRAFT_EXTEND:
                extend_prefix_lens = torch.tensor([0] * len(self.reqs), device='cuda') # TODO: 需要检查这样设置对不对

        # # Create seq_lens_cpu when needed
        # if (
        #     (
        #         global_server_args_dict["use_mla_backend"]
        #         and global_server_args_dict["attention_backend"] == "flashinfer"
        #     )
        #     or global_server_args_dict["attention_backend"] == "flashmla"
        #     or global_server_args_dict["attention_backend"] == "fa3"
        # ):
        #     seq_lens_cpu = self.seq_lens.cpu()
        # else:
        #     seq_lens_cpu = None

        global bid
        bid += 1
        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            cpu_reqs_rids=self.cpu_reqs_rids,
            gpu_reqs_rids=self.gpu_reqs_rids,
            input_ids=self.input_ids,
            cpu_req_pool_indices=self.cpu_req_pool_indices,
            gpu_req_pool_indices=self.gpu_req_pool_indices,
            seq_lens=self.seq_lens,
            cpu_seq_lens=self.cpu_seq_lens,
            gpu_seq_lens=self.gpu_seq_lens,
            seq_lens_sum=self.seq_lens_sum,
            gpu_out_cache_loc=self.gpu_out_cache_loc,
            cpu_out_cache_loc=self.cpu_out_cache_loc,
            cpu_seq_lens_sum=self.cpu_seq_lens_sum,
            gpu_seq_lens_sum=self.gpu_seq_lens_sum,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            policy=self.policy,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
        )
    
    def remove_finished_req(self, requests: List[Req]):
        # 找到request在reqs中的索引
        for request in requests:
            index = self.reqs.index(request)
            self.reqs.remove(request)
            if self.spec_algorithm.is_none():
                self.is_req_on_gpu.pop(index) # TODO: 只更改这个好像不太好，最好重新分配比例
        
            if request.rid in self.cpu_reqs_rids:
                self.cpu_reqs_rids.remove(request.rid)
            if request.rid in self.gpu_reqs_rids:
                self.gpu_reqs_rids.remove(request.rid)
        
        new_bs = len(self.reqs)
        if self.policy.gpu_attention_ratio != 0:
            self.policy.gpu_attention_micro_batch_size = int(new_bs * self.policy.gpu_attention_ratio)
            self.policy.gpu_attention_nano_batch_size = min(self.policy.gpu_attention_nano_batch_size, new_bs)


    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Req] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] is not chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        self.reqs = [self.reqs[i] for i in keep_indices]
        self.cpu_req_pool_indices = self.cpu_req_pool_indices[keep_indices_device]
        # TODO: 释放request的gpu/cpu req_to_token_pool

        self.seq_lens = self.seq_lens[keep_indices_device]
        self.out_cache_loc = None
        # XXX
        self.is_req_on_gpu = [self.is_req_on_gpu[i] for i in keep_indices]
        self.gpu_seq_lens_sum = sum(self.seq_lens[i] for i in self.gpu_reqs_id)
        self.cpu_seq_lens_sum = sum(self.seq_lens[i] for i in self.cpu_reqs_id)
        self.output_ids = self.output_ids[keep_indices_device]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        # self.has_stream = any(req.stream for req in self.reqs)
        # self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(keep_indices_device)

    def prepare_for_extend(self):
        logger.debug("prepare_for_extend")
        self.forward_mode = ForwardMode.EXTEND

        # Allocate req slots
        bs = len(self.reqs)
        cpu_req_pool_indices = self.cpu_alloc_req_slots(bs)
        gpu_req_pool_indices = self.gpu_alloc_req_slots(bs)

        # Init tensors
        reqs = self.reqs
        input_ids = [r.fill_ids for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        # logger.debug(f"seq_lens: {seq_lens}")
        prefix_lens = [0 for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]

        cpu_req_pool_indices_tensor = torch.tensor(cpu_req_pool_indices, dtype=torch.int64) # should located in CPU
        gpu_req_pool_indices_tensor = torch.tensor(gpu_req_pool_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        prefix_lens_tensor = torch.tensor(
            prefix_lens, dtype=torch.int64, device=self.device
        )
        extend_lens_tensor = seq_lens_tensor - prefix_lens_tensor

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.cpu_req_pool_idx = cpu_req_pool_indices[i]
            req.gpu_req_pool_idx = gpu_req_pool_indices[i]

            assert seq_len - pre_len == req.extend_input_len

            if pre_len > 0:
                assert False

        # Allocate memory
        cpu_out_cache_loc = self.cpu_alloc_token_slots_prefill()
        # logger.debug("cpu_out_cache_loc: %s", cpu_out_cache_loc)
        gpu_out_cache_loc, origin_gpu_kv_cache_state = self.gpu_alloc_token_slots(extend_num_tokens, backup_state=True)
        self.gpu_token_to_kv_pool_allocator.restore_state(origin_gpu_kv_cache_state) # XXX: 恢复GPU kv cache的状态，让下一个micro-batch从头开始分配

        # Set fields
        self.input_ids = input_ids_tensor
        self.cpu_req_pool_indices = cpu_req_pool_indices_tensor
        self.gpu_req_pool_indices = gpu_req_pool_indices_tensor
        self.seq_lens = seq_lens_tensor
        self.cpu_seq_lens = []
        self.gpu_seq_lens = self.seq_lens # in prefill, all requests run on GPU
        self.cpu_out_cache_loc = cpu_out_cache_loc
        self.gpu_out_cache_loc = gpu_out_cache_loc
        self.gpu_seq_lens_sum = self.gpu_seq_lens.sum()
        self.cpu_seq_lens_sum = 0

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        # self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # Write to req_to_token_pool
        if global_server_args_dict["attention_backend"] != "torch_native":
            # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)

            write_req_to_token_pool_triton[(bs,)](
                self.gpu_req_to_token_pool.req_to_token,
                gpu_req_pool_indices_tensor,
                prefix_lens_tensor,
                seq_lens_tensor,
                extend_lens_tensor,
                gpu_out_cache_loc,
                self.gpu_req_to_token_pool.req_to_token.shape[1],
            )

            pt = 0
            for i in range(bs):
                # logger.debug(f"req {i}, prefix_lens: {prefix_lens[i]}, seq_lens: {seq_lens[i]}, extend_lens: {extend_lens[i]}, cpu_req_pool_indices: {cpu_req_pool_indices[i]}")
                self.cpu_req_to_token_pool.write(
                    (cpu_req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    torch.tensor(cpu_out_cache_loc[pt : pt + extend_lens[i]], dtype=torch.int32),
                )
                pt += extend_lens[i]
        else:
            assert False
            pt = 0
            for i in range(bs):
                self.req_to_token_pool.write(
                    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                    out_cache_loc[pt : pt + extend_lens[i]],
                )
                pt += extend_lens[i]

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)
        
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )
        
        if self.spec_algorithm.is_eagle():
            # if spec decoding is used, the decode batch is prepared inside
            # `forward_batch_speculative_generation` after running draft models.
            # logger.debug("Eagle is used, skip prepare_for_decode")
            return

        # gpu_req_num = math.ceil(len(self.reqs)*self.policy.gpu_attention_ratio)
        gpu_req_num = self.policy.gpu_attention_micro_batch_size
        gpu_nano_bs = self.policy.gpu_attention_nano_batch_size
        gpu_bs = gpu_req_num
        cpu_bs = len(self.reqs) - gpu_bs

        # Update fields
        # 根据r.output_ids[-1]来确定input_ids
        self.input_ids = [r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs]
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.int32, device='cuda')
        self.output_ids = None

        self.seq_lens = torch.tensor([len(r.fill_ids) for r in self.reqs], dtype=torch.int64, device=self.device)
        is_req_on_gpu_tensor = torch.tensor(self.is_req_on_gpu, dtype=torch.bool, device=self.device)
        self.cpu_seq_lens = self.seq_lens[~is_req_on_gpu_tensor].to(device='cpu')
        self.gpu_seq_lens = self.seq_lens[is_req_on_gpu_tensor]
        # logger.debug(f"ScheduleBatch::prepare_for_decode bs: {bs}")
        # logger.debug(f"ScheduleBatch::prepare_for_decode seq_lens: {self.seq_lens}")
        
        self.gpu_seq_lens_sum = self.gpu_seq_lens.sum()
        self.cpu_seq_lens_sum = self.cpu_seq_lens.sum()

        self.seq_lens_sum = self.gpu_seq_lens_sum + self.cpu_seq_lens_sum

        # allocate CPU KV slot for requets run attention on CPU
        self.cpu_out_cache_loc = self.cpu_alloc_token_slots_decode([r.rid for r in self.reqs], [1] * bs)
        for idx in range(bs):
            if not self.is_req_on_gpu[idx]:
                self.reqs[idx].gpu_req_pool_idx = None

        # allocate GPU KV slot for requets run attention on GPU
        if gpu_bs > 0:
            # TODO: 分配的长度要是seq_len-1，因为新生成的token还没算好kv cache呢
            self.gpu_req_pool_indices = torch.tensor(self.gpu_alloc_req_slots(gpu_bs), dtype=torch.int64).to(
                self.device, non_blocking=True
            )
            
            i = 0
            for idx in range(bs):
                if self.is_req_on_gpu[idx]:
                    self.reqs[idx].gpu_req_pool_idx = self.gpu_req_pool_indices[i]
                i += 1

            self.gpu_kv_cache_loc = []
            self.gpu_out_cache_loc = []

            for i in range((gpu_bs + gpu_nano_bs - 1) // gpu_nano_bs):
                begin = i * gpu_nano_bs
                end = min(begin + gpu_nano_bs, gpu_bs)

                nano_bs = end - begin
                len_sum = sum(self.gpu_seq_lens[begin:end])
                nano_seq_lens = self.gpu_seq_lens[begin:end]

                gpu_out_cache_loc = self.gpu_alloc_token_slots(len_sum - nano_bs, backup_state=False) # 因为新生成的token没有kv cache，所以减去nano_bs
                self.gpu_kv_cache_loc.append(gpu_out_cache_loc)
                
                write_req_to_token_pool_triton[(nano_bs,)](
                    req_to_token_ptr=self.gpu_req_to_token_pool.req_to_token,
                    req_pool_indices=self.gpu_req_pool_indices[begin:end],
                    pre_lens=torch.tensor([0] * nano_bs, dtype=torch.int64).to(
                        self.device, non_blocking=True
                    ),
                    seq_lens=torch.tensor(nano_seq_lens - 1, dtype=torch.int64).to(
                        self.device, non_blocking=True
                    ),
                    extend_lens=torch.tensor(nano_seq_lens - 1, dtype=torch.int64).to(
                        self.device, non_blocking=True
                    ),
                    out_cache_loc=gpu_out_cache_loc,
                    req_to_token_ptr_stride=self.gpu_req_to_token_pool.req_to_token.shape[1],
                )

                gpu_out_cache_loc = self.gpu_alloc_token_slots(nano_bs) 
                self.gpu_out_cache_loc.append(gpu_out_cache_loc)
                # print(f"gpu_out_cache_loc: {self.gpu_out_cache_loc[-1]}")
                # print(f"begin: {begin}, end: {end}")
                # print(f"nano_seq_lens: {nano_seq_lens}")
                self.gpu_req_to_token_pool.write(
                    (self.gpu_req_pool_indices[begin:end], nano_seq_lens - 1), self.gpu_out_cache_loc[-1].to(torch.int32) # XXX: Note, -1 !
                )
                self.gpu_token_to_kv_pool_allocator.clear()

    def cleanup_after_prefill(self):
        # logger.debug("cleanup_after_prefill")
        # 释放request的gpu kv cache slot
        self.gpu_req_to_token_pool.free(self.gpu_req_pool_indices.tolist())
        # logger.debug(f"gpu_out_cache_loc: {self.gpu_out_cache_loc}")
        self.gpu_token_to_kv_pool_allocator.free(self.gpu_out_cache_loc)
    
    def cleanup_after_decode(self):
        # XXX: 这个cleanup必要吗，如果更改GPU kv cache slot的分配形式，让GPU KV cache slot和CPU KV cache slot保持一致行不行？好像有点麻烦。
        # logger.debug("cleanup_after_decode")
        # 释放request的gpu kv cache slot
        if self.gpu_req_pool_indices is not None:
            self.gpu_req_to_token_pool.free(self.gpu_req_pool_indices.tolist())
            # logger.debug(f"gpu_out_cache_loc: {self.gpu_out_cache_loc}")
            if self.gpu_out_cache_loc is not None:
                for out_loc in self.gpu_out_cache_loc:
                    self.gpu_token_to_kv_pool_allocator.free(out_loc)
            if hasattr(self, 'gpu_kv_cache_loc') and self.gpu_kv_cache_loc is not None:
                for kv_loc in self.gpu_kv_cache_loc:
                    self.gpu_token_to_kv_pool_allocator.free(kv_loc)
            # self.gpu_token_to_kv_pool_allocator.restore_state(self.gpu_kv_cache_backup)

    def cpu_alloc_req_slots(self, num_reqs: int):
        req_pool_indices = self.cpu_req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "alloc_req_slots runs out of memory. "
                "Please set a smaller number for `--max-running-requests`. "
                f"{self.cpu_req_to_token_pool.available_size()=}, "
                f"{num_reqs=}, "
            )
        return req_pool_indices

    def gpu_alloc_req_slots(self, num_reqs: int):
        req_pool_indices = self.gpu_req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError(
                "alloc_req_slots runs out of memory. "
            )
        return req_pool_indices

    def cpu_alloc_token_slots_prefill(self, backup_state: bool = False):
        out_cache_loc_list = []
        error = False
        for rq in self.reqs:
            rid = rq.rid
            alloc_res = self.cpu_token_to_kv_pool_allocator.alloc_for_a_new_request(rid, len(rq.fill_ids))
            if alloc_res is None: 
                error = True
                break
            out_cache_loc_list.extend(alloc_res)

        if backup_state:
            state = self.cpu_token_to_kv_pool_allocator.backup_state()

        if error:
            phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
            error_msg = (
                f"{phase_str} out of memory. Try to lower your batch size.\n"
                f"Avaliable tokens: {self.gpu_token_to_kv_pool_allocator.available_size()}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc_list, state
        else:
            return out_cache_loc_list

    # def cpu_alloc_token_slots_decode(self, num_tokens: List[int], backup_state: bool = False):
    def cpu_alloc_token_slots_decode(self, rids: List[int], num_tokens: List[int], backup_state: bool = False):
        # if self.cpu_token_to_kv_pool_allocator.available_size() < sum(num_tokens):
        #     logger.error(f"cpu_alloc_token_slots runs out of memory. {num_tokens=}")

        if backup_state:
            state = self.cpu_token_to_kv_pool_allocator.backup_state()
        
        out_cache_loc_list = []
        error = False
        for rid, num_token in zip(rids, num_tokens):
            alloc_res = self.cpu_token_to_kv_pool_allocator.alloc(rid, num_token)
            if alloc_res is None: 
                error = True
                break
            out_cache_loc_list.extend(alloc_res)


        if error:
            phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
            error_msg = (
                f"{phase_str} out of memory. Try to lower your batch size.\n"
                # f"Begin: {self.cpu_token_to_kv_pool_allocator.rid_to_kvcache_begin}\n"
                # f"Now: {self.cpu_token_to_kv_pool_allocator.rid_to_kvcache_now}\n"
                # f"End: {self.cpu_token_to_kv_pool_allocator.rid_to_kvcache_end}\n"
                # f"Try to allocate: {num_tokens}\n"
                # f"out_cache_loc_list: {out_cache_loc_list}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc_list, state
        else:
            return out_cache_loc_list

    def gpu_alloc_token_slots(self, num_tokens: int, backup_state: bool = False, specified_allocator: Optional[TokenToKVPoolAllocatorGPU] = None):
        if specified_allocator is None:
            allocator = self.gpu_token_to_kv_pool_allocator
        else:
            allocator = specified_allocator
        if allocator.available_size() < num_tokens:
            logger.error(f"gpu_alloc_token_slots runs out of memory. {num_tokens=}")

        if backup_state:
            state = allocator.backup_state()

        out_cache_loc = allocator.alloc(num_tokens)
        if out_cache_loc is None:
            phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
            error_msg = (
                f"{phase_str} out of memory. Try to lower your batch size.\n"
                f"Try to allocate {num_tokens} tokens.\n"
                f"Avaliable tokens: {allocator.available_size()}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if backup_state:
            return out_cache_loc, state
        else:
            return out_cache_loc
    
    def split_for_draft(self, gpu_req_num: int) -> Tuple["ScheduleBatch", "ScheduleBatch"]:
        """
        将当前batch分裂为GPU执行batch和CPU执行batch
        
        Args:
            gpu_req_num: GPU执行的request数量
            
        Returns:
            Tuple[gpu_batch, cpu_batch]: GPU执行batch和CPU执行batch
        """
        if gpu_req_num > 0:
            # 创建GPU batch
            gpu_batch = ScheduleBatch.init_new(
                reqs=self.reqs[:gpu_req_num],
                is_req_on_gpu=[True] * gpu_req_num,
                gpu_req_to_token_pool=self.gpu_req_to_token_pool,
                gpu_token_to_kv_pool_allocator=self.gpu_token_to_kv_pool_allocator,
                cpu_req_to_token_pool=self.cpu_req_to_token_pool,
                cpu_token_to_kv_pool_allocator=self.cpu_token_to_kv_pool_allocator,
                model_config=self.model_config,
                spec_algorithm=self.spec_algorithm,
                policy=self.policy,
                split_for_draft=True,
            )
            
            if self.seq_lens is not None:
                gpu_batch.seq_lens = self.seq_lens[:gpu_req_num]
                gpu_batch.seq_lens_sum = gpu_batch.seq_lens.sum().item()
            
            if self.cpu_seq_lens is not None:
                gpu_batch.cpu_seq_lens = torch.tensor([], dtype=self.cpu_seq_lens.dtype, device=self.cpu_seq_lens.device)
                gpu_batch.cpu_seq_lens_sum = 0
                
            if self.gpu_seq_lens is not None:
                gpu_batch.gpu_seq_lens = self.seq_lens[:gpu_req_num] if self.seq_lens is not None else self.gpu_seq_lens[:gpu_req_num]
                gpu_batch.gpu_seq_lens_sum = gpu_batch.gpu_seq_lens.sum().item() if len(gpu_batch.gpu_seq_lens) > 0 else 0
            
            # if self.input_ids is not None:
            #     extend_lens_gpu = self.extend_lens[:gpu_req_num] if self.extend_lens else []
            #     gpu_extend_tokens = sum(extend_lens_gpu)
            #     gpu_batch.input_ids = self.input_ids[:gpu_extend_tokens]
            #     gpu_batch.extend_lens = extend_lens_gpu
            #     gpu_batch.extend_num_tokens = gpu_extend_tokens
            
            if self.sampling_info is not None:
                gpu_batch.sampling_info = self.sampling_info
            
            # 分裂spec_info
            if self.spec_info is not None:
                gpu_batch.spec_info = EagleDraftInput()
                gpu_batch.spec_info.accept_length = self.spec_info.accept_length[:gpu_req_num]
                gpu_process_verified = (gpu_batch.spec_info.accept_length + 1).sum()
                gpu_batch.spec_info.verified_id = self.spec_info.verified_id[:gpu_process_verified]
                gpu_batch.spec_info.accept_length_cpu = self.spec_info.accept_length_cpu[:gpu_req_num]
                gpu_batch.spec_info.seq_lens_for_draft_extend = self.spec_info.seq_lens_for_draft_extend[:gpu_req_num]
            
            if hasattr(self.spec_info, 'first_step_after_prefill'):
                gpu_batch.spec_info.first_step_after_prefill = self.spec_info.first_step_after_prefill
            
            gpu_batch.out_cache_loc = self.out_cache_loc
        else:
            gpu_batch = None
        
        # 创建CPU batch
        cpu_req_num = len(self.reqs) - gpu_req_num
        if cpu_req_num > 0:
            cpu_batch = ScheduleBatch.init_new(
                reqs=self.reqs[gpu_req_num:],
                is_req_on_gpu=[False] * cpu_req_num,
                gpu_req_to_token_pool=self.gpu_req_to_token_pool,
                gpu_token_to_kv_pool_allocator=self.gpu_token_to_kv_pool_allocator,
                cpu_req_to_token_pool=self.cpu_req_to_token_pool,
                cpu_token_to_kv_pool_allocator=self.cpu_token_to_kv_pool_allocator,
                model_config=self.model_config,
                spec_algorithm=self.spec_algorithm,
                policy=self.policy,
                split_for_draft=True,
            )
            
            # 分裂相关tensor
            if self.seq_lens is not None:
                cpu_batch.seq_lens = self.seq_lens[gpu_req_num:]
                cpu_batch.seq_lens_sum = cpu_batch.seq_lens.sum().item()
            
            if self.cpu_seq_lens is not None:
                cpu_batch.cpu_seq_lens = self.seq_lens[gpu_req_num:] if self.seq_lens is not None else self.cpu_seq_lens[gpu_req_num:]
                cpu_batch.cpu_seq_lens_sum = cpu_batch.cpu_seq_lens.sum().item() if len(cpu_batch.cpu_seq_lens) > 0 else 0
                
            if self.gpu_seq_lens is not None:
                cpu_batch.gpu_seq_lens = torch.tensor([], dtype=self.gpu_seq_lens.dtype, device=self.gpu_seq_lens.device)
                cpu_batch.gpu_seq_lens_sum = 0
            
            # # 对于extend模式，需要根据extend_lens来分裂input_ids
            # extend_lens_cpu = self.extend_lens[gpu_req_num:] if self.extend_lens else []
            # gpu_extend_tokens = sum(self.extend_lens[:gpu_req_num]) if self.extend_lens else 0
            # cpu_batch.input_ids = self.input_ids[gpu_extend_tokens:]
            # cpu_batch.extend_lens = extend_lens_cpu
            # cpu_batch.extend_num_tokens = sum(extend_lens_cpu)
            
            # 分裂sampling info
            if self.sampling_info is not None:
                cpu_batch.sampling_info = self.sampling_info
            
            # 分裂spec_info
            if self.spec_info is not None:
                cpu_batch.spec_info = EagleDraftInput()
                cpu_batch.spec_info.accept_length = self.spec_info.accept_length[gpu_req_num:]
                gpu_process_verified = (gpu_batch.spec_info.accept_length + 1).sum()
                cpu_batch.spec_info.verified_id = self.spec_info.verified_id[gpu_process_verified:]
                cpu_batch.spec_info.accept_length_cpu = self.spec_info.accept_length_cpu[gpu_req_num:]
                cpu_batch.spec_info.seq_lens_for_draft_extend = self.spec_info.seq_lens_for_draft_extend[gpu_req_num:]
            
            if hasattr(self.spec_info, 'first_step_after_prefill'):
                gpu_batch.spec_info.first_step_after_prefill = self.spec_info.first_step_after_prefill
            
            gpu_batch.out_cache_loc = self.out_cache_loc
        else:
            cpu_batch = None
            
        return gpu_batch, cpu_batch
    
    def get_seq_len_by_rid(self, rid: int):
        if self.gpu_reqs_rids is not None:
            if rid in self.gpu_reqs_rids:
                return self.gpu_seq_lens[self.gpu_reqs_rids.index(rid)]
        if self.cpu_reqs_rids is not None:
            if rid in self.cpu_reqs_rids:
                return self.cpu_seq_lens[self.cpu_reqs_rids.index(rid)]
        return None


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # Policy
    policy: Policy
    # The request location
    # gpu_reqs_id: torch.Tensor # list[int]
    # cpu_reqs_id: torch.Tensor # list[int]
    cpu_reqs_rids: List[int] # list[int]
    gpu_reqs_rids: List[int] # list[int]
    # The indices of requests in the req_to_token_pool
    cpu_req_pool_indices: torch.Tensor
    gpu_req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    cpu_seq_lens: Optional[torch.Tensor]
    gpu_seq_lens: Optional[torch.Tensor]
    # The indices of output tokens in the token_to_kv_pool_allocator
    cpu_out_cache_loc: torch.Tensor
    gpu_out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int
    cpu_seq_lens_sum: int
    gpu_seq_lens_sum: int

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # The input Embeds
    input_embeds: Optional[torch.tensor] = None

    # Speculative decoding
    spec_algorithm: "SpeculativeAlgorithm" = None
    spec_info: Optional[Union["EagleVerifyInput", "EagleDraftInput"]] = None
    # # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None

    def sample(self, logits: torch.Tensor):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(self.sampling_info.temperatures)
        # logits.add_(self.logit_bias)

        probs = torch.softmax(logits, dim=-1)
        # print(probs.shape)
        # print(self.sampling_info.top_ps.shape)
        # print(self.sampling_info.top_ks.shape)
        # print(self.sampling_info.top_ps)
        # print(self.sampling_info.top_ks)
        probs_sort, probs_idx = _top_p_top_k(probs, self.sampling_info.top_ps, self.sampling_info.top_ks)
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(
            -1
        )
        batch_next_token_probs = torch.gather(
            probs_sort, dim=1, index=sampled_index
        ).view(-1)

        # logger.debug(f"batch_next_token_ids: {batch_next_token_ids}")
        # logger.debug(f"batch_next_token_probs: {batch_next_token_probs}")

        return batch_next_token_ids, batch_next_token_probs


@torch.compile(dynamic=True, backend="inductor")
def get_last_loc(req_to_token, req_pool_indices_tensor, prefix_lens_tensor):
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )

@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,  # total request length
    extend_lens,  # total request length - prefix hit length
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )
def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    top_ps = top_ps.view(-1, 1)
    top_ks = top_ks.view(-1, 1)
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    # logger.debug(f"probs_sort: {probs_sort}")
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= top_ks
    ] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    return probs_sort, probs_idx
