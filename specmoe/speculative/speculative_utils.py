import torch
import torch.nn.functional as F
import math
from typing import List, Optional, TYPE_CHECKING, Tuple
from dataclasses import dataclass
import triton
import triton.language as tl
import logging
logger = logging.getLogger(__name__)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.utils import next_power_of_2, fast_topk
from sglang.srt.speculative.build_eagle_tree import build_tree_kernel_efficient
from sgl_kernel import (
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)

from specmoe.utils.data_info import ScheduleBatch, CaptureHiddenMode, write_req_to_token_pool_triton
from specmoe.backend.memory import ReqToTokenPool, TokenToKVPoolAllocatorGPU, TokenToKVPoolAllocatorCPU
from specmoe.utils.data_info import get_last_loc
from specmoe.utils.data_info import global_server_args_dict
from specmoe.backend.policy import Policy, SpecPolicy
from specmoe.layers.logits_processor import LogitsProcessorOutput

@dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    # For GPU Requests
    topk_p_gpu: torch.Tensor = None
    topk_index_gpu: torch.Tensor = None
    hidden_states_gpu: torch.Tensor = None # shape: (b, hidden_size)
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None

    seq_lens_for_draft_extend: torch.Tensor = None # 用于extend阶段，表示当前req的seq len，已经算上accepted token了

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    all_padding_lens: Optional[torch.Tensor] = None

    first_step_after_prefill: bool = False

    # for CPU attention backend init in CPU Draft Genration
    seq_len_for_draft_cpu_attention_init: torch.Tensor = None
    custom_mask_for_draft_cpu_attention_init: torch.Tensor = None
    mask_indptr_for_draft_cpu_attention_init: torch.Tensor = None
    qo_indptr_for_draft_cpu_attention_init: torch.Tensor = None
    mask_len_for_draft_cpu_attention_init: int = None

    def split_to_one_req(self, i: int):
        return EagleDraftInput(
            topk_p=self.topk_p[i],
            topk_index=self.topk_index[i],
            hidden_states=self.hidden_states[i],
            topk_p_gpu=self.topk_p_gpu[i] if self.topk_p_gpu is not None else None,
            topk_index_gpu=self.topk_index_gpu[i] if self.topk_index_gpu is not None else None,
            capture_hidden_mode=self.capture_hidden_mode,
            verified_id=self.verified_id[i],
            first_step_after_prefill=self.first_step_after_prefill,
        )
    
    @classmethod
    def merge_spec_infos(cls, spec_infos: List["EagleDraftInput"], gpu_req_num: int): # 用于在prefill结束后，decode开始前，将多个req的spec_info合并成一个
        # for spec_info in spec_infos:
        #     logger.debug(f"spec_info: \n hidden_states: {spec_info.hidden_states.shape}, topk_p: {spec_info.topk_p.shape}, topk_index: {spec_info.topk_index.shape}")
        return cls(
            topk_p=torch.cat([spec_info.topk_p.unsqueeze(0) for spec_info in spec_infos[gpu_req_num:]]),
            topk_index=torch.cat([spec_info.topk_index.unsqueeze(0) for spec_info in spec_infos[gpu_req_num:]]),
            hidden_states=torch.cat([spec_info.hidden_states.unsqueeze(0) for spec_info in spec_infos[gpu_req_num:]]),
            topk_p_gpu=torch.cat([spec_info.topk_p_gpu.unsqueeze(0) for spec_info in spec_infos[:gpu_req_num]]) if gpu_req_num > 0 else None,
            topk_index_gpu=torch.cat([spec_info.topk_index_gpu.unsqueeze(0) for spec_info in spec_infos[:gpu_req_num]]) if gpu_req_num > 0 else None,
            hidden_states_gpu=torch.cat([spec_info.hidden_states.unsqueeze(0) for spec_info in spec_infos[:gpu_req_num]]) if gpu_req_num > 0 else None,
            capture_hidden_mode=spec_infos[0].capture_hidden_mode,
            verified_id=torch.cat([spec_info.verified_id.unsqueeze(0) for spec_info in spec_infos]),
            first_step_after_prefill=spec_infos[0].first_step_after_prefill,
        )

    def prepare_for_extend(self, batch: ScheduleBatch):
        # Prefill only generate 1 token.
        assert len(self.verified_id) == len(batch.seq_lens)

        # 将1到seq_lens的tokenid，与target model在prefill阶段生成的tokenid拼在一起
        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
        draft_model_placement: str,
        gpu_execution: bool = True,
    ):
        if gpu_execution and batch.policy.draft_gpu_req_num > 0:
            accept_length_cpu = batch.spec_info.accept_length_cpu
            batch.extend_lens = [x + 1 for x in accept_length_cpu]
            batch.extend_num_tokens = sum(batch.extend_lens)
            batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend # 这个seq_lens就已经包含verified tokens了
            seq_lens_cpu = batch.seq_lens.tolist()

            new_verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)
            self.accept_length += 1
            self.positions = torch.empty_like(self.verified_id, dtype=torch.long)

            create_extend_spec_info[(self.accept_length.numel(),)](
                self.verified_id,
                batch.seq_lens,
                self.accept_length,
                torch.cumsum(self.accept_length, axis=0, dtype=torch.int),
                self.positions,
                new_verified_id,
                next_power_of_2(speculative_num_steps + 1),
            )
        else:
            # accept_length_cpu = batch.spec_info.accept_length_cpu
            # 用accept_length而不是accept_length_cpu，可以避免在ForwardBatch.init_new()中的一次HtoD传输，虽然很短，但是有可能被GPU Draft阻塞
            accept_length = batch.spec_info.accept_length
            batch.extend_lens = accept_length + 1
            batch.extend_num_tokens = batch.extend_lens.sum()
            batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend # 这个seq_lens就已经包含verified tokens了
            seq_lens_cpu = batch.seq_lens.tolist()

            new_verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)
            self.accept_length += 1
            self.positions = torch.empty_like(self.verified_id, dtype=torch.long)

                
            bs = self.accept_length.numel()
            accept_len_cum = torch.cumsum(self.accept_length, dim=0, dtype=torch.int)

            start_pos = batch.seq_lens - self.accept_length
            
            # 2. Create global indices for all positions
            total_positions = accept_len_cum[-1].item()
            indices = torch.arange(total_positions, dtype=torch.long, device=self.positions.device)
            
            # 3. Find which request each position belongs to
            request_ids = torch.searchsorted(accept_len_cum, indices, right=True)
            
            # 4. Calculate offset within each request
            offsets = torch.cat([torch.zeros(1, dtype=torch.int, device=self.accept_length.device), 
                                accept_len_cum[:-1]])
            position_offsets = indices - offsets[request_ids]
            
            # 5. Compute actual position values
            actual_positions = start_pos[request_ids] + position_offsets
            
            # 6. Assign to positions array
            self.positions[:total_positions] = actual_positions
            
            # 7. Select the last verified_id for each request (vectorized)
            verified_id_offsets = accept_len_cum - 1
            new_verified_id = self.verified_id[verified_id_offsets] 

        batch.seq_lens_sum = sum(seq_lens_cpu)
        batch.input_ids = self.verified_id
        self.verified_id = new_verified_id 
        
        return batch

    
    def init_draft_extend_cpu_attention_metadata(self, batch: ScheduleBatch):
        bs = len(batch.seq_lens)
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int64, device="cpu")
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0) # 这里的accept_length加过1了
        seq_lens = batch.seq_lens.clone().detach().to(device="cpu", dtype=torch.int64)
        # seq_lens = seq_lens + self.accept_length.to(device='cpu', dtype=torch.int64) # XXX: 需要检查一下在纯CPU计算时，注释掉这一行是否还正确
        return seq_lens, qo_indptr

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = self.accept_length.numel()

        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # TODO: replace cum_kv_seq_len[-1] with paged_kernel_lens_sum to avoid the device sync.
        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def filter_batch(self, new_indices: torch.Tensor):
        self.topk_p = self.topk_p[: len(new_indices)]
        self.topk_index = self.topk_index[: len(new_indices)]
        self.hidden_states = self.hidden_states[: len(new_indices)]
        self.verified_id = self.verified_id[: len(new_indices)]


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepeted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepeted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepeted indices from logits_output.next_token_logits
    accepeted_indices: torch.Tensor


@dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    draft_token_num: int
    spec_steps: int
    capture_hidden_mode: CaptureHiddenMode
    # for gpu request
    custom_mask_gpu: torch.Tensor = None
    custom_mask_gpu_len: int = None
    retrive_index_gpu: torch.Tensor = None
    retrive_next_token_gpu: torch.Tensor = None
    retrive_next_sibling_gpu: torch.Tensor = None
    retrive_cum_len_gpu: torch.Tensor = None
    draft_token_num_gpu: int = 0
    spec_steps_gpu: int = 0

    @classmethod
    def merge(cls, gpu_verify_input: "EagleVerifyInput", cpu_verify_input: "EagleVerifyInput", merge_in_gpu_execution: bool = False):
        if gpu_verify_input is None:
            return cpu_verify_input

        if not merge_in_gpu_execution: # 合并
            if gpu_verify_input.retrive_index is None: # 这种情况下,gpu_verify_input只有gpu attention request的数据
                return cls(
                    draft_token=torch.cat([gpu_verify_input.draft_token, cpu_verify_input.draft_token]),
                    custom_mask=torch.cat([gpu_verify_input.custom_mask, cpu_verify_input.custom_mask]),
                    positions=torch.cat([gpu_verify_input.positions, cpu_verify_input.positions]),
                    retrive_index=cpu_verify_input.retrive_index,
                    retrive_next_token=cpu_verify_input.retrive_next_token,
                    retrive_next_sibling=cpu_verify_input.retrive_next_sibling,
                    retrive_cum_len=cpu_verify_input.retrive_cum_len,
                    draft_token_num=cpu_verify_input.draft_token_num,
                    spec_steps=cpu_verify_input.spec_steps,
                    capture_hidden_mode=gpu_verify_input.capture_hidden_mode,
                    custom_mask_gpu=gpu_verify_input.custom_mask_gpu,
                    custom_mask_gpu_len=gpu_verify_input.custom_mask.shape[0],
                    retrive_index_gpu=gpu_verify_input.retrive_index_gpu,
                    retrive_next_token_gpu=gpu_verify_input.retrive_next_token_gpu,
                    retrive_next_sibling_gpu=gpu_verify_input.retrive_next_sibling_gpu,
                    retrive_cum_len_gpu=gpu_verify_input.retrive_cum_len_gpu,
                    draft_token_num_gpu=gpu_verify_input.draft_token_num_gpu,
                    spec_steps_gpu=gpu_verify_input.spec_steps_gpu,
                )
            else: # 这种情况下gpu_verify_input也有部分cpu attn request的数据
                return cls(
                    draft_token=torch.cat([gpu_verify_input.draft_token, cpu_verify_input.draft_token]),
                    custom_mask=torch.cat([gpu_verify_input.custom_mask, cpu_verify_input.custom_mask]),
                    positions=torch.cat([gpu_verify_input.positions, cpu_verify_input.positions]),
                    retrive_index=torch.cat([gpu_verify_input.retrive_index, cpu_verify_input.retrive_index + gpu_verify_input.retrive_index.shape[0] * gpu_verify_input.retrive_index.shape[1]]),
                    retrive_next_token=torch.cat([gpu_verify_input.retrive_next_token, cpu_verify_input.retrive_next_token]),
                    retrive_next_sibling=torch.cat([gpu_verify_input.retrive_next_sibling, cpu_verify_input.retrive_next_sibling]),
                    retrive_cum_len=None, # XXX: check
                    draft_token_num=cpu_verify_input.draft_token_num,
                    spec_steps=cpu_verify_input.spec_steps,
                    capture_hidden_mode=gpu_verify_input.capture_hidden_mode,
                    custom_mask_gpu=gpu_verify_input.custom_mask_gpu,
                    custom_mask_gpu_len=gpu_verify_input.custom_mask_gpu_len,
                    retrive_index_gpu=gpu_verify_input.retrive_index_gpu,
                    retrive_next_token_gpu=gpu_verify_input.retrive_next_token_gpu,
                    retrive_next_sibling_gpu=gpu_verify_input.retrive_next_sibling_gpu,
                    retrive_cum_len_gpu=gpu_verify_input.retrive_cum_len_gpu,
                    draft_token_num_gpu=gpu_verify_input.draft_token_num_gpu,
                    spec_steps_gpu=gpu_verify_input.spec_steps_gpu,
                )
        else: # 这种情况下要将gpu_verify_input和cpu_verify_input合并成另一个gpu_verify_input，其中'_gpu'部分存放gpu_attn_requets数据，'_cpu'部分存放cpu_attn_requets数据
            if cpu_verify_input is None:
                return cls(
                    draft_token=gpu_verify_input.draft_token,
                    custom_mask=gpu_verify_input.custom_mask,
                    positions=gpu_verify_input.positions,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    draft_token_num=None,
                    spec_steps=None,
                    capture_hidden_mode=gpu_verify_input.capture_hidden_mode,
                    custom_mask_gpu=gpu_verify_input.custom_mask,
                    custom_mask_gpu_len=gpu_verify_input.custom_mask.shape[0],
                    retrive_index_gpu=gpu_verify_input.retrive_index,
                    retrive_next_token_gpu=gpu_verify_input.retrive_next_token,
                    retrive_next_sibling_gpu=gpu_verify_input.retrive_next_sibling,
                    retrive_cum_len_gpu=gpu_verify_input.retrive_cum_len,
                    draft_token_num_gpu=gpu_verify_input.draft_token_num,
                    spec_steps_gpu=gpu_verify_input.spec_steps,
                )
            else:
                return cls(
                    draft_token=torch.cat([gpu_verify_input.draft_token, cpu_verify_input.draft_token]),
                    custom_mask=torch.cat([gpu_verify_input.custom_mask, cpu_verify_input.custom_mask]),
                    positions=torch.cat([gpu_verify_input.positions, cpu_verify_input.positions]),
                    retrive_index=cpu_verify_input.retrive_index,
                    retrive_next_token=cpu_verify_input.retrive_next_token,
                    retrive_next_sibling=cpu_verify_input.retrive_next_sibling,
                    retrive_cum_len=cpu_verify_input.retrive_cum_len,
                    draft_token_num=cpu_verify_input.draft_token_num,
                    spec_steps=cpu_verify_input.spec_steps,
                    capture_hidden_mode=gpu_verify_input.capture_hidden_mode,
                    custom_mask_gpu=gpu_verify_input.custom_mask,
                    custom_mask_gpu_len=gpu_verify_input.custom_mask.shape[0],
                    retrive_index_gpu=gpu_verify_input.retrive_index,
                    retrive_next_token_gpu=gpu_verify_input.retrive_next_token,
                    retrive_next_sibling_gpu=gpu_verify_input.retrive_next_sibling,
                    retrive_cum_len_gpu=gpu_verify_input.retrive_cum_len,
                    draft_token_num_gpu=gpu_verify_input.draft_token_num,
                    spec_steps_gpu=gpu_verify_input.spec_steps,
                )

    @classmethod
    def create(
        cls,
        verified_id: torch.Tensor,
        score_list: List[torch.Tensor],
        token_list: List[torch.Tensor],
        parents_list: List[torch.Tensor],
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        topk: int,
        spec_steps: int,
        num_verify_tokens: int,
    ):
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id,
            score_list,
            token_list,
            parents_list,
            seq_lens,
            seq_lens_sum,
            topk,
            spec_steps,
            num_verify_tokens,
        )

        return cls(
            draft_tokens,
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            None,
            num_verify_tokens,
            spec_steps,
            CaptureHiddenMode.FULL,
        )

    def prepare_for_verify(self, batch: ScheduleBatch, policy: Policy):
        # 为draft tokens 分配 GPU/CPU token slots
        # 根据policy确定在CPU/GPU上的分配数量
        batch.input_ids = self.draft_token

        logger.debug(f"input_ids shape: {batch.input_ids.shape}")
        if policy.spec_policy == SpecPolicy.SequentialGPUonly:
            assert False, "Not Supported "
            batch.gpu_out_cache_loc = batch.gpu_alloc_token_slots(len(batch.input_ids))

            end_offset = batch.seq_lens + self.draft_token_num

            bs = batch.batch_size()
            assign_req_to_token_pool[(bs,)](
                batch.gpu_req_pool_indices,
                batch.gpu_req_to_token_pool.req_to_token,
                batch.seq_lens,
                end_offset,
                batch.gpu_out_cache_loc,
                batch.gpu_req_to_token_pool.req_to_token.shape[1],
                next_power_of_2(bs),
            )
        elif policy.spec_policy == SpecPolicy.SequentialCPUonly:
            if len(batch.gpu_seq_lens) > 0:
                assert False, "SequentialCPUOnly don't compatible with GPU attention"
            else:
                assert len(batch.input_ids) == len(batch.seq_lens) * self.draft_token_num
                batch.cpu_out_cache_loc = torch.tensor(batch.cpu_alloc_token_slots_decode([r.rid for r in batch.reqs], [self.draft_token_num] * len(batch.seq_lens)))
        elif policy.spec_policy == SpecPolicy.SequentialCGCoop:
            # assert len(batch.gpu_seq_lens) > 0, "SequentialCGCoop must have GPU attention part"
            # logger.debug(f"len(batch.input_ids): {len(batch.input_ids)}, len(batch.cpu_seq_lens): {len(batch.cpu_seq_lens)}, len(batch.gpu_seq_lens): {len(batch.gpu_seq_lens)}")
            assert len(batch.input_ids) == len(batch.cpu_seq_lens) * (0 if self.draft_token_num is None else self.draft_token_num) + len(batch.gpu_seq_lens) * (0 if self.draft_token_num_gpu is None else self.draft_token_num_gpu)
            # 在CPU上为所有request分配kv cache slot
            batch.cpu_out_cache_loc = torch.tensor(batch.cpu_alloc_token_slots_decode([r.rid for r in batch.reqs], [self.draft_token_num_gpu] * len(batch.gpu_seq_lens) + [self.draft_token_num] * len(batch.cpu_seq_lens)))

            # 在GPU上为gpu request分配prefix kv cache以及input id kv cache
            batch.gpu_req_pool_indices = torch.tensor(batch.gpu_alloc_req_slots(len(batch.gpu_seq_lens)), dtype=torch.int64).to(
                batch.device, non_blocking=True
            )
            if len(batch.gpu_seq_lens) > 0:
                gpu_req_num = batch.policy.gpu_attention_micro_batch_size
                gpu_nano_bs = batch.policy.gpu_attention_nano_batch_size
                gpu_bs = gpu_req_num
                batch.gpu_kv_cache_loc = []
                batch.gpu_out_cache_loc = []

                for i in range((gpu_bs + gpu_nano_bs - 1) // gpu_nano_bs):
                    begin = i * gpu_nano_bs
                    end = min(begin + gpu_nano_bs, gpu_bs)

                    nano_bs = end - begin
                    len_sum = sum(batch.gpu_seq_lens[begin:end])
                    nano_seq_lens = batch.gpu_seq_lens[begin:end]

                    kv_cache_loc = batch.gpu_alloc_token_slots(len_sum, backup_state=False) 
                    batch.gpu_kv_cache_loc.append(kv_cache_loc)

                    write_req_to_token_pool_triton[(nano_bs,)](
                        req_to_token_ptr=batch.gpu_req_to_token_pool.req_to_token,
                        req_pool_indices=batch.gpu_req_pool_indices[begin:end],
                        pre_lens=torch.tensor([0] * nano_bs, dtype=torch.int64).to(
                            batch.device, non_blocking=True
                        ),
                        seq_lens=nano_seq_lens.clone().detach().to(
                            batch.device, non_blocking=True
                        ),
                        extend_lens=nano_seq_lens.clone().detach().to(
                            batch.device, non_blocking=True
                        ),
                        out_cache_loc=kv_cache_loc,
                        req_to_token_ptr_stride=batch.gpu_req_to_token_pool.req_to_token.shape[1],
                    )

                    gpu_out_cache_loc = batch.gpu_alloc_token_slots(nano_bs * self.draft_token_num_gpu) 
                    batch.gpu_out_cache_loc.append(gpu_out_cache_loc)
                    assign_req_to_token_pool[(nano_bs,)](
                        batch.gpu_req_pool_indices[begin:end],
                        batch.gpu_req_to_token_pool.req_to_token,
                        nano_seq_lens,
                        nano_seq_lens + self.draft_token_num_gpu,
                        batch.gpu_out_cache_loc[-1],
                        batch.gpu_req_to_token_pool.req_to_token.shape[1],
                        next_power_of_2(nano_bs),
                    )
                    batch.gpu_token_to_kv_pool_allocator.clear()

                # batch.gpu_kv_cache_loc = batch.gpu_alloc_token_slots(batch.gpu_seq_lens_sum, backup_state=False) # 因为新生成的token没有kv cache

                # batch.gpu_out_cache_loc = batch.gpu_alloc_token_slots(len(batch.gpu_seq_lens) * self.draft_token_num_gpu)

                # write_req_to_token_pool_triton[(len(batch.gpu_seq_lens),)](
                #     req_to_token_ptr=batch.gpu_req_to_token_pool.req_to_token,
                #     req_pool_indices=batch.gpu_req_pool_indices,
                #     pre_lens=torch.tensor([0] * len(batch.gpu_seq_lens), dtype=torch.int64).to(
                #         batch.device, non_blocking=True
                #     ),
                #     seq_lens=batch.gpu_seq_lens.clone().detach().to(
                #         batch.device, non_blocking=True
                #     ),
                #     extend_lens=batch.gpu_seq_lens.clone().detach().to(
                #         batch.device, non_blocking=True
                #     ),
                #     out_cache_loc=batch.gpu_kv_cache_loc,
                #     req_to_token_ptr_stride=batch.gpu_req_to_token_pool.req_to_token.shape[1],
                # )

                # assign_req_to_token_pool[(len(batch.gpu_seq_lens),)](
                #     batch.gpu_req_pool_indices,
                #     batch.gpu_req_to_token_pool.req_to_token,
                #     batch.gpu_seq_lens,
                #     batch.gpu_seq_lens + self.draft_token_num_gpu,
                #     batch.gpu_out_cache_loc,
                #     batch.gpu_req_to_token_pool.req_to_token.shape[1],
                #     next_power_of_2(len(batch.gpu_seq_lens)),
                # )
        else:
            assert False, f"spec policy: {policy.spec_policy} not supported yet"

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device="cuda",
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device="cuda",
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask
    
    def drop_accepted_tokens(self, accept_index: torch.Tensor, accept_length: torch.Tensor, ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        用于控制accept rate的消融实验。
        每个accepted token有ratio的几率被丢弃，从一个request的accepted tokens中从后往前开始算，期望每个request被丢弃ratio/(1-ratio)个accepted tokens。
        '''
        if ratio == 0:
            return accept_index, accept_length
        bs = accept_length.shape[0]
        for i in range(bs):
            drop_count = 0
            for j in range(accept_length[i]):
                if torch.rand(1) < ratio:
                    drop_count += 1
            accept_length[i] -= drop_count
            accept_index[i, accept_length[i]+1:] = -1
        return accept_index, accept_length

    def verify_func(self, batch: ScheduleBatch, logits_output: torch.Tensor, token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU, process_cpu_requets: bool = True, experiment_ablation_drop_accepted_tokens_ratio: float = 0) -> torch.Tensor:
        if process_cpu_requets:
            bs = self.retrive_index.shape[0]
            candidates = self.draft_token[len(batch.gpu_seq_lens)*self.draft_token_num_gpu:].reshape(bs, self.draft_token_num)
        else:
            if len(batch.gpu_seq_lens) == 0:
                return None, None, None, None, None, None, None, None
            bs = self.retrive_index_gpu.shape[0]
            candidates = self.draft_token[:bs*self.draft_token_num_gpu].reshape(bs, self.draft_token_num_gpu)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device="cuda")
        accept_index = torch.full(
            (bs, self.spec_steps if process_cpu_requets else self.spec_steps_gpu + 1), -1, dtype=torch.int32, device="cuda"
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device="cuda")

        # Sample tokens
        # apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures[len(batch.gpu_seq_lens):] if process_cpu_requets else sampling_info.temperatures[:len(batch.gpu_seq_lens)], self.draft_token_num if process_cpu_requets else self.draft_token_num_gpu, dim=0
        )  # (bs * draft_token_num, 1)

        target_probs = F.softmax(
            logits_output.next_token_logits[len(batch.gpu_seq_lens)*self.draft_token_num_gpu:] if process_cpu_requets else logits_output.next_token_logits[:len(batch.gpu_seq_lens)*self.draft_token_num_gpu] / expanded_temperature, dim=-1
        )  # (bs * draft_token_num, vocab_size)
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ks, self.draft_token_num if process_cpu_requets else self.draft_token_num_gpu, dim=0
            ),
        )  # (bs * draft_token_num, vocab_size)
        target_probs = top_p_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ps, self.draft_token_num if process_cpu_requets else self.draft_token_num_gpu, dim=0
            ),
        )
        target_probs = target_probs.reshape(bs, self.draft_token_num if process_cpu_requets else self.draft_token_num_gpu, -1)

        draft_probs = torch.zeros(
            target_probs.shape, dtype=torch.float32, device="cuda"
        )
        coins = torch.rand_like(candidates, dtype=torch.float32, device="cuda")
        tree_speculative_sampling_target_only(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_length,  # mutable
            candidates=candidates.to(torch.int32),
            retrive_index=self.retrive_index.to(torch.int32) if process_cpu_requets else self.retrive_index_gpu.to(torch.int32),
            retrive_next_token=self.retrive_next_token.to(torch.int32) if process_cpu_requets else self.retrive_next_token_gpu.to(torch.int32),
            retrive_next_sibling=self.retrive_next_sibling.to(torch.int32) if process_cpu_requets else self.retrive_next_sibling_gpu.to(torch.int32),
            uniform_samples=coins,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=global_server_args_dict[
                "speculative_accept_threshold_single"
            ],
            threshold_acc=global_server_args_dict[
                "speculative_accept_threshold_acc"
            ],
            deterministic=True,
        )

        accept_index, accept_length = self.drop_accepted_tokens(accept_index, accept_length, ratio=experiment_ablation_drop_accepted_tokens_ratio)

        new_accept_index = []
        unfinished_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs[len(batch.gpu_seq_lens):] if process_cpu_requets else batch.reqs[:len(batch.gpu_seq_lens)], accept_index_cpu)):
            new_accept_index_ = []
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                # if not found_finished:
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    new_accept_index_.append(idx)
            if not req.finished():
                new_accept_index.extend(new_accept_index_)
                unfinished_index.append(i)
            req.spec_verify_ct += 1

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        accept_index = accept_index[accept_index != -1]  # 只保留被接受的index
        verified_id = predict[accept_index] # 得到被成功验证的token id
        reserve_mask = torch.full_like(self.draft_token[len(batch.gpu_seq_lens)*self.draft_token_num_gpu:] if process_cpu_requets else self.draft_token[:len(batch.gpu_seq_lens)*self.draft_token_num_gpu], False, dtype=torch.bool, device="cpu")
        reserve_mask[accept_index] = True
        logger.debug(f"accept_index: {accept_index}, verified_id: {verified_id}")
        return has_finished, accept_index, verified_id, reserve_mask, accept_length, unfinished_index, new_accept_index, predict

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
        token_to_kv_pool_allocator: TokenToKVPoolAllocatorCPU,
        experiment_ablation_drop_accepted_tokens_ratio: float = 0
    ) -> Tuple[EagleVerifyOutput, List[int]]:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepeted token logits.
        """

        has_finished_gpu, accept_index_gpu, verified_id_gpu, reserve_mask_gpu, accept_length_gpu, unfinished_index_gpu, new_accept_index_gpu, predict_gpu = self.verify_func(batch, logits_output, token_to_kv_pool_allocator, process_cpu_requets=False, experiment_ablation_drop_accepted_tokens_ratio=experiment_ablation_drop_accepted_tokens_ratio)
        has_finished_cpu, accept_index_cpu, verified_id_cpu, reserve_mask_cpu, accept_length_cpu, unfinished_index_cpu, new_accept_index_cpu, predict_cpu = self.verify_func(batch, logits_output, token_to_kv_pool_allocator, process_cpu_requets=True, experiment_ablation_drop_accepted_tokens_ratio=experiment_ablation_drop_accepted_tokens_ratio)
        if has_finished_gpu is not None:
            gpu_req_ratio = batch.policy.gpu_attention_ratio
            has_finished = has_finished_gpu or has_finished_cpu
            accept_length = torch.cat([accept_length_gpu, accept_length_cpu])
            reserve_mask = torch.cat([reserve_mask_gpu, reserve_mask_cpu])
            accept_index = torch.cat([accept_index_gpu, accept_index_cpu + reserve_mask_gpu.shape[0]])
            verified_id = torch.cat([verified_id_gpu, verified_id_cpu])
            unfinished_index = unfinished_index_gpu + [i + len(batch.gpu_seq_lens) for i in unfinished_index_cpu]
            new_accept_index = new_accept_index_gpu + [i + reserve_mask_gpu.shape[0] for i in new_accept_index_cpu]
        else:
            gpu_req_ratio = 0
            has_finished = has_finished_cpu
            accept_length = accept_length_cpu
            reserve_mask = reserve_mask_cpu
            accept_index = accept_index_cpu
            verified_id = verified_id_cpu
            unfinished_index = unfinished_index_cpu
            new_accept_index = new_accept_index_cpu

        # print(f"accept_length: {accept_length + 1}")

        # token_to_kv_pool_allocator.free(batch.out_cache_loc[reserve_mask]) # 将未被成功验证的token slot释放
        rids = [r.rid for r in batch.reqs]
        new_cpu_out_cache_loc = token_to_kv_pool_allocator.compact_after_verify_optimized(rids, self.draft_token_num, batch.cpu_out_cache_loc[reserve_mask], self.draft_token_num_gpu, len(batch.gpu_seq_lens))
        # 将成功验证的token移到序列后面，并释放掉未成功验证的token slot
        finished_gpudraft_req_rids: List[Tuple[int, int]] = [] # rid and seqlen
        # Construct EagleVerifyOutput
        if not has_finished:
            batch.out_cache_loc = new_cpu_out_cache_loc
            batch.seq_lens = batch.seq_lens.to('cuda')
            batch.seq_lens.add_(accept_length + 1)
            accept_length_cpu = accept_length.tolist()

            draft_input = EagleDraftInput()
            draft_input.hidden_states = batch.spec_info.hidden_states[accept_index]
            draft_input.verified_id = verified_id
            draft_input.accept_length = accept_length
            draft_input.accept_length_cpu = accept_length_cpu
            draft_input.seq_lens_for_draft_extend = batch.seq_lens # 这里的seq len就已经算上accepted token了
            # draft_input.req_pool_indices_for_draft_extend = batch.req_pool_indices

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_cpu,
                accepeted_indices=accept_index,
            ), finished_gpudraft_req_rids
        else:
            # assign_req_to_token_pool[(bs,)](
            #     batch.req_pool_indices,
            #     batch.req_to_token_pool.req_to_token,
            #     batch.seq_lens,
            #     batch.seq_lens + accept_length + 1,
            #     batch.out_cache_loc[accept_index],
            #     batch.req_to_token_pool.req_to_token.shape[1],
            #     next_power_of_2(bs),
            # )
            # TODO: 还没写处理结束任务的逻辑
            # assert False, "还没写处理结束任务的逻辑"
            batch.seq_lens = batch.seq_lens.to('cuda')
            batch.seq_lens.add_(accept_length + 1)
            accept_length_cpu = accept_length.tolist()

            draft_input = EagleDraftInput()
            if len(new_accept_index) > 0:
                new_accept_index_device = torch.tensor(new_accept_index, device="cuda")
                unfinished_index_device = torch.tensor(unfinished_index, device="cuda")
                draft_input.hidden_states = batch.spec_info.hidden_states[
                    new_accept_index_device
                ]
                if has_finished_gpu is not None:
                    gpu_vefified_id = predict_gpu[new_accept_index_gpu]
                    cpu_vefified_id = predict_cpu[new_accept_index_cpu]
                    draft_input.verified_id = torch.cat([gpu_vefified_id, cpu_vefified_id])
                else:
                    draft_input.verified_id = predict_cpu[new_accept_index_cpu]
                draft_input.accept_length_cpu = [
                    accept_length_cpu[i] for i in unfinished_index
                ]
                draft_input.accept_length = accept_length[unfinished_index_device]
                if has_finished:
                    draft_input.seq_lens_for_draft_extend = batch.seq_lens[
                        unfinished_index_device
                    ]
                else:
                    draft_input.seq_lens_for_draft_extend = batch.seq_lens
            batch.out_cache_loc = batch.cpu_out_cache_loc[new_accept_index]

            # 算出已完成的req的rid，然后返回，用于更新shadow kv cache
            for i in range(len(batch.seq_lens)):
                if i in unfinished_index: continue
                if i < batch.policy.draft_gpu_req_num:
                    finished_gpudraft_req_rids.append((batch.reqs[i].rid, batch.seq_lens[i].item()-accept_length_cpu[i]-1))

            batch.seq_lens = batch.seq_lens[unfinished_index]
            # TODO: 重新分配CPU/GPU request
            gpu_req_num = int(len(unfinished_index) * gpu_req_ratio)
            batch.policy.gpu_attention_micro_batch_size = gpu_req_num
            batch.policy.gpu_attention_nano_batch_size = min(batch.policy.gpu_attention_nano_batch_size, gpu_req_num)
            cpu_req_num = len(unfinished_index) - gpu_req_num
            batch.gpu_reqs_rids = [batch.reqs[i].rid for i in unfinished_index[:gpu_req_num]]
            batch.cpu_reqs_rids = [batch.reqs[i].rid for i in unfinished_index[gpu_req_num:]]
            batch.is_req_on_gpu = [True] * gpu_req_num + [False] * cpu_req_num
            batch.gpu_seq_lens = batch.seq_lens[:gpu_req_num]
            batch.cpu_seq_lens = batch.seq_lens[gpu_req_num:]
            batch.gpu_seq_lens_sum = batch.gpu_seq_lens.sum()
            batch.cpu_seq_lens_sum = batch.cpu_seq_lens.sum()

            return (
                EagleVerifyOutput(
                    draft_input=draft_input,
                    logits_output=logits_output,
                    verified_id=verified_id,
                    accept_length_per_req_cpu=accept_length_cpu,
                    accepeted_indices=accept_index,
                ),
                finished_gpudraft_req_rids,
            )

@triton.jit
def create_extend_spec_info(
    verified_id,
    seq_len,
    accept_len,
    accept_len_cum,
    positions,
    new_verified_id,
    accept_len_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = 0 if pid == 0 else tl.load(accept_len_cum + pid - 1)
    seq_length = tl.load(seq_len + pid)
    accept_length = tl.load(accept_len + pid)
    positions_ptr = positions + offset
    data = tl.arange(0, accept_len_upper)
    mask = data < accept_length
    tl.store(positions_ptr + data, seq_length - accept_length + data, mask)

    offset = tl.load(accept_len_cum + pid) - 1
    verified_id_data = tl.load(verified_id + offset)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(seq_lens + pid)

    if page_size == 1 or topk == 1:
        kv_end = tl.load(seq_lens + pid) + topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        num_new_page = (
            last_page_len + speculative_num_steps + page_size - 1
        ) // page_size
        kv_end = prefix_len // page_size * page_size + num_new_page * (page_size * topk)

    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    num_loop = tl.cdiv(topk * speculative_num_steps, BLOCK_SIZE)
    for i in range(num_loop):
        save_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE + kv_start
        load_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    num_seqs: tl.constexpr,
    topk: tl.constexpr,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    extend_data = tl.load(
        token_pool_ptr + seq_len + tl.arange(0, iter_upper) * topk + topk_id,
        mask=extend_offset < iters,
    )
    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(evict_mask + bid * num_draft_tokens + t_range, mask=io_mask)

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


# @torch.compile(dynamic=True)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device="cuda")
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        device = topk_index.device
        topk_p = topk_p.to(device=device)
        topk_index = topk_index.to(device=device)
        hidden_states = hidden_states.to(device=device)
        scores = scores.to(device=device)
        # logger.debug(f"scores shape: {scores.shape}, topk_p shape: {topk_p.shape}, topk_index shape: {topk_index.shape}, hidden_states shape: {hidden_states.shape}")
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk) # 每个上轮的topk都会衍生成topk个子节点，需要跟父节点的概率相乘作为自己cs_p
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk( # 从topk个父节点衍生出的topk*topk个子节点中选择概率最大的topk个
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2) # (b, topk*topk)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten() # input_ids 此时就按照cs_p从大到小排好顺序了, (b*topk)

        selected_input_index = topk_cs_index.flatten() // topk + torch.arange( 
            0, hidden_states.shape[0], step=topk, device=device
        ).repeat_interleave(topk) # (bs * topk)

        hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def _generate_simulated_accept_index(
    accept_index,
    predict,
    accept_length,
    simulate_acc_len,
    bs,
    spec_steps,
):
    simulate_acc_len_float = float(simulate_acc_len)
    simulated_values = torch.normal(
        mean=simulate_acc_len_float,
        std=1.0,
        size=(1,),
        device="cpu",
    )
    # clamp simulated values to be between 1 and self.spec_steps
    simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps)
    simulate_acc_len = int(simulated_values.round().item())

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    accept_length.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(seq_lens + pid)

    if page_size == 1 or topk == 1:
        kv_end = tl.load(seq_lens + pid) + topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        num_new_page = (
            last_page_len + speculative_num_steps + page_size - 1
        ) // page_size
        kv_end = prefix_len // page_size * page_size + num_new_page * (page_size * topk)

    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    num_loop = tl.cdiv(topk * speculative_num_steps, BLOCK_SIZE)
    for i in range(num_loop):
        save_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE + kv_start
        load_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)

def split_schedule_batch_for_draft(original_batch: ScheduleBatch, gpu_req_num: int) -> Tuple["ScheduleBatch", "ScheduleBatch"]:
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
                reqs=original_batch.reqs[:gpu_req_num],
                is_req_on_gpu=[True] * gpu_req_num,
                gpu_req_to_token_pool=original_batch.gpu_req_to_token_pool,
                gpu_token_to_kv_pool_allocator=original_batch.gpu_token_to_kv_pool_allocator,
                cpu_req_to_token_pool=original_batch.cpu_req_to_token_pool,
                cpu_token_to_kv_pool_allocator=original_batch.cpu_token_to_kv_pool_allocator,
                model_config=original_batch.model_config,
                spec_algorithm=original_batch.spec_algorithm,
                policy=original_batch.policy,
                split_for_draft=True,
            )

            gpu_batch.forward_mode = original_batch.forward_mode
            
            if original_batch.seq_lens is not None:
                gpu_batch.seq_lens = original_batch.seq_lens[:gpu_req_num]
                gpu_batch.seq_lens_sum = gpu_batch.seq_lens.sum().item()
            
            if original_batch.cpu_seq_lens is not None:
                gpu_batch.cpu_seq_lens = torch.tensor([], dtype=original_batch.cpu_seq_lens.dtype, device=original_batch.cpu_seq_lens.device)
                gpu_batch.cpu_seq_lens_sum = 0
                
            if original_batch.gpu_seq_lens is not None:
                gpu_batch.gpu_seq_lens = original_batch.seq_lens[:gpu_req_num] if original_batch.seq_lens is not None else original_batch.gpu_seq_lens[:gpu_req_num]
                gpu_batch.gpu_seq_lens_sum = gpu_batch.gpu_seq_lens.sum().item() if len(gpu_batch.gpu_seq_lens) > 0 else 0
            
            if original_batch.sampling_info is not None:
                gpu_batch.sampling_info = original_batch.sampling_info
            
            # 分裂spec_info
            if original_batch.spec_info is not None:
                gpu_batch.spec_info = EagleDraftInput()
                if original_batch.spec_info.accept_length is not None:
                    gpu_batch.spec_info.accept_length = original_batch.spec_info.accept_length[:gpu_req_num]
                if original_batch.spec_info.verified_id is not None:
                    if original_batch.spec_info.accept_length is not None:
                        gpu_process_verified = (gpu_batch.spec_info.accept_length + 1).sum()
                    else:
                        gpu_process_verified = gpu_req_num
                    gpu_batch.spec_info.verified_id = original_batch.spec_info.verified_id[:gpu_process_verified]
                if original_batch.spec_info.accept_length_cpu is not None:
                    gpu_batch.spec_info.accept_length_cpu = original_batch.spec_info.accept_length_cpu[:gpu_req_num]
                if original_batch.spec_info.seq_lens_for_draft_extend is not None:
                    gpu_batch.spec_info.seq_lens_for_draft_extend = original_batch.spec_info.seq_lens_for_draft_extend[:gpu_req_num]
                
                
                if original_batch.spec_info.first_step_after_prefill:
                    gpu_batch.spec_info.topk_p = original_batch.spec_info.topk_p[:gpu_req_num-original_batch.policy.gpu_attention_micro_batch_size]
                    gpu_batch.spec_info.topk_index = original_batch.spec_info.topk_index[:gpu_req_num-original_batch.policy.gpu_attention_micro_batch_size]
                    gpu_batch.spec_info.hidden_states = original_batch.spec_info.hidden_states[:gpu_req_num-original_batch.policy.gpu_attention_micro_batch_size]
                    if original_batch.spec_info.topk_p_gpu is not None:
                        gpu_batch.spec_info.topk_p_gpu = original_batch.spec_info.topk_p_gpu[:original_batch.policy.gpu_attention_micro_batch_size]
                        gpu_batch.spec_info.topk_index_gpu = original_batch.spec_info.topk_index_gpu[:original_batch.policy.gpu_attention_micro_batch_size]
                        gpu_batch.spec_info.hidden_states_gpu = original_batch.spec_info.hidden_states_gpu[:original_batch.policy.gpu_attention_micro_batch_size]
                    gpu_batch.spec_info.capture_hidden_mode = original_batch.spec_info.capture_hidden_mode
                else:
                    gpu_batch.spec_info.hidden_states = original_batch.spec_info.hidden_states[:gpu_process_verified]
                    gpu_batch.spec_info.capture_hidden_mode = original_batch.spec_info.capture_hidden_mode
            
            if hasattr(original_batch.spec_info, 'first_step_after_prefill'):
                gpu_batch.spec_info.first_step_after_prefill = original_batch.spec_info.first_step_after_prefill
            
            if hasattr(original_batch, "out_cache_loc"):
                gpu_batch.out_cache_loc = original_batch.out_cache_loc
        else:
            gpu_batch = None
        
        # 创建CPU batch
        cpu_req_num = len(original_batch.reqs) - gpu_req_num
        if cpu_req_num > 0:
            cpu_batch = ScheduleBatch.init_new(
                reqs=original_batch.reqs[gpu_req_num:],
                is_req_on_gpu=[False] * cpu_req_num,
                gpu_req_to_token_pool=original_batch.gpu_req_to_token_pool,
                gpu_token_to_kv_pool_allocator=original_batch.gpu_token_to_kv_pool_allocator,
                cpu_req_to_token_pool=original_batch.cpu_req_to_token_pool,
                cpu_token_to_kv_pool_allocator=original_batch.cpu_token_to_kv_pool_allocator,
                model_config=original_batch.model_config,
                spec_algorithm=original_batch.spec_algorithm,
                policy=original_batch.policy,
                split_for_draft=True,
            )
            cpu_batch.forward_mode = original_batch.forward_mode
            
            # 分裂相关tensor
            if original_batch.seq_lens is not None:
                cpu_batch.seq_lens = original_batch.seq_lens[gpu_req_num:]
                cpu_batch.seq_lens_sum = cpu_batch.seq_lens.sum().item()
            
            if original_batch.cpu_seq_lens is not None:
                cpu_batch.cpu_seq_lens = original_batch.seq_lens[gpu_req_num:] if original_batch.seq_lens is not None else original_batch.cpu_seq_lens[gpu_req_num:]
                cpu_batch.cpu_seq_lens_sum = cpu_batch.cpu_seq_lens.sum().item() if len(cpu_batch.cpu_seq_lens) > 0 else 0
                
            if original_batch.gpu_seq_lens is not None:
                cpu_batch.gpu_seq_lens = torch.tensor([], dtype=original_batch.gpu_seq_lens.dtype, device=original_batch.gpu_seq_lens.device)
                cpu_batch.gpu_seq_lens_sum = 0
           
            # 分裂sampling info
            if original_batch.sampling_info is not None:
                cpu_batch.sampling_info = original_batch.sampling_info
            
            # 分裂spec_info
            if original_batch.spec_info is not None:
                cpu_batch.spec_info = EagleDraftInput()
                if original_batch.spec_info.accept_length is not None:
                    cpu_batch.spec_info.accept_length = original_batch.spec_info.accept_length[gpu_req_num:]
                if original_batch.spec_info.verified_id is not None:
                    if gpu_batch is not None and gpu_batch.spec_info.accept_length is not None:
                        gpu_process_verified = (gpu_batch.spec_info.accept_length + 1).sum()
                    else:
                        gpu_process_verified = gpu_req_num
                    cpu_batch.spec_info.verified_id = original_batch.spec_info.verified_id[gpu_process_verified:]
                if original_batch.spec_info.accept_length_cpu is not None:
                    cpu_batch.spec_info.accept_length_cpu = original_batch.spec_info.accept_length_cpu[gpu_req_num:]
                if original_batch.spec_info.seq_lens_for_draft_extend is not None:
                    cpu_batch.spec_info.seq_lens_for_draft_extend = original_batch.spec_info.seq_lens_for_draft_extend[gpu_req_num:]

            if original_batch.spec_info.first_step_after_prefill:
                cpu_batch.spec_info.topk_p = original_batch.spec_info.topk_p[gpu_req_num-original_batch.policy.gpu_attention_micro_batch_size:]
                cpu_batch.spec_info.topk_index = original_batch.spec_info.topk_index[gpu_req_num-original_batch.policy.gpu_attention_micro_batch_size:]
                cpu_batch.spec_info.hidden_states = original_batch.spec_info.hidden_states[gpu_req_num-original_batch.policy.gpu_attention_micro_batch_size:]
                cpu_batch.spec_info.capture_hidden_mode = original_batch.spec_info.capture_hidden_mode
            else:
                cpu_batch.spec_info.hidden_states = original_batch.spec_info.hidden_states[gpu_process_verified:]
                cpu_batch.spec_info.capture_hidden_mode = original_batch.spec_info.capture_hidden_mode

            if hasattr(original_batch.spec_info, 'first_step_after_prefill'):
                cpu_batch.spec_info.first_step_after_prefill = original_batch.spec_info.first_step_after_prefill
            
            if hasattr(original_batch, "out_cache_loc"):
                cpu_batch.out_cache_loc = original_batch.out_cache_loc
                cpu_batch.out_cache_loc_begin_id_draft_extend_after_verify = (gpu_batch.spec_info.accept_length + 1).sum() if gpu_batch is not None else 0
        else:
            cpu_batch = None
            
        return gpu_batch, cpu_batch

