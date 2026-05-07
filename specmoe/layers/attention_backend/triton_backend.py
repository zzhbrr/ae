from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, List

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.utils import get_bool_env_var, get_device_core_count

from specmoe.layers.attention_backend.base_attention_backend import AttentionBackend
from specmoe.backend.forward_batch_info import ForwardBatch, ForwardMode, DecodePart
from specmoe.layers.attention import RadixAttention, AttentionType
from specmoe.backend.forward_batch_info import ForwardBatch, ForwardMode, DecodePart
from specmoe.backend.model_runner import ModelRunner
from specmoe.backend.memory import ReqToTokenPool

if TYPE_CHECKING:
    from specmoe.layers.attention import RadixAttention
    # from sglang.srt.model_executor.model_runner import ModelRunner
    # from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor


class TritonAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        for_draft_generate: bool = False, 
        draft_gpu_req_to_token_pool: Optional[ReqToTokenPool] = None, # Draft model有独立于model_runner.gpu_req_to_token_pool的req_to_token_pool，因此要独立处理；此参数仅用于for_draft_generate = True的情况
    ):
        # Lazy import to avoid the initialization of cuda context
        # from sglang.srt.layers.attention.triton_ops.decode_attention import (
        #     decode_attention_fwd,
        # )
        # from sglang.srt.layers.attention.triton_ops.extend_attention import (
        #     extend_attention_fwd,
        # )
        from specmoe.layers.attention_backend.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from specmoe.layers.attention_backend.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_fwd

        self.skip_prefill = skip_prefill

        self.max_context_len = model_runner.server_args.max_seq_length

        if for_draft_generate:
            max_bs = draft_gpu_req_to_token_pool.size
        else:
            max_bs = model_runner.gpu_req_to_token_pool.size

        # if kv_indptr_buf is None:
        #     self.kv_indptr = torch.zeros(
        #         (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        #     )
        # else:
        #     self.kv_indptr = kv_indptr_buf

        if for_draft_generate:
            self.req_to_token = draft_gpu_req_to_token_pool.req_to_token
        else:
            self.req_to_token = model_runner.gpu_req_to_token_pool.req_to_token

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        if model_runner.server_args.decode_gpu_attention_ratio > 0:
            self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens_gpu
        else:
            self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.num_key_value_heads

        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.gpu_token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.forward_metadata: ForwardMetadata = None
        self.forward_metadata_list: List[ForwardMetadata] = []

        # self.max_context_len = model_runner.model_config.context_len

        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if self.static_kv_splits or self.device_core_count <= 0:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch, micro_batch_id: int = -1, begin: int = 0, end: int = -1, draft_gpu_execute_cpu_attn_request: bool = False):
        """Init auxiliary variables for triton attention backend."""

        if end == -1:
            end = len(forward_batch.gpu_seq_lens)
        bs = end - begin
            
        kv_indptr = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=self.device
        )
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle(): 
            if spec_info is None: # for normal decode
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.gpu_seq_lens[begin:end], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.gpu_seq_lens[begin:end].sum().item(), dtype=torch.int32, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.gpu_req_pool_indices[begin:end],
                    forward_batch.gpu_seq_lens[begin:end],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else: # for draft generate
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)

            if draft_gpu_execute_cpu_attn_request:
                self.get_num_kv_splits(num_kv_splits, forward_batch.cpu_seq_lens)
            else:
                self.get_num_kv_splits(num_kv_splits, forward_batch.gpu_seq_lens[begin:end])

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify(): # for target verify
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.gpu_seq_lens[begin:end], dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int32, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.gpu_req_pool_indices[begin:end],
                forward_batch.gpu_seq_lens[begin:end],
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            # 先算出来整体的mask_indptr，为的是取出begin--end部分的mask
            oringinal_seq_mask_len = self.num_draft_tokens * ( 
                forward_batch.gpu_seq_lens + self.num_draft_tokens
            )
            mask_indptr = torch.zeros(
                (len(forward_batch.gpu_seq_lens) + 1,), dtype=torch.int64, device=self.device
            )
            mask_indptr[1 : len(forward_batch.gpu_seq_lens) + 1] = torch.cumsum(oringinal_seq_mask_len, dim=0)
            mask_indptr = mask_indptr[: len(forward_batch.gpu_seq_lens) + 1]
            # 取出mask
            custom_mask = spec_info.custom_mask[mask_indptr[begin]:mask_indptr[end]]
            # 计算最终的mask_indptr
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.gpu_seq_lens[begin:end] + self.num_draft_tokens
            )
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_batch.forward_mode.is_draft_extend(): # for draft extend
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.gpu_req_pool_indices,
                    forward_batch.gpu_seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.accept_length_cpu)`.
            # It might have been forgotten to update somewhere.
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.extend_prefix_lens.sum().item(),
                dtype=torch.int32,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.gpu_req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.qo_indptr.clone()
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
            num_kv_splits = None

        forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
        )

        if micro_batch_id != -1:
            if len(self.forward_metadata_list) <= micro_batch_id:
                self.forward_metadata_list.append(forward_metadata)
            else:
                self.forward_metadata_list[micro_batch_id] = forward_metadata
        else:
            self.forward_metadata = forward_metadata

    # def init_cuda_graph_state(
    #     self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None
    # ):
    #     self.cuda_graph_attn_logits = torch.zeros(
    #         (max_bs, self.num_head, self.max_kv_splits, self.v_head_dim),
    #         dtype=torch.float32,
    #         device=self.device,
    #     )
    #     self.cuda_graph_attn_lse = torch.zeros(
    #         (max_bs, self.num_head, self.max_kv_splits),
    #         dtype=torch.float32,
    #         device=self.device,
    #     )
    #     self.cuda_graph_num_kv_splits = torch.full(
    #         (max_bs,), self.max_kv_splits, dtype=torch.int32, device=self.device
    #     )
    #     if kv_indices_buf is None:
    #         self.cuda_graph_kv_indices = torch.zeros(
    #             (max_bs * self.max_context_len),
    #             dtype=torch.int32,
    #             device=self.device,
    #         )
    #     else:
    #         self.cuda_graph_kv_indices = kv_indices_buf

    #     if not self.skip_prefill:
    #         self.cuda_graph_custom_mask = torch.zeros(
    #             (max_bs * self.max_context_len),
    #             dtype=torch.uint8,
    #             device=self.device,
    #         )

    # def init_forward_metadata_capture_cuda_graph(
    #     self,
    #     bs: int,
    #     num_tokens: int,
    #     req_pool_indices: torch.Tensor,
    #     seq_lens: torch.Tensor,
    #     encoder_lens: Optional[torch.Tensor],
    #     forward_mode: ForwardMode,
    #     spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    # ):
    #     assert encoder_lens is None, "Not supported"

    #     if forward_mode.is_decode_or_idle():
    #         if spec_info is None:
    #             kv_indptr = self.kv_indptr
    #             kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
    #             kv_indptr = kv_indptr[: bs + 1]
    #             kv_indices = self.cuda_graph_kv_indices
    #             create_flashinfer_kv_indices_triton[(bs,)](
    #                 self.req_to_token,
    #                 req_pool_indices,
    #                 seq_lens,
    #                 kv_indptr,
    #                 None,
    #                 kv_indices,
    #                 self.req_to_token.stride(0),
    #             )
    #         else:
    #             kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

    #         attn_logits = self.cuda_graph_attn_logits
    #         attn_lse = self.cuda_graph_attn_lse
    #         max_extend_len = None
    #         num_kv_splits = self.cuda_graph_num_kv_splits
    #         qo_indptr = None
    #         custom_mask = None
    #         mask_indptr = None
    #     elif forward_mode.is_target_verify():
    #         qo_indptr = self.qo_indptr[: bs + 1]
    #         qo_indptr[: bs + 1] = torch.arange(
    #             0,
    #             (1 + bs) * self.num_draft_tokens,
    #             step=self.num_draft_tokens,
    #             dtype=torch.int32,
    #             device=self.device,
    #         )
    #         kv_indptr = self.kv_indptr[: bs + 1]
    #         kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
    #         kv_indices = self.cuda_graph_kv_indices
    #         create_flashinfer_kv_indices_triton[(bs,)](
    #             self.req_to_token,
    #             req_pool_indices,
    #             seq_lens,
    #             kv_indptr,
    #             None,
    #             kv_indices,
    #             self.req_to_token.stride(0),
    #         )

    #         custom_mask = self.cuda_graph_custom_mask
    #         seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
    #         mask_indptr = self.mask_indptr[: bs + 1]
    #         mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
    #         max_extend_len = self.num_draft_tokens
    #         num_kv_splits = None
    #         attn_logits = None
    #         attn_lse = None
    #     else:
    #         raise ValueError(
    #             f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
    #         )

    #     self.forward_metadata = ForwardMetadata(
    #         attn_logits,
    #         attn_lse,
    #         max_extend_len,
    #         num_kv_splits,
    #         kv_indptr,
    #         kv_indices,
    #         qo_indptr,
    #         custom_mask,
    #         mask_indptr,
    #     )

    # def init_forward_metadata_replay_cuda_graph(
    #     self,
    #     bs: int,
    #     req_pool_indices: torch.Tensor,
    #     seq_lens: torch.Tensor,
    #     seq_lens_sum: int,
    #     encoder_lens: Optional[torch.Tensor],
    #     forward_mode: ForwardMode,
    #     spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    #     seq_lens_cpu: Optional[torch.Tensor],
    # ):
    #     # NOTE: encoder_lens expected to be zeros or None
    #     if forward_mode.is_decode_or_idle():
    #         # Update kv_indptr, kv_indices
    #         kv_indptr = self.kv_indptr
    #         kv_indices = self.cuda_graph_kv_indices
    #         num_kv_splits = self.cuda_graph_num_kv_splits
    #         if spec_info is None:
    #             kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
    #             kv_indptr = kv_indptr[: bs + 1]
    #             create_flashinfer_kv_indices_triton[(bs,)](
    #                 self.req_to_token,
    #                 req_pool_indices[:bs],
    #                 seq_lens[:bs],
    #                 kv_indptr,
    #                 None,
    #                 kv_indices,
    #                 self.req_to_token.stride(0),
    #             )
    #             num_token = bs
    #         else:
    #             kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
    #             kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
    #             num_token = spec_info.kv_indptr.shape[0] - 1
    #         self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])
    #     elif forward_mode.is_target_verify():
    #         # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
    #         bs = len(req_pool_indices)
    #         qo_indptr = self.qo_indptr[: bs + 1]
    #         qo_indptr[: bs + 1] = torch.arange(
    #             0,
    #             (1 + bs) * self.num_draft_tokens,
    #             step=self.num_draft_tokens,
    #             dtype=torch.int32,
    #             device=self.device,
    #         )
    #         kv_indptr = self.kv_indptr[: bs + 1]
    #         kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
    #         kv_indices = self.cuda_graph_kv_indices
    #         create_flashinfer_kv_indices_triton[(bs,)](
    #             self.req_to_token,
    #             req_pool_indices,
    #             seq_lens,
    #             kv_indptr,
    #             None,
    #             kv_indices,
    #             self.req_to_token.stride(0),
    #         )
    #         custom_mask = self.cuda_graph_custom_mask
    #         custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
    #         seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
    #         mask_indptr = self.mask_indptr[: bs + 1]
    #         mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
    #     else:
    #         raise ValueError(
    #             f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
    #         )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        micro_batch_id: int = -1,
        nano_batch_id: int = -1,
        gpu_attn_init_slot: int = -1,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        
        use_gpu_two_buffer = False
        if forward_batch.forward_mode == ForwardMode.TARGET_VERIFY and forward_batch.gpu_attn_input_ids_size > 1:
            use_gpu_two_buffer = True
        
        two_buffer_offset = -1
        if save_kv_cache:
            gpu_out_cache_loc = forward_batch.gpu_out_cache_loc
            if nano_batch_id != -1:
                # two_buffer_offset = nano_batch_id % 2
                two_buffer_offset = 0 # XXX: 目前只用到一层kv cache，没有GPU attn和KV cache fetch的overlap
                # print(f"nano_batch_id: {nano_batch_id}")
                # print(f"len(gpu_out_cache_loc): {len(gpu_out_cache_loc)}")
                gpu_out_cache_loc = gpu_out_cache_loc[nano_batch_id]
            forward_batch.gpu_token_to_kv_pool.set_kv_buffer(
                layer, gpu_out_cache_loc, k, v, gpu_two_buffer=use_gpu_two_buffer, two_buffer_offset=two_buffer_offset
            )
        
        if micro_batch_id != -1:
            self.forward_metadata = self.forward_metadata_list[micro_batch_id]
            if gpu_attn_init_slot != -1:
                self.forward_metadata = self.forward_metadata_list[gpu_attn_init_slot]
        
        causal = True
        if layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.gpu_token_to_kv_pool.get_key_buffer(layer.layer_id, use_gpu_two_buffer, two_buffer_offset),
            forward_batch.gpu_token_to_kv_pool.get_value_buffer(layer.layer_id, use_gpu_two_buffer, two_buffer_offset),
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        micro_batch_id: int = -1,
        nano_batch_id: int = -1,
        gpu_attn_init_slot: int = -1,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        
        use_gpu_two_buffer = False
        if forward_batch.forward_mode == ForwardMode.TARGET_VERIFY or forward_batch.forward_mode == ForwardMode.DECODE:
            if len(forward_batch.gpu_seq_lens) > 0:
                use_gpu_two_buffer = True

        # XXX: 其实在DECODE和TargetVerify的情况下没必要保存KV cache
        two_buffer_offset = -1
        if save_kv_cache:
            gpu_out_cache_loc = forward_batch.gpu_out_cache_loc
            if nano_batch_id != -1:
                # two_buffer_offset = nano_batch_id % 2
                two_buffer_offset = 0 # XXX: 目前只用到一层kv cache，没有GPU attn和KV cache fetch的overlap
                gpu_out_cache_loc = gpu_out_cache_loc[nano_batch_id]
            forward_batch.gpu_token_to_kv_pool.set_kv_buffer(
                layer, gpu_out_cache_loc, k, v, gpu_two_buffer=use_gpu_two_buffer, two_buffer_offset=two_buffer_offset
            )
        
        if micro_batch_id != -1:
            self.forward_metadata = self.forward_metadata_list[micro_batch_id]
            if gpu_attn_init_slot != -1:
                self.forward_metadata = self.forward_metadata_list[gpu_attn_init_slot]
        
        # print(f"layer {layer.layer_id}, microbatch {micro_batch_id}, nano_batch {nano_batch_id}, kv_indptr: {self.forward_metadata.kv_indptr}")

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.gpu_token_to_kv_pool.get_key_buffer(layer.layer_id, use_gpu_two_buffer, two_buffer_offset),
            forward_batch.gpu_token_to_kv_pool.get_value_buffer(layer.layer_id, use_gpu_two_buffer, two_buffer_offset),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            layer.logit_cap,
        )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        for_draft_generate: bool = False, 
        draft_gpu_req_to_token_pool: Optional[ReqToTokenPool] = None, # Draft model有独立于model_runner.gpu_req_to_token_pool的req_to_token_pool，因此要独立处理；此参数仅用于for_draft_generate = True的情况
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.gpu_req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    for_draft_generate=for_draft_generate,
                    draft_gpu_req_to_token_pool=draft_gpu_req_to_token_pool,
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        if for_draft_generate:
            self.pool_len = draft_gpu_req_to_token_pool.req_to_token.shape[1]
        else:
            self.pool_len = model_runner.gpu_req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int, process_cpu_attn_request: bool = False
    ):
        if process_cpu_attn_request:
            num_seqs = len(forward_batch.cpu_seq_lens)
            seq_lens_sum = forward_batch.cpu_seq_lens_sum
            cpu_req_start_idx = len(forward_batch.gpu_seq_lens)  # CPU requests的起始索引
            effective_req_indices = forward_batch.gpu_req_pool_indices[cpu_req_start_idx:cpu_req_start_idx + num_seqs]
        else:
            num_seqs = len(forward_batch.gpu_seq_lens)
            seq_lens_sum = forward_batch.gpu_seq_lens_sum
            effective_req_indices = forward_batch.gpu_req_pool_indices
        bs = self.topk * num_seqs

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            effective_req_indices,
            forward_batch.gpu_req_to_token_pool.req_to_token,
            forward_batch.gpu_seq_lens if not process_cpu_attn_request else forward_batch.cpu_seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            num_seqs,
            self.topk,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
        )

        for i in range(self.speculative_num_steps):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch, process_cpu_attn_request: bool = False):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch: ForwardBatch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch, draft_gpu_execute_cpu_attn_request=process_cpu_attn_request)

        self.common_template(forward_batch, kv_indices, call_fn, process_cpu_attn_request)