# CPU Attention Backend
import torch
import logging
from dataclasses import dataclass
import math
from typing import List, Callable, Optional


from specmoe.layers.attention_backend.base_attention_backend import AttentionBackend
from specmoe.backend.forward_batch_info import ForwardBatch, ForwardMode, DecodePart
from specmoe.backend.model_runner import ModelRunner
from specmoe.layers.attention import RadixAttention
from specmoe._cpu_kernel import (
    token_attention_cpu,
    token_attention_cpu_verified3,
    token_attention_cpu_draft_extend_with_mask_optimized_2,
    token_attention_cpu_verified3_less_mask,
    token_attention_cpu_draft_extend_with_less_mask_optimized,
    token_attention_cpu_draft_extend_optimized_2
)

logger = logging.getLogger(__name__)

def CpuAttention(
    forward_mode: ForwardMode,
    cpu_output: torch.Tensor, 
    q: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    seq_lens: torch.Tensor, 
    start_loc: torch.Tensor,
    # blockmask: BlockMask,
    is_gqa: bool = False,
    layer: int = 0, 
    custom_mask: Optional[torch.Tensor] = None,
    mask_indptr: Optional[torch.Tensor] = None, 
    qo_indptr: Optional[torch.Tensor] = None, 
    mask_len: Optional[int] = None,
    use_less_mask_cpuattention: bool = True
):
    """使用PyTorch FlexAttention实现CPU上的注意力计算，支持变长序列和GQA
    
    参数:
        cpu_output: 输出tensor，形状为[sum(q_len), num_heads, head_dim]
        q: 查询tensor，形状为[sum(q_len), num_heads, head_dim]
        k: 键值tensor，形状为[slot_number, num_heads/gqa_ratio, head_dim]
        v: 值tensor，形状为[slot_number, num_heads/gqa_ratio, head_dim]
        seq_lens: 每个序列的实际长度，形状为[batch_size]
        start_loc: 每个序列在packed表示中的起始位置，形状为[batch_size]
        blockmask: 预先计算的BlockMask，用于加速稀疏计算
        document_ids: 每个token对应的序列ID
        is_gqa: 是否使用Grouped Query Attention
        kv_indices: KV cache中有效位置的索引，形状为[sum(kv_len)]，可用于避免冗余计算
    """
    batch_size = seq_lens.size(0)
    head_dim = k_cache.size(-1)
    q_heads = q.size(1)
    kv_heads = k_cache.size(1)

    if cpu_output is None:
        cpu_output = torch.empty_like(q, device=q.device, pin_memory=True)

    if forward_mode == ForwardMode.TARGET_VERIFY:
        # 需要将 q 从 [sum(q_len), num_heads, head_dim] 重塑为 [batch_size, q_len_per_batch, num_heads, head_dim]
        # 这里假设每个batch中的q_len都是1（decode阶段的常见情况）
        q_reshaped = q.view(batch_size, -1, q_heads, head_dim)
        if use_less_mask_cpuattention:
            token_attention_cpu_verified3_less_mask(cpu_output, q_reshaped, k_cache, v_cache, seq_lens, start_loc, custom_mask, mask_indptr, mask_len, head_dim**-0.5)
        else:
            token_attention_cpu_verified3(cpu_output, q_reshaped, k_cache, v_cache, seq_lens, start_loc, custom_mask, mask_indptr, head_dim**-0.5)
    elif forward_mode == ForwardMode.DRAFT_EXTEND:
        qlensum = q.size(0)
        q = q.reshape(qlensum, -1, head_dim)
        cpu_output = cpu_output.reshape(qlensum, -1, head_dim)
        token_attention_cpu_draft_extend_optimized_2(cpu_output, q, k_cache, v_cache, seq_lens, start_loc, qo_indptr, head_dim**-0.5)
        cpu_output = cpu_output.reshape(qlensum, -1)
    elif forward_mode == ForwardMode.DECODE and qo_indptr is not None: # for draft generate
        qlensum = q.size(0)
        q = q.reshape(qlensum, -1, head_dim)
        cpu_output = cpu_output.reshape(qlensum, -1, head_dim)
        if use_less_mask_cpuattention:
            token_attention_cpu_draft_extend_with_less_mask_optimized(cpu_output, q, k_cache, v_cache, seq_lens, start_loc, qo_indptr, custom_mask, mask_indptr, mask_len, head_dim**-0.5)
        else:
            token_attention_cpu_draft_extend_with_mask_optimized_2(cpu_output, q, k_cache, v_cache, seq_lens, start_loc, qo_indptr, custom_mask, mask_indptr, head_dim**-0.5)
        cpu_output = cpu_output.reshape(qlensum, -1)
    else:
        token_attention_cpu(cpu_output, q.unsqueeze(1), k_cache, v_cache, seq_lens, start_loc, head_dim**-0.5)
    # logger.debug(f"cpu_output: {cpu_output.shape}")
    # cpu_output.copy_(output)

    return cpu_output

@dataclass
class ForwardMetadata:
    """存储前向传播需要的元数据"""
    # blockmask: BlockMask  # 用于加速注意力计算的BlockMask
    start_loc: torch.Tensor # List[int]
    seq_lens: torch.Tensor # List[int]
    custom_mask: torch.Tensor # len(custom_mask) = sum(seq_lens) + bs * draft_token_num
    mask_indptr: torch.Tensor 
    qo_indptr: torch.Tensor
    mask_len: Optional[int] = None # use for less_mask_cpu_attention

class CPUAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        self.tp_q_head_num = model_runner.model_config.num_attention_heads
        self.tp_k_head_num = model_runner.model_config.num_key_value_heads
        self.tp_v_head_num = model_runner.model_config.num_key_value_heads
        self.head_dim = model_runner.model_config.head_dim
        self.is_gqa = (self.tp_q_head_num != self.tp_k_head_num)  # 检测是否使用GQA
        self.forward_metadata = None
        self.forward_metadata_list: List[ForwardMetadata] = []
        self.use_special_less_mask_cpuattention = model_runner.server_args.use_special_less_mask_cpuattention

        self.model_runner = model_runner
    
        # self.flex_attention = torch.compile(flex_attention)
    
    def init_forward_metadata(self, forward_batch: ForwardBatch, process_cpu_requets: bool = True, micro_batch_id: int = -1):
        """初始化前向传播所需的元数据，特别是创建BlockMask和KV索引
        
        参数:
            forward_batch: 包含批次信息的对象
        """
        if process_cpu_requets:
            batch_size = len(forward_batch.cpu_seq_lens)
        else:
            batch_size = len(forward_batch.gpu_seq_lens)

        if process_cpu_requets:
            start_loc = self.model_runner.cpu_token_to_kv_pool_allocator.get_start_loc(forward_batch.cpu_batch_rids)
            seq_lens = forward_batch.cpu_seq_lens.clone().detach()
        else:
            start_loc = self.model_runner.cpu_token_to_kv_pool_allocator.get_start_loc(forward_batch.gpu_batch_rids)
            seq_lens = forward_batch.gpu_seq_lens.clone().detach()

        if forward_batch.spec_info is not None and forward_batch.forward_mode == ForwardMode.TARGET_VERIFY: # for target verify
            # XXX: 在verify kernel实现的过程中，好像seq_len包含了draft token，我也不是很确定
            seq_lens.add_(forward_batch.spec_info.draft_token_num)
            if self.use_special_less_mask_cpuattention:
                mask_len = forward_batch.spec_info.draft_token_num
                custom_mask = forward_batch.spec_info.custom_mask[forward_batch.spec_info.custom_mask_gpu_len:].to('cpu')
                mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
                seq_mask_len = forward_batch.spec_info.draft_token_num * (forward_batch.cpu_seq_lens + forward_batch.spec_info.draft_token_num)
                mask_indptr[1 : batch_size + 1] = torch.cumsum(seq_mask_len[:batch_size], dim=0)

                # select_slice = torch.tensor([False] * forward_batch.spec_info.custom_mask.
                # shape[0])
                # for i in range(batch_size):
                #     start = mask_indptr[i]
                #     seq_len = seq_lens[i] # include draft token
                #     for j in range(forward_batch.spec_info.draft_token_num):
                #         select_slice[start + (j+1)*seq_len-mask_len : start + (j+1)*seq_len] = True

                # 向量化优化版本：避免双重循环
                # 创建batch和draft token的索引网格
                batch_indices = torch.arange(batch_size, device="cpu")
                draft_indices = torch.arange(forward_batch.spec_info.draft_token_num, device="cpu")
                batch_grid, draft_grid = torch.meshgrid(batch_indices, draft_indices, indexing='ij')
                
                # 展平为一维数组，方便向量化操作
                batch_flat = batch_grid.flatten()  # [batch_size * draft_token_num]
                draft_flat = draft_grid.flatten()  # [batch_size * draft_token_num]
                
                # 向量化计算所有的start位置和seq_len
                starts = mask_indptr[batch_flat]  # [batch_size * draft_token_num]
                seq_lens_expanded = seq_lens[batch_flat].to('cpu')  # [batch_size * draft_token_num]
                
                # 向量化计算所有范围的开始和结束位置
                range_starts = starts + (draft_flat + 1) * seq_lens_expanded - mask_len
                range_ends = starts + (draft_flat + 1) * seq_lens_expanded
                
                # 计算每个范围的长度
                range_lengths = range_ends - range_starts
                total_indices = range_lengths.sum().item()
                
                # 生成所有需要设置为True的索引位置
                if total_indices > 0:
                    # 使用repeat_interleave高效生成所有索引
                    offsets = torch.cat([torch.arange(length, device="cpu") for length in range_lengths])
                    starts_repeated = torch.repeat_interleave(range_starts, range_lengths)
                    all_indices = starts_repeated + offsets
                    
                    # 创建select_slice并一次性设置所有True位置
                    select_slice = torch.zeros(forward_batch.spec_info.custom_mask.shape[0], dtype=torch.bool, device="cpu")
                    select_slice[all_indices] = True
                else:
                    select_slice = torch.zeros(forward_batch.spec_info.custom_mask.shape[0], dtype=torch.bool, device="cpu")
                
                custom_mask = custom_mask[select_slice]
                custom_mask = torch.where(custom_mask, torch.tensor(0, dtype=torch.float32), torch.tensor(-float("inf"), dtype=torch.float32))
                seq_mask_len = torch.tensor([mask_len * mask_len] * batch_size)
                mask_indptr[1 : batch_size + 1] = torch.cumsum(seq_mask_len[:batch_size], dim=0)
                mask_indptr = mask_indptr[: batch_size + 1]
                qo_indptr = None    
            else:
                custom_mask = forward_batch.spec_info.custom_mask[forward_batch.spec_info.custom_mask_gpu_len:].to('cpu')
                custom_mask = torch.where(custom_mask, torch.tensor(0, dtype=torch.float32), torch.tensor(-float("inf"), dtype=torch.float32))
                # custom_mask = torch.zeros_like(custom_mask, dtype=torch.float32)
                seq_mask_len = forward_batch.spec_info.draft_token_num * (forward_batch.cpu_seq_lens + forward_batch.spec_info.draft_token_num)
                mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
                mask_indptr[1 : batch_size + 1] = torch.cumsum(seq_mask_len[:batch_size], dim=0)
                mask_indptr = mask_indptr[: batch_size + 1]
                qo_indptr = None    
                mask_len = None
        elif forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND: # for draft extend
            seq_lens, qo_indptr = forward_batch.spec_info.init_draft_extend_cpu_attention_metadata(forward_batch)
            start_loc = self.model_runner.cpu_token_to_kv_pool_allocator.get_start_loc(forward_batch.gpu_batch_rids + forward_batch.cpu_batch_rids)
            custom_mask = None
            mask_indptr = None
            mask_len = None
        elif forward_batch.forward_mode == ForwardMode.DECODE and forward_batch.spec_info is not None: # for draft generate
            seq_lens = forward_batch.spec_info.seq_len_for_draft_cpu_attention_init
            custom_mask = forward_batch.spec_info.custom_mask_for_draft_cpu_attention_init
            mask_indptr = forward_batch.spec_info.mask_indptr_for_draft_cpu_attention_init
            qo_indptr = forward_batch.spec_info.qo_indptr_for_draft_cpu_attention_init
            mask_len = forward_batch.spec_info.mask_len_for_draft_cpu_attention_init
        else: # for normal decode
            custom_mask = None
            mask_indptr = None
            qo_indptr = None
            mask_len = None
        # 存储元数据以在层间共享
        forward_metadata = ForwardMetadata(
            # blockmask=blockmask,
            start_loc=torch.tensor(start_loc), 
            seq_lens=seq_lens.to('cpu'), 
            custom_mask=custom_mask,
            mask_indptr=mask_indptr,
            qo_indptr=qo_indptr,
            mask_len=mask_len
        )
        if micro_batch_id != -1:
            if len(self.forward_metadata_list) <= micro_batch_id:
                self.forward_metadata_list.append(forward_metadata)
            else:
                self.forward_metadata_list[micro_batch_id] = forward_metadata
        else:
            self.forward_metadata = forward_metadata
        
        # logger.debug(f"已初始化CPU Attention元数据，批次大小={batch_size}")

    def forward_decode(self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch, save_kv_cache: bool, micro_batch_id: int = -1, **kwargs):
        # logger.debug(f"CPU Attention Backend Forward, layer: {layer.layer_id}")
        
        # 从前向批次中获取QKV数据
        if q is None: # draft verify / normal decode阶段
            assert forward_batch.decode_part == DecodePart.CPU_ATTN
            qkv = forward_batch.qkv_pin.view(-1, self.tp_q_head_num + 2 * self.tp_k_head_num, self.head_dim)
            q = qkv[:, :self.tp_q_head_num, :].contiguous()
            k = qkv[:, self.tp_q_head_num : self.tp_q_head_num + self.tp_k_head_num, :].contiguous()
            v = qkv[:, self.tp_q_head_num + self.tp_k_head_num :, :].contiguous()

            if len(forward_batch.gpu_seq_lens) > 0:
                # TODO: 帮助gpu request存kv cache
                qkv_gpu = forward_batch.qkv_pin_gpu.view(-1, self.tp_q_head_num + 2 * self.tp_k_head_num, self.head_dim)
                k_gpu = qkv_gpu[:, self.tp_q_head_num : self.tp_q_head_num + self.tp_k_head_num, :].contiguous()
                v_gpu = qkv_gpu[:, self.tp_q_head_num + self.tp_k_head_num :, :].contiguous()
                forward_batch.cpu_token_to_kv_pool.set_kv_buffer(layer, forward_batch.cpu_out_cache_loc[:forward_batch.gpu_attn_input_ids_size], k_gpu, v_gpu)
            if len(forward_batch.gpu_seq_lens) > 0:
                forward_batch.cpu_token_to_kv_pool.set_kv_buffer(layer, forward_batch.cpu_out_cache_loc[forward_batch.gpu_attn_input_ids_size:], k, v)
            else:
                forward_batch.cpu_token_to_kv_pool.set_kv_buffer(layer, forward_batch.cpu_out_cache_loc, k, v)
        else:
            forward_batch.cpu_token_to_kv_pool.set_kv_buffer(layer, forward_batch.cpu_out_cache_loc, k, v)
        # if layer.layer_id <= 3:
        #     logger.debug(f"q: {q.reshape(-1, 32*128)[:, :4]}")
        #     logger.debug(f"k: {k.reshape(-1, 8*128)[:, :4]}")
        #     logger.debug(f"v: {v.reshape(-1, 8*128)[:, :4]}")

        if micro_batch_id != -1:
            seq_lens = self.forward_metadata_list[micro_batch_id].seq_lens
            start_loc = self.forward_metadata_list[micro_batch_id].start_loc
            custom_mask = self.forward_metadata_list[micro_batch_id].custom_mask
            mask_indptr = self.forward_metadata_list[micro_batch_id].mask_indptr
            qo_indptr = self.forward_metadata_list[micro_batch_id].qo_indptr
            mask_len = self.forward_metadata_list[micro_batch_id].mask_len
        else:
            seq_lens = self.forward_metadata.seq_lens
            start_loc = self.forward_metadata.start_loc
            custom_mask = self.forward_metadata.custom_mask
            mask_indptr = self.forward_metadata.mask_indptr
            qo_indptr = self.forward_metadata.qo_indptr
            mask_len = self.forward_metadata.mask_len

        # 调用FlexAttention实现
        res = CpuAttention(
            forward_mode=forward_batch.forward_mode,
            # flex_attention=self.flex_attention,
            cpu_output=forward_batch.hidden_pin,
            q=q, 
            k_cache=forward_batch.cpu_token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=forward_batch.cpu_token_to_kv_pool.get_value_buffer(layer.layer_id),
            seq_lens=seq_lens,
            start_loc=start_loc,
            # blockmask=self.forward_metadata.blockmask,
            is_gqa=self.is_gqa,
            layer=layer.layer_id, 
            custom_mask=custom_mask,
            mask_indptr=mask_indptr, 
            qo_indptr=qo_indptr,
            mask_len=mask_len,
            use_less_mask_cpuattention=self.use_special_less_mask_cpuattention,
        )
        
        return res

    def forward_extend(self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch, save_kv_cache: bool, micro_batch_id: int = -1, **kwargs):
        return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, micro_batch_id, **kwargs)

class CPUMultiStepDraftBackend:
    """
    Wrap multiple CPU attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.model_runner = model_runner
        max_bs = model_runner.cpu_req_to_token_pool.size * self.topk
        
        # 为每个步骤创建独立的 CPU attention backend
        self.attn_backends: List[CPUAttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(CPUAttnBackend(model_runner))
        
        self.tp_q_head_num = model_runner.model_config.num_attention_heads
        self.tp_k_head_num = model_runner.model_config.num_key_value_heads
        self.head_dim = model_runner.model_config.head_dim
        self.max_context_len = model_runner.server_args.max_seq_length
        self.use_special_less_mask_cpuattention = model_runner.server_args.use_special_less_mask_cpuattention
        
        # CPU 不需要复杂的 GPU device 管理
        logger.debug(f"初始化 CPU 多步 draft backend，步数={speculative_num_steps}, topk={topk}")

    def init_forward_metadata(self, forward_batch: ForwardBatch, process_cpu_requets: bool = True):
        """为所有步骤初始化前向传播元数据"""
        
        def call_fn(i, forward_batch: ForwardBatch):
            # 为每个步骤初始化其对应的 backend 元数据
            self.attn_backends[i].init_forward_metadata(forward_batch, process_cpu_requets)


        self.common_template(forward_batch, call_fn, process_cpu_requets)

    def common_template(self, forward_batch: ForwardBatch, call_fn, process_cpu_requets: bool = True):
        """处理多步 draft decoding 的通用模板"""
        if process_cpu_requets:
            bs = len(forward_batch.cpu_seq_lens)
        else:
            bs = len(forward_batch.gpu_seq_lens)

        if self.use_special_less_mask_cpuattention:
            for i in range(self.speculative_num_steps):
                if process_cpu_requets:
                    forward_batch.spec_info.seq_len_for_draft_cpu_attention_init = (forward_batch.cpu_seq_lens + (i + 1) * self.topk).to(dtype=torch.int64, device='cpu')
                else:
                    forward_batch.spec_info.seq_len_for_draft_cpu_attention_init = (forward_batch.gpu_seq_lens + (i + 1) * self.topk).to(dtype=torch.int64, device='cpu')
                
                mask_indptr = torch.zeros(bs + 1, dtype=torch.int64, device="cpu")
                seq_mask_len = self.topk * torch.tensor([(i+1)*self.topk] * bs)
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                forward_batch.spec_info.mask_indptr_for_draft_cpu_attention_init = mask_indptr
                mask_len = (i + 1) * self.topk
                forward_batch.spec_info.mask_len_for_draft_cpu_attention_init = mask_len

                mask = torch.zeros((i+1) * self.topk * self.topk, dtype=torch.float32, device="cpu")
                
                begin = 0
                for k in range(self.topk):
                    mask[begin:begin+(i+1)*self.topk] = -float("inf") # 将当前step的token遮住
                    mask[begin + i*self.topk + k] = 0 
                    if i > 0:
                        mask[begin : begin + i*self.topk + k] = -float("inf") # 先将前面step的位置掩盖住，后续会根据树形依赖修改
                        for ii in range(i):
                            mask[begin + k + ii*self.topk] = 0
                    begin += (i + 1) * self.topk

                # 将mask重复bs遍
                forward_batch.spec_info.custom_mask_for_draft_cpu_attention_init = mask.repeat(bs)
                qo_indptr = torch.zeros((bs + 1,), dtype=torch.int64, device="cpu")
                qo_indptr[1:] = torch.cumsum(torch.tensor([self.topk] * bs), dim=0) 
                forward_batch.spec_info.qo_indptr_for_draft_cpu_attention_init = qo_indptr
                # 为每个步骤调用相应的函数
                call_fn(i, forward_batch)
        else:
            for i in range(self.speculative_num_steps):
                if process_cpu_requets:
                    forward_batch.spec_info.seq_len_for_draft_cpu_attention_init = (forward_batch.cpu_seq_lens + (i + 1) * self.topk).to(dtype=torch.int64, device='cpu')
                else:
                    forward_batch.spec_info.seq_len_for_draft_cpu_attention_init = (forward_batch.gpu_seq_lens + (i + 1) * self.topk).to(dtype=torch.int64, device='cpu')
                seq_mask_len = self.topk * forward_batch.spec_info.seq_len_for_draft_cpu_attention_init
                mask_indptr = torch.zeros(bs + 1, dtype=torch.int64, device="cpu")
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                forward_batch.spec_info.mask_indptr_for_draft_cpu_attention_init = mask_indptr
                mask = torch.zeros(mask_indptr[-1].item(), dtype=torch.float32, device="cpu")
                
                # 完全向量化的优化版本：避免嵌套循环
                begins = mask_indptr[:-1]  # [bs]
                seq_lens = forward_batch.spec_info.seq_len_for_draft_cpu_attention_init  # [bs]
                if process_cpu_requets:
                    real_seq_lens = forward_batch.cpu_seq_lens.to(dtype=torch.int64, device='cpu')  # [bs]
                else:
                    real_seq_lens = forward_batch.gpu_seq_lens.to(dtype=torch.int64, device='cpu')  # [bs]
                
                # 向量化计算所有batch和topk的组合
                # 创建batch和topk的网格
                batch_grid, topk_grid = torch.meshgrid(
                    torch.arange(bs, device='cpu'),
                    torch.arange(self.topk, device='cpu'),
                    indexing='ij'
                )  # [bs, topk]
                
                # 展平为一维，方便向量化操作
                batch_flat = batch_grid.flatten()  # [bs * topk]
                topk_flat = topk_grid.flatten()    # [bs * topk]
                
                # 向量化计算每个(batch, topk)对的开始位置
                begins_expanded = begins[batch_flat]  # [bs * topk]
                seq_lens_expanded = seq_lens[batch_flat]  # [bs * topk]
                real_seq_lens_expanded = real_seq_lens[batch_flat]  # [bs * topk]
                
                # 累计偏移：每个topk位置的begin需要加上前面topk的seql
                offset_multiplier = topk_flat  # [bs * topk]
                seq_offsets = offset_multiplier * seq_lens_expanded  # [bs * topk]
                current_begins = begins_expanded + seq_offsets  # [bs * topk]
                
                # 计算每个位置需要设置为-inf的范围
                range_starts = current_begins + seq_lens_expanded - self.topk  # [bs * topk]
                range_ends = current_begins + seq_lens_expanded  # [bs * topk]
                
                # 计算需要设置为0的位置
                zero_positions = current_begins + seq_lens_expanded - (self.topk - topk_flat)  # [bs * topk]
                
                # 完全向量化生成-inf位置索引
                # 创建有效范围的掩码
                valid_ranges = range_starts < range_ends
                valid_starts = range_starts[valid_ranges]
                valid_ends = range_ends[valid_ranges]
                
                # 计算总的索引数量
                range_lengths = valid_ends - valid_starts
                total_indices = range_lengths.sum().item()
                
                # 使用repeat_interleave高效生成所有索引
                if total_indices > 0:
                    # 创建偏移量数组
                    # XXX: 下列代码经过大量小tensor的分配释放后，内存变得碎片化，分配器需要整理内存，所以有时候会触发swap.
                    # offsets = torch.cat([torch.arange(length, device='cpu') for length in range_lengths])
                    # 预分配整个offsets tensor，这样可以减少内存分配器的压力
                    offsets = torch.empty(total_indices, dtype=torch.int64, device='cpu')
                    start_idx = 0
                    for length in range_lengths:
                        end_idx = start_idx + length
                        offsets[start_idx:end_idx] = torch.arange(length, device='cpu')
                        start_idx = end_idx
                    # 使用repeat_interleave生成起始位置
                    starts_repeated = torch.repeat_interleave(valid_starts, range_lengths)
                    # 生成所有mask索引
                    all_mask_indices = [starts_repeated + offsets]
                else:
                    all_mask_indices = []
                
                # 处理i > 0的情况（前面step的依赖）
                all_zero_indices = [zero_positions]
                if i > 0:
                    # 向量化计算前面step的-inf范围
                    prev_starts = begins + real_seq_lens  # [bs]
                    prev_ends = begins + real_seq_lens + i * self.topk  # [bs]
                    prev_valid = prev_starts < prev_ends
                    
                    if prev_valid.any():
                        prev_valid_starts = prev_starts[prev_valid]
                        prev_valid_ends = prev_ends[prev_valid]
                        prev_lengths = prev_valid_ends - prev_valid_starts
                        prev_total = prev_lengths.sum().item()
                        
                        if prev_total > 0:
                            prev_offsets = torch.cat([torch.arange(length, device='cpu') for length in prev_lengths])
                            prev_starts_repeated = torch.repeat_interleave(prev_valid_starts, prev_lengths)
                            all_mask_indices.append(prev_starts_repeated + prev_offsets)
                    
                    # 完全向量化计算依赖位置
                    # 为每个batch创建依赖索引
                    batch_indices_expanded = torch.arange(bs, device='cpu').unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]
                    step_indices = torch.arange(i, device='cpu').unsqueeze(0).unsqueeze(2)  # [1, i, 1]
                    topk_indices_dep = torch.arange(self.topk, device='cpu').unsqueeze(0).unsqueeze(0)  # [1, 1, topk]
                    
                    # 计算依赖位置：begin + real_seql + step * topk + k
                    dep_positions = (begins.unsqueeze(1).unsqueeze(2) +  # [bs, 1, 1]
                                real_seq_lens.unsqueeze(1).unsqueeze(2) +  # [bs, 1, 1]
                                step_indices * self.topk +  # [1, i, 1]
                                topk_indices_dep)  # [1, 1, topk]
                    # 结果形状: [bs, i, topk]
                    
                    all_zero_indices.append(dep_positions.flatten())
                
                # 批量设置mask值
                if all_mask_indices:
                    mask_indices = torch.cat(all_mask_indices)
                    mask[mask_indices] = -float("inf")
                
                if all_zero_indices:
                    zero_indices = torch.cat(all_zero_indices)
                    mask[zero_indices] = 0
                
                forward_batch.spec_info.custom_mask_for_draft_cpu_attention_init = mask
                qo_indptr = torch.zeros((bs + 1,), dtype=torch.int64, device="cpu")
                qo_indptr[1:] = torch.cumsum(torch.tensor([self.topk] * bs), dim=0) 
                forward_batch.spec_info.qo_indptr_for_draft_cpu_attention_init = qo_indptr
                forward_batch.spec_info.mask_len_for_draft_cpu_attention_init = None
                # 为每个步骤调用相应的函数
                call_fn(i, forward_batch)
    
    def dynamically_change_metadata(self, i:int, forward_batch: ForwardBatch):
        """当得知真实的依赖关系后，需要更改mask"""
        pass