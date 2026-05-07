from typing import Optional, Tuple, List, Dict
import torch
import triton
import triton.language as tl
import logging
import math
import time
import random
from concurrent.futures import ThreadPoolExecutor, Future

from sglang.srt.utils import fast_topk
from sglang.srt.utils import get_available_gpu_memory

from specmoe.backend.tp_worker import TpModelWorker
from specmoe.utils.server_args import ServerArgs
from specmoe.utils.data_info import ScheduleBatch, ForwardMode, CaptureHiddenMode, ModelWorkerBatch, write_req_to_token_pool_triton
from specmoe.speculative.spec_info import SpeculativeAlgorithm
from specmoe.speculative.speculative_utils import EagleDraftInput, EagleVerifyInput, EagleVerifyOutput, assign_draft_cache_locs, select_top_k_tokens, split_schedule_batch_for_draft
from specmoe.backend.forward_batch_info import ForwardBatch
from specmoe.layers.logits_processor import LogitsProcessorOutput
from specmoe.backend.policy import Policy
from specmoe.backend.memory import TokenToKVPoolAllocatorGPU, MHATokenToKVPool, ReqToTokenPool

logger = logging.getLogger(__name__)

class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
        prefill_policy: Policy, 
        decode_policy: Policy,
    ):
        # super().__init__(server_args, gpu_id, tp_rank, dp_rank, nccl_port, target_worker)
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy

        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.padded_static_len = self.speculative_num_steps + 1
        self.gpu_topk = server_args.speculative_eagle_topk_gpu
        self.gpu_speculative_num_steps = server_args.speculative_num_steps_gpu

        self.enable_nan_detection = server_args.enable_nan_detection

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        
        self.draft_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2) # CPU一个worker， GPU一个worker
        self.draft_futures: List[Future] = [None, None]

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.gpu_req_to_token_pool, self.gpu_token_to_kv_pool_allocator, self.cpu_req_to_token_pool, self.cpu_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        super().__init__(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            server_args=server_args,
            nccl_port=nccl_port,
            dp_rank=dp_rank,
            is_draft_worker=True,
            gpu_req_to_token_pool=self.gpu_req_to_token_pool,
            gpu_token_to_kv_pool_allocator=self.gpu_token_to_kv_pool_allocator,
            cpu_req_to_token_pool=self.cpu_req_to_token_pool,
            cpu_token_to_kv_pool_allocator=self.cpu_token_to_kv_pool_allocator,
            offload=False
        )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 models don't share lm_head
            # self.draft_model_runner.model.set_embed(embed.clone())
            self.draft_model_runner.model.set_embed(embed) 
        else:
            # Share the embedding and lm_head
            # self.draft_model_runner.model.set_embed_and_head(embed.clone(), head.clone())
            self.draft_model_runner.model.set_embed_and_head(embed, head)

    @property
    def draft_model_runner(self):
        return self.model_runner

    def init_attention_backend(self):
        # 更新gpu_attn_backend的req_to_token为Draft专用的
        self.draft_model_runner.gpu_attn_backend.req_to_token = self.draft_gpu_req_to_token_pool.req_to_token
        # 初始化专门用于draft generate的attention backend，(用于prefill的attention backend已经在scheduler里draft_worker.init_execution_engine里初始化了)
        from specmoe.layers.attention_backend.triton_backend import (
            TritonMultiStepDraftBackend,
        )
        self.draft_gpu_exec_attn_backend = TritonMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            for_draft_generate=True,
            draft_gpu_req_to_token_pool=self.draft_gpu_req_to_token_pool,
        )
        if self.server_args.decode_gpu_attention_ratio != 0:
            self.draft_gpu_exec_attn_backend_for_gpu_request = TritonMultiStepDraftBackend(
                self.draft_model_runner,
                self.gpu_topk,
                self.gpu_speculative_num_steps,
                for_draft_generate=True, 
                draft_gpu_req_to_token_pool=self.draft_gpu_req_to_token_pool,
            )
        self.draft_extend_attn_backend = None
        self.padded_static_len = self.speculative_num_steps + 1
        self.has_prefill_wrapper_verify = False
        # draft model only run on the GPU, so there is no need to init cpu attention backend

        from specmoe.layers.attention_backend.cpu_backend import CPUMultiStepDraftBackend
        self.draft_cpu_exec_attn_backend = CPUMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )


    def switch_to_decode(self):
        begin_time = time.time()
        # 看是否将draft model从GPU上移到CPU上，并且删除gpu kv cache
        if self.server_args.draft_model_placement == 'CPU':
            # 将draft model从GPU上移到CPU上
            # self.draft_model_runner.model = self.draft_model_runner.model.to('cpu')
            # self.draft_model_runner.model.model.embed_tokens.weight = self.draft_model_runner.model.model.embed_tokens.weight.to('cpu')
            # self.draft_model_runner.model.lm_head.weight = self.draft_model_runner.model.lm_head.weight.to('cpu')
            # 删除gpu kv cache
            pass
            # self.draft_model_runner.gpu_req_to_token_pool.clear()
        elif self.server_args.draft_model_placement == 'GPU':
            pass
        else:
            assert False, "Should never reach here"
        
        # if self.decode_policy.draft_gpu_execution_ratio == 0:
        #     self.draft_model_runner.gpu_token_to_kv_pool._clear_buffers()
        #     # assert False
        # else:
        # self.draft_model_runner.gpu_token_to_kv_pool.adjust_capacity(1, self.server_args.max_draft_gpu_req_total_length_in_a_batch, use_pinned_memory=False)
        draft_gpu_slot_num = max(int(self.server_args.allocate_cpu_slot * self.decode_policy.draft_gpu_execution_ratio), 1)
        # draft_gpu_slot_num = self.server_args.max_draft_gpu_req_total_length_in_a_batch # XXX: FOR DEBUG
        if self.server_args.draft_kv_cache_slot != 1:
            draft_gpu_slot_num = self.server_args.draft_kv_cache_slot
        self.draft_model_runner.gpu_token_to_kv_pool.adjust_capacity(1, draft_gpu_slot_num, use_pinned_memory=False)
        # XXX: 需要为draft gpu执行分配新的req_to_token_pool和allocator，因为在decode阶段和target model执行不同步共享request了
        self.draft_gpu_req_to_token_pool = ReqToTokenPool(
            # size=int(self.decode_policy.draft_gpu_req_num * 1.3), # reqtotokenpool占用内存很小，而req数量也不是固定的只有draft_gpu_exec_ratio个，所以多分配一些
            size=self.server_args.max_running_requests, # reqtotokenpool占用内存很小，而req数量也不是固定的只有draft_gpu_exec_ratio个，所以多分配一些
            max_context_len=self.server_args.max_seq_length + self.server_args.max_output_length + self.server_args.max_speculative_draft_tokens,
            device='cuda',
        )
        self.draft_gpu_token_to_kv_pool_allocator = TokenToKVPoolAllocatorGPU(
            size=draft_gpu_slot_num,
            dtype=self.draft_model_runner.kv_cache_dtype,
            device='cuda',
            kvcache=self.draft_model_runner.gpu_token_to_kv_pool,
        )
        self.draft_gpu_cached_rids: List[int] = []
        self.draft_gpu_cached_rids_every_mb: Dict[int, List[int]] = {} # schedule micro batch id to cached rids
        self.draft_gpu_pool_req_indices: Dict[int, int] = {} # rid to req_pool_index
        self.have_new_finish_reqs: bool = False

            
        self.gpu_draft_stream = torch.cuda.Stream(priority=0)
        self.cpu_draft_stream = torch.cuda.Stream(priority=-10)

        self.init_attention_backend()

        end_time = time.time()
        logger.info(f"Draft Model Runner switch to decode stage, time cost: {end_time - begin_time:.2f}s. After switch, gpu available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB")

    def forward_batch_speculative_generation(self, schedule_micro_batches: List[ScheduleBatch]):
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        forward_mode = schedule_micro_batches[0].forward_mode
        if forward_mode == ForwardMode.EXTEND:
            logits_outputs, next_token_ids, bids = self.forward_target_extend(schedule_micro_batches)
            self.forward_draft_extend(
                schedule_micro_batches, logits_outputs, next_token_ids
            )
            return logits_outputs, next_token_ids, bids, 0, 0, None, None, None
        elif forward_mode == ForwardMode.DECODE:
            spec_infos, draft_times = self.draft(schedule_micro_batches)
            # logger.debug(f"spec_infos after draft: {spec_infos}")
            logits_outputs, verify_outputs, model_worker_micro_batches, verify_time, target_model_performance_recoder = self.verify(
                schedule_micro_batches, spec_infos
            )

            verified_id_list = []
            bid_list = []
            accept_length = 0
            num_total_requets = 0

            for i, verify_output in enumerate(verify_outputs):
                verified_id_list.append(verify_output.verified_id)
                bid_list.append(model_worker_micro_batches[i].bid)
                accept_length += sum(verify_output.accept_length_per_req_cpu)
                num_total_requets += len(model_worker_micro_batches[i].cpu_reqs_rids) + len(model_worker_micro_batches[i].gpu_reqs_rids)

            return (
                logits_outputs,
                verified_id_list,
                bid_list,
                accept_length, 
                num_total_requets,
                draft_times,
                verify_time,
                target_model_performance_recoder
            )
        else:
            assert False

    def _prepare_kvcache_for_draft(self, schedule_batch: ScheduleBatch, batch_id: int, gpu_execution: bool = True):
        '''
            为request分配draft KV cache空间，并将现有的KV cache传输到GPU中
        '''

        # TODO: 如果将new verified token extend放到draft之前，就需要在req中维护一下上次还未计算kv cache的verified token id，并且在分配内存时要考虑这些。（因为r.fill_ids有可能已经包含了verified token id，但是实际上draft model还没得到其kv cache）
        if schedule_batch.seq_lens is None:
            # 如果紧接着prefill阶段后面，则seq_lens是None，重新计算这些值
            assert schedule_batch.spec_info.first_step_after_prefill
            seq_lens = torch.tensor([len(r.fill_ids) for r in schedule_batch.reqs], dtype=torch.int64)
            seq_lens_sum = seq_lens.sum().item()
            if gpu_execution:
                schedule_batch.seq_lens = seq_lens.to('cuda')
            else:
                schedule_batch.seq_lens = seq_lens.to('cpu')
            schedule_batch.seq_lens_sum = seq_lens_sum

            if gpu_execution:
                gpu_req_num = schedule_batch.policy.draft_gpu_req_num #
            else:
                gpu_req_num = 0
            is_req_on_gpu_tensor = torch.tensor([True] * gpu_req_num + [False] * (len(schedule_batch.reqs) - gpu_req_num), dtype=torch.bool)
            schedule_batch.cpu_seq_lens = schedule_batch.seq_lens[~is_req_on_gpu_tensor].to(device='cpu')
            schedule_batch.gpu_seq_lens = schedule_batch.seq_lens[is_req_on_gpu_tensor].to(device='cuda')

            schedule_batch.gpu_seq_lens_sum = schedule_batch.gpu_seq_lens.sum()
            schedule_batch.cpu_seq_lens_sum = schedule_batch.cpu_seq_lens.sum()
        else:
            # 如果当前是decode阶段后面的另一个decode，那么此时seq_lens应该已经包含了target verified tokens，因此要加载这些kv cache，需要刨除这些virified tokens.
            assert not schedule_batch.spec_info.first_step_after_prefill
            seq_lens = schedule_batch.seq_lens - schedule_batch.spec_info.accept_length - 1 # TODO: 需要检查是否要有这个1
            seq_lens_sum = seq_lens.sum().item()

        if gpu_execution and schedule_batch.policy.draft_gpu_req_num > 0:
            # logger.debug(f"seq_lens: {seq_lens}, seq_lens_sum: {seq_lens_sum}, num_seqs: {num_seqs}")

            num_seqs = schedule_batch.policy.draft_gpu_req_num
            assert num_seqs == len(schedule_batch.seq_lens)



            # # 分配req slots
            # schedule_batch.gpu_req_pool_indices = torch.tensor(schedule_batch.gpu_alloc_req_slots(num_seqs), device='cuda')
            schedule_batch.gpu_req_pool_indices = torch.tensor([self.draft_gpu_pool_req_indices[r.rid] for r in schedule_batch.reqs[:num_seqs]], device='cuda')

    def draft_tokens_generate(self, schedule_batch: ScheduleBatch, process_cpu_attn_requets: bool = True, gpu_execution: bool = True) -> EagleVerifyInput:
        '''
        情况1: process_cpu_attn_requets = True, gpu_execution = False: 用CPU执行CPU attn requests
        情况2: process_cpu_attn_requets = True, gpu_execution = True: 用GPU执行CPU attn requests
        情况3: process_cpu_attn_requets = False, gpu_execution = True: 用GPU执行GPU attn requests
        '''
        # 在分裂后的batch中，gpu_seq_lens和cpu_seq_lens已经正确分配
        # 不需要再用gpu_request_num来区分
        if process_cpu_attn_requets:
            num_seqs = len(schedule_batch.cpu_seq_lens)
            if num_seqs == 0:
                return None
        else:
            num_seqs = len(schedule_batch.gpu_seq_lens)
            if num_seqs == 0:
                return None
                
        spec_info = schedule_batch.spec_info
        # logger.debug(f"spec info: \n hidden_states: {spec_info.hidden_states.shape}, topk_p: {spec_info.topk_p.shape}, topk_index: {spec_info.topk_index.shape}")

        # 为draft tokens分配新的token slot
        if gpu_execution:
            if process_cpu_attn_requets:
                gpu_out_cache_loc, draft_gpu_token_to_kv_pool_state_backup = schedule_batch.gpu_alloc_token_slots(
                    num_seqs * self.topk * self.speculative_num_steps, backup_state=True, specified_allocator=self.draft_gpu_token_to_kv_pool_allocator
                )
                # Allocate cache locations
                # Layout of the out_cache_loc
                # [       iter 0         ] [       iter 1         ]
                # [topk=0, topk=1, topk=2] [topk=0, topk=1, topk=2]
                
                # 在分裂后的batch中，gpu_req_pool_indices应该对应正确的请求索引
                # 对于CPU requests处理，使用cpu_seq_lens对应的索引
                # 在分裂后的batch中，需要根据实际的请求来分配
                cpu_req_start_idx = len(schedule_batch.gpu_seq_lens)  # CPU requests的起始索引
                effective_req_indices = schedule_batch.gpu_req_pool_indices[cpu_req_start_idx:cpu_req_start_idx + num_seqs]
                
                assign_draft_cache_locs[(num_seqs,)](
                    effective_req_indices,
                    self.draft_gpu_req_to_token_pool.req_to_token,
                    schedule_batch.cpu_seq_lens.to(device='cuda'),
                    gpu_out_cache_loc.to(device='cuda'),
                    self.draft_gpu_req_to_token_pool.req_to_token.shape[1],
                    self.topk,
                    self.speculative_num_steps,
                    1,
                )
                schedule_batch.gpu_out_cache_loc = gpu_out_cache_loc
            else:
                gpu_out_cache_loc, draft_gpu_token_to_kv_pool_state_backup = schedule_batch.gpu_alloc_token_slots(
                    num_seqs * self.gpu_topk * self.gpu_speculative_num_steps, backup_state=True, 
                    specified_allocator=self.draft_gpu_token_to_kv_pool_allocator
                )
                # Allocate cache locations
                # Layout of the out_cache_loc
                # [       iter 0         ] [       iter 1         ]
                # [topk=0, topk=1, topk=2] [topk=0, topk=1, topk=2]
                
                assign_draft_cache_locs[(num_seqs,)](
                    schedule_batch.gpu_req_pool_indices[:num_seqs],
                    self.draft_gpu_req_to_token_pool.req_to_token,
                    schedule_batch.gpu_seq_lens.to(device='cuda'),
                    gpu_out_cache_loc.to(device='cuda'),
                    self.draft_gpu_req_to_token_pool.req_to_token.shape[1],
                    self.gpu_topk,
                    self.gpu_speculative_num_steps,
                    1,
                )
                schedule_batch.gpu_out_cache_loc = gpu_out_cache_loc
        else:
            if process_cpu_attn_requets:
                num_tokens = [self.topk * self.speculative_num_steps] * num_seqs
            else:
                assert False, "GPU Attn Request's Draft must processed in GPU"
            cpu_out_cache_loc, cpu_token_to_kv_pool_state_backup = schedule_batch.cpu_alloc_token_slots_decode( 
                [r.rid for r in schedule_batch.reqs], num_tokens, backup_state=True
            )
            schedule_batch.cpu_out_cache_loc = torch.tensor(cpu_out_cache_loc)
           
        # XXX: target model和draft model共用allocater，合理吗？
        # draft阶段，draft model会申请一些gpu slot用于存放draft tokens kv，当计算完这些draft token logits后，恢复gpu slot状态。
        # target model申请cpu slot用于存放draft tokens kv，计算完这些draft token logits后，进行Verify验证哪些token是否成功，将成功验证的token进行紧凑操作，然后释放掉未成功验证的token slot。
        # 构造 DraftInput，用于下一回合draft model将成功生成的token进行extend。
        
        # logger.debug(f"schedule_batch.seq_lens: {schedule_batch.seq_lens}")
        if process_cpu_attn_requets:
            spec_info.positions = schedule_batch.cpu_seq_lens.repeat_interleave(self.topk, dim=0)
        else:
            spec_info.positions = schedule_batch.gpu_seq_lens.repeat_interleave(self.gpu_topk, dim=0)

        # Get forward batch
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = schedule_batch.get_model_worker_batch()

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )

        # XXX
        forward_batch.gpu_req_to_token_pool = self.draft_gpu_req_to_token_pool
        forward_batch.gpu_token_to_kv_pool = self.draft_gpu_token_to_kv_pool_allocator._kvcache

        # Initialize attention backend
        if gpu_execution:
            if process_cpu_attn_requets:
                self.draft_gpu_exec_attn_backend.init_forward_metadata(forward_batch, process_cpu_attn_request=True)
            else:
                self.draft_gpu_exec_attn_backend_for_gpu_request.init_forward_metadata(forward_batch, process_cpu_attn_request=False)
        else:
            if process_cpu_attn_requets:
                self.draft_cpu_exec_attn_backend.init_forward_metadata(forward_batch)
            else:
                assert False, "GPU Attn Request's Draft must processed in GPU"
            
        # Run forward steps
        score_list, token_list, parents_list = self.draft_token_generate_forward(forward_batch, process_cpu_attn_requets, gpu_execution)

        # 恢复GPU和CPU的token slot状态
        if gpu_execution:
            self.draft_gpu_token_to_kv_pool_allocator.restore_state(draft_gpu_token_to_kv_pool_state_backup) 
        else:
            self.cpu_token_to_kv_pool_allocator.restore_state(cpu_token_to_kv_pool_state_backup)

        if process_cpu_attn_requets:
            # 在分裂后的batch中，verified_id应该对应正确的request
            if gpu_execution:
                verified_id = spec_info.verified_id[len(schedule_batch.gpu_seq_lens):]
            else:
                verified_id = spec_info.verified_id
            seq_lens = schedule_batch.cpu_seq_lens.to(device='cuda')
            seq_lens_sum = schedule_batch.cpu_seq_lens_sum
            topk = self.topk
            speculative_num_steps = self.speculative_num_steps
            num_draft_tokens = self.server_args.speculative_num_draft_tokens
        else:
            if gpu_execution:
                verified_id = spec_info.verified_id[:len(schedule_batch.gpu_seq_lens)]
            else:
                assert False, "GPU Attn Request's Draft must processed in GPU"
            seq_lens = schedule_batch.gpu_seq_lens.to(device='cuda')
            seq_lens_sum = schedule_batch.gpu_seq_lens_sum
            topk = self.gpu_topk
            speculative_num_steps = self.gpu_speculative_num_steps
            num_draft_tokens = self.server_args.speculative_num_draft_tokens_gpu

        res = EagleVerifyInput.create(
            verified_id,
            score_list,
            token_list,
            parents_list,
            seq_lens,
            seq_lens_sum, 
            topk,
            speculative_num_steps,
            num_draft_tokens,
        )
        logger.debug(f"draft_tokens: {res.draft_token}")
        return res
    
    


    def draft_thread_func(self, schedule_micro_batches: List[ScheduleBatch], gpu_execution: bool = True):
        '''
        处理分裂后的micro-batch的draft操作
        gpu_execution=True时处理GPU执行的batch
        gpu_execution=False时处理CPU执行的batch
        '''
        draft_begin_time = time.time()
        execution_type = "GPU" if gpu_execution else "CPU"
        logger.debug(f"{execution_type} draft phase starts")
        ret = []
        
        with torch.cuda.stream(self.gpu_draft_stream if gpu_execution else self.cpu_draft_stream):
            for i, schedule_batch in enumerate(schedule_micro_batches):
                if schedule_batch is None:
                    ret.append(None)
                    continue
                # 为draft model KV cache分配GPU显存空间，并且传输KV cache到GPU中
                self._prepare_kvcache_for_draft(schedule_batch, i, gpu_execution)
                origin_draft_gpu_token_to_kv_pool_state_before_tokengenerate = None
                if not schedule_batch.spec_info.first_step_after_prefill:
                    # 如果不是prefill后的第一次decode，就得先过extend算出来还未计算的verified tokens kv cache
                    origin_draft_gpu_token_to_kv_pool_state_before_tokengenerate = self.forward_draft_extend_after_decode(schedule_batch, i, gpu_execution)
                else:
                    schedule_batch.spec_info.first_step_after_prefill = False
                

                # 根据是GPU执行还是CPU执行来选择不同的策略
                if gpu_execution:
                    # GPU线程：先处理GPU attention request，再处理CPU attention request中剩余的部分
                    schedule_batch.gpu_seq_lens = schedule_batch.seq_lens[:schedule_batch.policy.gpu_attention_micro_batch_size]
                    schedule_batch.cpu_seq_lens = schedule_batch.seq_lens[schedule_batch.policy.gpu_attention_micro_batch_size:]
                    schedule_batch.gpu_seq_lens_sum = schedule_batch.gpu_seq_lens.sum()
                    schedule_batch.cpu_seq_lens_sum = schedule_batch.cpu_seq_lens.sum()

                    if len(schedule_batch.gpu_seq_lens) > 0:
                        # 处理GPU attention request
                        res_gpu = self.draft_tokens_generate(schedule_batch, process_cpu_attn_requets=False, gpu_execution=True)
                    else:
                        res_gpu = None
                        
                    if len(schedule_batch.cpu_seq_lens) > 0:
                        # 处理CPU attention request中剩余的部分
                        res_cpu = self.draft_tokens_generate(schedule_batch, process_cpu_attn_requets=True, gpu_execution=True)
                    else:
                        res_cpu = None
                        
                    res = EagleVerifyInput.merge(res_gpu, res_cpu, merge_in_gpu_execution=True)
                else:
                    # CPU线程：只处理CPU execution的request
                    # 由于传入的batch已经是分裂后的CPU batch，直接处理即可
                    res = self.draft_tokens_generate(schedule_batch, process_cpu_attn_requets=True, gpu_execution=False)

                # 清理资源
                if gpu_execution:
                    if origin_draft_gpu_token_to_kv_pool_state_before_tokengenerate is not None:
                        self.draft_gpu_token_to_kv_pool_allocator.restore_state(origin_draft_gpu_token_to_kv_pool_state_before_tokengenerate)
                else:
                    pass # 没有东西需要清理，已经在draft_tokens_generate中清理了

                ret.append(res)
            
        draft_time = time.time() - draft_begin_time
        logger.debug(f"{execution_type} draft phase ends")
        return ret, draft_time

    def preprocess_draft_kv_cache(self, schedule_micro_batches: List[ScheduleBatch]):
        if schedule_micro_batches[0].spec_info.first_step_after_prefill:
            # 初始化draft kv cache，遍历每个micro-batches，逐个添加到draft kv cache里，直到draft kv cache满了
            seq_lens_list: List[torch.Tensor] = []
            for schedule_batch in schedule_micro_batches:
                self.draft_gpu_cached_rids_every_mb[schedule_batch.decode_schedule_batch_id] = []
                seq_lens_list.append(torch.tensor([len(r.fill_ids) for r in schedule_batch.reqs], dtype=torch.int64))
            mb_req_ptr: List[int] = [0] * len(schedule_micro_batches)
            fill_finished = False
            while not fill_finished :
                if sum(mb_req_ptr) == sum([len(seq_lens_list[i]) for i in range(len(schedule_micro_batches))]):
                    fill_finished = True
                    break
                for i in range(len(schedule_micro_batches)):
                    if fill_finished: break
                    if len(seq_lens_list[i]) == 0:
                        continue
                    if len(seq_lens_list[i]) <= mb_req_ptr[i]:
                        continue
                    # print(f"batch: {i}, mb_req_ptr: {mb_req_ptr[i]}")
                    choosed_req = schedule_micro_batches[i].reqs[mb_req_ptr[i]]
                    choosed_seq_len = seq_lens_list[i][mb_req_ptr[i]]
                    mb_req_ptr[i] += 1
                    if self.draft_gpu_token_to_kv_pool_allocator.available_size() * 0.9 < choosed_seq_len + (len(self.draft_gpu_cached_rids) + 1) * self.server_args.max_speculative_draft_tokens: # 留下一点空间
                        fill_finished = True
                        break
                    req_pool_index = self.draft_gpu_req_to_token_pool.alloc(1)[0] # 分配req slot
                    gpu_loc = self.draft_gpu_token_to_kv_pool_allocator.alloc(choosed_seq_len).to(dtype=torch.int32) # 分配kv slot
                    self.draft_gpu_cached_rids.append(choosed_req.rid) # 更新cached_rids
                    self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id].append(choosed_req.rid)
                    self.draft_gpu_pool_req_indices.update({choosed_req.rid: req_pool_index}) # 更新pool_req_indices
                    
                    # 更新gpu_req_to_token_pool，写入prefix token的地址
                    # print(f"choosed_seq_len: {choosed_seq_len}, gpu_loc: {len(gpu_loc)}")
                    self.draft_gpu_req_to_token_pool.write((req_pool_index, slice(0, choosed_seq_len)), gpu_loc) 

                    # 传输kv cache
                    req_start_loc = self.cpu_token_to_kv_pool_allocator.get_start_loc([choosed_req.rid])[0]
                    request_kvcache_cpu_slice = slice(req_start_loc, req_start_loc + choosed_seq_len)
                    cpu_kv_cache = self.draft_model_runner.cpu_token_to_kv_pool.get_kv_buffer(0)[:, request_kvcache_cpu_slice, ...]
                    self.draft_gpu_token_to_kv_pool_allocator._kvcache.load_kv_cache_from_cpu(layer_id = 0, indices = gpu_loc, src = cpu_kv_cache, non_blocking=False)
            for i, schedule_micro_batch in enumerate(schedule_micro_batches):
                schedule_micro_batch.policy.draft_gpu_req_num = len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batch.decode_schedule_batch_id])
                assert schedule_micro_batch.policy.draft_gpu_req_num >= schedule_micro_batch.policy.gpu_attention_micro_batch_size, "should not happen"
        else:
            # 判断是否应该丢弃一些request
            cache_req_cnt = len(self.draft_gpu_cached_rids)

            total_accept_tokens = 0
            for i in range(len(schedule_micro_batches)):
                cached_cnt_one_micro_batch = len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id])
                total_accept_tokens += sum(schedule_micro_batches[i].spec_info.accept_length_cpu[:cached_cnt_one_micro_batch]) + cached_cnt_one_micro_batch # (spec_info.accept_length+1，要算上bonos token)

            print(f"available_size: {self.draft_gpu_token_to_kv_pool_allocator.available_size()}, cache_req_cnt: {cache_req_cnt}, total_accept_tokens: {total_accept_tokens}, need_size: {cache_req_cnt * self.server_args.max_speculative_draft_tokens + total_accept_tokens}")
            if self.draft_gpu_token_to_kv_pool_allocator.available_size() < cache_req_cnt * self.server_args.max_speculative_draft_tokens + total_accept_tokens:
                # 丢弃一些request
                drop_finished = False
                while not drop_finished:
                    iteration_range = [i for i in range(len(schedule_micro_batches))]
                    have_one_micro_batch_can_drop = False
                    random.shuffle(iteration_range)
                    for i in iteration_range:
                        if drop_finished: break
                        # 不能drop gpu attn req
                        if len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id]) <= len(schedule_micro_batches[i].gpu_seq_lens):
                            continue
                        if len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id]) == 0:
                            continue
                        have_one_micro_batch_can_drop = True
                        # schedule_micro_batches[i].policy.draft_gpu_req_num -= 1 # 更新 micro-batch's draft_gpu_req_num
                        choosed_req = self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id].pop()
                        choosed_req_len = schedule_micro_batches[i].get_seq_len_by_rid(choosed_req)
                        self.draft_gpu_cached_rids.remove(choosed_req) # 从cached_rids中移除
                        req_pool_index = self.draft_gpu_pool_req_indices.pop(choosed_req) # 从pool_req_indices中移除
                        token_slots = self.draft_gpu_req_to_token_pool.req_to_token[req_pool_index][:choosed_req_len] # 获取token slots
                        self.draft_gpu_req_to_token_pool.free(req_pool_index) # 释放req slot
                        self.draft_gpu_token_to_kv_pool_allocator.free(token_slots) # 释放token slot
                        total_accept_tokens -= schedule_micro_batches[i].spec_info.accept_length_cpu[len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id])] + 1
                        if self.draft_gpu_token_to_kv_pool_allocator.available_size() * 0.9 >= len(self.draft_gpu_cached_rids) * self.server_args.max_speculative_draft_tokens + total_accept_tokens:
                            drop_finished = True
                            break
                    if not have_one_micro_batch_can_drop:
                        drop_finished = True
                        assert False, "should not happen"
                for schedule_micro_batch in schedule_micro_batches:
                    schedule_micro_batch.policy.draft_gpu_req_num = len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batch.decode_schedule_batch_id])
                    assert schedule_micro_batch.policy.draft_gpu_req_num >= schedule_micro_batch.policy.gpu_attention_micro_batch_size, "should not happen"
            else:
                # 判断是否应该添加一些request（标准要严格，否则会出现抖动，最好是只有req结束会触发添加request的行为）
                if self.have_new_finish_reqs:
                    fill_finished = False
                    mb_req_ptr = [len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id]) for i in range(len(schedule_micro_batches))]
                    total_req_num = sum([len(schedule_micro_batches[i].reqs) for i in range(len(schedule_micro_batches))])
                    while not fill_finished:
                        if len(self.draft_gpu_cached_rids) == total_req_num:
                            fill_finished = True
                            break
                        iteration_ranges = [i for i in range(len(schedule_micro_batches))]
                        random.shuffle(iteration_ranges)
                        # 如果某个micro batch的gpu attn req还没有放到GPU，优先处理它
                        # 没有完全解决“每个microbatch的gpu attn req需要全都在GPU”这个事情
                        for i in range(len(schedule_micro_batches)):
                            if schedule_micro_batches[i].policy.gpu_attention_micro_batch_size > len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id]):
                                iteration_ranges = [i]
                                break
                        for i in iteration_ranges:
                            if fill_finished: break
                            if len(schedule_micro_batches[i].seq_lens) == 0:
                                continue
                            if len(schedule_micro_batches[i].reqs) <= mb_req_ptr[i]:
                                continue
                            choosed_req = schedule_micro_batches[i].reqs[mb_req_ptr[i]]
                            choosed_seq_len = schedule_micro_batches[i].seq_lens[mb_req_ptr[i]]
                            mb_req_ptr[i] += 1
                            cached_cnt_one_micro_batch = len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id])
                            if self.draft_gpu_token_to_kv_pool_allocator.available_size() * 0.9 < choosed_seq_len + (len(self.draft_gpu_cached_rids) + 1) * self.server_args.max_speculative_draft_tokens + total_accept_tokens + schedule_micro_batches[i].spec_info.accept_length_cpu[cached_cnt_one_micro_batch] + 1: # 留下一点空间
                                fill_finished = True
                                break
                            total_accept_tokens += schedule_micro_batches[i].spec_info.accept_length_cpu[cached_cnt_one_micro_batch] + 1
                            req_pool_index = self.draft_gpu_req_to_token_pool.alloc(1)[0] # 分配req slot
                            gpu_loc = self.draft_gpu_token_to_kv_pool_allocator.alloc(choosed_seq_len).to(dtype=torch.int32) # 分配kv slot
                            self.draft_gpu_cached_rids.append(choosed_req.rid) # 更新cached_rids
                            self.draft_gpu_cached_rids_every_mb[schedule_micro_batches[i].decode_schedule_batch_id].append(choosed_req.rid)
                            self.draft_gpu_pool_req_indices.update({choosed_req.rid: req_pool_index}) # 更新pool_req_indices

                            # 更新gpu_req_to_token_pool，写入prefix token的地址
                            self.draft_gpu_req_to_token_pool.write((req_pool_index, slice(0, choosed_seq_len)), gpu_loc) 

                            # 传输kv cache
                            req_start_loc = self.cpu_token_to_kv_pool_allocator.get_start_loc([choosed_req.rid])[0]
                            request_kvcache_cpu_slice = slice(req_start_loc, req_start_loc + choosed_seq_len)
                            cpu_kv_cache = self.draft_model_runner.cpu_token_to_kv_pool.get_kv_buffer(0)[:, request_kvcache_cpu_slice, ...]
                            self.draft_gpu_token_to_kv_pool_allocator._kvcache.load_kv_cache_from_cpu(layer_id = 0, indices = gpu_loc, src = cpu_kv_cache, non_blocking=False)
                    for i, schedule_micro_batch in enumerate(schedule_micro_batches):
                        schedule_micro_batch.policy.draft_gpu_req_num = len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batch.decode_schedule_batch_id])
        self.have_new_finish_reqs = False

    def draft(self, schedule_micro_batches: List[ScheduleBatch]) -> Tuple[List[EagleVerifyInput], float]:
        draft_begin_time = time.time()
        logger.debug("draft phase starts")

        self.preprocess_draft_kv_cache(schedule_micro_batches)

        # 准备GPU和CPU的micro batches
        gpu_micro_batches = []
        cpu_micro_batches = []
        
        for i, schedule_batch in enumerate(schedule_micro_batches):
            # 计算GPU执行的request数量
            schedule_batch.policy.draft_gpu_req_num = len(self.draft_gpu_cached_rids_every_mb[schedule_batch.decode_schedule_batch_id])
            gpu_req_num = schedule_batch.policy.draft_gpu_req_num
            assert gpu_req_num >= schedule_batch.policy.gpu_attention_micro_batch_size, f"should not happen, gpu_req_num: {gpu_req_num}, schedule_batch.policy.gpu_attention_micro_batch_size: {schedule_batch.policy.gpu_attention_micro_batch_size}"
            print(f"batch {i}, gpu_req_num: {gpu_req_num}")
            
            # 分裂batch
            # gpu_batch, cpu_batch = schedule_batch.split_for_draft(gpu_req_num)
            gpu_batch, cpu_batch = split_schedule_batch_for_draft(schedule_batch, gpu_req_num)
            
            gpu_micro_batches.append(gpu_batch)
            cpu_micro_batches.append(cpu_batch)
        
        # 启动两个线程分别处理GPU和CPU执行
        gpu_future = None
        cpu_future = None
        
        if len(gpu_micro_batches) > 0:
            gpu_future = self.draft_executor.submit(self.draft_thread_func, gpu_micro_batches, gpu_execution=True)
        
        if len(cpu_micro_batches) > 0:
            cpu_future = self.draft_executor.submit(self.draft_thread_func, cpu_micro_batches, gpu_execution=False)
        
        # 等待线程完成并收集结果
        gpu_results = []
        cpu_results = []
        gpu_draft_time, cpu_draft_time = 0, 0
        
        if gpu_future is not None:
            gpu_results, gpu_draft_time = gpu_future.result()
        
        if cpu_future is not None:
            cpu_results, cpu_draft_time = cpu_future.result()
 
        for i, schedule_batch in enumerate(schedule_micro_batches):
            schedule_batch.spec_info.first_step_after_prefill = False
            gpu_batch, cpu_batch = gpu_micro_batches[i], cpu_micro_batches[i]
            if gpu_batch is None:
                schedule_batch.seq_lens = cpu_batch.seq_lens
                schedule_batch.seq_lens_sum = cpu_batch.seq_lens_sum
                schedule_batch.gpu_seq_lens = torch.tensor([], device=schedule_batch.seq_lens.device)
                schedule_batch.cpu_seq_lens = schedule_batch.seq_lens
            elif cpu_batch is None:
                # assert schedule_batch.policy.draft_gpu_execution_ratio == 1
                schedule_batch.seq_lens = gpu_batch.seq_lens
                schedule_batch.seq_lens_sum = gpu_batch.seq_lens_sum
                schedule_batch.gpu_seq_lens = gpu_batch.gpu_seq_lens
                schedule_batch.cpu_seq_lens = gpu_batch.cpu_seq_lens
            else:
                schedule_batch.seq_lens = torch.cat([gpu_batch.seq_lens, cpu_batch.seq_lens.to('cuda')])
                schedule_batch.seq_lens_sum = gpu_batch.seq_lens_sum + cpu_batch.seq_lens_sum
                schedule_batch.gpu_seq_lens = gpu_batch.gpu_seq_lens
                schedule_batch.cpu_seq_lens = torch.cat([gpu_batch.cpu_seq_lens, cpu_batch.cpu_seq_lens.to('cuda')])
                schedule_batch.is_req_on_gpu_tensor = torch.tensor([True] * schedule_batch.policy.gpu_attention_micro_batch_size + [False] * (len(schedule_batch.reqs) - schedule_batch.policy.gpu_attention_micro_batch_size), dtype=torch.bool)
            schedule_batch.gpu_seq_lens_sum = gpu_batch.gpu_seq_lens_sum if gpu_batch is not None else 0
            schedule_batch.cpu_seq_lens_sum = (gpu_batch.cpu_seq_lens_sum if gpu_batch is not None else 0) + (cpu_batch.cpu_seq_lens_sum if cpu_batch is not None else 0)

        # 合并结果
        merged_results = []
        
        for i, schedule_batch in enumerate(schedule_micro_batches):
            if gpu_micro_batches[i] == None and cpu_micro_batches[i] == None:
                assert False, "这种情况应该这个schedule batch早会被删除"
            if gpu_micro_batches[i] is None:
                merged_results.append(cpu_results[i])
            elif cpu_micro_batches[i] is None:
                merged_results.append(gpu_results[i])
            else:
                merged_results.append(EagleVerifyInput.merge(gpu_results[i], cpu_results[i]))
        
        draft_time = time.time() - draft_begin_time
        logger.debug("draft phase ends")
        return merged_results, (draft_time, gpu_draft_time, cpu_draft_time)

    def draft_token_generate_forward(self, forward_batch: ForwardBatch, process_cpu_requets: bool = True, gpu_execution: bool = True):
        # Parse args
        spec_info = forward_batch.spec_info
        gpu_out_cache_loc = forward_batch.gpu_out_cache_loc
        cpu_out_cache_loc = forward_batch.cpu_out_cache_loc
        if process_cpu_requets:
            topk_p, topk_index, hidden_states = (
                spec_info.topk_p,
                spec_info.topk_index,
                spec_info.hidden_states, # (bs, hidden_dimension)，这是上一个verify阶段生成的token的hidden
            )
        else:
            topk_p, topk_index, hidden_states = (
                spec_info.topk_p_gpu,
                spec_info.topk_index_gpu,
                spec_info.hidden_states_gpu, # (bs, hidden_dimension)，这是上一个verify阶段生成的token的hidden
            )
        # logger.debug(f"spec_info.hidden_states: {spec_info.hidden_states.shape}")

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps if process_cpu_requets else self.gpu_speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk if process_cpu_requets else self.gpu_topk
            )
            score_list.append(tree_info[0].to(device='cuda'))
            token_list.append(tree_info[1].to(device='cuda'))
            parents_list.append(tree_info[2].to(device='cuda'))

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == (self.speculative_num_steps if process_cpu_requets else self.gpu_speculative_num_steps) - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            batch_size = len(forward_batch.cpu_seq_lens) if process_cpu_requets else len(forward_batch.gpu_seq_lens)
            if gpu_execution:
                gpu_out_cache_loc = gpu_out_cache_loc.view(batch_size, -1)
                if process_cpu_requets:
                    forward_batch.gpu_attn_backend = self.draft_gpu_exec_attn_backend.attn_backends[i]
                    forward_batch.gpu_out_cache_loc = gpu_out_cache_loc[
                        :, self.topk * i : self.topk * (i + 1)
                    ].flatten()
                else:
                    forward_batch.gpu_attn_backend = self.draft_gpu_exec_attn_backend_for_gpu_request.attn_backends[i]
                    forward_batch.gpu_out_cache_loc = gpu_out_cache_loc[
                        :, self.gpu_topk * i : self.gpu_topk * (i + 1)
                    ].flatten()
            else:
                cpu_out_cache_loc = cpu_out_cache_loc.view(batch_size, -1)
                if process_cpu_requets:
                    forward_batch.cpu_attn_backend = self.draft_cpu_exec_attn_backend.attn_backends[i]
                    forward_batch.cpu_out_cache_loc = cpu_out_cache_loc[
                        :, self.topk * i : self.topk * (i + 1)
                    ].flatten()
                else:
                    assert False, "GPU Attn Request's Draft must processed in GPU"
            forward_batch.positions.add_(1)
            if process_cpu_requets:
                spec_info.hidden_states = hidden_states
            else:
                spec_info.hidden_states_gpu = hidden_states

            # Run forward
            logits_output: LogitsProcessorOutput = self.draft_model_runner.draft_forward(
                [forward_batch], self.decode_policy, is_draft_generate=True, process_cpu_requets=process_cpu_requets, gpu_execution=gpu_execution
            )[0]
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            if process_cpu_requets:
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            else:
                topk_p, topk_index = fast_topk(probs, self.gpu_topk, dim=-1)
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch, batch_id: int, gpu_execution: bool = True):
        logger.debug("forward_draft_extend_after_decode")
        # Backup fileds that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone() # TODO: 这里的seq_lens应该是包含verified tokens的，需要检查一下
        accept_length_backup = batch.spec_info.accept_length.clone()

        # 在分裂后的batch中，需要根据batch的实际结构来处理
        gpu_req_num = len(batch.gpu_seq_lens) if hasattr(batch, 'gpu_seq_lens') and batch.gpu_seq_lens is not None else 0
        cpu_req_num = len(batch.cpu_seq_lens) if hasattr(batch, 'cpu_seq_lens') and batch.cpu_seq_lens is not None else 0
        total_req_num = len(batch.reqs)

        # Prepare GPU slots
        if gpu_execution and gpu_req_num > 0:
            assert gpu_req_num == batch.policy.draft_gpu_req_num
            # 对于GPU执行，只处理GPU部分的requests
            gpu_accept_length = batch.spec_info.accept_length
            gpu_out_cache_loc = batch.gpu_alloc_token_slots(
                (gpu_accept_length + 1).sum(), backup_state=False, specified_allocator=self.draft_gpu_token_to_kv_pool_allocator
            ).to('cuda')
            origin_draft_gpu_token_to_kv_pool_state_before_tokengenerate = self.draft_gpu_token_to_kv_pool_allocator.backup_state()
            # 同步更新shadow kv cache
            # self.shadow_gpu_token_to_kv_pool[batch_id].set_free_slots_force(batch.gpu_token_to_kv_pool_allocator.free_slots.clone())
            batch.gpu_out_cache_loc = gpu_out_cache_loc
            # pt = 0
            # 以下这部分for循环非常耗时，需要优化
            # 更新gpu_req_to_token_pool，将新分配的verified token slot写入对应的req中
            # for i, r in enumerate(batch.reqs[:gpu_req_num]):
            #     self.gpu_req_to_token_pool.write( 
            #         (batch.gpu_req_pool_indices[i], slice(batch.seq_lens[i] - batch.spec_info.accept_length[i] - 1, batch.seq_lens[i])),
            #         gpu_out_cache_loc[pt : pt + batch.spec_info.accept_length[i] + 1],
            #     )
            #     # 同步更新shadow kv cache
            #     self.shadow_gpu_req_to_token_pool[batch_id].write(
            #         (self.shadow_pool_req_indices[batch_id][r.rid], slice(batch.seq_lens[i] - batch.spec_info.accept_length[i] - 1, batch.seq_lens[i])),
            #         gpu_out_cache_loc[pt : pt + batch.spec_info.accept_length[i] + 1],
            #     )
            #     pt += batch.spec_info.accept_length[i] + 1 
            
            write_req_to_token_pool_triton[(gpu_req_num,)](
                req_to_token_ptr=self.draft_gpu_req_to_token_pool.req_to_token,
                req_pool_indices=batch.gpu_req_pool_indices,
                pre_lens=batch.seq_lens - batch.spec_info.accept_length - 1,
                seq_lens=batch.seq_lens,
                extend_lens=batch.spec_info.accept_length + 1,
                out_cache_loc=gpu_out_cache_loc,
                req_to_token_ptr_stride=self.draft_gpu_req_to_token_pool.req_to_token.shape[1],
            )
            # # 同步更新shadow kv cache
            # shadow_req_indices_ = torch.tensor([self.shadow_pool_req_indices[batch_id][r.rid] for r in batch.reqs[:gpu_req_num]], device='cuda')
            # write_req_to_token_pool_triton[(gpu_req_num,)](
            #     req_to_token_ptr=self.shadow_gpu_req_to_token_pool[batch_id].req_to_token,
            #     req_pool_indices=shadow_req_indices_,
            #     pre_lens=batch.seq_lens - batch.spec_info.accept_length - 1,
            #     seq_lens=batch.seq_lens,
            #     extend_lens=batch.spec_info.accept_length + 1,
            #     out_cache_loc=gpu_out_cache_loc,
            #     req_to_token_ptr_stride=self.shadow_gpu_req_to_token_pool[batch_id].req_to_token.shape[1],
            # )
            
            # CPU部分的out_cache_loc
            if cpu_req_num > 0:
                assert False, "???"
                batch.cpu_out_cache_loc = batch.out_cache_loc[:gpu_out_cache_loc.shape[0]]
        else:
            # CPU执行或没有GPU requests
            if cpu_req_num > 0:
                # 对于CPU执行，处理CPU部分的requests
                batch.cpu_out_cache_loc = batch.out_cache_loc[batch.out_cache_loc_begin_id_draft_extend_after_verify:]

        # Prepare metadata
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(
            batch,
            max(1 if self.gpu_speculative_num_steps is None else self.gpu_speculative_num_steps, self.speculative_num_steps),
            self.server_args.draft_model_placement,
            gpu_execution,
        ) # XXX: 在这里已经将accept_length加1了

        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        # XXX
        forward_batch.gpu_req_to_token_pool = self.draft_gpu_req_to_token_pool
        forward_batch.gpu_token_to_kv_pool = self.draft_gpu_token_to_kv_pool_allocator._kvcache

        # Run
        logits_outputs = self.draft_model_runner.draft_forward([forward_batch], forward_batch.policy, gpu_execution=gpu_execution)
        assert len(logits_outputs) == 1 # 这里假设extend after decode是一个batch一个batch进行的
        logits_output = logits_outputs[0]

        self._detect_nan_if_needed(logits_output)
        if gpu_execution:
            self.capture_for_decode(logits_output, forward_batch.spec_info, stage="decode", gpu_attn_req_num=batch.policy.gpu_attention_micro_batch_size)
        else:
            self.capture_for_decode(logits_output, forward_batch.spec_info, stage="decode", gpu_attn_req_num=0)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup
        if gpu_req_num > 0:
            batch.gpu_seq_lens = batch.seq_lens[:gpu_req_num]
        if cpu_req_num > 0:
            batch.cpu_seq_lens = batch.seq_lens[gpu_req_num:gpu_req_num + cpu_req_num]
        
        batch.gpu_seq_lens_sum = batch.gpu_seq_lens.sum() if len(batch.gpu_seq_lens) > 0 else 0
        batch.cpu_seq_lens_sum = batch.cpu_seq_lens.sum() if len(batch.cpu_seq_lens) > 0 else 0
        
        batch.spec_info.accept_length = accept_length_backup

        if gpu_execution:
            # 需要将这些verified tokens kv cache卸载到CPU上
            src = self.draft_model_runner.gpu_token_to_kv_pool.get_layer_kv_cache(0, batch.gpu_out_cache_loc)
            offload_lens = (batch.spec_info.accept_length + 1).tolist()
            rids = [r.rid for r in batch.reqs]
            dst_indices = self.cpu_token_to_kv_pool_allocator.get_tail_offload_indices(rids, offload_lens)
            self.cpu_token_to_kv_pool_allocator.offload_kv_cache_new_verified(0, src, dst_indices, offload_lens, kv_cache=self.draft_model_runner.cpu_token_to_kv_pool.kv_buffer)
            
            return origin_draft_gpu_token_to_kv_pool_state_before_tokengenerate
        else:
            return None
    
    def process_finished_gpudraft_req_rids(self, finished_gpudraft_req_rids: List[Tuple[int, int]], schedule_micro_batch: ScheduleBatch):
        if len(finished_gpudraft_req_rids) == 0:
            return
        for rid, seq_len in finished_gpudraft_req_rids:
            req_slot_index = self.draft_gpu_pool_req_indices[rid]
            self.draft_gpu_pool_req_indices.pop(rid)
            self.draft_gpu_cached_rids.remove(rid)
            self.draft_gpu_cached_rids_every_mb[schedule_micro_batch.decode_schedule_batch_id].remove(rid)
            token_slots = self.draft_gpu_req_to_token_pool.req_to_token[req_slot_index][:seq_len]
            self.draft_gpu_req_to_token_pool.free(req_slot_index)
            self.draft_gpu_token_to_kv_pool_allocator.free(token_slots)
        self.have_new_finish_reqs = True
        schedule_micro_batch.policy.draft_gpu_req_num = len(self.draft_gpu_cached_rids_every_mb[schedule_micro_batch.decode_schedule_batch_id])

    def verify(self, schedule_micro_batches: List[ScheduleBatch], spec_infos: List[EagleVerifyInput]) -> Tuple[List[LogitsProcessorOutput], List[EagleVerifyOutput], List[ModelWorkerBatch]]:
        verify_begin_time = time.time()
        verify_time = [0, 0]
        logger.debug("verify phase starts")
        model_worker_micro_batches:List[ModelWorkerBatch] = []

        for i, schedule_batch in enumerate(schedule_micro_batches):
            spec_info = spec_infos[i]
            spec_info.prepare_for_verify(schedule_batch, self.decode_policy)
            schedule_batch.forward_mode = ForwardMode.TARGET_VERIFY
            schedule_batch.spec_info = spec_info
            model_worker_batch = schedule_batch.get_model_worker_batch()
            model_worker_micro_batches.append(model_worker_batch)

        logits_outputs, _, target_model_performance_recoder = self.target_worker.forward_batch_generation(
            model_worker_micro_batches, policy=self.decode_policy, skip_sample=True
        )

        # logger.debug(f"hidden_states in verify: {logits_outputs[0].hidden_states[:, :3]}, logits: {logits_outputs[0].next_token_logits[:, :3]}")

        verify_time[0] = time.time() - verify_begin_time # target model running time
        verify_begin_time = time.time()
        for i, logits_output in enumerate(logits_outputs):
            if i == 0: continue
            if logits_output.next_token_logits_cpu is None:
                logits_output.next_token_logits_cpu = torch.empty_like(logits_output.next_token_logits, device='cpu', pin_memory=True)
            logits_output.next_token_logits_cpu.copy_(logits_output.next_token_logits, non_blocking=False)
            logits_output.next_token_logits = None
        ret_verify_outputs = []
        for i, logits_output in enumerate(logits_outputs):
            spec_infos[i].hidden_states = logits_output.hidden_states
            if i !=0:
                logits_output.next_token_logits = logits_output.next_token_logits_cpu.to('cuda')
                logits_output.next_token_logits_cpu = None
            self._detect_nan_if_needed(logits_output)
            # TODO: may need split to cpu request and gpu request verify
            res, finished_gpudraft_req_rids = spec_infos[i].verify(
                schedule_micro_batches[i],
                logits_outputs[i],
                self.cpu_token_to_kv_pool_allocator,
                experiment_ablation_drop_accepted_tokens_ratio=self.server_args.experiment_ablation_drop_accepted_tokens_ratio,
            )
            self.process_finished_gpudraft_req_rids(finished_gpudraft_req_rids, schedule_micro_batches[i])
            ret_verify_outputs.append(res)
            # Post process based on verified outputs.
            # Pick indices that we care (accepeted)
            logits_output.next_token_logits = logits_output.next_token_logits[
                res.accepeted_indices
            ]
            logits_output.hidden_states = logits_output.hidden_states[res.accepeted_indices]

            # Prepare the batch for the next draft forwards.
            schedule_micro_batches[i].forward_mode = ForwardMode.DECODE
            schedule_micro_batches[i].spec_info = res.draft_input

        logger.debug("verify phase ends")
        verify_time[1] = time.time() - verify_begin_time # verify time
        return logits_outputs, ret_verify_outputs, model_worker_micro_batches, verify_time, target_model_performance_recoder

    def forward_target_extend(
        self, schedule_micro_batches: List[ScheduleBatch]
    ) -> Tuple[List[LogitsProcessorOutput], List[List[int]], List[int]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
                List of logitprocessoroutput.
                Len: micro_batch_cnt
            next_token_ids: Next token ids generated. 
                List of tensor.
                Len: micro_batch_cnt
                    shape: (micro_batch_size, )
            bid: The model batch ID. Used for overlap schedule.
                List of bids.
                Len: micro_batch_cnt
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.

        model_worker_micro_batches:List[ModelWorkerBatch] = []
        bids = []

        for schedule_batch in schedule_micro_batches:
            model_worker_micro_batches.append(schedule_batch.get_model_worker_batch())
            model_worker_micro_batches[-1].capture_hidden_mode = CaptureHiddenMode.FULL
            bids.append(model_worker_micro_batches[-1].bid)

        logits_outputs, next_token_ids, _ = self.target_worker.forward_batch_generation(
            model_worker_micro_batches, 
            policy=self.prefill_policy
        )

        # logger.debug(f"logits_outputs type: {type(logits_outputs)}, len: {len(logits_outputs)}, {logits_outputs}")
        # logger.debug(f"next_token_ids type: {type(next_token_ids)}, shape: {next_token_ids[0].shape}, len: {len(next_token_ids)}, {next_token_ids}")
        # logger.debug(f"bids: {bids}")

        return logits_outputs, next_token_ids, bids

    def forward_draft_extend(
        self,
        schedule_micro_batches: List[ScheduleBatch],
        logits_outputs: List[LogitsProcessorOutput],
        next_token_ids: List[List[int]],
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        model_worker_micro_batches = []
        for i, schedule_batch in enumerate(schedule_micro_batches):
            schedule_batch.spec_info = EagleDraftInput(
                hidden_states=logits_outputs[i].hidden_states_cpu,
                verified_id=next_token_ids[i],
                first_step_after_prefill=True,
            )
            schedule_batch.spec_info.prepare_for_extend(schedule_batch)
            schedule_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
            model_worker_micro_batches.append(schedule_batch.get_model_worker_batch())

        forward_micro_batches: List[ForwardBatch] = []
        for model_worker_micro_batch in model_worker_micro_batches:
            forward_micro_batches.append(ForwardBatch.init_new(model_worker_micro_batch, self.draft_model_runner))
        logits_outputs = self.draft_model_runner.draft_forward(forward_micro_batches, self.prefill_policy)

        assert isinstance(schedule_micro_batches[0].spec_info, EagleDraftInput)
        assert schedule_micro_batches[0].spec_info is schedule_micro_batches[0].spec_info

        for logits_output, forward_batch in zip(logits_outputs, forward_micro_batches):
            self._detect_nan_if_needed(logits_output)
            self.capture_for_decode(logits_output, forward_batch.spec_info, stage="prefill")

        # spec_info_store: Dict[int, EagleDraftInput] = {} # 为每个request保存一个EagleDraftInput，因为extend结束后schedule_batch就会销毁，需要在decode开始前重新对构造出的schedule_batch的spec_info进行初始化
        for i, schedule_batch in enumerate(schedule_micro_batches):
            spec_info = schedule_batch.spec_info
            for j, rq in enumerate(schedule_batch.reqs):
                # spec_info_store[rq.rid] = spec_info.split_to_one_req(j)
                # rq.spec_info = spec_info_store[rq.rid]
                rq.spec_info = spec_info.split_to_one_req(j)

        # return spec_info_store

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput, stage: str = "decode", gpu_attn_req_num: int = 0
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        if stage == "decode":
            if gpu_attn_req_num != 0:
                draft_input.topk_p_gpu, draft_input.topk_index_gpu = fast_topk(probs[:gpu_attn_req_num, ...], self.gpu_topk, dim=-1)
                draft_input.hidden_states_gpu = logits_output.hidden_states[:gpu_attn_req_num, ...]
            draft_input.topk_p, draft_input.topk_index = fast_topk(probs[gpu_attn_req_num:, ...], self.topk, dim=-1)
            draft_input.hidden_states = logits_output.hidden_states[gpu_attn_req_num:, ...]
        elif stage == "prefill":
            draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.server_args.decode_gpu_attention_ratio != 0:
                draft_input.topk_p_gpu, draft_input.topk_index_gpu = fast_topk(probs, self.gpu_topk, dim=-1)
            draft_input.hidden_states = logits_output.hidden_states
        else: 
            assert False, "Should never reach here"

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")
