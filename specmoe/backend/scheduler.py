from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import psutil
import signal
import os
import setproctitle
import faulthandler
import logging
import sys
import traceback
import zmq
from collections import deque
import torch
import torch.profiler
from time import sleep
import time
from math import ceil
import pathlib
from contextlib import nullcontext

from sglang.srt.utils import get_bool_env_var, set_gpu_proc_affinity, kill_itself_when_parent_died, suppress_other_loggers, broadcast_pyobj, DynamicGradMode, get_zmq_socket
from sglang.srt.hf_transformers_utils import get_tokenizer

from specmoe.utils.utils import configure_logger
from specmoe.utils.server_args import ServerArgs, PortArgs
from specmoe.backend.tp_worker import TpModelWorker
from specmoe.backend.eagle_worker import EAGLEWorker
from specmoe.speculative.spec_info import SpeculativeAlgorithm
from specmoe.utils.data_info import Req, SamplingBatchInfo, ScheduleBatch, ModelWorkerBatch
from specmoe.backend.policy import Policy
from specmoe.speculative.speculative_utils import EagleDraftInput 
from specmoe.optimizer.solver import Solver

logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, tokenizer_path: str, server_args: ServerArgs):
        self.tokenizer = get_tokenizer(
            tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code
        )
        self.server_args = server_args
        self.eos_token_id = None
        self.truncate_input_length = server_args.truncate_input_length

    def encode(self, prompt: str) -> List[int]:
        # 1. use template
        chat_input = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(chat_input, tokenize=False)
        tmp = self.tokenizer(prompt)
        input_ids = tmp['input_ids']
        if self.truncate_input_length > 0:
            input_ids = input_ids[:self.truncate_input_length-4]
            input_ids.extend([733, 28748, 16289, 28793])
            # print(f"truncate input_ids to {len(input_ids)}")
        # 2. no template
        # tmp = self.tokenizer(prompt)
        # input_ids = tmp['input_ids']
        return input_ids

    def decode(self, ids: List[torch.Tensor]) -> List[str]:
        res = []
        for id_ in ids:
            res.append(self.tokenizer.decode(id_))
        return res

@dataclass
class GenerationBatchResult:
    # logits_output: LogitsProcessorOutput
    next_token_ids: List[int]
    # extend_input_len_per_req: List[int]
    # extend_logprob_start_len_per_req: List[int]
    bid: int

class Scheduler:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        solver: Solver = None
    ):
        logger.debug("{scheduler init}")
        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.solver = solver

        context = zmq.Context(2)
        if self.tp_rank == 0:
            self.recv_from_engine = get_zmq_socket(context, zmq.PULL, self.port_args.scheduler_input_ipc_name, False)
            self.send_to_engine = get_zmq_socket(context, zmq.PUSH, self.port_args.engine_ipc_name, False)
        else:
            self.recv_from_engine = None

        self.tokenizer = Tokenizer(server_args.tokenizer_path, server_args)

        self.tp_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=self.port_args.nccl_port,
            offload=server_args.offload,
        )

        self.prefill_policy = Policy.get_prefill_policy(self.server_args, self.tp_worker.model_config)
        self.decode_policy = Policy.get_decode_policy(self.server_args, self.tp_worker.model_config)

        self.tp_worker.model_runner.init_execution_engine(self.prefill_policy)

        self.tokenizer.eos_token_id = self.tp_worker.model_config.eos_token_id
        self.model_config = self.tp_worker.model_config

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            self.draft_worker = EAGLEWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                nccl_port=self.port_args.nccl_port,
                target_worker=self.tp_worker,
                prefill_policy=self.prefill_policy,
                decode_policy=self.decode_policy,
            )
            self.draft_worker.model_runner.init_execution_engine(self.prefill_policy, self.tp_worker.model_runner.execution_engine)
            # self.draft_worker.init_attention_backend()
        else:
            self.draft_worker = None
        self.init_memory_pool_and_cache()
        self.init_metrics()

    def init_memory_pool_and_cache(self):
        server_args = self.server_args

        self.gpu_req_to_token_pool, self.gpu_token_to_kv_pool_allocator, self.cpu_req_to_token_pool, self.cpu_token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        self.decode_mem_cache_buf_multiplier = (
            1
            if self.spec_algorithm.is_none()
            else (
                server_args.speculative_num_draft_tokens
                + (
                    server_args.speculative_eagle_topk
                    * server_args.speculative_num_steps
                )
            )
        )
    
    def change_plan(self, new_global_batchsize: int, now_generation_length: int, running_micro_batch_num: int = 0, log: bool = True):
        # logger.debug("change plan")
        change, plan_dict = self.solver.change_plan(new_global_batchsize=new_global_batchsize, now_generation_length=now_generation_length, running_micro_batch_num=running_micro_batch_num, log=log)
        if change:
            assert self.draft_worker is not None, "draft_worker is not initialized"
            self.server_args.speculative_eagle_topk = plan_dict['speculative_eagle_topk']
            self.server_args.speculative_num_steps = plan_dict['speculative_num_steps']
            self.server_args.speculative_num_draft_tokens = plan_dict['speculative_num_draft_tokens']
            self.server_args.speculative_eagle_topk_gpu = plan_dict['speculative_eagle_topk_gpu']
            self.server_args.speculative_num_steps_gpu = plan_dict['speculative_num_steps_gpu']
            self.server_args.speculative_num_draft_tokens_gpu = plan_dict['speculative_num_draft_tokens_gpu']
            self.draft_worker.topk = plan_dict['speculative_eagle_topk']
            self.draft_worker.speculative_num_steps = plan_dict['speculative_num_steps']
            self.draft_worker.gpu_topk = plan_dict['speculative_eagle_topk_gpu']
            self.draft_worker.gpu_speculative_num_steps = plan_dict['speculative_num_steps_gpu']
            self.draft_worker.init_attention_backend()
            self.tp_worker.model_runner.execution_engine.context.change_pin_size(self.decode_policy, self.server_args)
            logger.info(f"change plan, now_global_batchsize: {new_global_batchsize}, now_generation_length: {now_generation_length}, topk: {self.server_args.speculative_eagle_topk}, num_steps: {self.server_args.speculative_num_steps}, num_draft_tokens: {self.server_args.speculative_num_draft_tokens}, gpu_topk: {self.server_args.speculative_eagle_topk_gpu}, gpu_num_steps: {self.server_args.speculative_num_steps_gpu}, gpu_num_draft_tokens: {self.server_args.speculative_num_draft_tokens_gpu}")
    
    
    def switch_to_another_policy(self, plan_dict: Dict, solver: Solver):
        global_batch_size = plan_dict['global_batch_size']
        prefill_micro_batch_size = plan_dict['prefill_micro_batch_size']
        prefill_micro_batch_num = plan_dict['prefill_micro_batch_num']
        prefill_weight_cache_ratio = plan_dict['prefill_weight_cache_ratio']
        decode_micro_batch_num = plan_dict['decode_micro_batch_num']
        decode_micro_batch_size = plan_dict['decode_micro_batch_size']
        decode_weight_cache_ratio = plan_dict['decode_weight_cache_ratio']
        decode_gpu_attention_ratio = plan_dict['decode_gpu_attention_ratio']
        draft_gpu_execution_ratio = plan_dict['draft_gpu_execution_ratio']
        speculative_num_steps = plan_dict['speculative_num_steps']
        speculative_eagle_topk = plan_dict['speculative_eagle_topk']
        speculative_num_draft_tokens = plan_dict['speculative_num_draft_tokens']
        speculative_num_steps_gpu = plan_dict['speculative_num_steps_gpu']
        speculative_eagle_topk_gpu = plan_dict['speculative_eagle_topk_gpu']
        speculative_num_draft_tokens_gpu = plan_dict['speculative_num_draft_tokens_gpu']
        decode_spec_policy = plan_dict['decode_spec_policy']
        real_target_kv_cache_size = plan_dict['real_target_kv_cache_size']

        self.server_args.global_batch_size = global_batch_size
        self.server_args.prefill_micro_batch_size = prefill_micro_batch_size
        self.server_args.prefill_micro_batch_num = prefill_micro_batch_num
        self.server_args.prefill_weight_cache_ratio = prefill_weight_cache_ratio
        self.server_args.decode_micro_batch_num = decode_micro_batch_num
        self.server_args.decode_micro_batch_size = decode_micro_batch_size
        self.server_args.decode_weight_cache_ratio = decode_weight_cache_ratio
        self.server_args.decode_gpu_attention_ratio = decode_gpu_attention_ratio
        self.server_args.draft_gpu_execution_ratio = draft_gpu_execution_ratio
        self.server_args.speculative_num_steps = speculative_num_steps
        self.server_args.speculative_eagle_topk = speculative_eagle_topk
        self.server_args.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.server_args.speculative_num_steps_gpu = speculative_num_steps_gpu
        self.server_args.speculative_eagle_topk_gpu = speculative_eagle_topk_gpu
        self.server_args.speculative_num_draft_tokens_gpu = speculative_num_draft_tokens_gpu
        self.server_args.decode_spec_policy = decode_spec_policy

        self.solver = solver
        
        self.prefill_policy = Policy.get_prefill_policy(self.server_args, self.tp_worker.model_config)
        self.decode_policy = Policy.get_decode_policy(self.server_args, self.tp_worker.model_config)
    

    def init_metrics(self):
        # The largest prefill length of a single request
        self._largest_prefill_len: int = 0
        # The largest context length (prefill + generation) of a single request
        self._largest_prefill_decode_len: int = 0
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.decode_times = []
        self.history_accept_rate = []
        self.draft_times = []
        self.gpu_draft_times = []
        self.cpu_draft_times = []
        self.target_model_times = []
        self.verify_times = []
        self.avg_generate_length = 0.0
        self.cpu_speculative_num_draft_token_list = []
        self.gpu_speculative_num_draft_token_list = []
        self.cpu_attention_times = []
        self.gpu_post_attention_times = []
        self.htod_transfer_times = []

    
    def log_metrics(self):
        if self.spec_num_total_forward_ct != 0:
            logger.info(f"Speculative num total accepted tokens: {self.spec_num_total_accepted_tokens}")
            logger.info(f"Speculative num total forward ct: {self.spec_num_total_forward_ct}")
            logger.info(f"Speculative average accepted tokens: {self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct:.2f}")
            self.history_accept_rate.append(self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct)
            self.cum_spec_accept_length += self.spec_num_total_accepted_tokens
            self.cum_spec_accept_count += self.spec_num_total_forward_ct
            self.avg_generate_length += self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct
            self.spec_num_total_accepted_tokens = 0
            self.spec_num_total_forward_ct = 0
            self.cpu_speculative_num_draft_token_list.append(self.server_args.speculative_num_draft_tokens)
            self.gpu_speculative_num_draft_token_list.append(self.server_args.speculative_num_draft_tokens_gpu)

    def _run_batch(self, schedule_micro_batches: List[ScheduleBatch], policy: Policy) -> List[GenerationBatchResult]:
        logger.debug(f"_run_batch")
        bids:List[int] = []
        ret: List[GenerationBatchResult] = []
        if self.spec_algorithm.is_none():
            model_worker_micro_batches:List[ModelWorkerBatch] = []

            # 使用 torch.profiler.record_function 标记不同阶段
            for schedule_batch in schedule_micro_batches:
                model_worker_micro_batches.append(schedule_batch.get_model_worker_batch())

            logits_outputs, next_token_ids, performance_recoder = self.tp_worker.forward_batch_generation(
                model_worker_micro_batches,
                policy,
            )
            if performance_recoder is not None:
                self.cpu_attention_times.append(performance_recoder["cpu_attention_time"])
                self.gpu_post_attention_times.append(performance_recoder["post_attention_time"])
                self.htod_transfer_times.append(performance_recoder["htod_transfer_time"])

            for wb in model_worker_micro_batches:
                bids.append(wb.bid)
            for schedule_batch, next_token_id in zip(schedule_micro_batches, next_token_ids):
                schedule_batch.output_ids = next_token_id
                for r, nid in zip(schedule_batch.reqs, next_token_id):
                    r.add_output_ids(nid)
                ret.append(GenerationBatchResult(
                    next_token_ids=next_token_id,
                    bid=wb.bid,
                ))

        else:
            logits_outputs, next_token_ids, bids, num_accepted_tokens, num_total_requets, draft_times, verify_time, target_model_performance_recoder = (
                self.draft_worker.forward_batch_speculative_generation(
                    schedule_micro_batches
                )
            )
            self.spec_num_total_accepted_tokens += num_accepted_tokens + num_total_requets
            self.spec_num_total_forward_ct += num_total_requets
            if draft_times is not None:
                self.draft_times.append(round(draft_times[0], 2))
                self.gpu_draft_times.append(round(draft_times[1], 2))
                self.cpu_draft_times.append(round(draft_times[2], 2))
                self.verify_times.append(round(verify_time[1], 2))
                self.target_model_times.append(round(verify_time[0], 2))
            if target_model_performance_recoder is not None:
                self.cpu_attention_times.append(round(target_model_performance_recoder["cpu_attention_time"], 2))
                self.gpu_post_attention_times.append(round(target_model_performance_recoder["post_attention_time"], 2))
                self.htod_transfer_times.append(round(target_model_performance_recoder["htod_transfer_time"], 2))
            for schedule_batch, logits_output, next_token_id, bid in zip(schedule_micro_batches, logits_outputs, next_token_ids, bids):
                if policy.stage == "prefill":
                    for r, nid in zip(schedule_batch.reqs, next_token_id):
                        r.add_output_ids(nid)
                schedule_batch.output_ids = next_token_id
                ret.append(GenerationBatchResult(
                    next_token_ids=next_token_id,
                    bid=bid,
                ))

        logger.debug(f"ret: {ret}")
        return ret

    def run_batch(self, all_req: List[Req]):
        # 创建profile输出目录
        if self.server_args.enable_profile and self.tp_rank == 0:
            pathlib.Path(self.server_args.profile_output_dir).mkdir(parents=True, exist_ok=True)
            profile_path = os.path.join(self.server_args.profile_output_dir, f"tp{self.tp_rank}")
            logger.info(f"Profiling enabled, output will be saved to {profile_path}")
            profiler_instance = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # record_shapes=True,
                # profile_memory=True,
                # with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path)
            )
            # profiler_instance.start()
        else:
            profiler_instance = None

        # XXX: add batch scheduling here, i.e., split the global batch into multiple batches
        batchsize = len(all_req)
        logger.debug(f"batchsize: {batchsize}")

        ret_ids = [[] for _ in range(len(all_req))]
        aborted_request_rids = []

        # 1. 根据可用CPU内存筛选出一部分Global Batch，这里先假设所有request都放到global batch中。TODO: 细化
        global_batchs: List[List[Req]] = []
        global_batchs.append(all_req)
        for global_batch in global_batchs:
            # 2. 对一个Global Batch求解prefill策略
            #    调整GPU KV cache pool容量
            # self.gpu_token_to_kv_pool_allocator.adjust_capacity(self.prefill_policy.gpu_kv_pool_slot_num)
            #    TODO: 调整GPU expert buffer容量

            # 3. 根据policy构建schedule batch
            #    prepare for forward at the first time
            for r in global_batch:
                r.init_next_round_input()

            schedule_batch_num = ceil(len(global_batch) / (self.prefill_policy.batchsize))

            logger.info(f"schedule_batch_num: {schedule_batch_num}")

            prefill_start_time = time.time()
            #    TODO: 这里构建batch的策略也可以优化
            for i in range(schedule_batch_num):
                logger.info(f"Schedule global batch {i} to do prefill")
                schedule_micro_batches: List[ScheduleBatch] = []

                for j in range(self.prefill_policy.micro_batch_num):
                    now_micro_batch_size = min(
                        self.prefill_policy.micro_batch_size, 
                        len(global_batch) - i * self.prefill_policy.batchsize - j * self.prefill_policy.micro_batch_size,
                    )

                    schedule_batch = ScheduleBatch.init_new(
                        global_batch[
                            i * self.prefill_policy.batchsize + j * self.prefill_policy.micro_batch_size
                            : min(
                                len(global_batch),
                                i * self.prefill_policy.batchsize + (j + 1) * self.prefill_policy.micro_batch_size,
                            )
                        ],
                        [
                            1 for i in range(now_micro_batch_size)
                        ],  # in the prefill stage, all requests' attention computation is on the gpu
                        self.gpu_req_to_token_pool,
                        self.gpu_token_to_kv_pool_allocator,
                        self.cpu_req_to_token_pool,
                        self.cpu_token_to_kv_pool_allocator,
                        self.model_config,
                        self.spec_algorithm,
                        self.prefill_policy,
                    )
                    schedule_batch.prepare_for_extend()
                    schedule_micro_batches.append(schedule_batch)

                # Operate Prefill
                ret = self._run_batch(schedule_micro_batches, self.prefill_policy)

                for mbid, mb in enumerate(schedule_micro_batches):
                    for r, next_token_id in zip(mb.reqs, ret[mbid].next_token_ids):
                        ret_ids[r.rid-1] = r.output_ids
                # TODO: 清理这个batch的GPU上KV cache相关内容（删除req_to_token_pool和kv_pool上相关内容）
                for schedule_batch in schedule_micro_batches:
                    schedule_batch.cleanup_after_prefill()
                # logger.debug(f"GPU kv cache available size: {self.gpu_token_to_kv_pool_allocator.available_size()}")
            
            logger.info(f"Global batch {i} prefill finished")

            if profiler_instance is not None:
                profiler_instance.start()
            decode_start_time = time.time()
            # 这里所有prefill已经做完了
            finished_micro_batches = []
            # 5. 构建decode batch
            decodecnt = 0

            max_gpu_attn_req_total_length_in_a_batch = 1
            max_draft_gpu_req_total_length_in_a_batch = 1

            micro_batches_pool: List[ScheduleBatch] = []
            for j in range(self.decode_policy.micro_batch_num):
                now_micro_batch_size = min(
                    self.decode_policy.micro_batch_size, 
                    len(global_batch) - j * self.decode_policy.micro_batch_size,
                )
                logger.debug(f"now_micro_batch_size: {now_micro_batch_size}")
                schedule_batch = ScheduleBatch.init_new(
                    global_batch[
                        j * self.decode_policy.micro_batch_size
                        : min(
                            len(global_batch),
                            (j + 1) * self.decode_policy.micro_batch_size,
                        )
                    ],
                    [
                        0 for i in range(now_micro_batch_size)
                    ],  
                    self.gpu_req_to_token_pool,
                    self.gpu_token_to_kv_pool_allocator,
                    self.cpu_req_to_token_pool,
                    self.cpu_token_to_kv_pool_allocator,
                    self.model_config,
                    self.spec_algorithm,
                    self.decode_policy,
                )
                if self.spec_algorithm.is_eagle(): # 组装spec_info
                    gpu_req_num = ceil(len(schedule_batch.reqs)*self.decode_policy.gpu_attention_ratio)
                    schedule_batch.spec_info = EagleDraftInput.merge_spec_infos([req.spec_info for req in schedule_batch.reqs], gpu_req_num)
                    for r in schedule_batch.reqs:
                        # TODO: 感觉还有很多地方像这里，需要手动解除一些引用，否则显存不会自己释放
                        r.spec_info = None
                micro_batches_pool.append(schedule_batch)

                # 统计gpu处理的最多长度，用于kv cache slot的分配
                gpu_attn_req_num = self.decode_policy.gpu_attention_micro_batch_size
                draft_gpu_req_num = int(self.decode_policy.micro_batch_size * self.decode_policy.draft_gpu_execution_ratio)
                assert draft_gpu_req_num >= gpu_attn_req_num
                draft_gpu_req_total_length = 0
                gpu_attn_req_total_length = 0
                for req_index in range(draft_gpu_req_num):
                    if req_index >= len(schedule_batch.reqs):
                        break
                    spec_draft_tokens = max(0, max(0 if self.server_args.speculative_num_draft_tokens is None else self.server_args.speculative_num_draft_tokens, 0 if self.server_args.speculative_num_draft_tokens_gpu is None else self.server_args.speculative_num_draft_tokens_gpu))

                    seql = len(schedule_batch.reqs[req_index].original_input_ids) + self.server_args.max_speculative_draft_tokens + self.server_args.max_output_length + spec_draft_tokens
                    draft_gpu_req_total_length += seql
                    if req_index < gpu_attn_req_num:
                        gpu_attn_req_total_length += seql
                # max_draft_gpu_req_total_length_in_a_batch = max(max_draft_gpu_req_total_length_in_a_batch, draft_gpu_req_total_length)
                max_draft_gpu_req_total_length_in_a_batch = max_draft_gpu_req_total_length_in_a_batch + draft_gpu_req_total_length
                max_gpu_attn_req_total_length_in_a_batch = max(max_gpu_attn_req_total_length_in_a_batch, gpu_attn_req_total_length)
            self.server_args.max_draft_gpu_req_total_length_in_a_batch = max_draft_gpu_req_total_length_in_a_batch
            self.server_args.max_gpu_attn_req_total_length_in_a_batch = max_gpu_attn_req_total_length_in_a_batch
            logger.info(f"max_draft_gpu_req_total_length_in_a_batch: {max_draft_gpu_req_total_length_in_a_batch}, max_gpu_attn_req_total_length_in_a_batch: {max_gpu_attn_req_total_length_in_a_batch}")

            # 4. 对一个Global Batch求解decode策略
            #    调整GPU KV cache pool容量
            self.tp_worker.model_runner.switch_to_decode(self.decode_policy)
            if self.spec_algorithm.is_eagle():
                self.draft_worker.switch_to_decode()
            
            while True:
                # break
                # for r in global_batch:
                #     r.init_next_round_input()
                # if decodecnt >= 3:
                #     break
                one_step_begin_time = time.time()
                schedule_micro_batches: List[ScheduleBatch] = micro_batches_pool
                # XXX: NEED CHECK! reset gpu kv cache allocator，这样每次decode新分配gpu kv cache slot，就都是连续的内存了
                self.gpu_token_to_kv_pool_allocator.clear()

                for micro_batch in schedule_micro_batches:
                    if self.spec_algorithm.is_none(): # XXX: spec decoding的seqlen信息在eagle_worker中更新
                        for r in micro_batch.reqs:
                            r.init_next_round_input()
                    micro_batch.prepare_for_decode()

                self._run_batch(schedule_micro_batches, self.decode_policy)
                new_finished_mb = -1
                remained_request_num = 0
                can_break = False
                for micro_batch in schedule_micro_batches:
                    micro_batch.cleanup_after_decode()
                    finished_reqs = []
                    for r in micro_batch.reqs:
                        r.init_next_round_input()
                        r.check_finished()
                        if r.finished():
                            # logger.debug(f"Request {r.rid} finished, output_ids: {r.output_ids}")
                            finished_reqs.append(r)
                    if len(finished_reqs) > 0:
                        micro_batch.remove_finished_req(finished_reqs)
                    if len(micro_batch.reqs) == 0:
                        if new_finished_mb == -1:
                            new_finished_mb = len(finished_micro_batches)
                        finished_micro_batches.append(micro_batch)
                    remained_request_num += len(micro_batch.reqs)
                if new_finished_mb != -1:
                    for i in range(new_finished_mb, len(finished_micro_batches)):
                        index = schedule_micro_batches.index(finished_micro_batches[i])
                        schedule_micro_batches.pop(index)
                        finished_micro_batches[i] = None # 释放micro-batch的引用应该可以释放资源？
                if remained_request_num <= self.server_args.global_batch_size * 0.05:
                    can_break = True
                    for micro_batch in schedule_micro_batches:
                        aborted_request_rids.extend([r.rid for r in micro_batch.reqs])

                decodecnt += 1
                self.log_metrics()
                logger.info(f"Decode step: {decodecnt}, time: {time.time() - one_step_begin_time:.2f}s, remained request num: {remained_request_num}")
                self.decode_times.append(time.time() - one_step_begin_time)
                if len(finished_micro_batches) == self.decode_policy.micro_batch_num:
                    break
                if can_break:
                    break
                if self.spec_algorithm.is_eagle():
                    if self.solver is not None:
                        self.change_plan(remained_request_num, self.avg_generate_length, running_micro_batch_num=len(schedule_micro_batches), log=False)
        total_generate_tokens_cnt = 0
        for r in all_req:
            if r.rid in aborted_request_rids:
                r.output_ids = [self.tokenizer.eos_token_id] * r.max_new_tokens
                continue
            total_generate_tokens_cnt += len(r.output_ids)
        end_time = time.time()
        logger.info(f"Total generate tokens: {total_generate_tokens_cnt}")
        if self.cum_spec_accept_count != 0:
            logger.info(f"Speculative average accepted tokens: {self.cum_spec_accept_length / self.cum_spec_accept_count:.2f}")
        logger.info(f"Decoding time: {end_time - decode_start_time}")
        logger.info(f"Decoding Throughput: {total_generate_tokens_cnt / (end_time - decode_start_time)}")
        logger.info(f"Overall time: {end_time - prefill_start_time}")
        logger.info(f"Overall Throughput: {total_generate_tokens_cnt / (end_time - prefill_start_time)}")

        if profiler_instance is not None:
            profiler_instance.stop()
            logger.debug("profiler instance stopped")
        
        metrics = {
            "total_generate_tokens_cnt": total_generate_tokens_cnt,
            "decode_time": end_time - decode_start_time,
            "decode_throughput": total_generate_tokens_cnt / (end_time - decode_start_time),
            "overall_time": end_time - prefill_start_time,
            "overall_throughput": total_generate_tokens_cnt / (end_time - prefill_start_time),
        }
        if self.cum_spec_accept_count != 0:
            metrics["speculative_average_accepted_tokens"] = self.cum_spec_accept_length / self.cum_spec_accept_count
        
        inference_infos = {
            "decode_time": self.decode_times,
            "history_accept_rate": self.history_accept_rate,
            "cpu_speculative_num_draft_token_list": self.cpu_speculative_num_draft_token_list,
            "gpu_speculative_num_draft_token_list": self.gpu_speculative_num_draft_token_list,
        }
        if len(self.draft_times) > 0:
            inference_infos["draft_time"] = self.draft_times
            inference_infos["gpu_draft_time"] = self.gpu_draft_times
            inference_infos["cpu_draft_time"] = self.cpu_draft_times
            inference_infos["target_model_time"] = self.target_model_times
            inference_infos["verify_time"] = self.verify_times
        if len(self.cpu_attention_times) > 0:
            inference_infos["cpu_attention_time"] = self.cpu_attention_times
            inference_infos["gpu_post_attention_time"] = self.gpu_post_attention_times
            inference_infos["htod_transfer_time"] = self.htod_transfer_times

        # 切分大batch，分别进行推理，逐个返回结果
        # ids = []
        # for r in all_req:
        #     # print(r.original_input_ids)
        #     ids.append(r.output_ids)

        self.send_outputs((self.tokenizer.decode(ret_ids), ret_ids, metrics, inference_infos))

        # do some cleanup
        self.init_metrics()

    def send_outputs(self, outputs: Tuple[List[str], List[List[int]], Dict]):
        self.send_to_engine.send_pyobj(outputs)

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            if len(recv_reqs) != 0:
                assert len(recv_reqs) == 1
                recv_reqs = recv_reqs[0]
                # logger.debug(f"recv_reqs: {recv_reqs}")
                reqs = []
                rid = 0
                for req in recv_reqs:
                    original_prompts = req["prompts"]
                    sampling_params = SamplingBatchInfo(
                        temperatures=req["sampling_params"].get("temperatures", 1.0),
                        greedy=req["sampling_params"].get("greedy", False), 
                        top_ks=req["sampling_params"].get("top_k", 1),
                        top_ps=req["sampling_params"].get("top_p", 0.86),
                    )
                    max_new_tokens = req.get("max_new_tokens", self.server_args.max_output_length)
                    original_input_ids = self.tokenizer.encode(original_prompts) if req["input_ids"] is None else req["input_ids"]
                    reqs.append(
                        Req(
                            original_prompt=original_prompts,
                            sampling_params=sampling_params,
                            max_new_tokens=max_new_tokens,
                            original_input_ids=original_input_ids,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    )
                self.run_batch(reqs)

    def recv_requests(self) -> Dict:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
        if self.tp_rank == 0:
            recv_reqs = []

            while True:
                try:
                    recv_req = self.recv_from_engine.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)

        return recv_reqs


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str

def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    solver = None
):
    # Generate the prefix
    if dp_rank is None:
        prefix = f" TP{tp_rank}"
    else:
        prefix = f" DP{dp_rank} TP{tp_rank}"

    # Config the process
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"specmoe::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank, solver)
        pipe_writer.send(
            {
                "status": "ready",
                # "max_total_num_tokens": scheduler.max_total_num_tokens,
                # "max_req_input_len": scheduler.max_req_input_len,
            }
        )
        scheduler.event_loop_overlap()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
