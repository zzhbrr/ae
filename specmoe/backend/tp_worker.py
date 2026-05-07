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
"""A tensor parallel worker."""

import logging
import threading
from typing import Optional, Tuple, List

import torch

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed

from specmoe.utils.server_args import ServerArgs
from specmoe.utils.model_config import ModelConfig
from specmoe.backend.model_runner import ModelRunner
from specmoe.speculative.spec_info import SpeculativeAlgorithm
from specmoe.backend.memory import ReqToTokenPool, TokenToKVPoolAllocatorGPU, TokenToKVPoolAllocatorCPU
from specmoe.utils.data_info import ModelWorkerBatch, global_server_args_dict
from specmoe.backend.forward_batch_info import ForwardBatch
from specmoe.backend.policy import Policy
from specmoe.layers.logits_processor import LogitsProcessorOutput

logger = logging.getLogger(__name__)


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        offload: bool = True,
        gpu_req_to_token_pool: Optional[ReqToTokenPool] = None,
        gpu_token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocatorGPU] = None,
        cpu_req_to_token_pool: Optional[ReqToTokenPool] = None,
        cpu_token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocatorCPU] = None,
    ):
        # Parse args
        self.tp_rank = tp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig(
            (
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            is_draft_model=is_draft_worker,
            spec_algorithm=SpeculativeAlgorithm.from_string(server_args.speculative_algorithm),
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            pin_experts_when_load_weight=server_args.pin_experts_when_load_weight,
        )
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            offload=offload,
            gpu_req_to_token_pool=gpu_req_to_token_pool,
            gpu_token_to_kv_pool_allocator=gpu_token_to_kv_pool_allocator,
            cpu_req_to_token_pool=cpu_req_to_token_pool,
            cpu_token_to_kv_pool_allocator=cpu_token_to_kv_pool_allocator,
        )
        self.device = self.model_runner.device

        # # Profile number of tokens
        # self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        # self.max_prefill_tokens = server_args.max_prefill_tokens
        # self.max_running_requests = min(
        #     (
        #         self.max_total_num_tokens // 2
        #         if server_args.max_running_requests is None
        #         else server_args.max_running_requests
        #         // (server_args.dp_size if server_args.enable_dp_attention else 1)
        #     ),
        #     self.model_runner.req_to_token_pool.size,
        # )
        # assert self.max_running_requests > 0, "max_running_request is zero"
        # self.max_req_len = min(
        #     self.model_config.context_len - 1,
        #     self.max_total_num_tokens - 1,
        # )
        # self.max_req_input_len = self.max_req_len - 5
        # assert (
        #     self.max_req_len > 0 and self.max_req_input_len > 0
        # ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    # def get_tp_cpu_group(self):
    #     return self.model_runner.tp_group.cpu_group

    # def get_attention_tp_cpu_group(self):
    #     return self.model_runner.attention_tp_group.cpu_group

    def get_memory_pool(self):
        return (
            self.model_runner.gpu_req_to_token_pool,
            self.model_runner.gpu_token_to_kv_pool_allocator,
            self.model_runner.cpu_req_to_token_pool,
            self.model_runner.cpu_token_to_kv_pool_allocator,
        )

    def forward_batch_generation(
        self,
        model_worker_micro_batches: List[ModelWorkerBatch],
        policy: Policy,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[List[LogitsProcessorOutput], Optional[List[torch.Tensor]], Optional[dict]]:
        forward_micro_batches: List[ForwardBatch] = []
        for model_worker_micro_batch in model_worker_micro_batches:
            forward_micro_batches.append(ForwardBatch.init_new(model_worker_micro_batch, self.model_runner))
        
        res, performance_recoder = self.model_runner.forward(forward_micro_batches, policy)
        
        if self.model_config.spec_algorithm.is_none():
            for i in range(len(res)):
                assert res[i].hidden_states is None
        else:
            # for i in range(len(res)):
            #     res[i].hidden_states = res[i].hidden_states.to('cpu').pin_memory() # XXX: Should we offload hidden state for EAGLE to CPU ?????
            pass
        # logger.debug(f"logits_outputs: len: {len(logits_outputs)}, shape: {logits_outputs[0].shape}")
        # for i in range(logits_outputs[0].shape[0]):
        #     logger.debug(f"logits_outputs[{i}]: {logits_outputs[0][i][:10]}")
        if launch_done:
            launch_done.set()
        
        if self.model_runner.logit_bias is not None:
            for i in range(len(res)):
                res[i].next_token_logits.add_(self.model_runner.logit_bias.repeat(res[i].next_token_logits.shape[0], 1))

        
        next_token_ids:Optional[List[torch.Tensor]] = []
        if skip_sample:
            next_token_ids = None
        else:
            logits_outputs = [res[i].next_token_logits for i in range(len(res))]
            for logits_output, model_worker_micro_batch in zip(logits_outputs, model_worker_micro_batches):
                next_token_ids.append(self.model_runner.sample(logits_output, model_worker_micro_batch))

        return res, next_token_ids, performance_recoder