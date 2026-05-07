import torch
from torch import nn
import dataclasses 
from typing import Optional
import logging

from specmoe.backend.forward_batch_info import ForwardMode, ForwardBatch

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    hidden_states: Optional[torch.Tensor] = None

    hidden_states_cpu: Optional[torch.Tensor] = None
    next_token_logits_cpu: Optional[torch.Tensor] = None


class LogitsProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = 1

    def forward(self, input_ids, hidden_states, weight, forward_batch: ForwardBatch, aux_hidden_states: Optional[torch.Tensor] = None):
        # assert not forward_batch.return_logprob
        if forward_batch.forward_mode == ForwardMode.DECODE or forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:
            pruned_states = hidden_states
        elif forward_batch.forward_mode.is_extend():
            last_index = (
                torch.cumsum(
                    forward_batch.extend_seq_lens,
                    dim=0,
                    dtype=torch.long,
                )
                - 1
            ).to(device=hidden_states.device)
            pruned_states = hidden_states[last_index]

        last_logits = torch.matmul(pruned_states, weight.T)

        last_logits = last_logits[:, : self.config.vocab_size]


        if forward_batch.capture_hidden_mode.need_capture():
            hidden_states_to_store: Optional[torch.Tensor] = None
            if forward_batch.capture_hidden_mode.is_full():
                hidden_states_to_store = hidden_states
            elif forward_batch.capture_hidden_mode.is_last():
                hidden_states_to_store = pruned_states
            return LogitsProcessorOutput(last_logits, hidden_states_to_store)
        return LogitsProcessorOutput(last_logits, None)
