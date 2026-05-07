import logging
import torch
import multiprocessing as mp
from torch import nn
from typing import Optional, Union, List, Dict, Iterator
import zmq
import zmq.asyncio
import atexit
from typing import Tuple, Dict
import os
import signal
import asyncio
import uvloop
import threading
# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

from sglang.srt.utils import prepare_model_and_tokenizer, set_prometheus_multiproc_dir, set_ulimit, maybe_set_triton_cache_manager, assert_pkg_version, kill_process_tree, get_zmq_socket

from specmoe.utils.server_args import ServerArgs, PortArgs
from specmoe.backend.scheduler import run_scheduler_process
from specmoe.utils.utils import configure_logger

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


class SpecMoeEngine:
    '''
    Manage Scheduler, responsible for receiving requests and handing them over to the Scheduler at rank=0
    '''
    def __init__(self, solver = None, **kwargs):
        '''
        初始化serverargs, 创建scheduler
        '''
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)
        configure_logger(server_args)
        self.server_args = server_args

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)
        logging.info(f"{server_args=}")

        scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=port_args,
            solver=solver,
        )

        self.server_args = server_args
        self.scheduler_info = scheduler_info

        context = zmq.Context(2)
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.engine_ipc_name, True
        )

    def shutdown(self):
        """Shutdown the engine"""
        kill_process_tree(os.getpid(), include_parent=False) 

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompts: Optional[Union[List[str], str]] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 1,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        '''
        Receive requests and hand them over to the Scheduler at rank=0.
        '''
        # logging.debug(f"generate: {prompts=} {sampling_params=} {input_ids=}")
        send_reqs = []
        for prompt in prompts:
            send_reqs.append(
                {
                    "prompts": prompt,
                    "sampling_params": {"temperatures": temperature, "greedy": False, "top_p": top_p, "top_k": top_k},
                    "input_ids": input_ids,
                }
            )
        self.send_to_scheduler.send_pyobj(send_reqs)
        recv_outputs = []
        while True:
            try:
                recv_outputs = self.recv_from_scheduler.recv_pyobj()
                break
            except zmq.ZMQError:
                continue
        return recv_outputs

def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None, solver = None
) -> Tuple[Dict]:
    """
    Launch the Scheduler in a subprocess
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logging.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    
    scheduler_pipe_readers = []
    tp_size_per_node = server_args.tp_size // server_args.nnodes
    tp_rank_range = range(
        tp_size_per_node * server_args.node_rank,
        tp_size_per_node * (server_args.node_rank + 1),
    )
    for tp_rank in tp_rank_range:
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = (
            server_args.base_gpu_id
            + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
        )
        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, gpu_id, tp_rank, None, writer, solver),
        )
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logging.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logging.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    return scheduler_info

def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.3",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )

    def sigchld_handler(signum, frame):
        pid, exitcode = os.waitpid(0, os.WNOHANG)
        if exitcode != 0:
            logging.warning(
                "Child process unexpectedly failed with an exit code %d. pid=%d",
                exitcode,
                pid,
            )

    signal.signal(signal.SIGCHLD, sigchld_handler)

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        logging.error(
            "Received sigquit from a child process. It usually means the child failed."
        )
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)
