from specmoe.engine import SpecMoeEngine
from specmoe.optimizer.solver import Solver
import sys
sys.path.append("/home/zzh/codes/specmoe/test/examples")
from apps.loaddata import loaddat
from log_result import log_result
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_seql", type=int)
    parser.add_argument("--max_seql", type=int)
    parser.add_argument("--max_output_length", type=int, default=32)
    parser.add_argument("--avg_seq_len", type=int, default=1205)
    parser.add_argument("--cpu_memory", type=int, default=230)
    parser.add_argument("--first_name", type=str)
    parser.add_argument("--second_name", type=str)
    args = parser.parse_args()

    print(args)
    
    max_seq_len = args.max_seql + args.max_output_length + 60

    target_model_path = "/data1/zzh/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1"
    draft_model_path = "/data1/zzh/huggingface/hub/models--yuhuili--EAGLE-mixtral-instruct-8x7B/snapshots/f2e9cd1e1efaf0dec41c2da1b1fae4327727871d"

    ROOT = "/home/zzh/codes/specmoe/test"
    solver = Solver(
        target_model_path=target_model_path,
        draft_model_path=draft_model_path,
        average_seq_len=args.avg_seq_len, 
        max_seq_len=max_seq_len,
        max_speculative_len=20,
        gpu_memory_gb=24,
        cpu_memory_gb=args.cpu_memory,
        max_output_length=args.max_output_length,
        speculative_algorithm="EAGLE",
        tp_size=1,
        profile_root_path=f"{ROOT}/simulator_file",
        gpu_moe_fit_file_path=f"{ROOT}/simulator_file/fit_results/gpu_moe_fit.json",
        target_cpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/target_cpu_attn_fit.json", 
        target_gpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/target_gpu_attn_fit.json", 
        draft_cpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/draft_cpu_attn_fit.json", 
        draft_gpu_attn_fit_file_path=f"{ROOT}/simulator_file/fit_results/draft_gpu_attn_fit.json",
        htod_transfer_fit_file_path=f"{ROOT}/simulator_file/fit_results/htod_transfer_fit.json",
        log_level='info'
    )
    execution_plan = solver.solve()
    plan_dict = execution_plan.to_dict()

    print(plan_dict)
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
    draft_model_placement = plan_dict['draft_model_placement']
    throughput = plan_dict['throughput']
    cpu_dram_for_kv_cache = plan_dict['cpu_dram_for_kv_cache']
    target_cg_nano_kv_cache_slot = plan_dict['target_cg_nano_kv_cache_slot']
    draft_kv_cache_slot = plan_dict['draft_kv_cache_slot']
    
    prompt_num = global_batch_size
    prompts, sum_length = loaddata(prompt_num, min_len=args.min_seql, max_len=args.max_seql, return_sum_length=True)
    real_target_kv_cache_size = solver.get_real_target_kv_cache_size(sum_length, max_speculative_len=max(speculative_eagle_topk*speculative_num_steps, speculative_eagle_topk_gpu*speculative_num_steps_gpu))
    print("Overall length: ", sum_length)
    print("Real Target KV Cache Size: ", real_target_kv_cache_size, " GB")

    engine = SpecMoeEngine(
        solver=solver,
        gpu_hbm_size=23,
        available_cpu_dram_for_kvcache=real_target_kv_cache_size+2,
        target_cg_nano_kv_cache_slot=target_cg_nano_kv_cache_slot,
        draft_kv_cache_slot=draft_kv_cache_slot,
        model_path=target_model_path,
        speculative_draft_model_path=draft_model_path if speculative_num_draft_tokens!=1 else None,
        speculative_algorithm="EAGLE" if speculative_num_draft_tokens!=1 else None,
        log_level="info",
        port=8000,
        tp_size=1, 
        attention_backend="triton", 
        max_running_requests = global_batch_size + 1, 
        max_seq_length = max_seq_len, 
        max_output_length = args.max_output_length,
        max_speculative_draft_tokens = max(speculative_eagle_topk*speculative_num_steps, speculative_eagle_topk_gpu*speculative_num_steps_gpu),
        global_batch_size = global_batch_size,
        prefill_weight_cache_ratio = prefill_weight_cache_ratio,
        decode_weight_cache_ratio = decode_weight_cache_ratio,
        mem_fraction_static = 0.8,
        prefill_micro_batch_num = prefill_micro_batch_num,
        prefill_micro_batch_size = prefill_micro_batch_size,
        decode_micro_batch_num = decode_micro_batch_num,
        decode_micro_batch_size = decode_micro_batch_size,
        decode_gpu_attention_ratio = decode_gpu_attention_ratio,
        decode_gpu_attention_micro_batch_size = int(global_batch_size * decode_gpu_attention_ratio),
        decode_gpu_attention_nano_batch_size = int(global_batch_size * decode_gpu_attention_ratio),
        speculative_num_steps = speculative_num_steps, 
        speculative_eagle_topk = speculative_eagle_topk,
        speculative_num_draft_tokens = speculative_num_draft_tokens,
        speculative_num_steps_gpu = speculative_num_steps_gpu, 
        speculative_eagle_topk_gpu = speculative_eagle_topk_gpu,
        speculative_num_draft_tokens_gpu = speculative_num_draft_tokens_gpu,
        decode_spec_policy = "SequentialCGCoop",
        draft_gpu_execution_ratio = draft_gpu_execution_ratio,
        draft_model_placement = draft_model_placement,
        profile_output_dir="/home/zzh/codes/specmoe/test/logs/profile",
        dont_output_eos=True,
    )
    output, output_ids, metrics, inference_infos = engine.generate(prompts=prompts, top_k=1)
    print(metrics)
    print(inference_infos)
    log_result(engine, output, output_ids, metrics, inference_infos, global_batch_size, args.first_name, args.second_name)
