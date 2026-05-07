import torch
import threading
from specmoe.backend.memory import MHATokenToKVPool
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Union
from queue import PriorityQueue
import logging
import time

logger = logging.getLogger(__name__)


class TransferTaskType(Enum):
    HIDDEN_LOADING = auto()
    KV_CACHE_LOADING = auto()
    DRAFT_KV_CACHE_LOADING = auto()
    WEIGHT_LOADING = auto()


# 定义任务优先级（数字越小优先级越高）
TASK_PRIORITY_MAPPING = {
    TransferTaskType.HIDDEN_LOADING: 1,      # 最高优先级
    TransferTaskType.KV_CACHE_LOADING: 2,    # 次高优先级
    TransferTaskType.DRAFT_KV_CACHE_LOADING: 3,    # 次高优先级
    TransferTaskType.WEIGHT_LOADING: 4,      # 最低优先级
}


@dataclass
class TransferTask:
    """传输任务数据结构"""
    task_type: TransferTaskType
    src: torch.Tensor = None
    dst: torch.Tensor = None
    stream: torch.cuda.Stream = None
    event: Optional[torch.cuda.Event] = None
    priority: int = 0
    current_id: Optional[int] = None  # 用于切片任务
    total_id: Optional[int] = None    # 用于切片任务
    task_id: Optional[str] = None     # 任务标识符，用于条件变量通知
    layer_id: Optional[int] = None    # 层ID
    
    def __post_init__(self):
        if self.priority == 0:
            self.priority = TASK_PRIORITY_MAPPING[self.task_type]
    
    def __lt__(self, other):
        # 优先级队列使用，数字越小优先级越高
        # 首先按照优先级排序
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # 如果优先级相同，先按layer_id从小到大排序；如果layer_id相同，再按current_id从小到大排序
        self_layer_id = self.layer_id if self.layer_id is not None else 0
        other_layer_id = other.layer_id if other.layer_id is not None else 0
        if self_layer_id != other_layer_id:
            if self_layer_id == 0:
                return False  # self在后面
            if other_layer_id == 0:
                return True   # self在前面
            return self_layer_id < other_layer_id
        self_current_id = self.current_id if self.current_id is not None else 0
        other_current_id = other.current_id if other.current_id is not None else 0
        
        return self_current_id < other_current_id

@dataclass
class KVCacheTransferTask(TransferTask):
    src_kvcache: torch.Tensor = None
    dst_indices: slice = None
    gpu_kv_pool: MHATokenToKVPool = None
    two_buffer_offset: int = -1

class HtoDTransferEngine:
    """Host-to-Device传输引擎"""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 默认1MB切片大小
        self.task_queue = PriorityQueue()
        self.chunk_size = chunk_size
        self.running = False
        self.worker_thread = None
        
        # 条件变量字典，用于通知切片任务完成
        self.completion_conditions: Dict[str, threading.Condition] = {}
        self.completion_status: Dict[str, bool] = {}
        
        # 线程锁
        self.condition_lock = threading.Lock()
        
        # 工作线程控制变量
        self.worker_enabled = True
        self.worker_enabled_lock = threading.Lock()
        
        # WEIGHT_LOADING任务计时相关数据结构
        self.weight_loading_start_times: Dict[str, float] = {}  # 记录每个task_id的开始时间
        self.weight_loading_end_times: Dict[str, float] = {}    # 记录每个task_id的结束时间
        self.weight_loading_timing_lock = threading.Lock()      # 计时数据的线程锁
        
    def start(self):
        """启动传输引擎"""
        if self.running:
            logger.warning("HtoDTransferEngine is already running")
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("HtoDTransferEngine Started")
    
    def stop(self):
        """停止传输引擎"""
        if not self.running:
            return
            
        self.running = False
        # 添加一个停止任务来唤醒worker线程
        self.task_queue.put(None)
        
        if self.worker_thread:
            self.worker_thread.join()
            
        # 清理资源
        with self.condition_lock:
            self.completion_conditions.clear()
            self.completion_status.clear()
        
        # 清理计时相关数据
        with self.weight_loading_timing_lock:
            self.weight_loading_start_times.clear()
            self.weight_loading_end_times.clear()
            
        # 清空队列
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break
                
        logger.info("HtoDTransferEngine Stopped")
    
    def submit_hidden_loading_task(self, src: torch.Tensor, dst: torch.Tensor, 
                                 stream: torch.cuda.Stream, event: torch.cuda.Event):
        """提交hidden loading任务"""
        '''直接传输，避免后续步骤开始时event还未设置'''
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
            event.record(stream)
        
        # logger.debug(f"完成hidden loading任务，张量大小: {src.numel()}")
        
    def submit_kv_cache_loading_task(self, src_kvcache: torch.Tensor, dst_indices: slice, gpu_kv_pool: MHATokenToKVPool, event: torch.cuda.Event, stream: torch.cuda.Stream, task_id: str, layer_id: int, two_buffer_offset: int = -1):
        """提交KV cache loading任务（会被切片）"""
        with self.condition_lock:
            if task_id not in self.completion_conditions:
                self.completion_conditions[task_id] = threading.Condition()
                self.completion_status[task_id] = False
        
        with torch.cuda.stream(stream):
            gpu_kv_pool.load_kv_cache_from_cpu(layer_id, dst_indices, src_kvcache, non_blocking=True, gpu_two_buffer=True, two_buffer_offset=two_buffer_offset) 
        event.record(stream)
        # task = KVCacheTransferTask(
        #     task_type=TransferTaskType.KV_CACHE_LOADING,
        #     src_kvcache=src_kvcache,
        #     dst_indices=dst_indices,
        #     gpu_kv_pool=gpu_kv_pool,
        #     two_buffer_offset=two_buffer_offset,
        #     stream=stream,
        #     event=event,
        #     task_id=task_id,
        #     layer_id=layer_id, 
        # )
        # self.task_queue.put(task)
    def submit_kv_cache_loading_task_in_batch(self, src_kvcache_list: List[torch.Tensor], dst_indices_list: List[slice], gpu_kv_pool: MHATokenToKVPool, event: torch.cuda.Event, stream: torch.cuda.Stream, stream2: torch.cuda.Stream, task_id: str, layer_id: int, two_buffer_offset: int = -1):
        # 停止工作线程调度
        with self.worker_enabled_lock:
            self.worker_enabled = False
        
        try:
            with torch.cuda.stream(stream):
                gpu_kv_pool.load_kv_cache_from_cpu_in_batch(layer_id, dst_indices_list, src_kvcache_list, non_blocking=False, gpu_two_buffer=True, two_buffer_offset=two_buffer_offset, stream=stream, stream2=stream2) 
            # event.record(stream)
        finally:
            # 恢复工作线程调度
            with self.worker_enabled_lock:
                self.worker_enabled = True
    
    def submit_kv_cache_loading_task_in_batch2(self, src_pinned: torch.Tensor, src_indices_list: List[tuple[int, int]], dst_indices_list: List[slice], gpu_kv_pool: MHATokenToKVPool, event: torch.cuda.Event, stream: torch.cuda.Stream, layer_id: int, two_buffer_offset: int = -1):
        # 停止工作线程调度
        # with self.worker_enabled_lock:
        #     self.worker_enabled = False
        
        try:
            with torch.cuda.stream(stream):
                tmp = src_pinned[src_indices_list[0][0]:src_indices_list[-1][1], ...].to('cuda', non_blocking=False)
                # for i in range(len(src_indices_list)):
                #     # print(f"layer_id: {layer_id}")
                #     src_len = src_indices_list[i][1] - src_indices_list[i][0]
                #     dst_len = dst_indices_list[i].stop - dst_indices_list[i].start
                #     # print(f"src_len: {src_len}, dst_len: {dst_len}")
                #     assert src_len == dst_len, f"源区间和目标区间长度不一致: src_len={src_len}, dst_len={dst_len}"
                #     gpu_kv_pool.kv_buffer[two_buffer_offset][0, dst_indices_list[i], ...] = tmp[src_indices_list[i][0]-src_indices_list[0][0]:src_indices_list[i][1]-src_indices_list[0][0], 0, ...]
                #     gpu_kv_pool.kv_buffer[two_buffer_offset][1, dst_indices_list[i], ...] = tmp[src_indices_list[i][0]-src_indices_list[0][0]:src_indices_list[i][1]-src_indices_list[0][0], 1, ...]
                # 区间合并优化：将连续的区间合并为更大的块进行批量拷贝
                merged_intervals = []
                i = 0
                while i < len(src_indices_list):
                    # 开始一个新的合并区间
                    merged_src_start = src_indices_list[i][0] - src_indices_list[0][0]  # 相对于tmp的偏移
                    merged_src_end = src_indices_list[i][1] - src_indices_list[0][0]
                    merged_dst_start = dst_indices_list[i].start
                    merged_dst_end = dst_indices_list[i].stop
                    
                    # 检查是否可以与后续区间合并
                    j = i + 1
                    while j < len(src_indices_list):
                        # 检查源区间和目标区间是否都连续
                        src_is_continuous = (src_indices_list[j][0] - src_indices_list[0][0]) == merged_src_end
                        dst_is_continuous = dst_indices_list[j].start == merged_dst_end
                        
                        if src_is_continuous and dst_is_continuous:
                            # 扩展合并区间
                            merged_src_end = src_indices_list[j][1] - src_indices_list[0][0]
                            merged_dst_end = dst_indices_list[j].stop
                            j += 1
                        else:
                            break
                    
                    # 添加合并后的区间
                    merged_intervals.append({
                        'src_start': merged_src_start,
                        'src_end': merged_src_end,
                        'dst_start': merged_dst_start,
                        'dst_end': merged_dst_end,
                        'original_count': j - i  # 记录合并了多少个原始区间
                    })
                    
                    i = j
                
                # 使用合并后的区间进行批量拷贝
                for interval in merged_intervals:
                    src_slice = slice(interval['src_start'], interval['src_end'])
                    dst_slice = slice(interval['dst_start'], interval['dst_end'])
                    
                    # 验证长度一致性
                    src_len = interval['src_end'] - interval['src_start']
                    dst_len = interval['dst_end'] - interval['dst_start']
                    assert src_len == dst_len, f"合并区间长度不一致: src_len={src_len}, dst_len={dst_len}"
                    
                    # 批量拷贝key和value
                    gpu_kv_pool.kv_buffer[two_buffer_offset][0, dst_slice, ...] = tmp[src_slice, 0, ...]
                    gpu_kv_pool.kv_buffer[two_buffer_offset][1, dst_slice, ...] = tmp[src_slice, 1, ...]
            # event.record(stream)
        finally:
            pass
        #     # 恢复工作线程调度
        #     with self.worker_enabled_lock:
        #         self.worker_enabled = True
    
    def submit_weight_loading_task(self, src: torch.Tensor, dst: torch.Tensor,
                                 stream: torch.cuda.Stream, task_id: str, num_chunks: int, layer_id: int):
        """提交weight loading任务（会被切片）"""
        self._submit_chunked_task(
            task_type=TransferTaskType.WEIGHT_LOADING,
            src=src,
            dst=dst,
            stream=stream,
            task_id=task_id,
            num_chunks=num_chunks,
            layer_id=layer_id
        )
        # logger.debug(f"提交weight loading任务，张量大小: {src.numel()}, 任务ID: {task_id}")
    
    def submit_draft_kv_cache_loading_task(self, src: torch.Tensor, dst: torch.Tensor, stream: torch.cuda.Stream, task_id: str, num_chunks: int, layer_id: int=0):
        self._submit_chunked_task(
            task_type=TransferTaskType.DRAFT_KV_CACHE_LOADING,
            src=src,
            dst=dst,
            stream=stream,
            task_id=task_id,
            num_chunks=num_chunks,
            layer_id=layer_id
        )

    def _submit_chunked_task(self, task_type: TransferTaskType, src: torch.Tensor, 
                           dst: torch.Tensor, stream: torch.cuda.Stream, task_id: str, num_chunks: int, layer_id: int):
        """提交需要切片的任务"""
        # 初始化条件变量
        with self.condition_lock:
            if task_id not in self.completion_conditions:
                self.completion_conditions[task_id] = threading.Condition()
                self.completion_status[task_id] = False
        
        if num_chunks == 1:
            # 不需要切片，直接传输
            task = TransferTask(
                task_type=task_type,
                src=src,
                dst=dst,
                stream=stream,
                current_id=1,
                total_id=1,
                task_id=task_id,
                layer_id=layer_id,
            )
            self.task_queue.put(task)
        else:
            # 确保tensor是连续的
            src_flat = src.contiguous().view(-1)
            dst_flat = dst.contiguous().view(-1)
            
            # 计算每个chunk的大小，处理不能整除的情况
            total_size = src_flat.size(0)
            base_chunk_size = total_size // num_chunks
            remainder = total_size % num_chunks
            
            # 创建切片任务
            current_pos = 0
            for i in range(num_chunks):
                # 前remainder个chunk每个多分配1个元素
                current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
                
                start_idx = current_pos
                end_idx = current_pos + current_chunk_size
                
                src_chunk = src_flat[start_idx:end_idx]
                dst_chunk = dst_flat[start_idx:end_idx]
                
                task = TransferTask(
                    task_type=task_type,
                    src=src_chunk,
                    dst=dst_chunk,
                    stream=stream,
                    current_id=i + 1,
                    total_id=num_chunks,
                    task_id=task_id,
                    layer_id=layer_id,
                )
                self.task_queue.put(task)
                
                current_pos = end_idx
    
    def has_task_id(self, task_id: str) -> bool:
        with self.condition_lock:
            return task_id in self.completion_conditions
    
    def wait_for_completion(self, task_id: str, timeout: Optional[float] = None) -> Union[bool, tuple[bool, Optional[float]]]:
        """等待切片任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间
            
        Returns:
            对于WEIGHT_LOADING任务: (success: bool, execution_time: Optional[float])
            对于其他任务: success: bool
        """
        with self.condition_lock:
            if task_id not in self.completion_conditions:
                logger.warning(f"任务ID {task_id} 不存在")
                return False
            
            condition = self.completion_conditions[task_id]
            
        with condition:
            if not self.completion_status[task_id]:
                result = condition.wait(timeout)
                if not result:
                    logger.warning(f"任务 {task_id} 等待超时")
                    return False
            # self.completion_status[task_id] = False
            
            # 检查是否为WEIGHT_LOADING任务，如果是则返回执行时间
            execution_time = None
            is_weight_loading = False
            
            with self.weight_loading_timing_lock:
                if (task_id in self.weight_loading_start_times and 
                    task_id in self.weight_loading_end_times):
                    is_weight_loading = True
                    start_time = self.weight_loading_start_times[task_id]
                    end_time = self.weight_loading_end_times[task_id]
                    execution_time = end_time - start_time
            
            if is_weight_loading:
                return True, execution_time
            else:
                return True
    
    def clear_condition(self, task_ids: List[str]):
        with self.condition_lock:
            for task_id in task_ids:
                if task_id in self.completion_conditions:
                    del self.completion_conditions[task_id]
                    del self.completion_status[task_id]
        
        # 清理WEIGHT_LOADING任务的计时数据
        with self.weight_loading_timing_lock:
            for task_id in task_ids:
                if task_id in self.weight_loading_start_times:
                    del self.weight_loading_start_times[task_id]
                if task_id in self.weight_loading_end_times:
                    del self.weight_loading_end_times[task_id]
    
    def _worker_loop(self):
        """工作线程主循环"""
        # logger.info("传输引擎工作线程开始运行")
        
        while self.running:
            try:
                # 检查工作线程是否启用
                while self.running:
                    with self.worker_enabled_lock:
                        if self.worker_enabled:
                            break
                    time.sleep(0.001)  # 1ms延迟，避免忙等待
                
                if not self.running:
                    break
                
                # 获取任务（阻塞）
                task = self.task_queue.get()
                
                # 检查停止信号
                if task is None:
                    break
                
                # 执行任务
                self._execute_task(task)
                
                # 标记任务完成
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"执行传输任务时发生错误: {e}")
                # 仍然标记任务完成
                try:
                    self.task_queue.task_done()
                except:
                    pass
                continue
        
        logger.info("HtoDTransferEngine Worker Thread Ended")
    
    def _execute_task(self, task: Union[TransferTask, KVCacheTransferTask]):
        """执行单个传输任务"""
        try:
            # 执行数据传输
            start_time = time.time()
            
            # 为WEIGHT_LOADING任务记录开始时间（仅对第一个切片任务记录）
            if (task.task_type == TransferTaskType.WEIGHT_LOADING and 
                task.current_id == 1 and task.task_id):
                with self.weight_loading_timing_lock:
                    self.weight_loading_start_times[task.task_id] = start_time
            
            if task.task_type == TransferTaskType.HIDDEN_LOADING:
                # 使用阻塞传输，确保完成
                assert False, "Should never reach here"
                with torch.cuda.stream(task.stream):
                    task.dst.copy_(task.src, non_blocking=True)
                task.event.record(task.stream)
            elif task.task_type == TransferTaskType.KV_CACHE_LOADING:
                assert False, "Should never reach here"
                with torch.cuda.stream(task.stream):
                    task.gpu_kv_pool.load_kv_cache_from_cpu(task.layer_id, task.dst_indices, task.src_kvcache, non_blocking=True, gpu_two_buffer=True, two_buffer_offset=task.two_buffer_offset) 
            else:
                assert task.task_type == TransferTaskType.DRAFT_KV_CACHE_LOADING or task.task_type == TransferTaskType.WEIGHT_LOADING
                with torch.cuda.stream(task.stream):
                    task.dst.copy_(task.src, non_blocking=False)
                # logger.debug(f"transfer task done: {task.task_id}, {task.current_id}/{task.total_id}, use time: {(time.time() - start_time)*1000:.2f}ms")
                # print(task.current_id, task.total_id, task.task_id)
                
                # 为WEIGHT_LOADING任务记录结束时间（仅对最后一个切片任务记录）
                if (task.task_type == TransferTaskType.WEIGHT_LOADING and 
                    task.current_id is not None and task.total_id is not None and 
                    task.current_id == task.total_id and task.task_id):
                    with self.weight_loading_timing_lock:
                        self.weight_loading_end_times[task.task_id] = time.time()
                
                # 如果是切片任务的最后一片，通知完成
                if (task.current_id is not None and task.total_id is not None and 
                    task.current_id == task.total_id and task.task_id):
                    self._notify_completion(task.task_id)
            
        except Exception as e:
            logger.error(f"执行传输任务失败: {e}")
            if task.task_id:
                self._notify_completion(task.task_id, success=False)
    
    def _notify_completion(self, task_id: str, success: bool = True):
        """通知切片任务完成"""
        with self.condition_lock:
            if task_id in self.completion_conditions:
                condition = self.completion_conditions[task_id]
                self.completion_status[task_id] = success
                
                with condition:
                    condition.notify_all()
                
                # logger.debug(f"任务 {task_id} 完成通知已发送，状态: {success}")
    
    def clear_completed_tasks(self, task_ids: list):
        """清理已完成任务的条件变量"""
        with self.condition_lock:
            for task_id in task_ids:
                if task_id in self.completion_conditions:
                    del self.completion_conditions[task_id]
                    del self.completion_status[task_id]
        
        # 清理WEIGHT_LOADING任务的计时数据
        with self.weight_loading_timing_lock:
            for task_id in task_ids:
                if task_id in self.weight_loading_start_times:
                    del self.weight_loading_start_times[task_id]
                if task_id in self.weight_loading_end_times:
                    del self.weight_loading_end_times[task_id]
    
    def __del__(self):
        """析构函数"""
        self.stop() 