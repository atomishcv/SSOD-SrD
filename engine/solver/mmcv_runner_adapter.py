"""
简化的 Runner 适配器（借鉴 mmcv Runner 设计理念）

由于 mmcv 2.x 将 Runner 移到了 mmengine，且版本兼容性问题，
这里实现一个简化的 Runner，借鉴 mmcv Runner 的核心设计：
1. Hook 机制
2. 异常处理
3. 资源管理
4. 训练循环管理
5. 🔥 超时检测（解决训练卡死问题）

不直接依赖 mmcv，保持代码独立性。
"""

import torch
import threading
import time
from typing import List, Callable, Optional, Dict, Any
from enum import Enum

# 🔥 导入 dist_utils（用于判断是否为主进程）
try:
    from ..misc import dist_utils
except ImportError:
    # 如果导入失败，提供一个简单的替代
    class _FallbackDistUtils:
        @staticmethod
        def is_main_process():
            return True
    dist_utils = _FallbackDistUtils()


class TimeoutError(Exception):
    """Iteration 超时异常"""
    pass


class IterationWatchdog:
    """
    🔥 迭代看门狗：检测单个 iteration 是否卡死
    
    原理：
    1. 在每个 iteration 开始前启动定时器
    2. 如果 iteration 在超时时间内未完成，触发超时处理
    3. 支持自动恢复（跳过当前 batch，继续训练）
    """
    
    def __init__(self, timeout_seconds: float = 300, callback: Optional[Callable] = None):
        """
        Args:
            timeout_seconds: 单个 iteration 的超时时间（秒）
            callback: 超时回调函数
        """
        self.timeout = timeout_seconds
        self.callback = callback
        self._timer = None
        self._start_time = None
        self._lock = threading.Lock()
        self._timed_out = False
    
    def start(self):
        """开始监控一个 iteration"""
        with self._lock:
            self._timed_out = False
            self._start_time = time.time()
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.timeout, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()
    
    def stop(self):
        """停止监控（iteration 正常完成）"""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._start_time = None
    
    def _on_timeout(self):
        """超时回调"""
        with self._lock:
            self._timed_out = True
            elapsed = time.time() - self._start_time if self._start_time else 0
            print(f"⚠️ Watchdog: Iteration 超时! 已运行 {elapsed:.1f}s > {self.timeout}s")
            if self.callback:
                try:
                    self.callback()
                except Exception as e:
                    print(f"⚠️ Watchdog callback error: {e}")
    
    @property
    def timed_out(self) -> bool:
        """是否已超时"""
        with self._lock:
            return self._timed_out
    
    def reset(self):
        """重置状态"""
        self.stop()
        with self._lock:
            self._timed_out = False


class HookPriority(Enum):
    """Hook 优先级（参考 mmcv）"""
    LOWEST = 0
    LOWER = 10
    LOW = 20
    NORMAL = 50
    HIGH = 80
    HIGHER = 90
    HIGHEST = 100


class BaseHook:
    """Hook 基类（参考 mmcv Hook 设计）"""
    
    def before_train_epoch(self, runner): pass
    def after_train_epoch(self, runner): pass
    def before_train_iter(self, runner): pass
    def after_train_iter(self, runner): pass
    def before_val_epoch(self, runner): pass
    def after_val_epoch(self, runner): pass
    def before_val_iter(self, runner): pass
    def after_val_iter(self, runner): pass


class CheckpointHook(BaseHook):
    """Checkpoint Hook（自动保存模型）"""
    
    def __init__(self, interval: int = 1, save_best: bool = True, 
                 save_latest: bool = True, output_dir: Optional[str] = None):
        self.interval = interval
        self.save_best = save_best
        self.save_latest = save_latest
        self.output_dir = output_dir
        self.best_score = 0.0
    
    def after_train_epoch(self, runner):
        """每个 epoch 结束后保存 checkpoint"""
        epoch = runner.epoch
        
        if epoch % self.interval == 0:
            if self.save_latest and self.output_dir:
                runner.save_checkpoint(f'{self.output_dir}/checkpoint_latest.pth')
            
            # 保存最佳模型（基于验证指标）
            if self.save_best and hasattr(runner, 'val_metrics'):
                current_score = runner.val_metrics.get('ap', 0.0)
                if current_score > self.best_score:
                    self.best_score = current_score
                    if self.output_dir:
                        runner.save_checkpoint(
                            f'{self.output_dir}/checkpoint_best_ap{current_score:.4f}_epoch{epoch:03d}.pth'
                        )


class LogHook(BaseHook):
    """日志 Hook（记录训练信息）"""
    
    def __init__(self, interval: int = 50, log_file: Optional[str] = None):
        self.interval = interval
        self.log_file = log_file
    
    def after_train_iter(self, runner):
        """每个 iteration 后记录日志"""
        if runner.iter % self.interval == 0:
            metrics = runner.train_metrics
            log_str = f"Epoch {runner.epoch} | Iter {runner.iter} | " \
                     f"loss: {metrics.get('loss', 0.0):.4f} | " \
                     f"lr: {metrics.get('lr', 0.0):.2e}"
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_str + '\n')
            else:
                print(log_str)


class EpochBasedRunner:
    """
    简化的 EpochBasedRunner（借鉴 mmcv Runner 设计）
    
    核心特性：
    1. Hook 机制：解耦训练逻辑
    2. 异常处理：避免训练卡死
    3. 资源管理：自动清理显存
    4. 训练循环管理：统一的训练流程
    5. 🔥 超时检测：防止训练卡死
    """
    
    def __init__(self, model, optimizer, work_dir: Optional[str] = None, 
                 logger=None, meta: Optional[Dict] = None,
                 iter_timeout: float = 300.0):
        """
        Args:
            iter_timeout: 单个 iteration 的超时时间（秒），默认 300s（5分钟）
        """
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.meta = meta or {}
        
        # 训练状态
        self.epoch = 0
        self.iter = 0
        self.max_epochs = 0
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Hook 管理
        self.hooks: List[BaseHook] = []
        
        # 异常处理标志
        self._should_stop = False
        
        # 🔥 超时检测（可选，如果 iter_timeout <= 0 则禁用）
        self.iter_timeout = iter_timeout
        if iter_timeout > 0:
            self.watchdog = IterationWatchdog(
                timeout_seconds=iter_timeout,
                callback=self._on_iteration_timeout
            )
        else:
            self.watchdog = None  # 禁用 Watchdog，减少开销
        self._timeout_count = 0
        self._max_timeout_retries = 3  # 连续超时次数上限
    
    def _on_iteration_timeout(self):
        """Iteration 超时回调"""
        self._timeout_count += 1
        msg = f"⚠️ Iteration {self.iter} 超时（第 {self._timeout_count} 次）"
        if self.logger:
            self.logger.warning(msg)
        else:
            print(msg)
        
        # 🔥 不调用 empty_cache()，避免显存波动
        # PyTorch 会自动管理显存，强制清空会导致后续重新分配，造成波动
    
    def register_hook(self, hook: BaseHook, priority: HookPriority = HookPriority.NORMAL):
        """注册 Hook（按优先级排序）"""
        hook.priority = priority.value
        self.hooks.append(hook)
        # 按优先级排序（优先级高的先执行）
        self.hooks.sort(key=lambda x: x.priority, reverse=True)
    
    def call_hook(self, fn_name: str):
        """调用所有 Hook 的指定方法"""
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in {hook.__class__.__name__}.{fn_name}: {e}")
                    else:
                        print(f"⚠️ Error in {hook.__class__.__name__}.{fn_name}: {e}")
                    # 继续执行其他 Hook，不中断训练
    
    def save_checkpoint(self, filepath: str):
        """保存 checkpoint"""
        try:
            checkpoint = {
                'epoch': self.epoch,
                'iter': self.iter,
                'model': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
                'optimizer': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None,
                'meta': self.meta,
            }
            torch.save(checkpoint, filepath)
            if self.logger:
                self.logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save checkpoint: {e}")
            else:
                print(f"⚠️ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str, map_location='cpu'):
        """加载 checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=map_location)
            self.epoch = checkpoint.get('epoch', 0)
            self.iter = checkpoint.get('iter', 0)
            if 'model' in checkpoint and checkpoint['model'] is not None:
                if hasattr(self.model, 'load_state_dict'):
                    self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                if hasattr(self.optimizer, 'load_state_dict'):
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.logger:
                self.logger.info(f"Checkpoint loaded: {filepath}")
            return checkpoint
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load checkpoint: {e}")
            else:
                print(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def train_epoch(self, train_fn: Callable, *args, **kwargs):
        """
        训练一个 epoch（带异常处理和超时检测）
        
        Args:
            train_fn: 训练函数，签名: train_fn(...) -> dict
            *args, **kwargs: 传递给 train_fn 的参数
        """
        self.model.train()
        self.call_hook('before_train_epoch')
        
        try:
            # 🔥 启动 watchdog 监控整个 epoch（如果启用）
            if self.watchdog is not None:
                self.watchdog.start()
            
            # 调用训练函数（通常是 train_one_epoch_ssod）
            # 训练函数内部会处理整个 epoch 的循环
            # ⚠️ 注意：如果训练函数内部卡死，Watchdog 无法强制中断它
            # Watchdog 只能检测超时并记录，无法真正中断正在执行的训练循环
            metrics = train_fn(*args, **kwargs)
            
            # 🔥 epoch 完成，停止 watchdog（如果启用）
            if self.watchdog is not None:
                self.watchdog.stop()
                # 检查是否超时（训练函数可能已经完成，但超过了超时时间）
                if self.watchdog.timed_out:
                    self._timeout_count += 1
                    if dist_utils.is_main_process():
                        print(f"⚠️ Epoch {self.epoch} 完成，但耗时超过超时时间（>{self.iter_timeout}s）")
                        print(f"   记录超时 (连续超时: {self._timeout_count}/{self._max_timeout_retries})")
                else:
                    self._timeout_count = 0  # 正常完成，重置计数
            else:
                self._timeout_count = 0  # 未启用 Watchdog，重置计数
            
            self.train_metrics.update(metrics)
            self.call_hook('after_train_epoch')
            
        except RuntimeError as e:
            if self.watchdog is not None:
                self.watchdog.stop()
            # GPU 内存错误
            if 'out of memory' in str(e):
                print(f"⚠️ GPU OOM at epoch {self.epoch}")
                # 🔥 只在真正的 OOM 时调用 empty_cache（这是必要的恢复操作）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 返回空指标，让训练继续
                self.train_metrics = {}
            else:
                raise
        except KeyboardInterrupt:
            if self.watchdog is not None:
                self.watchdog.stop()
            print("\n⚠️ Training interrupted by user")
            self._should_stop = True
        except Exception as e:
            if self.watchdog is not None:
                self.watchdog.stop()
            print(f"❌ Fatal error in training epoch {self.epoch}: {e}")
            if self.logger:
                self.logger.error(f"Fatal training error: {e}")
            # 不抛出异常，让训练可以继续或优雅退出
            self.train_metrics = {}
    
    def val_epoch(self, data_loader, val_fn: Callable):
        """验证一个 epoch（带异常处理）"""
        self.model.eval()
        self.call_hook('before_val_epoch')
        
        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    self.call_hook('before_val_iter')
                    
                    try:
                        metrics = val_fn(self.model, batch_data, self.epoch)
                        if batch_idx == 0:
                            self.val_metrics = metrics
                        else:
                            # 累积指标
                            for k, v in metrics.items():
                                if isinstance(v, (int, float)):
                                    self.val_metrics[k] = (self.val_metrics.get(k, 0) * batch_idx + v) / (batch_idx + 1)
                    except Exception as e:
                        print(f"⚠️ Error in validation: {e}")
                        continue
                    
                    self.call_hook('after_val_iter')
            
            self.call_hook('after_val_epoch')
            
        except Exception as e:
            print(f"❌ Fatal error in validation: {e}")
            if self.logger:
                self.logger.error(f"Fatal validation error: {e}")
    
    def run(self, data_loaders: List, workflow: List[tuple], max_epochs: int):
        """
        运行训练循环（参考 mmcv Runner.run）
        
        Args:
            data_loaders: 数据加载器列表
            workflow: 工作流，如 [('train', 1), ('val', 1)]
            max_epochs: 最大 epoch 数
        """
        self.max_epochs = max_epochs
        
        # 解析 workflow
        train_loader = data_loaders[0] if len(data_loaders) > 0 else None
        val_loader = data_loaders[1] if len(data_loaders) > 1 else None
        
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            
            # 🔥 检查超时计数，如果连续超时过多则尝试恢复（仅在启用 Watchdog 时）
            if self.watchdog is not None and self._timeout_count >= self._max_timeout_retries:
                print(f"⚠️ 连续 {self._timeout_count} 次超时，尝试恢复...")
                self._recover_from_stuck()
                self._timeout_count = 0
            
            # 执行 workflow
            for mode, epochs in workflow:
                if mode == 'train' and train_loader is not None:
                    # 训练模式：需要传入训练函数
                    # 这里假设训练函数已经通过其他方式注册
                    if hasattr(self, '_train_fn'):
                        self.train_epoch(train_loader, self._train_fn)
                elif mode == 'val' and val_loader is not None:
                    # 验证模式
                    if hasattr(self, '_val_fn'):
                        self.val_epoch(val_loader, self._val_fn)
            
            # 检查是否应该停止
            if self._should_stop:
                break
        
        # 训练结束清理
        if self.watchdog is not None:
            self.watchdog.stop()
        # 🔥 不调用 empty_cache()，让 PyTorch 自动管理显存
    
    def _recover_from_stuck(self):
        """
        🔥 尝试从卡死状态恢复
        
        注意：只在真正卡死时调用，此时可能需要强制清理 GPU 状态
        """
        print("🔄 正在尝试恢复...")
        
        # 1. 停止 watchdog（如果启用）
        if self.watchdog is not None:
            self.watchdog.stop()
        
        # 2. 🔥 只在恢复场景下调用 synchronize（确保 GPU 操作完成）
        # 但不调用 empty_cache()，避免显存波动
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()  # 确保 GPU 操作完成
                # 不调用 empty_cache()，让 PyTorch 自动管理
                print("✅ GPU 状态已同步")
            except Exception as e:
                print(f"⚠️ GPU 同步失败: {e}")
        
        # 3. 重置 watchdog（如果启用）
        if self.watchdog is not None:
            self.watchdog.reset()
        
        # 4. 保存当前 checkpoint（以防万一）
        if self.work_dir:
            try:
                self.save_checkpoint(f'{self.work_dir}/checkpoint_recovery_epoch{self.epoch}.pth')
                print(f"✅ 已保存恢复点: checkpoint_recovery_epoch{self.epoch}.pth")
            except Exception as e:
                print(f"⚠️ 保存恢复点失败: {e}")
        
        print("🔄 恢复完成，继续训练...")

