"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import re
import torch.multiprocessing as mp
from datetime import datetime

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

# ✅ 增加 NCCL 超时时间（防止长时间训练超时）
# 默认是 30 分钟，增加到 3 小时
os.environ.setdefault('NCCL_TIMEOUT', '10800')  # 秒
os.environ.setdefault('NCCL_DEBUG', 'WARN')  # 启用调试信息
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # 允许 NCCL 在通信异常时及时抛错

def main(args, ) -> None:
    """main
    """
    # ========== 设置日志保存（仅 rank 0） ==========
    if args.save_log and (not hasattr(args, 'local_rank') or args.local_rank == 0):
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 自动生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        log_file = os.path.join(log_dir, f'{config_name}_{timestamp}.log')
        
        # 重定向 stdout 和 stderr 到日志文件（同时保留终端输出）
        import sys
        
        class TeeLogger:
            """同时输出到终端和文件"""
            def __init__(self, file_path):
                self.terminal = sys.stdout
                self.log_file = open(file_path, 'a', encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log_file.write(message)
                self.log_file.flush()  # 实时写入
            
            def flush(self):
                self.terminal.flush()
                self.log_file.flush()
            
            def close(self):
                self.log_file.close()
        
        # 创建日志记录器
        tee_logger = TeeLogger(log_file)
        sys.stdout = tee_logger
        sys.stderr = tee_logger
        
        print(f"{'='*80}")
        print(f"Training log will be saved to: {log_file}")
        print(f"Start time: {timestamp}")
        print(f"Config: {args.config}")
        print(f"{'='*80}\n")
    
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)

    # 支持从 warmup 检查点（仅监督阶段）恢复，并直接进入半监督训练
    # 使用方式：--resume-warmup outputs/.../checkpoint_warmup_epoch009.pth
    resume_warmup_path = getattr(args, 'resume_warmup', None)
    warmup_extra_updates = {}
    if resume_warmup_path is not None:
        try:
            import torch
            state = torch.load(resume_warmup_path, map_location='cpu')
            ckpt_epoch = state.get('epoch', state.get('last_epoch', None))
        except Exception:
            ckpt_epoch = None
        # 兜底：从文件名中解析 epoch 编号
        if ckpt_epoch is None:
            m = re.search(r'epoch(\d+)', os.path.basename(resume_warmup_path))
            if m:
                ckpt_epoch = int(m.group(1))
        # 设置恢复参数
        warmup_extra_updates['resume'] = resume_warmup_path
        if ckpt_epoch is not None:
            warmup_extra_updates['last_epoch'] = ckpt_epoch

        print(f"[Warmup-Resume] load={resume_warmup_path}, parsed_epoch={ckpt_epoch}")

    # 合并 CLI 参数与额外的 warmup 恢复参数
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})
    update_dict.update(warmup_extra_updates)

    cfg = YAMLConfig(args.config, **update_dict)

    if cfg.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()

# --- helpers for self-launched distributed (must be at top-level for spawn) ---
def _parse_port_from_url(url: str) -> str:
    m = re.search(r':(\d+)$', url)
    return m.group(1) if m else '29500'

def _worker(local_rank: int, args):
    # Set basic envs for env:// init
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', _parse_port_from_url(args.dist_url))
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(args.num_gpus)
    
    # ✅ 在子进程中重新设置 NCCL 环境变量（multiprocessing spawn 不会自动继承）
    os.environ.setdefault('NCCL_TIMEOUT', '10800')  # 3小时超时
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # 阻塞等待，更容易定位问题
    
    # 🔥 额外NCCL优化（针对800×800分辨率和大模型通信）
    os.environ.setdefault('NCCL_IB_DISABLE', '0')  # 启用InfiniBand（如果可用）
    os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # 启用P2P通信（GPU间直接通信）
    os.environ.setdefault('NCCL_SHM_DISABLE', '0')  # 启用共享内存
    
    # 🔥 CUDA内存优化（避免OOM和显存碎片）
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')
    
    # Propagate to argparse namespace
    args.local_rank = local_rank
    main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # priority 0
    parser.add_argument('-c', '--config', type=str, default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, default=None, help='device (auto-detect in distributed training)',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true',default=True, help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, default='outputs/deimv2_dinov3_x_dji_800_ssod', help='output directory')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summary')
    parser.add_argument('--test-only', action='store_true', default=False,)
    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank id')
    # self-launch distributed (for running script directly without torchrun)
    parser.add_argument('--num-gpus', type=int, default=2, help='number of GPUs to use (spawn processes if >1)')
    parser.add_argument('--gpu-ids', type=str, default='1,2,3,4', help='comma-separated GPU ids, e.g. "0,1"')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:29500', help='url used to set up distributed training')
    # 日志保存
    parser.add_argument('--save-log', action='store_true', default=True, 
                       help='save training log to file (auto-generate filename with timestamp in logs/ directory)')
    # 从仅监督 warmup 检查点恢复，直接开始半监督训练
    parser.add_argument('--resume-warmup', type=str, default=None,
                        help='path to supervised-only warmup checkpoint')
    args = parser.parse_args()
 
    # Optional: set visible devices from CLI
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    if args.num_gpus and args.num_gpus > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        mp.spawn(_worker, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        # single process
        main(args)
