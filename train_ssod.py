"""
DEIM 半监督训练入口脚本（基于 DEIMv2 现有的 SSOD 框架）

使用方式：
    单卡：python train_ssod.py -c configs/ssod/deim_ssod_dji.yml
    多卡：python train_ssod.py -c configs/ssod/deim_ssod_dji.yml --num-gpus 4

也可以使用现有的 DEIMv2 SSOD 配置：
    python train_ssod.py -c configs/deimv2_ssod/deimv2_dinov3_x_coco.yml --num-gpus 2
"""

import os
import sys
import argparse
import re
from datetime import datetime

import torch
import torch.multiprocessing as mp

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

# NCCL 环境变量设置（防止长时间训练超时）
os.environ.setdefault('NCCL_TIMEOUT', '10800')  # 3小时超时
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')


def main(args):
    """主函数"""
    # 设置日志
    if args.save_log and (not hasattr(args, 'local_rank') or args.local_rank == 0):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        log_file = os.path.join(log_dir, f'{config_name}_{timestamp}.log')
        
        class TeeLogger:
            def __init__(self, file_path):
                self.terminal = sys.stdout
                self.log_file = open(file_path, 'a', encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log_file.write(message)
                self.log_file.flush()
            
            def flush(self):
                self.terminal.flush()
                self.log_file.flush()
        
        tee_logger = TeeLogger(log_file)
        sys.stdout = tee_logger
        sys.stderr = tee_logger
        
        print(f"{'='*80}")
        print(f"Training log: {log_file}")
        print(f"Start time: {timestamp}")
        print(f"Config: {args.config}")
        print(f"{'='*80}\n")
    
    # 设置分布式环境
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    
    # 确保 resume 和 tuning 不能同时使用
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'
    
    # 解析 CLI 更新参数
    update_dict = yaml_utils.parse_cli(args.update)
    
    # 合并 CLI 参数
    update_dict.update({k: v for k, v in args.__dict__.items() 
                       if k not in ['update'] and v is not None})
    
    # 加载配置
    cfg = YAMLConfig(args.config, **update_dict)
    
    # 如果 resume 或 tuning，禁用预训练
    if cfg.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    print('cfg: ', cfg.__dict__)
    
    # 获取任务类型并创建 Solver
    task = cfg.yaml_cfg.get('task', 'detection')
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASKS.keys())}")
    
    solver = TASKS[task](cfg)
    
    # 运行训练或测试
    if args.test_only:
        solver.val()
    else:
        solver.fit()
    
    dist_utils.cleanup()


def _parse_port_from_url(url: str) -> str:
    """从 URL 解析端口号"""
    m = re.search(r':(\d+)$', url)
    return m.group(1) if m else '29500'


def _worker(local_rank: int, args):
    """分布式训练 worker"""
    # 设置基本环境变量
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', _parse_port_from_url(args.dist_url))
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(args.num_gpus)
    
    # 设置 NCCL 环境变量
    os.environ.setdefault('NCCL_TIMEOUT', '10800')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
    
    # CUDA 内存优化
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')
    
    args.local_rank = local_rank
    main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIM Semi-Supervised Training')
    
    # 配置文件
    parser.add_argument('-c', '--config', type=str, 
                        default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
                        help='Path to config file')
    
    # 训练参数
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, default=None,
                        help='Tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Device (auto-detect in distributed training)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--summary-dir', type=str, default=None,
                        help='Tensorboard summary directory')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Only run evaluation')
    
    # 更新参数
    parser.add_argument('-u', '--update', nargs='+', default=None,
                        help='Update yaml config')
    
    # 打印设置
    parser.add_argument('--print-method', type=str, default='builtin',
                        help='Print method')
    parser.add_argument('--print-rank', type=int, default=0,
                        help='Print rank id')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank id')
    
    # 分布式设置
    parser.add_argument('--num-gpus', type=int, default=3,
                        help='Number of GPUs')
    parser.add_argument('--gpu-ids', type=str, default='2,3,5',
                        help='Comma-separated GPU IDs (e.g., "0")')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:29500',
                        help='URL for distributed training')
    
    # 日志
    parser.add_argument('--save-log', action='store_true', default=True,
                        help='Save training log')
    
    args = parser.parse_args()
    
    # 设置可见 GPU
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # 启动训练
    if args.num_gpus > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        mp.spawn(_worker, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        main(args)
