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
import socket
import torch.multiprocessing as mp

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

# ✅ 增加 NCCL 超时时间（防止长时间训练超时）
os.environ.setdefault('NCCL_TIMEOUT', '10800')  # 秒
os.environ.setdefault('NCCL_DEBUG', 'WARN')  # 启用调试信息
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # 允许 NCCL 在通信异常时及时抛错

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
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
    
    # 🔥 NCCL 通信优化（针对双卡训练）
    os.environ.setdefault('NCCL_IB_DISABLE', '1')  # 禁用 InfiniBand（避免系统错误）
    os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # 启用 P2P 通信（GPU 间直接通信）
    os.environ.setdefault('NCCL_SHM_DISABLE', '0')  # 启用共享内存
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')  # 使用本地回环接口（单机多卡）
    
    # 🔥 CUDA内存优化（避免OOM和显存碎片）
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')
    
    # Propagate to argparse namespace
    args.local_rank = local_rank
    main(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='configs/deimv2/deimv2_dinov3_x_dji_optimized.yml')
    parser.add_argument('-r', '--resume', type=str, default='outputs/deimv2_dinov3_x_dji_800/best_stg2.pth', help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, default=None, help='device (auto-detect in distributed training)',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true',default=True, help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy', default='outputs/deimv2_dinov3_x_dji_800')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank id')
    
    # self-launch distributed (for running script directly without torchrun)
    parser.add_argument('--num-gpus', type=int, default=1, help='number of GPUs to use (spawn processes if >1)')
    parser.add_argument('--gpu-ids', type=str, default='1', help='comma-separated GPU ids, e.g. "1,2"')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:29500', help='url used to set up distributed training')
    
    args = parser.parse_args()

    # Optional: set visible devices from CLI
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"✅ 设置 CUDA_VISIBLE_DEVICES={args.gpu_ids}")

    if args.num_gpus and args.num_gpus > 1:
        # 检查端口是否可用
        import socket
        port = _parse_port_from_url(args.dist_url)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', int(port)))
            sock.close()
            print(f"✅ 端口 {port} 可用")
        except OSError:
            print(f"⚠️  警告: 端口 {port} 已被占用，尝试使用其他端口")
            # 尝试使用其他端口
            for p in range(29501, 29510):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(('127.0.0.1', p))
                    sock.close()
                    args.dist_url = f'tcp://127.0.0.1:{p}'
                    print(f"✅ 使用端口 {p}")
                    break
                except OSError:
                    continue
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        mp.spawn(_worker, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        # single process
        main(args)
