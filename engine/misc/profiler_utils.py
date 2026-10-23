"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

try:
    # 可选依赖：用于统计 FLOPs/MACs。缺失时不应影响训练/推理/可视化脚本导入。
    from calflops import calculate_flops  # type: ignore
except Exception:  # noqa: BLE001
    calculate_flops = None  # type: ignore

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    """
    计算模型的 FLOPs、MACs 和参数数量。
    如果 FLOPs 计算失败（某些模型结构不兼容），则跳过并只返回参数数量。
    """
    try:
        if calculate_flops is None:
            raise ImportError("calflops not installed")
        base_size = cfg.train_dataloader.collate_fn.base_size
        input_shape = (1, 3, base_size, base_size)

        model_for_info = copy.deepcopy(cfg.model).deploy()

        flops, macs, _ = calculate_flops(model=model_for_info,
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4,
                                            print_detailed=False)
        params = sum(p.numel() for p in model_for_info.parameters())
        del model_for_info

        return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
    
    except Exception as e:
        # FLOPs 计算失败时（如模型结构不兼容 calflops），跳过并只统计参数
        print(f"[Warning] FLOPs calculation failed: {type(e).__name__}: {e}")
        print("[Warning] Skipping FLOPs, only counting parameters...")
        try:
            params = sum(p.numel() for p in cfg.model.parameters())
            return params, {"Model Params: %d (FLOPs calculation skipped)" % params}
        except Exception:
            return 0, {"Model stats unavailable"}
