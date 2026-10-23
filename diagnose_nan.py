#!/usr/bin/env python3
"""
🔍 NaN 诊断工具
用于检查模型和预训练权重中的 NaN 值
"""

import torch
import sys

print("=" * 60)
print("🔍 检查预训练权重")
print("=" * 60)

# 1. 检查预训练权重
ckpt_path = "./ckpts/vitt_distill.pt"
print(f"\n📂 加载权重: {ckpt_path}")

try:
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    print(f"✅ 权重加载成功，包含 {len(state_dict)} 个参数")
    
    # 检查每个参数
    nan_params = []
    inf_params = []
    zero_params = []
    
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
            
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            total_count = param.numel()
            nan_params.append((name, nan_count, total_count))
            print(f"❌ NaN found in {name}: {nan_count}/{total_count} ({100*nan_count/total_count:.2f}%)")
        
        if torch.isinf(param).any():
            inf_count = torch.isinf(param).sum().item()
            total_count = param.numel()
            inf_params.append((name, inf_count, total_count))
            print(f"⚠️  Inf found in {name}: {inf_count}/{total_count}")
        
        if (param == 0).all():
            zero_params.append(name)
    
    print("\n" + "=" * 60)
    print("📊 诊断结果")
    print("=" * 60)
    
    if nan_params:
        print(f"\n❌ 发现 {len(nan_params)} 个参数包含 NaN:")
        for name, count, total in nan_params:
            print(f"   - {name}: {count}/{total}")
        print("\n💡 建议：重新下载预训练权重或使用其他权重文件")
        sys.exit(1)
    
    if inf_params:
        print(f"\n⚠️  发现 {len(inf_params)} 个参数包含 Inf:")
        for name, count, total in inf_params:
            print(f"   - {name}: {count}/{total}")
        print("\n💡 建议：预训练权重可能不稳定")
    
    if zero_params:
        print(f"\n⚠️  发现 {len(zero_params)} 个参数全为 0:")
        for name in zero_params[:5]:  # 只显示前 5 个
            print(f"   - {name}")
        if len(zero_params) > 5:
            print(f"   ... 还有 {len(zero_params) - 5} 个")
    
    if not nan_params and not inf_params:
        print("\n✅ 预训练权重检查通过！")
        print("   - 无 NaN 值")
        print("   - 无 Inf 值")
        
        # 显示一些统计信息
        print("\n📈 权重统计:")
        for name, param in list(state_dict.items())[:5]:
            if isinstance(param, torch.Tensor):
                print(f"   - {name}:")
                print(f"     mean={param.mean().item():.6f}, "
                      f"std={param.std().item():.6f}, "
                      f"min={param.min().item():.6f}, "
                      f"max={param.max().item():.6f}")
        
        print("\n🎯 结论：预训练权重正常，NaN 问题可能来自：")
        print("   1. 模型初始化代码")
        print("   2. 前向传播过程中的数值不稳定")
        print("   3. 学习率或优化器设置")

except Exception as e:
    print(f"❌ 加载权重时出错: {e}")
    print(f"\n💡 可能的原因:")
    print("   - 权重文件损坏")
    print("   - 文件格式不正确")
    print("   - 路径错误")
    sys.exit(1)

print("\n" + "=" * 60)

