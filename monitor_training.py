"""
训练监控脚本：实时检测 NaN/梯度爆炸
用法：在另一个终端运行此脚本监控训练日志
"""
import re
import sys
import time
from collections import deque

def monitor_log(log_file, window_size=10):
    """
    监控训练日志，检测异常模式
    
    Args:
        log_file: 日志文件路径（或使用 tail -f 输出）
        window_size: 滑动窗口大小，用于检测趋势
    """
    loss_history = deque(maxlen=window_size)
    lr_history = deque(maxlen=window_size)
    
    print("🔍 开始监控训练...")
    print("-" * 80)
    
    # 如果是从标准输入读取
    if log_file == '-':
        log_stream = sys.stdin
    else:
        log_stream = open(log_file, 'r')
    
    for line in log_stream:
        line = line.strip()
        
        # 检测 NaN/Inf 警告
        if 'NaN' in line or 'Inf' in line:
            print(f"⚠️  {time.strftime('%H:%M:%S')} - {line}")
        
        # 检测梯度跳过
        if 'Skipping' in line:
            print(f"❌ {time.strftime('%H:%M:%S')} - {line}")
        
        # 检测损失值异常
        if 'loss:' in line:
            # 提取损失值
            match = re.search(r'loss:\s+([\d.]+)', line)
            if match:
                loss = float(match.group(1))
                loss_history.append(loss)
                
                # 检测损失爆炸（相比平均值增加 10 倍）
                if len(loss_history) >= 5:
                    avg_loss = sum(loss_history) / len(loss_history)
                    if loss > avg_loss * 10:
                        print(f"🔥 {time.strftime('%H:%M:%S')} - 损失爆炸！当前: {loss:.4f}, 平均: {avg_loss:.4f}")
                
                # 检测损失为 0（可能跳过了太多 batch）
                if loss == 0.0:
                    print(f"⚠️  {time.strftime('%H:%M:%S')} - 损失为 0！")
        
        # 检测学习率变化
        if 'lr:' in line:
            match = re.search(r'lr:\s+([\d.e-]+)', line)
            if match:
                lr = float(match.group(1))
                lr_history.append(lr)
        
        # 检测正常的训练进度（每 500 步打印一次统计）
        if re.search(r'\[\s*\d+/\d+\]', line):
            # 提取步数
            match = re.search(r'\[\s*(\d+)/(\d+)\]', line)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                if current % 500 == 0:
                    print(f"✅ {time.strftime('%H:%M:%S')} - 进度: [{current}/{total}]")
                    if loss_history:
                        print(f"   平均损失: {sum(loss_history)/len(loss_history):.4f}")
                    if lr_history:
                        print(f"   当前学习率: {lr_history[-1]:.6f}")

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║              DEIM 训练监控脚本                                ║
║  用法：                                                       ║
║    1. 训练时重定向日志：python ssod_train.py 2>&1 | tee train.log ║
║    2. 运行监控：python monitor_training.py train.log         ║
║  或：                                                         ║
║    tail -f train.log | python monitor_training.py -          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        monitor_log(sys.argv[1])
    else:
        print("📝 从标准输入读取日志...")
        monitor_log('-')

