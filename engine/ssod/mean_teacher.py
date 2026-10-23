"""
Mean Teacher EMA 更新机制（参考 Semi-DETR）
"""

import torch
import torch.nn as nn


class MeanTeacher:
    """
    Mean Teacher EMA 更新（参考 Semi-DETR MeanTeacher Hook）
    
    通过指数移动平均（EMA）更新教师模型权重：
    teacher = momentum * teacher + (1 - momentum) * student
    
    特点：
    1. 支持动量预热：早期使用较小的动量，让教师更快跟随学生
    2. Semi-DETR 风格的动量计算：
       momentum = min(target_momentum, 1 - (1 + warm_up) / (step + 1 + warm_up))
    """
    
    def __init__(self, 
                 teacher: nn.Module,
                 student: nn.Module,
                 momentum: float = 0.999,
                 warm_up: int = 100):
        """
        Args:
            teacher: 教师模型
            student: 学生模型
            momentum: 目标 EMA 动量（默认 0.999）
            warm_up: 动量预热步数（默认 100）
        """
        self.teacher = teacher
        self.student = student
        self.momentum = momentum
        self.warm_up = warm_up
        
        # 冻结教师模型
        self._freeze_teacher()
        
        # 初始化：将学生权重复制到教师
        self._copy_student_to_teacher()
    
    def _freeze_teacher(self):
        """冻结教师模型"""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def _copy_student_to_teacher(self):
        """将学生权重复制到教师"""
        for (_, tgt_param), (_, src_param) in zip(
            self.teacher.named_parameters(),
            self.student.named_parameters()
        ):
            tgt_param.data.copy_(src_param.data)
    
    def get_momentum(self, curr_step: int) -> float:
        """
        计算当前步的 EMA 动量（Semi-DETR 风格）
        
        Args:
            curr_step: 当前训练步数
            
        Returns:
            momentum: 当前 EMA 动量
        """
        # Semi-DETR 公式：momentum = min(target_momentum, 1 - (1 + warm_up) / (step + 1 + warm_up))
        momentum = min(
            self.momentum,
            1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        return max(0.0, momentum)
    
    def update(self, curr_step: int):
        """
        EMA 更新教师模型
        
        Args:
            curr_step: 当前训练步数
        """
        momentum = self.get_momentum(curr_step)
        
        with torch.no_grad():
            for (_, tgt_param), (_, src_param) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()
            ):
                tgt_param.data.mul_(momentum).add_(src_param.data, alpha=1 - momentum)
    
    @property
    def module(self):
        """返回教师模型（兼容 ModelEMA 接口）"""
        return self.teacher

