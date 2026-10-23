"""
基于类别原型的伪标签质量评估器

使用预训练模型提取的类别原型特征，通过余弦相似度评估教师模型生成的伪标签质量
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PrototypeFilter:
    """
    基于类别原型的伪标签过滤器
    
    工作流程：
    1. 加载预训练时提取的类别原型
    2. 对教师模型生成的每个伪标签，提取其特征
    3. 计算特征与对应类别原型的余弦相似度
    4. 相似度低于阈值的伪标签被过滤掉
    """
    
    def __init__(self, 
                 prototype_path: str,
                 similarity_threshold: float = 0.7,
                 device: str = 'cuda'):
        """
        Args:
            prototype_path: 类别原型文件路径
            similarity_threshold: 余弦相似度阈值（0-1），低于此值的伪标签被过滤
            device: 设备
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # 加载原型
        print(f"📦 加载类别原型: {prototype_path}")
        prototype_data = torch.load(prototype_path, map_location='cpu')
        
        self.prototypes = prototype_data['prototypes']  # {class_id: feature}
        self.stats = prototype_data['stats']  # {class_id: {'count', 'std'}}
        
        # 将原型移到设备
        for class_id in self.prototypes:
            self.prototypes[class_id] = self.prototypes[class_id].to(device)
        
        print(f"✅ 加载完成: {len(self.prototypes)} 个类别原型")
        print(f"   - 特征维度: {prototype_data.get('feature_dim', 'unknown')}")
        print(f"   - 相似度阈值: {similarity_threshold}")
    
    @torch.no_grad()
    def extract_features_from_pseudo_labels(self,
                                           encoder_features: torch.Tensor,
                                           pseudo_boxes: torch.Tensor,
                                           pseudo_labels: torch.Tensor) -> torch.Tensor:
        """
        从伪标签中提取特征
        
        Args:
            encoder_features: Encoder 输出特征 (B, C, H, W) 或 (C, H, W)
            pseudo_boxes: 伪标签框 (N, 4) [cx, cy, w, h] normalized
            pseudo_labels: 伪标签类别 (N,)
        
        Returns:
            features: (N, C) 每个伪标签的特征
        """
        if encoder_features.dim() == 4:
            # (B, C, H, W) -> 假设只有一张图
            encoder_features = encoder_features[0]
        
        C, H, W = encoder_features.shape
        N = len(pseudo_boxes)
        
        if N == 0:
            return torch.zeros((0, C), device=self.device)
        
        features = []
        
        for box in pseudo_boxes:
            # 将归一化坐标转换为特征图坐标
            cx, cy, bw, bh = box.tolist()
            x1 = max(0, int((cx - bw/2) * W))
            y1 = max(0, int((cy - bh/2) * H))
            x2 = min(W, int((cx + bw/2) * W))
            y2 = min(H, int((cy + bh/2) * H))
            
            if x2 <= x1 or y2 <= y1:
                # 无效框，使用零特征
                features.append(torch.zeros(C, device=self.device))
                continue
            
            # 提取 RoI 特征
            roi_feat = encoder_features[:, y1:y2, x1:x2]  # (C, h, w)
            
            # 自适应平均池化
            roi_feat = F.adaptive_avg_pool2d(roi_feat.unsqueeze(0), (1, 1))  # (1, C, 1, 1)
            roi_feat = roi_feat.squeeze()  # (C,)
            
            # L2 归一化
            roi_feat = F.normalize(roi_feat, p=2, dim=0)
            
            features.append(roi_feat)
        
        return torch.stack(features)  # (N, C)
    
    @torch.no_grad()
    def compute_similarities(self,
                            features: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """
        计算特征与对应类别原型的余弦相似度
        
        Args:
            features: (N, C) 伪标签特征
            labels: (N,) 伪标签类别
        
        Returns:
            similarities: (N,) 余弦相似度
        """
        N = len(features)
        similarities = torch.zeros(N, device=self.device)
        
        for idx, label in enumerate(labels):
            label_id = int(label.item())
            
            # 检查是否有对应的原型
            if label_id not in self.prototypes:
                # 没有原型，设为低相似度
                similarities[idx] = 0.0
                continue
            
            # 获取原型
            prototype = self.prototypes[label_id]  # (C,)
            
            # 计算余弦相似度
            feat = features[idx]  # (C,)
            similarity = torch.dot(feat, prototype).item()  # 两个 L2 归一化向量的点积
            
            similarities[idx] = similarity
        
        return similarities
    
    @torch.no_grad()
    def filter_pseudo_labels(self,
                            encoder_features: torch.Tensor,
                            pseudo_boxes: torch.Tensor,
                            pseudo_labels: torch.Tensor,
                            pseudo_scores: Optional[torch.Tensor] = None,
                            return_similarities: bool = False) -> Tuple:
        """
        过滤伪标签
        
        Args:
            encoder_features: Encoder 输出特征
            pseudo_boxes: 伪标签框 (N, 4)
            pseudo_labels: 伪标签类别 (N,)
            pseudo_scores: 伪标签分数 (N,) 可选
            return_similarities: 是否返回相似度
        
        Returns:
            filtered_boxes: 过滤后的框
            filtered_labels: 过滤后的标签
            filtered_scores: 过滤后的分数（如果提供）
            similarities: 相似度（如果 return_similarities=True）
        """
        if len(pseudo_boxes) == 0:
            if return_similarities:
                return pseudo_boxes, pseudo_labels, pseudo_scores, torch.tensor([])
            else:
                return pseudo_boxes, pseudo_labels, pseudo_scores
        
        # 提取特征
        features = self.extract_features_from_pseudo_labels(
            encoder_features, pseudo_boxes, pseudo_labels
        )
        
        # 计算相似度
        similarities = self.compute_similarities(features, pseudo_labels)
        
        # 过滤
        mask = similarities >= self.similarity_threshold
        
        filtered_boxes = pseudo_boxes[mask]
        filtered_labels = pseudo_labels[mask]
        filtered_scores = pseudo_scores[mask] if pseudo_scores is not None else None
        
        if return_similarities:
            return filtered_boxes, filtered_labels, filtered_scores, similarities[mask]
        else:
            return filtered_boxes, filtered_labels, filtered_scores
    
    def get_statistics(self) -> Dict:
        """获取原型统计信息"""
        return {
            'num_classes': len(self.prototypes),
            'feature_dim': next(iter(self.prototypes.values())).shape[0] if self.prototypes else 0,
            'class_stats': self.stats,
            'similarity_threshold': self.similarity_threshold,
        }
    
    def __repr__(self):
        return (f"PrototypeFilter("
                f"classes={len(self.prototypes)}, "
                f"threshold={self.similarity_threshold}, "
                f"device={self.device})")


def load_prototype_filter(prototype_path: str, 
                          similarity_threshold: float = 0.7,
                          device: str = 'cuda') -> Optional[PrototypeFilter]:
    """
    便捷函数：加载原型过滤器
    
    Args:
        prototype_path: 原型文件路径
        similarity_threshold: 相似度阈值
        device: 设备
    
    Returns:
        PrototypeFilter 或 None（如果文件不存在）
    """
    if not Path(prototype_path).exists():
        print(f"⚠️  原型文件不存在: {prototype_path}")
        return None
    
    try:
        return PrototypeFilter(prototype_path, similarity_threshold, device)
    except Exception as e:
        print(f"❌ 加载原型过滤器失败: {e}")
        return None


# ========== 使用示例 ==========

if __name__ == '__main__':
    """
    使用示例
    """
    # 1. 加载原型过滤器
    filter = load_prototype_filter(
        prototype_path='outputs/deimv2_dinov3_x_coco/class_prototypes.pth',
        similarity_threshold=0.7,
        device='cuda'
    )
    
    if filter is None:
        print("无法加载原型过滤器")
        exit(1)
    
    # 2. 模拟伪标签
    encoder_features = torch.randn(1, 256, 20, 20).cuda()  # (B, C, H, W)
    pseudo_boxes = torch.tensor([
        [0.5, 0.5, 0.3, 0.4],  # [cx, cy, w, h] normalized
        [0.2, 0.3, 0.1, 0.2],
        [0.7, 0.8, 0.2, 0.3],
    ]).cuda()
    pseudo_labels = torch.tensor([0, 1, 2]).cuda()  # person, bicycle, car
    pseudo_scores = torch.tensor([0.9, 0.8, 0.85]).cuda()
    
    # 3. 过滤伪标签
    print("\n开始过滤伪标签...")
    filtered_boxes, filtered_labels, filtered_scores, similarities = filter.filter_pseudo_labels(
        encoder_features=encoder_features[0],  # (C, H, W)
        pseudo_boxes=pseudo_boxes,
        pseudo_labels=pseudo_labels,
        pseudo_scores=pseudo_scores,
        return_similarities=True
    )
    
    # 4. 打印结果
    print(f"\n原始伪标签数: {len(pseudo_boxes)}")
    print(f"过滤后伪标签数: {len(filtered_boxes)}")
    print(f"过滤比例: {(1 - len(filtered_boxes)/len(pseudo_boxes))*100:.1f}%")
    print(f"\n相似度:")
    for idx, (label, score, sim) in enumerate(zip(pseudo_labels, pseudo_scores, similarities)):
        print(f"  [{idx}] Class {label.item()}, Score {score.item():.3f}, Similarity {sim.item():.3f}")


