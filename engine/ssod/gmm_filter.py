"""
GMM 伪标签筛选器（参考 Semi-DETR）

使用高斯混合模型（GMM）动态估计匹配代价阈值，筛选高质量伪标签
"""

import torch
import numpy as np

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


class GMMFilter:
    """
    GMM 伪标签筛选器（参考 Semi-DETR）
    
    原理：
    - 低匹配代价 = 高质量伪标签（预测与伪标签匹配好）
    - 高匹配代价 = 低质量伪标签（预测与伪标签匹配差）
    - GMM 拟合两个高斯分量（低代价 + 高代价），取低代价分量的边界作为阈值
    """
    
    def __init__(self, covariance_type='diag'):
        """
        Args:
            covariance_type: GMM 协方差类型 ('diag', 'full', 'spherical', 'tied')
        """
        if skm is None:
            raise ImportError('Please run "pip install sklearn" to install sklearn first.')
        
        self.covariance_type = covariance_type
    
    def fit(self, data_points: torch.Tensor, device=None) -> float:
        """
        拟合 GMM 模型，返回代价阈值
        
        Args:
            data_points: 匹配代价数据 [N]
            device: 设备（用于返回值转换）
            
        Returns:
            cost_thr: 代价阈值（低于此值被认为是高质量伪标签）
        """
        if len(data_points) == 0:
            return 0.0
        
        if len(data_points) < 2:
            return data_points[0].item() if isinstance(data_points, torch.Tensor) else data_points[0]
        
        # 转换为 numpy
        if isinstance(data_points, torch.Tensor):
            data_points = data_points.detach().cpu()
        
        # 排序（从低到高）
        pos_cost_gmm, _ = data_points.sort()
        pos_cost_gmm = pos_cost_gmm.view(-1, 1).numpy()
        
        min_cost, max_cost = pos_cost_gmm.min(), pos_cost_gmm.max()
        
        # 初始化 GMM 参数
        means_init = np.array([min_cost, max_cost]).reshape(2, 1)
        weights_init = np.array([0.5, 0.5])
        precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)
        
        if self.covariance_type == 'spherical':
            precisions_init = precisions_init.reshape(2)
        elif self.covariance_type == 'diag':
            precisions_init = precisions_init.reshape(2, 1)
        elif self.covariance_type == 'tied':
            precisions_init = np.array([[1.0]])
        
        # 拟合 GMM
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            covariance_type=self.covariance_type,
            reg_covar=1e-5
        )
        
        try:
            gmm.fit(pos_cost_gmm)
        except Exception:
            # 拟合失败，返回中位数作为阈值
            return float(np.median(pos_cost_gmm))
        
        gmm_assignment = gmm.predict(pos_cost_gmm)
        scores = gmm.score_samples(pos_cost_gmm)
        gmm_assignment = torch.from_numpy(gmm_assignment)
        scores = torch.from_numpy(scores)
        
        # 取低代价分量（分量 0）的最大代价作为阈值
        pseudo_mask = gmm_assignment == 0
        cost_thr = None
        
        if pseudo_mask.nonzero().numel():
            _, pseudo_thr_ind = scores[pseudo_mask].topk(1)
            cost_thr = pos_cost_gmm[pseudo_mask.numpy()][pseudo_thr_ind.item()]
        
        if cost_thr is not None:
            return float(cost_thr)
        else:
            # 如果低代价分量为空，使用高代价分量
            pseudo_mask = gmm_assignment == 1
            if pseudo_mask.nonzero().numel():
                _, pseudo_thr_ind = scores[pseudo_mask].topk(1)
                cost_thr = pos_cost_gmm[pseudo_mask.numpy()][pseudo_thr_ind.item()]
                return float(cost_thr)
            else:
                return float(np.median(pos_cost_gmm))


def compute_matching_cost(
    cls_scores: torch.Tensor,
    bbox_preds: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_bboxes: torch.Tensor,
    img_size: tuple,
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
):
    """
    计算匹配代价矩阵（用于 GMM 筛选）
    
    Args:
        cls_scores: 分类分数 [num_queries, num_classes]
        bbox_preds: 边界框预测 [num_queries, 4]，cxcywh 格式，归一化
        gt_labels: GT 标签 [num_gt]
        gt_bboxes: GT 边界框 [num_gt, 4]，xyxy 格式，像素坐标
        img_size: 图像尺寸 (H, W)
        cost_class: 分类代价权重
        cost_bbox: L1 代价权重
        cost_giou: GIoU 代价权重
        
    Returns:
        cost: 代价矩阵 [num_queries, num_gt]
    """
    from ..deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
    
    num_queries = cls_scores.shape[0]
    num_gt = gt_labels.shape[0]
    
    if num_queries == 0 or num_gt == 0:
        return cls_scores.new_zeros(num_queries, num_gt)
    
    H, W = img_size
    factor = gt_bboxes.new_tensor([W, H, W, H]).unsqueeze(0)
    
    # 分类代价（Focal Loss 风格）
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (cls_scores ** gamma) * (-(1 - cls_scores + 1e-8).log())
    pos_cost_class = alpha * ((1 - cls_scores) ** gamma) * (-(cls_scores + 1e-8).log())
    cost_cls = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]
    
    # L1 代价
    normalize_gt_bboxes = gt_bboxes / factor  # 归一化
    # 将 xyxy 转为 cxcywh
    x1, y1, x2, y2 = normalize_gt_bboxes.unbind(-1)
    gt_bboxes_cxcywh = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
    cost_l1 = torch.cdist(bbox_preds, gt_bboxes_cxcywh, p=1)
    
    # GIoU 代价
    pred_bboxes_xyxy = box_cxcywh_to_xyxy(bbox_preds) * factor
    cost_giou_mat = -generalized_box_iou(pred_bboxes_xyxy, gt_bboxes)
    
    # 总代价
    cost = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_mat
    
    return cost

