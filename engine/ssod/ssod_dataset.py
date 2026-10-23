"""
半监督数据集和数据管道（参考 Semi-DETR）

核心组件：
1. SemiDataset：组合有标签和无标签数据集
2. MultiBranchPipeline：对无标签数据生成弱增广和强增广两个视图
3. SemiBalanceSampler：按比例采样有标签和无标签数据
"""

import copy
import random
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF

from pycocotools.coco import COCO


# ============================================================================
# 几何变换记录（参考 Semi-DETR GeometricTransformationBase）
# ============================================================================

class GeometricTransformationBase:
    """
    记录几何变换的变换矩阵，用于坐标空间对齐（参考 Semi-DETR）
    
    支持的变换：
    - scale: 缩放
    - flip: 水平/垂直翻转
    - shift: 平移
    - rotate: 旋转
    """
    
    @classmethod
    def get_identity_matrix(cls):
        """返回单位矩阵"""
        return np.eye(3, dtype=np.float32)
    
    @classmethod
    def get_scale_matrix(cls, sx, sy):
        """缩放矩阵"""
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
    
    @classmethod
    def get_flip_matrix(cls, img_w, direction='horizontal'):
        """翻转矩阵"""
        if direction == 'horizontal':
            return np.array([[-1, 0, img_w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        else:  # vertical
            return np.array([[1, 0, 0], [0, -1, img_w], [0, 0, 1]], dtype=np.float32)
    
    @classmethod
    def get_shift_matrix(cls, dx, dy):
        """平移矩阵"""
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
    
    @classmethod
    def get_rotate_matrix(cls, degree, cx=0, cy=0):
        """旋转矩阵（绕中心点）"""
        rad = degree * np.pi / 180
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        # 先平移到原点，旋转，再平移回去
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        return T2 @ R @ T1
    
    @classmethod
    def compose(cls, *matrices):
        """组合多个变换矩阵"""
        result = np.eye(3, dtype=np.float32)
        for m in matrices:
            result = m @ result
        return result


GTrans = GeometricTransformationBase


# ============================================================================
# 数据增强管道（参考 Semi-DETR）
# ============================================================================

class WeakAugmentation:
    """
    弱增广（用于 Teacher）
    
    仅包含：
    - 随机缩放
    - 随机水平翻转
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (800, 800),
        flip_prob: float = 0.5,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.size = size
        self.flip_prob = flip_prob
        self.mean = mean
        self.std = std
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            image: PIL Image
            target: 包含 'boxes' (xyxy, 像素坐标) 和 'labels' 的字典
            
        Returns:
            image: 归一化后的张量 [3, H, W]
            target: 更新后的目标字典，包含 meta['transform_matrix']
        """
        orig_w, orig_h = image.size
        
        # 初始化变换矩阵
        transform_matrix = GTrans.get_identity_matrix()
        
        # 1. 缩放到目标尺寸
        new_h, new_w = self.size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        image = image.resize((new_w, new_h), Image.BILINEAR)
        transform_matrix = GTrans.get_scale_matrix(scale_x, scale_y) @ transform_matrix
        
        # 更新边界框
        if 'boxes' in target and target['boxes'].shape[0] > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
        
        # 2. 随机水平翻转
        flip = random.random() < self.flip_prob
        if flip:
            image = TF.hflip(image)
            transform_matrix = GTrans.get_flip_matrix(new_w) @ transform_matrix
            
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = new_w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        # 3. 转换为张量并归一化
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        
        # 4. 记录元信息
        target['meta'] = {
            'transform_matrix': transform_matrix,
            'scale_factor': (scale_x, scale_y),
            'flip': flip,
            'orig_size': (orig_h, orig_w),
            'img_size': (new_h, new_w),
        }
        target['tag'] = 'unsup_teacher'
        
        return image, target


class StrongAugmentation:
    """
    强增广（用于 Student）
    
    包含：
    - 随机缩放
    - 随机水平翻转
    - 颜色抖动（ColorJitter）
    - 随机灰度
    - 随机高斯模糊
    - 随机擦除（Cutout）
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (800, 800),
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        grayscale_prob: float = 0.2,
        blur_prob: float = 0.5,
        cutout_prob: float = 0.5,
        cutout_ratio: float = 0.3,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.size = size
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio
        self.mean = mean
        self.std = std
        
        from torchvision import transforms
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            image: PIL Image
            target: 包含 'boxes' (xyxy, 像素坐标) 和 'labels' 的字典
            
        Returns:
            image: 归一化后的张量 [3, H, W]
            target: 更新后的目标字典，包含 meta['transform_matrix']
        """
        orig_w, orig_h = image.size
        
        # 初始化变换矩阵
        transform_matrix = GTrans.get_identity_matrix()
        
        # 1. 缩放到目标尺寸
        new_h, new_w = self.size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        image = image.resize((new_w, new_h), Image.BILINEAR)
        transform_matrix = GTrans.get_scale_matrix(scale_x, scale_y) @ transform_matrix
        
        # 更新边界框
        if 'boxes' in target and target['boxes'].shape[0] > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
        
        # 2. 随机水平翻转
        flip = random.random() < self.flip_prob
        if flip:
            image = TF.hflip(image)
            transform_matrix = GTrans.get_flip_matrix(new_w) @ transform_matrix
            
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = new_w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        # 3. 颜色抖动
        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)
        
        # 4. 随机灰度
        if random.random() < self.grayscale_prob:
            image = TF.to_grayscale(image, num_output_channels=3)
        
        # 5. 随机高斯模糊
        if random.random() < self.blur_prob:
            sigma = random.uniform(0.1, 2.0)
            image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)
        
        # 6. 转换为张量
        image = TF.to_tensor(image)
        
        # 7. 随机擦除（Cutout）- 在张量上操作
        if random.random() < self.cutout_prob:
            _, h, w = image.shape
            cut_h = int(h * self.cutout_ratio)
            cut_w = int(w * self.cutout_ratio)
            y = random.randint(0, h - cut_h)
            x = random.randint(0, w - cut_w)
            image[:, y:y+cut_h, x:x+cut_w] = 0
        
        # 8. 归一化
        image = TF.normalize(image, self.mean, self.std)
        
        # 9. 记录元信息
        target['meta'] = {
            'transform_matrix': transform_matrix,
            'scale_factor': (scale_x, scale_y),
            'flip': flip,
            'orig_size': (orig_h, orig_w),
            'img_size': (new_h, new_w),
        }
        target['tag'] = 'unsup_student'
        
        return image, target


class MultiBranchPipeline:
    """
    多分支数据管道（参考 Semi-DETR MultiBranch）
    
    对同一张图像生成弱增广和强增广两个版本
    """
    
    def __init__(
        self,
        weak_aug: Callable = None,
        strong_aug: Callable = None,
        size: Tuple[int, int] = (800, 800),
    ):
        self.weak_aug = weak_aug or WeakAugmentation(size=size)
        self.strong_aug = strong_aug or StrongAugmentation(size=size)
    
    def __call__(self, image: Image.Image, target: Dict) -> Dict:
        """
        Args:
            image: PIL Image
            target: 原始目标字典
            
        Returns:
            result: 包含 'weak' 和 'strong' 两个分支的字典
        """
        # 深拷贝 target，避免相互影响
        target_weak = copy.deepcopy(target)
        target_strong = copy.deepcopy(target)
        
        # 应用弱增广
        image_weak, target_weak = self.weak_aug(image.copy(), target_weak)
        
        # 应用强增广
        image_strong, target_strong = self.strong_aug(image.copy(), target_strong)
        
        return {
            'weak': (image_weak, target_weak),
            'strong': (image_strong, target_strong),
        }


# ============================================================================
# 半监督数据集（参考 Semi-DETR SemiDataset）
# ============================================================================

class CocoUnlabeledDataset(Dataset):
    """
    COCO 格式的无标签数据集
    
    返回多分支（弱增广 + 强增广）数据
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        pipeline: MultiBranchPipeline = None,
        size: Tuple[int, int] = (800, 800),
    ):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.pipeline = pipeline or MultiBranchPipeline(size=size)
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # 加载图像
        img_path = f"{self.img_folder}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        
        # 无标签数据，创建空的 target
        target = {
            'boxes': torch.zeros(0, 4),
            'labels': torch.zeros(0, dtype=torch.long),
            'image_id': img_id,
        }
        
        # 应用多分支管道
        result = self.pipeline(image, target)
        
        return result


class CocoLabeledDataset(Dataset):
    """
    COCO 格式的有标签数据集
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Callable = None,
        size: Tuple[int, int] = (800, 800),
    ):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        
        # 过滤没有标注的图像
        self.img_ids = [
            img_id for img_id in self.img_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        
        self.transforms = transforms
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # 类别映射
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # 加载图像
        img_path = f"{self.img_folder}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(self.cat2label[ann['category_id']])
        
        if len(boxes) == 0:
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
        }
        
        # 应用变换
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # 默认变换
            new_h, new_w = self.size
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            image = image.resize((new_w, new_h), Image.BILINEAR)
            
            if target['boxes'].shape[0] > 0:
                target['boxes'][:, [0, 2]] *= scale_x
                target['boxes'][:, [1, 3]] *= scale_y
            
            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)
        
        # 转换为 cxcywh 归一化坐标
        if target['boxes'].shape[0] > 0:
            H, W = self.size
            boxes = target['boxes']
            # xyxy -> cxcywh
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / H
            w = (boxes[:, 2] - boxes[:, 0]) / W
            h = (boxes[:, 3] - boxes[:, 1]) / H
            target['boxes'] = torch.stack([cx, cy, w, h], dim=-1)
        
        target['tag'] = 'sup'
        
        return image, target


class SemiDataset(Dataset):
    """
    半监督数据集（参考 Semi-DETR SemiDataset）
    
    组合有标签和无标签数据集
    """
    
    def __init__(
        self,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        sample_ratio: Tuple[int, int] = (1, 4),
    ):
        """
        Args:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            sample_ratio: 采样比例 (labeled, unlabeled)，例如 (1, 4) 表示每 5 个样本中 1 个有标签
        """
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.sample_ratio = sample_ratio
        
        # 计算总长度
        ratio_sum = sum(sample_ratio)
        labeled_per_epoch = len(labeled_dataset)
        self._length = labeled_per_epoch * ratio_sum // sample_ratio[0]
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        ratio_sum = sum(self.sample_ratio)
        
        if idx % ratio_sum < self.sample_ratio[0]:
            # 有标签数据
            labeled_idx = (idx // ratio_sum) % len(self.labeled_dataset)
            return {'sup': self.labeled_dataset[labeled_idx]}
        else:
            # 无标签数据
            unlabeled_idx = idx % len(self.unlabeled_dataset)
            unsup_data = self.unlabeled_dataset[unlabeled_idx]
            return {'unsup': unsup_data}


# ============================================================================
# 采样器（参考 Semi-DETR SemiBalanceSampler）
# ============================================================================

class SemiBalanceSampler(Sampler):
    """
    半监督平衡采样器（参考 Semi-DETR）
    
    确保每个 batch 中有标签和无标签数据的比例固定
    """
    
    def __init__(
        self,
        dataset: SemiDataset,
        batch_size: int,
        sample_ratio: Tuple[int, int] = (1, 4),
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            indices = torch.randperm(len(indices), generator=g).tolist()
        
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ============================================================================
# Collate 函数
# ============================================================================

def semi_collate_fn(batch: List[Dict]) -> Dict:
    """
    半监督数据 collate 函数
    
    将 batch 中的数据按类型分组
    """
    sup_images = []
    sup_targets = []
    unsup_weak_images = []
    unsup_weak_targets = []
    unsup_strong_images = []
    unsup_strong_targets = []
    
    for item in batch:
        if 'sup' in item:
            image, target = item['sup']
            sup_images.append(image)
            sup_targets.append(target)
        
        if 'unsup' in item:
            unsup_data = item['unsup']
            weak_image, weak_target = unsup_data['weak']
            strong_image, strong_target = unsup_data['strong']
            unsup_weak_images.append(weak_image)
            unsup_weak_targets.append(weak_target)
            unsup_strong_images.append(strong_image)
            unsup_strong_targets.append(strong_target)
    
    result = {}
    
    if sup_images:
        result['sup'] = (torch.stack(sup_images), sup_targets)
    
    if unsup_weak_images:
        result['unsup_weak'] = (torch.stack(unsup_weak_images), unsup_weak_targets)
        result['unsup_strong'] = (torch.stack(unsup_strong_images), unsup_strong_targets)
    
    return result

