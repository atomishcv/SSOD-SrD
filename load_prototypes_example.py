"""
语义原型加载与使用示例
演示如何正确加载和使用提取的语义原型
"""

import torch
import torch.nn.functional as F
from pathlib import Path


def load_semantic_prototypes(prototype_path='semantic_prototypes.pth', device='cpu'):
    """
    加载语义原型文件
    
    Args:
        prototype_path: 原型文件路径
        device: 加载到的设备
    
    Returns:
        dict: 包含原型和统计信息的字典
    """
    print(f"📂 正在加载: {prototype_path}")
    
    # 加载文件
    data = torch.load(prototype_path, map_location=device)
    
    print("\n✅ 加载成功！包含以下内容：")
    print(f"  - prototypes: {data['prototypes'].shape}")  # [num_classes, 256]
    print(f"  - counts: {data['counts'].shape}")  # [num_classes]
    print(f"  - stds: {data['stds'].shape}")  # [num_classes, 256]
    print(f"  - similarities: {len(data['similarities'])} 个类别的统计信息")
    print(f"  - config: {data['config']}")
    
    return data


def load_all_features(all_features_path='semantic_prototypes_all_features.pth', device='cpu'):
    """
    加载所有原始特征（可选，用于高级分析）
    
    Args:
        all_features_path: 所有特征文件路径
        device: 加载到的设备
    
    Returns:
        dict: {class_id: Tensor[N, 256]} 每个类别的所有特征
    """
    print(f"📂 正在加载: {all_features_path}")
    
    all_features = torch.load(all_features_path, map_location=device)
    
    print("\n✅ 加载成功！")
    print(f"  共 {len(all_features)} 个类别")
    for class_id, features in all_features.items():
        print(f"  类别 {class_id:2d}: {features.shape[0]:4d} 个特征, 形状 {features.shape}")
    
    return all_features


def compute_similarity_to_prototypes(query_features, prototypes, normalize=True):
    """
    计算查询特征与各个类别原型的相似度
    
    Args:
        query_features: [N, 256] 查询特征
        prototypes: [num_classes, 256] 类别原型
        normalize: 是否先进行 L2 归一化（如果特征已归一化则不需要）
    
    Returns:
        similarities: [N, num_classes] 相似度矩阵
    """
    if normalize:
        query_features = F.normalize(query_features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
    
    # 余弦相似度: [N, 256] @ [256, num_classes] = [N, num_classes]
    similarities = query_features @ prototypes.T
    
    return similarities


def filter_by_similarity(query_features, query_labels, prototypes, similarity_thresh=0.7):
    """
    根据与原型的相似度过滤查询
    
    Args:
        query_features: [N, 256] 查询特征
        query_labels: [N] 预测的类别标签
        prototypes: [num_classes, 256] 类别原型
        similarity_thresh: 相似度阈值
    
    Returns:
        valid_mask: [N] bool tensor，表示是否通过过滤
        similarities: [N] 与对应类别原型的相似度
    """
    # 计算与所有原型的相似度
    all_similarities = compute_similarity_to_prototypes(query_features, prototypes)
    
    # 提取每个查询与其预测类别原型的相似度
    # 方法1: 使用 gather
    query_class_similarities = all_similarities.gather(
        dim=1, 
        index=query_labels.unsqueeze(1)
    ).squeeze(1)  # [N]
    
    # 方法2: 使用索引（等价但更直观）
    # query_class_similarities = all_similarities[torch.arange(len(query_labels)), query_labels]
    
    # 过滤：相似度 >= 阈值
    valid_mask = query_class_similarities >= similarity_thresh
    
    return valid_mask, query_class_similarities


def analyze_class_diversity(prototypes_data):
    """
    分析各个类别的多样性
    
    Args:
        prototypes_data: 从 torch.load() 加载的原型数据
    """
    prototypes = prototypes_data['prototypes']
    stds = prototypes_data['stds']
    counts = prototypes_data['counts']
    similarities = prototypes_data['similarities']
    
    print("\n" + "="*80)
    print("📊 类别多样性分析")
    print("="*80)
    print(f"{'类别':>4} | {'样本数':>6} | {'平均Std':>9} | {'类内相似度':>11} | {'建议'}")
    print("-"*80)
    
    high_diversity_classes = []
    
    for class_id in range(len(prototypes)):
        if counts[class_id] == 0:
            continue
        
        # 计算平均标准差
        avg_std = stds[class_id].mean().item()
        
        # 获取类内相似度
        if class_id in similarities:
            intra_sim = similarities[class_id]['mean']
        else:
            intra_sim = -1
        
        # 判断是否需要多原型
        if avg_std > 0.2 or (intra_sim > 0 and intra_sim < 0.8):
            suggestion = "⚠️ 多原型"
            high_diversity_classes.append(class_id)
        else:
            suggestion = "✅ 单原型"
        
        print(f"{class_id:>4} | {counts[class_id]:>6} | {avg_std:>9.4f} | {intra_sim:>11.4f} | {suggestion}")
    
    print("-"*80)
    print(f"\n高多样性类别（建议使用多原型）: {len(high_diversity_classes)} 个")
    if high_diversity_classes:
        print(f"  类别ID: {high_diversity_classes}")


# ============================================================================
# 使用示例
# ============================================================================

def example_1_basic_loading():
    """示例1: 基本加载"""
    print("\n" + "="*80)
    print("示例1: 基本加载语义原型")
    print("="*80)
    
    # 加载原型
    prototypes_data = load_semantic_prototypes('semantic_prototypes.pth', device='cpu')
    
    # 提取各个字段
    prototypes = prototypes_data['prototypes']  # [80, 256]
    counts = prototypes_data['counts']  # [80]
    stds = prototypes_data['stds']  # [80, 256]
    config = prototypes_data['config']
    
    print(f"\n📊 统计信息：")
    print(f"  总类别数: {len(prototypes)}")
    print(f"  非空类别数: {(counts > 0).sum().item()}")
    print(f"  总特征数: {counts.sum().item()}")
    
    return prototypes_data


def example_2_compute_similarity():
    """示例2: 计算与原型的相似度"""
    print("\n" + "="*80)
    print("示例2: 计算查询特征与原型的相似度")
    print("="*80)
    
    # 加载原型
    prototypes_data = load_semantic_prototypes('semantic_prototypes.pth', device='cpu')
    prototypes = prototypes_data['prototypes']  # [80, 256]
    
    # 模拟一些查询特征（假设从模型推理得到）
    num_queries = 300
    query_features = torch.randn(num_queries, 256)
    query_features = F.normalize(query_features, dim=-1)  # L2 归一化
    
    # 计算相似度
    similarities = compute_similarity_to_prototypes(query_features, prototypes, normalize=False)
    # 形状: [300, 80]
    
    print(f"\n查询特征: {query_features.shape}")
    print(f"原型: {prototypes.shape}")
    print(f"相似度矩阵: {similarities.shape}")
    print(f"相似度范围: [{similarities.min():.4f}, {similarities.max():.4f}]")
    
    # 找到每个查询最相似的类别
    max_similarities, best_classes = similarities.max(dim=1)
    
    print(f"\n前10个查询的最相似类别:")
    for i in range(min(10, num_queries)):
        print(f"  查询 {i}: 类别 {best_classes[i]:2d}, 相似度 {max_similarities[i]:.4f}")
    
    return similarities


def example_3_filter_pseudo_labels():
    """示例3: 过滤伪标签（基于原型相似度）"""
    print("\n" + "="*80)
    print("示例3: 使用原型相似度过滤伪标签")
    print("="*80)
    
    # 加载原型
    prototypes_data = load_semantic_prototypes('semantic_prototypes.pth', device='cpu')
    prototypes = prototypes_data['prototypes']  # [80, 256]
    
    # 模拟模型的预测结果
    num_detections = 100
    pred_features = torch.randn(num_detections, 256)
    pred_features = F.normalize(pred_features, dim=-1)
    pred_labels = torch.randint(0, 80, (num_detections,))
    pred_scores = torch.rand(num_detections)
    
    print(f"原始检测数: {num_detections}")
    
    # 过滤1: 置信度阈值
    score_thresh = 0.5
    score_mask = pred_scores >= score_thresh
    print(f"置信度过滤后: {score_mask.sum().item()} (阈值={score_thresh})")
    
    # 过滤2: 原型相似度
    similarity_thresh = 0.7
    valid_mask, similarities = filter_by_similarity(
        pred_features[score_mask], 
        pred_labels[score_mask], 
        prototypes,
        similarity_thresh=similarity_thresh
    )
    print(f"原型相似度过滤后: {valid_mask.sum().item()} (阈值={similarity_thresh})")
    
    # 最终的伪标签
    final_mask = score_mask.clone()
    final_mask[score_mask] = valid_mask
    
    print(f"最终保留: {final_mask.sum().item()}")
    print(f"过滤率: {(1 - final_mask.sum().item() / num_detections) * 100:.1f}%")
    
    # 打印一些示例
    print(f"\n示例（前10个通过置信度过滤的检测）:")
    score_filtered_indices = torch.where(score_mask)[0][:10]
    for i, idx in enumerate(score_filtered_indices):
        if i < len(similarities):
            status = "✅ 通过" if valid_mask[i] else "❌ 过滤"
            print(f"  检测 {idx.item():2d}: 类别 {pred_labels[idx]:2d}, "
                  f"置信度 {pred_scores[idx]:.4f}, "
                  f"相似度 {similarities[i]:.4f} {status}")


def example_4_multi_prototype():
    """示例4: 使用所有特征进行多原型建模"""
    print("\n" + "="*80)
    print("示例4: 多原型建模（处理类内多样性）")
    print("="*80)
    
    # 加载所有特征
    all_features = load_all_features('semantic_prototypes_all_features.pth', device='cpu')
    
    # 选择一个类别进行聚类（例如类别0：person）
    class_id = 0
    if class_id in all_features:
        class_features = all_features[class_id]  # [N, 256]
        
        print(f"\n类别 {class_id} 的特征: {class_features.shape}")
        
        # 使用 K-means 聚类生成多个原型
        from sklearn.cluster import KMeans
        
        n_prototypes = 3  # 每个类别3个原型
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(class_features.numpy())
        
        # 获取聚类中心作为多个原型
        multi_prototypes = torch.from_numpy(kmeans.cluster_centers_).float()  # [3, 256]
        multi_prototypes = F.normalize(multi_prototypes, dim=-1)
        
        print(f"生成 {n_prototypes} 个原型: {multi_prototypes.shape}")
        
        # 分析每个子类别
        for i in range(n_prototypes):
            cluster_features = class_features[labels == i]
            if len(cluster_features) > 0:
                # 计算类内相似度
                cluster_features_norm = F.normalize(cluster_features, dim=-1)
                similarities = cluster_features_norm @ multi_prototypes[i].unsqueeze(1)
                print(f"  子原型 {i}: {len(cluster_features):4d} 个样本, "
                      f"相似度 {similarities.mean():.4f}±{similarities.std():.4f}")
        
        # 使用多原型进行匹配
        print(f"\n使用多原型匹配（取最大相似度）:")
        query = torch.randn(1, 256)
        query = F.normalize(query, dim=-1)
        
        # 单原型
        single_prototype = class_features.mean(dim=0, keepdim=True)
        single_prototype = F.normalize(single_prototype, dim=-1)
        single_sim = (query @ single_prototype.T).item()
        
        # 多原型
        multi_sim = (query @ multi_prototypes.T).max().item()
        
        print(f"  单原型相似度: {single_sim:.4f}")
        print(f"  多原型相似度: {multi_sim:.4f}")


def main():
    """运行所有示例"""
    
    # 检查文件是否存在
    if not Path('semantic_prototypes.pth').exists():
        print("❌ 错误: 找不到 semantic_prototypes.pth")
        print("   请先运行 extract_semantic_prototypes.py 生成原型文件")
        return
    
    # 示例1: 基本加载
    prototypes_data = example_1_basic_loading()
    
    # 分析类别多样性
    analyze_class_diversity(prototypes_data)
    
    # 示例2: 计算相似度
    example_2_compute_similarity()
    
    # 示例3: 过滤伪标签
    example_3_filter_pseudo_labels()
    
    # 示例4: 多原型（需要 all_features 文件）
    if Path('semantic_prototypes_all_features.pth').exists():
        example_4_multi_prototype()
    else:
        print("\n⚠️  未找到 semantic_prototypes_all_features.pth，跳过示例4")
    
    print("\n" + "="*80)
    print("✅ 所有示例完成！")
    print("="*80)


if __name__ == '__main__':
    main()

