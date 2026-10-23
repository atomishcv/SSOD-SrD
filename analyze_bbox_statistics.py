"""
Analyze bounding box size distribution in COCO format annotation file
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


def load_annotations(json_path):
    """
    Load COCO format annotation file
    
    Args:
        json_path: Path to JSON annotation file
        
    Returns:
        annotations: List of annotation dictionaries
        images: List of image dictionaries
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    images = data.get('images', [])
    
    return annotations, images


def extract_bbox_statistics(annotations):
    """
    Extract bounding box statistics
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        stats_dict: Dictionary containing width, height, area, aspect_ratio arrays
    """
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            if h > 0:
                aspect_ratios.append(w / h)
            else:
                aspect_ratios.append(0)
    
    return {
        'widths': np.array(widths),
        'heights': np.array(heights),
        'areas': np.array(areas),
        'aspect_ratios': np.array(aspect_ratios)
    }


def print_statistics(stats_dict):
    """
    Print detailed statistics
    
    Args:
        stats_dict: Dictionary containing bbox statistics
    """
    print("\n" + "="*80)
    print("Bounding Box Size Statistics")
    print("="*80)
    
    print(f"\nTotal number of bounding boxes: {len(stats_dict['widths'])}")
    
    metrics = {
        'Width': stats_dict['widths'],
        'Height': stats_dict['heights'],
        'Area': stats_dict['areas'],
        'Aspect Ratio (W/H)': stats_dict['aspect_ratios']
    }
    
    for name, values in metrics.items():
        if len(values) > 0:
            print(f"\n{name}:")
            print(f"  Min: {np.min(values):.2f}")
            print(f"  Max: {np.max(values):.2f}")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Median: {np.median(values):.2f}")
            print(f"  Std: {np.std(values):.2f}")
            print(f"  25th Percentile: {np.percentile(values, 25):.2f}")
            print(f"  75th Percentile: {np.percentile(values, 75):.2f}")
            print(f"  90th Percentile: {np.percentile(values, 90):.2f}")
            print(f"  95th Percentile: {np.percentile(values, 95):.2f}")
            print(f"  99th Percentile: {np.percentile(values, 99):.2f}")


def visualize_distributions(stats_dict, save_path=None):
    """
    Visualize bounding box size distributions
    
    Args:
        stats_dict: Dictionary containing bbox statistics
        save_path: Path to save the visualization (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Width distribution
    axes[0, 0].hist(stats_dict['widths'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Width Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height distribution
    axes[0, 1].hist(stats_dict['heights'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Height Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Area distribution (log scale)
    areas = stats_dict['areas']
    axes[0, 2].hist(areas, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 2].set_title('Area Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Area (pixels²)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Aspect ratio distribution
    aspect_ratios = stats_dict['aspect_ratios']
    axes[1, 0].hist(aspect_ratios, bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[1, 0].set_title('Aspect Ratio Distribution (W/H)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Aspect Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Width vs Height scatter plot
    axes[1, 1].scatter(stats_dict['widths'], stats_dict['heights'], 
                       alpha=0.3, s=1, c='blue')
    axes[1, 1].set_title('Width vs Height Scatter', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Area distribution (log scale for better visualization)
    axes[1, 2].hist(np.log10(areas + 1), bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 2].set_title('Area Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Log10(Area + 1)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close()


def analyze_size_categories(stats_dict):
    """
    Categorize bounding boxes by size
    
    Args:
        stats_dict: Dictionary containing bbox statistics
        
    Returns:
        categories: Dictionary with size category counts
    """
    areas = stats_dict['areas']
    widths = stats_dict['widths']
    heights = stats_dict['heights']
    
    # Define size categories based on area
    small_threshold = np.percentile(areas, 33)
    medium_threshold = np.percentile(areas, 67)
    
    small_boxes = areas < small_threshold
    medium_boxes = (areas >= small_threshold) & (areas < medium_threshold)
    large_boxes = areas >= medium_threshold
    
    categories = {
        'Small': np.sum(small_boxes),
        'Medium': np.sum(medium_boxes),
        'Large': np.sum(large_boxes)
    }
    
    # Also categorize by absolute size
    # Common definitions: small < 32x32, large > 96x96
    small_abs = (widths < 32) & (heights < 32)
    large_abs = (widths > 96) | (heights > 96)
    medium_abs = ~(small_abs | large_abs)
    
    categories_abs = {
        'Small (< 32x32)': np.sum(small_abs),
        'Medium (32x32 - 96x96)': np.sum(medium_abs),
        'Large (> 96x96)': np.sum(large_abs)
    }
    
    return categories, categories_abs


def visualize_size_categories(categories, categories_abs, save_path=None):
    """
    Visualize size category distribution
    
    Args:
        categories: Relative size categories
        categories_abs: Absolute size categories
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Relative size categories
    labels1 = list(categories.keys())
    values1 = list(categories.values())
    axes[0].bar(labels1, values1, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0].set_title('Size Categories (Relative - Percentile-based)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(values1):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Absolute size categories
    labels2 = list(categories_abs.keys())
    values2 = list(categories_abs.values())
    axes[1].bar(labels2, values2, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[1].set_title('Size Categories (Absolute - Pixel-based)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(values2):
        axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Category visualization saved to: {save_path}")
    
    plt.close()


def print_category_statistics(categories, categories_abs):
    """
    Print category statistics
    
    Args:
        categories: Relative size categories
        categories_abs: Absolute size categories
    """
    print("\n" + "="*80)
    print("Size Category Statistics")
    print("="*80)
    
    total = sum(categories.values())
    print("\nRelative Size Categories (Percentile-based):")
    for name, count in categories.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {name}: {count} ({percentage:.2f}%)")
    
    total_abs = sum(categories_abs.values())
    print("\nAbsolute Size Categories (Pixel-based):")
    for name, count in categories_abs.items():
        percentage = (count / total_abs) * 100 if total_abs > 0 else 0
        print(f"  {name}: {count} ({percentage:.2f}%)")


def categorize_by_size(annotations, size_method='min_edge', threshold_method='percentile75'):
    """
    Categorize annotations into 'small' and 'big' based on pixel size (width/height)
    
    Args:
        annotations: List of annotation dictionaries
        size_method: Method to determine size
            - 'min_edge': use minimum of width and height (default)
            - 'max_edge': use maximum of width and height
            - 'avg_edge': use average of width and height
            - 'width': use width only
            - 'height': use height only
        threshold_method: Method to determine threshold 
            - 'percentile75': 75th percentile (default, ~25% big)
            - 'percentile80': 80th percentile (~20% big)
            - 'percentile90': 90th percentile (~10% big)
            - 'median': 50th percentile (50% big)
            - 'mean': mean value
            - or a number: direct threshold value in pixels
        
    Returns:
        updated_annotations: Annotations with 'size_category' field added
        threshold: The size threshold used (in pixels)
        stats: Statistics about the categorization
    """
    # Calculate sizes based on chosen method
    sizes = []
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if size_method == 'min_edge':
                size = min(w, h)
            elif size_method == 'max_edge':
                size = max(w, h)
            elif size_method == 'avg_edge':
                size = (w + h) / 2
            elif size_method == 'width':
                size = w
            elif size_method == 'height':
                size = h
            else:
                raise ValueError(f"Unknown size_method: {size_method}")
            sizes.append(size)
        else:
            sizes.append(0)
    
    sizes = np.array(sizes)
    
    # Determine threshold
    if threshold_method == 'percentile75':
        threshold = np.percentile(sizes, 75)
    elif threshold_method == 'percentile80':
        threshold = np.percentile(sizes, 80)
    elif threshold_method == 'percentile90':
        threshold = np.percentile(sizes, 90)
    elif threshold_method == 'median':
        threshold = np.median(sizes)
    elif threshold_method == 'mean':
        threshold = np.mean(sizes)
    elif isinstance(threshold_method, (int, float)):
        threshold = threshold_method
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    # Categorize annotations
    updated_annotations = []
    small_count = 0
    big_count = 0
    
    for i, ann in enumerate(annotations):
        ann_copy = ann.copy()
        if sizes[i] < threshold:
            ann_copy['size_category'] = 'small'
            small_count += 1
        else:
            ann_copy['size_category'] = 'big'
            big_count += 1
        updated_annotations.append(ann_copy)
    
    stats = {
        'threshold': threshold,
        'size_method': size_method,
        'small_count': small_count,
        'big_count': big_count,
        'small_percentage': (small_count / len(annotations)) * 100,
        'big_percentage': (big_count / len(annotations)) * 100
    }
    
    return updated_annotations, threshold, stats


def categorize_by_area(annotations, threshold_method='percentile75'):
    """
    Categorize annotations into 'small' and 'big' based on area
    
    Args:
        annotations: List of annotation dictionaries
        threshold_method: Method to determine threshold 
            - 'percentile75': 75th percentile (default, ~25% big)
            - 'percentile80': 80th percentile (~20% big)
            - 'percentile90': 90th percentile (~10% big)
            - 'median': 50th percentile (50% big)
            - 'mean': mean value
            - or a number: direct threshold value
        
    Returns:
        updated_annotations: Annotations with 'size_category' field added
        threshold: The area threshold used
        stats: Statistics about the categorization
    """
    # Calculate areas
    areas = []
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            areas.append(w * h)
        else:
            areas.append(0)
    
    areas = np.array(areas)
    
    # Determine threshold
    if threshold_method == 'percentile75':
        threshold = np.percentile(areas, 75)
    elif threshold_method == 'percentile80':
        threshold = np.percentile(areas, 80)
    elif threshold_method == 'percentile90':
        threshold = np.percentile(areas, 90)
    elif threshold_method == 'median':
        threshold = np.median(areas)
    elif threshold_method == 'mean':
        threshold = np.mean(areas)
    elif isinstance(threshold_method, (int, float)):
        threshold = threshold_method
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    # Categorize annotations
    updated_annotations = []
    small_count = 0
    big_count = 0
    
    for i, ann in enumerate(annotations):
        ann_copy = ann.copy()
        if areas[i] < threshold:
            ann_copy['size_category'] = 'small'
            small_count += 1
        else:
            ann_copy['size_category'] = 'big'
            big_count += 1
        updated_annotations.append(ann_copy)
    
    stats = {
        'threshold': threshold,
        'size_method': 'area',
        'small_count': small_count,
        'big_count': big_count,
        'small_percentage': (small_count / len(annotations)) * 100,
        'big_percentage': (big_count / len(annotations)) * 100
    }
    
    return updated_annotations, threshold, stats


def save_categorized_annotations(json_path, annotations, images, output_path, threshold, stats, threshold_method='percentile75', size_method='min_edge'):
    """
    Save annotations with size categories to a new JSON file
    
    Args:
        json_path: Original JSON file path (to copy structure)
        annotations: Updated annotations with size_category
        images: Images list
        output_path: Path to save the new JSON file
        threshold: Size threshold used (in pixels)
        stats: Statistics about categorization
        threshold_method: Method used to determine threshold
        size_method: Method used to determine size (min_edge, max_edge, etc.)
    """
    # Load original file to preserve structure
    with open(json_path, 'r') as f:
        original_data = json.load(f)
    
    # Update with categorized annotations
    new_data = original_data.copy()
    new_data['annotations'] = annotations
    new_data['images'] = images
    
    # Add metadata about categorization
    if 'info' not in new_data:
        new_data['info'] = {}
    
    # Determine method type based on size_method
    if size_method == 'area':
        method_type = 'area_based'
        threshold_unit = 'pixels²'
    else:
        method_type = 'pixel_size_based'
        threshold_unit = 'pixels'
    
    new_data['info']['size_categorization'] = {
        'method': method_type,
        'size_method': size_method,
        'threshold_method': threshold_method,
        'threshold': float(threshold),
        'small_count': int(stats['small_count']),
        'big_count': int(stats['big_count']),
        'small_percentage': float(stats['small_percentage']),
        'big_percentage': float(stats['big_percentage'])
    }
    
    # Save to new file
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"\nCategorized annotations saved to: {output_path}")
    print(f"  Method: {method_type}")
    print(f"  Threshold: {threshold:.2f} {threshold_unit}")
    print(f"  Small boxes: {stats['small_count']} ({stats['small_percentage']:.2f}%)")
    print(f"  Big boxes: {stats['big_count']} ({stats['big_percentage']:.2f}%)")


def main():
    """Main function"""
    json_path = "/data/DEIMv2/DJI_dataset/annotations/train_full.json"
    output_dir = Path("/data/DEIMv2") / "bbox_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading annotations from: {json_path}")
    
    # Load annotations
    annotations, images = load_annotations(json_path)
    print(f"Loaded {len(annotations)} annotations from {len(images)} images")
    
    # Extract statistics
    print("\nExtracting bounding box statistics...")
    stats_dict = extract_bbox_statistics(annotations)
    
    # Print statistics
    print_statistics(stats_dict)
    
    # Analyze size categories
    categories, categories_abs = analyze_size_categories(stats_dict)
    print_category_statistics(categories, categories_abs)
    
    # Visualize distributions
    print("\nGenerating visualizations...")
    dist_save_path = output_dir / "bbox_distributions.png"
    visualize_distributions(stats_dict, save_path=str(dist_save_path))
    
    # Visualize categories
    cat_save_path = output_dir / "bbox_categories.png"
    visualize_size_categories(categories, categories_abs, save_path=str(cat_save_path))
    
    # Categorize by area (small/big)
    print("\n" + "="*80)
    print("Categorizing annotations by area (small/big)")
    print("="*80)
    
    # Use 75th percentile of area as threshold (so big boxes are ~25% of total)
    # Based on the area distribution, this gives a good split
    updated_annotations, threshold, stats = categorize_by_area(
        annotations, threshold_method='percentile75'
    )
    
    print(f"\nCategorization Results:")
    print(f"  Threshold (75th percentile area): {threshold:.2f} pixels²")
    print(f"  Small boxes: {stats['small_count']} ({stats['small_percentage']:.2f}%)")
    print(f"  Big boxes: {stats['big_count']} ({stats['big_percentage']:.2f}%)")
    
    # Save categorized annotations
    output_json_path = Path("/data/DEIMv2/DJI_dataset/annotations/train_full_categorized.json")
    save_categorized_annotations(
        json_path, updated_annotations, images, 
        output_json_path, threshold, stats, 
        threshold_method='percentile75', size_method='area'
    )
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()

