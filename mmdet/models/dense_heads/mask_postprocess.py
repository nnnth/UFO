import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def keep_largest_n_components(binary_mask, n=5):
    """
    保留二值掩码中面积最大的前n个连通区域。

    参数:
    binary_mask (numpy.ndarray): 输入的二值掩码，值为0或1。
    n (int): 要保留的最大连通区域数量。

    返回:
    numpy.ndarray: 只包含最大的前n个连通区域的二值掩码。
    """
    # 确保输入是布尔类型
    binary_mask = binary_mask.astype(bool)
    
    # 标记所有连通区域
    labeled_mask = measure.label(binary_mask, connectivity=2)  # 8连通
    
    # 计算每个连通区域的面积
    regions = measure.regionprops(labeled_mask)
    
    # 如果没有找到任何连通区域，返回全零的掩码
    if not regions:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    
    # 根据面积对区域进行排序（从大到小）
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
    
    # 获取前n个区域的标签
    top_n = sorted_regions[:n]
    top_n_labels = [region.label for region in top_n]
    
    # 创建只包含前n个连通区域的掩码
    largest_n_component_mask = np.isin(labeled_mask, top_n_labels).astype(np.uint8)
    
    return largest_n_component_mask

# 示例用法
if __name__ == "__main__":
    # 创建一个示例二值掩码
    binary_mask = np.zeros((300, 300), dtype=np.uint8)
    binary_mask[50:100, 50:100] = 1    # 第一个区域
    binary_mask[120:180, 120:180] = 1  # 第二个更大的区域
    binary_mask[200:250, 50:100] = 1    # 第三个区域
    binary_mask[50:80, 200:250] = 1     # 第四个区域
    binary_mask[220:270, 220:270] = 1   # 第五个区域
    binary_mask[10:30, 10:30] = 1       # 第六个小区域

    # 保留前5个最大的连通区域
    largest_5_mask = keep_largest_n_components(binary_mask, n=5)

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(binary_mask, cmap='gray')
    axes[0].set_title('原始掩码')
    axes[0].axis('off')

    axes[1].imshow(largest_5_mask, cmap='gray')
    axes[1].set_title('仅保留前5个最大区域')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
