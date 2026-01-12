# make sample with size of 14*14*200
import argparse
import os
import json
import h5py
import numpy as np
from scipy import sparse
from tqdm import tqdm
import anndata as ad
from scipy import stats
import random
from functools import lru_cache
from typing import List, Tuple, Dict, Any
from scipy.sparse import csr_matrix 

# find high-variance genes and save
def find_hv_genes(input_dir, depth,hv_genes_path):
    """优化后的高变基因选择函数，核心改进：
    1. 使用内存映射代替全量加载矩阵
    2. 增量统计替代全数据存储
    3. 分块处理减少内存峰值
    4. 向量化计算提升效率
    """
    print('Finding high-variance genes...')
    # 阶段1: 全局基因ID收集 (低内存消耗)
    print('1: Collecting all gene IDs...')
    all_genes = set()
    for fname in os.listdir(input_dir):
        if not fname.endswith('.h5'):
            continue
        with h5py.File(os.path.join(input_dir, fname), 'r') as f:
            for group in f.values():
                if 'vocab_index' not in group:
                    continue
                # 高效处理字节字符串转换
                vocab = group['vocab_index'][:]
                all_genes.update(v.decode() if isinstance(v, bytes) else str(v) for v in vocab)

    # 阶段2: 增量式统计计算
    print('2: Computing gene statistics...')
    gene_stats = {gene: [0, 0.0, 0.0] for gene in all_genes}  # [count, sum, sum_sq]
    chunk_size = 2000  # 优化内存的关键参数
    
    for fname in os.listdir(input_dir):
        if not fname.endswith('.h5'):
            continue
            
        with h5py.File(os.path.join(input_dir, fname), 'r') as f:
            for group in f.values():
                # 跳过无效数据集
                if 'expr_map' not in group or 'vocab_index' not in group:
                    continue
                    
                ds = group['expr_map']
                vocab = group['vocab_index'][:]
                gene_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in vocab]
                
                n_genes = ds.shape[1]
                
                # 分块处理大数据集
                for start in range(0, n_genes, chunk_size):
                    end = min(start + chunk_size, n_genes)
                    # 使用memory-map避免全量加载
                    chunk = ds[:, start:end]
                    
                    # 向量化计算统计量
                    sums = chunk.sum(axis=0)
                    sum_sqs = (chunk**2).sum(axis=0)
                    counts = np.full(chunk.shape[1], chunk.shape[0])
                    
                    # 增量更新统计值
                    for i in range(chunk.shape[1]):
                        gene = gene_ids[start + i]
                        cnt, s, sq = gene_stats[gene]
                        cnt_new = cnt + counts[i]
                        s_new = s + sums[i]
                        sq_new = sq + sum_sqs[i]
                        gene_stats[gene] = [cnt_new, s_new, sq_new]

    # 阶段3: 高效计算变异系数
    print('3: Computing gene CVs...')
    epsilon = 1e-7  # 防止除零
    cv_values = []
    
    for gene, (n, s, sq) in gene_stats.items():
        if n == 0:
            cv = 0.0
        else:
            mean = s / n
            # 使用数值稳定公式计算方差
            variance = max(0.0, (sq - s**2 / n) / (n - 1)) if n > 1 else 0.0
            cv = np.sqrt(variance) / (mean + epsilon)
        cv_values.append((gene, cv))
    
    # 阶段4: 基于变异系数排序取TopK
    print(f'4: Top {depth} high-variance genes:')
    cv_values.sort(key=lambda x: x[1], reverse=True)
    hv_genes_id = [gene for gene, _ in cv_values[:depth]]
    # 阶段5: 保存Top K 基因ID
    print('5: Saving Top K gene IDs...')
    output_path = hv_genes_path
    with open(output_path, 'w') as f:
        json.dump(hv_genes_id, f, indent=4)
    return hv_genes_id

# make h5 file into samples
'''
def make_samples(input_dir, output_dir, seed=0, height=14, width=14, depth=200,min_spot=10, pad_id=0, n_samples_multiplier=2):
    """
    处理空间转录组数据，创建随机采样的空间区域样本，保存为H5AD格式
    
    修改亮点:
    1. 添加pad_id参数: 当样本基因数量不足depth时，用pad_id填充
    2. 每个样本独立计算高变基因，然后加权随机选择depth个基因
    3. 随机采样空间区域而非整体分割
    4. 增加采样次数(原本区块数*2)
    
    参数:
    input_dir: 包含.h5输入文件的目录
    output_dir: 输出样本的目录
    seed: 随机种子(默认0)
    height: 每个样本的y轴高度(默认14)
    width: 每个样本的x轴宽度(默认14)
    depth: 使用的基因数量(默认200)
    min_spot: 样本包含的最小spot数(默认10)
    pad_id: 用于填充不足基因的标识符(默认"PAD")
    n_samples_multiplier: 采样次数倍数(默认2)
    
    # Spot级别元数据
    adata.obs = {
        'original_index': 原始索引,
        'x_abs': 绝对X坐标,
        'y_abs': 绝对Y坐标,
        'x_rel': 样本内相对X坐标,
        'y_rel': 样本内相对Y坐标
    }

    # 基因级别元数据
    adata.var = {
        'gene_ids': 基因ID列表(含pad_id填充),
        'is_selected': 基因是否被选择的布尔向量
    }

    # 样本元数据
    adata.uns['region'] = {
        'x_start': 区域起始X,
        'y_start': 区域起始Y,
        'width': 区域宽度,
        'height': 区域高度
    }
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有h5文件
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    if not h5_files:
        print("⚠️ 警告: 输入目录中没有找到.h5文件")
        return
    
    print(f"🔍 找到 {len(h5_files)} 个.h5文件，开始处理...")
    print(f"⚙️ 配置参数: height={height}, width={width}, depth={depth}, min_spot={min_spot}")
    print(f"🎲 采样策略: 随机采样{height}x{width}区域")
    print(f"💾 输出格式: H5AD (AnnData格式)")
    
    # 统计变量
    total_samples = 0
    skipped_files = 0
    skipped_groups = 0
    skipped_blocks = 0
    
    # 进度条：文件处理
    file_pbar = tqdm(h5_files, desc="文件处理")
    
    for file_name in file_pbar:
        file_pbar.set_postfix(file=file_name)
        file_path = os.path.join(input_dir, file_name)
        
        try:
            with h5py.File(file_path, 'r') as h5_file:
                groups = list(h5_file.keys())
                if not groups:
                    skipped_files += 1
                    print(f"⚠️ 跳过: 文件 {file_name} 中没有找到任何组")
                    continue
                
                # 组处理进度
                group_pbar = tqdm(groups, desc=f"样本处理", leave=False)
                
                for group_name in group_pbar:
                    group_pbar.set_postfix(group=group_name)
                    
                    try:
                        group = h5_file[group_name]
                        
                        # 提取数据
                        coords = group['coords_map'][:]
                        expr = group['expr_map'][:]
                        vocab = group['vocab_index'][:]
                        
                        # 处理基因ID类型
                        if isinstance(vocab[0], bytes):
                            vocab = np.array([g.decode('utf-8') for g in vocab])
                        elif np.issubdtype(vocab.dtype, np.integer):
                            vocab = np.array([str(g) for g in vocab])
                        
                        n_spots, n_genes = expr.shape
                        
                        # 计算每个基因的整体方差（用于权重）
                        overall_var = np.var(expr, axis=0)
                        
                        # 确定空间范围
                        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                        
                        # 计算原本的分块数量（用于确定采样次数）
                        x_blocks = int(np.ceil((x_max - x_min + 1) / width))
                        y_blocks = int(np.ceil((y_max - y_min + 1) / height))
                        n_samples = x_blocks * y_blocks * n_samples_multiplier
                        
                        sample_count = 1
                        spots_added = 0
                        
                        # 区块处理进度
                        block_pbar = tqdm(
                            total=n_samples, 
                            desc=f"随机采样", 
                            leave=False
                        )
                        
                        # 随机采样空间区块
                        for _ in range(n_samples):
                            block_pbar.update(1)
                            
                            # 随机选择起始点
                            x_start = random.randint(int(x_min), max(int(x_min), int(x_max) - width))
                            y_start = random.randint(int(y_min), max(int(y_min), int(y_max) - height))
                            x_end = x_start + width
                            y_end = y_start + height
                            
                            # 选择区块内的spots
                            in_block = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) & \
                                       (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
                            
                            # 跳过spot不足的区块
                            num_spots = np.sum(in_block)
                            if num_spots < min_spot:
                                skipped_blocks += 1
                                continue
                            
                            block_coords = coords[in_block]
                            block_expr = expr[in_block]
                            rel_coords = block_coords - np.array([x_start, y_start])
                            
                            # === 选择高变基因 ===
                            
                            # 计算区域内每个基因的方差
                            block_var = np.var(block_expr, axis=0)
                            
                            # 结合全局方差和区域方差的权重
                            alpha = 1.0  # 区域方差权重
                            beta = 0   # 全局方差权重
                            gene_weights = alpha * block_var + beta * overall_var
                            
                            # 标准化权重并添加小常数避免零权重
                            gene_weights = (gene_weights - np.min(gene_weights)) / \
                                          (np.max(gene_weights) - np.min(gene_weights) + 1e-8) + 1e-8
                            
                            # 加权随机选择depth个基因（无重复）
                            selected_indices = random.choices(
                                range(n_genes), 
                                weights=gene_weights, 
                                k=min(depth, n_genes)
                            )
                            
                            # 如果需要，使用pad_id填充
                            selected_gene_ids = []
                            selected_expr = np.zeros((num_spots, depth))
                            
                            # 复制选中的基因表达值
                            for i, idx in enumerate(selected_indices):
                                selected_gene_ids.append(vocab[idx])
                                selected_expr[:, i] = block_expr[:, idx]
                            
                            # 填充不足的基因
                            if len(selected_indices) < depth:
                                fill_count = depth - len(selected_indices)
                                selected_gene_ids.extend([pad_id] * fill_count)
                            
                            # 创建是否被选中的标记
                            is_selected = [True] * len(selected_indices) + [False] * (depth - len(selected_indices))
                            
                            # 创建AnnData对象
                            adata = ad.AnnData(
                                X=selected_expr,
                                obs={
                                    'original_index': np.where(in_block)[0],
                                    'x_abs': block_coords[:, 0],
                                    'y_abs': block_coords[:, 1],
                                    'x_rel': rel_coords[:, 0],
                                    'y_rel': rel_coords[:, 1]
                                },
                                var={
                                    'gene_ids': selected_gene_ids,
                                    'is_selected': is_selected
                                }
                            )
                            
                            # 添加空间信息到obsm
                            adata.obsm['coords_map'] = block_coords
                            adata.obsm['coords_sample'] = rel_coords
                            
                            # 添加额外元数据
                            adata.uns['region'] = {
                                'x_start': x_start,
                                'y_start': y_start,
                                'width': width,
                                'height': height
                            }
                            
                            adata.uns['gene_selection'] = {
                                'method': 'weighted_random',
                                'alpha': alpha,
                                'beta': beta
                            }
                            
                            # 保存为H5AD
                            output_name = f"{group_name}_sample_{sample_count}.h5ad"
                            output_path = os.path.join(output_dir, output_name)
                            adata.write(output_path)
                            
                            sample_count += 1
                            total_samples += 1
                            spots_added += num_spots
                        
                        block_pbar.close()
                        group_pbar.set_postfix(spots=f"{spots_added} spots")
                    
                    except Exception as e:
                        skipped_groups += 1
                        print(f"⚠️ 跳过组 {group_name}: {str(e)}")
                
                group_pbar.close()
        
        except Exception as e:
            skipped_files += 1
            print(f"🚨 错误处理文件 {file_name}: {str(e)}")
    
    # 处理完成后的统计信息
    print(f"\n✅ 处理完成!")
    print(f"📊 统计:")
    print(f"  创建样本: {total_samples}个")
    print(f"  采样尝试: {x_blocks * y_blocks * n_samples_multiplier}次")
    print(f"  跳过文件: {skipped_files}个")
    print(f"  跳过样本组: {skipped_groups}个")
    print(f"  跳过区块: {skipped_blocks}个 (spot<{min_spot})")
    
    if total_samples == 0:
        print("❌ 未创建任何样本，请检查输入数据和参数配置")
    else:
        print(f"💾 输出目录: {output_dir} (H5AD格式)")
        print(f"🔬 每个样本包含 {depth} 个基因 (使用 {pad_id} 填充不足)")

def make_samples_h5(input_dir, output_dir, seed=0, height=14, width=14, depth=200,
                 min_spot=10, pad_id=0, n_samples_multiplier=2):
    """
    处理空间转录组数据，以细胞坐标为中心创建空间区域样本，保存为H5AD格式
    
    修改亮点:
    1. 添加pad_id参数: 当样本基因数量不足depth时，用pad_id填充
    2. 每个样本独立计算高变基因，然后加权随机选择depth个基因
    3. 以细胞坐标为中心采样空间区域而非随机区域
    4. 采样次数为细胞数量的两倍
    
    参数:
    input_dir: 包含.h5输入文件的目录
    output_dir: 输出样本的目录
    seed: 随机种子(默认0)
    height: 每个样本的y轴高度(默认14)
    width: 每个样本的x轴宽度(默认14)
    depth: 使用的基因数量(默认200)
    min_spot: 样本包含的最小spot数(默认10)
    pad_id: 用于填充不足基因的标识符(默认"PAD")
    n_samples_multiplier: 采样次数倍数(默认2)
    
    # Spot级别元数据
    adata.obs = {
        'original_index': 原始索引,
        'x_abs': 绝对X坐标,
        'y_abs': 绝对Y坐标,
        'x_rel': 样本内相对X坐标,
        'y_rel': 样本内相对Y坐标
    }

    # 基因级别元数据
    adata.var = {
        'gene_ids': 基因ID列表(含pad_id填充),
        'is_selected': 基因是否被选择的布尔向量
    }

    # 样本元数据
    adata.uns['region'] = {
        'x_start': 区域起始X,
        'y_start': 区域起始Y,
        'width': 区域宽度,
        'height': 区域高度
    }
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有h5文件
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    if not h5_files:
        print("⚠️ 警告: 输入目录中没有找到.h5文件")
        return
    
    print(f"🔍 找到 {len(h5_files)} 个.h5文件，开始处理...")
    print(f"⚙️ 配置参数: height={height}, width={width}, depth={depth}, min_spot={min_spot}")
    print(f"🎯 采样策略: 以细胞坐标为中心采样{height}x{width}区域")
    print(f"💾 输出格式: H5AD (AnnData格式)")
    
    # 统计变量
    total_samples = 0
    skipped_files = 0
    skipped_groups = 0
    skipped_blocks = 0
    
    # 进度条：文件处理
    file_pbar = tqdm(h5_files, desc="文件处理")
    
    for file_name in file_pbar:
        file_pbar.set_postfix(file=file_name)
        file_path = os.path.join(input_dir, file_name)
        
        try:
            with h5py.File(file_path, 'r') as h5_file:
                groups = list(h5_file.keys())
                if not groups:
                    skipped_files += 1
                    print(f"⚠️ 跳过: 文件 {file_name} 中没有找到任何组")
                    continue
                
                # 组处理进度
                group_pbar = tqdm(groups, desc=f"样本处理", leave=False)
                
                for group_name in group_pbar:
                    group_pbar.set_postfix(group=group_name)
                    
                    try:
                        group = h5_file[group_name]
                        
                        # 提取数据
                        coords = group['coords_map'][:]
                        expr = group['expr_map'][:]
                        vocab = group['vocab_index'][:]
                        
                        # 处理基因ID类型
                        if isinstance(vocab[0], bytes):
                            vocab = np.array([g.decode('utf-8') for g in vocab])
                        elif np.issubdtype(vocab.dtype, np.integer):
                            vocab = np.array([str(g) for g in vocab])
                        
                        n_spots, n_genes = expr.shape
                        
                        # 计算每个基因的整体方差（用于权重）
                        overall_var = np.var(expr, axis=0)
                        
                        # 确定空间范围
                        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                        
                        # 计算采样次数：细胞数量的两倍
                        n_samples = n_spots * n_samples_multiplier
                        
                        sample_count = 1
                        spots_added = 0
                        
                        # 区块处理进度
                        block_pbar = tqdm(
                            total=n_samples, 
                            desc=f"细胞中心采样", 
                            leave=False
                        )
                        
                        # 以细胞坐标为中心进行采样
                        for _ in range(n_samples):
                            block_pbar.update(1)
                            
                            # 随机选择一个细胞作为中心点
                            center_idx = random.randint(0, n_spots - 1)
                            center_x, center_y = coords[center_idx]
                            
                            # 计算采样区域的起始坐标（确保区域在切片范围内）
                            x_start = max(x_min, center_x - width // 2)
                            y_start = max(y_min, center_y - height // 2)
                            
                            # 调整起始点，确保区域不超出边界
                            x_start = min(x_start, x_max - width)
                            y_start = min(y_start, y_max - height)
                            
                            # 确保起始坐标为整数
                            x_start = int(x_start)
                            y_start = int(y_start)
                            
                            x_end = x_start + width
                            y_end = y_start + height
                            
                            # 选择区块内的spots
                            in_block = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) & \
                                       (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
                            
                            # 跳过spot不足的区块
                            num_spots = np.sum(in_block)
                            if num_spots < min_spot:
                                skipped_blocks += 1
                                continue
                            
                            block_coords = coords[in_block]
                            block_expr = expr[in_block]
                            rel_coords = block_coords - np.array([x_start, y_start])
                            
                            # === 选择高变基因 ===
                            
                            # 计算区域内每个基因的方差
                            block_var = np.var(block_expr, axis=0)
                            
                            # 结合全局方差和区域方差的权重
                            alpha = 1.0  # 区域方差权重
                            beta = 0   # 全局方差权重
                            gene_weights = alpha * block_var + beta * overall_var
                            
                            # 标准化权重并添加小常数避免零权重
                            gene_weights = (gene_weights - np.min(gene_weights)) / \
                                          (np.max(gene_weights) - np.min(gene_weights) + 1e-8) + 1e-8
                            
                            # 加权随机选择depth个基因（无重复）
                            selected_indices = random.choices(
                                range(n_genes), 
                                weights=gene_weights, 
                                k=min(depth, n_genes)
                            )
                            
                            # 如果需要，使用pad_id填充
                            selected_gene_ids = []
                            selected_expr = np.zeros((num_spots, depth))
                            
                            # 复制选中的基因表达值
                            for i, idx in enumerate(selected_indices):
                                selected_gene_ids.append(vocab[idx])
                                selected_expr[:, i] = block_expr[:, idx]
                            
                            # 填充不足的基因
                            if len(selected_indices) < depth:
                                fill_count = depth - len(selected_indices)
                                selected_gene_ids.extend([pad_id] * fill_count)
                            
                            # 创建是否被选中的标记
                            is_selected = [True] * len(selected_indices) + [False] * (depth - len(selected_indices))
                            
                            # 创建AnnData对象
                            adata = ad.AnnData(
                                X=selected_expr,
                                obs={
                                    'original_index': np.where(in_block)[0],
                                    'x_abs': block_coords[:, 0],
                                    'y_abs': block_coords[:, 1],
                                    'x_rel': rel_coords[:, 0],
                                    'y_rel': rel_coords[:, 1]
                                },
                                var={
                                    'gene_ids': selected_gene_ids,
                                    'is_selected': is_selected
                                }
                            )
                            
                            # 添加空间信息到obsm
                            adata.obsm['coords_map'] = block_coords
                            adata.obsm['coords_sample'] = rel_coords
                            
                            # 添加额外元数据
                            adata.uns['region'] = {
                                'x_start': x_start,
                                'y_start': y_start,
                                'width': width,
                                'height': height
                            }
                            
                            adata.uns['gene_selection'] = {
                                'method': 'weighted_random',
                                'alpha': alpha,
                                'beta': beta
                            }
                            
                            # 添加中心细胞信息
                            adata.uns['center_cell'] = {
                                'x': center_x,
                                'y': center_y,
                                'index': center_idx
                            }
                            
                            # 保存为H5AD
                            output_name = f"{group_name}_sample_{sample_count}.h5ad"
                            output_path = os.path.join(output_dir, output_name)
                            adata.write(output_path)
                            
                            sample_count += 1
                            total_samples += 1
                            spots_added += num_spots
                        
                        block_pbar.close()
                        group_pbar.set_postfix(spots=f"{spots_added} spots")
                    
                    except Exception as e:
                        skipped_groups += 1
                        print(f"⚠️ 跳过组 {group_name}: {str(e)}")
                
                group_pbar.close()
        
        except Exception as e:
            skipped_files += 1
            print(f"🚨 错误处理文件 {file_name}: {str(e)}")
    
    # 处理完成后的统计信息
    print(f"\n✅ 处理完成!")
    print(f"📊 统计:")
    print(f"  创建样本: {total_samples}个")
    print(f"  采样尝试: {n_samples}次 (细胞数量×{n_samples_multiplier})")
    print(f"  跳过文件: {skipped_files}个")
    print(f"  跳过样本组: {skipped_groups}个")
    print(f"  跳过区块: {skipped_blocks}个 (spot<{min_spot})")
    
    if total_samples == 0:
        print("❌ 未创建任何样本，请检查输入数据和参数配置")
    else:
        print(f"💾 输出目录: {output_dir} (H5AD格式)")
        print(f"🔬 每个样本包含 {depth} 个基因 (使用 {pad_id} 填充不足)")
'''



def scale_coordinates(coords: np.ndarray, target_size: int = 560) -> np.ndarray:
    """
    将坐标缩放至target_size x target_size范围
    
    参数:
    coords: 原始坐标数组
    target_size: 目标尺寸
    
    返回:
    缩放后的坐标数组
    """
    if len(coords) == 0:
        return coords
        
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 避免除零错误
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    
    # 缩放坐标
    scaled_x = ((x_coords - x_min) / x_range) * (target_size - 1)
    scaled_y = ((y_coords - y_min) / y_range) * (target_size - 1)
    
    # 四舍五入为整数
    scaled_coords = np.column_stack((np.round(scaled_x).astype(int), 
                                   np.round(scaled_y).astype(int)))
    
    return scaled_coords

def map_to_grid(block_coords: np.ndarray, x_start: int, y_start: int, 
                height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将区块内的坐标映射到height*width的网格中
    
    参数:
    block_coords: 区块内所有点的绝对坐标
    x_start: 区块起始x坐标
    y_start: 区块起始y坐标  
    height: 网格高度
    width: 网格宽度
    
    返回:
    rel_coords: 相对网格坐标(浮点数格式的整数)
    valid_mask: 有效点位的掩码(未被丢弃的点)
    """
    # 计算相对坐标
    rel_coords = block_coords - np.array([x_start, y_start])
    
    # 四舍五入到最近的网格坐标
    grid_coords = np.round(rel_coords).astype(int)
    
    # 确保坐标在网格范围内
    grid_coords[:, 0] = np.clip(grid_coords[:, 0], 0, width - 1)
    grid_coords[:, 1] = np.clip(grid_coords[:, 1], 0, height - 1)
    
    # 创建网格占用记录
    grid_occupied = np.zeros((height, width), dtype=bool)
    valid_mask = np.ones(len(rel_coords), dtype=bool)
    
    # 检查每个点位的网格是否被占用
    for i, (x, y) in enumerate(grid_coords):
        if grid_occupied[y, x]:
            # 网格已被占用，丢弃该点位
            valid_mask[i] = False
        else:
            # 标记网格为已占用
            grid_occupied[y, x] = True
            # 使用浮点数格式存储整数坐标
            rel_coords[i] = [float(x), float(y)]
    
    return rel_coords, valid_mask

def create_sample_data(block_coords: np.ndarray, block_expr: np.ndarray, 
                      vocab: np.ndarray, x_start: int, y_start: int,
                      selected_indices: List[int], selected_gene_ids: List[str], 
                      is_selected: List[bool]) -> ad.AnnData:
    """
    创建样本数据并返回AnnData对象
    
    参数:
    block_coords: 区块坐标
    block_expr: 区块表达数据
    vocab: 基因词汇表
    x_start: 区域起始X坐标
    y_start: 区域起始Y坐标
    selected_indices: 选中的基因索引
    selected_gene_ids: 选中的基因ID
    is_selected: 基因是否被选中的标记
    
    返回:
    AnnData对象
    """
    # 计算相对坐标
    rel_coords = block_coords - np.array([x_start, y_start])
    
    # 将表达矩阵转换为稀疏矩阵并优化数据类型
    if len(selected_indices) < block_expr.shape[1]:
        expr_data = block_expr[:, selected_indices]
    else:
        expr_data = block_expr
    
    # 转换为CSR稀疏矩阵并优化数据类型
    expr_sparse = csr_matrix(expr_data.astype(np.float32))
    
    # 创建AnnData对象
    adata = ad.AnnData(
        X=expr_sparse,  # 使用稀疏矩阵
        obs={
            'original_index': np.arange(len(block_coords)),
            'x_abs': block_coords[:, 0],
            'y_abs': block_coords[:, 1],
            'x_rel': rel_coords[:, 0],
            'y_rel': rel_coords[:, 1]
        },
        var={
            'gene_ids': selected_gene_ids,
            'is_selected': is_selected
        }
    )
    
    # 添加空间信息到obsm
    adata.obsm['coords_map'] = block_coords
    adata.obsm['coords_sample'] = rel_coords
    
    return adata

def process_group(group: h5py.Group, height: int, width: int, depth: int, 
                 min_spot: int, pad_id: int, n_samples_multiplier: int, 
                 output_dir: str, original_coords: np.ndarray) -> Tuple[int, int, int]:
    """
    处理单个组的数据
    
    参数:
    group: HDF5组对象
    height: 样本高度
    width: 样本宽度
    depth: 基因深度
    min_spot: 最小spot数
    pad_id: 填充ID
    n_samples_multiplier: 采样倍数
    output_dir: 输出目录
    original_coords: 原始坐标（未缩放的）
    
    返回:
    (样本数, 跳过区块数, 添加spot数)
    """
    # 提取数据
    coords = group['coords_map'][:]
    expr = group['expr_map'][:]
    vocab = group['vocab_index'][:]
    
    # 处理基因ID类型
    if isinstance(vocab[0], bytes):
        vocab = np.array([g.decode('utf-8') for g in vocab])
    elif np.issubdtype(vocab.dtype, np.integer):
        vocab = np.array([str(g) for g in vocab])
    target_size = 140
    # 缩放坐标到140x140范围
    coords = scale_coordinates(coords, target_size)
    
    n_spots, n_genes = expr.shape
    
    # 确定空间范围（使用原始坐标计算面积）
    orig_x_min, orig_x_max = original_coords[:, 0].min(), original_coords[:, 0].max()
    orig_y_min, orig_y_max = original_coords[:, 1].min(), original_coords[:, 1].max()
    
    # 计算原始样本面积
    orig_sample_area = (orig_x_max - orig_x_min) * (orig_y_max - orig_y_min)
    sample_area = height * width
    
    # 计算采样次数：样本面积/sample面积 * 2
    n_samples = int((target_size * target_size / sample_area) * 2)
    
    # 确定缩放后的空间范围
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    sample_count = 0
    skipped_blocks = 0
    spots_added = 0
    
    # 随机采样空间区域
    for _ in range(n_samples):
        # 随机选择区域的起始坐标
        x_start = random.randint(int(x_min), int(x_max - width))
        y_start = random.randint(int(y_min), int(y_max - height))
        
        x_end = x_start + width
        y_end = y_start + height
        
        # 选择区块内的spots
        in_block = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) & \
                   (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
        
        # 跳过spot不足的区块
        num_spots = np.sum(in_block)
        if num_spots < min_spot:
            skipped_blocks += 1
            continue
        
        block_coords = coords[in_block]
        block_expr = expr[in_block]
        
        # 映射坐标到网格并处理冲突
        rel_coords, valid_mask = map_to_grid(
            block_coords, x_start, y_start, height, width
        )
        
        # 使用有效掩码过滤坐标和表达数据
        block_coords = block_coords[valid_mask]
        block_expr = block_expr[valid_mask]
        rel_coords = rel_coords[valid_mask]
        
        # 更新spot数量
        num_spots = np.sum(valid_mask)
        
        # 如果有效点位数不足，跳过该区块
        if num_spots < min_spot:
            skipped_blocks += 1
            continue
        
        selected_indices = list(range(n_genes))  # 选择所有基因
        
        # 保存所有基因ID
        selected_gene_ids = vocab.tolist()
        
        # 创建是否被选中的标记
        is_selected = [True] * n_genes
        
        # 创建样本数据
        adata = create_sample_data(
            block_coords, block_expr, vocab, x_start, y_start,
            selected_indices, selected_gene_ids, is_selected
        )
        

        
        # 保存为H5AD
        output_name = f"{group.name}_sample_{sample_count}.h5ad"
        output_name = output_name.lstrip('/\\')
        output_path = os.path.join(output_dir, output_name)
        adata.write(output_path)
        
        sample_count += 1
        spots_added += num_spots
    
    return sample_count, skipped_blocks, spots_added

def make_samples_h5(input_dir: str, output_dir: str, seed: int = 0, height: int = 14, 
                   width: int = 14, depth: int = 200, min_spot: int = 10, 
                   pad_id: int = 0, n_samples_multiplier: int = 2) -> Dict[str, int]:
    """
    处理空间转录组数据，随机采样空间区域，保存为H5AD格式
    
    参数:
    input_dir: 包含.h5输入文件的目录
    output_dir: 输出样本的目录
    seed: 随机种子(默认0)
    height: 每个样本的y轴高度(默认14)
    width: 每个样本的x轴宽度(默认14)
    depth: 使用的基因数量(默认200)
    min_spot: 样本包含的最小spot数(默认10)
    pad_id: 用于填充不足基因的标识符(默认0)
    n_samples_multiplier: 采样次数倍数(默认2)
    
    返回:
    处理统计信息
    """
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有h5文件
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    if not h5_files:
        print("⚠️ 警告: 输入目录中没有找到.h5文件")
        return {}
    
    print(f"🔍 找到 {len(h5_files)} 个.h5文件，开始处理...")
    print(f"⚙️ 配置参数: height={height}, width={width}, depth={depth}, min_spot={min_spot}")
    print(f"🎯 采样策略: 随机采样{height}x{width}区域")
    print(f"💾 输出格式: H5AD (AnnData格式)")
    
    # 统计变量
    total_samples = 0
    skipped_files = 0
    skipped_groups = 0
    skipped_blocks = 0
    total_spots = 0
    
    # 进度条：文件处理
    file_pbar = tqdm(h5_files, desc="文件处理")
    
    for file_name in file_pbar:
        file_pbar.set_postfix(file=file_name)
        file_path = os.path.join(input_dir, file_name)
        
        try:
            with h5py.File(file_path, 'r') as h5_file:
                groups = list(h5_file.keys())
                if not groups:
                    skipped_files += 1
                    print(f"⚠️ 跳过: 文件 {file_name} 中没有找到任何组")
                    continue
                
                # 组处理进度
                group_pbar = tqdm(groups, desc=f"样本处理", leave=False)
                
                for group_name in group_pbar:
                    group_pbar.set_postfix(group=group_name)
                    
                    try:
                        group = h5_file[group_name]
                        
                        # 获取原始坐标（未缩放的）
                        original_coords = group['coords_map'][:]
                        
                        samples, skipped, spots = process_group(
                            group, height, width, depth, min_spot, 
                            pad_id, n_samples_multiplier, output_dir, original_coords
                        )
                        
                        total_samples += samples
                        skipped_blocks += skipped
                        total_spots += spots
                        
                        group_pbar.set_postfix(spots=f"{spots} spots")
                    
                    except Exception as e:
                        skipped_groups += 1
                        print(f"⚠️ 跳过组 {group_name}: {str(e)}")
                
                group_pbar.close()
        
        except Exception as e:
            skipped_files += 1
            print(f"🚨 错误处理文件 {file_name}: {str(e)}")
    
    # 处理完成后的统计信息
    print(f"\n✅ 处理完成!")
    print(f"📊 统计:")
    print(f"  创建样本: {total_samples}个")
    print(f"  采样尝试: {total_samples + skipped_blocks}次")
    print(f"  跳过文件: {skipped_files}个")
    print(f"  跳过样本组: {skipped_groups}个")
    print(f"  跳过区块: {skipped_blocks}个 (spot<{min_spot})")
    print(f"  总处理spots: {total_spots}个")
    
    if total_samples == 0:
        print("❌ 未创建任何样本，请检查输入数据和参数配置")
    else:
        print(f"💾 输出目录: {output_dir} (H5AD格式)")
        print(f"🔬 每个样本包含所有基因 (使用稀疏矩阵存储)")
    
    return {
        'total_samples': total_samples,
        'skipped_files': skipped_files,
        'skipped_groups': skipped_groups,
        'skipped_blocks': skipped_blocks,
        'total_spots': total_spots
    }

def main(input_dir,vocab_path,output_dir,hv_genes_path,seed=0,height=14,width=14,depth=200,min_spot=10):
    if os.path.exists(vocab_path):
        print(f"从 {vocab_path} 加载基因索引...")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = None
        print(f"警告: 未找到词汇表文件 {vocab_path}")
    #hv_genes_id = find_hv_genes(input_dir,depth,hv_genes_path)
    make_samples_h5(input_dir,output_dir,seed,height,width,depth,min_spot,vocab["<pad>"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="make sample with size of 14*14*200"
    )
    parser.add_argument('-i', '--input-dir',      default='project1/spatial_data/all_data',    help='Directory of raw inputs')
    parser.add_argument('-o', '--output-dir',     default='project1/spatial_data/samples16',     help='Base output directory')
    parser.add_argument('-v', '--vocab-path',     default='project1/spatial_data/spatial_data/new_vocab.json',               help='Path to vocab JSON file')
    parser.add_argument('-hv', '--hv-genes-path', default='project1/spatial_data/spatial_data/samples/hv_genes_id.json',     help='Path to hv genes vocab JSON file')    
    parser.add_argument('-min', '--min-spot', type=int, default=57,              help='min spot of each sample (height*width*0.1)')
    parser.add_argument('-s', '--seed',       type=int, default=0,              help='Random seed')
    parser.add_argument('-H', '--height',     type=int, default=16,             help='height of sample')
    parser.add_argument('-W', '--width',      type=int, default=16,             help='width of sample')
    parser.add_argument('-D', '--depth',      type=int, default=512,            help='depth of sample')
    args = parser.parse_args()
    main(args.input_dir, args.vocab_path, args.output_dir, args.hv_genes_path, args.seed,args.height,args.width,args.depth,args.min_spot)


    