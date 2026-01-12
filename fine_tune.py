import torch
import numpy as np
from scipy import sparse
import anndata as ad
from torch import optim
import os
import json
import anndata as ad
from utils.utils import get_least_used_gpu
from model.MAE import MAEModel
import scipy
import scanpy as sc
from scipy.spatial import KDTree  # 用于高效的空间最近邻搜索
from tqdm import tqdm
from dataset.dataset import SpatialTranscriptomicsDataset

def fine_tune_model(model, adata, vocab, epochs, config, device):
    """
    微调模型，使用空间转录组数据（AnnData）进行训练。
    
    参数:
        model: 要微调的模型，预期输入 expression [batch, c, h, w] 和 gene_ids [batch, c]
        adata: AnnData 对象，包含 .X（表达矩阵）、.obsm['spatial']（空间坐标）和 .var（基因名称）
        vocab: 字典，将基因名称映射到ID，必须包含 "<pad>" 用于填充
        epochs: 训练周期数
        config: 配置字典，包含 'h'（空间高度）、'w'（空间宽度）、'c'（基因数量）、
                'batch_size'（批次大小）、'lr'（学习率，可选）
        device: 训练设备（如 'cuda' 或 'cpu'）
    
    返回:
        model: 微调后的模型
    """
    # 从配置中获取参数
    h = config['h']  # 空间网格高度
    w = config['w']  # 空间网格宽度
    c = config['c']  # 要使用的基因数量
    batch_size = config.get('batch_size', 1)  # 默认批次大小为1
    lr = config.get('lr', 1e-5)  # 默认学习率

    # 检查 adata 是否包含空间坐标
    if 'spatial' not in adata.obsm:
        raise ValueError("adata must have spatial coordinates in .obsm['spatial']")
    
    # 提取坐标和表达矩阵
    coords = adata.obsm['spatial']  # 保持原始格式（可能是浮点数）
    expr_matrix = adata.X  # 表达矩阵，可能为稀疏
    if sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()  # 转换为密集矩阵以便处理
    n_cells = adata.n_obs  # 细胞数量
    n_genes_adata = adata.n_vars  # adata 中的基因数量
    gene_names = adata.var_names  # 基因名称列表

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    # 训练循环
    for epoch in range(epochs):
        # 随机打乱细胞顺序
        indices = np.random.permutation(n_cells)
        
        # 使用tqdm添加进度条[1,2](@ref)
        progress_bar = tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{epochs}", unit="cell")
        
        # 批次处理循环[6,7](@ref)
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # 初始化批次数据
            batch_expressions = []
            batch_gene_ids = []
            
            # 处理当前批次中的每个细胞
            for idx in batch_indices:
                # 获取当前细胞的坐标
                x_center, y_center = coords[idx]
                
                # 计算网格边界（使用浮点数）
                half_w = w / 2
                half_h = h / 2
                x_min = x_center - half_w
                x_max = x_center + half_w
                y_min = y_center - half_h
                y_max = y_center + half_h
                
                # 创建网格坐标系统
                x_grid = np.linspace(x_min, x_max, w, endpoint=False)
                y_grid = np.linspace(y_min, y_max, h, endpoint=False)
                
                # 计算每个网格点的中心坐标
                grid_centers = np.array([(x + (x_grid[1]-x_grid[0])/2, 
                                        y + (y_grid[1]-y_grid[0])/2) 
                                       for y in y_grid for x in x_grid])
                
                # 为网格点创建KDTree
                grid_tree = KDTree(grid_centers)
                
                # 查找在当前细胞周围矩形范围内的所有细胞
                in_range_indices = []
                for i, (x, y) in enumerate(coords):
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        in_range_indices.append(i)
                
                n_spots = len(in_range_indices)
                
                # 初始化网格表达矩阵，形状 [h, w, n_genes_adata]，初始为0
                grid_expr = np.zeros((h, w, n_genes_adata))
                
                # 为范围内的每个细胞查找最近网格并填充
                for cell_idx in in_range_indices:
                    cell_coord = coords[cell_idx]
                    # 查找距离当前细胞最近的网格点
                    dist, grid_idx = grid_tree.query(cell_coord)
                    
                    # 将网格索引转换回二维坐标
                    grid_y = grid_idx // w
                    grid_x = grid_idx % w
                    
                    # 将细胞表达值填充到最近网格
                    grid_expr[grid_y, grid_x, :] = expr_matrix[cell_idx, :]
                
                # 计算高变基因：基于网格内细胞的表达方差
                if n_spots > 0:
                    spot_expr = expr_matrix[in_range_indices, :]  # 网格内细胞的表达矩阵
                    var_per_gene = np.var(spot_expr, axis=0)  # 每个基因的方差
                else:
                    var_per_gene = np.zeros(n_genes_adata)  # 无细胞时方差为0
                
                # 选择 top c 高变基因的索引
                if n_genes_adata >= c:
                    top_gene_indices = np.argsort(var_per_gene)[-c:][::-1]  # 方差最大的 c 个基因索引
                else:
                    top_gene_indices = np.arange(n_genes_adata)  # 所有基因，后续填充
                
                # 创建减少后的表达矩阵 [h, w, c]
                reduced_expr = grid_expr[:, :, top_gene_indices]
                if n_genes_adata < c:
                    # 填充至 c 个基因
                    padded_expr = np.zeros((h, w, c))
                    padded_expr[:, :, :n_genes_adata] = reduced_expr
                    reduced_expr = padded_expr
                
                # 创建 gene_ids 列表
                gene_ids_list = []
                for i in range(c):
                    if i < n_genes_adata:
                        gene_name = gene_names[top_gene_indices[i]]
                        gene_id = vocab.get(gene_name, vocab.get("<unk>", vocab["<pad>"]))  # 使用未知 token 或 pad
                    else:
                        gene_id = vocab["<pad>"]  # 填充 pad
                    gene_ids_list.append(gene_id)
                
                # 转换为 tensor 并调整维度
                expression_tensor = torch.tensor(reduced_expr, dtype=torch.float32).unsqueeze(0)  # [1, h, w, c]
                expression_tensor = expression_tensor.permute(0, 3, 1, 2)  # [1, c, h, w]
                gene_ids_tensor = torch.tensor(gene_ids_list, dtype=torch.long).unsqueeze(0)  # [1, c]
                
                # 添加到批次数据
                batch_expressions.append(expression_tensor)
                batch_gene_ids.append(gene_ids_tensor)
            
            # 合并批次数据
            if batch_expressions:
                batch_expression = torch.cat(batch_expressions, dim=0)  # [batch_size, c, h, w]
                batch_gene_id = torch.cat(batch_gene_ids, dim=0)  # [batch_size, c]
                
                # 移动到设备
                batch_expression = batch_expression.to(device)
                batch_gene_id = batch_gene_id.to(device)
                
                # 模型前向传播
                model_output = model(batch_expression, batch_gene_id)
                recon = model_output[0] if isinstance(model_output, tuple) else model_output['recon']
                mask = model_output[2] if isinstance(model_output, tuple) else model_output['combined_mask']
                
                # 计算损失
                loss = model.loss_function(batch_expression, recon, mask)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 更新进度条[1,2](@ref)
            progress_bar.update(len(batch_indices))
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}" if 'loss' in locals() else "N/A"})
        
        # 完成当前epoch
        progress_bar.close()
        print(f"Epoch {epoch+1}/{epochs} completed")
    
    return model

'''
# 根据方差加权随机选择 c 个基因
def fine_tune_model(model, adata, vocab, epochs, config, device):
    """
    微调模型，使用空间转录组数据（AnnData）进行训练。
    
    参数:
        model: 要微调的模型，预期输入 expression [batch, c, h, w] 和 gene_ids [batch, c]
        adata: AnnData 对象，包含 .X（表达矩阵）、.obsm['spatial']（空间坐标）和 .var（基因名称）
        vocab: 字典，将基因名称映射到ID，必须包含 "<pad>" 用于填充
        epochs: 训练周期数
        config: 配置字典，包含 'h'（空间高度）、'w'（空间宽度）、'c'（基因数量）、
                'batch_size'（批次大小）、'lr'（学习率，可选）
        device: 训练设备（如 'cuda' 或 'cpu'）
    
    返回:
        model: 微调后的模型
    """
    # 从配置中获取参数
    h = config['h']  # 空间网格高度
    w = config['w']  # 空间网格宽度
    c = config['c']  # 要使用的基因数量
    batch_size = config.get('batch_size', 1)  # 默认批次大小为1
    lr = config.get('lr', 1e-5)  # 默认学习率

    # 检查 adata 是否包含空间坐标
    if 'spatial' not in adata.obsm:
        raise ValueError("adata must have spatial coordinates in .obsm['spatial']")
    
    # 提取坐标和表达矩阵
    coords = adata.obsm['spatial']  # 保持原始格式（可能是浮点数）
    expr_matrix = adata.X  # 表达矩阵，可能为稀疏
    if sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()  # 转换为密集矩阵以便处理
    n_cells = adata.n_obs  # 细胞数量
    n_genes_adata = adata.n_vars  # adata 中的基因数量
    gene_names = adata.var_names  # 基因名称列表

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    # 训练循环
    for epoch in range(epochs):
        # 随机打乱细胞顺序
        indices = np.random.permutation(n_cells)
        
        # 使用tqdm添加进度条
        progress_bar = tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{epochs}", unit="cell")
        
        # 批次处理循环
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # 初始化批次数据
            batch_expressions = []
            batch_gene_ids = []
            
            # 处理当前批次中的每个细胞
            for idx in batch_indices:
                # 获取当前细胞的坐标
                x_center, y_center = coords[idx]
                
                # 计算网格边界（使用浮点数）
                half_w = w / 2
                half_h = h / 2
                x_min = x_center - half_w
                x_max = x_center + half_w
                y_min = y_center - half_h
                y_max = y_center + half_h
                
                # 创建网格坐标系统
                x_grid = np.linspace(x_min, x_max, w, endpoint=False)
                y_grid = np.linspace(y_min, y_max, h, endpoint=False)
                
                # 计算每个网格点的中心坐标
                grid_centers = np.array([(x + (x_grid[1]-x_grid[0])/2, 
                                        y + (y_grid[1]-y_grid[0])/2) 
                                       for y in y_grid for x in x_grid])
                
                # 为网格点创建KDTree
                grid_tree = KDTree(grid_centers)
                
                # 查找在当前细胞周围矩形范围内的所有细胞
                in_range_indices = []
                for i, (x, y) in enumerate(coords):
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        in_range_indices.append(i)
                
                n_spots = len(in_range_indices)
                
                # 初始化网格表达矩阵，形状 [h, w, n_genes_adata]，初始为0
                grid_expr = np.zeros((h, w, n_genes_adata))
                
                # 为范围内的每个细胞查找最近网格并填充
                for cell_idx in in_range_indices:
                    cell_coord = coords[cell_idx]
                    # 查找距离当前细胞最近的网格点
                    dist, grid_idx = grid_tree.query(cell_coord)
                    
                    # 将网格索引转换回二维坐标
                    grid_y = grid_idx // w
                    grid_x = grid_idx % w
                    
                    # 将细胞表达值填充到最近网格
                    grid_expr[grid_y, grid_x, :] = expr_matrix[cell_idx, :]
                
                # 计算高变基因：基于网格内细胞的表达方差
                if n_spots > 0:
                    spot_expr = expr_matrix[in_range_indices, :]  # 网格内细胞的表达矩阵
                    var_per_gene = np.var(spot_expr, axis=0)  # 每个基因的方差
                else:
                    var_per_gene = np.zeros(n_genes_adata)  # 无细胞时方差为0
                
                # 根据方差加权随机选择 c 个基因 [6,7](@ref)
                if n_genes_adata > 0:
                    # 处理方差为0的情况，给一个极小值避免除零错误
                    var_per_gene = np.where(var_per_gene == 0, 1e-10, var_per_gene)
                    
                    # 计算选择概率（方差越大，被选中的概率越高）
                    total_variance = np.sum(var_per_gene)
                    selection_probs = var_per_gene / total_variance
                    
                    # 确保选择的基因数量不超过总基因数
                    select_count = min(c, n_genes_adata)
                    
                    # 使用numpy的random.choice进行加权随机选择（不重复选择）
                    selected_gene_indices = np.random.choice(
                        n_genes_adata, 
                        size=select_count, 
                        replace=False,  # 不重复选择
                        p=selection_probs
                    )
                else:
                    selected_gene_indices = np.array([], dtype=int)
                
                # 创建减少后的表达矩阵 [h, w, c]
                reduced_expr = np.zeros((h, w, c))
                if len(selected_gene_indices) > 0:
                    reduced_expr[:, :, :len(selected_gene_indices)] = grid_expr[:, :, selected_gene_indices]
                
                # 创建 gene_ids 列表
                gene_ids_list = []
                for i in range(c):
                    if i < len(selected_gene_indices):
                        gene_name = gene_names[selected_gene_indices[i]]
                        gene_id = vocab.get(gene_name, vocab.get("<unk>", vocab["<pad>"]))  # 使用未知 token 或 pad
                    else:
                        gene_id = vocab["<pad>"]  # 填充 pad
                    gene_ids_list.append(gene_id)
                
                # 转换为 tensor 并调整维度
                expression_tensor = torch.tensor(reduced_expr, dtype=torch.float32).unsqueeze(0)  # [1, h, w, c]
                expression_tensor = expression_tensor.permute(0, 3, 1, 2)  # [1, c, h, w]
                gene_ids_tensor = torch.tensor(gene_ids_list, dtype=torch.long).unsqueeze(0)  # [1, c]
                
                # 添加到批次数据
                batch_expressions.append(expression_tensor)
                batch_gene_ids.append(gene_ids_tensor)
            
            # 合并批次数据
            if batch_expressions:
                batch_expression = torch.cat(batch_expressions, dim=0)  # [batch_size, c, h, w]
                batch_gene_id = torch.cat(batch_gene_ids, dim=0)  # [batch_size, c]
                
                # 移动到设备
                batch_expression = batch_expression.to(device)
                batch_gene_id = batch_gene_id.to(device)
                
                # 模型前向传播
                model_output = model(batch_expression, batch_gene_id)
                recon = model_output[0] if isinstance(model_output, tuple) else model_output['recon']
                mask = model_output[1] if isinstance(model_output, tuple) else model_output['combined_mask']
                
                # 计算损失
                loss = model.loss_function(batch_expression, recon, mask)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 更新进度条
            progress_bar.update(len(batch_indices))
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}" if 'loss' in locals() else "N/A"})
        
        # 完成当前epoch
        progress_bar.close()
        print(f"Epoch {epoch+1}/{epochs} completed")
    
    return model
'''

def _is_log_transformed(X, threshold=10):
    """
    判断数据是否已经过对数化处理
    
    参数
    ----
    X : numpy.ndarray, scipy.sparse.spmatrix
        输入数据矩阵
    threshold : float
        判断阈值，最大值超过此值则认为未对数化
    
    返回
    ----
    bool
        是否已经过对数化处理
    """
    # 处理稀疏矩阵
    if scipy.sparse.issparse(X):
        X_data = X.data
    else:
        X_data = X
        
    # 检查数据中最大值
    max_val = np.max(X_data) if len(X_data) > 0 else 0
    
    # 如果最大值大于阈值，则认为数据未对数化
    if max_val > threshold:
        return False
    
    # 进一步检查数据特征：对数化后的数据通常包含小数
    if scipy.sparse.issparse(X):
        sample_values = X_data[:min(1000, len(X_data))]
    else:
        sample_values = X.flatten()[::max(1, X.size // 1000)]
    
    # 检查小数部分占比
    has_fraction = np.any(np.modf(sample_values)[0] != 0)
    
    # 如果最大值较小且包含小数，则认为已对数化
    return has_fraction or max_val < 5

def log1p_with_backup(adata, layer_raw='counts', layer_log='log1p'):
    """
    对 adata.X 做 log1p（如果尚未对数化），同时完整保留原始计数。
    
    参数
    ----
    adata : AnnData
        输入对象
    layer_raw : str
        用来存放原始计数的 layer 名，默认 'counts'
    layer_log : str
        用来存放 log1p 后值的 layer 名，默认 'log1p'
    """
    # 检查数据是否已经对数化
    if _is_log_transformed(adata.X):
        print("✅ 数据已经过对数化处理，跳过 log1p 转换")
        # 确保原始数据已备份
        if adata.raw is None:
            adata.raw = adata.copy()
        if layer_raw not in adata.layers:
            adata.layers[layer_raw] = adata.raw.X.copy()
        if layer_log not in adata.layers:
            adata.layers[layer_log] = adata.X.copy()
        return
    
    # 1) 拷贝原始计数到 adata.raw（scanpy 推荐做法）
    adata.raw = adata.copy()
    
    # 2) 再存一份到指定 layer，方便矩阵操作
    adata.layers[layer_raw] = adata.X.copy()
    
    # 3) 真正执行 log1p（只在 adata.X 上操作）
    sc.pp.log1p(adata)          # 原位修改 adata.X
    
    # 4) 把 log1p 后的结果也存一份 layer，方便调用
    adata.layers[layer_log] = adata.X.copy()
    
    print("✅ log1p 完成：")
    print(f"   原始计数 -> adata.raw, adata.layers['{layer_raw}']")
    print(f"   log1p 值 -> adata.X, adata.layers['{layer_log}']")

def scale_spatial_coordinates(adata, target_range=(0, 1)):
    """
    将 adata.obsm['spatial'] 中的坐标缩放到目标范围；
    若 adata.obsm 中已存在 'ori_spatial'，则视为已缩放过，直接跳过。

    参数
    ----
    adata : AnnData
    target_range : tuple, 默认 (0, 1)

    返回
    ----
    adata : AnnData（原位修改，同时返回方便链式调用）
    """
    # 如果已经保存过原始坐标，则不再重复缩放
    if 'ori_spatial' in adata.obsm:
        print("检测到 'ori_spatial'，空间坐标已缩放过，跳过本次操作。")
        return adata

    spatial = adata.obsm['spatial'].copy()
    min_vals = spatial.min(axis=0)
    max_vals = spatial.max(axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # 防止除零

    target_min, target_max = target_range
    scaled_spatial = (spatial - min_vals) / ranges * (target_max - target_min) + target_min

    adata.obsm['ori_spatial'] = spatial        # 保存原始坐标
    adata.obsm['spatial'] = scaled_spatial     # 更新缩放后坐标

    print(f"空间坐标已从范围 {min_vals} - {max_vals} 缩放到 {target_min} - {target_max}")
    return adata


if __name__ == '__main__':
    vocab_path = "project1/spatial_data/spatial_data/new_vocab.json"
    if os.path.exists(vocab_path):
        print(f"从 {vocab_path} 加载基因索引...")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = None
        print(f"警告: 未找到词汇表文件 {vocab_path}")

    # 配置参数
    config = {
        'normalize':10000, # 是否进行归一化
        'is_bin': False,   # 是否进行bin处理
        'bins': 50,
        'is_mask':False,    # 模型内部是否掩码
        'c': 512,           # 最大基因长度
        'depth':512,
        'h': 14,            # 高度
        'w': 14,            # 宽度
        'patch_size': 1,     # 块大小
        'emb_dim': 256,      # 嵌入维度
        'en_dim': 256,       # 编码器维度
        'de_dim': 256,       # 解码器维度
        'mlp1_depth': 2,     # MLP1深度
        'mlp2_depth': 4,     # MLP2深度
        'mask_ratio': 0.0,   # 掩码比例
        'lr': 1e-5,           # 学习率
        'weight_decay': 0.05, # 权重衰减
        'batch_size': 32,     # 批次大小
        'num_workers': 4,     # 数据加载工作进程数
        'epochs': 30,        # 训练轮数
        'mask_ratio_list': [0.0], # 掩码比例列表
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': vocab["<pad>"] if vocab else 0,  # 填充ID
        'num_genes': max(vocab.values()) + 1 if vocab else 1000, # 基因数量 (包括pad)
        'model_output_dir': 'project1/model_outputs/maskdual_valid128', # 模型输出目录
        'model_type': 'transformer', # 模型类型
        'model_path': 'project1/model_outputs/maskdual_valid128/checkpoint_epoch_70.pth',
        'batch_size': 32,
    }
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = ad.read_h5ad(h5ad_path)
    log1p_with_backup(adata)
    if torch.cuda.is_available():
        device = get_least_used_gpu()
        print(f"使用GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")

    # 初始化模型
    model = MAEModel(config).to(device)
    
    model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])
    
    # 冻结编码器参数
    for param in model.encoder.parameters():
        param.requires_grad = False
    '''
    for param in model.decoder.parameters():
        param.requires_grad = False
    '''
    model = fine_tune_model(model, adata, vocab, config['epochs'], config, device)
    state = {
        'model_state_dict': model.state_dict(),
    }
    # 创建输出目录
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # 保存最新检查点
    checkpoint_path = os.path.join(config['model_output_dir'], "model0+30.pth")
    torch.save(state, checkpoint_path)
