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
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random


class SpatialMLPClassifier(nn.Module):
    """
    MLP分类器模块，用于细胞类型分类
    """
    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=5, dropout=0.1):
        super(SpatialMLPClassifier, self).__init__()
        layers = []
        
        # 构建MLP层
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


def create_spatial_grid(center_coord, coords, expr_matrix, h, w, c, gene_names, vocab):
    """
    创建空间网格和基因选择
    """
    x_center, y_center = center_coord
    
    # 计算网格边界
    half_w = w / 2
    half_h = h / 2
    x_min = x_center - half_w
    x_max = x_center + half_w
    y_min = y_center - half_h
    y_max = y_center + half_h
    
    # 创建网格坐标系统
    x_grid = np.linspace(x_min, x_max, w, endpoint=False)
    y_grid = np.linspace(y_min, y_max, h, endpoint=False)
    
    # 计算网格中心点
    grid_centers = np.array([(x + (x_grid[1]-x_grid[0])/2, 
                            y + (y_grid[1]-y_grid[0])/2) 
                           for y in y_grid for x in x_grid])
    
    # 创建KDTree用于快速查找
    grid_tree = KDTree(grid_centers)
    
    # 查找范围内的细胞
    in_range_indices = []
    for i, (x, y) in enumerate(coords):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            in_range_indices.append(i)
    
    n_spots = len(in_range_indices)
    n_genes_adata = expr_matrix.shape[1]
    
    # 初始化网格表达矩阵
    grid_expr = np.zeros((h, w, n_genes_adata))
    
    # 将绝对坐标转换为相对坐标（相对于网格左下角）
    relative_coords = []
    for cell_idx in in_range_indices:
        x_abs, y_abs = coords[cell_idx]
        # 计算相对于网格左下角的坐标[2,6](@ref)
        x_rel = x_abs - x_min
        y_rel = y_abs - y_min
        relative_coords.append((x_rel, y_rel))
    
    # 确保中心坐标填充到中心网格
    center_grid_y = h // 2
    center_grid_x = w // 2
    
    # 将细胞填充到最近网格
    for idx, cell_idx in enumerate(in_range_indices):
        cell_coord = coords[cell_idx]
        
        # 如果是中心坐标，直接分配到中心网格
        if abs(cell_coord[0] - x_center) < 1e-6 and abs(cell_coord[1] - y_center) < 1e-6:
            grid_y = center_grid_y
            grid_x = center_grid_x
        else:
            # 使用相对坐标进行网格分配[6,8](@ref)
            rel_x, rel_y = relative_coords[idx]
            grid_x = int(rel_x / (x_grid[1] - x_grid[0]))
            grid_y = int(rel_y / (y_grid[1] - y_grid[0]))
            
            # 确保网格索引在有效范围内
            grid_x = max(0, min(w - 1, grid_x))
            grid_y = max(0, min(h - 1, grid_y))
        
        grid_expr[grid_y, grid_x, :] = expr_matrix[cell_idx, :]
    
    # 计算高变基因
    if n_spots > 0:
        spot_expr = expr_matrix[in_range_indices, :]
        var_per_gene = np.var(spot_expr, axis=0)
    else:
        var_per_gene = np.zeros(n_genes_adata)
    
    # 选择基因
    if n_genes_adata >= c:
        # 加权随机选择
        var_per_gene = np.where(var_per_gene == 0, 1e-10, var_per_gene)
        selection_probs = var_per_gene / np.sum(var_per_gene)
        select_count = min(c, n_genes_adata)
        selected_gene_indices = np.random.choice(
            n_genes_adata, size=select_count, replace=False, p=selection_probs
        )
    else:
        selected_gene_indices = np.arange(n_genes_adata)
    
    # 创建表达矩阵和gene_ids
    reduced_expr = np.zeros((h, w, c))
    if len(selected_gene_indices) > 0:
        reduced_expr[:, :, :len(selected_gene_indices)] = grid_expr[:, :, selected_gene_indices]
    
    gene_ids_list = []
    for i in range(c):
        if i < len(selected_gene_indices):
            gene_name = gene_names[selected_gene_indices[i]]
            gene_id = vocab.get(gene_name, vocab.get("<unk>", vocab["<pad>"]))
        else:
            gene_id = vocab["<pad>"]
        gene_ids_list.append(gene_id)
    
    return reduced_expr, gene_ids_list
def prepare_training_data(adata, vocab, config, device):
    """
    准备训练数据
    """
    # 提取坐标和表达矩阵
    coords = adata.obsm['spatial']
    expr_matrix = adata.X
    if sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()
    
    n_cells = adata.n_obs
    gene_names = adata.var_names
    
    # 修改1: 从config获取标签列名[1](@ref)
    label_column = config.get('cell_type', 'cell_type')  # 默认为'cell_type'
    
    # 检查标签列是否存在
    if label_column not in adata.obs:
        raise ValueError(f"Label column '{label_column}' not found in adata.obs")
    
    # 获取标签数据并确保为字符串类型[1](@ref)
    labels = adata.obs[label_column].astype(str).values
    
    # 使用LabelEncoder将字符串标签转换为数值标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        range(n_cells), test_size=config.get('val_ratio', 0.2), random_state=42
    )
    
    return coords, expr_matrix, gene_names, encoded_labels, label_encoder, train_indices, val_indices

def f1(model, adata, vocab, epochs, config, device):
    """
    主训练函数：结合基因表达重建和细胞类型分类
    """
    # 配置参数
    h = config['h']
    w = config['w']
    c = config['c']
    batch_size = config.get('batch_size', 1)
    lr = config.get('lr', 1e-6)
    val_ratio = config.get('val_ratio', 0.2)
    
    # 准备数据
    coords, expr_matrix, gene_names, labels, label_encoder, train_indices, val_indices = prepare_training_data(
        adata, vocab, config, device
    )
    
    n_cells = len(train_indices)
    num_classes = len(label_encoder.classes_)
    
    # 初始化MLP分类器
    mlp_classifier = SpatialMLPClassifier(
        input_size=config['emb_dim'],  # 根据实际编码维度调整
        hidden_sizes=[256, 128],
        num_classes=num_classes
    ).to(device)
    
    # 优化器
    main_optimizer = optim.Adam(model.parameters(), lr=lr)
    classifier_optimizer = optim.Adam(mlp_classifier.parameters(), lr=lr)
    
    # 训练模式
    model.to(device)
    model.train()
    mlp_classifier.train()
    # 训练循环
    for epoch in range(epochs):

        # 随机打乱训练集
        np.random.shuffle(train_indices)
        progress_bar = tqdm(total=len(train_indices), desc=f"Epoch {epoch+1}/{epochs}")
        
        total_loss = 0
        total_classification_loss = 0
        
        for batch_start in range(0, len(train_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(train_indices))
            batch_indices = train_indices[batch_start:batch_end]
            
            batch_expressions = []
            batch_gene_ids = []
            batch_labels = []
            
            # 准备批次数据
            for idx in batch_indices:
                center_coord = coords[idx]
                expression, gene_ids = create_spatial_grid(
                    center_coord, coords, expr_matrix, h, w, c, gene_names, vocab
                )
                
                # 转换为tensor
                expression_tensor = torch.tensor(expression, dtype=torch.float32).unsqueeze(0)
                expression_tensor = expression_tensor.permute(0, 3, 1, 2)
                gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long).unsqueeze(0)
                
                batch_expressions.append(expression_tensor)
                batch_gene_ids.append(gene_ids_tensor)
                batch_labels.append(labels[idx])
            
            # 合并批次
            batch_expression = torch.cat(batch_expressions, dim=0).to(device)
            batch_gene_id = torch.cat(batch_gene_ids, dim=0).to(device)
            batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # 前向传播
            model_output = model(batch_expression, batch_gene_id)
            recon = model_output[0] if isinstance(model_output, tuple) else model_output['recon']
            mask = model_output[2] if isinstance(model_output, tuple) else model_output['combined_mask']
            enc_output = model_output[3] if isinstance(model_output, tuple) else model_output['enc_output_spatial']
            
            # 计算重建损失
            #reconstruction_loss = model.loss_function(batch_expression, recon, mask)
            
            # 分类损失
            # 使用中心细胞的编码输出进行分类
            classifier_input = enc_output[:, :, h//2, w//2]
            classifier_output = mlp_classifier(classifier_input)  # 中心位置
            classification_loss = nn.CrossEntropyLoss()(classifier_output, batch_labels_tensor)
            
            # 总损失
            loss = classification_loss * config.get('cls_weight', 1.0) #+ reconstruction_loss
            total_classification_loss += classification_loss.item()
            total_loss += loss.item()
            
            # 反向传播
            main_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss.backward()
            main_optimizer.step()
            classifier_optimizer.step()
            
            # 更新进度条
            progress_bar.update(len(batch_indices))
            avg_loss = total_loss / ((batch_end + batch_size - 1) // batch_size)
            avg_cls_loss = total_classification_loss / ((batch_end + batch_size - 1) // batch_size)
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}", 
                "ClsLoss": f"{avg_cls_loss:.4f}"
            })
        
        progress_bar.close()
        
        # 验证集评估
        if val_indices:
            validate_model(model, mlp_classifier, coords, expr_matrix, labels, 
                         val_indices, h, w, c, gene_names, vocab, device)
    
    return model, mlp_classifier, label_encoder

def validate_model(model, classifier, coords, expr_matrix, labels, val_indices, 
                  h, w, c, gene_names, vocab, device):
    """
    验证模型性能
    """
    model.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx in val_indices:
            center_coord = coords[idx]
            expression, gene_ids = create_spatial_grid(
                center_coord, coords, expr_matrix, h, w, c, gene_names, vocab
            )
            
            # 转换为tensor
            expression_tensor = torch.tensor(expression, dtype=torch.float32).unsqueeze(0)
            expression_tensor = expression_tensor.permute(0, 3, 1, 2).to(device)
            gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            # 前向传播
            model_output = model(expression_tensor, gene_ids_tensor)
            enc_output = model_output[3] if isinstance(model_output, tuple) else model_output['enc_output_spatial']
            
            # 分类预测
            classifier_input = enc_output[:, :, h//2, w//2]
            classifier_output = classifier(classifier_input)
            pred = torch.argmax(classifier_output, dim=1)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels[idx])
    
    # 计算准确率
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    model.train()
    classifier.train()
    
    return accuracy

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
        'is_bin': False,   # 是否进行bin处理
        'bins': 50,
        'is_mask':False,    # 模型内部是否掩码
        'c': 512,           # 最大基因长度
        'depth':512,
        'h': 14,            # 高度
        'w': 14,            # 宽度
        'patch_size': 1,     # 块大小
        'emb_dim': 512,      # 嵌入维度
        'en_dim': 512,       # 编码器维度
        'de_dim': 512,       # 解码器维度
        'mlp1_depth': 2,     # MLP1深度
        'mlp2_depth': 4,     # MLP2深度
        'mask_ratio': 0.0,   # 掩码比例
        'lr': 0.0001,           # 学习率
        'weight_decay': 0.05, # 权重衰减
        'batch_size': 32,     # 批次大小
        'num_workers': 4,     # 数据加载工作进程数
        'epochs': 100,        # 训练轮数
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': vocab["<pad>"] if vocab else 0,  # 填充ID
        'num_genes': max(vocab.values()) + 1 if vocab else 1000, # 基因数量 (包括pad)
        'model_output_dir': 'project1/model_outputs', # 模型输出目录
        'model_type': 'transformer', # 模型类型
        'model_path': 'project1/model_outputs/maskdual_valid512/best_model.pth',
        'batch_size': 32,
        'cell_type': 'sce.layer_guess'
    }
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = ad.read_h5ad(h5ad_path)
    log1p_with_backup(adata)
    adata = scale_spatial_coordinates(adata,target_range=(0, 100))
    if torch.cuda.is_available():
        device = get_least_used_gpu()
        print(f"使用GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")


    
    # 初始化模型
    model = MAEModel(config).to(device)
    if config['model_path']:
        model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])

    trained_model, trained_classifier, le = f1(model, adata, vocab, 100, config, device)

    state = {
        'model_state_dict': model.state_dict(),
    }
    # 创建输出目录
    #os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # 保存最新检查点
    #checkpoint_path = os.path.join(config['model_output_dir'], "checkpoint_epoch_7+30.pth")
    #torch.save(state, checkpoint_path)
