import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score  
from utils.utils import get_least_used_gpu


def train_mlp_on_adata(adata, obsm_key, obs_label_key, test_size=0.2, 
                       hidden_dims=[512, 256, 128], batch_size=32, epochs=100, 
                       lr=0.001, device='auto', knn_smooth_k=0,
                       warm_up_epochs=10, warm_up_factor=0.1, structure='mlp',
                       knn_k=43, predict_key='mlp_prediction', save_path=None):
    """
    在空间转录组数据上训练MLP模型，支持KNN平滑预处理和warm up策略
    
    Args:
        adata (AnnData): 空间转录组数据对象
        obsm_key (str): obsm中的列名或数字(表示top n高变基因)
        obs_label_key (str): obs中存储标签的列名
        test_size (float): 测试集比例
        hidden_dims (list): 隐藏层维度
        batch_size (int): 批大小
        epochs (int): 训练轮数
        lr (float): 学习率
        device (str): 计算设备
        knn_smooth_k (int): 基于特征空间的KNN平滑的邻居数量
        warm_up_epochs (int): warm up阶段的epoch数
        warm_up_factor (float): warm up初始学习率比例
        structure (str): 模型结构
        knn_k (int): 基于空间坐标的KNN平滑的邻居数量
        predict_key (str): 预测结果保存到obs中的键名
        save_path (str): 保存AnnData对象的路径
    
    Returns:
        tuple: (训练好的模型, 测试集特征, 测试集标签, 训练历史, 标签编码器, 最终测试准确率, 最终测试ARI)
    """
    feature_info = {}  # 在特征提取部分之前初始化
    # 设置设备
    if device == 'auto':
        device = get_least_used_gpu() if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 准备特征数据
    if obsm_key.isdigit():
        # 使用top k高变基因
        n_top_genes = int(obsm_key)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        adata = adata[:, adata.var.highly_variable]
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        feature_type = f'top_{n_top_genes}_highly_variable_genes'
        
        # 应用基于特征空间的KNN平滑
        if knn_smooth_k is not None and knn_smooth_k > 0:
            print(f"Applying feature-based KNN smoothing with k={knn_smooth_k}")
            
            knn = NearestNeighbors(n_neighbors=knn_smooth_k+1)  # 包括自身
            knn.fit(X)
            distances, indices = knn.kneighbors(X)
            # 计算邻居均值进行平滑
            X = np.mean(X[indices], axis=1)
        feature_info['type'] = 'highly_variable_genes'
        feature_info['gene_names'] = adata.var_names.tolist()  # 保存基因名
        feature_info['n_top_genes'] = int(obsm_key)
    else:
        # 使用obsm中的特征
        if obsm_key not in adata.obsm:
            raise ValueError(f"obsm_key '{obsm_key}' not found in adata.obsm")
        X = adata.obsm[obsm_key]
        feature_type = f'obsm_{obsm_key}'
        feature_info['type'] = 'obsm'
        feature_info['obsm_key'] = obsm_key
    print(f"Using features from: {feature_type}")
    print(f"Feature matrix shape: {X.shape}")
    
    # 新增：应用基于空间坐标的KNN平滑
    if knn_k > 0:
        print(f"Applying spatial KNN smoothing with k={knn_k} based on 'source_file' groups and spatial coordinates")
        
        # 检查必要的列是否存在
        if 'source_file' not in adata.obs:
            raise ValueError("'source_file' not found in adata.obs. Required for spatial KNN smoothing.")
        if 'spatial' not in adata.obsm:
            raise ValueError("'spatial' not found in adata.obsm. Required for spatial coordinates.")
        
        # 获取空间坐标
        spatial_coords = adata.obsm['spatial']
        
        # 初始化平滑后的特征矩阵
        X_smoothed = np.zeros_like(X)
        
        # 按source_file分组处理
        unique_files = adata.obs['source_file'].unique()
        
        for file in unique_files:
            # 获取当前文件对应的细胞索引
            file_mask = adata.obs['source_file'] == file
            file_indices = np.where(file_mask)[0]
            
            if len(file_indices) == 0:
                continue
                
            # 提取当前文件组的空间坐标和特征
            file_spatial = spatial_coords[file_indices]
            file_X = X[file_indices]
            
            # 计算实际可用的k值（排除自身后）
            n_cells_in_file = len(file_indices)
            k_actual = min(knn_k, n_cells_in_file - 1)  # 最多为n-1，因为要排除自身
            
            if k_actual < 1:
                # 如果没有邻居可用，直接使用原始特征
                X_smoothed[file_indices] = file_X
                continue
                
            # 在当前文件组内构建KDTree进行最近邻搜索
            from sklearn.neighbors import NearestNeighbors
            knn_spatial = NearestNeighbors(n_neighbors=k_actual+1)  # +1是为了包含自身
            knn_spatial.fit(file_spatial)
            
            # 查找每个细胞的k_actual+1个最近邻居（包括自身）
            distances, indices = knn_spatial.kneighbors(file_spatial)
            
            # 对每个细胞计算邻居平均特征（排除自身）
            for i, local_idx in enumerate(range(len(file_indices))):
                # indices[i, 0]是自身，所以从1开始取k_actual个邻居
                neighbor_indices = indices[i, 1:1+k_actual]  # 排除自身
                
                if len(neighbor_indices) > 0:
                    # 计算邻居的平均特征
                    neighbor_features = file_X[neighbor_indices]
                    avg_features = np.mean(neighbor_features, axis=0)
                else:
                    # 如果没有邻居，使用自身特征
                    avg_features = file_X[i]
                
                # 将结果存回全局索引位置
                global_idx = file_indices[i]
                X_smoothed[global_idx] = avg_features
        
        # 用平滑后的特征替换原始特征
        X = X_smoothed
        print(f"After spatial KNN smoothing: {X.shape}")
    
    if obs_label_key not in adata.obs:
        raise ValueError(f"obs_label_key '{obs_label_key}' not found in adata.obs")

    y = adata.obs[obs_label_key].values

    # 将标签转换为字符串并处理NaN值
    y_str = y.astype(str)

    # 首先同时处理标签NaN和特征NaN，确保维度一致
    # 创建标签有效掩码
    label_nan_mask = pd.notna(y) & (y_str != 'nan')

    # 检查特征中是否有NaN值
    if np.isnan(X).any():
        print(f"发现特征矩阵中有 {np.isnan(X).sum()} 个NaN值")
        # 创建特征NaN掩码：检查每一行是否有NaN
        feature_nan_mask = ~np.isnan(X).any(axis=1)
        print(f"特征NaN掩码将删除 {np.sum(~feature_nan_mask)} 个样本")
    else:
        feature_nan_mask = np.ones(len(X), dtype=bool)

    # 合并标签和特征掩码
    valid_indices = label_nan_mask & feature_nan_mask

    # 应用合并后的掩码
    X = X[valid_indices]
    y_str = y_str[valid_indices]

    print(f"After removing NaN labels and features: {X.shape[0]} samples remaining")
    print(f"删除的样本数量: {len(label_nan_mask) - np.sum(valid_indices)}")
    
    # 编码字符串标签为数字
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    label_classes = le.classes_
    print(f"Encoded labels: {dict(enumerate(label_classes))}")
    print(f"处理前X的nan数量：{np.isnan(X).sum()}")
    
    # 划分训练测试集 - 修改为返回索引以便标记每个spot的分组
    valid_indices_positions = np.where(valid_indices)[0]  # 获取有效样本的原始位置
    train_idx, test_idx = train_test_split(
        np.arange(len(valid_indices_positions)),  # 有效样本的相对索引
        test_size=test_size, 
        stratify=y, 
        random_state=42
    )
    
    # 映射回原始adata的索引
    original_train_indices = valid_indices_positions[train_idx]
    original_test_indices = valid_indices_positions[test_idx]
    
    # 功能1: 将每个spot被分为训练集还是测试集保存到obs属性中
    adata.obs['data_split'] = 'unused'  # 初始化，标记未使用的样本
    adata.obs.iloc[original_train_indices, adata.obs.columns.get_loc('data_split')] = 'train'
    adata.obs.iloc[original_test_indices, adata.obs.columns.get_loc('data_split')] = 'test'
    print(f"标记数据分割: {np.sum(adata.obs['data_split'] == 'train')} 个训练样本, "
          f"{np.sum(adata.obs['data_split'] == 'test')} 个测试样本, "
          f"{np.sum(adata.obs['data_split'] == 'unused')} 个未使用样本")
    
    # 提取训练测试集特征
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        structure=structure,
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 训练循环
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Warm up策略：在前warm_up_epochs个epoch中线性增加学习率
        if epoch < warm_up_epochs:
            warm_up_lr = lr * warm_up_factor + (lr - lr * warm_up_factor) * epoch / warm_up_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_up_lr
        
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # 计算训练准确率
        train_acc = 100 * correct / total
        train_loss = epoch_loss / len(train_loader)
        
        # 计算测试准确率
        test_acc, test_ari = evaluate_model(model, X_test_tensor, y_test_tensor)
        
        # 更新学习率（warm up阶段后使用scheduler）
        if epoch >= warm_up_epochs:
            scheduler.step(epoch_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'LR: {current_lr:.6f}, '
                  f'Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Test Acc: {test_acc:.2f}%, '
                  f'Test ARI: {test_ari:.2f}')
    
    # 最终评估
    final_test_acc, final_test_ari = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f'Final Test Accuracy: {final_test_acc:.2f}%, '
          f'Final Test ARI: {final_test_ari:.2f}')
    
    return model, X_test, y_test, history, le, scaler, feature_info, final_test_acc, final_test_ari

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / y_test.size(0)
        y_test_np = y_test.cpu().numpy()  # 真实标签
        predicted_np = predicted.cpu().numpy()  # 预测标签
        ari = adjusted_rand_score(y_test_np, predicted_np)  # 计算ARI
    return accuracy, ari  

class MLPClassifier(nn.Module):
    """支持MLP和Transformer结构的分类器"""
    def __init__(self, input_dim, hidden_dims, num_classes, structure='mlp'):
        super(MLPClassifier, self).__init__()
        self.structure = structure
        self.input_dim = input_dim
        self.num_classes = num_classes

        if structure == 'mlp':
            # 原始MLP结构：全连接层+ReLU+Dropout
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)

        elif structure == 'transformer':
            # Transformer结构：使用编码器处理序列数据
            self.d_model = 512  # 嵌入维度
            self.nhead = 8       # 注意力头数
            self.num_layers = 2  # Transformer编码器层数
            self.dim_feedforward = 512  # 前馈网络中间维度

            self.input_projection = nn.Linear(input_dim, self.d_model)
            self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))
            self.pos_embedding = nn.Parameter(torch.randn(1, 2, self.d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.output_layer = nn.Linear(self.d_model, num_classes)

        else:
            raise ValueError("structure参数必须是'mlp'或'transformer'")

    def forward(self, x):
        if self.structure == 'mlp':
            return self.network(x)
        
        elif self.structure == 'transformer':
            batch_size = x.shape[0]
            
            x_proj = self.input_projection(x)
            x_proj = x_proj.unsqueeze(1)
            
            class_tokens = self.class_token.expand(batch_size, -1, -1)
            sequence = torch.cat([class_tokens, x_proj], dim=1)
            sequence = sequence + self.pos_embedding
            
            encoded = self.transformer_encoder(sequence)
            class_output = encoded[:, 0, :]
            
            return self.output_layer(class_output)

def predict_mlp_on_adata(model, adata, feature_info, scaler, le, 
                         knn_smooth_k=0, knn_k=0, device='auto', 
                         output_obs_key='predicted_labels', save_path=None):
    """
    使用训练好的MLP模型预测adata的标签，并保存到obs中
    
    Args:
        model: 训练好的MLP模型
        adata: 要预测的AnnData对象
        feature_info: 训练时保存的特征信息字典，包含特征类型、基因列表等
        scaler: 训练时使用的StandardScaler对象
        le: 训练时使用的LabelEncoder对象
        knn_smooth_k: 基于特征空间的KNN平滑k值，与训练时一致
        knn_k: 基于空间坐标的KNN平滑k值，与训练时一致
        device: 计算设备，'auto'为自动选择
        output_obs_key: obs中保存预测标签的列名
        save_path: 保存adata的路径，如果为None则不保存
        
    Returns:
        adata: 预测后的AnnData对象，标签保存在obs[output_obs_key]中
    """
    # 设置设备
    if device == 'auto':
        device = get_least_used_gpu() if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print("开始预测新adata的标签...")
    
    # 1. 特征提取：与训练时一致
    if feature_info['type'] == 'highly_variable_genes':
        # 使用训练时的高变基因列表
        required_genes = feature_info['gene_names']
        available_genes = adata.var_names
        # 检查基因是否匹配
        missing_genes = set(required_genes) - set(available_genes)
        if missing_genes:
            raise ValueError(f"新数据缺失训练时使用的基因: {len(missing_genes)}个")
        # 按训练时的基因顺序选择特征
        adata_sub = adata[:, required_genes]
        X = adata_sub.X.toarray() if hasattr(adata_sub.X, 'toarray') else adata_sub.X
        print(f"使用高变基因特征，形状: {X.shape}")
        
    elif feature_info['type'] == 'obsm':
        # 使用obsm中的特征
        obsm_key = feature_info['obsm_key']
        if obsm_key not in adata.obsm:
            raise ValueError(f"obsm键 '{obsm_key}' 不在新数据中")
        X = adata.obsm[obsm_key]
        print(f"使用obsm特征 '{obsm_key}'，形状: {X.shape}")
    else:
        raise ValueError(f"未知特征类型: {feature_info['type']}")
    
    # 2. 应用基于特征空间的KNN平滑（与训练时一致）
    if knn_smooth_k > 0:
        print(f"应用特征空间KNN平滑, k={knn_smooth_k}")
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=knn_smooth_k+1)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        X = np.mean(X[indices], axis=1)
        print(f"平滑后特征形状: {X.shape}")
    
    # 3. 应用基于空间坐标的KNN平滑（与训练时一致）
    if knn_k > 0:
        print(f"应用空间坐标KNN平滑, k={knn_k}")
        if 'spatial' not in adata.obsm:
            raise ValueError("新数据缺少空间坐标 ('spatial' not in obsm)")
        if 'source_file' not in adata.obs:
            raise ValueError("新数据缺少 'source_file' 分组信息")
        
        spatial_coords = adata.obsm['spatial']
        X_smoothed = np.zeros_like(X)
        unique_files = adata.obs['source_file'].unique()
        
        for file in unique_files:
            file_mask = adata.obs['source_file'] == file
            file_indices = np.where(file_mask)[0]
            if len(file_indices) == 0:
                continue
                
            file_spatial = spatial_coords[file_indices]
            file_X = X[file_indices]
            n_cells = len(file_indices)
            k_actual = min(knn_k, n_cells - 1)
            
            if k_actual < 1:
                X_smoothed[file_indices] = file_X
                continue
                
            from sklearn.neighbors import NearestNeighbors
            knn_spatial = NearestNeighbors(n_neighbors=k_actual+1)
            knn_spatial.fit(file_spatial)
            distances, indices = knn_spatial.kneighbors(file_spatial)
            
            for i, local_idx in enumerate(range(len(file_indices))):
                neighbor_indices = indices[i, 1:1+k_actual]
                if len(neighbor_indices) > 0:
                    neighbor_features = file_X[neighbor_indices]
                    avg_features = np.mean(neighbor_features, axis=0)
                else:
                    avg_features = file_X[i]
                global_idx = file_indices[i]
                X_smoothed[global_idx] = avg_features
        
        X = X_smoothed
        print(f"空间平滑后特征形状: {X.shape}")
    
    # 4. 处理NaN值：用特征均值填充（与训练时移除策略不同，预测时需保留所有细胞）
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"警告: 特征中有 {nan_count} 个NaN值，使用列均值填充")
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
    
    # 5. 标准化：使用训练时的scaler
    X = scaler.transform(X)  # 注意：使用transform而非fit_transform
    print(f"标准化后特征形状: {X.shape}")
    
    # 6. 转换为Tensor并预测
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_labels = predicted.cpu().numpy()
    
    # 7. 解码标签
    predicted_labels_decoded = le.inverse_transform(predicted_labels)
    
    # 8. 保存到obs
    adata.obs[output_obs_key] = pd.Categorical(predicted_labels_decoded)
    print(f"预测完成，标签已保存到 adata.obs['{output_obs_key}']")
    
    # 9. 保存文件（可选）
    if save_path:
        adata.write(save_path)
        print(f"数据已保存到: {save_path}")
    
    return adata

def main():
    """
    自动训练多个obsm_key的模型，并将结果保存到文本文件
    """
    # 定义要测试的obsm_key列表
    obsm_keys = [
        '512',
        'X_emb_scGPT',
        'X_emb_scGPTspatial',
        'X_emb512_model40_8',
        'X_emb512_model40_16',
        'X_emb512_model40_24',
        'X_emb512_model40_30',
    ]
    
    # 结果保存文件
    result_file = 'obsm_key_results.txt'
    
    # 存储所有结果的列表
    all_results = []
    
    # 加载数据（只加载一次，提高效率）
    directory_path = 'project1/spatial_data/down_stream_data/Human_tonsil'
    obs_label_key= 'final_annot'
    h5ad_files = [f for f in os.listdir(directory_path) if f.endswith('.h5ad')]
    train_files = [f for f in h5ad_files]
    adata_train_list = []
    
    for train_file in train_files:
        file_path = os.path.join(directory_path, train_file)
        print(f"Loading training file: {file_path}")
        adata = ad.read_h5ad(file_path)

        # 给 spot 名加前缀
        prefix = train_file.replace('.h5ad', '')
        adata.obs_names = f"{prefix}_" + adata.obs_names

        adata.obs['source_file'] = train_file   # 保留来源信息
        adata_train_list.append(adata)

    # 合并数据
    if len(adata_train_list) == 1:
        adata_train = adata_train_list[0]
    else:
        adata_train = adata_train_list[0].concatenate(
            adata_train_list[1:],
            batch_key='source_file',
            index_unique=None
        )
    
    print(f"合并后的数据形状: {adata_train.shape}")
    num_repeats = 5
    
    # 遍历所有obsm_key进行训练
    for obsm_key in obsm_keys:
        print(f"\n{'='*60}")
        print(f"开始训练 obsm_key: {obsm_key}")
        print(f"{'='*60}")
        
        # 根据obsm_key设置knn_k参数
        if obsm_key == 'X_emb_scGPTspatial':
            knn_k_val = 16  # 特殊处理
            print(f"检测到特殊obsm_key '{obsm_key}'，设置 knn_k=16")
        else:
            knn_k_val = 0   # 默认值
            print(f"使用默认 knn_k=0")
        
        # 存储每次实验的结果
        accuracy_results = []
        ari_results = []
        
        for repeat in range(num_repeats):
            print(f"\n--- 第 {repeat + 1}/{num_repeats} 次重复实验 ---")
            try:
                # 训练模型
                model, X_test, y_test, history, le, scaler, feature_info, final_acc, final_ari = train_mlp_on_adata(
                    adata_train, 
                    obsm_key=obsm_key,
                    obs_label_key=obs_label_key,
                    test_size=0.8,
                    epochs=500,
                    warm_up_epochs=20,
                    warm_up_factor=0.1,
                    structure='mlp',
                    knn_k=knn_k_val,
                    predict_key=f'pred_{obsm_key}',
                    save_path=None  # 不自动保存，避免文件冲突
                )
                
                # 将结果添加到临时列表，不要添加到all_results
                accuracy_results.append(final_acc)
                ari_results.append(final_ari)
                
                print(f"第 {repeat + 1} 次实验完成: 准确率={final_acc:.2f}%, ARI={final_ari:.4f}")
                
            except Exception as e:
                print(f"第 {repeat + 1} 次实验失败: {str(e)}")
                continue
        
        # 计算统计结果
        if accuracy_results:
            avg_accuracy = np.mean(accuracy_results)
            std_accuracy = np.std(accuracy_results)
            
            avg_ari = np.mean(ari_results)
            std_ari = np.std(ari_results)
            
            success_count = len(accuracy_results)
            
            print(f"\n{obsm_key} 的 {success_count}/{num_repeats} 次实验统计结果:")
            print(f"准确率: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
            print(f"ARI: {avg_ari:.4f} ± {std_ari:.4f}")
            
            # 同时提供范围信息
            accuracy_range = f"({np.min(accuracy_results):.2f}% - {np.max(accuracy_results):.2f}%)"
            ari_range = f"({np.min(ari_results):.4f} - {np.max(ari_results):.4f})"
            print(f"准确率范围: {accuracy_range}")
            print(f"ARI范围: {ari_range}")
            
            # 保存统计结果到all_results（只添加一次）
            result = {
                'obsm_key': obsm_key,
                'num_repeats': num_repeats,
                'success_count': success_count,
                'accuracy_mean_std': (avg_accuracy, std_accuracy),
                'ari_mean_std': (avg_ari, std_ari),
                'accuracy_range': (np.min(accuracy_results), np.max(accuracy_results)),
                'ari_range': (np.min(ari_results), np.max(ari_results)),
                'all_accuracies': accuracy_results,
                'all_aris': ari_results,
                'knn_k_used': knn_k_val
            }
        else:
            # 处理所有实验失败的情况
            print(f"{obsm_key} 的所有实验均失败")
            result = {
                'obsm_key': obsm_key,
                'num_repeats': num_repeats,
                'success_count': 0,
                'accuracy_mean_std': (0.0, 0.0),
                'ari_mean_std': (0.0, 0.0),
                'accuracy_range': (0.0, 0.0),
                'ari_range': (0.0, 0.0),
                'all_accuracies': [],
                'all_aris': [],
                'knn_k_used': knn_k_val
            }
        
        # 只将统计结果添加到all_results（每个obsm_key添加一次）
        all_results.append(result)
    
    # 将结果写入文件
    print(f"\n{'='*60}")
    print("正在将结果写入文件...")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("obsm_key\tnum_repeats\tsuccess_count\taccuracy_mean_std\tari_mean_std\taccuracy_range\tari_range\tknn_k_used\tall_accuracies\tall_aris\n")
        
        for result in all_results:
            accuracies_str = ','.join([f"{acc:.2f}" for acc in result['all_accuracies']])
            aris_str = ','.join([f"{ari:.4f}" for ari in result['all_aris']])
            acc_mean, acc_std = result['accuracy_mean_std']
            ari_mean, ari_std = result['ari_mean_std']
            acc_min, acc_max = result['accuracy_range']
            ari_min, ari_max = result['ari_range']
            
            f.write(f"{result['obsm_key']}\t{result['num_repeats']}\t{result['success_count']}\t"
                    f"{acc_mean:.2f}±{acc_std:.2f}\t{ari_mean:.4f}±{ari_std:.4f}\t"
                    f"{acc_min:.2f}-{acc_max:.2f}\t{ari_min:.4f}-{ari_max:.4f}\t"
                    f"{result['knn_k_used']}\t{accuracies_str}\t{aris_str}\n")
    
    print(f"结果已保存到: {result_file}")
    
    # 打印汇总结果
    print(f"\n{'='*110}")
    print(directory_path)
    print("训练结果汇总 (多次重复实验) - 格式: 平均值 ± 标准差")
    print(f"{'obsm_key':<30} {'Reps':<6} {'Success':<8} {'Accuracy':<15} {'ARI':<20} {'Acc Range':<15} {'ARI Range':<15}")
    print("-" * 110)
    
    for result in all_results:
        acc_mean, acc_std = result['accuracy_mean_std']
        ari_mean, ari_std = result['ari_mean_std']
        acc_min, acc_max = result['accuracy_range']
        ari_min, ari_max = result['ari_range']
        
        accuracy_display = f"{acc_mean:.2f}±{acc_std:.2f}%"
        ari_display = f"{ari_mean:.4f}±{ari_std:.4f}"
        acc_range = f"{acc_min:.2f}-{acc_max:.2f}"
        ari_range = f"{ari_min:.4f}-{ari_max:.4f}"
        
        print(f"{result['obsm_key']:<30} {result['num_repeats']:<6} {result['success_count']:<8} "
              f"{accuracy_display:<15} {ari_display:<20} {acc_range:<15} {ari_range:<15}")
    
    print(f"{'='*110}")
    
    return all_results
if __name__ == "__main__":
    main()