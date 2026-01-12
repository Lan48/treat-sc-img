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


def train_mlp_on_adata(adata, obsm_key, obs_label_key, test_size=0.2, 
                       hidden_dims=[512, 256, 128], batch_size=32, epochs=100, 
                       lr=0.001, device='auto', knn_smooth_k=0,
                       warm_up_epochs=10, warm_up_factor=0.1, structure='mlp'):
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
        knn_smooth_k (int): KNN平滑的邻居数量，默认为5。仅当使用top n高变基因时生效[1](@ref)。
        warm_up_epochs (int): warm up阶段的epoch数
        warm_up_factor (float): warm up初始学习率比例
    
    Returns:
        tuple: (训练好的模型, 测试集特征, 测试集标签, 训练历史, 标签编码器)
    """
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 准备特征数据
    if obsm_key.isdigit():
        # 使用top k高变基因
        n_top_genes = int(obsm_key)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        adata = adata[:, adata.var.highly_variable]
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        feature_type = f'top_{n_top_genes}_highly_variable_genes'
        
        # 应用KNN平滑[1](@ref) - 修改后的版本
        if knn_smooth_k is not None and knn_smooth_k > 0:
            print(f"Applying KNN smoothing with k={knn_smooth_k} within same source_file")
            
            # 检查必要的字段是否存在
            if 'source_file' not in adata.obs:
                raise ValueError("'source_file' column not found in adata.obs for KNN smoothing")
            if 'spatial' not in adata.obsm:
                raise ValueError("'spatial' coordinates not found in adata.obsm for KNN smoothing")
            
            # 创建平滑后的矩阵
            X_smoothed = np.zeros_like(X)
            
            # 按source_file分组进行KNN平滑
            unique_files = adata.obs['source_file'].unique()
            
            for file_name in unique_files:
                # 获取当前source_file的细胞索引
                file_mask = adata.obs['source_file'] == file_name
                file_indices = np.where(file_mask)[0]
                
                if len(file_indices) == 0:
                    continue
                
                # 获取当前source_file的空间坐标
                spatial_coords = adata.obsm['spatial'][file_indices]
                
                # 如果细胞数量少于knn_smooth_k，调整k值
                current_k = min(knn_smooth_k, len(file_indices) - 1)
                if current_k <= 0:
                    # 如果只有一个细胞，直接使用原始表达量
                    X_smoothed[file_indices] = X[file_indices]
                    continue
                
                # 在当前source_file内构建KNN
                knn = NearestNeighbors(n_neighbors=current_k+1)  # +1 包括自身
                knn.fit(spatial_coords)
                
                # 查找邻居
                distances, indices = knn.kneighbors(spatial_coords)
                
                # 对当前source_file的每个细胞进行平滑
                for i, idx in enumerate(file_indices):
                    # 获取邻居在原始adata中的索引
                    neighbor_indices = file_indices[indices[i]]
                    # 计算邻居的均值进行平滑
                    X_smoothed[idx] = np.mean(X[neighbor_indices], axis=0)
            
            X = X_smoothed
    else:
        # 使用obsm中的特征
        if obsm_key not in adata.obsm:
            raise ValueError(f"obsm_key '{obsm_key}' not found in adata.obsm")
        X = adata.obsm[obsm_key]
        feature_type = f'obsm_{obsm_key}'
    
    print(f"Using features from: {feature_type}")
    print(f"Feature matrix shape: {X.shape}")
    
    # 准备标签数据
    if obs_label_key not in adata.obs:
        raise ValueError(f"obs_label_key '{obs_label_key}' not found in adata.obs")
    
    y = adata.obs[obs_label_key].values
    
    # 将标签转换为字符串并处理NaN值
    y_str = y.astype(str)
    
    # 删除标签为'nan'的样本
    valid_indices = y_str != 'nan'
    X = X[valid_indices]
    y_str = y_str[valid_indices]
    
    print(f"After removing NaN labels: {X.shape[0]} samples remaining")
    
    # 编码字符串标签为数字
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    label_classes = le.classes_
    print(f"Encoded labels: {dict(enumerate(label_classes))}")
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
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
        test_acc = evaluate_model(model, X_test_tensor, y_test_tensor)
        
        # 更新学习率（warm up阶段后使用scheduler）
        if epoch >= warm_up_epochs:
            scheduler.step(epoch_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'LR: {current_lr:.6f}, '
                  f'Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Test Acc: {test_acc:.2f}%')
    
    # 最终评估
    final_test_acc = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f'Final Test Accuracy: {final_test_acc:.2f}%')
    
    return model, X_test, y_test, history, le
def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / y_test.size(0)
    return accuracy

class MLPClassifier(nn.Module):
    """支持MLP和Transformer结构的分类器"""
    def __init__(self, input_dim, hidden_dims, num_classes, structure='mlp'):
        super(MLPClassifier, self).__init__()
        self.structure = structure
        self.input_dim = input_dim
        self.num_classes = num_classes

        if structure == 'mlp':
            # 原始MLP结构：全连接层+ReLU+Dropout [1,5](@ref)
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
            # Transformer结构：使用编码器处理序列数据 [3,7](@ref)
            # 默认参数设置（可根据需要调整）
            self.d_model = 512  # 嵌入维度
            self.nhead = 8       # 注意力头数
            self.num_layers = 2  # Transformer编码器层数
            self.dim_feedforward = 512  # 前馈网络中间维度

            # 将输入向量投影到Transformer所需的嵌入空间 [3](@ref)
            self.input_projection = nn.Linear(input_dim, self.d_model)
            
            # 可学习的分类令牌（类似ViT的class token）[3](@ref)
            self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))
            
            # 位置编码（序列长度=1输入+1类令牌）[3](@ref)
            self.pos_embedding = nn.Parameter(torch.randn(1, 2, self.d_model))
            
            # Transformer编码器层 [3,8](@ref)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                batch_first=True  # 使用(batch, seq, feature)格式
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            
            # 输出分类层 [1](@ref)
            self.output_layer = nn.Linear(self.d_model, num_classes)

        else:
            raise ValueError("structure参数必须是'mlp'或'transformer'")

    def forward(self, x):
        if self.structure == 'mlp':
            return self.network(x)
        
        elif self.structure == 'transformer':
            batch_size = x.shape[0]
            
            # 1. 投影输入到d_model维度 [3](@ref)
            x_proj = self.input_projection(x)  # (batch_size, input_dim) -> (batch_size, d_model)
            x_proj = x_proj.unsqueeze(1)       # (batch_size, 1, d_model)
            
            # 2. 添加分类令牌和位置编码 [3](@ref)
            class_tokens = self.class_token.expand(batch_size, -1, -1)  # 扩展至batch大小
            sequence = torch.cat([class_tokens, x_proj], dim=1)  # (batch_size, 2, d_model)
            sequence = sequence + self.pos_embedding
            
            # 3. 通过Transformer编码器 [3,8](@ref)
            encoded = self.transformer_encoder(sequence)  # (batch_size, 2, d_model)
            
            # 4. 取分类令牌对应的输出作为最终特征 [3](@ref)
            class_output = encoded[:, 0, :]  # (batch_size, d_model)
            
            # 5. 输出分类结果
            return self.output_layer(class_output)  # (batch_size, num_classes)

# 3. 示例使用方式
def main():
    # 所有切片上随机划分训练测试
    directory_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC'
    # 获取目录下所有.h5ad文件 [1,5](@ref)
    h5ad_files = [f for f in os.listdir(directory_path) if f.endswith('.h5ad')]
    train_files = [f for f in h5ad_files]
    adata_train_list = []
    
    for train_file in train_files:
        file_path = os.path.join(directory_path, train_file)
        print(f"Loading training file: {file_path}")
        adata = ad.read_h5ad(file_path)
        adata.obs['source_file'] = train_file  # 标记来源文件
        adata_train_list.append(adata)
    
    # 合并训练集[7](@ref)
    if len(adata_train_list) == 1:
        adata_train = adata_train_list[0]
    else:
        adata_train = adata_train_list[0].concatenate(
            adata_train_list[1:], 
            batch_key='source_file',
            index_unique=None  # 避免索引重复
        )
    # 随机划分训练集测试集

    model, X_test, y_test, history, le = train_mlp_on_adata(
        adata_train, 
        obsm_key='X_emb_scGPTspatial',  # 使用的特征
        obs_label_key='sce.layer_guess',  # 细胞类型标签DLPFC:sce.layer_guess manual-anno
        test_size=0.8,
        epochs=800,
        warm_up_epochs=20,  # 使用warm up策略
        warm_up_factor=0.1, # 初始学习率为最终学习率的1%
        structure='mlp',  # 使用'mlp'或'transformer'
        knn_smooth_k=0,
    )
    
# 切片上训练测试
if __name__ == "__main__":
    main()