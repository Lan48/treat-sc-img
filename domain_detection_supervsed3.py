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


def train_mlp_on_adata(adata, obsm_key, obs_label_key, test_size=0.2, 
                       hidden_dims=[512, 256, 128], batch_size=128, epochs=100, 
                       lr=0.001, device='auto', knn_smooth_k=0,
                       warm_up_epochs=10, warm_up_factor=0.1, structure='mlp'):
    """
    在空间转录组数据上训练MLP模型，支持KNN平滑预处理和warm up策略
    训练集和测试集根据slice_id划分，确保同一切片细胞不分散
    
    Args:
        adata (AnnData): 空间转录组数据对象
        obsm_key (str): obsm中的列名或数字(表示top n高变基因)
        obs_label_key (str): obs中存储标签的列名
        test_size (float): 测试集比例（基于切片数量）
        hidden_dims (list): 隐藏层维度
        batch_size (int): 批大小
        epochs (int): 训练轮数
        lr (float): 学习率
        device (str): 计算设备
        knn_smooth_k (int): KNN平滑的邻居数量
        warm_up_epochs (int): warm up阶段的epoch数
        warm_up_factor (float): warm up初始学习率比例
    
    Returns:
        tuple: (训练好的模型, 测试集特征, 测试集标签, 训练历史, 标签编码器)
    """
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 检查slice_id是否存在
    if 'slice_id' not in adata.obs.columns:
        raise ValueError("'slice_id' not found in adata.obs. Please ensure slice_id information is available.")
    
    # 准备特征数据
    if obsm_key.isdigit():
        # 使用top k高变基因
        n_top_genes = int(obsm_key)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        adata = adata[:, adata.var.highly_variable]
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        feature_type = f'top_{n_top_genes}_highly_variable_genes'
        
        # 应用KNN平滑
        if knn_smooth_k is not None and knn_smooth_k > 0:
            print(f"Applying KNN smoothing with k={knn_smooth_k}")
            
            knn = NearestNeighbors(n_neighbors=knn_smooth_k+1)  # 包括自身
            knn.fit(X)
            distances, indices = knn.kneighbors(X)
            X = np.mean(X[indices], axis=1)
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
    
    # 同时过滤slice_id
    slice_ids = adata.obs['slice_id'].values[valid_indices]
    
    print(f"After removing NaN labels: {X.shape[0]} samples remaining")
    print(f"Number of unique slices: {len(np.unique(slice_ids))}")
    
    # 编码字符串标签为数字
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    label_classes = le.classes_
    print(f"Encoded labels: {dict(enumerate(label_classes))}")
    
    # 修改：根据slice_id划分训练测试集[2,4](@ref)
    unique_slices = np.unique(slice_ids)
    print(f"Total unique slices: {len(unique_slices)}")
    
    # 对切片进行分层抽样，确保每个类别的切片在训练测试集中分布均匀[5](@ref)
    slice_labels = []
    for slice_id in unique_slices:
        # 找到该切片的所有细胞标签，取最常见的标签作为该切片的代表标签
        slice_mask = slice_ids == slice_id
        slice_y = y[slice_mask]
        if len(slice_y) > 0:
            # 使用该切片中最常见的标签作为该切片的标签
            most_common_label = np.bincount(slice_y).argmax()
            slice_labels.append(most_common_label)
        else:
            slice_labels.append(-1)  # 空切片的标记
    
    # 使用分层抽样划分切片[5](@ref)
    train_slices, test_slices, _, _ = train_test_split(
        unique_slices, slice_labels, 
        test_size=test_size, 
        stratify=slice_labels,  # 分层抽样确保标签分布
        random_state=42
    )
    
    print(f"Training slices: {len(train_slices)}, Test slices: {len(test_slices)}")
    
    # 根据切片划分创建细胞掩码
    train_mask = np.isin(slice_ids, train_slices)
    test_mask = np.isin(slice_ids, test_slices)
    
    # 划分数据
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Training set: {X_train.shape[0]} cells")
    print(f"Test set: {X_test.shape[0]} cells")
    
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
    
    # 返回切片划分信息用于后续分析
    slice_info = {
        'train_slices': train_slices,
        'test_slices': test_slices,
        'train_mask': train_mask,
        'test_mask': test_mask
    }
    
    return model, X_test, y_test, history, le, slice_info

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
            self.d_model = 512
            self.nhead = 8
            self.num_layers = 2
            self.dim_feedforward = 512

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

# 修改后的主函数
def main():
    np.random.seed(42)
    # 根据切片id划分训练集测试集
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = ad.read_h5ad(h5ad_path)

    # 假设adata已经包含slice_id信息
    model, X_test, y_test, history, le, slice_info = train_mlp_on_adata(
        adata, 
        obsm_key='X_emb_scGPTspatial',
        obs_label_key='sce.layer_guess',
        test_size=0.2,
        epochs=100,
        warm_up_epochs=20,
        warm_up_factor=0.1,
        structure='transformer',
    )
    
    # 可以访问切片划分信息
    print(f"Used {len(slice_info['train_slices'])} slices for training")
    print(f"Used {len(slice_info['test_slices'])} slices for testing")
    
if __name__ == "__main__":
    # 根据切片划分训练集测试集
    main()