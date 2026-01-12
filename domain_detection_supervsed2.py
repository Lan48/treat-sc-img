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
import os  # 添加os模块用于目录操作

def train_mlp_on_directory(
    data_directory, 
    test_file_name, 
    obsm_key, 
    obs_label_key, 
    hidden_dims=[512, 256, 128], 
    batch_size=32, 
    epochs=100, 
    lr=0.001, 
    device='auto', 
    knn_smooth_k=0,
    warm_up_epochs=10, 
    warm_up_factor=0.1, 
    structure='mlp',
    # 1. 新增参数：控制HVG从训练集还是测试集选择
    hvg_selection_source: str = 'test'  # 可选值：'train'（默认）、'test'
):
    """
    在包含多个H5AD文件的目录上训练MLP模型，指定一个文件作为测试集，其余作为训练集
    
    Args:
        data_directory (str): 包含H5AD文件的目录路径
        test_file_name (str): 作为测试集的文件名
        obsm_key (str): obsm中的列名或数字(表示top n高变基因)
        obs_label_key (str): obs中存储标签的列名
        hidden_dims (list): 隐藏层维度
        batch_size (int): 批大小
        epochs (int): 训练轮数
        lr (float): 学习率
        device (str): 计算设备
        knn_smooth_k (int): KNN平滑的邻居数量
        warm_up_epochs (int): warm up阶段的epoch数
        warm_up_factor (float): warm up初始学习率比例
        structure (str): 模型结构，'mlp'或'transformer'
        hvg_selection_source (str): 高变基因（HVG）的计算数据源，可选'train'（从训练集选）或'test'（从测试集选），默认'train'
    
    Returns:
        tuple: (训练好的模型, 测试集特征, 测试集标签, 训练历史, 标签编码器)
    """
    
    # 2. 新增：校验hvg_selection_source参数合法性
    if hvg_selection_source not in ['train', 'test']:
        raise ValueError(
            f"Invalid 'hvg_selection_source' value: {hvg_selection_source}\n"
            "Only 'train' (select HVG from training set) and 'test' (select HVG from test set) are supported."
        )
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 1. 加载目录中的H5AD文件并分割训练测试集
    print(f"Loading H5AD files from directory: {data_directory}")
    
    # 获取目录中所有H5AD文件
    all_files = [f for f in os.listdir(data_directory) if f.endswith('.h5ad')]
    if not all_files:
        raise ValueError(f"No H5AD files found in directory: {data_directory}")
    
    if test_file_name not in all_files:
        raise ValueError(f"Test file {test_file_name} not found in directory. Available files: {all_files}")
    
    # 分离训练和测试文件
    train_files = [f for f in all_files if f != test_file_name]
    
    if not train_files:
        raise ValueError("No training files found after excluding test file")
    
    print(f"Training files: {train_files}")
    print(f"Test file: {test_file_name}")
    
    # 2. 加载并合并训练集文件
    adata_train_list = []
    for train_file in train_files:
        file_path = os.path.join(data_directory, train_file)
        print(f"Loading training file: {file_path}")
        adata = sc.read_h5ad(file_path)
        adata.obs['source_file'] = train_file  # 标记来源文件
        adata_train_list.append(adata)
    
    # 合并训练集
    if len(adata_train_list) == 1:
        adata_train = adata_train_list[0]
    else:
        adata_train = adata_train_list[0].concatenate(
            adata_train_list[1:], 
            batch_key='source_file',
            index_unique=None  # 避免索引重复
        )
    print(f"Training data after merge: {adata_train}")
    
    # 3. 加载测试集文件
    test_file_path = os.path.join(data_directory, test_file_name)
    print(f"Loading test file: {test_file_path}")
    adata_test = sc.read_h5ad(test_file_path)
    
    print(f"Training set shape: {adata_train.shape}")
    print(f"Test set shape: {adata_test.shape}")
    
    # 4. 准备训练集特征数据（核心修改：HVG选择逻辑）
    if obsm_key.isdigit():
        # 使用top k高变基因（根据hvg_selection_source选择数据源）
        n_top_genes = int(obsm_key)
        feature_type = f'top_{n_top_genes}_highly_variable_genes（from {hvg_selection_source} set）'
        print(f"\n=== Selecting {feature_type} ===")
        
        # 3. 核心分支：根据参数从训练集/测试集计算HVG
        if hvg_selection_source == 'train':
            # 原逻辑：从训练集计算HVG
            print(f"Step 1/3: Calculating HVG on TRAIN set (top {n_top_genes} genes)")
            sc.pp.highly_variable_genes(adata_train, n_top_genes=n_top_genes, flavor='seurat')
            hvg_names = adata_train.var_names[adata_train.var.highly_variable]  # 获取HVG基因名
        else:  # hvg_selection_source == 'test'
            # 新逻辑：从测试集计算HVG
            print(f"Step 1/3: Calculating HVG on TEST set (top {n_top_genes} genes)")
            sc.pp.highly_variable_genes(adata_test, n_top_genes=n_top_genes, flavor='seurat')
            hvg_names = adata_test.var_names[adata_test.var.highly_variable]  # 获取HVG基因名
        
        # Step 2/3：训练集、测试集均对齐到HVG列表（确保特征维度一致）
        print(f"Step 2/3: Aligning train/test set to HVG list (total {len(hvg_names)} genes)")
        # 检查HVG是否都在训练集/测试集中（避免基因名不匹配）
        missing_in_train = [gene for gene in hvg_names if gene not in adata_train.var_names]
        missing_in_test = [gene for gene in hvg_names if gene not in adata_test.var_names]
        if missing_in_train:
            raise ValueError(f"HVG list contains {len(missing_in_train)} genes not found in TRAIN set: {missing_in_train[:5]}...")
        if missing_in_test:
            raise ValueError(f"HVG list contains {len(missing_in_test)} genes not found in TEST set: {missing_in_test[:5]}...")
        
        # 筛选HVG特征
        adata_train_hvg = adata_train[:, hvg_names].copy()
        adata_test_hvg = adata_test[:, hvg_names].copy()
        
        # Step 3/3：提取特征矩阵并应用KNN平滑（仅训练集）
        X_train = adata_train_hvg.X.toarray() if hasattr(adata_train_hvg.X, 'toarray') else adata_train_hvg.X
        X_test = adata_test_hvg.X.toarray() if hasattr(adata_test_hvg.X, 'toarray') else adata_test_hvg.X
        
        # 应用KNN平滑到训练集（原逻辑不变，仅训练集预处理）
        if knn_smooth_k is not None and knn_smooth_k > 0:
            print(f"Step 3/3: Applying KNN smoothing to TRAIN set (k={knn_smooth_k})")
            knn = NearestNeighbors(n_neighbors=knn_smooth_k+1)
            knn.fit(X_train)
            distances, indices = knn.kneighbors(X_train)
            X_train = np.mean(X_train[indices], axis=1)
        
    else:
        # 使用obsm中的特征（原逻辑不变）
        if obsm_key not in adata_train.obsm:
            raise ValueError(f"obsm_key '{obsm_key}' not found in training data obsm")
        if obsm_key not in adata_test.obsm:
            raise ValueError(f"obsm_key '{obsm_key}' not found in test data obsm")
        
        X_train = adata_train.obsm[obsm_key]
        X_test = adata_test.obsm[obsm_key]
        feature_type = f'obsm_{obsm_key}'
    
    print(f"\nUsing features from: {feature_type}")
    print(f"Training feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")
    
    # 5. 准备标签数据（原逻辑不变）
    if obs_label_key not in adata_train.obs:
        raise ValueError(f"obs_label_key '{obs_label_key}' not found in training data obs")
    if obs_label_key not in adata_test.obs:
        raise ValueError(f"obs_label_key '{obs_label_key}' not found in test data obs")
    
    y_train_str = adata_train.obs[obs_label_key].values.astype(str)
    y_test_str = adata_test.obs[obs_label_key].values.astype(str)
    
    # 删除标签为'nan'的样本
    train_valid_indices = y_train_str != 'nan'
    test_valid_indices = y_test_str != 'nan'
    
    X_train = X_train[train_valid_indices]
    y_train_str = y_train_str[train_valid_indices]
    X_test = X_test[test_valid_indices]
    y_test_str = y_test_str[test_valid_indices]
    
    print(f"After removing NaN labels - Training: {X_train.shape[0]} samples")
    print(f"After removing NaN labels - Test: {X_test.shape[0]} samples")
    
    # 编码标签（使用训练集的编码器确保一致性）
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)  # 只在训练集上拟合
    y_test = le.transform(y_test_str)        # 用训练集的编码器变换测试集
    
    label_classes = le.classes_
    print(f"Encoded labels: {dict(enumerate(label_classes))}")
    
    # 6. 标准化特征（使用训练集的参数，原逻辑不变）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 在训练集上拟合
    X_test = scaler.transform(X_test)        # 用训练集参数变换测试集
    
    # 7. 转换为Tensor（原逻辑不变）
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 8. 创建训练数据加载器（原逻辑不变）
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 9. 初始化模型（原逻辑不变）
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        structure=structure,
    ).to(device)
    
    # 10. 定义损失函数和优化器（原逻辑不变）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 11. 训练循环（原逻辑不变）
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Warm up策略
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
    print(f'\nFinal Test Accuracy: {final_test_acc:.2f}%')
    
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
            # MLP结构：全连接层+ReLU+Dropout
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
            # Transformer结构
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
    # 根据文件划分训练集测试集
    seed = 42
    np.random.seed(seed)
    # 现在输入目录路径
    data_directory = 'project1/spatial_data/raw_data_DLPFC/DLPFC'  # 包含H5AD文件的目录
    test_file_name = '151507.h5ad'  # 指定作为测试集的文件名
    
    model, X_test, y_test, history, le = train_mlp_on_directory(
        data_directory=data_directory,
        test_file_name=test_file_name,
        obsm_key='X_emb_scGPTspatial',           # 使用的特征
        obs_label_key='sce.layer_guess',  # 细胞类型标签
        epochs=100,
        warm_up_epochs=20,
        warm_up_factor=0.1,
        structure='mlp',
        knn_smooth_k=0,
        hvg_selection_source = 'test',
    )

if __name__ == "__main__":
    main()