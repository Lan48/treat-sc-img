# 模型配置
MAE_CONFIG = {
    'c': 200,           # 基因通道数（最大基因数量）
    'h': 14,            # 空间高度
    'w': 14,            # 空间宽度
    'patch_size': 2,    # 块大小
    'mask_ratio': 0.75, # 掩码比例
    'emb_dim': 256,     # 嵌入维度
    'en_dim': 512,      # 编码器输出维度
    'de_dim': 512,      # 解码器输出维度
    'mlp1_depth': 2,    # MLP1网络深度
    'mlp2_depth': 2,    # MLP2网络深度
    'decoder_layers': 6, # 解码器层数
    'nhead': 8,         # 注意力头数
    'dim_feedforward': 1024, # 前馈网络维度
    'num_classes': 10,  # 分类类别数
    'num_genes': 2000   # 基因总数
}

# 简化配置（用于快速测试）
SIMPLE_MAE_CONFIG = {
    'c': 200,           # 基因通道数
    'h': 14,            # 空间高度
    'w': 14,            # 空间宽度
    'patch_size': 2,    # 块大小
    'mask_ratio': 0.75, # 掩码比例
    'emb_dim': 128,     # 嵌入维度（减小以节省内存）
    'en_dim': 256,      # 编码器输出维度
    'de_dim': 256,      # 解码器输出维度
    'mlp1_depth': 2,    # MLP1网络深度
    'mlp2_depth': 2,    # MLP2网络深度
    'decoder_layers': 3, # 解码器层数（减少以节省内存）
    'nhead': 4,         # 注意力头数
    'dim_feedforward': 512, # 前馈网络维度
    'num_classes': 5,   # 分类类别数
    'num_genes': 2000   # 基因总数
}

# 训练配置
TRAIN_CONFIG = {
    'data_dir': 'project1/spatial_data/samples',
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'cls_loss_weight': 0.1,  # 分类损失权重
    'grad_clip_norm': 1.0,   # 梯度裁剪
    'save_interval': 10,     # 保存间隔（epoch）
    'val_interval': 5,       # 验证间隔（epoch）
}

# 简化训练配置
SIMPLE_TRAIN_CONFIG = {
    'data_dir': 'project1/spatial_data/samples',
    'num_epochs': 10,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'cls_loss_weight': 0.1,
    'grad_clip_norm': 1.0,
    'save_interval': 5,
    'max_batches_per_epoch': 6,  # 每个epoch最大批次数（用于快速测试）
}

# 数据集配置
DATASET_CONFIG = {
    'pad_id': 0,        # 填充ID
    'num_workers': 4,   # 数据加载器工作进程数
    'pin_memory': True, # 是否使用固定内存
    'persistent_workers': True,  # 是否保持工作进程
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'type': 'AdamW',
    'betas': (0.9, 0.95),
    'eps': 1e-8,
}

# 学习率调度器配置
SCHEDULER_CONFIG = {
    'type': 'CosineAnnealingLR',
    'eta_min_factor': 0.1,  # 最小学习率因子
}

# 混合精度训练配置
AMP_CONFIG = {
    'enabled': True,
    'dtype': 'float16',
}

# 模型保存配置
SAVE_CONFIG = {
    'save_dir': './checkpoints',
    'best_model_name': 'best_mae_model.pth',
    'checkpoint_prefix': 'mae_checkpoint_epoch_',
    'save_best_only': True,
    'save_last': True,
} 