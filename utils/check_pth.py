import torch

# 1. 加载 .pth 文件（替换为你的文件路径）
pth_path = "project1/model_outputs/maskdual_valid512(3)/checkpoint_epoch_10.pth"  # 例如你保存的 history_path
state = torch.load(pth_path, map_location='cpu')  # 用 map_location='cpu' 确保在CPU上加载（避免GPU设备问题）

# 2. 查看所有保存的参数的键（即有哪些参数）
print("保存的参数键：", state.keys())
# 输出会类似：dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'warmup_scheduler_state_dict', 'best_loss', 'best_epoch'])

# 3. 查看具体参数的值
# 查看 epoch（下一个开始的epoch）
print("下一个开始的epoch：", state['epoch'])

# 查看最佳损失和最佳epoch
print("最佳损失：", state['best_loss'])
print("最佳epoch：", state['best_epoch'])

# 查看模型参数（model_state_dict 是模型各层的权重/偏置）
# 先看模型参数的键（各层名称）
#print("\n模型参数的层名称：", state['model_state_dict'].keys())
# 示例输出可能包含：'conv1.weight', 'conv1.bias', 'fc.weight' 等（取决于你的模型结构）

# 查看某一层的具体参数（例如查看第一个卷积层的权重）
# 注意：如果模型很大，这会输出大量数值，建议只查看形状或部分值
#first_layer_key = next(iter(state['model_state_dict'].keys()))  # 获取第一个层的键
#print(f"\n层 {first_layer_key} 的参数形状：", state['model_state_dict'][first_layer_key].shape)
# 输出类似：torch.Size([64, 3, 7, 7])（表示64个输出通道，3个输入通道，7x7卷积核）

# 查看优化器参数（optimizer_state_dict 包含优化器的配置和状态）
# 例如查看优化器的参数组（包含学习率、权重衰减等）
print("\n优化器参数组：", state['optimizer_state_dict']['param_groups'])
# 输出会包含每个参数组的 'lr'（学习率）、'weight_decay' 等配置

# 查看调度器状态（例如学习率调度器的状态）
print("\n调度器状态：", state['scheduler_state_dict'])
# 可能包含调度器的当前步数、最后一次调整的epoch等信息