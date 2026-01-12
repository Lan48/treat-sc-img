import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import json
import time
import numpy as np
from tqdm import tqdm

from model.MAE_sp_reg import MAEModel
from dataset.dataset import SpatialTranscriptomicsDataset
from utils.utils import get_least_used_gpu
import warnings

# 设置可见的GPU设备[1,4](@ref)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # 根据实际GPU数量修改

def train_one_epoch(model, dataloader, optimizer, scheduler, device, warmup_steps, warmup_scheduler):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (binned_expression, expression, gene_ids) in enumerate(progress_bar):
        # 将数据移动到主GPU设备[4](@ref)
        binned_expression = binned_expression.to(device, non_blocking=True)
        expression = expression.to(device, non_blocking=True)
        gene_ids = gene_ids.to(device, non_blocking=True)
        
        # 清除梯度
        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播
        recon, cls_pred, mask, _ = model(binned_expression, gene_ids)  # 前向传播可通过DataParallel对象进行
        # 计算损失时，通过 .module 访问原始模型的 loss_function
        recon_loss = model.module.loss_function(expression, recon, mask)
        
        # 添加分类损失
        cls_loss = 0.0
        loss = recon_loss + cls_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 学习率调度
        if batch_idx < warmup_steps:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # 更新进度条
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.8f}")
    
    return avg_loss

def save_checkpoint(epoch, model, optimizer, scheduler, warmup_scheduler, best_loss, best_epoch, is_best, model_output_dir):
    """保存检查点文件，正确处理多GPU模型保存[2,3](@ref)"""
    # 处理多GPU模型的状态字典[2](@ref)
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model_state_dict,  # 使用处理后的状态字典
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
        'best_loss': best_loss,
        'best_epoch': best_epoch
    }
    
    # 创建输出目录
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 保存最新检查点
    checkpoint_path = os.path.join(model_output_dir, "checkpoint.pth")
    torch.save(state, checkpoint_path)
    
    # 如果是当前最佳模型，单独保存
    if is_best:
        best_model_path = os.path.join(model_output_dir, "best_model.pth")
        torch.save(state, best_model_path)
    
    # 保留历史检查点
    if (epoch + 1) % 10 == 0:
        history_path = os.path.join(model_output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(state, history_path)
    
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, warmup_scheduler, model_output_dir):
    """加载检查点文件，处理多GPU模型加载[3](@ref)"""
    checkpoint_path = os.path.join(model_output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"恢复训练从检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # 处理多GPU模型的状态字典加载[3](@ref)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', -1)
        
        return start_epoch, best_loss, best_epoch, True
    return 0, float('inf'), -1, False

def train(config, model, dataloader, optimizer, scheduler, device, warmup_steps, warmup_scheduler, model_output_dir):
    """训练主函数，支持多GPU训练[1](@ref)"""
    # 加载检查点
    start_epoch, best_loss, best_epoch, resume_status = load_checkpoint(
        model, optimizer, scheduler, warmup_scheduler, model_output_dir
    )
    
    total_epochs = config['epochs']
    print(f"{'恢复训练' if resume_status else '开始训练'} ({start_epoch}/{total_epochs} epochs)")
    
    # 主训练循环
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        epoch_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler, device, warmup_steps, warmup_scheduler
        )
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {epoch_loss:.8f} | Time: {epoch_time:.1f}s")
        
        # 检查是否是最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            print(f"发现新最佳模型! Loss: {best_loss:.8f} (Epoch {best_epoch})")
            is_best = True
        else:
            is_best = False
        
        # 记录损失
        record_loss(epoch + 1, epoch_loss, model_output_dir)
        
        # 保存检查点
        checkpoint_path = save_checkpoint(
            epoch, model, optimizer, scheduler, warmup_scheduler, 
            best_loss, best_epoch, is_best, model_output_dir
        )
        
        # 监控GPU内存使用
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"GPU内存使用: {gpu_mem:.1f} GB | 检查点保存至: {checkpoint_path}")

    # 训练完成后输出最佳结果
    if best_epoch > 0:
        print(f"训练完成! 最佳损失: {best_loss:.8f} (出现于第 {best_epoch} 个epoch)")
    else:
        print(f"训练完成! 最终损失: {best_loss:.8f}")
    
    return best_loss, best_epoch

def main(config):
    # 加载数据集
    dataset = SpatialTranscriptomicsDataset(config)

    # 创建数据加载器，调整batch_size以适应多GPU训练[4](@ref)
    effective_batch_size = config['batch_size']
    dataloader = DataLoader(
        dataset, 
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 设备设置 - 多GPU支持[1,2](@ref)
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device_ids = list(range(n_gpu))
        device = torch.device(f"cuda:{device_ids[0]}")
        
        print(f"发现 {n_gpu} 个GPU设备")
        print(f"使用GPU: {device_ids}")
        
        # 初始化模型并移至主设备
        model = MAEModel(config).to(device)
        
        # 使用DataParallel包装模型以实现多GPU训练[1](@ref)
        if n_gpu > 1:
            print(f"使用 {n_gpu} 个GPU进行训练")
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            print("使用单GPU训练")
            
    else:
        device = torch.device("cpu")
        device_ids = None
        model = MAEModel(config).to(device)
        print("使用CPU进行训练")

    # 优化器设置
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )

    # 学习率调度
    total_steps = len(dataloader) * config['epochs']
    warmup_steps = int(total_steps * 0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps
    )
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    # 开始训练
    best_loss, best_epoch = train(
        config=config,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        warmup_steps=warmup_steps,
        warmup_scheduler=warmup_scheduler,
        model_output_dir=config['model_output_dir']
    )
    
    # 保存最终训练结果
    with open(os.path.join(config['model_output_dir'], "training_result.txt"), "w") as f:
        f.write(f"最佳损失: {best_loss:.8f}\n")
        f.write(f"出现于epoch: {best_epoch}\n")
        f.write(f"总训练epochs: {config['epochs']}\n")
        f.write(f"使用的GPU数量: {len(device_ids) if device_ids else 1}\n")

def record_loss(epoch, loss, model_output_dir):
    """记录损失到文件"""
    os.makedirs(model_output_dir, exist_ok=True)
    loss_file = os.path.join(model_output_dir, "epoch_losses.txt")
    
    if not os.path.exists(loss_file):
        with open(loss_file, "w") as f:
            f.write("Epoch\tLoss\n")
    
    with open(loss_file, "a") as f:
        f.write(f"{epoch}\t{loss:.8f}\n")

if __name__ == '__main__':
    vocab_path = "project1/spatial_data/spatial_data/new_vocab.json"
    if os.path.exists(vocab_path):
        print(f"从 {vocab_path} 加载基因索引...")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = None
        print(f"警告: 未找到词汇表文件 {vocab_path}")

    # 配置参数 - 调整batch_size以适应多GPU训练[4](@ref)
    config = {
        'dataset_ratio': 0.5,
        'normalize': 10000,
        'encoder_layers': 6,
        'decoder_layers': 2,
        'is_bin': False,
        'bins': 50,
        'c': 512,
        'h': 16,
        'w': 16,
        'patch_size': 1,
        'emb_dim': 256,
        'en_dim': 256,
        'de_dim': 256,
        'mlp1_depth': 2,
        'mlp2_depth': 4,
        'mask_ratio': 0.3,
        'lr': 2e-4,
        'weight_decay': 0.05,
        'batch_size': 32,  # 增加batch_size以充分利用多GPU[4](@ref)
        'num_workers': 4,   # 根据GPU数量调整工作进程数
        'epochs': 50,
        'data_dir': "project1/spatial_data/samples16",
        'pad_id': vocab["<pad>"] if vocab else 0,
        'num_genes': max(vocab.values()) + 1 if vocab else 1000,
        'model_output_dir': 'project1/model_outputs/maskdual_valid512(16)_sp_reg',
        'model_type': 'transformer',
        'tips': '多GPU训练 - 基因id和表达值分别embedding'
    }
    # 确保输出目录存在 /mnt/data/test2/anaconda3/envs/project1/bin/python /mnt/data/test2/project1/train2.py
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config['model_output_dir'], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    main(config)
