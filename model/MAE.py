import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
import torch.nn.functional as F

'''
class MAEModel(nn.Module):
    def __init__(self, config):
        """
        MAE编码器-解码器结构实现
        参数配置:
        - c: 基因通道数
        - h: 空间高度
        - w: 空间宽度
        - num_genes： 基因总数
        - patch_size: 块大小
        - mask_ratio: 掩码比例
        - emb_dim: 嵌入维度
        - en_dim: 编码器输出维度
        - de_dim: 解码器输出维度
        - mlp1_depth: MLP1网络深度
        - mlp2_depth: MLP2网络深度
        """
        super().__init__()
        self.c = config['c']
        self.h = config['h']
        self.w = config['w']
        self.patch_size = config['patch_size']
        self.mask_ratio = config.get('mask_ratio', 0.75)
        self.emb_dim = config['emb_dim']
        self.en_dim = config['en_dim']
        self.de_dim = config['de_dim']
        self.num_genes = config['num_genes']
        self.model_type = config['model_type']
        # 验证输入尺寸有效性
        assert self.h % self.patch_size == 0, "高度必须能被块大小整除"
        assert self.w % self.patch_size == 0, "宽度必须能被块大小整除"
        
        # 计算块数量
        self.num_patches = (self.h // self.patch_size) * (self.w // self.patch_size)
        self.patch_dim = self.c * self.patch_size**2
        
        # 共享MLP1网络
        self.mlp1 = self.build_mlp(self.patch_dim, self.emb_dim, config['mlp1_depth'])
        # 共享MLP2网络
        self.mlp2 = self.build_mlp(self.de_dim, self.patch_dim, config['mlp2_depth'])
        # 基因ID嵌入层
        self.gene_id_embedding = nn.Embedding(self.num_genes, self.emb_dim)
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, self.emb_dim))
        if self.model_type == 'mlp':
            # 编码器MLP
            self.encoder = nn.Sequential(
                nn.Linear(self.emb_dim, self.en_dim),
                nn.ReLU(),
                nn.Linear(self.en_dim, self.en_dim)
            )
        elif self.model_type == 'transformer':
            # 编码器transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=config.get('nhead', 8),
                dim_feedforward=config.get('dim_feedforward', self.emb_dim*4),
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer,config.get('encoder_layers', 6))
        # 解码器（Transformer）
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.en_dim,
                nhead=config.get('nhead', 8),
                dim_feedforward=config.get('dim_feedforward', self.en_dim*4),
                batch_first=True
            ),
            num_layers=config.get('decoder_layers', 3)
        )
        
        # CLS处理模块
        self.cls_token = nn.Parameter(torch.zeros(1, self.patch_dim))
        
        # 掩码标记（可学习）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.en_dim))
        
        # 初始化参数
        self.initialize_weights()
    
    def build_mlp(self, in_dim, out_dim, depth):
        """构建多层感知机"""
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(out_dim, out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def initialize_weights(self):
        """初始化权重参数"""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.gene_id_embedding.weight)
    
    def patchify(self, x):
        """将输入图像分块（支持批量处理）"""
        return rearrange(
            x,
            'b c (h ph) (w pw) -> b (h w) (c ph pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            h=self.h//self.patch_size,
            w=self.w//self.patch_size
        )

    def unpatchify(self, x):
        """将分块数据重组为图像（支持批量处理）"""
        return rearrange(
            x,
            'b (h w) (c ph pw) -> b c (h ph) (w pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            h=self.h//self.patch_size,
            w=self.w//self.patch_size,
            c=self.c
        )
    
    def random_masking(self, batch_patches):
        """生成随机掩码"""
        N = batch_patches.size(1)
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_patches.size(0), N, device=batch_patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        mask = torch.ones(batch_patches.size(0), N, device=batch_patches.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return mask.bool(), ids_restore

    def forward(self, expression, gene_ids):
        """
        前向传播
        输入:
        - expression: 基因表达矩阵 [batch, c, h, w]
        - gene_ids: 基因ID [batch, c]
        """
        batch_size = expression.size(0)
        
        # ============== 1. 分块处理 ==============
        patches = self.patchify(expression)  # [batch, num_patches, patch_dim]
        
        # ============== 2. 掩码处理 ==============
        mask, ids_restore = self.random_masking(patches)
        
        # 提取可见块
        visible_patches = patches[~mask].view(batch_size, -1, self.patch_dim)
        num_visible = visible_patches.size(1)
        
        # ============== 3. 编码器处理 ==============
        # 通过共享MLP1
        x = self.mlp1(visible_patches)  # [batch, num_visible, emb_dim]
        
        # 基因ID嵌入处理
        gene_emb = self.gene_id_embedding(gene_ids).mean(dim=1)  # [batch, emb_dim]
        gene_emb = gene_emb.unsqueeze(1).expand(-1, num_visible, -1)  # [batch, num_visible, emb_dim]
        
        # 加入基因嵌入和位置嵌入
        visible_positions = torch.arange(self.num_patches, device=x.device)[None, :].expand(batch_size, -1)
        visible_positions = visible_positions[~mask].view(batch_size, num_visible)
        x = x + gene_emb + self.pos_embedding[visible_positions]
        
        # 通过编码器MLP
        enc_output = self.encoder(x)  # [batch, num_visible, en_dim]
        
        # ============== 4. 解码器准备 ==============
        # 添加掩码标记
        mask_tokens = self.mask_token.expand(batch_size, self.num_patches - num_visible, self.en_dim)
        x = torch.cat([enc_output, mask_tokens], dim=1)
        
        # 恢复原始顺序
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.en_dim))
        
        # 添加位置嵌入（所有块）
        all_positions = self.pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + all_positions
        
        # ============== 5. 解码器处理 ==============
        dec_output = self.decoder(x)  # [batch, num_patches, en_dim]
        
        # 通过共享MLP2
        reconstructed = self.mlp2(dec_output)  # [batch, num_patches, patch_dim]
        reconstructed_image = self.unpatchify(reconstructed)  # [batch, c, h, w]
        
        # ============== 6. CLS处理分支 ==============
        cls_input = self.cls_token.expand(batch_size, -1)  # [batch, patch_dim]
        cls_x = self.mlp1(cls_input)  # [batch, emb_dim]
        cls_gene_emb = gene_emb[:, 0]  # 取第一个基因嵌入 [batch, emb_dim]
        cls_pos = torch.zeros(batch_size, self.emb_dim, device=x.device)
        cls_output = self.encoder(cls_x + cls_gene_emb + cls_pos)  # [batch, en_dim]
        
        return reconstructed_image, cls_output, mask,enc_output

    def loss_function(self, original, reconstructed, mask):
        """计算损失函数（仅掩码区域）"""
        # 原始图像分块
        orig_patches = self.patchify(original)
        
        # 重建图像分块
        rec_patches = self.patchify(reconstructed)
        
        # 仅计算掩码区域的损失
        loss = (orig_patches[mask] - rec_patches[mask]) ** 2
        return loss.mean()

# 基因ID和表达值统一编码
class MAEModel(nn.Module):
    def __init__(self, config):
        """
        MAE编码器-解码器结构实现 (针对基因表达数据修改版)
        参数配置:
        - is_mask: 是否使用掩码
        - c: 基因通道数
        - h: 空间高度
        - w: 空间宽度
        - num_genes： 基因总数
        - patch_size: 块大小 (固定为1)
        - mask_ratio: 基因级掩码比例 (列表, 如 [0.25, 0.5, 0.75])
        - emb_dim: 嵌入维度
        - en_dim: 编码器输出维度
        - de_dim: 解码器输出维度
        - mlp1_depth: MLP1网络深度
        - mlp2_depth: MLP2网络深度
        """
        super().__init__()
        self.is_mask = config.get('is_mask', True)
        self.c = config['c']
        self.h = config['h']
        self.w = config['w']
        self.patch_size = 1  # 固定为1
        # 基因级掩码比例列表
        self.mask_ratio_list = config.get('mask_ratio_list', [0.25, 0.5, 0.75])
        self.emb_dim = config['emb_dim']
        self.en_dim = config['en_dim']
        self.de_dim = config['de_dim']
        self.num_genes = config['num_genes']
        self.model_type = config['model_type']
        
        # 计算块数量
        self.num_patches = self.h * self.w
        self.patch_dim = self.c * self.patch_size**2
        
        # 共享MLP1网络
        self.mlp1 = self.build_mlp(self.patch_dim, self.emb_dim, config['mlp1_depth'])
        # 共享MLP2网络
        self.mlp2 = self.build_mlp(self.de_dim, self.patch_dim, config['mlp2_depth'])
        # 基因ID嵌入层
        self.gene_id_embedding = nn.Embedding(self.num_genes, self.emb_dim)
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, self.emb_dim))
        
        if self.model_type == 'mlp':
            # 编码器MLP
            self.encoder = nn.Sequential(
                nn.Linear(self.emb_dim, self.en_dim),
                nn.ReLU(),
                nn.Linear(self.en_dim, self.en_dim)
            )
        elif self.model_type == 'transformer':
            # 编码器transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=config.get('nhead', 8),
                dim_feedforward=config.get('dim_feedforward', self.emb_dim*4),
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.get('encoder_layers', 6))
        
        # 解码器（Transformer）
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.en_dim,
            nhead=config.get('nhead', 8),
            dim_feedforward=config.get('dim_feedforward', self.en_dim*4),
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.get('decoder_layers', 3))
        
        # CLS处理模块
        self.cls_token = nn.Parameter(torch.zeros(1, self.patch_dim))
        
        # 掩码标记（可学习）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.en_dim))
        
        # 初始化参数
        self.initialize_weights()
    
    def build_mlp(self, in_dim, out_dim, depth):
        """构建多层感知机"""
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(out_dim, out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def initialize_weights(self):
        """初始化权重参数"""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.gene_id_embedding.weight)
    
    def patchify(self, x):
        """将输入图像分块（支持批量处理）"""
        return rearrange(
            x,
            'b c h w -> b (h w) c',
            h=self.h,
            w=self.w,
            c=self.c
        )

    def unpatchify(self, x):
        """将分块数据重组为图像（支持批量处理）"""
        return rearrange(
            x,
            'b (h w) c -> b c h w',
            h=self.h,
            w=self.w,
            c=self.c
        )
    
    def create_dual_masks(self, expression):
        """
        创建双掩码策略:
        1. 基因级掩码: 在每个有效Spot内随机选择基因进行掩码
        2. Spot级掩码: 在空间中心(w/2, c/2)的Spot进行全掩码
        
        返回:
        - gene_mask: 基因级掩码 (与expression同形)
        - spot_mask: Spot级掩码 (与expression同形)
        - combined_mask: 合并掩码，标记所有需要重建的位置
        """
        batch_size, c, h, w = expression.shape
        device = expression.device
        
        # 初始化掩码张量
        gene_mask = torch.zeros_like(expression, dtype=torch.bool, device=device)
        spot_mask = torch.zeros_like(expression, dtype=torch.bool, device=device)
        
        # 1. 识别有效Spot (基因表达非零的Spot)
        # 假设每个Spot是c维的基因表达向量
        spot_has_expression = (expression.sum(dim=1) > 0)  # [b, h, w]
        
        # 2. 基因级掩码: 在每个有效Spot内随机选择基因进行掩码
        for b in range(batch_size):
            for i in range(h):
                for j in range(w):
                    if spot_has_expression[b, i, j]:
                        # 随机选择掩码比例
                        mask_ratio = np.random.choice(self.mask_ratio_list)
                        # 在这个Spot内随机选择基因进行掩码
                        num_genes_to_mask = int(c * mask_ratio)
                        genes_to_mask = torch.randperm(c, device=device)[:num_genes_to_mask]
                        gene_mask[b, genes_to_mask, i, j] = True
        
        # 3. Spot级掩码: 在空间中心(w/2, c/2)的Spot进行全掩码
        center_i, center_j = h // 2, w // 2
        spot_mask[:, :, center_i, center_j] = True
        
        # 4. 合并掩码: 标记所有需要重建的位置
        combined_mask = gene_mask
        
        return gene_mask, spot_mask, combined_mask
    
    def forward(self, expression, gene_ids):
        """
        前向传播
        输入:
        - expression: 基因表达矩阵 [batch, c, h, w]
        - gene_ids: 基因ID [batch, c] (假设每个位置的基因顺序相同)
        """
        batch_size, c, h, w = expression.shape
        device = expression.device
        
        # ============== 1. 创建双掩码 ==============
        gene_mask, spot_mask, combined_mask = self.create_dual_masks(expression)
        
        # 创建掩码后的输入
        masked_expression = expression.clone()
        if self.is_mask:
            masked_expression[combined_mask] = 0  # 将被掩码的基因表达值置零
        
        # ============== 2. 分块处理 ==============
        patches = self.patchify(masked_expression)  # [batch, num_patches, c]
        original_patches = self.patchify(expression)  # 原始分块，用于损失计算
        
        # 识别有效点位（基因表达值不全为0的点位）
        valid_mask = (patches.sum(dim=-1) != 0)  # [batch, num_patches]
        
        # 记录有效点位的位置
        valid_positions = []
        for b in range(batch_size):
            valid_indices = torch.where(valid_mask[b])[0]
            valid_positions.append(valid_indices)
        
        # 只保留有效点位
        valid_patches = []
        for b in range(batch_size):
            valid_patches.append(patches[b, valid_positions[b]])
        
        # 通过共享MLP1处理有效点位
        mlp1_outputs = []
        for b in range(batch_size):
            if len(valid_patches[b]) > 0:
                mlp1_output = self.mlp1(valid_patches[b])  # [num_valid_patches, emb_dim]
                mlp1_outputs.append(mlp1_output)
            else:
                mlp1_outputs.append(torch.empty(0, self.emb_dim, device=device))
        
        # ============== 3. 编码器处理 ==============
        # 基因ID嵌入处理 (假设所有Patch共享相同的基因ID)
        gene_emb = self.gene_id_embedding(gene_ids).mean(dim=1)  # [batch, emb_dim]
        
        # 加入基因嵌入和位置嵌入（只对有效点位）
        encoder_inputs = []
        for b in range(batch_size):
            if len(mlp1_outputs[b]) > 0:
                # 获取有效点位对应的位置嵌入
                pos_emb = self.pos_embedding[valid_positions[b]]  # [num_valid_patches, emb_dim]
                
                # 扩展基因嵌入到有效点位数
                gene_emb_expanded = gene_emb[b].unsqueeze(0).expand(len(valid_positions[b]), -1)  # [num_valid_patches, emb_dim]
                
                # 组合输入
                encoder_input = mlp1_outputs[b] + gene_emb_expanded + pos_emb  # [num_valid_patches, emb_dim]
                encoder_inputs.append(encoder_input)
            else:
                encoder_inputs.append(torch.empty(0, self.emb_dim, device=device))
        
        # 通过编码器
        enc_outputs = []
        for b in range(batch_size):
            if len(encoder_inputs[b]) > 0:
                if self.model_type == 'mlp':
                    enc_output = self.encoder(encoder_inputs[b])  # [num_valid_patches, en_dim]
                elif self.model_type == 'transformer':
                    enc_output = self.encoder(encoder_inputs[b].unsqueeze(0)).squeeze(0)  # [num_valid_patches, en_dim]
                enc_outputs.append(enc_output)
            else:
                enc_outputs.append(torch.empty(0, self.en_dim, device=device))
        
        # ============== 4. 解码器处理 ==============
        dec_outputs = []
        for b in range(batch_size):
            if len(enc_outputs[b]) > 0:
                dec_output = self.decoder(enc_outputs[b].unsqueeze(0)).squeeze(0)  # [num_valid_patches, en_dim]
                dec_outputs.append(dec_output)
            else:
                dec_outputs.append(torch.empty(0, self.en_dim, device=device))
        
        # 通过共享MLP2
        reconstructed_valid_patches = []
        for b in range(batch_size):
            if len(dec_outputs[b]) > 0:
                reconstructed_patch = self.mlp2(dec_outputs[b])  # [num_valid_patches, c]
                reconstructed_valid_patches.append(reconstructed_patch)
            else:
                reconstructed_valid_patches.append(torch.empty(0, c, device=device))
        
        # 将重建的有效点位填充回全零矩阵
        reconstructed_patches_full = torch.zeros(batch_size, self.num_patches, c, device=device)
        for b in range(batch_size):
            if len(reconstructed_valid_patches[b]) > 0:
                reconstructed_patches_full[b, valid_positions[b]] = reconstructed_valid_patches[b]
        
        # 重组为图像形状
        reconstructed_image = self.unpatchify(reconstructed_patches_full)  # [batch, c, h, w]
        
        # 将编码器输出也填充到全零矩阵
        enc_output_full = torch.zeros(batch_size, self.num_patches, self.en_dim, device=device)
        for b in range(batch_size):
            if len(enc_outputs[b]) > 0:
                enc_output_full[b, valid_positions[b]] = enc_outputs[b]
        
        # 重组编码器输出为空间形状 [batch, en_dim, h, w]
        enc_output_spatial = rearrange(
            enc_output_full,
            'b (h w) d -> b d h w',
            h=self.h,
            w=self.w
        )
        
        # ============== 5. CLS处理分支 ==============
        cls_input = self.cls_token.expand(batch_size, -1)  # [batch, c]
        cls_x = self.mlp1(cls_input)  # [batch, emb_dim]
        cls_gene_emb = gene_emb  # [batch, emb_dim]
        cls_pos = self.pos_embedding[0:1].expand(batch_size, -1)  # 使用第一个位置编码 [batch, emb_dim]
        cls_output = self.encoder((cls_x + cls_gene_emb + cls_pos).unsqueeze(1))
        cls_output = cls_output.squeeze(1)  # [batch, en_dim]
        
        return reconstructed_image, cls_output, combined_mask, enc_output_spatial

    def loss_function(self, original, reconstructed, mask):
        """
        计算损失函数（所有被掩码位置）
        参数:
        - original: 原始基因表达矩阵 [batch, c, h, w]
        - reconstructed: 重建的基因表达矩阵 [batch, c, h, w]
        - mask: 掩码矩阵 [batch, c, h, w]，True表示被掩码的位置
        """
        # 确保mask是布尔型
        mask = mask.bool()
        
        # 计算所有被掩码位置的MSE损失
        loss = (original[mask] - reconstructed[mask]) ** 2
        return loss.mean()   

'''

# 基因ID和表达值分别编码
class MAEModel(nn.Module):
    def __init__(self, config):
        """
        MAE编码器-解码器结构实现 (优化版)
        主要优化点：
        - 向量化操作替代循环
        - 批量处理提高GPU利用率  
        - 优化内存访问模式
        - 简化掩码生成逻辑
        """
        super().__init__()
        self.is_mask = config.get('is_mask', True)
        self.c = config['c']
        self.h = config['h']
        self.w = config['w']
        self.patch_size = 1  # 固定为1
        self.mask_ratio_list = config.get('mask_ratio_list', [0.25, 0.5, 0.75])
        self.emb_dim = config['emb_dim']
        self.en_dim = config['en_dim']
        self.de_dim = config['de_dim']
        self.num_genes = config['num_genes']
        self.model_type = config['model_type']
        
        # 计算块数量
        self.num_patches = self.h * self.w
        self.patch_dim = self.c * self.patch_size**2
        
        # 基因值编码器 (单个基因表达值到emb_dim的映射)
        self.gene_value_encoder = nn.Sequential(
            nn.Linear(1, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        
        # 基因ID嵌入层
        self.gene_id_embedding = nn.Embedding(self.num_genes, self.emb_dim)
        
        # 共享MLP2网络 (解码器输出到基因表达值的映射)
        self.mlp2 = self.build_mlp(self.de_dim, self.patch_dim, config['mlp2_depth'])
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.emb_dim))
        
        if self.model_type == 'mlp':
            # 编码器MLP
            self.encoder = nn.Sequential(
                nn.Linear(self.emb_dim, self.en_dim),
                nn.ReLU(),
                nn.Linear(self.en_dim, self.en_dim)
            )
        elif self.model_type == 'transformer':
            # 编码器transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=config.get('nhead', 8),
                dim_feedforward=config.get('dim_feedforward', self.emb_dim*4),
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.get('encoder_layers', 6))
        
        # 解码器（Transformer）
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.en_dim,
            nhead=config.get('nhead', 8),
            dim_feedforward=config.get('dim_feedforward', self.en_dim*4),
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.get('decoder_layers', 3))
        
        # CLS处理模块
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        
        # 掩码标记（可学习）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.en_dim))
        
        # 初始化参数
        self.initialize_weights()
    
    def build_mlp(self, in_dim, out_dim, depth):
        """构建多层感知机"""
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(out_dim, out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def initialize_weights(self):
        """初始化权重参数"""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.gene_id_embedding.weight)
        # 初始化基因值编码器
        for module in self.gene_value_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def patchify(self, x):
        """将输入图像分块（支持批量处理）"""
        return rearrange(
            x,
            'b c h w -> b (h w) c',
            h=self.h,
            w=self.w,
            c=self.c
        )

    def unpatchify(self, x):
        """将分块数据重组为图像（支持批量处理）"""
        return rearrange(
            x,
            'b (h w) c -> b c h w',
            h=self.h,
            w=self.w,
            c=self.c
        )
    
    def create_dual_masks(self, expression):
        """
        优化后的双掩码生成策略 - 向量化实现
        """
        batch_size, c, h, w = expression.shape
        device = expression.device
        
        # 初始化掩码张量
        gene_mask = torch.zeros_like(expression, dtype=torch.bool, device=device)
        spot_mask = torch.zeros_like(expression, dtype=torch.bool, device=device)
        
        # 1. 识别有效Spot (基因表达非零的Spot) - 向量化
        spot_has_expression = (expression.sum(dim=1) > 0)  # [b, h, w]
        
        # 2. 基因级掩码: 向量化实现
        # 为每个batch的每个spot生成随机掩码比例
        mask_ratio = torch.tensor(self.mask_ratio_list, device=device)
        random_ratios = mask_ratio[torch.randint(0, len(mask_ratio), (batch_size, h, w))]
        
        # 为每个spot生成基因掩码
        for b in range(batch_size):
            for i in range(h):
                for j in range(w):
                    if spot_has_expression[b, i, j]:
                        num_genes_to_mask = int(c * random_ratios[b, i, j])
                        if num_genes_to_mask > 0:
                            genes_to_mask = torch.randperm(c, device=device)[:num_genes_to_mask]
                            gene_mask[b, genes_to_mask, i, j] = True
        
        # 3. Spot级掩码: 在空间中心(h//2, w//2)的Spot进行全掩码
        center_i, center_j = h // 2, w // 2
        spot_mask[:, :, center_i, center_j] = True
        
        # 4. 合并掩码
        combined_mask = gene_mask
        
        return gene_mask, spot_mask, combined_mask
    
    def encode_genes(self, gene_ids, expressions):
        """
        优化后的基因编码方法：单个基因ID和表达值分别嵌入后相加
        """
        batch_size, num_patches, c = expressions.shape
        device = expressions.device
        
        # 基因ID嵌入: 对每个基因ID嵌入后求和 [6](@ref)
        # gene_ids形状: [batch_size, c] -> 嵌入后: [batch_size, c, emb_dim] -> 求和: [batch_size, emb_dim]
        gene_id_emb = self.gene_id_embedding(gene_ids).sum(dim=1)  # 改为求和而不是平均
        
        # 基因表达值编码: 每个基因表达值单独编码后求和
        # 将表达值reshape为 [batch_size * num_patches * c, 1] 进行批量处理
        flat_expressions = expressions.reshape(-1, 1)  # [batch_size * num_patches * c, 1]
        
        # 应用基因值编码器 [1](@ref)
        gene_value_emb = self.gene_value_encoder(flat_expressions)  # [batch_size * num_patches * c, emb_dim]
        
        # reshape回原始维度并求和
        gene_value_emb = gene_value_emb.reshape(batch_size, num_patches, c, self.emb_dim)
        gene_value_emb_sum = gene_value_emb.sum(dim=2)  # [batch_size, num_patches, emb_dim]
        
        # 扩展基因ID嵌入到每个patch
        gene_id_emb_expanded = gene_id_emb.unsqueeze(1).expand(-1, num_patches, -1)  # [batch_size, num_patches, emb_dim]
        
        # 合并基因ID嵌入和基因值嵌入
        total_gene_emb = gene_id_emb_expanded + gene_value_emb_sum
        
        return total_gene_emb
    
    def forward(self, expression, gene_ids):
        """
        优化后的前向传播 - 批量处理实现
        """
        batch_size, c, h, w = expression.shape
        device = expression.device
        
        # ============== 1. 创建双掩码 ==============
        gene_mask, spot_mask, combined_mask = self.create_dual_masks(expression)
        
        # 创建掩码后的输入
        masked_expression = expression.clone()
        if self.is_mask:
            masked_expression[combined_mask] = 0 
        
        # ============== 2. 分块处理 ==============
        patches = self.patchify(masked_expression)  # [batch_size, num_patches, c]
        original_patches = self.patchify(expression)  # 原始分块，用于损失计算
        
        # 识别有效点位（批量处理）
        valid_mask = (patches.sum(dim=-1) != 0)  # [batch_size, num_patches]
        
        # ============== 3. 基因编码 ==============
        # 使用优化后的基因编码方法
        total_gene_emb = self.encode_genes(gene_ids, patches)  # [batch_size, num_patches, emb_dim]
        
        # 添加位置嵌入 [5](@ref)
        pos_emb = self.pos_embedding.expand(batch_size, -1, -1)  # [batch_size, num_patches, emb_dim]
        encoder_input = total_gene_emb + pos_emb  # [batch_size, num_patches, emb_dim]
        
        # 只处理有效点位
        encoder_input = encoder_input * valid_mask.unsqueeze(-1).float()
        
        # ============== 4. 编码器处理 ==============
        if self.model_type == 'mlp':
            enc_output = self.encoder(encoder_input)  # [batch_size, num_patches, en_dim]
        elif self.model_type == 'transformer':
            # 转换维度以适应transformer
            enc_output = self.encoder(encoder_input)  # [batch_size, num_patches, en_dim]
        
        # ============== 5. 解码器处理 ==============
        dec_output = self.decoder(enc_output)  # [batch_size, num_patches, en_dim]
        
        # ============== 6. 重建基因表达值 ==============
        reconstructed_patches = self.mlp2(dec_output)  # [batch_size, num_patches, c]
        
        # 重组为图像形状
        reconstructed_image = self.unpatchify(reconstructed_patches)  # [batch_size, c, h, w]
        
        # 重组编码器输出为空间形状
        enc_output_spatial = rearrange(
            enc_output,
            'b (h w) d -> b d h w',
            h=self.h,
            w=self.w
        )

        # 重组编码器输入为空间形状
        enc_input_spatial  = rearrange(
            enc_output,
            'b (h w) d -> b d h w',
            h=self.h,
            w=self.w
        )
        
        # ============== 7. CLS处理分支 ==============
        cls_input = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_dim]
        
        # CLS分支的基因编码（使用基因ID和零表达值）
        zero_expression = torch.zeros(batch_size, 1, c, device=device)
        cls_gene_emb = self.encode_genes(gene_ids, zero_expression).squeeze(1)  # [batch_size, emb_dim]
        
        cls_total = cls_input.squeeze(1) + cls_gene_emb  # [batch_size, emb_dim]
        cls_output = self.encoder(cls_total.unsqueeze(1)).squeeze(1)  # [batch_size, en_dim]
        
        return reconstructed_image, cls_output, combined_mask, enc_output_spatial#, enc_input_spatial

    def loss_function(self, original, reconstructed, mask):
        """
        计算损失函数（所有被掩码位置）
        """
        mask = mask.bool()
        loss = F.mse_loss(original[mask], reconstructed[mask])
        return loss

if __name__ == '__main__':
    config = {
        'c': 32,
        'h': 14,
        'w': 14,
        'patch_size': 1,
        'emb_dim': 512,
        'en_dim': 512,
        'de_dim': 512,
        'mlp1_depth': 2,
        'mlp2_depth': 2,
        'decoder_layers': 6,
        'num_classes': 10,
        'num_genes':2000,
        'mask_ratio': 0.5,
        'model_type': 'transformer'
    }
    def get_least_used_gpu():
        import pynvml
        """获取当前内存使用最少的GPU设备"""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            # 获取每个GPU的剩余内存
            gpu_mem = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem.append(info.free)
            
            # 选择剩余内存最多的GPU
            best_gpu = torch.device(f"cuda:{gpu_mem.index(max(gpu_mem))}")
            print(f"Selected GPU {best_gpu.index} with {max(gpu_mem)/1024**2:.2f} MB free memory")
            return best_gpu
        except Exception as e:
            print(f"Error selecting GPU: {e}. Using default GPU")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = get_least_used_gpu()
    model = MAEModel(config).to(device)
    # 创建带有无效点位的数据
    batch_size = 8
    c, h, w = 32, 14, 14
    
    # 创建随机表达矩阵
    expression = torch.randn(batch_size, c, h, w).to(device)
    
    # 随机选择一些点位设置为全0（创建无效点位）
    for b in range(batch_size):
        # 随机选择一些点位设置为全0
        num_zero_spots = torch.randint(0, h*w//2, (1,)).item()  # 随机选择0到一半的点位
        zero_spots = torch.randperm(h*w)[:num_zero_spots]  # 随机选择点位
        
        for spot_idx in zero_spots:
            i = spot_idx // w
            j = spot_idx % w
            expression[b, :, i, j] = 0  # 将该点位的所有基因表达值设为0
    
    gene_ids = torch.randint(0, 1000, (batch_size, 32)).to(device) # 基因ID

    # 打印有效点位数量信息
    patches = rearrange(expression, 'b c h w -> b (h w) c')
    valid_mask = (patches.sum(dim=-1) != 0)
    valid_counts = valid_mask.sum(dim=1)
    print(f"每个样本的有效点位数量: {valid_counts.tolist()}")
    print(f"总有效点位比例: {valid_counts.sum().item()/(batch_size*h*w)*100:.2f}%")

    # 前向传播
    recon, cls_output, mask, enc_output = model(expression, gene_ids)
    
    # 检查输出形状
    print(f"输入形状: {expression.shape}")
    print(f"重建形状: {recon.shape}")
    print(f"CLS输出形状: {cls_output.shape}")
    print(f"掩码形状: {mask.shape}")
    print(f"编码器输出形状: {enc_output.shape}")
    
    # 损失计算
    loss = model.loss_function(expression, recon, mask)
    print(f"损失值: {loss.item():.4f}")
    
    # 验证重建效果
    # 只计算有效点位的重建误差
    valid_patches = rearrange(expression, 'b c h w -> b (h w) c')
    valid_mask = (valid_patches.sum(dim=-1) != 0)
    
    recon_patches = rearrange(recon, 'b c h w -> b (h w) c')
    
    # 计算有效点位的MSE
    mse_valid = ((valid_patches[valid_mask] - recon_patches[valid_mask]) ** 2).mean()
    print(f"有效点位重建MSE: {mse_valid.item():.4f}")
    
    # 计算掩码点位的重建MSE
    mask_patches = rearrange(mask, 'b c h w -> b (h w) c')
    mask_spots = mask_patches.any(dim=-1)  # 只要有一个基因被掩码，整个点位就算被掩码
    
    mse_masked = ((valid_patches[mask_spots] - recon_patches[mask_spots]) ** 2).mean()
    print(f"掩码点位重建MSE: {mse_masked.item():.4f}")


