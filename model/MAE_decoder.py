import torch
import torch.nn as nn
import torch.nn.functional as F

class MAEDecoder(nn.Module):
    def __init__(self, dim=512, decoder_dim=256, patches=196, num_genes=200):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.patches = patches
        self.num_genes = num_genes
        
        # 掩码标记
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 位置嵌入 - 匹配序列长度
        self.pos_embed = nn.Parameter(torch.randn(1, patches * num_genes, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 解码器结构
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True) for _ in range(4)
        ])
        
        self.dim_adjust = nn.Linear(dim, decoder_dim)
        
        # 重建层 - 每个位置重建一个基因表达值
        self.gene_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_dim * 2, 1)  # 输出单个值
        )
        
        # 设备将由父模块设置
        self.device = None

    def forward(self, x, gene_ids_restore):
        # 确保输入在正确设备上
        if self.device is None:
            self.device = next(self.parameters()).device
        x = x.to(self.device)
        gene_ids_restore = gene_ids_restore.to(self.device)
        
        B, num_tokens, D = x.shape
        
        # 分离CLS token
        cls_token = x[:, :1, :]
        seq_tokens = x[:, 1:, :]  # [B, num_visible_patches*C, D]
        
        # 调整维度
        seq_tokens = self.dim_adjust(seq_tokens)
        cls_token = self.dim_adjust(cls_token)
        
        # 添加mask tokens
        num_visible = seq_tokens.shape[1]
        num_total = gene_ids_restore.shape[1]
        num_masked = num_total - num_visible
        mask_tokens = self.mask_token.expand(B, num_masked, self.decoder_dim)
        full_emb = torch.cat([seq_tokens, mask_tokens], dim=1)
        
        # 恢复原始顺序
        full_emb = torch.gather(
            full_emb, 1, 
            gene_ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )
        
        # 添加位置嵌入
        full_emb += self.pos_embed
        
        # 重新添加CLS token
        if cls_token is not None:
            full_emb = torch.cat([cls_token, full_emb], dim=1)
        
        # Transformer处理
        decoded = self.blocks(full_emb)
        
        # 跳过CLS token
        decoded = decoded[:, 1:, :]  # [B, L*C, decoder_dim]
        
        # 应用重建层 - 每个位置重建一个基因表达值
        recon_values = self.gene_head(decoded)  # [B, L*C, 1]
        
        # 重塑为 [B, L, C]
        recon_patches = recon_values.view(B, self.patches, self.num_genes)
        
        return recon_patches

