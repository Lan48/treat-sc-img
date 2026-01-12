import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import anndata
import os
import torch
import json
import scipy.sparse
from model.MAE import MAEModel
from utils.utils import get_least_used_gpu
from typing import List, Dict, Union
import anndata as ad
import warnings
from anndata._warnings import ImplicitModificationWarning
from scipy.sparse import issparse,csr_matrix
import re
import os
import math
import json
import numpy as np
import anndata
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

class SpatialPatchDataset(Dataset):
    """
    针对一个 AnnData (一个切片)，对其每个 spot 生成:
       expression_patch: [C,H,W]
       gene_ids: [C]
    并记录中心 spot 的索引用于回写 embedding。
    """
    def __init__(
        self,
        adata: anndata.AnnData,
        vocab: Dict[str, int],
        config: Dict,
        gene_selection_method: str = 'hvg'
    ):
        self.adata = adata
        self.vocab = vocab
        self.config = config
        self.gene_selection_method = gene_selection_method
        self.C = config['c']
        self.H = config['h']
        self.W = config['w']
        self.emb_name = config['emb_name']
        self.pad_id = config.get('pad_id', 0)

        # 提取表达矩阵 (稀疏转稠密)
        X = adata.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        self.expr = X  # shape [N_spots, N_genes]
        self.genes = list(adata.var_names)
        self.gene_index_map = {
            g: i for i, g in enumerate(self.genes) if g in self.vocab
        }
        self.coord = adata.obsm['spatial']  # [N_spots,2]
        self.N = self.coord.shape[0]

        # 预估网格尺度
        self.dx, self.dy = self._estimate_spacing()

        # 中心点索引列表（每个 spot 都做）
        self.centers = list(range(self.N))

        # 准备 gene variance 全局（可辅助 HVG 或过滤）
        self.global_var = np.var(self.expr, axis=0)

    def _estimate_spacing(self) -> Tuple[float, float]:
        xs = np.sort(np.unique(self.coord[:,0]))
        ys = np.sort(np.unique(self.coord[:,1]))
        def median_diff(arr):
            if len(arr) < 2:
                return 1.0
            diffs = np.diff(arr)
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                return 1.0
            return float(np.median(diffs))
        dx = median_diff(xs)
        dy = median_diff(ys)
        return dx if dx > 0 else 1.0, dy if dy > 0 else 1.0

    def __len__(self):
        return len(self.centers)

    def _select_genes(self, spot_indices: List[int]) -> List[int]:
        """
        基于局部 patch 的表达矩阵 (spots_in_patch × genes) 计算基因方差。
        若基因数 < C -> 抛错
        若 >= C -> 选方差最大的前 C 个（或改成按方差加权随机采样）。
        返回的是在 vocab 空间中的 gene id 列表（长度 C）。
        """
        # 收集所有可用基因（在 vocab 中）
        valid_gene_indices = [self.gene_index_map[g] for g in self.genes if g in self.gene_index_map]
        if len(valid_gene_indices) < self.C:
            raise ValueError(f"可用基因数量({len(valid_gene_indices)}) < 需求 C({self.C}).")

        sub_expr = self.expr[np.array(spot_indices)[:,None], valid_gene_indices]  # [S, G_valid]
        local_var = np.var(sub_expr, axis=0)  # [G_valid]

        # Top-C
        top_idx = np.argsort(-local_var)[:self.C]
        chosen_gene_global_indices = [valid_gene_indices[i] for i in top_idx]

        # 转换到 vocab id
        chosen_gene_vocab_ids = [self.vocab[self.genes[g_idx]] for g_idx in chosen_gene_global_indices]
        return chosen_gene_global_indices, chosen_gene_vocab_ids

    def _build_patch(self, center_idx: int):
        cx, cy = self.coord[center_idx]
        half_w = (self.W - 1) / 2.0
        half_h = (self.H - 1) / 2.0

        # 计算所有点相对中心的网格坐标（连续）
        rel_x = (self.coord[:,0] - cx) / self.dx
        rel_y = (self.coord[:,1] - cy) / self.dy

        # 四舍五入到最近网格
        grid_x = np.rint(rel_x).astype(int)
        grid_y = np.rint(rel_y).astype(int)

        # 筛选在范围内 [-half_w, half_w], [-half_h, half_h]
        mask = (
            (grid_x >= -half_w) & (grid_x <= half_w) &
            (grid_y >= -half_h) & (grid_y <= half_h)
        )
        candidate_indices = np.where(mask)[0]
        if candidate_indices.size == 0:
            # 全部为空 -> 返回全 0
            raise ValueError("中心点没有任何邻域点（可能 dx/dy 估计不合适或坐标异常）")

        # 基因选择（使用所有落入范围的 spot）
        chosen_gene_global_indices, chosen_gene_vocab_ids = self._select_genes(candidate_indices)

        # 初始化 patch: [C,H,W] 全 0
        patch = np.zeros((self.C, self.H, self.W), dtype=np.float32)

        # 映射坐标到 [0,W-1]/[0,H-1]
        for spot_i in candidate_indices:
            gx = grid_x[spot_i]
            gy = grid_y[spot_i]
            # 转换到索引
            ix = int(gx + half_w)
            iy = int(gy + half_h)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                # 取该 spot 在选中基因集合中的表达
                expr_vec = self.expr[spot_i, chosen_gene_global_indices]  # [C]
                patch[:, iy, ix] = expr_vec

        gene_ids = np.array(chosen_gene_vocab_ids, dtype=np.int64)

        return patch, gene_ids

    def __getitem__(self, idx):
        center = self.centers[idx]
        patch, gene_ids = self._build_patch(center)
        return {
            "expression": torch.from_numpy(patch),     # [C,H,W]
            "gene_ids": torch.from_numpy(gene_ids),    # [C]
            "center_index": center
        }


def collate_fn(batch):
    expressions = torch.stack([b["expression"] for b in batch], dim=0)  # [B,C,H,W]
    gene_ids = torch.stack([b["gene_ids"] for b in batch], dim=0)       # [B,C]
    centers = [b["center_index"] for b in batch]
    return expressions, gene_ids, centers


@torch.no_grad()
def embed_slice(
    adata: anndata.AnnData,
    model: MAEModel,
    vocab: Dict[str,int],
    config: Dict,
    gene_selection_method: str,
    device: torch.device,
    batch_size: int = 32
):
    dataset = SpatialPatchDataset(
        adata=adata,
        vocab=vocab,
        config=config,
        gene_selection_method=gene_selection_method
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn
    )

    model.eval()
    emb_list = []
    center_indices = []
    H = config['h']
    W = config['w']
    center_y = H // 2
    center_x = W // 2

    for expressions, gene_ids, centers in tqdm(loader, desc="Embedding spots"):
        expressions = expressions.to(device)  # [B,C,H,W]
        gene_ids = gene_ids.to(device)        # [B,C]
        _, _, _, enc_output_spatial = model(expressions, gene_ids)  # enc_output_spatial:[B,E,H,W]
        center_emb = enc_output_spatial[:, :, center_y, center_x]   # [B,E]
        emb_list.append(center_emb.cpu().numpy())
        center_indices.extend(centers)

    embeddings = np.concatenate(emb_list, axis=0)  # [N_spots, E] (顺序与 center_indices 一致)
    # 需要按原 spot 顺序放回
    E = embeddings.shape[1]
    out = np.zeros((adata.n_obs, E), dtype=np.float32)
    for i, spot_idx in enumerate(center_indices):
        out[spot_idx] = embeddings[i]

    adata.obsm[config['emb_name']] = out
    return adata


def process_directory(
    directory_path: str,
    model: MAEModel,
    vocab_path: str,
    config: Dict,
    gene_selection_method: str,
    device: torch.device,
    output_suffix="_with_emb",
    batch_size: int = 32
):
    # 读取 vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    files = [f for f in os.listdir(directory_path) if f.endswith('.h5ad')]
    if not files:
        print("目录中未找到 .h5ad 文件")
        return

    os.makedirs(config['model_output_dir'], exist_ok=True)

    for fname in files:
        fpath = os.path.join(directory_path, fname)
        print(f"处理: {fpath}")
        adata = anndata.read_h5ad(fpath)
        adata = embed_slice(
            adata=adata,
            model=model,
            vocab=vocab,
            config=config,
            gene_selection_method=gene_selection_method,
            device=device,
            batch_size=batch_size
        )
        out_name = fname#.replace('.h5ad', f'{output_suffix}.h5ad')
        out_path = os.path.join(directory_path, out_name)
        adata.write(out_path)
        print(f"已写出: {out_path}")


    
if __name__ == '__main__':
    seed = 42  

    # 1. 设置Python内置随机数生成器
    random.seed(seed)

    # 2. 设置NumPy的随机数生成器
    np.random.seed(seed)

    # 3. 设置PyTorch的随机数生成器
    torch.manual_seed(seed)
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
        'normalize':10000,
        "encoder_layers": 6,
        "decoder_layers": 2,
        'is_bin': False,   # 是否进行bin处理
        'bins': 50,
        'is_mask':False,    # 模型内部是否掩码
        'c': 512,           # 最大基因长度
        'depth':512,
        'h': 16,            # 高度
        'w': 16,            # 宽度
        'patch_size': 1,     # 块大小
        'emb_dim': 256,      # 嵌入维度
        'en_dim': 256,       # 编码器维度
        'de_dim': 256,       # 解码器维度
        'mlp1_depth': 2,     # MLP1深度
        'mlp2_depth': 4,     # MLP2深度
        'mask_ratio': 0.0,   # 掩码比例
        'mask_ratio_list': [0.0], # 掩码比例列表
        'lr': 2e-5,           # 学习率
        'weight_decay': 0.05, # 权重衰减
        'batch_size': 32,     # 批次大小
        'num_workers': 4,     # 数据加载工作进程数
        'epochs': 100,        # 训练轮数
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': vocab["<pad>"] if vocab else 0,  # 填充ID
        'num_genes': max(vocab.values()) + 1 if vocab else 1000, # 基因数量 (包括pad)
        'model_output_dir': 'project1/model_outputs', # 模型输出目录
        'model_type': 'transformer', # 模型类型
        'model_path': 'project1/model_outputs/maskdual_valid512(16)/checkpoint_epoch_40.pth',
        'emb_name': 'X_emb512_model40_16_canter',
        'mode': 'spatial', # 模式选择 spatial or single or padding
        'pad_size': 1.0
    }
    gene_selection_method = 'hvg' #"hvg": 使用高变基因"weighted_random": 加权随机选择）"uniform_random": 直接均匀随机选择
    directory_path = 'project1/spatial_data/down_stream_data/raw_data_DLPFC/DLPFC'
    if torch.cuda.is_available():
        device = get_least_used_gpu()
        print(f"使用GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")

    # 初始化模型
    model = MAEModel(config).to(device)
    model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])

    
    process_directory(
        directory_path=directory_path,
        model=model,
        vocab_path="project1/spatial_data/spatial_data/new_vocab.json",
        config=config,
        gene_selection_method="hvg",
        device=device,
        output_suffix="_with_emb",
        batch_size=config['batch_size']
    )