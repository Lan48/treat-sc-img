import scanpy as sc
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, random_split, TensorDataset
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, adjusted_rand_score
import warnings
import datetime
import random
import pandas as pd
import anndata as ad
warnings.filterwarnings("ignore", category=UserWarning)


# ========================== 1. 多文件数据读取函数（适配两种特征模式）==========================
def read_data_for_features(h5ad_path, args):
    """
    统一数据读取：支持单文件或目录输入
    - 若 h5ad_path 是文件，则读取单个文件
    - 若 h5ad_path 是目录，则读取目录下所有 .h5ad 文件并按文件拼接
    - feature_mode="emb"：从各 AnnData 的 obsm[emb_name] 取 embedding
    - feature_mode="topk_genes"：使用所有文件 HVG 的交集作为统一基因集合
    - KNN 平滑只在每个文件内部进行（如果 args.k_smooth > 0）
    """
    import numpy as np
    import scanpy as sc
    from sklearn.neighbors import NearestNeighbors
    
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"{h5ad_path} does not exist.")
    
    # 收集所有文件
    if path.is_dir():
        h5_files = sorted([p for p in path.glob("*.h5ad") if p.is_file()])
        if len(h5_files) == 0:
            raise ValueError(f"No .h5ad files found in directory: {h5ad_path}")
        print(f"Found {len(h5_files)} .h5ad files in directory")
    else:
        h5_files = [path]
        print(f"Processing single file: {path.name}")
    
    all_features = []
    all_cls_names = []
    all_feature_valid_mask = []
    adata_list = []
    
    # 针对 topk_genes 模式需预处理收集所有文件的 HVG 交集
    gene_intersection = None
    if args.feature_mode == "topk_genes":
        print("Computing HVG intersection across all files...")
        hvg_sets = []
        for f in h5_files:
            a = sc.read_h5ad(f)
            if args.label_key not in a.obs.columns:
                raise ValueError(f"Label key {args.label_key} not found in {f}. Available: {list(a.obs.columns)}")
            # 过滤空标签
            valid_mask_lbl = a.obs[args.label_key].astype(str).str.strip() != ""
            a = a[valid_mask_lbl].copy()
            sc.pp.highly_variable_genes(a, n_top_genes=args.topk, inplace=True)
            hvg = a.var[a.var["highly_variable"]].index.tolist()
            hvg_sets.append(set(hvg))
        # 求交集
        gene_intersection = set.intersection(*hvg_sets) if len(hvg_sets) > 1 else hvg_sets[0]
        if len(gene_intersection) == 0:
            raise ValueError("Intersection of HVGs across files is empty. Consider lowering topk or using embedding mode.")
        if len(gene_intersection) < args.topk:
            print(f"[Warning] Intersection HVG count {len(gene_intersection)} < requested topk {args.topk}. Using intersection.")
        gene_intersection = sorted(gene_intersection)
        print(f"Using {len(gene_intersection)} HVGs from intersection")
    
    # 遍历所有文件，逐个处理
    for f in h5_files:
        adata = sc.read_h5ad(f)
        print(f"\n[Load] {f.name}: shape={adata.shape}")
        
        # 标签检查与过滤
        if args.label_key not in adata.obs.columns:
            raise ValueError(f"Label key {args.label_key} not found in {f}. Available: {list(adata.obs.columns)}")
        cls_names = adata.obs[args.label_key].tolist()
        valid_lbl_mask = [isinstance(lab, str) and lab.strip() != "" for lab in cls_names]
        adata = adata[valid_lbl_mask].copy()
        cls_names = adata.obs[args.label_key].tolist()
        
        # 特征提取
        if args.feature_mode == "emb":
            if args.emb_name not in adata.obsm.keys():
                raise ValueError(f"Emb key {args.emb_name} not found in {f}. Available: {list(adata.obsm.keys())}")
            feats = adata.obsm[args.emb_name].copy()
            if not isinstance(feats, np.ndarray):
                feats = np.asarray(feats)
            print(f"[Feature] emb '{args.emb_name}' from {f.name}: {feats.shape}")
        
        elif args.feature_mode == "topk_genes":
            # 检查交集基因在当前文件中是否存在
            missing_genes = [g for g in gene_intersection if g not in adata.var_names]
            if len(missing_genes) > 0:
                print(f"[Warning] {f.name} missing {len(missing_genes)} genes from intersection. They will be dropped.")
                use_genes = [g for g in gene_intersection if g in adata.var_names]
            else:
                use_genes = gene_intersection
            
            # 提取子集用于特征计算
            sub = adata[:, use_genes]
            feats = sub.X
            if hasattr(feats, "toarray"):
                feats = feats.toarray()
            print(f"[Feature] topk intersection genes ({len(use_genes)}) from {f.name}: {feats.shape}")
        else:
            raise ValueError(f"Invalid feature_mode: {args.feature_mode}")
        
        # 特征有效性过滤
        feature_valid_mask = [len(feat) > 0 and not np.isnan(feat).any() for feat in feats]
        feats = feats[feature_valid_mask]
        filtered_cls = [cls_names[i] for i in range(len(cls_names)) if feature_valid_mask[i]]
        
        # KNN 平滑（在每个文件内部单独进行）
        if hasattr(args, 'k_smooth') and args.k_smooth > 0:
            if 'spatial' not in adata.obsm:
                print(f"[Warning] {f.name} has no adata.obsm['spatial'] for smoothing. Skipping smoothing for this file.")
            else:
                # 获取过滤后的空间坐标
                spatial_coords = adata.obsm['spatial'].copy()
                # 应用标签过滤
                spatial_coords = spatial_coords[valid_lbl_mask]
                # 应用特征有效性过滤
                spatial_coords = spatial_coords[feature_valid_mask]
                
                if spatial_coords.shape[0] != feats.shape[0]:
                    raise ValueError(f"Spatial coords count {spatial_coords.shape[0]} doesn't match feature count {feats.shape[0]} in file: {f.name}")
                
                k = args.k_smooth
                print(f"[Smoothing] {f.name} with k={k}")
                nbrs = NearestNeighbors(n_neighbors=min(k + 1, spatial_coords.shape[0]), metric='euclidean')
                nbrs.fit(spatial_coords)
                indices = nbrs.kneighbors(return_distance=False)
                
                smoothed = np.zeros_like(feats)
                for i in range(len(feats)):
                    neigh = indices[i][1:1 + k]  # 去掉自身
                    if len(neigh) == 0:
                        smoothed[i] = feats[i]
                    else:
                        smoothed[i] = feats[neigh].mean(axis=0)
                feats = smoothed
        
        # 累积结果
        all_features.append(feats)
        all_cls_names.extend(filtered_cls)
        all_feature_valid_mask.extend([True] * feats.shape[0])
        
        # 为合并的 AnnData 做准备
        adata_filtered = adata[feature_valid_mask].copy()
        batch_name = f.stem
        adata_filtered.obs['orig_batch'] = batch_name
        adata_list.append(adata_filtered)
    
    # 拼接所有特征
    if len(all_features) == 0:
        raise ValueError("No valid features extracted from any file.")
    
    features = np.vstack(all_features)
    print(f"\n[Concat] Total cells: {features.shape[0]}, feature dim: {features.shape[1]}")
    
    # 构建合并的 AnnData 对象
    combined_adata = None
    if len(adata_list) == 1:
        combined_adata = adata_list[0]
    else:
        if args.feature_mode == "emb":
            # 对于emb模式，构建新的AnnData
            obs_dfs = []
            for a in adata_list:
                obs_dfs.append(a.obs)
            merged_obs = pd.concat(obs_dfs, axis=0)
            combined_adata = ad.AnnData(X=features, obs=merged_obs)
        else:
            # 对于topk_genes模式，使用concat
            combined_adata = ad.concat(adata_list, join='inner', merge='unique', index_unique=None)
            # 将特征存储在obsm中，避免替换原始X矩阵
            combined_adata.obsm['X_features'] = features
    
    return {
        "features": features,
        "cls_name": all_cls_names,
        "adata": combined_adata,
        "feature_valid_mask": all_feature_valid_mask
    }


# ========================== 2. 数据集类 ==========================
class FeatureDataset:
    def __init__(self, data):
        self.features = data["features"]
        self.cls_names = data["cls_name"]
        self.adata = data.get("adata", None)
        self.feature_valid_mask = data.get("feature_valid_mask", None)
        self.class_label_map = {label: idx for idx, label in enumerate(sorted(set(self.cls_names)))}
        self.num_classes = len(self.class_label_map)
        self.labels = [self.class_label_map[name] for name in self.cls_names]


# ========================== 3. KNN评估函数 ==========================
def evaluate_knn(train_feat, train_labels, eval_feat, eval_labels, k=10):
    """KNN评估，返回准确率、ARI、预测标签和KNN模型"""
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', n_jobs=4)
    knn.fit(train_feat, train_labels)
    predicted_labels = knn.predict(eval_feat)
    accuracy = accuracy_score(eval_labels, predicted_labels) * 100
    ari = adjusted_rand_score(eval_labels, predicted_labels)
    return accuracy, ari, predicted_labels, knn


# ========================== 4. 固定随机种子 ==========================
def set_seed(seed_value=None):
    if seed_value is None:
        seed_value = random.randint(0, 10000)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# ========================== 5. 核心运行函数 ==========================
def run_knn_only(args, dataset, trial_num=0):
    """只进行KNN评估，返回准确率、ARI、索引和预测标签"""
    num_train = int(len(dataset.labels) * args.train_ratio)
    indices = np.random.permutation(len(dataset.labels))
    
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    train_feat = dataset.features[train_indices]
    train_labels = np.array(dataset.labels)[train_indices]
    test_feat = dataset.features[test_indices]
    test_labels = np.array(dataset.labels)[test_indices]
    
    print(f"Train set: {len(train_indices)} samples, Test set: {len(test_indices)} samples")
    
    knn_accuracy, ari, predicted_labels, knn_model = evaluate_knn(
        train_feat, train_labels, test_feat, test_labels, k=args.knn_k
    )
    print(f'KNN Accuracy (k={args.knn_k}): {knn_accuracy:.2f}%, ARI: {ari:.4f}')
    
    return knn_accuracy, ari, train_indices, test_indices, predicted_labels, knn_model


# ========================== 6. 保存KNN预测结果 ==========================
def save_knn_predictions_to_adata(dataset, train_indices, test_indices, predicted_labels, trial_num, args):
    """将训练集真实标签和测试集预测标签合并保存到adata.obs中"""
    if dataset.adata is None:
        print("Warning: No adata object found in dataset, skipping prediction saving.")
        return None

    combined_label_col_name = f"knn_labels_trial_{trial_num}"
    test_pred_col_name = f"knn_pred_only_trial_{trial_num}"

    N = dataset.features.shape[0]
    all_labels = np.array(['unassigned'] * N, dtype=object)
    pred_only = np.array(['unpredicted'] * N, dtype=object)

    # 训练集真实标签
    for tr_idx in train_indices:
        all_labels[tr_idx] = dataset.cls_names[tr_idx]

    # 测试集预测标签
    inv_map = {v: k for k, v in dataset.class_label_map.items()}
    for i, te_idx in enumerate(test_indices):
        pred_label_idx = predicted_labels[i]
        pred_label_name = inv_map[pred_label_idx]
        all_labels[te_idx] = pred_label_name
        pred_only[te_idx] = pred_label_name

    dataset.adata.obs[combined_label_col_name] = all_labels
    dataset.adata.obs[test_pred_col_name] = pred_only
    print(f"Saved combined labels to obs['{combined_label_col_name}', test predictions to obs['{test_pred_col_name}']")
    return dataset.adata


# ========================== 7. 保存结果到文件 ==========================
def save_results_to_files(results, args, final_adata=None, trial_num=None):
    """将结果保存到文本文件和h5ad文件"""
    import json
    import os
    from datetime import datetime
    
    result_file = f"knn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w') as f:
        f.write("Trial\tAccuracy\tARI\n")
        for i, (acc, ari) in enumerate(zip(results['all_accuracies'], results['all_aris'])):
            f.write(f"{i+1}\t{acc:.4f}\t{ari:.4f}\n")
        f.write(f"Mean\t{results['mean_accuracy']:.4f}\t{results['mean_ari']:.4f}\n")
        f.write(f"Std\t{results['std_accuracy']:.4f}\t{results['std_ari']:.4f}\n")
    print(f"Results saved to: {result_file}")
    
    if final_adata is not None:
        out_path = f"knn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
        final_adata.write(out_path)
        print(f"AnnData with predictions saved to: {out_path}")


# ========================== 8. 主函数 ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------------- 关键参数 --------------------------
    parser.add_argument("--feature_mode", type=str, default="topk_genes", choices=["emb", "topk_genes"],
                        help="Feature type: 'emb' (use pre-stored emb) or 'topk_genes' (use top-k HVGs)")
    parser.add_argument("--h5ad_path", default="project1/spatial_data/down_stream_data/human_breast_cancer", 
                        type=str, help="Path to a .h5ad file OR a directory containing multiple .h5ad files")
    parser.add_argument("--emb_name", type=str, default="X_emb512_model40_16",
                        help="Key of pre-stored emb in adata.obsm (required if feature_mode='emb')")
    parser.add_argument("--topk", type=int, default=512, 
                        help="Number of top highly variable genes (required if feature_mode='topk_genes')")
    parser.add_argument("--label_key", default="ground_truth", type=str, 
                        help="Key of cell type label in adata.obs")
    parser.add_argument("--knn_k", type=int, default=10, help="k for KNN classifier")
    parser.add_argument("--k_smooth", type=int, default=0, help="k for KNN smooth (0 to disable)")
    
    # -------------------------- 其他参数 --------------------------
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of train set (0-1)")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of repeated trials")
    parser.add_argument("--save_predictions", type=bool, default=True, 
                        help="Whether to save KNN predictions to adata object")
    
    args = parser.parse_args()
    
    # 参数校验
    if args.feature_mode == "emb" and args.emb_name is None:
        parser.error("--emb_name is required when feature_mode='emb'")
    if args.feature_mode == "topk_genes" and args.topk <= 0:
        parser.error("--topk must be a positive integer when feature_mode='topk_genes'")
    
    # 初始化结果存储
    all_knn_accuracies = []
    all_aris = []
    final_adata = None
    
    print("="*80)
    print(f"Multi-file KNN Classification Evaluation")
    print(f"Feature mode: {args.feature_mode}, Trials: {args.num_trials}")
    print(f"KNN k={args.knn_k}, Smoothing k={args.k_smooth}, Train ratio={args.train_ratio}")
    print(f"Input path: {args.h5ad_path}")
    print("="*80)
    
    # 多轮试验
    for trial in range(args.num_trials):
        print(f"\n--- Trial {trial+1}/{args.num_trials} ---")
        set_seed(trial)  # 每轮试验固定种子
        
        # 1. 读取数据（支持多文件）
        data = read_data_for_features(args.h5ad_path, args)
        
        # 2. 构建数据集
        dataset = FeatureDataset(data)
        print(f"Dataset: {len(dataset.labels)} cells, {dataset.num_classes} cell types, "
              f"feature dim: {dataset.features.shape[1]}")
        
        # 3. 运行KNN评估
        knn_acc, ari, train_indices, test_indices, pred_labels, knn_model = run_knn_only(args, dataset, trial_num=trial+1)
        all_knn_accuracies.append(knn_acc)
        all_aris.append(ari)
        
        print(f"Trial {trial+1} - KNN Accuracy: {knn_acc:.2f}%, ARI: {ari:.4f}")
        
        # 4. 保存预测结果（最后一次试验）
        if args.save_predictions and trial == args.num_trials - 1:
            final_adata = save_knn_predictions_to_adata(
                dataset, train_indices, test_indices, pred_labels, trial+1, args
            )
    
    # 5. 结果汇总
    mean_knn_acc = np.mean(all_knn_accuracies)
    std_knn_acc = np.std(all_knn_accuracies)
    mean_ari = np.mean(all_aris)
    std_ari = np.std(all_aris)
    
    results_summary = {
        'all_accuracies': all_knn_accuracies,
        'all_aris': all_aris,
        'mean_accuracy': mean_knn_acc,
        'std_accuracy': std_knn_acc,
        'mean_ari': mean_ari,
        'std_ari': std_ari
    }
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS (Multi-file)")
    print(f"KNN Accuracy: {mean_knn_acc:.2f}% ± {std_knn_acc:.2f}%")
    print(f"ARI: {mean_ari:.4f} ± {std_ari:.4f}")
    print(f"Range (Accuracy): {min(all_knn_accuracies):.2f}% - {max(all_knn_accuracies):.2f}%")
    print("="*80)
    
    # 6. 保存所有结果到文件
    if args.save_predictions:
        save_results_to_files(results_summary, args, final_adata, args.num_trials)