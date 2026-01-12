import scanpy as sc
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, adjusted_rand_score
import warnings
import datetime
import random
import pandas as pd
import anndata as ad
warnings.filterwarnings("ignore", category=UserWarning)


# ========================== 1. 多文件读取与特征构建 ==========================
def read_data_for_features(h5ad_path, args):
    """
    统一数据读取：
    - 若 h5ad_path 是文件，则行为与原版本基本一致（在多 obsm_key / 多 trial 的循环中重复调用）
    - 若 h5ad_path 是目录，则读取目录下所有 .h5ad 文件并按文件为 batch 拼接
    - feature_mode = 'emb': 从各 AnnData 的 obsm[emb_name] 取 embedding（要求各文件维度一致）
    - feature_mode = 'topk_genes':
        * 每个文件内分别执行 HVG 选择（n_top_genes=args.topk）
        * 取所有文件 HVG 的交集作为统一基因集合（若交集小于 args.topk 给出警告）
        * 在该统一基因集合上取表达值并拼接（稀疏转稠密）
    - KNN 平滑只在每个文件内部进行（如果 args.k_smooth > 0）
    - 返回：
        {
          "features": (N, D) np.array
          "cls_name":  list[str] 长度 N
          "adata": 合并后的 AnnData（仅在需要保存预测时使用）
          "feature_valid_mask": list[bool] 长度 N （所有文件拼接后）
        }
    """
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"{h5ad_path} does not exist.")

    # 收集所有文件
    if path.is_dir():
        h5_files = sorted([p for p in path.glob("*.h5ad") if p.is_file()])
        if len(h5_files) == 0:
            raise ValueError(f"No .h5ad files found in directory: {h5ad_path}")
    else:
        h5_files = [path]

    all_features = []
    all_cls_names = []
    all_feature_valid_mask = []
    adata_list = []

    # 针对 topk 模式需预处理收集 HVG
    gene_intersection = None
    if args.feature_mode == "topk_genes":
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

    # 遍历文件，逐个提取特征并（可选）平滑
    global_offset = 0
    for f in h5_files:
        adata = sc.read_h5ad(f)
        print(f"[Load] {f.name}: shape={adata.shape}")

        # 标签检查与过滤
        if args.label_key not in adata.obs.columns:
            raise ValueError(f"Label key {args.label_key} not found in {f}. Available: {list(adata.obs.columns)}")
        cls_names = adata.obs[args.label_key].tolist()
        valid_lbl_mask = [isinstance(lab, str) and lab.strip() != "" for lab in cls_names]
        adata = adata[valid_lbl_mask].copy()
        cls_names = adata.obs[args.label_key].tolist()

        # ---- 特征提取 ----
        if args.feature_mode == "emb":
            if args.emb_name not in adata.obsm.keys():
                raise ValueError(f"Emb key {args.emb_name} not found in {f}. Available: {list(adata.obsm.keys())}")
            feats = adata.obsm[args.emb_name].copy()
            if not isinstance(feats, np.ndarray):
                feats = np.asarray(feats)
            print(f"[Feature] emb '{args.emb_name}' from {f.name}: {feats.shape}")

        elif args.feature_mode == "topk_genes":
            missing_genes = [g for g in gene_intersection if g not in adata.var_names]
            if len(missing_genes) > 0:
                # 对于缺失基因（极少情况），跳过该文件或选择忽略这些基因
                print(f"[Warning] {f.name} missing {len(missing_genes)} genes from intersection. They will be dropped.")
                use_genes = [g for g in gene_intersection if g in adata.var_names]
            else:
                use_genes = gene_intersection
            sub = adata[:, use_genes]
            feats = sub.X
            if hasattr(feats, "toarray"):
                feats = feats.toarray()
            print(f"[Feature] topk intersection genes ({len(use_genes)}) from {f.name}: {feats.shape}")
        else:
            raise ValueError(f"Invalid feature_mode: {args.feature_mode}")

        # ---- 特征有效性过滤 ----
        feature_valid_mask = [len(feat) > 0 and not np.isnan(feat).any() for feat in feats]
        feats = feats[feature_valid_mask]
        filtered_cls = [cls_names[i] for i in range(len(cls_names)) if feature_valid_mask[i]]

        # ---- KNN 平滑（文件内）----
        if hasattr(args, 'k_smooth') and args.k_smooth > 0:
            if 'spatial' not in adata.obsm:
                raise ValueError(f"{f.name} has no adata.obsm['spatial'] for smoothing.")
            
            # 修复：先获取过滤标签后的空间坐标，再应用特征有效性过滤
            spatial_coords = adata.obsm['spatial'].copy()
            if spatial_coords.shape[0] != len(feature_valid_mask):
                raise ValueError(f"Spatial coords count {spatial_coords.shape[0]} doesn't match feature count {len(feature_valid_mask)} in file: {f.name}")
            
            spatial_coords = spatial_coords[feature_valid_mask]
            if spatial_coords.shape[0] != feats.shape[0]:
                raise ValueError("Spatial coords and feature count mismatch after filtering in file: " + f.name)
            
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

        # ---- 累积 ----
        all_features.append(feats)
        all_cls_names.extend(filtered_cls)
        # 需要把文件内的 feature_valid_mask 也扩展（只保留 True 的行数对应）
        # 由于 feats 已经过滤，这里将其折算成一个全 True 段
        all_feature_valid_mask.extend([True] * feats.shape[0])

        # 为合并的 AnnData 做准备：只保留过滤且 feature_valid_mask=True 的细胞
        # 重新构建一个子集 AnnData
        # valid_lbl_mask -> adata 已经用它筛过
        # feature_valid_mask 再筛
        adata_filtered = adata[feature_valid_mask].copy()
        # 增加来源 batch 信息
        batch_name = f.stem
        adata_filtered.obs['orig_batch'] = batch_name
        adata_list.append(adata_filtered)

        global_offset += feats.shape[0]

    # ---- 拼接所有特征 ----
    features = np.vstack(all_features)
    print(f"[Concat] Total cells: {features.shape[0]}, feature dim: {features.shape[1]}")

    # ---- （可选）检查 embedding 维度一致性 ----
    if args.feature_mode == "emb":
        # 这里只需确保没有不一致（前面若有不一致会在 vstack 抛错）
        pass

    # 构建合并 AnnData（用于保存预测标签）
    combined_adata = None
    if len(adata_list) == 1:
        combined_adata = adata_list[0]
    else:
        # anndata.concat：保留所有 obs/var；对 var（基因）不统一时可产生并集
        # 对 embedding 模式：var 可能不同，此时可创建一个虚拟基因矩阵
        # 这里做法：
        #   - 如果 feature_mode='emb'，不依赖 X，直接构建一个 dummy AnnData
        #   - 如果 feature_mode='topk_genes'，各文件已经在 gene_intersection 上统一，concat 安全
        if args.feature_mode == "emb":
            # 构建一个以特征为 X 的 AnnData（注意这不是基因表达矩阵，但可用于存 obs）
            obs_dfs = []
            for a in adata_list:
                obs_dfs.append(a.obs)
            merged_obs = pd.concat(obs_dfs, axis=0)
            combined_adata = ad.AnnData(X=features, obs=merged_obs)
        else:
            combined_adata = ad.concat(adata_list, join='inner', merge='unique', index_unique=None)
            # 替换 X 为我们用的 features (因为 smoothing 后 features 可能与 X 一致也可能已改)
            combined_adata.X = features

    return {
        "features": features,
        "cls_name": all_cls_names,
        "adata": combined_adata,
        "feature_valid_mask": all_feature_valid_mask
    }


# ========================== 2. 数据集封装 ==========================
class FeatureDataset:
    def __init__(self, data):
        self.features = data["features"]
        self.cls_names = data["cls_name"]
        self.adata = data.get("adata", None)
        self.feature_valid_mask = data.get("feature_valid_mask", None)
        self.class_label_map = {label: idx for idx, label in enumerate(sorted(set(self.cls_names)))}
        self.num_classes = len(self.class_label_map)
        self.labels = [self.class_label_map[name] for name in self.cls_names]


# ========================== 3. 评估函数 ==========================
def evaluate_knn(train_feat, train_labels, eval_feat, eval_labels, k=10):
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


# ========================== 5. 仅 KNN 评估 ==========================
def run_knn_only(args, dataset, trial_num=0):
    num_train = int(len(dataset.labels) * args.train_ratio)
    indices = np.random.permutation(len(dataset.labels))
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_feat = dataset.features[train_indices]
    train_labels = np.array(dataset.labels)[train_indices]
    test_feat = dataset.features[test_indices]
    test_labels = np.array(dataset.labels)[test_indices]

    print(f"Train set: {len(train_indices)} - Test set: {len(test_indices)}")
    knn_accuracy, ari, predicted_labels, knn_model = evaluate_knn(
        train_feat, train_labels, test_feat, test_labels, k=args.knn_k
    )
    print(f'KNN Accuracy (k={args.knn_k}): {knn_accuracy:.2f}%, ARI: {ari:.4f}')
    return knn_accuracy, ari, train_indices, test_indices, predicted_labels, knn_model


# ========================== 6. 保存预测到 AnnData ==========================
def save_knn_predictions_to_adata(dataset, train_indices, test_indices, predicted_labels, trial_num, args):
    if dataset.adata is None:
        print("Warning: No adata object present; skip saving predictions.")
        return None

    combined_label_col_name = f"knn_labels_trial_{trial_num}"
    test_pred_col_name = f"knn_pred_only_trial_{trial_num}"

    N = dataset.features.shape[0]
    # 初始化
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
    print(f"Saved combined labels to obs['{combined_label_col_name}'], test predictions to obs['{test_pred_col_name}']")
    return dataset.adata


# ========================== 7. 保存结果文件 ==========================
def save_results_to_files(results, args, final_adata=None):
    import json
    from datetime import datetime
    result_file = f"obsm_key_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w') as f:
        f.write("obsm_key\tmean_accuracy\tstd_accuracy\tmean_ari\tstd_ari\tk_smooth_used\n")
        for res in results:
            f.write(f"{res['obsm_key']}\t{res['mean_accuracy']:.4f}\t{res['std_accuracy']:.4f}\t"
                    f"{res['mean_ari']:.4f}\t{res['std_ari']:.4f}\t{res['k_smooth_used']}\n")
    print(f"Results saved to: {result_file}")

    if final_adata is not None:
        out_dir = Path("knn_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"knn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
        final_adata.write(out_path)
        print(f"AnnData with predictions saved to: {out_path}")


# ========================== 8. 主函数 ==========================
def main():# /mnt/data/test2/anaconda3/envs/project1/bin/python project1/cls_knn2.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_mode", type=str, default="topk_genes", choices=["emb", "topk_genes"])
    parser.add_argument("--h5ad_path", type=str,
                        default="project1/spatial_data/down_stream_data/raw_data_DLPFC/DLPFC",
                        help="Path to a .h5ad file OR a directory containing multiple .h5ad files")
    parser.add_argument("--emb_name", type=str, default="X_emb512_model40_16",
                        help="obsm key for embeddings (if feature_mode='emb')")
    parser.add_argument("--topk", type=int, default=512,
                        help="Top-k HVGs (if feature_mode='topk_genes')")
    parser.add_argument("--label_key", type=str, default="sce.layer_guess")
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--k_smooth", type=int, default=0,
                        help="K for within-file KNN smoothing (0 to disable)")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--save_predictions", type=bool, default=True)

    args = parser.parse_args()
    if args.feature_mode == "emb" and not args.emb_name:
        parser.error("--emb_name required for feature_mode='emb'")
    if args.feature_mode == "topk_genes" and args.topk <= 0:
        parser.error("--topk must be positive for feature_mode='topk_genes'")

    # 可配置需要测试的 obsm_keys（仅在 emb 模式下有意义）
    if args.feature_mode == "emb":
        obsm_keys = [
            'X_emb_scGPT',              # 示例键2  
            'X_emb_scGPTspatial',       # 特别处理：knn_k=16
            'X_emb512_model40_16',  # 示例键1
            'X_emb512_model40_8',
            'X_emb512_model40_24',
            'X_emb512_model40_30',
            #'512',
        ]
    else:
        # topk_genes 模式只需占位一个 key 便于统一逻辑
        obsm_keys = ['TOPK_GENES_MODE']

    all_results = []

    print("=" * 80)
    print("Multi-file KNN Classification")
    print(f"Path: {args.h5ad_path}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"KNN k={args.knn_k}, train_ratio={args.train_ratio}, trials={args.num_trials}")
    print("=" * 80)

    for obsm_key in obsm_keys:
        print(f"\n{'=' * 60}")
        print(f"Testing key: {obsm_key}")
        print(f"{'=' * 60}")

        # 特殊处理 k_smooth：只有当 key == 'X_emb_scGPTspatial' 时改为 16（其余为用户指定值或 0）
        if args.feature_mode == "emb" and obsm_key == 'X_emb_scGPTspatial':
            k_smooth_val = 16
            print("Special key detected -> k_smooth=16")
        else:
            k_smooth_val = args.k_smooth

        # 存储统计
        trial_acc = []
        trial_ari = []
        final_adata = None

        for t in range(args.num_trials):
            print(f"\n--- Trial {t + 1}/{args.num_trials} ---")
            set_seed(t)

            # 动态更新 emb_name / k_smooth
            current_args = args
            if args.feature_mode == "emb":
                current_args.emb_name = obsm_key
            current_args.k_smooth = k_smooth_val

            # 读取并构建数据
            data = read_data_for_features(current_args.h5ad_path, current_args)
            dataset = FeatureDataset(data)
            print(f"Dataset total cells: {len(dataset.labels)}, classes: {dataset.num_classes}, dim: {dataset.features.shape[1]}")

            # KNN 评估
            knn_acc, ari, train_indices, test_indices, pred_labels, _ = run_knn_only(current_args, dataset, trial_num=t + 1)
            trial_acc.append(knn_acc)
            trial_ari.append(ari)

            # 保存最后一次 trial 的预测
            if current_args.save_predictions and t == args.num_trials - 1:
                final_adata = save_knn_predictions_to_adata(dataset, train_indices, test_indices, pred_labels, t + 1, current_args)

        mean_acc = np.mean(trial_acc)
        std_acc = np.std(trial_acc)
        mean_ari = np.mean(trial_ari)
        std_ari = np.std(trial_ari)

        result = {
            'obsm_key': obsm_key,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'k_smooth_used': k_smooth_val
        }
        all_results.append(result)

        print(f"\n=== Summary for {obsm_key} ===")
        print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"ARI: {mean_ari:.4f} ± {std_ari:.4f}")
        print(f"k_smooth_used: {k_smooth_val}")

        # 写结果（只在每个 key 循环结束后写一次可选）
        if current_args.save_predictions:
            save_results_to_files([result], current_args, final_adata=final_adata)

    # 全部结果汇总
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'obsm_key':<30} {'Accuracy (mean±std)':<28} {'ARI (mean±std)':<24} k_smooth")
    print("-" * 80)
    for r in all_results:
        print(f"{r['obsm_key']:<30} {r['mean_accuracy']:.2f}% ± {r['std_accuracy']:.2f}%"
              f"{' ' * (8 - len(f'{r['std_accuracy']:.2f}%'))}"
              f"{r['mean_ari']:.4f} ± {r['std_ari']:.4f}{' ' * (6 - len(f'{r['std_ari']:.4f}'))}"
              f" {r['k_smooth_used']}")
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()