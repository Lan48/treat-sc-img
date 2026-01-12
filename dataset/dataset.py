import os
import anndata
import numpy as np
import scipy.sparse as sp
import pandas as pd
import heapq
import random
import math

'''
class SpatialTranscriptomicsDataset:
    def __init__(self, config):
        """
        初始化Dataset类
        :param data_dir: 包含h5ad文件的目录路径
        :param h: 固定高度
        :param w: 固定宽度
        :param pad_id: <pad>的基因ID（整数）
        :param max_length: 最大基因长度
        """
        self.data_dir = config['data_dir']
        self.h = config['h']
        self.w = config['w']
        self.pad_id = int(config['pad_id'])  # 确保pad_id为整数
        self.max_length = config['max_length']
        self.file_list = sorted([f for f in os.listdir(config['data_dir']) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(config['data_dir'], f) for f in self.file_list]
        self.is_bin = config.get('is_bin', False)
        self.bins = config.get('bins', 50)
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        adata = anndata.read_h5ad(file_path)
        
        # 创建全零矩阵 (max_length, h, w)
        expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        
        # 获取有效基因信息
        is_selected = adata.var['is_selected'].values
        gene_ids = adata.var['gene_ids'].values.astype(np.int32)  # 确保基因ID为整数
        valid_gene_ids = gene_ids[is_selected]
        n_valid = len(valid_gene_ids)
        
        # 构建基因ID序列 (长度为max_length的整数数组)
        gene_id_seq = np.full(self.max_length, self.pad_id, dtype=np.int32)
        valid_len = min(n_valid, self.max_length)
        gene_id_seq[:valid_len] = valid_gene_ids[:self.max_length]
        
        # 获取坐标并确保为整数类型
        coords = adata.obsm['coords_sample'].astype(np.int32)
        
        # 只取有效基因的表达数据
        X_valid = adata.X[:, is_selected].toarray() if hasattr(adata.X, 'toarray') else adata.X[:, is_selected]
        X_valid = X_valid[:, :valid_len]  # 截断到有效长度
        
        # 优化后的单层循环（spot数量不多）
        for i in range(X_valid.shape[0]):
            x, y = coords[i]
            # 检查坐标是否在有效范围内
            if 0 <= x < self.h and 0 <= y < self.w:
                # 将当前spot的基因表达向量（一维）放入矩阵
                expression_matrix[:valid_len, x, y] = X_valid[i]
        
        return expression_matrix, ori_expression_matrix,gene_id_seq

class SpatialTranscriptomicsDataset:
    def __init__(self, config):
        """
        初始化Dataset类
        :param data_dir: 包含h5ad文件的目录路径
        :param h: 固定高度
        :param w: 固定宽度
        :param pad_id: <pad>的基因ID（整数）
        :param max_length: 最大基因长度
        :param is_bin: 是否进行分位数分箱
        :param bins: 分箱数量
        """
        self.data_dir = config['data_dir']
        self.h = config['h']
        self.w = config['w']
        self.pad_id = int(config['pad_id'])  # 确保pad_id为整数
        self.max_length = config['c']
        self.file_list = sorted([f for f in os.listdir(config['data_dir']) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(config['data_dir'], f) for f in self.file_list]
        self.is_bin = config.get('is_bin', False)
        self.bins = config.get('bins', 50)
    
    def _quantile_bin(self, expression_data):
        """
        对基因表达数据进行分位数分箱
        :param expression_data: 原始表达矩阵
        :return: 分箱后的表达矩阵
        """
        # 展平表达矩阵以便分箱
        flat_data = expression_data.flatten()
        
        # 移除零值（背景）或根据需求处理
        non_zero_data = flat_data[flat_data > 0]
        
        if len(non_zero_data) == 0:
            # 如果没有非零值，返回全零矩阵
            binned_data = np.zeros_like(flat_data)
        else:
            # 计算分位数
            quantiles = np.percentile(non_zero_data, np.linspace(0, 100, self.bins + 1))
            
            # 确保分位数是唯一的且递增
            quantiles = np.unique(quantiles)
            if len(quantiles) < 2:
                quantiles = np.array([0, np.max(non_zero_data)])
            
            # 使用pd.cut进行分箱
            try:
                binned_flat = pd.cut(flat_data, bins=quantiles, labels=False, include_lowest=True)
                binned_flat = np.nan_to_num(binned_flat, nan=0.0)  # 将NaN替换为0
                binned_data = binned_flat.astype(np.float32)
            except Exception as e:
                print(f"分箱过程中出错: {e}")
                binned_data = np.zeros_like(flat_data, dtype=np.float32)
        
        # 恢复原始形状
        return binned_data.reshape(expression_data.shape)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        adata = anndata.read_h5ad(file_path)
        
        # 创建全零矩阵 (max_length, h, w)
        binned_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        ori_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        
        # 获取有效基因信息
        is_selected = adata.var['is_selected'].values
        gene_ids = adata.var['gene_ids'].values.astype(np.int32)  # 确保基因ID为整数
        valid_gene_ids = gene_ids[is_selected]
        n_valid = len(valid_gene_ids)
        
        # 构建基因ID序列 (长度为max_length的整数数组)
        gene_id_seq = np.full(self.max_length, self.pad_id, dtype=np.int32)
        valid_len = min(n_valid, self.max_length)
        gene_id_seq[:valid_len] = valid_gene_ids[:self.max_length]
        
        # 获取坐标并确保为整数类型
        coords = adata.obsm['coords_sample'].astype(np.int32)
        
        # 只取有效基因的表达数据
        X_valid = adata.X[:, is_selected].toarray() if hasattr(adata.X, 'toarray') else adata.X[:, is_selected]
        X_valid = X_valid[:, :valid_len]  # 截断到有效长度
        
        # 创建临时矩阵存储原始表达数据
        temp_ori_matrix = np.zeros((valid_len, self.h, self.w), dtype=np.float32)
        
        # 填充原始表达矩阵
        for i in range(X_valid.shape[0]):
            x, y = coords[i]
            if 0 <= x < self.h and 0 <= y < self.w:
                temp_ori_matrix[:valid_len, x, y] = X_valid[i].T  # 转置以使基因维度在前
        
        # 存储原始表达矩阵
        ori_expression_matrix[:valid_len, :, :] = temp_ori_matrix
        
        # 根据是否分箱处理表达矩阵
        if self.is_bin:
            # 对每个基因进行分箱处理
            for gene_idx in range(valid_len):
                gene_data = temp_ori_matrix[gene_idx, :, :]
                binned_data = self._quantile_bin(gene_data)
                binned_expression_matrix[gene_idx, :, :] = binned_data
        else:
            # 不使用分箱，直接使用原始数据
            binned_expression_matrix[:valid_len, :, :] = temp_ori_matrix
        
        return binned_expression_matrix, ori_expression_matrix, gene_id_seq

class SpatialTranscriptomicsDataset:
    def __init__(self, config):
        """
        初始化Dataset类
        :param config: 配置字典，包含data_dir、h、w、pad_id、max_length等参数
        """
        self.data_dir = config['data_dir']
        self.h = config['h']
        self.w = config['w']
        self.pad_id = int(config['pad_id'])  # 确保pad_id为整数
        self.max_length = config['c']
        self.file_list = sorted([f for f in os.listdir(config['data_dir']) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(config['data_dir'], f) for f in self.file_list]
        self.is_bin = config.get('is_bin', False)
        self.bins = config.get('bins', 50)
        self.normalize = config.get('normalize', 0)
    @staticmethod
    def weighted_random_sample(items_weights, k):
        """
        使用A-Res算法进行加权随机采样（不放回）。
        :param items_weights: 列表，每个元素为(item, weight)元组
        :param k: 采样数量
        :return: 采样到的item列表
        """
        heap = []  # 最小堆，存储(ki, item)
        for item, weight in items_weights:
            ui = random.uniform(0, 1)
            ki = ui ** (1 / weight)  # 计算特征值
            if len(heap) < k:
                heapq.heappush(heap, (ki, item))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, item))
                if len(heap) > k:
                    heapq.heappop(heap)
        return [item for _, item in heap]  # 返回采样到的items
    
    def _quantile_bin(self, expression_data):
        """
        对基因表达数据进行分位数分箱（保持不变）。
        :param expression_data: 原始表达矩阵
        :return: 分箱后的表达矩阵
        """
        flat_data = expression_data.flatten()
        non_zero_data = flat_data[flat_data > 0]
        if len(non_zero_data) == 0:
            binned_data = np.zeros_like(flat_data)
        else:
            quantiles = np.percentile(non_zero_data, np.linspace(0, 100, self.bins + 1))
            quantiles = np.unique(quantiles)
            if len(quantiles) < 2:
                quantiles = np.array([0, np.max(non_zero_data)])
            try:
                binned_flat = pd.cut(flat_data, bins=quantiles, labels=False, include_lowest=True)
                binned_flat = np.nan_to_num(binned_flat, nan=0.0)
                binned_data = binned_flat.astype(np.float32)
            except Exception as e:
                print(f"分箱过程中出错: {e}")
                binned_data = np.zeros_like(flat_data, dtype=np.float32)
        return binned_data.reshape(expression_data.shape)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        adata = anndata.read_h5ad(file_path)
        
        # 初始化输出矩阵
        binned_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        ori_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        
        # 获取有效基因信息
        is_selected = adata.var['is_selected'].values
        gene_ids = adata.var['gene_ids'].values.astype(np.int32)
        valid_gene_ids = gene_ids[is_selected]
        n_valid = len(valid_gene_ids)
        
        # 获取表达数据并确保为密集矩阵
        X_valid = adata.X[:, is_selected]
        if hasattr(X_valid, 'toarray'):
            X_valid = X_valid.toarray()
        X_valid = X_valid.astype(np.float32)
        
        # 计算每个基因的方差作为权重
        variances = np.var(X_valid, axis=0)  # 轴0: 计算每个基因的方差
        variances = np.maximum(variances, 1e-10)  # 确保权重为正，避免除零错误
        
        # 根据基因数量决定是否采样
        if n_valid > self.max_length:
            # 加权随机采样：选择方差大的基因
            items_weights = list(zip(range(n_valid), variances))  # 每个元素为(索引, 方差)
            sampled_indices = self.weighted_random_sample(items_weights, self.max_length)
            valid_gene_ids_sampled = valid_gene_ids[sampled_indices]
            X_valid_sampled = X_valid[:, sampled_indices]
            valid_len = self.max_length
        else:
            # 直接取所有基因
            valid_gene_ids_sampled = valid_gene_ids
            X_valid_sampled = X_valid
            valid_len = n_valid
        
        # 构建基因ID序列
        gene_id_seq = np.full(self.max_length, self.pad_id, dtype=np.int32)
        gene_id_seq[:valid_len] = valid_gene_ids_sampled
        
        # 获取坐标并确保为整数
        coords = adata.obsm['coords_sample'].astype(np.int32)
        
        # 创建临时矩阵存储原始表达数据
        temp_ori_matrix = np.zeros((valid_len, self.h, self.w), dtype=np.float32)
        for i in range(X_valid_sampled.shape[0]):  # 遍历每个spot
            x, y = coords[i]
            if 0 <= x < self.h and 0 <= y < self.w:
                temp_ori_matrix[:, x, y] = X_valid_sampled[i]  # 分配表达数据
        
        # 存储原始表达矩阵
        ori_expression_matrix[:valid_len, :, :] = temp_ori_matrix
        
        # 根据是否分箱处理表达矩阵
        if self.is_bin:
            for gene_idx in range(valid_len):
                gene_data = temp_ori_matrix[gene_idx, :, :]
                binned_data = self._quantile_bin(gene_data)
                binned_expression_matrix[gene_idx, :, :] = binned_data
        else:
            binned_expression_matrix[:valid_len, :, :] = temp_ori_matrix
        
        return binned_expression_matrix, ori_expression_matrix, gene_id_seq
'''
'''
class SpatialTranscriptomicsDataset:
    def __init__(self, config):
        """
        初始化Dataset类
        :param config: 配置字典，包含data_dir、h、w、pad_id、max_length等参数
        """
        self.data_dir = config['data_dir']
        self.h = config['h']
        self.w = config['w']
        self.pad_id = int(config['pad_id'])  # 确保pad_id为整数
        self.max_length = config['c']
        self.file_list = sorted([f for f in os.listdir(config['data_dir']) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(config['data_dir'], f) for f in self.file_list]
        self.is_bin = config.get('is_bin', False)
        self.bins = config.get('bins', 50)
        self.normalize = config.get('normalize', 0)
    @staticmethod
    def weighted_random_sample(items_weights, k):
        """
        使用A-Res算法进行加权随机采样（不放回）。
        :param items_weights: 列表，每个元素为(item, weight)元组
        :param k: 采样数量
        :return: 采样到的item列表
        """
        heap = []  # 最小堆，存储(ki, item)
        for item, weight in items_weights:
            ui = random.uniform(0, 1)
            ki = ui ** (1 / weight)  # 计算特征值
            if len(heap) < k:
                heapq.heappush(heap, (ki, item))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, item))
                if len(heap) > k:
                    heapq.heappop(heap)
        return [item for _, item in heap]  # 返回采样到的items
    
    def _quantile_bin(self, expression_data):
        """
        对基因表达数据进行分位数分箱（保持不变）。
        :param expression_data: 原始表达矩阵
        :return: 分箱后的表达矩阵
        """
        flat_data = expression_data.flatten()
        non_zero_data = flat_data[flat_data > 0]
        if len(non_zero_data) == 0:
            binned_data = np.zeros_like(flat_data)
        else:
            quantiles = np.percentile(non_zero_data, np.linspace(0, 100, self.bins + 1))
            quantiles = np.unique(quantiles)
            if len(quantiles) < 2:
                quantiles = np.array([0, np.max(non_zero_data)])
            try:
                binned_flat = pd.cut(flat_data, bins=quantiles, labels=False, include_lowest=True)
                binned_flat = np.nan_to_num(binned_flat, nan=0.0)
                binned_data = binned_flat.astype(np.float32)
            except Exception as e:
                print(f"分箱过程中出错: {e}")
                binned_data = np.zeros_like(flat_data, dtype=np.float32)
        return binned_data.reshape(expression_data.shape)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        adata = anndata.read_h5ad(file_path)
        
        # 初始化输出矩阵
        binned_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        ori_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        
        # 获取有效基因信息
        is_selected = adata.var['is_selected'].values
        gene_ids = adata.var['gene_ids'].values.astype(np.int32)
        valid_gene_ids = gene_ids[is_selected]
        n_valid = len(valid_gene_ids)
        
        # 获取表达数据并确保为密集矩阵
        X_valid = adata.X[:, is_selected]
        if hasattr(X_valid, 'toarray'):
            X_valid = X_valid.toarray()
        X_valid = X_valid.astype(np.float32)
        if self.normalize > 0:
            # 新增：判断是否经过log1p处理并还原[1,2](@ref)
            if self._is_log1p_processed(X_valid):
                X_valid2 = X_valid
                X_valid = np.expm1(X_valid)  # 还原为原始尺度
            
            # 新增：应用normalize_total风格归一化[6,7](@ref)
            X_normalized = self._normalize_total(X_valid, target_sum=self.normalize)
            # 新增：在归一化后重新应用log1p处理[2,6](@ref)
            X_normalized = np.log1p(X_normalized)
        # 计算每个基因的方差作为权重（使用归一化后数据）
        variances = np.var(X_normalized, axis=0)
        variances = np.maximum(variances, 1e-10)
        
        # 根据基因数量决定是否采样
        if n_valid > self.max_length:
            items_weights = list(zip(range(n_valid), variances))
            sampled_indices = self.weighted_random_sample(items_weights, self.max_length)
            valid_gene_ids_sampled = valid_gene_ids[sampled_indices]
            X_final = X_normalized[:, sampled_indices]  # 使用归一化后数据
            valid_len = self.max_length
        else:
            valid_gene_ids_sampled = valid_gene_ids
            X_final = X_normalized  # 使用归一化后数据
            valid_len = n_valid
        
        # 构建基因ID序列
        gene_id_seq = np.full(self.max_length, self.pad_id, dtype=np.int32)
        gene_id_seq[:valid_len] = valid_gene_ids_sampled
        
        # 获取坐标并确保为整数
        coords = adata.obsm['coords_sample'].astype(np.int32)
        
        # 创建临时矩阵存储处理后的表达数据
        temp_matrix = np.zeros((valid_len, self.h, self.w), dtype=np.float32)
        for i in range(X_final.shape[0]):  # 遍历每个spot
            x, y = coords[i]
            if 0 <= x < self.h and 0 <= y < self.w:
                temp_matrix[:, x, y] = X_final[i]  # 分配表达数据
        
        # 存储到输出矩阵
        ori_expression_matrix[:valid_len, :, :] = temp_matrix
        
        # 根据是否分箱处理表达矩阵
        if self.is_bin:
            for gene_idx in range(valid_len):
                gene_data = temp_matrix[gene_idx, :, :]
                binned_data = self._quantile_bin(gene_data)
                binned_expression_matrix[gene_idx, :, :] = binned_data
        else:
            binned_expression_matrix[:valid_len, :, :] = temp_matrix
        
        return binned_expression_matrix, ori_expression_matrix, gene_id_seq

    # 新增辅助方法
    def _is_log1p_processed(self, X):
        """
        判断数据是否经过log1p处理[1,5](@ref)
        启发式规则：最大值小于50且存在小数值时认为已处理
        """
        max_val = np.max(X)
        if max_val < 50:  # log处理后值通常较小[1](@ref)
            if np.any(X != np.round(X)):  # 检查非整数值
                return True
        return False

    def _normalize_total(self, X, target_sum=1e4):
        """
        实现类似sc.pp.normalize_total的归一化[6,7](@ref)
        """
        counts_per_cell = np.sum(X, axis=1)
        if target_sum is None:
            target_sum = np.median(counts_per_cell)  # 默认使用中值[8](@ref)
        
        # 处理零值避免除零错误
        counts_per_cell = np.where(counts_per_cell == 0, 1.0, counts_per_cell)
        scaling_factors = target_sum / counts_per_cell
        X_normalized = X * scaling_factors[:, np.newaxis]  # 广播缩放因子
        return X_normalized    
'''    

class SpatialTranscriptomicsDataset:
    def __init__(self, config):
        """
        初始化Dataset类
        :param config: 配置字典，包含如下参数：
            - data_dir: 数据目录
            - h: 高度
            - w: 宽度
            - pad_id: pad标记id
            - c: 最大基因通道数（max_length）
            - is_bin: 是否对表达矩阵做分箱（可选，默认False）
            - bins: 分箱数（可选，默认50）
            - normalize: normalize_total的目标和（可选，默认0表示不归一化）
            - dataset_ratio: 仅使用数据集前多少比例的样本（0~1，默认1.0；例如0.2表示仅使用前20%的文件）
        """
        self.data_dir = config['data_dir']
        self.h = config['h']
        self.w = config['w']
        self.pad_id = int(config['pad_id'])  # 确保pad_id为整数
        self.max_length = config['c']
        self.is_bin = config.get('is_bin', False)
        self.bins = config.get('bins', 50)
        self.normalize = config.get('normalize', 0)

        # 新增：控制数据集使用比例
        dataset_ratio = float(config.get('dataset_ratio', 1.0))
        if not np.isfinite(dataset_ratio):
            dataset_ratio = 1.0
        self.dataset_ratio = max(0.0, min(1.0, dataset_ratio))

        # 列出并排序所有.h5ad文件
        self.file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(self.data_dir, f) for f in self.file_list]

        # 仅保留前 dataset_ratio 比例的数据
        if self.dataset_ratio < 1.0:
            n_total = len(self.file_paths)
            n_keep = int(math.ceil(n_total * self.dataset_ratio))
            # 若比例>0但四舍五入后为0，则至少保留1个
            if self.dataset_ratio > 0.0 and n_total > 0 and n_keep == 0:
                n_keep = 1
            # 当dataset_ratio==0时，n_keep为0，数据集长度为0
            self.file_list = self.file_list[:n_keep]
            self.file_paths = self.file_paths[:n_keep]

    @staticmethod
    def weighted_random_sample(items_weights, k):
        """
        使用A-Res算法进行加权随机采样（不放回）。
        :param items_weights: 列表，每个元素为(item, weight)元组
        :param k: 采样数量
        :return: 采样到的item列表
        """
        heap = []  # 最小堆，存储(ki, item)
        for item, weight in items_weights:
            ui = random.uniform(0, 1)
            ki = ui ** (1 / weight)  # 计算特征值
            if len(heap) < k:
                heapq.heappush(heap, (ki, item))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, item))
                if len(heap) > k:
                    heapq.heappop(heap)
        return [item for _, item in heap]  # 返回采样到的items

    def _quantile_bin(self, expression_data):
        """
        对基因表达数据进行分位数分箱（保持不变）。
        :param expression_data: 原始表达矩阵
        :return: 分箱后的表达矩阵
        """
        flat_data = expression_data.flatten()
        non_zero_data = flat_data[flat_data > 0]
        if len(non_zero_data) == 0:
            binned_data = np.zeros_like(flat_data)
        else:
            quantiles = np.percentile(non_zero_data, np.linspace(0, 100, self.bins + 1))
            quantiles = np.unique(quantiles)
            if len(quantiles) < 2:
                quantiles = np.array([0, np.max(non_zero_data)])
            try:
                binned_flat = pd.cut(flat_data, bins=quantiles, labels=False, include_lowest=True)
                binned_flat = np.nan_to_num(binned_flat, nan=0.0)
                binned_data = binned_flat.astype(np.float32)
            except Exception as e:
                print(f"分箱过程中出错: {e}")
                binned_data = np.zeros_like(flat_data, dtype=np.float32)
        return binned_data.reshape(expression_data.shape)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        adata = anndata.read_h5ad(file_path)

        # 初始化输出矩阵
        binned_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)
        ori_expression_matrix = np.zeros((self.max_length, self.h, self.w), dtype=np.float32)

        # 获取有效基因信息
        is_selected = adata.var['is_selected'].values
        gene_ids = adata.var['gene_ids'].values.astype(np.int32)
        valid_gene_ids = gene_ids[is_selected]
        n_valid = len(valid_gene_ids)

        # 获取表达数据并确保为密集矩阵
        X_valid = adata.X[:, is_selected]
        if hasattr(X_valid, 'toarray'):
            X_valid = X_valid.toarray()
        X_valid = X_valid.astype(np.float32)

        if self.normalize > 0:
            # 新增：判断是否经过log1p处理并还原[1,2](@ref)
            if self._is_log1p_processed(X_valid):
                X_valid2 = X_valid
                X_valid = np.expm1(X_valid)  # 还原为原始尺度

            # 新增：应用normalize_total风格归一化[6,7](@ref)
            X_normalized = self._normalize_total(X_valid, target_sum=self.normalize)
            # 新增：在归一化后重新应用log1p处理[2,6](@ref)
            X_normalized = np.log1p(X_normalized)
        else:
            # 保证在不归一化时也能使用后续变量
            X_normalized = X_valid

        # 计算每个基因的方差作为权重（使用归一化后数据或原始数据）
        variances = np.var(X_normalized, axis=0)
        variances = np.maximum(variances, 1e-10)

        # 根据基因数量决定是否采样
        if n_valid > self.max_length:
            items_weights = list(zip(range(n_valid), variances))
            sampled_indices = self.weighted_random_sample(items_weights, self.max_length)
            valid_gene_ids_sampled = valid_gene_ids[sampled_indices]
            X_final = X_normalized[:, sampled_indices]  # 使用归一化后数据
            valid_len = self.max_length
        else:
            valid_gene_ids_sampled = valid_gene_ids
            X_final = X_normalized  # 使用归一化后数据
            valid_len = n_valid

        # 构建基因ID序列
        gene_id_seq = np.full(self.max_length, self.pad_id, dtype=np.int32)
        gene_id_seq[:valid_len] = valid_gene_ids_sampled

        # 获取坐标并确保为整数
        coords = adata.obsm['coords_sample'].astype(np.int32)

        # 创建临时矩阵存储处理后的表达数据
        temp_matrix = np.zeros((valid_len, self.h, self.w), dtype=np.float32)
        for i in range(X_final.shape[0]):  # 遍历每个spot
            x, y = coords[i]
            if 0 <= x < self.h and 0 <= y < self.w:
                temp_matrix[:, x, y] = X_final[i]  # 分配表达数据

        # 存储到输出矩阵
        ori_expression_matrix[:valid_len, :, :] = temp_matrix

        # 根据是否分箱处理表达矩阵
        if self.is_bin:
            for gene_idx in range(valid_len):
                gene_data = temp_matrix[gene_idx, :, :]
                binned_data = self._quantile_bin(gene_data)
                binned_expression_matrix[gene_idx, :, :] = binned_data
        else:
            binned_expression_matrix[:valid_len, :, :] = temp_matrix

        return binned_expression_matrix, ori_expression_matrix, gene_id_seq

    # 新增辅助方法
    def _is_log1p_processed(self, X):
        """
        判断数据是否经过log1p处理[1,5](@ref)
        启发式规则：最大值小于50且存在小数值时认为已处理
        """
        max_val = np.max(X)
        if max_val < 50:  # log处理后值通常较小[1](@ref)
            if np.any(X != np.round(X)):  # 检查非整数值
                return True
        return False

    def _normalize_total(self, X, target_sum=1e4):
        """
        实现类似sc.pp.normalize_total的归一化[6,7](@ref)
        """
        counts_per_cell = np.sum(X, axis=1)
        if target_sum is None:
            target_sum = np.median(counts_per_cell)  # 默认使用中值[8](@ref)

        # 处理零值避免除零错误
        counts_per_cell = np.where(counts_per_cell == 0, 1.0, counts_per_cell)
        scaling_factors = target_sum / counts_per_cell
        X_normalized = X * scaling_factors[:, np.newaxis]  # 广播缩放因子
        return X_normalized
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    config = {
        'normalize': 10000,
        'is_bin': False,  # 是否使用binning'
        'bins': 50,          # 块数
        'c': 512,           # 最大基因长度
        'h': 14,            # 高度
        'w': 14,            # 宽度
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': 0,  # 填充ID
        'num_genes': 1000, # 基因数量 (包括pad)
    }
    # 确保pad_id是整数类型
    ds = SpatialTranscriptomicsDataset(
        config)

    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        spatial_mats = batch[0]  # shape: [B, max_genes, H, W]
        gene_id_vecs = batch[1]  # shape: [B, max_genes]
        
        print(f"Spatial mats shape: {spatial_mats.shape}")
        print(spatial_mats[0][0][1])
        print(f"Gene ID vecs shape: {gene_id_vecs.shape}")
        print(gene_id_vecs[0][0][1])
        break