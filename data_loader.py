import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import ast # 用于把字符串 "[1, 2]" 转回列表

# ==========================================
# 1. 定义 PyTorch 数据集 (Dataset)
# ==========================================
class WordleDataset(Dataset):
    def __init__(self, csv_file, npy_file):
        # 1. 读取 CSV (包含字符索引、反馈向量、标签)
        self.df = pd.read_csv(csv_file)
        
        # 2. 读取 NPY (包含 Word2Vec 语义向量)
        self.semantic_vectors = np.load(npy_file)
        
        # 3. 准备数据
        # ast.literal_eval 安全地将字符串形式的列表转回 Python list
        self.char_features = [ast.literal_eval(x) for x in self.df['Target_Vec']]
        self.feedback_features = [ast.literal_eval(x) for x in self.df['Feedback_Vec']]
        self.labels = self.df['Difficulty_Score'].values
        
        # 检查对齐
        assert len(self.char_features) == len(self.semantic_vectors), \
            "错误：CSV行数与NPY向量数不一致！"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取第 idx 个样本的所有特征
        
        # A. 字符特征 (5个整数) -> 转为 LongTensor
        x_char = torch.LongTensor(self.char_features[idx])
        
        # B. 语义特征 (100个浮点数, 来自 GloVe) -> FloatTensor
        x_semantic = torch.FloatTensor(self.semantic_vectors[idx])
        
        # C. 反馈特征 (5个整数) -> FloatTensor
        x_feedback = torch.FloatTensor(self.feedback_features[idx])
        
        # D. 标签 (难度分数) -> FloatTensor
        y = torch.FloatTensor([self.labels[idx]])
        
        return (x_char, x_semantic, x_feedback), y

# ==========================================
# 2. 辅助函数：自动获取 DataLoader
# ==========================================
def get_dataloaders(batch_size=16, split_ratio=0.8):
    """
    自动寻找文件并返回训练集和测试集的 DataLoader
    """
    # 自动定位文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'wordle_preprocessed_final.csv')
    npy_path = os.path.join(script_dir, 'word_embeddings.npy')
    
    if not os.path.exists(csv_path) or not os.path.exists(npy_path):
        print("❌ 错误：在当前目录下找不到 .csv 或 .npy 文件！")
        return None, None

    # 实例化 Dataset
    full_dataset = WordleDataset(csv_path, npy_path)
    
    # 划分 训练集 / 测试集
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    print(f"数据加载成功！总样本: {len(full_dataset)}")
    print(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ==========================================
# 测试代码 (运行这个文件来检查数据管道是否通畅)
# ==========================================
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    
    if train_loader:
        # 取出一个 Batch 看看长什么样
        (x_char, x_sem, x_fb), y = next(iter(train_loader))
        
        print("\n=== 一个 Batch 的数据形状检查 ===")
        print(f"1. 字符输入 (Target_Vec): {x_char.shape}  -> [Batch, 5]")
        print(f"2. 语义输入 (Word2Vec):   {x_sem.shape}   -> [Batch, 100]")
        print(f"3. 反馈输入 (Feedback):   {x_fb.shape}    -> [Batch, 5]")
        print(f"4. 预测目标 (Difficulty): {y.shape}       -> [Batch, 1]")
        print("\n✅ 数据管道搭建完成！下一步可以写 BiLSTM 模型了。")