import pandas as pd
import numpy as np
import os

# ==========================================
# 0. 路径工具
# ==========================================
def get_file_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

class TimeSeriesBuilder:
    def __init__(self, window_size=7):
        """
        window_size: 回看过去多少天 (比如 7 天)
        """
        self.window_size = window_size
        
    def load_data(self):
        # 读取之前做好的数据
        csv_path = get_file_path("wordle_preprocessed_final.csv")
        npy_path = get_file_path("word_embeddings.npy")
        
        if not os.path.exists(csv_path) or not os.path.exists(npy_path):
            print("❌ 错误: 找不到预处理文件，请检查路径。")
            return None, None
            
        df = pd.read_csv(csv_path)
        embeddings = np.load(npy_path)
        
        # 我们需要两个核心序列：
        # 1. 难度分数序列 (主要预测目标)
        scores = df['Difficulty_Score'].values
        # 2. 语义向量序列 (辅助特征)
        # embeddings 形状是 (356, 100)
        
        return scores, embeddings

    def create_sequences(self):
        scores, embeddings = self.load_data()
        if scores is None: return

        print(f"--> 正在构建滑动窗口数据 (窗口大小={self.window_size})...")
        
        X_time = []   # 输入: 过去 N 天的数据
        y_target = [] # 输出: 第 N+1 天的难度
        
        total_days = len(scores)
        
        for i in range(total_days - self.window_size):
            # 窗口范围: [i, i+window_size]
            
            # 1. 获取过去 window_size 天的难度 (Shape: [7, 1])
            past_scores = scores[i : i + self.window_size].reshape(-1, 1)
            
            # 2. 获取过去 window_size 天的单词语义 (Shape: [7, 100])
            past_embeddings = embeddings[i : i + self.window_size]
            
            # 3. 组合特征: 把分数和语义拼起来 -> (Shape: [7, 101])
            # 这样模型既知道过去的分数走势，也知道过去的单词含义
            combined_window = np.concatenate((past_scores, past_embeddings), axis=1)
            
            # 4. 获取目标: 第 i + window_size 天的分数
            target_score = scores[i + self.window_size]
            
            X_time.append(combined_window)
            y_target.append(target_score)
            
        # 转为 numpy 数组
        X_time = np.array(X_time)
        y_target = np.array(y_target)
        
        print(f"--> 数据构建完成!")
        print(f"    输入形状 (X): {X_time.shape} (样本数, 时间步, 特征数)")
        print(f"    输出形状 (y): {y_target.shape} (样本数, )")
        
        return X_time, y_target

    def save_data(self, X, y):
        # 保存为新的 .npy 文件
        np.save(get_file_path("timeseries_X.npy"), X)
        np.save(get_file_path("timeseries_y.npy"), y)
        print(f"--> 文件已保存:\n    - timeseries_X.npy\n    - timeseries_y.npy")

if __name__ == "__main__":
    # 设定回顾过去 7 天来预测第 8 天
    builder = TimeSeriesBuilder(window_size=7)
    X, y = builder.create_sequences()
    
    if X is not None:
        builder.save_data(X, y)