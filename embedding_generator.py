import pandas as pd
import numpy as np
import gensim.downloader as api
import os
import sys

# ==========================================
# 1. 自动路径定位工具 (防止找不到文件)
# ==========================================
def get_file_path(filename):
    """
    智能寻找文件：优先在脚本所在目录找，其次在当前工作目录找。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 路径 1: 脚本同级目录
    path_in_script_dir = os.path.join(script_dir, filename)
    if os.path.exists(path_in_script_dir):
        return path_in_script_dir
    
    # 路径 2: 运行时的当前目录
    if os.path.exists(filename):
        return os.path.abspath(filename)
        
    return None

# ==========================================
# 2. 核心类: 词向量生成器
# ==========================================
class EmbeddingGenerator:
    def __init__(self, csv_path):
        print(f"-->正在读取预处理数据: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # 提取目标单词列 (Target_Word)，转小写并去空格
        # 注意：这里我们只为 Target Word 生成向量，因为这是 LSTM 的核心输入
        self.words = self.df['Target_Word'].astype(str).str.strip().str.lower().values
        self.vectors = []
        
        # 嵌入维度 (GloVe 100维)
        self.embedding_dim = 100 
        
    def load_pretrained_model(self, model_name="glove-wiki-gigaword-100"):
        """
        加载 Gensim 的预训练模型。
        参数 model_name: 
          - 'glove-wiki-gigaword-100' (推荐: 100维, 约128MB, 速度快效果好)
          - 'word2vec-google-news-300' (最强: 300维, 约1.6GB, 下载慢)
        """
        print(f"\n[1/3] 正在加载预训练模型 '{model_name}'...")
        print("      (首次运行会自动下载，可能需要几分钟，请耐心等待...)")
        
        try:
            self.model = api.load(model_name)
            print(f"✅ 模型加载成功! (词表大小: {len(self.model.index_to_key)})")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("提示: 请检查网络连接。如果下载太慢，可以尝试更换网络或使用 VPN。")
            sys.exit(1)

    def generate_embedding_matrix(self):
        """
        核心逻辑：遍历我们的单词表，去预训练模型里查表
        """
        print(f"\n[2/3] 正在为 {len(self.words)} 个单词生成向量...")
        
        matrix_list = []
        found_count = 0
        oov_count = 0 # Out-Of-Vocabulary (未登录词)
        
        for word in self.words:
            if word in self.model:
                # 情况 A: 单词在字典里 -> 直接获取向量
                vec = self.model[word]
                found_count += 1
            else:
                # 情况 B: 单词不在字典里 (生僻词) -> 使用随机向量初始化
                # 保持与预训练向量相似的分布 (均值0, 方差0.6)
                vec = np.random.normal(scale=0.6, size=(self.embedding_dim,))
                oov_count += 1
                print(f"   ⚠️ [生僻词发现] '{word}' 未在模型中找到，已使用随机向量代替。")
            
            matrix_list.append(vec)
            
        # 转换为 NumPy 矩阵 (Shape: N_samples x Embedding_dim)
        self.vectors = np.array(matrix_list)
        
        print(f"\n--> 统计结果:")
        print(f"    ✅ 完美匹配: {found_count} 个")
        print(f"    ⚠️ 未登录词: {oov_count} 个")
        print(f"    生成的矩阵形状: {self.vectors.shape} (行数应等于 {len(self.words)})")

    def save_results(self, output_filename='word_embeddings.npy'):
        """
        保存结果为 .npy 文件
        """
        print(f"\n[3/3] 保存结果...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, output_filename)
        
        np.save(save_path, self.vectors)
        print(f"🎉 成功! 词向量矩阵已保存至: {save_path}")
        print("------------------------------------------------")
        print("【交付指南】")
        print("请将 'word_embeddings.npy' 发给 Member A。")
        print("并在报告中说明：'使用了基于 Wikipedia 语料库训练的 GloVe-100d 模型进行迁移学习，以确保输入层的语义表达质量。'")

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 你的预处理文件
    input_csv = 'wordle_preprocessed_final.csv'
    
    # 获取文件路径
    file_path = get_file_path(input_csv)
    
    if file_path:
        # 实例化并运行
        generator = EmbeddingGenerator(file_path)
        
        # 步骤 1: 加载 GloVe 模型
        generator.load_pretrained_model("glove-wiki-gigaword-100")
        
        # 步骤 2: 生成向量
        generator.generate_embedding_matrix()
        
        # 步骤 3: 保存
        generator.save_results()
        
    else:
        print(f"❌ 错误: 找不到文件 '{input_csv}'。")
        print("请确保你已经完成了上一步的文本预处理，并且文件就在当前目录下。")