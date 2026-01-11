import pandas as pd
import numpy as np
import ast # 用于在读取CSV时将字符串 "[1, 2]" 转回列表

class WordlePreprocessor:
    def __init__(self, input_file):
        # 1. 读取清洗好的数据
        self.df = pd.read_csv(input_file)
        # 确保单词列是字符串并转小写
        self.words = self.df['Word'].astype(str).str.strip().str.lower().values
        self.scores = self.df['Average_Score'].values
        
        # 2. 构建词表 (Vocabulary)
        # 映射规则: a -> 1, b -> 2 ... z -> 26
        # 0 保留给 Padding (填充)
        self.chars = sorted(list("abcdefghijklmnopqrstuvwxyz"))
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1
        
        print(f"--> 初始化完成: 词表大小 {self.vocab_size} (包含 Padding 0)")

    def tokenize(self, word):
        """
        核心功能 1: Tokenization
        将单词转换为数字索引列表。例如: 'abc' -> [1, 2, 3]
        """
        return [self.char_to_idx.get(c, 0) for c in str(word)]

    def get_feedback_vector(self, guess, target):
        """
        核心功能 2: 处理 Wordle 反馈模式
        模拟游戏规则，计算 guess 面对 target 时的颜色反馈。
        
        编码映射:
        0: 灰色 (Gray)   - 字母不存在
        1: 黄色 (Yellow) - 字母存在但位置错误
        2: 绿色 (Green)  - 字母和位置都正确
        """
        feedback = [0] * 5
        target_chars = list(target)
        guess_chars = list(guess)
        
        # 第一步: 优先处理绿色 (Exact Match)
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback[i] = 2
                target_chars[i] = None # 标记为已匹配，防止被黄色重复统计
                guess_chars[i] = None  # 标记为已处理

        # 第二步: 处理黄色 (Misplaced Match)
        for i in range(5):
            if guess_chars[i] is not None: # 如果不是绿色
                if guess_chars[i] in target_chars:
                    feedback[i] = 1
                    # 移除目标列表中第一个匹配到的字符，确保一个字母只贡献一次黄色
                    target_chars[target_chars.index(guess_chars[i])] = None
        
        return feedback

    def process_and_save(self, output_file='wordle_preprocessed_final.csv'):
        """
        处理整个数据集，生成特征并保存
        """
        results = []
        
        # 模拟策略: 假设所有数据都是基于一个固定的常用起手词 'slate'
        # (这为 LSTM 提供了一个统一的参考系来衡量 Target 的难度)
        simulation_guess = 'slate' 
        guess_vec = self.tokenize(simulation_guess)
        
        print(f"--> 开始预处理... (使用模拟猜测词: '{simulation_guess}')")
        
        for i, target in enumerate(self.words):
            if len(target) != 5: continue # 跳过异常值
            
            # 1. 目标单词向量化 (Word Vector)
            target_vec = self.tokenize(target)
            
            # 2. 生成反馈特征 (Feedback Vector)
            # 这是一个关键特征：它代表了 "这个词对于标准猜测策略来说，会产生什么样的信号"
            feedback = self.get_feedback_vector(simulation_guess, target)
            
            # 3. 收集结果
            results.append({
                'Date': self.df.iloc[i]['Date'],
                'Target_Word': target,              # 原始单词
                'Target_Vec': target_vec,           # [19, 12, 21, 13, 16]
                'Sim_Guess': simulation_guess,      # slate
                'Sim_Guess_Vec': guess_vec,         # [19, 12, 1, 20, 5]
                'Feedback_Vec': feedback,           # [2, 2, 0, 0, 0]
                'Difficulty_Score': self.scores[i]  # 预测目标 y
            })
            
        # 转换为 DataFrame
        df_processed = pd.DataFrame(results)
        
        # 保存
        df_processed.to_csv(output_file, index=False)
        print(f"--> 处理完成! 结果已保存为: {output_file}")
        print(f"--> 数据形状: {df_processed.shape}")
        
        return df_processed

# ==========================================
# 运行部分
# ==========================================
if __name__ == "__main__":
    # 确保文件名与你本地的一致
    input_filename = 'cleaned_wordle_data.csv' 
    
    try:
        # 实例化处理器
        processor = WordlePreprocessor(input_filename)
        
        # 执行处理
        df_result = processor.process_and_save()
        
        # 打印预览，供你检查
        print("\n=== 数据预览 (Top 5) ===")
        print(df_result[['Target_Word', 'Target_Vec', 'Feedback_Vec', 'Difficulty_Score']].head())
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{input_filename}'。请确保它在当前目录下。")