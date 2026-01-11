import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_loader import get_dataloaders
from train_model import WordlePredictor  # 需要引用你刚才定义的模型类

# 设置中文字体 (防止热力图中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 加载模型与数据
# ==========================================
def load_trained_model(model_path='bi_lstm_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WordlePredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式
    return model, device

# ==========================================
# 2. 核心功能：提取 Attention 权重
# ==========================================
def analyze_attention(model, device, sample_data, char_map):
    """
    输入一个样本，返回预测值、真实值，以及 Attention 权重
    """
    x_char, x_sem, x_fb, y_true = sample_data
    
    # 增加 Batch 维度 (从 [5] 变成 [1, 5])
    x_char = x_char.unsqueeze(0).to(device)
    x_sem = x_sem.unsqueeze(0).to(device)
    x_fb = x_fb.unsqueeze(0).to(device)
    
    # 1. 前向传播 (手动分步执行，为了抓取 Attention)
    embedded = model.char_embedding(x_char)
    lstm_out, _ = model.lstm(embedded)
    
    # 这里的 model.attention 返回 (context, weights)
    # weights 形状: [1, 5, 1]
    context, attn_weights = model.attention(lstm_out)
    
    # 继续跑完剩下的层拿预测值
    combined = torch.cat((context, x_sem, x_fb), dim=1)
    prediction = model.regressor(combined)
    
    # 整理结果
    pred_score = prediction.item()
    true_score = y_true.item()
    # 把权重转回 numpy 数组: [0.1, 0.5, ...]
    attn_values = attn_weights.squeeze().detach().cpu().numpy()
    
    # 把数字索引转回单词 (比如 [1, 16, 16...] -> "apple")
    # char_map 是 idx -> char 的字典
    word_indices = x_char.squeeze().cpu().numpy()
    word_str = "".join([char_map.get(idx, '?') for idx in word_indices])
    
    return word_str, pred_score, true_score, attn_values

# ==========================================
# 3. 主可视化流程
# ==========================================
def run_analysis():
    print("--> 正在加载模型与测试集...")
    model, device = load_trained_model()
    _, test_loader = get_dataloaders(batch_size=1) # 这里的 Batch=1 方便逐个分析
    
    # 构建反向查表字典 (idx -> char)
    # 这里简单重建一下，正常应该从 preprocessor 导入
    chars = sorted(list("abcdefghijklmnopqrstuvwxyz"))
    idx_to_char = {i+1: c for i, c in enumerate(chars)}
    idx_to_char[0] = '_' # Padding
    
    results = []
    
    print("--> 正在分析测试集样本...")
    # 随机挑选 5 个样本进行深度分析
    count = 0
    target_samples = 5
    
    plt.figure(figsize=(12, 6))
    
    for (x_char, x_sem, x_fb), y in test_loader:
        word, pred, true, attn = analyze_attention(model, device, (x_char[0], x_sem[0], x_fb[0], y[0]), idx_to_char)
        
        # 计算误差
        error = abs(pred - true)
        
        # 挑选：如果你想看特定的词，可以在这里加 if word == 'slate':
        # 这里我们简单地挑选前 5 个
        if count < target_samples:
            ax = plt.subplot(1, target_samples, count+1)
            
            # 画热力图 (Heatmap) 
            # 颜色越深，代表模型觉得这个字母越重要
            sns.heatmap(attn.reshape(-1, 1), annot=True, cmap='Reds', cbar=False, 
                        xticklabels=[], yticklabels=list(word), ax=ax)
            
            ax.set_title(f"{word}\n真:{true:.2f} 测:{pred:.2f}")
            count += 1
        
        results.append({'Word': word, 'True': true, 'Pred': pred, 'Error': error})

    plt.tight_layout()
    plt.savefig('attention_analysis.png')
    print(f"--> Attention 热力图已保存为 'attention_analysis.png'")
    
    # --- 整体评估 ---
    df_res = pd.DataFrame(results)
    mse = np.mean((df_res['True'] - df_res['Pred'])**2)
    rmse = np.sqrt(mse)
    
    print("\n=== 模型最终成绩单 ===")
    print(f"测试集 RMSE: {rmse:.4f}")
    print(f"预测最准确的词:\n{df_res.sort_values('Error').head(3)}")
    print(f"预测偏差最大的词:\n{df_res.sort_values('Error').tail(3)}")
    
    # 保存预测结果表
    df_res.to_csv('final_predictions.csv', index=False)
    print("--> 详细预测结果已保存为 'final_predictions.csv'")

if __name__ == "__main__":
    run_analysis()