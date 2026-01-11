import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

# 尝试导入数据加载器 (要求 data_loader.py 在同一文件夹)
try:
    from data_loader import get_dataloaders
except ImportError:
    print("❌ 错误: 找不到 data_loader.py！请确保它和本脚本在同一目录下。")
    exit()

# ==========================================
# 1. 定义注意力机制层 (Attention Layer)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 定义一个线性层来计算注意力分数
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [Batch, Seq_Len, Hidden_Dim]
        
        # 1. 计算分数: [Batch, Seq_Len, 1]
        attn_weights = self.attn(lstm_output)
        
        # 2. 归一化 (Softmax): [Batch, Seq_Len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 3. 加权求和 (Context Vector): [Batch, Hidden_Dim]
        # 也就是把 Seq_Len (5个字符) 的特征融合成 1 个向量
        context = torch.sum(lstm_output * attn_weights, dim=1)
        
        return context, attn_weights

# ==========================================
# 2. 定义主模型 (BiLSTM + Attention + Fusion)
# ==========================================
class WordlePredictor(nn.Module):
    def __init__(self, vocab_size=28, char_embed_dim=32, hidden_dim=64, semantic_dim=100):
        super(WordlePredictor, self).__init__()
        
        # --- 分支 A: 字符级 BiLSTM ---
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=char_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True # <--- 双向 LSTM
        )
        # 双向会导致输出维度翻倍，所以 Attention 输入维度是 hidden_dim * 2
        self.attention = Attention(hidden_dim * 2)
        
        # --- 分支 B: 语义融合层 ---
        # 这里的 semantic_dim 就是 Word2Vec 的 100 维
        
        # --- 分支 C: 反馈融合层 ---
        # 反馈向量是 5 维
        
        # --- 最终预测层 (Fully Connected) ---
        # 输入总维度 = LSTM上下文(128) + 语义向量(100) + 反馈向量(5)
        fusion_dim = (hidden_dim * 2) + semantic_dim + 5
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # 防止过拟合
            nn.Linear(64, 1) # 输出 1 个值 (难度分数)
        )

    def forward(self, x_char, x_semantic, x_feedback):
        # 1. 处理字符序列
        # x_char: [Batch, 5] -> [Batch, 5, 32]
        embedded = self.char_embedding(x_char)
        
        # BiLSTM 输出: [Batch, 5, 128]
        lstm_out, _ = self.lstm(embedded)
        
        # Attention 聚合: [Batch, 128]
        context_vector, attn_weights = self.attention(lstm_out)
        
        # 2. 特征融合 (Fusion)
        # 将 "拼写特征"、"语义特征"、"反馈特征" 拼接在一起
        combined = torch.cat((context_vector, x_semantic, x_feedback), dim=1)
        
        # 3. 最终预测
        prediction = self.regressor(combined)
        return prediction

# ==========================================
# 3. 训练循环 (Training Loop)
# ==========================================
def train_model():
    # --- 配置 ---
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> 使用设备: {device}")

    # 1. 获取数据
    print("--> 正在加载数据...")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    if not train_loader: return

    # 2. 初始化模型
    model = WordlePredictor().to(device)
    criterion = nn.MSELoss() # 回归问题用均方误差
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录 Loss 用于画图
    train_losses = []
    test_losses = []

    print(f"--> 开始训练 ({EPOCHS} Epochs)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        batch_loss = 0
        
        for (x_char, x_sem, x_fb), y in train_loader:
            # 搬运数据到 GPU/CPU
            x_char, x_sem, x_fb, y = x_char.to(device), x_sem.to(device), x_fb.to(device), y.to(device)
            
            optimizer.zero_grad() # 清空梯度
            pred = model(x_char, x_sem, x_fb) # 前向传播
            loss = criterion(pred, y) # 计算 Loss
            loss.backward() # 反向传播
            optimizer.step() # 更新权重
            
            batch_loss += loss.item()
        
        avg_train_loss = batch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证集测试
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x_char, x_sem, x_fb), y in test_loader:
                x_char, x_sem, x_fb, y = x_char.to(device), x_sem.to(device), x_fb.to(device), y.to(device)
                pred = model(x_char, x_sem, x_fb)
                loss = criterion(pred, y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    print(f"\n训练完成! 耗时: {time.time() - start_time:.2f}秒")
    
    # --- 保存模型 ---
    torch.save(model.state_dict(), 'bi_lstm_model.pth')
    print("--> 模型已保存为 'bi_lstm_model.pth'")

    # --- 可视化 Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Model Training Curve (MSE Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png') # 保存图片而不是显示
    print("--> 训练曲线已保存为 'training_curve.png'")

if __name__ == "__main__":
    train_model()