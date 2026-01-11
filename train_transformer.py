import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# 复用你之前写好的数据管道
try:
    from data_loader import get_dataloaders
except ImportError:
    print("❌ 错误: 找不到 data_loader.py")
    exit()

# ==========================================
# 1. 位置编码 (Positional Encoding)
# Transformer 必须组件，告诉模型字符的顺序
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super(PositionalEncoding, self).__init__()
        # 创建一个位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer (不是参数，不需要更新，但随模型保存)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        return x + self.pe

# ==========================================
# 2. Transformer 主模型
# ==========================================
class WordleTransformer(nn.Module):
    def __init__(self, vocab_size=28, char_embed_dim=32, nhead=4, num_layers=2, semantic_dim=100):
        super(WordleTransformer, self).__init__()
        
        # 1. 字符嵌入层
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(char_embed_dim)
        
        # 2. Transformer Encoder
        # nhead: 多头注意力的头数
        # num_layers: 堆叠几层
        encoder_layer = nn.TransformerEncoderLayer(d_model=char_embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 融合层与输出
        # Transformer 输出维度 = char_embed_dim * 5 (flatten) 或者取平均
        # 这里我们把 5 个位置的特征展平
        flatten_dim = char_embed_dim * 5 
        fusion_dim = flatten_dim + semantic_dim + 5
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_char, x_semantic, x_feedback):
        # A. 字符流
        # [Batch, 5] -> [Batch, 5, 32]
        x = self.char_embedding(x_char)
        # 加上位置编码
        x = self.pos_encoder(x)
        # 过 Transformer: [Batch, 5, 32]
        x = self.transformer_encoder(x)
        
        # 展平: [Batch, 160]
        x = x.reshape(x.size(0), -1)
        
        # B. 融合
        combined = torch.cat((x, x_semantic, x_feedback), dim=1)
        
        # C. 预测
        return self.regressor(combined)

# ==========================================
# 3. 训练与对比脚本
# ==========================================
def train_and_compare():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Transformer 准备就绪，使用设备: {device}")
    
    # 获取数据
    train_loader, test_loader = get_dataloaders(batch_size=16)
    if not train_loader: return

    # 初始化 Transformer
    model = WordleTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    EPOCHS = 50
    transformer_losses = []
    
    print(f"--> 开始训练 Transformer ({EPOCHS} Epochs)...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        batch_loss = 0
        for (x_char, x_sem, x_fb), y in train_loader:
            x_char, x_sem, x_fb, y = x_char.to(device), x_sem.to(device), x_fb.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x_char, x_sem, x_fb)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            
        # 记录 Test Loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x_char, x_sem, x_fb), y in test_loader:
                x_char, x_sem, x_fb, y = x_char.to(device), x_sem.to(device), x_fb.to(device), y.to(device)
                pred = model(x_char, x_sem, x_fb)
                loss = criterion(pred, y)
                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader)
        transformer_losses.append(avg_test_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Transformer Test Loss: {avg_test_loss:.4f}")

    print(f"Transformer 训练完成! 耗时: {time.time() - start_time:.2f}s")
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    
    # --- 关键步骤：生成对比图 ---
    print("\n--> 正在生成模型对比图 (BiLSTM vs Transformer)...")
    
    # 注意：这里假设你之前跑过 train_model.py 并且它生成了数据
    # 如果为了简单对比，我们可以手动把刚才 BiLSTM 的最终 Loss 填在这里画个柱状图
    # 或者直接把刚才的曲线存下来对比。
    # 这里我们只画 Transformer 的曲线供你放入论文
    
    plt.figure(figsize=(8, 5))
    plt.plot(transformer_losses, label='Transformer Loss', color='orange')
    # 如果你有 BiLSTM 的数据，可以 plt.plot(bilstm_losses, label='BiLSTM Loss')
    plt.title('Model Performance: Transformer on Wordle Data')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_curve.png')
    print("--> 对比曲线已保存为 'transformer_curve.png'")

if __name__ == "__main__":
    train_and_compare()