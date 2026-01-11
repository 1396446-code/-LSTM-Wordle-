import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math

# ==========================================
# 1. 路径与配置
# ==========================================
def get_file_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 数据加载器 (Dataset)
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self):
        # 加载刚才生成的时序数据
        self.X = np.load(get_file_path("timeseries_X.npy")).astype(np.float32)
        self.y = np.load(get_file_path("timeseries_y.npy")).astype(np.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ==========================================
# 3. 模型定义: Time Series Transformer
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=101, d_model=64, nhead=4, num_layers=2):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. 特征投影 (把 101 维 -> 64 维)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 输出层
        # 我们取最后一个时间步的输出进行预测
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, src):
        # src shape: [Batch, 7, 101]
        
        # 投影: [Batch, 7, 64]
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        
        # 编码: [Batch, 7, 64]
        output = self.transformer_encoder(x)
        
        # 取最后一个时间步 (代表基于最新信息的总结): [Batch, 64]
        last_step_output = output[:, -1, :]
        
        # 预测: [Batch, 1]
        prediction = self.decoder(last_step_output)
        return prediction.squeeze(1)

# ==========================================
# 4. 训练主流程
# ==========================================
def train():
    print(f"--> 使用设备: {DEVICE}")
    
    # 准备数据
    dataset = TimeSeriesDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # 初始化模型
    model = TimeSeriesTransformer().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("--> 开始训练时序模型...")
    EPOCHS = 50
    train_losses = []
    test_losses = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
                
        test_loss = np.mean(val_losses)
        test_losses.append(test_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            
    print(f"训练完成! 耗时: {time.time()-start_time:.2f}s")
    
    # 保存模型
    torch.save(model.state_dict(), get_file_path('timeseries_transformer.pth'))
    
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Time Series Transformer Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(get_file_path('ts_training_curve.png'))
    print("--> 训练曲线已保存为 ts_training_curve.png")

if __name__ == "__main__":
    train()