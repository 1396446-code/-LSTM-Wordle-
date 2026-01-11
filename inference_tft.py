import torch
import numpy as np
import pandas as pd
import os
from train_tft import TimeSeriesTransformer # 引用你刚才定义的模型类

# ==========================================
# 配置与路径
# ==========================================
def get_file_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference():
    print(f"--> 正在加载时序模型与数据...")
    
    # 1. 加载数据
    X = np.load(get_file_path("timeseries_X.npy")).astype(np.float32)
    y = np.load(get_file_path("timeseries_y.npy")).astype(np.float32)
    
    # 加载原始CSV为了获取对应的日期 (注意：时序数据比原始数据少 Window_Size 天)
    df_raw = pd.read_csv(get_file_path("wordle_preprocessed_final.csv"))
    # 时序预测是从第 8 天开始的 (因为需要前 7 天做窗口)
    # 所以日期的切片是 [7:]
    dates = df_raw['Date'].values[7:]
    
    # 确保长度对齐 (以防万一数据处理时丢了一些尾部)
    min_len = min(len(dates), len(X))
    X = X[:min_len]
    y = y[:min_len]
    dates = dates[:min_len]

    # 2. 加载模型
    model = TimeSeriesTransformer().to(DEVICE)
    model_path = get_file_path('timeseries_transformer.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}，请先运行 train_tft.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. 批量预测
    print("--> 开始全量推理...")
    predictions = []
    
    # 为了显存安全，建议分批次预测，但这里数据量小，直接一次性推也行
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(DEVICE)
        # 模型输出 shape: [Batch]
        preds_tensor = model(X_tensor)
        predictions = preds_tensor.cpu().numpy()
        
    # 4. 保存结果
    results_df = pd.DataFrame({
        'Date': dates,
        'True_Score': y,
        'TFT_Prediction': predictions
    })
    
    save_path = get_file_path("tft_predictions.csv")
    results_df.to_csv(save_path, index=False)
    
    # 计算一下 RMSE
    mse = np.mean((y - predictions)**2)
    rmse = np.sqrt(mse)
    
    print(f"--> 推理完成!")
    print(f"    全量数据 RMSE: {rmse:.4f}")
    print(f"    结果已保存为: {save_path}")
    
    # --- 附加功能：预测未来 (Next Step Prediction) ---
    # 取最后一个窗口的数据，预测“明天”
    last_window = X[-1].reshape(1, 7, 101)
    with torch.no_grad():
        future_pred = model(torch.tensor(last_window).to(DEVICE)).item()
    
    print(f"\n🔮 [未来预测] 基于最后 7 天的数据，预测下一天的 Wordle 难度为: {future_pred:.2f}")

if __name__ == "__main__":
    run_inference()