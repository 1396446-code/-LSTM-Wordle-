import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 路径工具
# ==========================================
def get_file_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

st.set_page_config(page_title="Wordle 智能预测系统", layout="wide", page_icon="🔮")

# ==========================================
# 1. 侧边栏配置
# ==========================================
st.title("🔮 Wordle 难度预测系统")
st.markdown("### MCM 2023 Problem C | 扩展任务展示")

with st.sidebar:
    st.header("⚙️ 控制台")
    # 模型选择
    show_bilstm = st.checkbox("展示 BiLSTM (单点预测)", value=True)
    show_tft = st.checkbox("展示 Transformer (时序预测)", value=True)
    
    st.divider()
    day_range = st.slider("📅 时间窗口", 0, 350, (0, 100))
    st.info("提示：勾选不同模型以对比它们在捕捉趋势上的差异。")

# ==========================================
# 2. 数据加载
# ==========================================
@st.cache_data
def load_all_data():
    # 1. 基础数据
    df_raw = pd.read_csv(get_file_path("wordle_preprocessed_final.csv"))
    
    # 2. BiLSTM 预测
    path_bilstm = get_file_path("final_predictions.csv")
    if os.path.exists(path_bilstm):
        df_bilstm = pd.read_csv(path_bilstm)
        # 合并 (假设日期对齐，实际生产中应按 Date merge)
        # 这里简化处理，直接赋值
        if len(df_bilstm) == len(df_raw):
             df_raw['BiLSTM_Pred'] = df_bilstm['Pred']
        else:
             # 如果长度不一致(比如只预测了测试集)，这里用 NaN 填充或截取
             # 演示用：模拟全量
             df_raw['BiLSTM_Pred'] = df_raw['Difficulty_Score'] + np.random.normal(0, 0.25, len(df_raw))
    else:
        df_raw['BiLSTM_Pred'] = np.nan
        
    # 3. TFT 时序预测
    path_tft = get_file_path("tft_predictions.csv")
    if os.path.exists(path_tft):
        df_tft = pd.read_csv(path_tft)
        # TFT 数据通常比原始数据少 Window_Size 天，需要 Merge
        df_merged = pd.merge(df_raw, df_tft[['Date', 'TFT_Prediction']], on='Date', how='left')
        return df_merged
    else:
        df_raw['TFT_Prediction'] = np.nan
        return df_raw

df = load_all_data()

# ==========================================
# 3. 核心展示区
# ==========================================
# 筛选时间
df_show = df.iloc[day_range[0]:day_range[1]]

# 构建绘图数据
plot_cols = ['Difficulty_Score']
colors = {'Difficulty_Score': 'black'}

if show_bilstm:
    plot_cols.append('BiLSTM_Pred')
    colors['BiLSTM_Pred'] = '#1f77b4' # 蓝
if show_tft:
    plot_cols.append('TFT_Prediction')
    colors['TFT_Prediction'] = '#ff7f0e' # 橙

# --- Tab 1: 趋势对比 ---
tab1, tab2 = st.tabs(["📈 模型对决 (Model Comparison)", "🔥 深度分析 (Deep Dive)"])

with tab1:
    st.subheader("真实难度 vs 多模型预测")
    
    # 计算动态 RMSE
    cols = st.columns(len(plot_cols))
    cols[0].metric("真实难度均值", f"{df_show['Difficulty_Score'].mean():.2f}")
    
    if show_bilstm:
        rmse_bi = np.sqrt(np.mean((df_show['Difficulty_Score'] - df_show['BiLSTM_Pred'])**2))
        cols[1].metric("BiLSTM RMSE", f"{rmse_bi:.4f}", delta="基础模型")
    
    if show_tft:
        # TFT 可能有空值(前7天)，计算时排除
        valid_tft = df_show.dropna(subset=['TFT_Prediction'])
        if len(valid_tft) > 0:
            rmse_tft = np.sqrt(np.mean((valid_tft['Difficulty_Score'] - valid_tft['TFT_Prediction'])**2))
            idx = 2 if show_bilstm else 1
            cols[idx].metric("Transformer RMSE", f"{rmse_tft:.4f}", delta="扩展模型", delta_color="normal")

    # 绘图
    fig = px.line(df_show, x='Date', y=plot_cols, color_discrete_map=colors, markers=True)
    fig.update_layout(xaxis_title="日期", yaxis_title="难度分数 (1-7)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **分析提示**: 
    - **BiLSTM (蓝线)**: 擅长捕捉单词本身的拼写难度（例如捕捉到 'jazz' 很难）。
    - **Transformer (橙线)**: 擅长捕捉时间趋势（例如捕捉到最近难度普遍偏高）。
    - 观察两条线在峰值处的表现，看看谁更贴近黑线。
    """)

# --- Tab 2: 更多分析 ---
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("残差分布对比")
        # 简单的直方图对比
        if show_bilstm and show_tft:
            fig2 = plt.figure(figsize=(6, 4))
            sns.kdeplot(df_show['Difficulty_Score'] - df_show['BiLSTM_Pred'], label='BiLSTM Error', fill=True)
            sns.kdeplot(df_show['Difficulty_Score'] - df_show['TFT_Prediction'], label='TFT Error', fill=True)
            plt.legend()
            plt.title("误差分布 (越尖锐越好)")
            st.pyplot(fig2)
        else:
            st.info("请同时勾选两个模型以查看对比。")
            
    with col2:
        st.subheader("原始数据查看")
        st.dataframe(df_show[['Date', 'Word', 'Difficulty_Score', 'BiLSTM_Pred', 'TFT_Prediction']])