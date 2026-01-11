import pandas as pd
import os
import glob

def auto_locate_and_clean():
    print("="*40)
    print("【修正版：数据清洗程序】")
    
    # --- 1. 定位文件 ---
    # 获取脚本所在目录
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    os.chdir(script_dir)
    
    # 查找 Excel 文件
    files = glob.glob("*.xlsx") + glob.glob("*.csv")
    input_file = None
    # 优先找名字匹配的
    for f in files:
        if 'Problem_C' in f or 'Data' in f:
            input_file = f
            break
    
    if not input_file:
        if files: input_file = files[0]
        else:
            print("错误：未找到Excel文件！")
            return

    print(f"--> 使用文件: {input_file}")

    # --- 2. 读取数据 ---
    try:
        # 智能寻找表头
        if input_file.endswith('.csv'):
            df_raw = pd.read_csv(input_file, header=None, nrows=10)
        else:
            df_raw = pd.read_excel(input_file, header=None, nrows=10)
            
        header_row = 1 # 默认第2行
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower().tolist()
            if any('word' in str(x) for x in row_str) and any('date' in str(x) for x in row_str):
                header_row = i
                break
        
        print(f"--> 表头所在行: {header_row + 1}")
        
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file, header=header_row)
        else:
            df = pd.read_excel(input_file, header=header_row)

        # --- 3. 强制列名修复 (关键步骤) ---
        # 去除列名空格
        df.columns = [str(c).strip() for c in df.columns]
        
        # 建立严格的映射字典，解决列名不匹配问题
        rename_map = {
            'Contest number': 'Contest_Number',
            'Number of  reported results': 'N_Results', # 注意原文件可能有双空格
            'Number of reported results': 'N_Results',
            'Number in hard mode': 'N_Hard_Mode',
            '7 or more tries (X)': '7_tries_plus'      # <--- 修复你的报错点
        }
        df.rename(columns=rename_map, inplace=True)
        
        print(f"--> 列名标准化完成。当前列: {df.columns.tolist()}")

        # --- 4. 清洗逻辑 ---
        # 去空值
        df_clean = df.dropna(subset=['Word', 'Date']).copy()
        
        # 单词处理
        df_clean['Word'] = df_clean['Word'].astype(str).str.strip().str.lower()
        df_clean = df_clean[df_clean['Word'].apply(len) == 5]
        
        # 日期处理
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean.sort_values('Date', inplace=True)

        # 数值列转换
        cols_percent = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7_tries_plus']
        
        # 检查是否所有列都齐了
        missing_cols = [c for c in cols_percent if c not in df_clean.columns]
        if missing_cols:
            print(f"警告: 缺少以下列，无法计算平均分: {missing_cols}")
            # 如果真的缺列，为了不报错，填充为0
            for c in missing_cols:
                df_clean[c] = 0
        
        for col in cols_percent:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

        # --- 5. 计算特征 (Average_Score) ---
        # 只有列名对齐了，这里才能算
        print("--> 正在计算 Average_Score ...")
        weighted_sum = (
            df_clean['1 try'] * 1 +
            df_clean['2 tries'] * 2 +
            df_clean['3 tries'] * 3 +
            df_clean['4 tries'] * 4 +
            df_clean['5 tries'] * 5 +
            df_clean['6 tries'] * 6 +
            df_clean['7_tries_plus'] * 7
        )
        df_clean['Average_Score'] = weighted_sum / 100.0

        # --- 6. 保存 ---
        output_file = 'cleaned_wordle_data.csv'
        df_clean.to_csv(output_file, index=False)
        
        print("\n" + "="*40)
        print(f"SUCCESS! 全部完成！")
        print(f"文件已保存: {output_file}")
        print("数据预览:")
        # 这次肯定不会报错了
        print(df_clean[['Date', 'Word', 'Average_Score']].head())
        print("="*40)

    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    auto_locate_and_clean()