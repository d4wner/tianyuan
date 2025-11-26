import pandas as pd
import os

# 读取最新的CSV文件
exports_dir = 'outputs/exports'
files = [f for f in os.listdir(exports_dir) if f.startswith('sh512660_daily') and f.endswith('.csv')]
files.sort(reverse=True)

if not files:
    print("未找到导出的CSV文件")
    exit(1)

latest_file = files[0]
file_path = os.path.join(exports_dir, latest_file)
print(f"正在分析最新文件: {file_path}")

# 读取CSV文件
df = pd.read_csv(file_path)

# 检查日期列格式
print(f"日期列类型: {df['date'].dtype}")

# 转换日期列为datetime类型
df['date'] = pd.to_datetime(df['date'])

# 检查11月24-25日的数据
print("\n11月24-25日的数据:")
for date_str in ['2025-11-24', '2025-11-25']:
    mask = df['date'].dt.strftime('%Y-%m-%d') == date_str
    if mask.any():
        row = df[mask].iloc[0]
        print(f"\n{date_str}:")
        print(f"  底分型: {row['bottom_fractal']}")
        print(f"  背驰: {row['divergence']}")
        print(f"  MACD: {row['macd']}")
        print(f"  MACD信号: {row['macd_signal']}")
        print(f"  MACD柱状图: {row['macd_hist']}")
        print(f"  信号: {row['signal']}")
        print(f"  信号强度: {row['signal_strength']}")
    else:
        print(f"\n未找到{date_str}的数据")

# 检查手动标记逻辑
print("\n测试手动标记逻辑:")
for idx, row in df.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    if date_str in ['2025-11-24', '2025-11-25']:
        print(f"找到日期 {date_str} 在索引 {idx}")
        # 模拟手动标记
        df.loc[idx, 'bottom_fractal'] = True
        df.loc[idx, 'fractal_price'] = row['low']
        df.loc[idx, 'divergence'] = 'bull'
        df.loc[idx, 'divergence_indicator'] = 'macd'
        df.loc[idx, 'divergence_strength'] = 1.0
        df.loc[idx, 'divergence_count'] = 1
        print(f"  已标记为底分型和背驰")

# 验证标记结果
print("\n标记后的结果:")
for date_str in ['2025-11-24', '2025-11-25']:
    mask = df['date'].dt.strftime('%Y-%m-%d') == date_str
    if mask.any():
        row = df[mask].iloc[0]
        print(f"\n{date_str}:")
        print(f"  底分型: {row['bottom_fractal']}")
        print(f"  背驰: {row['divergence']}")