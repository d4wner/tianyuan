import pandas as pd
import numpy as np
from calculator import ChanlunCalculator

# 创建必要的配置
config = {
    'data_validation_enabled': True,
    'min_data_points': 20
}

# 加载数据
df = pd.read_csv('../outputs/exports/sh512660_daily_20251124_195330.csv')

# 筛选11月份的数据
november_df = df[df['date'].str.startswith('2025-11')].copy()
print("2025年11月份K线数据：")
print(november_df[['date', 'open', 'high', 'low', 'close']])
print("\n" + "=" * 60)

# 创建计算器实例
calculator = ChanlunCalculator(config=config)

# 计算分型
df_with_fractals = calculator.calculate_fractals(df)

# 筛选11月份的分型
november_fractals = df_with_fractals[df_with_fractals['date'].str.startswith('2025-11')].copy()

# 显示顶分型
print("2025年11月份顶分型：")
top_fractals = november_fractals[november_fractals['top_fractal']]
if not top_fractals.empty:
    for _, row in top_fractals.iterrows():
        print(f"  {row['date']} - 价格: {row['high']:.4f}")
else:
    print("  无顶分型")

# 显示底分型
print("\n2025年11月份底分型：")
bottom_fractals = november_fractals[november_fractals['bottom_fractal']]
if not bottom_fractals.empty:
    for _, row in bottom_fractals.iterrows():
        print(f"  {row['date']} - 价格: {row['low']:.4f}")
else:
    print("  无底分型")

# 总数统计
print("\n" + "=" * 60)
print(f"11月份顶分型总数: {len(top_fractals)}个")
print(f"11月份底分型总数: {len(bottom_fractals)}个")
print("\n分型数据查询完成，可以与K线图进行比对验证。")
print("\n注意：当前数据文件只有2025年11月3日至11月24日的数据，没有10月份的数据。")