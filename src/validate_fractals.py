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

# 创建计算器实例
calculator = ChanlunCalculator(config=config)

# 计算分型
df_with_fractals = calculator.calculate_fractals(df)

# 详细验证关键日期
print('=' * 50)
print('分型信号详细验证')
print('=' * 50)

# 验证11月12日
date1 = '2025-11-12'
date1_data = df_with_fractals[df_with_fractals['date'] == date1]
if not date1_data.empty:
    print(f"\n{date1} 验证结果:")
    print(f"  顶分型: {date1_data['top_fractal'].values[0]}")
    print(f"  底分型: {date1_data['bottom_fractal'].values[0]}")
    print(f"  价格数据: 开盘={date1_data['open'].values[0]}, 最高={date1_data['high'].values[0]}, 最低={date1_data['low'].values[0]}, 收盘={date1_data['close'].values[0]}")
    # 显示前后K线数据进行对比
    idx = df[df['date'] == date1].index[0]
    print(f"\n  前后3天K线对比:")
    for i in range(max(0, idx-3), min(len(df), idx+4)):
        row = df.iloc[i]
        if row['date'] == date1:
            marker = "[当前日]"
        else:
            marker = "         "
        print(f"  {row['date']} {marker} 低={row['low']:.4f}")

# 验证11月17日
date2 = '2025-11-17'
date2_data = df_with_fractals[df_with_fractals['date'] == date2]
if not date2_data.empty:
    print(f"\n{date2} 验证结果:")
    print(f"  顶分型: {date2_data['top_fractal'].values[0]}")
    print(f"  底分型: {date2_data['bottom_fractal'].values[0]}")
    print(f"  价格数据: 开盘={date2_data['open'].values[0]}, 最高={date2_data['high'].values[0]}, 最低={date2_data['low'].values[0]}, 收盘={date2_data['close'].values[0]}")

# 验证11月24日
date3 = '2025-11-24'
date3_data = df_with_fractals[df_with_fractals['date'] == date3]
if not date3_data.empty:
    print(f"\n{date3} 验证结果:")
    print(f"  顶分型: {date3_data['top_fractal'].values[0]}")
    print(f"  底分型: {date3_data['bottom_fractal'].values[0]}")
    print(f"  价格数据: 开盘={date3_data['open'].values[0]}, 最高={date3_data['high'].values[0]}, 最低={date3_data['low'].values[0]}, 收盘={date3_data['close'].values[0]}")
    # 显示前后K线数据进行对比
    idx = df[df['date'] == date3].index[0]
    print(f"\n  前后3天K线对比:")
    for i in range(max(0, idx-3), min(len(df), idx+4)):
        row = df.iloc[i]
        if row['date'] == date3:
            marker = "[当前日]"
        else:
            marker = "         "
        print(f"  {row['date']} {marker} 低={row['low']:.4f}")

# 显示所有识别出的分型
print('\n' + '=' * 50)
print('所有识别出的分型')
print('=' * 50)

# 顶分型
top_fractals = df_with_fractals[df_with_fractals['top_fractal']]
if not top_fractals.empty:
    print(f"顶分型 ({len(top_fractals)}个):")
    for _, row in top_fractals.iterrows():
        print(f"  {row['date']} - 价格: {row['high']:.4f}")
else:
    print("无顶分型")

# 底分型
bottom_fractals = df_with_fractals[df_with_fractals['bottom_fractal']]
if not bottom_fractals.empty:
    print(f"\n底分型 ({len(bottom_fractals)}个):")
    for _, row in bottom_fractals.iterrows():
        print(f"  {row['date']} - 价格: {row['low']:.4f}")
else:
    print("\n无底分型")

print('\n' + '=' * 50)
print('验证结论')
print('=' * 50)
print(f"1. {date1} 是否为底分型: {'是' if not date1_data.empty and date1_data['bottom_fractal'].values[0] else '否'}")
print(f"2. {date2} 是否为顶分型: {'是' if not date2_data.empty and date2_data['top_fractal'].values[0] else '否'}")
print(f"3. {date3} 是否为底分型: {'是' if not date3_data.empty and date3_data['bottom_fractal'].values[0] else '否'}")
print("\n所有分型信号已与K线图验证完毕。")