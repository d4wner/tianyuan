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
print("2025年11月份K线数据完整列表：")
print(df[['date', 'open', 'high', 'low', 'close']].to_string(index=True))
print("\n" + "=" * 80)

# 严格按照用户定义的分型逻辑进行判断
def is_top_fractal(df, i):
    """
    顶分型判断：中间K线的最高价是三根K线的最高点，并且右侧K线收盘价低于左侧K线收盘价
    顶分型形成日是右侧K线，即第三根K线
    """
    if i < 2 or i >= len(df):
        return False
    
    # 三根K线的条件：中间K线最高价最高
    left_high = df.iloc[i-2]['high']
    middle_high = df.iloc[i-1]['high']
    right_high = df.iloc[i]['high']
    
    # 顶分型条件1：中间K线最高价是三根中的最高
    if not (middle_high > left_high and middle_high > right_high):
        return False
    
    # 顶分型条件2：右侧K线收盘价低于左侧K线收盘价
    left_close = df.iloc[i-2]['close']
    right_close = df.iloc[i]['close']
    
    # 补充验证：11月6日的特殊检查
    if df.iloc[i-1]['date'] == '2025-11-06':
        # 检查11月6日是否高于11月4日高点
        if i-2 >= 0 and df.iloc[i-2]['date'] == '2025-11-04':
            print(f"\n11月6日顶分型验证: 11月4日高点={left_high}, 11月6日高点={middle_high}")
            if middle_high <= left_high:
                print("  11月6日不满足顶分型条件：未高于前一交易日高点")
                return False
    
    return right_close < left_close

def is_bottom_fractal(df, i):
    """
    底分型判断：中间K线的最低价是三根K线的最低点，并且右侧K线收盘价高于左侧K线收盘价
    底分型形成日是右侧K线，即第三根K线
    """
    if i < 2 or i >= len(df):
        return False
    
    # 三根K线的条件：中间K线最低价最低
    left_low = df.iloc[i-2]['low']
    middle_low = df.iloc[i-1]['low']
    right_low = df.iloc[i]['low']
    
    # 底分型条件1：中间K线最低价是三根中的最低
    if not (middle_low < left_low and middle_low < right_low):
        return False
    
    # 底分型条件2：右侧K线收盘价高于左侧K线收盘价
    left_close = df.iloc[i-2]['close']
    right_close = df.iloc[i]['close']
    
    # 补充验证：11月24日作为底分型形成日的特殊处理
    if df.iloc[i]['date'] == '2025-11-24':
        # 11月24日是底分型形成日，对应的中间K线是11月21日
        if i-1 >= 0 and df.iloc[i-1]['date'] == '2025-11-21':
            print(f"\n11月24日底分型验证: 中间K线(11月21日)低点={middle_low}, 右侧收盘价={right_close}, 左侧收盘价={left_close}")
            # 验证右侧收盘价是否高于左侧
            if right_close > left_close:
                print("  11月24日满足底分型形成日条件")
            else:
                print("  11月24日不满足底分型形成日条件")
    
    return right_close > left_close

# 创建顶底分型结果列表
top_fractals = []
bottom_fractals = []

print("\n严格按照分型定义进行判断...")

# 遍历数据进行判断
for i in range(len(df)):
    date = df.iloc[i]['date']
    
    # 判断顶分型（i是右侧K线，即形成日）
    if is_top_fractal(df, i):
        middle_idx = i-1
        top_fractal_info = {
            'date': date,  # 顶分型形成日
            'high': df.iloc[middle_idx]['high'],  # 中间K线的最高价
            'middle_date': df.iloc[middle_idx]['date'],  # 中间K线日期
            'left_date': df.iloc[i-2]['date'],  # 左侧K线日期
            'right_date': date  # 右侧K线日期（形成日）
        }
        top_fractals.append(top_fractal_info)
    
    # 判断底分型（i是右侧K线，即形成日）
    if is_bottom_fractal(df, i):
        middle_idx = i-1
        bottom_fractal_info = {
            'date': date,  # 底分型形成日
            'low': df.iloc[middle_idx]['low'],  # 中间K线的最低价
            'middle_date': df.iloc[middle_idx]['date'],  # 中间K线日期
            'left_date': df.iloc[i-2]['date'],  # 左侧K线日期
            'right_date': date  # 右侧K线日期（形成日）
        }
        bottom_fractals.append(bottom_fractal_info)

# 显示顶分型结果
print("\n" + "=" * 80)
print("2025年11月份顶分型（形成日）：")
if top_fractals:
    for fractal in top_fractals:
        if fractal['date'].startswith('2025-11'):
            print(f"  形成日: {fractal['date']} | 顶部价格: {fractal['high']:.4f} | 中间K线: {fractal['middle_date']} | 组成: [{fractal['left_date']}, {fractal['middle_date']}, {fractal['right_date']}]")
else:
    print("  无顶分型")

# 显示底分型结果
print("\n2025年11月份底分型（形成日）：")
if bottom_fractals:
    for fractal in bottom_fractals:
        if fractal['date'].startswith('2025-11'):
            print(f"  形成日: {fractal['date']} | 底部价格: {fractal['low']:.4f} | 中间K线: {fractal['middle_date']} | 组成: [{fractal['left_date']}, {fractal['middle_date']}, {fractal['right_date']}]")
else:
    print("  无底分型")

# 特别强调11月24日的底分型状态
print("\n" + "=" * 80)
print("重要结论：")
bottom_24_exists = any(f['date'] == '2025-11-24' for f in bottom_fractals)
if bottom_24_exists:
    print("✓ 11月24日被正确识别为底分型形成日")
else:
    print("✗ 11月24日未被识别为底分型形成日")

top_6_exists = any(f['date'] in ['2025-11-07', '2025-11-06'] for f in top_fractals)
if top_6_exists:
    print("⚠️  11月6日相关顶分型存在")
else:
    print("✓ 11月6日相关顶分型不存在（符合严格判断）")

# 总数统计
november_top_count = sum(1 for f in top_fractals if f['date'].startswith('2025-11'))
november_bottom_count = sum(1 for f in bottom_fractals if f['date'].startswith('2025-11'))
print(f"\n11月份顶分型形成日总数: {november_top_count}个")
print(f"11月份底分型形成日总数: {november_bottom_count}个")
print("\n分型分析完成，严格按照用户提供的定义：顶底分型的形成日是最后一日，高低点是中间K线。")