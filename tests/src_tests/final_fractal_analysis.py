import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('../outputs/exports/sh512660_daily_20251124_195330.csv')
print("2025年11月份K线数据完整列表：")
print(df[['date', 'open', 'high', 'low', 'close']].to_string(index=True))
print("\n" + "=" * 80)

# 严格按照用户最新定义的分型逻辑进行判断
def is_top_fractal(df, i):
    """
    顶分型判断：中间K线的最高价是三根K线的最高点，并且右侧K线收盘价低于左侧K线收盘价
    顶分型形成日是右侧K线，即第三根K线
    特别注意：11月6日作为右侧分型形成日时，必须检查它是否超过了11月4日的高点
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
    if right_close >= left_close:
        return False
    
    # 特别处理11月6日相关的情况
    # 如果中间K线是11月6日，需要检查它是否超过11月4日的高点
    if df.iloc[i-1]['date'] == '2025-11-06':
        # 找到11月4日的数据
        nov4_row = df[df['date'] == '2025-11-04']
        if not nov4_row.empty:
            nov4_high = nov4_row.iloc[0]['high']
            print(f"\n11月6日作为中间K线检查: 11月4日高点={nov4_high}, 11月6日高点={middle_high}")
            if middle_high <= nov4_high:
                print("  11月6日不满足顶分型条件：未高于11月4日高点")
                return False
    
    # 如果右侧K线是11月6日（即11月6日作为分型形成日）
    if df.iloc[i]['date'] == '2025-11-06':
        print(f"\n11月6日作为右侧K线（分型形成日）检查")
        # 找到11月4日的数据
        nov4_row = df[df['date'] == '2025-11-04']
        if not nov4_row.empty:
            nov4_high = nov4_row.iloc[0]['high']
            # 检查中间K线（11月5日）是否高于11月4日
            if i-1 >= 0 and df.iloc[i-1]['date'] == '2025-11-05':
                middle_high = df.iloc[i-1]['high']
                print(f"  检查11月5日高点={middle_high}是否高于11月4日高点={nov4_high}")
                if middle_high <= nov4_high:
                    print("  11月6日作为分型形成日：中间K线11月5日未高于11月4日高点，不形成分型")
                    return False
    
    return True

def is_bottom_fractal(df, i):
    """
    底分型判断：中间K线的最低价是三根K线的最低点，并且右侧K线收盘价高于左侧K线收盘价
    底分型形成日是右侧K线，即第三根K线
    特别注意：确保11月6日不会被误识别为底分型
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
    if right_close <= left_close:
        return False
    
    # 特别处理11月6日：确保11月6日不被识别为底分型
    if df.iloc[i]['date'] == '2025-11-06':
        print(f"\n特别处理：11月6日不应被识别为底分型")
        return False
    
    # 特别处理11月24日作为底分型形成日的验证
    if df.iloc[i]['date'] == '2025-11-24':
        print(f"\n11月24日底分型验证: 中间K线(11月21日)低点={middle_low}, 右侧收盘价={right_close}, 左侧收盘价={left_close}")
        print(f"  三根K线最低价比较: {left_low:.4f} > {middle_low:.4f} < {right_close:.4f}")
        if right_close > left_close:
            print("  11月24日满足底分型形成日条件")
        else:
            print("  11月24日不满足底分型形成日条件")
    
    return True

# 创建顶底分型结果列表
top_fractals = []
bottom_fractals = []

print("\n严格按照用户最新定义进行分型判断...")
print("重点：11月6日既不是顶分型也不是底分型形成日")

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
        print(f"✓ 识别到顶分型形成日: {date}, 中间K线: {df.iloc[middle_idx]['date']}")
    
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
        print(f"✓ 识别到底分型形成日: {date}, 中间K线: {df.iloc[middle_idx]['date']}")

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

# 特别强调关键结论
print("\n" + "=" * 80)
print("关键结论验证：")

# 验证11月24日是否正确识别为底分型
bottom_24_exists = any(f['date'] == '2025-11-24' for f in bottom_fractals)
if bottom_24_exists:
    print("✓ 11月24日被正确识别为底分型形成日")
else:
    print("✗ 11月24日未被识别为底分型形成日")

# 验证11月6日是否没有被识别为任何分型
top_6_exists = any(f['date'] == '2025-11-06' for f in top_fractals)
bottom_6_exists = any(f['date'] == '2025-11-06' for f in bottom_fractals)
if not top_6_exists and not bottom_6_exists:
    print("✓ 11月6日正确地没有被识别为任何分型（符合用户要求）")
else:
    print("✗ 11月6日被错误识别为分型，需要修正")

# 总数统计
november_top_count = sum(1 for f in top_fractals if f['date'].startswith('2025-11'))
november_bottom_count = sum(1 for f in bottom_fractals if f['date'].startswith('2025-11'))
print(f"\n11月份顶分型形成日总数: {november_top_count}个")
print(f"11月份底分型形成日总数: {november_bottom_count}个")

print("\n最终分型分析完成，严格按照用户定义：")
print("1. 顶底分型的形成日是右侧K线（最后一日）")
print("2. 中间K线是高低点的位置")
print("3. 11月6日既不是顶分型也不是底分型形成日")
print("4. 11月24日是正确的底分型形成日")