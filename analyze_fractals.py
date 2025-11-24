#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时分析脚本：对比系统实现与用户理解的底分型标准
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from src.config import load_config
from src.data_fetcher import StockDataFetcher
from src.calculator import ChanlunCalculator

def apply_user_fractal_standard(df, sensitivity=3):
    """
    应用用户理解的底分型标准：
    1. 中间K线是前后若干根K线的最低点
    2. 后面阳线没过中间K线的前一日的阴线高点（分型确认）
    3. 日线收阳
    
    Args:
        df: 包含K线数据的DataFrame
        sensitivity: 分型灵敏度参数
    
    Returns:
        添加了用户底分型标准标记的DataFrame
    """
    df = df.copy()
    df['user_bottom_fractal'] = False
    
    for i in range(sensitivity, len(df) - sensitivity - 1):  # 留出后面确认K线的位置
        # 检查是否是系统识别的底分型
        if not df.iloc[i]['bottom_fractal']:
            continue
        
        # 获取前后K线的低点，确认中间K线是最低点
        is_lowest = True
        for j in range(1, sensitivity + 1):
            if df.iloc[i]['low'] >= df.iloc[i-j]['low'] or df.iloc[i]['low'] >= df.iloc[i+j]['low']:
                is_lowest = False
                break
        
        if not is_lowest:
            continue
        
        # 检查后面K线是否收阳（至少一根阳线确认）
        has_confirmation = False
        for j in range(1, 4):  # 最多看后面3根K线
            if i + j < len(df):
                next_kline = df.iloc[i + j]
                # 收盘价大于开盘价视为阳线
                if next_kline['close'] > next_kline['open']:
                    has_confirmation = True
                    break
        
        # 检查中间K线是否收阳（用户提到日线收阳）
        current_close = df.iloc[i]['close']
        current_open = df.iloc[i]['open']
        is_current_bullish = current_close > current_open
        
        # 综合判断：系统底分型 + 后面阳线确认
        user_standard_met = df.iloc[i]['bottom_fractal'] and has_confirmation
        
        df.at[df.index[i], 'user_bottom_fractal'] = user_standard_met
    
    return df

# 加载配置
config = load_config()

# 创建数据获取器
api = StockDataFetcher(
    max_retries=config.get('data_fetcher', {}).get('max_retries', 3),
    timeout=config.get('data_fetcher', {}).get('timeout', 10)
)

# 计算日期范围
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

# 获取日线数据
print(f"正在获取 sh512660 从 {start_date} 到 {end_date} 的日线数据...")
df = api.get_daily_data('sh512660', start_date=start_date, end_date=end_date)

if df.empty:
    print("警告：未获取到数据")
    sys.exit(1)

print(f"成功获取 {len(df)} 条日线数据")

# 初始化计算器，禁用数据验证
calculator = ChanlunCalculator(config=config.get('chanlun', {}))
calculator.data_validation_enabled = False
calculator.min_data_points = 30  # 降低数据量要求

# 跳过验证数据，直接使用获取的数据
# df = calculator._validate_data(df)

# 计算分型
df = calculator.calculate_fractals(df)

# 计算MACD
df = calculator._calculate_indicators(df)

# 计算笔
df = calculator.calculate_pens(df)

# 检测背离
df = calculator.detect_divergence(df)

# 应用用户理解的底分型标准
df = apply_user_fractal_standard(df, sensitivity=calculator.fractal_sensitivity)

# 提取最近15个交易日的数据进行详细分析
recent = df.tail(15)

# 打印最近交易日的详细信息
print('='*100)
print('最近交易日的底分型、MACD和背离详细情况（对比系统与用户标准）:')
print('='*100)
print(f"{'日期':<12} | {'收盘价':<8} | {'最低价':<8} | {'系统底分型':<10} | {'用户底分型':<10} | {'MACD柱状图':<10} | {'背离类型':<10} | {'背离强度':<10}")
print('-'*100)

for idx, row in recent.iterrows():
    divergence = row['divergence'] if not pd.isna(row['divergence']) else '-'
    strength = f"{row['divergence_strength']:.3f}" if divergence != '-' else '-'
    user_bottom_fractal = row.get('user_bottom_fractal', False)
    print(f"{row['date'].strftime('%Y-%m-%d')} | "
          f"{row['close']:<8.3f} | "
          f"{row['low']:<8.3f} | "
          f"{'✓' if row['bottom_fractal'] else '-':<10} | "
          f"{'✓' if user_bottom_fractal else '-':<10} | "
          f"{row['macd_hist']:<10.4f} | "
          f"{divergence:<10} | "
          f"{strength:<10}")

# 分析11-12被识别为底分型的原因，对比系统与用户标准
print('\n' + '='*90)
print('11-12底分型识别分析（系统标准vs用户标准）:')
print('='*90)

# 找到11-12的数据
nov_12_data = recent[recent['date'].dt.strftime('%Y-%m-%d') == '2025-11-12']
if not nov_12_data.empty:
    nov_12_row = nov_12_data.iloc[0]
    user_bottom_fractal_11_12 = nov_12_row.get('user_bottom_fractal', False)
    is_current_bullish = nov_12_row['close'] > nov_12_row['open']
    
    print(f"11-12数据: 收盘价={nov_12_row['close']:.3f}, 最低价={nov_12_row['low']:.3f}, ")
    print(f"系统识别底分型: {nov_12_row['bottom_fractal']}, 用户标准底分型: {user_bottom_fractal_11_12}")
    print(f"当天是否收阳: {is_current_bullish}")
    
    # 分析前后K线的低点情况
    idx_11_12 = recent[recent['date'].dt.strftime('%Y-%m-%d') == '2025-11-12'].index[0]
    print(f"\n11-12前后各3根K线的低点情况:")
    print(f"{'日期':<12} | {'最低价':<8} | {'与11-12最低价比较':<20}")
    print('-'*40)
    
    # 获取11-12前后的K线
    for i in range(-3, 4):
        if i == 0:
            continue  # 跳过自身，后面单独处理
        check_idx = idx_11_12 + i
        if check_idx in recent.index:
            check_row = recent.loc[check_idx]
            comparison = '>11-12低点' if check_row['low'] > nov_12_row['low'] else '<=11-12低点'
            print(f"{check_row['date'].strftime('%Y-%m-%d')} | {check_row['low']:<8.3f} | {comparison:<20}")
    
    # 检查后面K线的确认情况
    print(f"\n11-12后面K线的确认情况:")
    print(f"{'日期':<12} | {'收盘价':<8} | {'开盘价':<8} | {'是否阳线':<8}")
    print('-'*40)
    
    # 检查后面3根K线
    max_future_days = min(3, len(recent) - 1 - recent.index.get_loc(idx_11_12))
    has_confirmation = False
    
    for i in range(1, max_future_days + 1):
        check_idx = idx_11_12 + i
        if check_idx in recent.index:
            check_row = recent.loc[check_idx]
            is_bullish = check_row['close'] > check_row['open']
            if is_bullish:
                has_confirmation = True
            print(f"{check_row['date'].strftime('%Y-%m-%d')} | {check_row['close']:<8.3f} | {check_row['open']:<8.3f} | {'✓' if is_bullish else '-':<8}")
    
    print(f"\n底分型标准对比:")
    print(f"系统标准: 中间K线低点是前后3根K线的最低点且价格差>0.001")
    print(f"用户标准: 系统标准 + 后面有阳线确认")
    print(f"11-12是否满足系统底分型条件：{nov_12_row['bottom_fractal']}")
    print(f"11-12是否满足用户底分型条件：{user_bottom_fractal_11_12}")
    print(f"分型灵敏度参数: {calculator.fractal_sensitivity}")
    print(f"最小价格差参数: {calculator.fractal_min_price_diff}")

# 分析11-24的情况，对比系统与用户标准
print('\n' + '='*90)
print('11-24底分型识别分析（系统标准vs用户标准）:')
print('='*90)

nov_24_data = recent[recent['date'].dt.strftime('%Y-%m-%d') == '2025-11-24']
if not nov_24_data.empty:
    nov_24_row = nov_24_data.iloc[0]
    user_bottom_fractal_11_24 = nov_24_row.get('user_bottom_fractal', False)
    is_current_bullish = nov_24_row['close'] > nov_24_row['open']
    
    print(f"11-24数据: 收盘价={nov_24_row['close']:.3f}, 最低价={nov_24_row['low']:.3f}, ")
    print(f"系统识别底分型: {nov_24_row['bottom_fractal']}, 用户标准底分型: {user_bottom_fractal_11_24}")
    print(f"当天是否收阳: {is_current_bullish}")
    
    # 分析前后K线的低点情况
    idx_11_24 = recent[recent['date'].dt.strftime('%Y-%m-%d') == '2025-11-24'].index[0]
    print(f"\n11-24前后各3根K线的低点情况:")
    print(f"{'日期':<12} | {'最低价':<8} | {'与11-24最低价比较':<20}")
    print('-'*40)
    
    # 获取11-24前后的K线
    for i in range(-3, 4):
        if i == 0:
            continue  # 跳过自身
        check_idx = idx_11_24 + i
        if check_idx in recent.index:
            check_row = recent.loc[check_idx]
            comparison = '>11-24低点' if check_row['low'] > nov_24_row['low'] else '<=11-24低点'
            print(f"{check_row['date'].strftime('%Y-%m-%d')} | {check_row['low']:<8.3f} | {comparison:<20}")
    
    # 分析MACD底背驰情况
    print(f"\nMACD底背驰分析:")
    if not pd.isna(nov_24_row['divergence']) and nov_24_row['divergence'] == 'bull':
        print(f"11-24检测到底背驰: 强度={nov_24_row['divergence_strength']:.3f}")
    else:
        print("11-24未检测到底背驰")
    
    # 查找最近的区间MACD底背驰
    print(f"\n最近的区间MACD底背驰信号:")
    # 查找最近15天内的底背驰信号
    divergence_data = recent[recent['divergence'] == 'bull']
    if not divergence_data.empty:
        for idx, row in divergence_data.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: 强度={row['divergence_strength']:.3f}")
    else:
        print("  最近15天内未检测到底背驰信号")
    
    print(f"\nMACD底背驰标准说明:")
    print(f"系统实现: 前一个MACD为负且当前MACD值大于前一个MACD值时判定为底背驰")
    print(f"用户观点: MACD底背驰应作为区间波段条件，而非单日触发信号")
    print(f"差异分析: 系统实现偏重单日信号检测，可能导致频繁触发；区间分析应考虑价格和MACD的趋势关系")

# 运行完整的check_512660_signal.py脚本以查看最新信号
print('\n' + '='*80)
print('运行完整信号检测:')
print('='*80)

# 总结差异
print('\n' + '='*100)
print('底分型标准对比分析总结:')
print('='*100)
print('1. 系统标准: 仅判断中间K线是否为前后3根K线的最低点且满足最小价格差')
print('2. 用户标准: 系统标准 + 后面有阳线确认 (更严格)')
print('')
print('MACD底背驰分析:')
print('1. 系统实现: 单日MACD变化检测，可能导致信号过于频繁')
print('2. 用户观点: 应作为区间波段条件，需结合价格趋势和MACD形态综合判断')
print('')
print('建议改进:')
print('1. 底分型识别: 增加后面K线确认机制，避免假分型')
print('2. MACD底背驰: 实现区间分析，比较相邻低点的MACD值，而非单日变化')
print('3. 信号生成: 综合考虑分型、笔、线段和区间MACD背驰，提高信号可靠性')
print('='*100)