#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析512660 2025年11月买点信号
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.daily_buy_signal_detector import BuySignalDetector
from src.weekly_trend_detector import WeeklyTrendDetector
from src.divergence_detector import DivergenceDetector

def analyze_november_buy_points():
    """
    分析512660在2025年11月的买点情况
    """
    print("=" * 60)
    print("512660 (军工ETF) 2025年11月买点信号分析")
    print("=" * 60)
    
    # 读取完整日线数据
    full_df = pd.read_csv('/Users/pingan/tools/trade/tianyuan/data/daily/512660_daily.csv')
    full_df['date'] = pd.to_datetime(full_df['date'])
    
    # 筛选11月数据
    nov_df = full_df[(full_df['date'] >= '2025-11-01') & (full_df['date'] <= '2025-11-30')]
    print(f"11月交易数据：{len(nov_df)}个交易日")
    print(f"日期范围：{nov_df['date'].min().strftime('%Y-%m-%d')} 至 {nov_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"价格范围：{nov_df['close'].min():.4f} - {nov_df['close'].max():.4f} 元")
    
    print("\n1. 日线买点信号检测：")
    print("-" * 30)
    
    # 初始化买点检测器
    detector = BuySignalDetector()
    
    # 使用完整数据进行检测（需要至少60天数据）
    # 获取11月之前的至少30天数据，加上11月的28天，共58天，接近60天要求
    start_date = pd.to_datetime('2025-11-01') - pd.Timedelta(days=32)
    detection_df = full_df[(full_df['date'] >= start_date) & (full_df['date'] <= '2025-11-30')]
    
    print(f"使用检测数据范围：{detection_df['date'].min().strftime('%Y-%m-%d')} 至 {detection_df['date'].max().strftime('%Y-%m-%d')} (共{len(detection_df)}天)")
    
    # 计算波动率并设置动态参数
    detector.adapt_to_volatility(detection_df)
    
    # 检测所有买点信号
    buy_signals = detector.detect_buy_signals(detection_df)
    
    # 过滤出11月的信号
    nov_buy_signals = {
        'first_buy': buy_signals['first_buy'] if buy_signals.get('first_buy', {}).get('date') and buy_signals['first_buy']['date'] >= pd.to_datetime('2025-11-01') else {},
        'second_buy': buy_signals['second_buy'] if buy_signals.get('second_buy', {}).get('date') and buy_signals['second_buy']['date'] >= pd.to_datetime('2025-11-01') else {},
        'third_buy': buy_signals['third_buy'] if buy_signals.get('third_buy', {}).get('date') and buy_signals['third_buy']['date'] >= pd.to_datetime('2025-11-01') else {},
        'break_central_pullback_buy': buy_signals['break_central_pullback_buy'] if buy_signals.get('break_central_pullback_buy', {}).get('date') and buy_signals['break_central_pullback_buy']['date'] >= pd.to_datetime('2025-11-01') else {}
    }
    
    print(f"日线一买信号：{'✅' if nov_buy_signals['first_buy'].get('is_valid') else '❌'}")
    if nov_buy_signals['first_buy'].get('is_valid'):
        print(f"  - 日期：{nov_buy_signals['first_buy']['date'].strftime('%Y-%m-%d')}，价格：{nov_buy_signals['first_buy']['price']:.4f}元")
        print(f"    详情：{nov_buy_signals['first_buy'].get('details', '无')}")
    
    print(f"日线二买信号：{'✅' if nov_buy_signals['second_buy'].get('is_valid') else '❌'}")
    if nov_buy_signals['second_buy'].get('is_valid'):
        print(f"  - 日期：{nov_buy_signals['second_buy']['date'].strftime('%Y-%m-%d')}，价格：{nov_buy_signals['second_buy']['price']:.4f}元")
        print(f"    详情：{nov_buy_signals['second_buy'].get('details', '无')}")
    
    print(f"日线三买信号：{'✅' if nov_buy_signals['third_buy'].get('is_valid') else '❌'}")
    if nov_buy_signals['third_buy'].get('is_valid'):
        print(f"  - 日期：{nov_buy_signals['third_buy']['date'].strftime('%Y-%m-%d')}，价格：{nov_buy_signals['third_buy']['price']:.4f}元")
        print(f"    详情：{nov_buy_signals['third_buy'].get('details', '无')}")
    
    print(f"破中枢反抽信号：{'✅' if nov_buy_signals['break_central_pullback_buy'].get('is_valid') else '❌'}")
    if nov_buy_signals['break_central_pullback_buy'].get('is_valid'):
        print(f"  - 日期：{nov_buy_signals['break_central_pullback_buy']['date'].strftime('%Y-%m-%d')}，价格：{nov_buy_signals['break_central_pullback_buy']['price']:.4f}元")
        print(f"    详情：{nov_buy_signals['break_central_pullback_buy'].get('details', '无')}")
    
    # 周线趋势分析
    print("\n2. 周线趋势分析：")
    print("-" * 30)
    
    # 读取周线数据
    try:
        weekly_df = pd.read_csv('/Users/pingan/tools/trade/tianyuan/data/weekly/512660_weekly.csv')
        weekly_df['date'] = pd.to_datetime(weekly_df['date'])
        
        weekly_detector = WeeklyTrendDetector()
        trend_result = weekly_detector.detect_weekly_bullish_trend(weekly_df)
        
        print(f"周线趋势：{'多头' if trend_result.get('is_bullish', False) else '空头'}")
        print(f"置信度：{trend_result.get('confidence_level', '未知')}")
        print(f"确认日期：{trend_result.get('confirm_date', '未知')}")
    except FileNotFoundError:
        print("周线数据文件未找到")
    
    # 综合分析
    print("\n3. 综合框架策略评估：")
    print("-" * 30)
    
    total_nov_signals = sum(1 for sig in nov_buy_signals.values() if sig.get('is_valid'))
    print(f"11月捕获的买点信号总数：{total_nov_signals}个")
    
    # 策略健康度评估
    print("\n策略健康度评估：")
    print(f"- 数据完整性：{'✅ 完整' if len(detection_df) >= 40 else '⚠️  数据量不足'}")
    print(f"- 波动率计算：{'✅ 完成'}")
    print(f"- 信号检测功能：{'✅ 正常' if buy_signals else '⚠️  信号检测异常'}")
    
    # 框架状态总结
    if total_nov_signals > 0:
        print("\n✅ 框架策略运行正常，在11月成功捕获到买点信号！")
        print("建议：关注捕获到的信号，结合其他指标进行综合判断")
    else:
        print("\n⚠️  11月未捕获到明显的买点信号")
        print("可能原因：")
        print("  1. 市场处于横盘整理状态")
        print("  2. 价格未形成明显的中枢结构")
        print("  3. 背驰条件未满足")
        print("  4. 数据量接近60天要求，可能影响检测效果")
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    analyze_november_buy_points()