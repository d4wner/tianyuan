#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试512660军工ETF的日线底背驰和底分型信号
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from src.config import load_config
from src.data_fetcher import StockDataFetcher
from src.calculator import ChanlunCalculator

def get_recent_daily_data(symbol, days=60):
    """
    获取最近N天的日线数据
    """
    # 加载配置
    config = load_config()
    
    # 创建数据获取器
    api = StockDataFetcher(
        max_retries=config.get('data_fetcher', {}).get('max_retries', 3),
        timeout=config.get('data_fetcher', {}).get('timeout', 10)
    )
    
    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 获取日线数据
    print(f"正在获取 {symbol} 从 {start_date} 到 {end_date} 的日线数据...")
    df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
    
    if df.empty:
        print(f"警告：未获取到 {symbol} 的数据")
        return None
    
    print(f"成功获取 {len(df)} 条日线数据")
    return df

def check_daily_bottom_signals(symbol):
    """
    检查日线底背驰和底分型信号
    """
    # 加载配置
    config = load_config()
    
    # 获取最近60天的日线数据
    df = get_recent_daily_data(symbol, days=60)
    if df is None or len(df) < 30:
        print("数据不足，无法进行缠论分析")
        return False
    
    # 创建缠论计算器
    calculator = ChanlunCalculator(config=config.get('chanlun', {}))
    
    # 执行完整的缠论计算流程
    # 1. 计算分型
    print("\n正在计算分型...")
    df_fractals = calculator.calculate_fractals(df)
    
    # 2. 计算笔划分
    print("正在计算笔划分...")
    df_pens = calculator.calculate_pens(df_fractals)
    
    # 3. 计算线段划分
    print("正在计算线段划分...")
    df_segments = calculator.calculate_segments(df_pens)
    
    # 4. 计算中枢（关键步骤）
    print("正在计算中枢...")
    df_central = calculator.calculate_central_banks(df_segments)
    
    # 5. 检测背离
    print("正在检测背离...")
    df_divergence = calculator.detect_divergence(df_central)
    
    # 6. 生成信号
    print("正在生成交易信号...")
    df_signals = calculator.generate_signals(df_divergence)
    
    # 检查最近10个交易日是否有MACD底背驰+底分型的组合（扩大时间范围）
    recent_days = 10
    print(f"\n检查最近 {recent_days} 个交易日的MACD底背驰和底分型信号：")
    
    # 复制原始数据，添加索引以便于显示
    result_df = df_signals.copy()
    result_df['date_str'] = result_df['date'].dt.strftime('%Y-%m-%d')
    
    # 提取最近N天的数据
    recent_df = result_df.tail(recent_days)
    
    # 显示最近N天的关键信息
    print("\n最近交易日的关键信息：")
    print("-" * 95)
    print(f"{'日期':<12} {'收盘价':<10} {'底分型':<8} {'MACD底背驰':<12} {'背驰指标':<15} {'信号':<10} {'信号强度':<10}")
    print("-" * 95)
    
    has_bottom_signal = False
    for _, row in recent_df.iterrows():
        bottom_fractal = "✓" if row['bottom_fractal'] else "-"
        
        # 特别检查MACD底背驰（现在基于MACD绿柱减小）
        is_macd_bottom_divergence = False
        if row['divergence'] == 'bull':
            # 由于我们现在只使用MACD，且所有bull类型的divergence都是MACD底背驰
            is_macd_bottom_divergence = True
        
        macd_divergence = "✓" if is_macd_bottom_divergence else "-"
        signal = row['signal']
        signal_strength = row['signal_strength']
        
        # 高亮显示同时满足底分型和MACD底背驰的行
        highlight = "**" if row['bottom_fractal'] and is_macd_bottom_divergence else "  "
        
        print(f"{highlight}{row['date_str']:<12} {row['close']:<10.3f} {bottom_fractal:<8} {macd_divergence:<12} {'MACD':<15} {signal:<10} {signal_strength:<10.3f}{highlight}")
        
        # 检查是否同时满足底分型和MACD底背驰
        if row['bottom_fractal'] and is_macd_bottom_divergence:
            has_bottom_signal = True
            print(f"\n📊 发现底分型+MACD底背驰组合信号：")
            print(f"   日期: {row['date_str']}")
            print(f"   价格: {row['close']:.3f}")
            print(f"   背驰强度: {row.get('divergence_strength', 0):.3f}")
            print(f"   背驰指标: {divergence_indicator}")
            
            # 计算当前价格与信号日价格的关系
            current_price = recent_df.iloc[-1]['close']
            price_change_pct = (current_price - row['close']) / row['close'] * 100
            print(f"   当前价格: {current_price:.3f} ({price_change_pct:+.2f}%)")
    
    # 检查最近30天内是否有MACD底背驰+底分型信号（更全面的检查）
    if not has_bottom_signal and len(df_signals) >= 30:
        print(f"\n正在检查最近30天内是否有MACD底背驰+底分型信号...")
        recent_30d_df = result_df.tail(30)
        
        # 寻找MACD底背驰+底分型的组合（现在只使用MACD，简化条件）
        macd_bottom_signals = recent_30d_df[
            recent_30d_df['bottom_fractal'] & 
            (recent_30d_df['divergence'] == 'bull')
        ]
        
        if not macd_bottom_signals.empty:
            print(f"\n📊 发现MACD底背驰+底分型组合信号（最近30天内）：")
            for _, row in macd_bottom_signals.iterrows():
                print(f"   日期: {row['date_str']}")
                print(f"   价格: {row['close']:.3f}")
                print(f"   背驰强度: {row.get('divergence_strength', 0):.3f}")
                print(f"   背驰指标: {row.get('divergence_indicator', '-')}")
            has_bottom_signal = True
    
    print("-" * 80)
    
    # 检查最近的买入信号
    recent_buy_signals = recent_df[recent_df['signal'] == 'buy']
    if not recent_buy_signals.empty:
        print(f"\n📈 最近 {recent_days} 天内检测到 {len(recent_buy_signals)} 个买入信号：")
        for _, row in recent_buy_signals.iterrows():
            print(f"   {row['date_str']}: 信号强度 {row['signal_strength']:.3f}, 来源: {row['signal_source']}")
    
    # 输出总结
    print("\n📋 信号总结:")
    if has_bottom_signal:
        print("✅ 已检测到日线MACD底背驰+底分型交易信号!")
    else:
        print("❌ 未检测到日线MACD底背驰+底分型交易信号")
    
    # 检查信号强度分布
    bottom_divergence_count = (df_signals['divergence'] == 'bull').sum()
    bottom_fractal_count = df_signals['bottom_fractal'].sum()
    
    # 统计MACD底背驰的数量（现在所有bull类型的divergence都是MACD底背驰）
    macd_bottom_divergence_count = (df_signals['divergence'] == 'bull').sum()
    
    print(f"\n📊 数据统计:")
    print(f"- 总底分型数量: {bottom_fractal_count}")
    print(f"- 总底背离数量: {bottom_divergence_count}")
    print(f"- 总MACD底背驰数量: {macd_bottom_divergence_count}")
    
    # 显示最近的MACD底背驰详细信息
    macd_divergence_rows = df_signals[df_signals['divergence'] == 'bull']
    if not macd_divergence_rows.empty:
        recent_macd = macd_divergence_rows.tail(1)
        print(f"\n📈 最近的MACD底背驰详情：")
        for _, row in recent_macd.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"   日期: {date_str}")
            print(f"   价格: {row['close']:.3f}")
            print(f"   背驰指标: {row.get('divergence_indicator', '-')}")
            print(f"   背驰强度: {row.get('divergence_strength', 0):.3f}")
    
    # 检查最近是否有底分型或底背离形成但信号未达到阈值的情况
    close_to_signal = False
    for _, row in recent_df.iterrows():
        if row['bottom_fractal'] and row['signal_strength'] > 0 and row['signal'] != 'buy':
            close_to_signal = True
            print(f"⚠️  {row['date_str']} 有底分型形成，但信号强度未达到买入阈值")
        elif row['divergence'] == 'bull' and row['signal_strength'] > 0 and row['signal'] != 'buy':
            close_to_signal = True
            print(f"⚠️  {row['date_str']} 有底背离形成，但信号强度未达到买入阈值")
    
    if close_to_signal:
        print("\n💡 注意：有底分型或底背离形成，但组合信号强度未达到买入阈值")
    
    return has_bottom_signal

if __name__ == "__main__":
    print("=" * 60)
    print("  512660军工ETF 日线底背驰+底分型信号检测工具")
    print("=" * 60)
    
    # 检查512660的信号
    symbol = "512660"
    has_signal = check_daily_bottom_signals(symbol)
    
    print("\n" + "=" * 60)
    print(f"检测完成! 是否检测到信号: {'✅ 是' if has_signal else '❌ 否'}")
    print("=" * 60)