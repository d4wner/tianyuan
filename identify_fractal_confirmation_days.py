#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格按照缠论定义识别顶底分型确认日
缠论定义：
- 底分型：由5根K线组成，中间K线的低点是5根中最低的，且左右K线的低点都高于它
- 顶分型：由5根K线组成，中间K线的高点是5根中最高的，且左右K线的高点都低于它
- 确认日：右侧第二根K线收盘价确认分型形成的日期
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FractalIdentifier')

def load_price_data(file_path):
    """加载价格数据"""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"成功加载价格数据，共{len(df)}条记录")
        return df
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        raise

def identify_bottom_fractals(df_prices):
    """
    识别底分型
    返回：底分型确认日列表，每个元素包含确认日和相关K线信息
    """
    bottom_fractals = []
    n = len(df_prices)
    
    for i in range(2, n - 2):  # 中间K线位置，需要前后各有2根K线
        # 检查是否符合底分型条件：中间K线低点是5根中最低
        if (df_prices.iloc[i]['low'] < df_prices.iloc[i-2]['low'] and 
            df_prices.iloc[i]['low'] < df_prices.iloc[i-1]['low'] and 
            df_prices.iloc[i]['low'] < df_prices.iloc[i+1]['low'] and 
            df_prices.iloc[i]['low'] < df_prices.iloc[i+2]['low']):
            
            # 确认日是右侧第二根K线（i+2位置）
            confirmation_date = df_prices.iloc[i+2]['date']
            
            bottom_fractals.append({
                'confirmation_date': confirmation_date,
                'fractal_center_date': df_prices.iloc[i]['date'],
                'fractal_low': df_prices.iloc[i]['low'],
                'center_close': df_prices.iloc[i]['close'],
                'confirmation_close': df_prices.iloc[i+2]['close'],
                'fractal_bars': {
                    'bar1': df_prices.iloc[i-2].to_dict(),
                    'bar2': df_prices.iloc[i-1].to_dict(),
                    'center': df_prices.iloc[i].to_dict(),
                    'bar4': df_prices.iloc[i+1].to_dict(),
                    'bar5': df_prices.iloc[i+2].to_dict()
                }
            })
    
    return bottom_fractals

def identify_top_fractals(df_prices):
    """
    识别顶分型
    返回：顶分型确认日列表，每个元素包含确认日和相关K线信息
    """
    top_fractals = []
    n = len(df_prices)
    
    for i in range(2, n - 2):  # 中间K线位置，需要前后各有2根K线
        # 检查是否符合顶分型条件：中间K线高点是5根中最高
        if (df_prices.iloc[i]['high'] > df_prices.iloc[i-2]['high'] and 
            df_prices.iloc[i]['high'] > df_prices.iloc[i-1]['high'] and 
            df_prices.iloc[i]['high'] > df_prices.iloc[i+1]['high'] and 
            df_prices.iloc[i]['high'] > df_prices.iloc[i+2]['high']):
            
            # 确认日是右侧第二根K线（i+2位置）
            confirmation_date = df_prices.iloc[i+2]['date']
            
            top_fractals.append({
                'confirmation_date': confirmation_date,
                'fractal_center_date': df_prices.iloc[i]['date'],
                'fractal_high': df_prices.iloc[i]['high'],
                'center_close': df_prices.iloc[i]['close'],
                'confirmation_close': df_prices.iloc[i+2]['close'],
                'fractal_bars': {
                    'bar1': df_prices.iloc[i-2].to_dict(),
                    'bar2': df_prices.iloc[i-1].to_dict(),
                    'center': df_prices.iloc[i].to_dict(),
                    'bar4': df_prices.iloc[i+1].to_dict(),
                    'bar5': df_prices.iloc[i+2].to_dict()
                }
            })
    
    return top_fractals

def filter_by_date_range(fractals, start_date, end_date):
    """按日期范围过滤分型"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    return [f for f in fractals if start <= f['confirmation_date'] <= end]

def analyze_october_november_fractals():
    """分析2025年10月和11月的顶底分型"""
    # 加载最新的价格数据
    price_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_daily_20251125_084859.csv'
    df_prices = load_price_data(price_file)
    
    # 识别所有底分型和顶分型
    bottom_fractals = identify_bottom_fractals(df_prices)
    top_fractals = identify_top_fractals(df_prices)
    
    logger.info(f"总共识别到底分型 {len(bottom_fractals)} 个")
    logger.info(f"总共识别到顶分型 {len(top_fractals)} 个")
    
    # 过滤出2025年10月和11月的分型
    october_bottoms = filter_by_date_range(bottom_fractals, '2025-10-01', '2025-10-31')
    november_bottoms = filter_by_date_range(bottom_fractals, '2025-11-01', '2025-11-30')
    
    october_tops = filter_by_date_range(top_fractals, '2025-10-01', '2025-10-31')
    november_tops = filter_by_date_range(top_fractals, '2025-11-01', '2025-11-30')
    
    # 输出10月份的分型确认日
    print("\n" + "="*60)
    print("2025年10月份顶底分型确认日（严格缠论定义）")
    print("="*60)
    
    print("\n底分型确认日：")
    if october_bottoms:
        for idx, fractal in enumerate(october_bottoms, 1):
            print(f"{idx}. 确认日: {fractal['confirmation_date'].strftime('%Y-%m-%d')}")
            print(f"   分型中心: {fractal['fractal_center_date'].strftime('%Y-%m-%d')}")
            print(f"   分型低点: {fractal['fractal_low']:.3f}")
            print(f"   确认日收盘价: {fractal['confirmation_close']:.3f}")
            print("   " + "-"*40)
    else:
        print("未识别到10月份的底分型")
    
    print("\n顶分型确认日：")
    if october_tops:
        for idx, fractal in enumerate(october_tops, 1):
            print(f"{idx}. 确认日: {fractal['confirmation_date'].strftime('%Y-%m-%d')}")
            print(f"   分型中心: {fractal['fractal_center_date'].strftime('%Y-%m-%d')}")
            print(f"   分型高点: {fractal['fractal_high']:.3f}")
            print(f"   确认日收盘价: {fractal['confirmation_close']:.3f}")
            print("   " + "-"*40)
    else:
        print("未识别到10月份的顶分型")
    
    # 输出11月份的分型确认日（如果有数据）
    print("\n" + "="*60)
    print("2025年11月份顶底分型确认日（严格缠论定义）")
    print("="*60)
    
    print("\n底分型确认日：")
    if november_bottoms:
        for idx, fractal in enumerate(november_bottoms, 1):
            print(f"{idx}. 确认日: {fractal['confirmation_date'].strftime('%Y-%m-%d')}")
            print(f"   分型中心: {fractal['fractal_center_date'].strftime('%Y-%m-%d')}")
            print(f"   分型低点: {fractal['fractal_low']:.3f}")
            print(f"   确认日收盘价: {fractal['confirmation_close']:.3f}")
            print("   " + "-"*40)
    else:
        print("未识别到11月份的底分型")
    
    print("\n顶分型确认日：")
    if november_tops:
        for idx, fractal in enumerate(november_tops, 1):
            print(f"{idx}. 确认日: {fractal['confirmation_date'].strftime('%Y-%m-%d')}")
            print(f"   分型中心: {fractal['fractal_center_date'].strftime('%Y-%m-%d')}")
            print(f"   分型高点: {fractal['fractal_high']:.3f}")
            print(f"   确认日收盘价: {fractal['confirmation_close']:.3f}")
            print("   " + "-"*40)
    else:
        print("未识别到11月份的顶分型")
    
    # 额外验证：打印10月份的所有K线，方便用户对照
    october_data = df_prices[(df_prices['date'] >= '2025-10-01') & (df_prices['date'] <= '2025-10-31')]
    
    print("\n" + "="*60)
    print("2025年10月军工ETF(512660)日线数据")
    print("="*60)
    print("日期       开盘价   最高价   最低价   收盘价")
    print("-"*50)
    
    for _, row in october_data.iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d')}  {row['open']:.3f}  {row['high']:.3f}  {row['low']:.3f}  {row['close']:.3f}")
    
    # 返回分析结果，便于后续使用
    return {
        'october_bottoms': october_bottoms,
        'october_tops': october_tops,
        'november_bottoms': november_bottoms,
        'november_tops': november_tops
    }

def main():
    try:
        logger.info("开始识别顶底分型确认日...")
        
        # 分析10月和11月的顶底分型
        results = analyze_october_november_fractals()
        
        # 统计结果
        total_october = len(results['october_bottoms']) + len(results['october_tops'])
        total_november = len(results['november_bottoms']) + len(results['november_tops'])
        
        print(f"\n总结：2025年10月共识别到 {total_october} 个分型，11月共识别到 {total_november} 个分型")
        print(f"注意：以上分型确认日严格按照缠论定义识别，确认日为右侧第二根K线")
        
    except Exception as e:
        logger.error(f"识别过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()