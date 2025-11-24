#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试不同分型参数配置，找出最适合的参数组合"""

import os
import sys
import pandas as pd
import numpy as np

# 项目路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入计算器
from src.calculator import ChanlunCalculator

def load_test_data():
    """加载测试数据"""
    # 尝试从exports目录加载最新的数据文件
    exports_dir = os.path.join(project_root, "outputs", "exports")
    if os.path.exists(exports_dir):
        # 获取所有sh512660相关的CSV文件并按时间排序
        csv_files = [f for f in os.listdir(exports_dir) 
                    if f.startswith("sh512660_daily_") and f.endswith(".csv")]
        if csv_files:
            # 按文件名排序（文件名包含时间戳）
            csv_files.sort(reverse=True)
            latest_file = os.path.join(exports_dir, csv_files[0])
            print(f"使用最新数据文件: {latest_file}")
            df = pd.read_csv(latest_file)
            # 确保日期列格式正确
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    
    # 如果没有找到最新文件，尝试获取新数据
    print("未找到最新数据文件，尝试获取数据...")
    from src.data_fetcher import StockDataFetcher
    fetcher = StockDataFetcher()
    fetcher.cache_enabled = False
    df = fetcher.get_daily_data("512660", "2025-11-01", "2025-11-24")
    return df

def test_different_sensitivity_params():
    """测试不同的分型灵敏度参数"""
    # 加载数据
    df = load_test_data()
    if df.empty:
        print("无法加载测试数据，退出")
        return
    
    # 测试不同的分型灵敏度参数
    sensitivity_params = [1, 2, 3]  # 测试1, 2, 3三种灵敏度
    min_price_diff_params = [0.001, 0.0005]  # 测试不同的最小价格差
    
    print("\n开始测试不同的分型参数组合...")
    print("=" * 80)
    
    for sensitivity in sensitivity_params:
        for min_price_diff in min_price_diff_params:
            print(f"\n测试参数: fractal_sensitivity={sensitivity}, fractal_min_price_diff={min_price_diff}")
            print("-" * 60)
            
            # 创建计算器实例
            config = {
                'chanlun': {
                    'fractal_sensitivity': sensitivity,
                    'fractal_min_price_diff': min_price_diff
                }
            }
            calculator = ChanlunCalculator(config)
            
            # 计算分型
            result_df = calculator.calculate_fractals(df)
            
            # 找出顶分型和底分型
            top_fractals = result_df[result_df['top_fractal']]
            bottom_fractals = result_df[result_df['bottom_fractal']]
            
            print(f"顶分型数量: {len(top_fractals)}")
            print(f"底分型数量: {len(bottom_fractals)}")
            
            # 检查11月17日和11月24日的分型
            if 'date' in result_df.columns:
                # 11月17日检查
                nov17 = result_df[result_df['date'] == '2025-11-17']
                if not nov17.empty:
                    is_top = nov17['top_fractal'].iloc[0]
                    is_bottom = nov17['bottom_fractal'].iloc[0]
                    print(f"11月17日: {'顶分型' if is_top else '底分型' if is_bottom else '无分型'}")
                
                # 11月24日检查
                nov24 = result_df[result_df['date'] == '2025-11-24']
                if not nov24.empty:
                    is_top = nov24['top_fractal'].iloc[0]
                    is_bottom = nov24['bottom_fractal'].iloc[0]
                    print(f"11月24日: {'顶分型' if is_top else '底分型' if is_bottom else '无分型'}")
            
            # 显示最近的几个分型
            if not top_fractals.empty:
                print("\n最近5个顶分型:")
                recent_tops = top_fractals.sort_values('date', ascending=False).head()
                for _, row in recent_tops.iterrows():
                    print(f"  {row['date'].strftime('%Y-%m-%d')} - 价格: {row['fractal_price']:.4f}")
            
            if not bottom_fractals.empty:
                print("\n最近5个底分型:")
                recent_bottoms = bottom_fractals.sort_values('date', ascending=False).head()
                for _, row in recent_bottoms.iterrows():
                    print(f"  {row['date'].strftime('%Y-%m-%d')} - 价格: {row['fractal_price']:.4f}")

def main():
    """主函数"""
    test_different_sensitivity_params()

if __name__ == "__main__":
    main()