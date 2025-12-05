#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""获取ETF的长期历史数据用于完整判定"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import argparse

# 项目路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.data_fetcher import StockDataFetcher
from src.data_validator import DataValidator

def main(symbol="512690", start_date=None, end_date=None):
    """主函数：获取指定ETF的长期历史数据"""
    print(f"开始获取{symbol}的历史数据...")
    
    # 初始化数据获取器
    fetcher = StockDataFetcher()
    
    # 设置默认日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"日期范围: {start_date} 至 {end_date}")
    
    # 获取日线数据
    print("正在获取日线数据...")
    daily_data = fetcher.get_daily_data(symbol, start_date, end_date, force_refresh=True)
    
    if daily_data.empty:
        print("警告：日线数据获取失败或为空！")
    else:
        print(f"日线数据获取成功：共 {len(daily_data)} 条记录")
        print(f"日线数据时间范围：{daily_data['date'].min().strftime('%Y-%m-%d')} 至 {daily_data['date'].max().strftime('%Y-%m-%d')}")
        
        # 保存日线数据
        daily_dir = os.path.join(current_dir, 'data', 'daily')
        os.makedirs(daily_dir, exist_ok=True)
        daily_file = os.path.join(daily_dir, f"{symbol}_daily.csv")
        daily_data.to_csv(daily_file, index=False, encoding='utf-8')
        print(f"日线数据已保存至：{daily_file}")
    
    # 获取周线数据
    print("正在获取周线数据...")
    weekly_data, actual_start, actual_end = fetcher.get_weekly_data(symbol, start_date, end_date)
    
    if weekly_data.empty:
        print("警告：周线数据获取失败或为空！")
    else:
        print(f"周线数据获取成功：共 {len(weekly_data)} 条记录")
        print(f"周线数据时间范围：{weekly_data['date'].min().strftime('%Y-%m-%d')} 至 {weekly_data['date'].max().strftime('%Y-%m-%d')}")
        
        # 保存周线数据
        weekly_dir = os.path.join(current_dir, 'data', 'weekly')
        os.makedirs(weekly_dir, exist_ok=True)
        weekly_file = os.path.join(weekly_dir, f"{symbol}_weekly.csv")
        weekly_data.to_csv(weekly_file, index=False, encoding='utf-8')
        print(f"周线数据已保存至：{weekly_file}")
    
    # 验证数据有效性
    print("\n开始验证数据有效性...")
    validator = DataValidator()
    daily_valid = validator.validate_daily_data(daily_data)
    weekly_valid = validator.validate_weekly_data(weekly_data)
    
    print(f"日线数据有效性: {'有效' if daily_valid else '无效'}")
    print(f"周线数据有效性: {'有效' if weekly_valid else '无效'}")
    
    if daily_valid and weekly_valid:
        print("\n数据获取和验证完成！数据足够进行完整的交易信号判定。")
    else:
        print("\n警告：数据可能不足以进行完整的交易信号判定，请检查数据质量。")
    
    return daily_data, weekly_data

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='获取ETF的长期历史数据')
    parser.add_argument('--symbol', type=str, default='512690', help='ETF代码，默认为512690')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期，格式: YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期，格式: YYYY-MM-DD')
    args = parser.parse_args()
    
    daily_data, weekly_data = main(args.symbol, args.start_date, args.end_date)