#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""更新股票数据脚本 - 用于获取最新的军工ETF(512660)数据"""

import os
import sys
import pandas as pd
from datetime import datetime

# 项目路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据获取器
from src.data_fetcher import StockDataFetcher

def update_sh512660_data(start_date="2025-11-01", end_date="2025-11-24"):
    """更新军工ETF(512660)数据"""
    print(f"开始更新军工ETF(512660)数据，日期范围: {start_date} 至 {end_date}")
    
    # 创建数据获取器实例
    fetcher = StockDataFetcher()  # 创建实例，使用默认参数
    
    # 手动禁用缓存
    fetcher.cache_enabled = False
    
    # 获取日线数据
    df = fetcher.get_daily_data("512660", start_date, end_date)
    
    if df.empty:
        print("警告: 获取数据失败，返回空数据框")
        return None
    
    print(f"成功获取数据: {len(df)} 条记录")
    print(f"数据日期范围: {df['date'].min()} 至 {df['date'].max()}")
    
    # 保存数据到exports目录
    exports_dir = os.path.join(project_root, "outputs", "exports")
    os.makedirs(exports_dir, exist_ok=True)
    
    # 保存为CSV
    csv_filename = f"sh512660_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(exports_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存至: {csv_path}")
    
    # 保存为JSON
    json_filename = f"sh512660_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(exports_dir, json_filename)
    
    # 将日期转换为字符串格式以便JSON序列化
    df_for_json = df.copy()
    if 'date' in df_for_json.columns:
        df_for_json['date'] = df_for_json['date'].dt.strftime('%Y-%m-%d')
    
    df_for_json.to_json(json_path, orient='records', force_ascii=False)
    print(f"数据已保存至: {json_path}")
    
    return df

def main():
    """主函数，支持命令行参数"""
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='更新军工ETF(512660)数据')
    parser.add_argument('--start_date', type=str, default='2025-11-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-11-24', help='结束日期 (YYYY-MM-DD)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取最新的军工ETF数据
    df = update_sh512660_data(args.start_date, args.end_date)
    
    if df is not None:
        # 显示最新的几条数据
        print("\n最新5条数据:")
        print(df.tail())
        
        # 检查是否包含11月24日的数据
        if 'date' in df.columns:
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
            has_nov24 = any(df['date_str'] == '2025-11-24')
            print(f"\n是否包含2025-11-24数据: {'是' if has_nov24 else '否'}")
            
            if has_nov24:
                nov24_data = df[df['date_str'] == '2025-11-24']
                print("\n2025-11-24数据:")
                print(nov24_data)

if __name__ == "__main__":
    main()