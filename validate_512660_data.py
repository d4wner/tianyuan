#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""详细验证512660（酒ETF）数据质量，生成完整的验证报告"""

import os
import sys
import pandas as pd
import logging

# 项目路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.data_validator import DataValidator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_saved_data():
    """加载已保存的数据"""
    logger.info("开始加载已保存的数据...")
    
    # 加载日线数据
    daily_file = os.path.join(current_dir, 'data', 'daily', '512660_daily.csv')
    if os.path.exists(daily_file):
        daily_df = pd.read_csv(daily_file)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        logger.info(f"成功加载日线数据，共{len(daily_df)}条记录")
    else:
        logger.error(f"日线数据文件不存在: {daily_file}")
        daily_df = pd.DataFrame()
    
    # 加载周线数据
    weekly_file = os.path.join(current_dir, 'data', 'weekly', '512660_weekly.csv')
    if os.path.exists(weekly_file):
        weekly_df = pd.read_csv(weekly_file)
        weekly_df['date'] = pd.to_datetime(weekly_df['date'])
        logger.info(f"成功加载周线数据，共{len(weekly_df)}条记录")
    else:
        logger.error(f"周线数据文件不存在: {weekly_file}")
        weekly_df = pd.DataFrame()
    
    return daily_df, weekly_df

def perform_detailed_validation(daily_df, weekly_df):
    """执行详细的数据验证"""
    logger.info("开始执行详细的数据验证...")
    
    # 创建验证器实例
    validator = DataValidator(min_daily_k_count=60, min_weekly_k_count=52)
    
    # 执行综合验证
    validation_result = validator.validate_all_data(daily_df, weekly_df)
    
    # 打印验证报告
    print("\n" + validation_result['validation_report'])
    
    # 执行额外的数据质量检查
    perform_additional_checks(daily_df, weekly_df)
    
    return validation_result['overall_valid']

def perform_additional_checks(daily_df, weekly_df):
    """执行额外的数据质量检查"""
    print("\n===== 额外数据质量检查 =====")
    
    # 日线数据额外检查
    if not daily_df.empty:
        print("\n【日线数据额外检查】")
        
        # 检查数据完整性
        missing_values = daily_df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"  警告: 存在缺失值\n  {missing_values[missing_values > 0]}")
        else:
            print("  ✓ 数据完整性良好，无缺失值")
        
        # 检查价格异常
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if daily_df[col].min() <= 0:
                print(f"  警告: {col}列存在非正数价格")
        
        # 检查交易量异常
        if 'volume' in daily_df.columns:
            if (daily_df['volume'] < 0).any():
                print("  警告: 交易量存在负值")
            if (daily_df['volume'] == 0).sum() > 0:
                print(f"  警告: 存在{sum(daily_df['volume'] == 0)}条交易量为0的记录")
        
        # 检查数据时间连续性
        daily_df_sorted = daily_df.sort_values('date')
        date_diffs = daily_df_sorted['date'].diff().dt.days
        # 找出非交易日的间隔
        gaps = date_diffs[date_diffs > 3]  # 超过3天的间隔可能是数据缺失
        if len(gaps) > 0:
            print(f"  警告: 存在{len(gaps)}个较大的日期间隔")
            print(f"  最大间隔: {date_diffs.max()}天")
        else:
            print("  ✓ 日期连续性良好")
    
    # 周线数据额外检查
    if not weekly_df.empty:
        print("\n【周线数据额外检查】")
        
        # 检查数据完整性
        missing_values = weekly_df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"  警告: 存在缺失值\n  {missing_values[missing_values > 0]}")
        else:
            print("  ✓ 数据完整性良好，无缺失值")
        
        # 检查周线数据的时间间隔
        weekly_df_sorted = weekly_df.sort_values('date')
        week_diffs = weekly_df_sorted['date'].diff().dt.days
        # 找出异常的周间隔（正常应该是5-8天）
        abnormal_weeks = week_diffs[(week_diffs < 5) | (week_diffs > 14)]
        if len(abnormal_weeks) > 0:
            print(f"  警告: 存在{len(abnormal_weeks)}个异常的周间隔")
        else:
            print("  ✓ 周线时间间隔正常")
    
    print("==========================")

def main():
    """主函数"""
    print("开始对512660（酒ETF）数据进行详细验证...")
    
    # 加载数据
    daily_df, weekly_df = load_saved_data()
    
    if daily_df.empty or weekly_df.empty:
        print("错误：无法加载数据，请先运行数据获取脚本")
        return False
    
    # 执行验证
    is_valid = perform_detailed_validation(daily_df, weekly_df)
    
    print(f"\n数据验证结果: {'✓ 通过' if is_valid else '✗ 未通过'}")
    
    if is_valid:
        print("✓ 数据质量满足要求，可以进行后续的交易信号分析")
    else:
        print("✗ 数据质量不满足要求，请检查数据问题")
    
    return is_valid

if __name__ == "__main__":
    main()