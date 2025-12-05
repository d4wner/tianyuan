#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析512660（酒ETF）周线多头趋势"""

import os
import sys
import pandas as pd
import logging
import json

# 项目路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.weekly_trend_detector import WeeklyTrendDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_weekly_data():
    """加载已保存的周线数据"""
    logger.info("开始加载512660周线数据...")
    
    weekly_file = os.path.join(current_dir, 'data', 'weekly', '512660_weekly.csv')
    if os.path.exists(weekly_file):
        weekly_df = pd.read_csv(weekly_file)
        weekly_df['date'] = pd.to_datetime(weekly_df['date'])
        weekly_df = weekly_df.sort_values('date')  # 确保数据按日期排序
        logger.info(f"成功加载周线数据，共{len(weekly_df)}条记录")
        logger.info(f"时间范围: {weekly_df['date'].min().strftime('%Y-%m-%d')} 至 {weekly_df['date'].max().strftime('%Y-%m-%d')}")
    else:
        logger.error(f"周线数据文件不存在: {weekly_file}")
        weekly_df = pd.DataFrame()
    
    return weekly_df

def perform_weekly_trend_analysis(weekly_df):
    """执行周线多头趋势分析"""
    logger.info("开始执行512660周线多头趋势分析...")
    
    # 创建周线趋势检测器
    detector = WeeklyTrendDetector(lookback_period=10, macd_fast=12, macd_slow=26, macd_signal=9)
    
    # 执行趋势检测
    result = detector.detect_weekly_bullish_trend(weekly_df)
    
    # 生成详细报告
    report = detector.generate_trend_report(result)
    
    # 输出报告
    print("\n" + report)
    
    # 保存分析结果
    save_analysis_result(result, report)
    
    return result

def save_analysis_result(result, report):
    """保存分析结果"""
    # 创建结果目录
    result_dir = os.path.join(current_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存JSON格式的结果数据
    json_file = os.path.join(result_dir, '512660_weekly_trend_analysis.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"分析结果已保存至: {json_file}")
    
    # 保存文本格式的报告
    report_file = os.path.join(result_dir, '512660_weekly_trend_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"分析报告已保存至: {report_file}")

def perform_additional_analysis(weekly_df):
    """执行额外的周线分析"""
    print("\n===== 周线数据额外分析 =====")
    
    # 计算近期表现
    recent_4_weeks = weekly_df.tail(4)
    recent_8_weeks = weekly_df.tail(8)
    recent_13_weeks = weekly_df.tail(13)  # 约3个月
    
    def calculate_performance(data):
        start_price = data.iloc[0]['close']
        end_price = data.iloc[-1]['close']
        percent_change = (end_price - start_price) / start_price * 100
        return percent_change
    
    # 计算不同周期的涨跌幅
    perf_4w = calculate_performance(recent_4_weeks)
    perf_8w = calculate_performance(recent_8_weeks)
    perf_13w = calculate_performance(recent_13_weeks)
    
    print(f"近4周涨跌幅: {perf_4w:.2f}%")
    print(f"近8周涨跌幅: {perf_8w:.2f}%")
    print(f"近13周涨跌幅: {perf_13w:.2f}%")
    
    # 计算波动性指标
    weekly_df['weekly_return'] = weekly_df['close'].pct_change()
    recent_volatility = weekly_df['weekly_return'].tail(26).std() * 100  # 年化波动率估计
    
    print(f"\n近26周波动率: {recent_volatility:.2f}%")
    
    # 检查均线系统
    weekly_df['ma5'] = weekly_df['close'].rolling(window=5).mean()
    weekly_df['ma10'] = weekly_df['close'].rolling(window=10).mean()
    weekly_df['ma20'] = weekly_df['close'].rolling(window=20).mean()
    
    recent_data = weekly_df.tail(1).iloc[0]
    print(f"\n最新均线状态:")
    print(f"  5周均线: {recent_data['ma5']:.3f}")
    print(f"  10周均线: {recent_data['ma10']:.3f}")
    print(f"  20周均线: {recent_data['ma20']:.3f}")
    
    # 检查均线排列
    ma_aligned = recent_data['ma5'] > recent_data['ma10'] > recent_data['ma20'] > recent_data['close']
    print(f"  均线多头排列: {'是' if ma_aligned else '否'}")
    
    print("============================")

def main():
    """主函数"""
    print("开始分析512660（酒ETF）周线多头趋势...")
    
    # 加载周线数据
    weekly_df = load_weekly_data()
    
    if weekly_df.empty:
        print("错误：无法加载周线数据，请先运行数据获取脚本")
        return False
    
    # 执行趋势分析
    result = perform_weekly_trend_analysis(weekly_df)
    
    # 执行额外分析
    perform_additional_analysis(weekly_df)
    
    print(f"\n周线多头趋势判定结果: {'✓ 确认' if result['bullish_trend'] else '✗ 未确认'}")
    
    return result['bullish_trend']

if __name__ == "__main__":
    main()