#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新ETF测试脚本

该脚本用于测试优化后的中枢判定和买卖信号逻辑对新ETF的效果。
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from src.daily_buy_signal_detector import BuySignalDetector
from src.etf_trend_detector import ETFTrendDetector
from src.invalid_signal_filter import InvalidSignalFilter
from src.data_fetcher import StockDataFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_new_etfs.log')
    ]
)
logger = logging.getLogger(__name__)

# 测试的新ETF代码
NEW_ETFS = [
    "512880",  # 证券ETF
    "510300",  # 沪深300ETF
    "512010",  # 医药ETF
    "512480",  # 半导体ETF
    "515000",  # 科技ETF
    "512000"   # 券商ETF
]

def test_single_etf(stock_code):
    """测试单个ETF的中枢判定和买卖信号
    
    Args:
        stock_code: ETF代码
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"测试ETF: {stock_code}")
    logger.info(f"{'='*60}")
    
    try:
        # 初始化数据获取器
        data_fetcher = StockDataFetcher()
        
        # 获取日线数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        df_daily = data_fetcher.get_daily_data(stock_code, start_date, end_date)
        
        if df_daily is None or df_daily.empty:
            logger.warning(f"无法获取 {stock_code} 的数据")
            return None
        
        logger.info(f"成功获取 {stock_code} 的日线数据: {len(df_daily)} 条记录")
        logger.info(f"数据时间范围: {df_daily.index[0]} 至 {df_daily.index[-1]}")
        
        # 测试1: 周线趋势检测
        logger.info("\n1. 测试周线趋势检测...")
        trend_detector = ETFTrendDetector()
        center_result = trend_detector.detect_sideways_market(df_daily)
        
        logger.info(f"中枢检测结果: {center_result}")
        
        # 2. 测试日线买点检测...
        logger.info("\n2. 测试日线买点检测...")
        buy_detector = BuySignalDetector()
        # 初始化动态参数（适配波动率）
        buy_detector.adapt_to_volatility(df_daily)
        buy_signals = buy_detector.detect_buy_signals(df_daily)
        
        for signal_type, signal_info in buy_signals['signals'].items():
            if signal_info['detected']:
                logger.info(f"  {signal_type}: 存在信号")
                logger.info(f"    详细信息: {signal_info['details']}")
            else:
                logger.info(f"  {signal_type}: 无信号")
        
        return {
            'stock_code': stock_code,
            'centers': center_result,
            'buy_signals': buy_signals
        }
        
    except Exception as e:
        logger.error(f"测试 {stock_code} 时发生错误: {str(e)}", exc_info=True)
        return None

def main():
    """主函数"""
    logger.info("开始测试新ETF的中枢判定和买卖信号优化效果")
    
    results = []
    
    for etf_code in NEW_ETFS:
        result = test_single_etf(etf_code)
        if result:
            results.append(result)
    
    logger.info(f"\n{'='*60}")
    logger.info("测试完成")
    logger.info(f"{'='*60}")
    
    # 生成测试报告
    report_file = 'new_etfs_test_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"新ETF测试报告\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        
        for result in results:
            f.write(f"ETF代码: {result['stock_code']}\n")
            f.write(f"{'='*30}\n")
            
            # 中枢信息
            centers = result['centers']
            if centers['is_sideways']:
                f.write(f"是否横盘: 是\n")
                f.write(f"振幅: {centers['amplitude']:.2f}%\n")
                f.write(f"中枢区间: 下限={centers['center_range']['lower']:.4f}, 上限={centers['center_range']['upper']:.4f}\n")
            else:
                f.write(f"是否横盘: 否\n")
                f.write(f"原因: {centers['reason']}\n")
            
            # 买卖信号
            f.write(f"\n买卖信号:\n")
            buy_signals = result['buy_signals']
            f.write(f"  最强信号: {buy_signals['strongest_signal']}\n")
            f.write(f"  满足买点数量: {buy_signals['satisfied_signals_count']}/4\n")
            f.write(f"\n各信号详情:\n")
            for signal_type, info in buy_signals['signals'].items():
                if info['detected']:
                    f.write(f"  {signal_type}: 存在信号\n")
                    f.write(f"    详细信息: {info['details']}\n")
                else:
                    f.write(f"  {signal_type}: 无信号\n")
            

            
            f.write("\n")
    
    logger.info(f"测试报告已生成: {report_file}")
    logger.info("所有测试完成")

if __name__ == "__main__":
    main()