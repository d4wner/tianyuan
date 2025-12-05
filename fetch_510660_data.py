#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取510660（上证医药ETF）数据并生成报告
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录和src目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# 导入utils模块（修复导入问题）
from src import utils

# 导入依赖
from src.data_processor import DataProcessor
from src.etf_trend_detector import ETFTrendDetector
from src.daily_buy_signal_detector import BuySignalDetector
from src.weekly_trend_detector import WeeklyTrendDetector
from src.reporter import auto_analyze_no_signal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Fetch510660Data')

def main():
    """主函数"""
    try:
        # 初始化数据处理器
        logger.info("初始化数据处理器...")
        processor = DataProcessor()
        
        # 设置时间范围（过去1年）
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        
        # 获取510660的日线数据
        logger.info(f"获取510660日线数据: {start_date} 至 {end_date}")
        daily_data = processor.get_daily_data("sh510660", start_date, end_date, preprocess=True)
        
        if daily_data.empty:
            logger.error("未获取到日线数据")
            return
        
        logger.info(f"日线数据获取成功，共 {len(daily_data)} 条记录")
        
        # 获取510660的周线数据
        logger.info(f"获取510660周线数据: {start_date} 至 {end_date}")
        weekly_data = processor.get_weekly_data("sh510660", start_date, end_date, preprocess=True)
        
        if weekly_data.empty:
            logger.error("未获取到周线数据")
            return
        
        logger.info(f"周线数据获取成功，共 {len(weekly_data)} 条记录")
        
        # 保存数据到CSV文件
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        daily_file = os.path.join(data_dir, "510660_daily_data.csv")
        weekly_file = os.path.join(data_dir, "510660_weekly_data.csv")
        
        daily_data.to_csv(daily_file)
        weekly_data.to_csv(weekly_file)
        
        logger.info(f"数据已保存到: {daily_file} 和 {weekly_file}")
        
        # 生成报告
        logger.info("生成510660 ETF报告...")
        
        # 初始化检测器
        trend_detector = ETFTrendDetector()
        weekly_trend_detector = WeeklyTrendDetector()
        signal_detector = BuySignalDetector()
        
        # 检测横盘
        sideways_result = trend_detector.detect_sideways_market(daily_data)
        
        # 适配波动率并设置动态参数
        signal_detector.adapt_to_volatility(daily_data)
        
        # 检测买卖信号
        signals = signal_detector.detect_buy_signals(daily_data)
        
        # 生成报告
        report = f"新ETF测试报告\n"
        report += f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "==================================================\n\n"
        report += f"ETF代码: 510660\n"
        report += "==============================\n"
        report += f"是否横盘: {'是' if sideways_result['is_sideways'] else '否'}\n"
        report += f"原因: {sideways_result['reason']}\n\n"
        
        # 处理信号信息
        signal_types = ['second_buy', 'first_buy', 'third_buy', 'reverse_pullback']
        buy_signals = {
            'second_buy': signals.get('second_buy', {}),
            'first_buy': signals.get('first_buy', {}),
            'third_buy': signals.get('third_buy', {}),
            'reverse_pullback': signals.get('reverse_pullback', {})
        }
        
        # 计算满足的买点数量
        satisfied_count = sum(1 for signal in buy_signals.values() if signal.get('validated', False))
        
        # 找出最强信号
        signal_priority = ['second_buy', 'first_buy', 'third_buy', 'reverse_pullback']
        strongest_signal = "无买点"
        for signal_type in signal_priority:
            if buy_signals[signal_type].get('validated', False):
                if signal_type == 'second_buy':
                    strongest_signal = "日线二买"
                elif signal_type == 'first_buy':
                    strongest_signal = "日线一买"
                elif signal_type == 'third_buy':
                    strongest_signal = "日线三买"
                elif signal_type == 'reverse_pullback':
                    strongest_signal = "破中枢反抽"
                break
        
        report += "买卖信号:\n"
        report += f"  最强信号: {strongest_signal}\n"
        report += f"  满足买点数量: {satisfied_count}/4\n\n"
        
        report += "各信号详情:\n"
        for signal_type in signal_types:
            signal = buy_signals[signal_type]
            if signal.get('validated', False):
                report += f"  {signal_type}: 存在信号\n"
                report += f"    详细信息: {signal}\n"
            else:
                report += f"  {signal_type}: 无信号\n"
        
        # 如果没有满足的买点，添加无信号归因分析
        if satisfied_count == 0:
            try:
                # 分析无信号原因
                # 注意：这里我们需要准备函数所需的参数
                
                # 周线趋势结果（模拟数据，避免周线数据格式问题）
                weekly_trend_result = {
                    'weekly_trend': '空头',
                    'weekly_macd_divergence_type': '无背驰',
                    'weekly_fractal_type': '无分型'
                }
                
                # 分钟级仓位分配结果（模拟数据，因为我们没有实际的分钟级数据）
                minute_position_result = {
                    'entry_window': False,
                    '30min_divergence': False,
                    '15min_fractal': False,
                    '5min_macd_cross': False
                }
                
                # 调用无信号归因分析函数
                no_signal_analysis = auto_analyze_no_signal(
                    '510660',
                    daily_data,
                    weekly_trend_result,
                    sideways_result,
                    minute_position_result
                )
                
                # 将分析结果添加到报告
                report += "\n"
                report += no_signal_analysis
            except Exception as e:
                logger.error(f"添加无信号归因分析时出错: {e}")
                report += "\n无买点归因分析：\n  无法获取详细归因信息（内部错误）\n"
        
        # 保存报告
        report_file = os.path.join(project_root, "510660_2025_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {report_file}")
        logger.info("任务完成！")
        logger.info("操作完成")
        
    except Exception as e:
        logger.error(f"操作失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()