#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周线趋势检测器测试脚本

本脚本用于测试周线背驰/分型过滤优化功能的正确性，
包括背驰过滤条件、分型过滤条件和过滤开关功能的测试。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入需要测试的模块
from weekly_trend_detector import WeeklyTrendDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_weekly_data():
    """创建模拟的周线数据
    
    Returns:
        模拟的周线数据框
    """
    # 创建日期索引
    dates = [datetime.now() - timedelta(weeks=i) for i in range(30)][::-1]
    
    # 创建基础数据
    data = {
        'date': dates,
        'open': np.random.rand(30) * 10 + 100,
        'high': np.random.rand(30) * 5 + 102,
        'low': np.random.rand(30) * 5 + 98,
        'close': np.random.rand(30) * 10 + 100,
        'volume': np.random.rand(30) * 1000000 + 500000
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # 确保数据有合理的高低点
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(30) * 2
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(30) * 2
    
    return df


def create_bullish_fractal_data():
    """创建包含底分型的周线数据
    
    Returns:
        包含底分型的周线数据框
    """
    df = create_mock_weekly_data()
    
    # 修改最近5根K线，创建底分型
    last5 = df.tail(5).copy()
    
    # 左阴
    last5.iloc[1, last5.columns.get_loc('open')] = 105
    last5.iloc[1, last5.columns.get_loc('close')] = 100
    
    # 中低
    middle_low = min(last5['low']) - 2
    last5.iloc[2, last5.columns.get_loc('low')] = middle_low
    last5.iloc[2, last5.columns.get_loc('open')] = middle_low + 0.5
    last5.iloc[2, last5.columns.get_loc('close')] = middle_low + 1.0
    
    # 右阳，涨幅≥2%
    right_open = last5.iloc[2]['close']
    right_close = right_open * 1.03  # 3%涨幅
    last5.iloc[3, last5.columns.get_loc('open')] = right_open
    last5.iloc[3, last5.columns.get_loc('close')] = right_close
    last5.iloc[3, last5.columns.get_loc('high')] = right_close + 0.5
    last5.iloc[3, last5.columns.get_loc('low')] = right_open - 0.5
    
    # 更新原始数据
    df.update(last5)
    
    return df


def create_bearish_fractal_data():
    """创建包含顶分型的周线数据
    
    Returns:
        包含顶分型的周线数据框
    """
    df = create_mock_weekly_data()
    
    # 修改最近5根K线，创建顶分型
    last5 = df.tail(5).copy()
    
    # 左阳
    last5.iloc[1, last5.columns.get_loc('open')] = 100
    last5.iloc[1, last5.columns.get_loc('close')] = 105
    
    # 中高
    middle_high = max(last5['high']) + 2
    last5.iloc[2, last5.columns.get_loc('high')] = middle_high
    last5.iloc[2, last5.columns.get_loc('open')] = middle_high - 1.0
    last5.iloc[2, last5.columns.get_loc('close')] = middle_high - 0.5
    
    # 右阴，跌幅≥2%
    right_open = last5.iloc[2]['close']
    right_close = right_open * 0.97  # 3%跌幅
    last5.iloc[3, last5.columns.get_loc('open')] = right_open
    last5.iloc[3, last5.columns.get_loc('close')] = right_close
    last5.iloc[3, last5.columns.get_loc('high')] = right_open + 0.5
    last5.iloc[3, last5.columns.get_loc('low')] = right_close - 0.5
    
    # 更新原始数据
    df.update(last5)
    
    return df


def create_macd_divergence_data():
    """创建包含MACD底背驰的周线数据
    
    Returns:
        包含MACD底背驰的周线数据框
    """
    df = create_mock_weekly_data()
    
    # 创建价格新低
    df.iloc[-5:, df.columns.get_loc('low')] = np.linspace(100, 95, 5)  # 价格逐渐降低
    df.iloc[-5:, df.columns.get_loc('close')] = df.iloc[-5:, df.columns.get_loc('low')] + np.random.rand(5) * 2
    df.iloc[-5:, df.columns.get_loc('open')] = df.iloc[-5:, df.columns.get_loc('close')] - np.random.rand(5) * 1
    df.iloc[-5:, df.columns.get_loc('high')] = df.iloc[-5:, df.columns.get_loc('close')] + np.random.rand(5) * 2
    
    # 确保成交量满足波动等级阈值
    avg_volume = df['volume'].mean()
    df.iloc[-1, df.columns.get_loc('volume')] = avg_volume * 1.1  # 高于均值
    
    return df


def test_fractal_filter():
    """测试分型过滤功能"""
    logger.info("=== 测试分型过滤功能 ===")
    
    detector = WeeklyTrendDetector()
    
    # 测试底分型过滤
    logger.info("1. 测试底分型过滤功能")
    bullish_data = create_bullish_fractal_data()
    
    # 开启过滤开关
    detector.filter_enabled['fractal'] = True
    result_with_filter = detector.calc_weekly_fractal_confidence(bullish_data)
    logger.info(f"底分型过滤开启时的结果: {result_with_filter}")
    
    # 关闭过滤开关
    detector.filter_enabled['fractal'] = False
    result_without_filter = detector.calc_weekly_fractal_confidence(bullish_data)
    logger.info(f"底分型过滤关闭时的结果: {result_without_filter}")
    
    # 测试顶分型过滤
    logger.info("2. 测试顶分型过滤功能")
    bearish_data = create_bearish_fractal_data()
    
    # 开启过滤开关
    detector.filter_enabled['fractal'] = True
    result_with_filter = detector.calc_weekly_fractal_confidence(bearish_data)
    logger.info(f"顶分型过滤开启时的结果: {result_with_filter}")
    
    # 关闭过滤开关
    detector.filter_enabled['fractal'] = False
    result_without_filter = detector.calc_weekly_fractal_confidence(bearish_data)
    logger.info(f"顶分型过滤关闭时的结果: {result_without_filter}")


def test_macd_divergence_filter():
    """测试MACD背驰过滤功能"""
    logger.info("=== 测试MACD背驰过滤功能 ===")
    
    detector = WeeklyTrendDetector()
    divergence_data = create_macd_divergence_data()
    
    # 计算MACD指标
    divergence_data = detector._calculate_macd(divergence_data)
    
    # 开启过滤开关
    detector.filter_enabled['macd_divergence'] = True
    result_with_filter = detector.calc_weekly_macd_divergence_confidence(divergence_data)
    logger.info(f"背驰过滤开启时的结果: {result_with_filter}")
    
    # 关闭过滤开关
    detector.filter_enabled['macd_divergence'] = False
    result_without_filter = detector.calc_weekly_macd_divergence_confidence(divergence_data)
    logger.info(f"背驰过滤关闭时的结果: {result_without_filter}")


def test_filter_switch():
    """测试过滤开关功能"""
    logger.info("=== 测试过滤开关功能 ===")
    
    detector = WeeklyTrendDetector()
    
    # 检查默认状态
    logger.info(f"默认背驰过滤开关状态: {detector.filter_enabled['macd_divergence']}")
    logger.info(f"默认分型过滤开关状态: {detector.filter_enabled['fractal']}")
    
    # 测试开关切换
    detector.filter_enabled['macd_divergence'] = False
    detector.filter_enabled['fractal'] = False
    logger.info(f"关闭后背驰过滤开关状态: {detector.filter_enabled['macd_divergence']}")
    logger.info(f"关闭后分型过滤开关状态: {detector.filter_enabled['fractal']}")
    
    detector.filter_enabled['macd_divergence'] = True
    detector.filter_enabled['fractal'] = True
    logger.info(f"开启后背驰过滤开关状态: {detector.filter_enabled['macd_divergence']}")
    logger.info(f"开启后分型过滤开关状态: {detector.filter_enabled['fractal']}")


def test_central_border_detection():
    """测试中枢边缘检测功能"""
    logger.info("=== 测试中枢边缘检测功能 ===")
    
    detector = WeeklyTrendDetector()
    test_data = create_mock_weekly_data()
    
    central_high, central_low = detector._detect_central_border(test_data)
    logger.info(f"检测到的中枢上沿: {central_high:.2f}, 中枢下沿: {central_low:.2f}")
    logger.info(f"中枢区间大小: {central_high - central_low:.2f}")


def test_volatility_level():
    """测试波动等级计算功能"""
    logger.info("=== 测试波动等级计算功能 ===")
    
    detector = WeeklyTrendDetector()
    test_data = create_mock_weekly_data()
    
    volatility_level = detector._get_volatility_level(test_data)
    logger.info(f"计算得到的波动等级: {volatility_level}")


if __name__ == "__main__":
    # 运行所有测试
    test_filter_switch()
    test_central_border_detection()
    test_volatility_level()
    test_fractal_filter()
    test_macd_divergence_filter()
    
    logger.info("所有测试完成！")