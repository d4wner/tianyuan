#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新功能：数据缺失补全、信号矛盾归因、评分体系统一、个性化建议生成
"""

import sys
import os
import pandas as pd
from typing import Dict, Any

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入reporter模块
from src.reporter import (
    auto_complete_missing_data,
    signal_conflict_analyzer,
    unify_score_system,
    generate_personalized_suggestions,
    auto_analyze_no_signal
)

# 测试数据
def create_test_data():
    """创建测试数据"""
    # 创建模拟的日线数据
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60)
    df = pd.DataFrame({
        'date': dates,
        'open': [1.0 + i * 0.01 for i in range(60)],
        'high': [1.05 + i * 0.01 for i in range(60)],
        'low': [0.95 + i * 0.01 for i in range(60)],
        'close': [1.0 + i * 0.01 for i in range(60)],
        'volume': [1000000 + i * 1000 for i in range(60)]
    })
    
    # 创建模拟的周线趋势结果（包含缺失数据的情况）
    weekly_trend_result_512660 = {
        'weekly_trend': '未知',  # 模拟周线数据缺失
        'weekly_macd_divergence_type': '底背驰',
        'divergence_strength': 69.47,
        'macd_line_position': '零轴上方',
        'macd_histogram_amplitude': 0.002957
    }
    
    weekly_trend_result_510660 = {
        'weekly_trend': '非多头',
        'weekly_macd_divergence_type': '无背驰',
        'divergence_strength': 0
    }
    
    sideways_result = {
        'amplitude': 7.89,
        'center_range': {'lower': 2.0950, 'upper': 2.25}
    }
    
    minute_position_result = {
        'entry_window': False,
        '30min_divergence': False,
        '15min_fractal': False,
        '5min_macd_cross': False
    }
    
    return {
        'df': df,
        'weekly_trend_result_512660': weekly_trend_result_512660,
        'weekly_trend_result_510660': weekly_trend_result_510660,
        'sideways_result': sideways_result,
        'minute_position_result': minute_position_result
    }

def test_all_features():
    """测试所有新功能"""
    print("=== 测试新功能 ===\n")
    
    # 获取测试数据
    test_data = create_test_data()
    df = test_data['df']
    weekly_trend_result_512660 = test_data['weekly_trend_result_512660']
    weekly_trend_result_510660 = test_data['weekly_trend_result_510660']
    sideways_result = test_data['sideways_result']
    minute_position_result = test_data['minute_position_result']
    
    print("1. 测试512660自动分析无信号功能（包含所有新特性）：")
    print("=" * 60)
    
    # 调用auto_analyze_no_signal函数，该函数已集成所有新功能
    report_512660 = auto_analyze_no_signal(
        symbol='512660',
        df=df,
        weekly_trend_result=weekly_trend_result_512660,
        sideways_result=sideways_result,
        minute_position_result=minute_position_result
    )
    
    print(report_512660)
    
    print("2. 测试510660自动分析无信号功能：")
    print("=" * 60)
    
    report_510660 = auto_analyze_no_signal(
        symbol='510660',
        df=df,
        weekly_trend_result=weekly_trend_result_510660,
        sideways_result=sideways_result,
        minute_position_result=minute_position_result
    )
    
    print(report_510660)
    
    print("3. 单独测试信号矛盾归因功能：")
    print("=" * 60)
    
    conflict_analysis = signal_conflict_analyzer(
        symbol='512660',
        weekly_trend_result=weekly_trend_result_512660,
        daily_buy_result={'strongest_signal': '无买点', 'has_buy_signal': False},
        macd_result={'divergence_type': '底部背离', 'divergence_strength': 69.47}
    )
    
    print(conflict_analysis)
    
    print("4. 单独测试评分体系统一功能：")
    print("=" * 60)
    
    unified_score = unify_score_system(
        symbol="512660",
        weekly_trend_result=weekly_trend_result_512660,
        daily_buy_result={'strongest_signal': '无买点', 'has_buy_signal': False},
        macd_result={'divergence_type': '底部背离', 'divergence_strength': 69.47},
        signal_result={}
    )
    print(f"最终评分: {unified_score['final_score']:.2f}分")
    print(f"置信度等级: {unified_score['confidence_level']}")
    
    print("\n5. 单独测试个性化建议生成功能：")
    print("=" * 60)
    
    suggestions_512660 = generate_personalized_suggestions(
        symbol='512660',
        weekly_trend_result=weekly_trend_result_512660,
        daily_buy_result={'strongest_signal': '无买点', 'has_buy_signal': False},
        macd_result={'divergence_type': '底部背离', 'divergence_strength': 69.47},
        signal_result={}
    )
    print(suggestions_512660)
    
    print("\n6. 单独测试数据缺失补全功能：")
    print("=" * 60)
    
    result = auto_complete_missing_data(
        symbol='512660',
        weekly_trend_result=weekly_trend_result_512660,
        daily_buy_result={'strongest_signal': '无买点'},
        signal_result={}
    )
    
    print(f"数据补全结果：")
    print(f"周线趋势结果：{result['weekly_trend_result']}")
    print(f"日线买点结果：{result['daily_buy_result']}")
    print(f"信号结果：{result['signal_result']}")

if __name__ == "__main__":
    test_all_features()