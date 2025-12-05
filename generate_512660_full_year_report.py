#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512660军工ETF 2025年全年交易下单检测详细报告生成脚本

该脚本用于生成512660在2025年全年的交易下单检测详细报告，
包括信号质量、交易有效性、价格模式分析和优化建议等内容。
"""

import json
import os
import pandas as pd
from datetime import datetime

def generate_full_year_report():
    """
    生成512660在2025年全年的交易下单检测详细报告
    """
    # 读取现有的分析数据
    analysis_file = "/Users/pingan/tools/trade/tianyuan/outputs/analysis/512660_buy_signal_detailed_analysis_20251204_085648.json"
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except Exception as e:
        print(f"读取分析数据失败: {e}")
        return
    
    # 创建报告内容
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("512660军工ETF 2025年全年交易下单检测详细报告")
    report_lines.append("="*80)
    report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据来源: {analysis_file}")
    report_lines.append("="*80)
    report_lines.append("\n")
    
    # 一、信号质量分析
    report_lines.append("1. 信号质量分析")
    report_lines.append("-"*40)
    
    signal_quality = analysis_data.get('signal_quality', {})
    report_lines.append(f"总信号数: {signal_quality.get('total_signals', 0)}个")
    
    strength_stats = signal_quality.get('strength_stats', {})
    report_lines.append(f"信号强度均值: {strength_stats.get('mean', 0):.3f}")
    report_lines.append(f"信号强度中位数: {strength_stats.get('median', 0):.3f}")
    report_lines.append(f"信号强度范围: {strength_stats.get('min', 0):.3f} - {strength_stats.get('max', 0):.3f}")
    
    report_lines.append("\n信号强度分布:")
    report_lines.append(f"  - 高强度信号(≥0.65): {strength_stats.get('high_count', 0)}个")
    report_lines.append(f"  - 中等强度信号(0.55-0.65): {strength_stats.get('medium_count', 0)}个")
    report_lines.append(f"  - 低强度信号(<0.55): {strength_stats.get('low_count', 0)}个")
    
    report_lines.append("\n信号原因分布:")
    reason_distribution = signal_quality.get('reason_distribution', {})
    for reason, count in reason_distribution.items():
        report_lines.append(f"  - {reason}: {count}个")
    
    report_lines.append("\n月度信号分布:")
    monthly_distribution = signal_quality.get('monthly_distribution', {})
    months = {"9": "九月", "10": "十月", "11": "十一月"}
    for month, count in monthly_distribution.items():
        report_lines.append(f"  - {months.get(month, f'月{month}')}: {count}个")
    
    report_lines.append("\n信号详情:")
    signals = signal_quality.get('signals', [])
    for i, signal in enumerate(signals, 1):
        signal_date = pd.to_datetime(signal['date'], unit='ms').strftime('%Y-%m-%d')
        report_lines.append(f"  {i}. 日期: {signal_date}, 价格: {signal['price']}, 强度: {signal['strength']:.2f}, 原因: {signal['reason']}")
    
    report_lines.append("\n")
    
    # 二、交易有效性分析
    report_lines.append("2. 交易有效性分析")
    report_lines.append("-"*40)
    
    signal_effectiveness = analysis_data.get('signal_effectiveness', {})
    report_lines.append(f"是否有实际交易: {'是' if signal_effectiveness.get('has_trades', False) else '否'}")
    
    analysis = signal_effectiveness.get('analysis', {})
    report_lines.append(f"总交易次数: {analysis.get('total_trades', 0)}次")
    report_lines.append(f"胜率: {analysis.get('win_rate', 0):.1f}%")
    report_lines.append(f"平均收益率: {analysis.get('avg_profit', 0):.3f}%")
    report_lines.append(f"最大收益率: {analysis.get('max_profit', 0):.3f}%")
    report_lines.append(f"最小收益率: {analysis.get('min_profit', 0):.3f}%")
    
    report_lines.append("\n强度-收益相关性:")
    strength_correlation = analysis.get('strength_profit_correlation', {})
    for strength_range, profit in strength_correlation.items():
        report_lines.append(f"  - 强度范围 {strength_range}: 平均收益 {profit:.3f}%")
    
    report_lines.append("\n交易详情:")
    trades = analysis.get('trades', [])
    for i, trade in enumerate(trades, 1):
        report_lines.append(f"  {i}. 买入日期: {trade['buy_date']}, 买入价格: {trade['buy_price']}, 买入强度: {trade['buy_strength']:.2f}")
        report_lines.append(f"     卖出日期: {trade['sell_date']}, 卖出价格: {trade['sell_price']}, 卖出强度: {trade['sell_strength']:.2f}")
        report_lines.append(f"     收益率: {trade['profit_percent']:.3f}%, 结果: {'盈利' if trade['is_win'] else '亏损'}")
    
    report_lines.append("\n")
    
    # 三、价格模式分析
    report_lines.append("3. 价格模式分析")
    report_lines.append("-"*40)
    
    price_patterns = analysis_data.get('price_patterns', {})
    report_lines.append(f"信号数量: {price_patterns.get('signal_count', 0)}个")
    
    time_gaps = price_patterns.get('time_gaps', {})
    report_lines.append(f"信号时间间隔均值: {time_gaps.get('mean_days', 0):.1f}天")
    report_lines.append(f"信号时间间隔中位数: {time_gaps.get('median_days', 0):.1f}天")
    report_lines.append(f"信号时间间隔范围: {time_gaps.get('min_days', 0)} - {time_gaps.get('max_days', 0)}天")
    
    price_analysis = price_patterns.get('price_analysis', {})
    report_lines.append(f"平均价格变动百分比: {price_analysis.get('avg_price_change_pct', 0):.3f}%")
    report_lines.append(f"价格变动标准差: {price_analysis.get('price_change_std', 0):.3f}%")
    
    report_lines.append("\n周度信号集中度:")
    weekly_concentration = price_patterns.get('weekly_concentration', {})
    for week, count in weekly_concentration.items():
        report_lines.append(f"  - {week}: {count}个信号")
    
    report_lines.append("\n")
    
    # 四、优化建议
    report_lines.append("4. 优化建议")
    report_lines.append("-"*40)
    
    recommendations = analysis_data.get('optimization_recommendations', [])
    for i, recommendation in enumerate(recommendations, 1):
        report_lines.append(f"  {i}. {recommendation}")
    
    report_lines.append("\n")
    
    # 五、总结
    report_lines.append("5. 总结")
    report_lines.append("-"*40)
    
    summary = analysis_data.get('summary', '')
    report_lines.append(summary)
    
    report_lines.append("\n")
    report_lines.append("="*80)
    report_lines.append("报告结束")
    report_lines.append("="*80)
    
    # 生成报告文本
    report_content = '\n'.join(report_lines)
    
    # 保存报告
    report_file = "/Users/pingan/tools/trade/tianyuan/outputs/reports/512660_2025_full_year_trading_detection_report.txt"
    report_dir = os.path.dirname(report_file)
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"报告已生成: {report_file}")
    print("\n报告预览:")
    print('\n'.join(report_lines[:50]))  # 预览前50行

if __name__ == "__main__":
    generate_full_year_report()