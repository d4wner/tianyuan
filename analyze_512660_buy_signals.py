#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""512660军工ETF买入信号深度分析脚本"""

import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('512660_Buy_Signal_Analyzer')


class BuySignalDeepAnalyzer:
    """买入信号深度分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.config_dir = '/Users/pingan/tools/trade/tianyuan/config'
        self.output_dir = '/Users/pingan/tools/trade/tianyuan/outputs/analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_latest_report(self) -> Dict:
        """加载最新的验证报告
        
        Returns:
            验证报告字典
        """
        reports_dir = '/Users/pingan/tools/trade/tianyuan/outputs/reports'
        
        # 获取所有验证报告文件
        report_files = []
        for file in os.listdir(reports_dir):
            if file.startswith('512660_validation_report_sep_nov_') and file.endswith('.json'):
                file_path = os.path.join(reports_dir, file)
                report_files.append((file_path, os.path.getmtime(file_path)))
        
        if not report_files:
            logger.error("未找到验证报告文件")
            return {}
        
        # 按修改时间排序，获取最新的报告
        report_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = report_files[0][0]
        
        logger.info(f"使用最新验证报告: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_raw_signals(self) -> List[Dict]:
        """加载原始信号文件
        
        Returns:
            原始信号列表
        """
        # 获取最新的信号文件
        signal_files = [
            '/Users/pingan/tools/trade/tianyuan/outputs/exports/512660_signals_20251126_202448.json',
            '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_enhanced.json',
            '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251125_084914.json',
            '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251124_120616.json'
        ]
        
        # 按修改时间排序，获取最新的文件
        valid_files = []
        for file_path in signal_files:
            if os.path.exists(file_path):
                valid_files.append((file_path, os.path.getmtime(file_path)))
        
        if not valid_files:
            logger.error("未找到有效的信号文件")
            return []
        
        # 按时间排序，获取最新的
        valid_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = valid_files[0][0]
        
        logger.info(f"加载原始信号文件: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_buy_signal_quality(self, buy_signals: List[Dict]) -> Dict:
        """分析买入信号质量
        
        Args:
            buy_signals: 买入信号列表
            
        Returns:
            信号质量分析结果
        """
        if not buy_signals:
            return {}
        
        # 提取信号强度数据
        strengths = [signal['strength'] for signal in buy_signals]
        
        # 按强度分组分析
        low_strength = [s for s in strengths if s < 0.55]
        medium_strength = [s for s in strengths if 0.55 <= s < 0.65]
        high_strength = [s for s in strengths if s >= 0.65]
        
        # 分析信号原因分布
        reasons = {}
        for signal in buy_signals:
            reason = signal.get('reason', '未知')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        # 分析信号时间分布
        month_dist = {9: 0, 10: 0, 11: 0}
        for signal in buy_signals:
            signal_date = datetime.datetime.fromtimestamp(signal['date']/1000)
            if signal_date.month in month_dist:
                month_dist[signal_date.month] += 1
        
        return {
            'total_signals': len(buy_signals),
            'strength_stats': {
                'mean': np.mean(strengths),
                'median': np.median(strengths),
                'min': min(strengths),
                'max': max(strengths),
                'std': np.std(strengths),
                'low_count': len(low_strength),
                'medium_count': len(medium_strength),
                'high_count': len(high_strength)
            },
            'reason_distribution': reasons,
            'monthly_distribution': month_dist,
            'signals': buy_signals
        }
    
    def analyze_signal_effectiveness(self, report: Dict) -> Dict:
        """分析信号有效性
        
        Args:
            report: 验证报告
            
        Returns:
            有效性分析结果
        """
        trades = report.get('trading_results', {}).get('overall', {}).get('trades', [])
        
        if not trades:
            return {
                'has_trades': False,
                'analysis': {}
            }
        
        # 分析交易结果
        profits = [t['profit_percent'] for t in trades]
        win_trades = [t for t in trades if t['is_win']]
        
        # 按信号强度分析交易结果
        strength_profit = {}
        for trade in trades:
            strength_range = f"{int(trade['buy_strength'] * 10) * 0.1}-{int(trade['buy_strength'] * 10) * 0.1 + 0.1}"
            if strength_range not in strength_profit:
                strength_profit[strength_range] = []
            strength_profit[strength_range].append(trade['profit_percent'])
        
        # 计算平均收益
        strength_avg_profit = {}
        for strength_range, profits in strength_profit.items():
            strength_avg_profit[strength_range] = np.mean(profits)
        
        return {
            'has_trades': True,
            'analysis': {
                'total_trades': len(trades),
                'win_rate': len(win_trades) / len(trades) * 100,
                'avg_profit': np.mean(profits),
                'max_profit': max(profits),
                'min_profit': min(profits),
                'profit_std': np.std(profits),
                'strength_profit_correlation': strength_avg_profit,
                'trades': trades
            }
        }
    
    def analyze_price_patterns(self, signals: List[Dict], lookback_days: int = 5, forward_days: int = 10) -> Dict:
        """分析买入信号前后的价格模式
        
        Args:
            signals: 信号列表
            lookback_days: 回看天数
            forward_days: 前瞻性天数
            
        Returns:
            价格模式分析
        """
        # 按日期排序信号
        sorted_signals = sorted(signals, key=lambda x: x['date'])
        
        # 分析信号间的时间间隔
        time_gaps = []
        for i in range(1, len(sorted_signals)):
            prev_date = datetime.datetime.fromtimestamp(sorted_signals[i-1]['date']/1000)
            curr_date = datetime.datetime.fromtimestamp(sorted_signals[i]['date']/1000)
            gap_days = (curr_date - prev_date).days
            time_gaps.append(gap_days)
        
        # 分析价格变化趋势
        price_changes = []
        if len(sorted_signals) > 1:
            for i in range(1, len(sorted_signals)):
                price_change_pct = ((sorted_signals[i]['price'] - sorted_signals[i-1]['price']) / sorted_signals[i-1]['price']) * 100
                price_changes.append(price_change_pct)
        
        # 信号集中度分析（按周）
        weekly_concentration = {}
        for signal in sorted_signals:
            signal_date = datetime.datetime.fromtimestamp(signal['date']/1000)
            week_key = f"{signal_date.year}-W{signal_date.isocalendar()[1]}"
            weekly_concentration[week_key] = weekly_concentration.get(week_key, 0) + 1
        
        return {
            'signal_count': len(sorted_signals),
            'time_gaps': {
                'mean_days': np.mean(time_gaps) if time_gaps else 0,
                'median_days': np.median(time_gaps) if time_gaps else 0,
                'min_days': min(time_gaps) if time_gaps else 0,
                'max_days': max(time_gaps) if time_gaps else 0
            },
            'price_analysis': {
                'avg_price_change_pct': np.mean(price_changes) if price_changes else 0,
                'price_change_std': np.std(price_changes) if price_changes else 0
            },
            'weekly_concentration': weekly_concentration,
            'sorted_signals': sorted_signals
        }
    
    def generate_optimization_recommendations(self, quality_analysis: Dict, effectiveness_analysis: Dict, pattern_analysis: Dict) -> List[str]:
        """生成优化建议
        
        Args:
            quality_analysis: 质量分析结果
            effectiveness_analysis: 有效性分析结果
            pattern_analysis: 模式分析结果
            
        Returns:
            优化建议列表
        """
        recommendations = []
        
        # 基于信号质量的建议
        strength_stats = quality_analysis.get('strength_stats', {})
        if strength_stats.get('mean', 0) < 0.6:
            recommendations.append("建议提高信号强度阈值至0.6以上，以过滤低质量信号")
        
        if strength_stats.get('low_count', 0) > strength_stats.get('high_count', 0):
            recommendations.append("信号强度普遍偏低，建议检查缠论参数设置")
        
        # 基于有效性的建议
        if effectiveness_analysis.get('has_trades', False):
            analysis = effectiveness_analysis['analysis']
            if analysis.get('win_rate', 0) < 60:
                recommendations.append(f"当前胜率为{analysis['win_rate']:.1f}%，建议结合止损策略提高胜率")
            
            if analysis.get('avg_profit', 0) < 1.0:
                recommendations.append(f"平均收益率为{analysis['avg_profit']:.2f}%，建议优化止盈策略")
        
        # 基于模式的建议
        time_gaps = pattern_analysis.get('time_gaps', {})
        if time_gaps.get('mean_days', 0) < 5:
            recommendations.append(f"信号平均间隔{time_gaps['mean_days']:.1f}天，过于频繁，建议增加信号过滤条件")
        elif time_gaps.get('mean_days', 0) > 20:
            recommendations.append(f"信号平均间隔{time_gaps['mean_days']:.1f}天，过于稀少，建议降低信号生成阈值")
        
        # 通用建议
        recommendations.append("建议增加底分型确认天数参数，提高信号可靠性")
        recommendations.append("考虑添加成交量条件，配合价格分型提高信号质量")
        recommendations.append("建议实现不同市场环境下的参数自适应调整")
        recommendations.append("考虑增加MACD等技术指标作为辅助确认条件")
        
        return recommendations
    
    def generate_visualizations(self, quality_analysis: Dict, effectiveness_analysis: Dict, pattern_analysis: Dict):
        """生成可视化图表
        
        Args:
            quality_analysis: 质量分析结果
            effectiveness_analysis: 有效性分析结果
            pattern_analysis: 模式分析结果
        """
        # 创建图表目录
        chart_dir = os.path.join(self.output_dir, 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 1. 信号强度分布图
        plt.figure(figsize=(10, 6))
        strengths = [signal['strength'] for signal in quality_analysis['signals']]
        plt.hist(strengths, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('512660买入信号强度分布')
        plt.xlabel('信号强度')
        plt.ylabel('信号数量')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(chart_dir, 'signal_strength_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 月度信号分布图
        plt.figure(figsize=(10, 6))
        month_dist = quality_analysis['monthly_distribution']
        months = ['9月', '10月', '11月']
        counts = [month_dist[9], month_dist[10], month_dist[11]]
        plt.bar(months, counts, color=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title('512660买入信号月度分布')
        plt.xlabel('月份')
        plt.ylabel('信号数量')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(chart_dir, 'monthly_signal_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 信号原因分布饼图
        plt.figure(figsize=(10, 8))
        reasons = quality_analysis['reason_distribution']
        plt.pie(reasons.values(), labels=reasons.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('512660买入信号原因分布')
        plt.axis('equal')  # 保证饼图是圆的
        plt.savefig(os.path.join(chart_dir, 'signal_reason_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 交易结果分析（如果有交易数据）
        if effectiveness_analysis.get('has_trades', False):
            trades = effectiveness_analysis['analysis']['trades']
            plt.figure(figsize=(12, 6))
            
            # 收益率柱状图
            profits = [t['profit_percent'] for t in trades]
            trade_dates = [t['buy_date'] for t in trades]
            
            colors = ['green' if p > 0 else 'red' for p in profits]
            plt.bar(trade_dates, profits, color=colors)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('512660交易收益率分析')
            plt.xlabel('买入日期')
            plt.ylabel('收益率 (%)')
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, 'trade_profit_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. 信号时间间隔分析
        plt.figure(figsize=(10, 6))
        if pattern_analysis['time_gaps']['mean_days'] > 0:
            gaps = pattern_analysis['time_gaps']
            gap_data = [gaps['min_days'], gaps['median_days'], gaps['mean_days'], gaps['max_days']]
            labels = ['最小间隔', '中位数间隔', '平均间隔', '最大间隔']
            
            plt.bar(labels, gap_data, color=['#ff9999', '#ffcc99', '#66b3ff', '#99ff99'])
            plt.title('512660买入信号时间间隔分析 (天)')
            plt.ylabel('天数')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(chart_dir, 'signal_time_gap_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"可视化图表已保存至: {chart_dir}")
    
    def generate_detailed_analysis(self) -> Dict:
        """生成详细分析报告
        
        Returns:
            详细分析报告
        """
        # 加载验证报告
        report = self.load_latest_report()
        if not report:
            return {}
        
        # 加载原始信号
        raw_signals = self.load_raw_signals()
        
        # 获取9-11月的信号
        start_date = datetime.datetime(2025, 9, 1)
        end_date = datetime.datetime(2025, 11, 30)
        
        sep_nov_signals = []
        for signal in raw_signals:
            signal_date = datetime.datetime.fromtimestamp(signal['date']/1000)
            if start_date <= signal_date <= end_date:
                sep_nov_signals.append(signal)
        
        # 筛选买入信号
        buy_signals = [s for s in sep_nov_signals if s['type'] == 'buy']
        
        # 进行各项分析
        quality_analysis = self.analyze_buy_signal_quality(buy_signals)
        effectiveness_analysis = self.analyze_signal_effectiveness(report)
        pattern_analysis = self.analyze_price_patterns(buy_signals)
        
        # 生成优化建议
        recommendations = self.generate_optimization_recommendations(
            quality_analysis, effectiveness_analysis, pattern_analysis
        )
        
        # 生成可视化
        self.generate_visualizations(quality_analysis, effectiveness_analysis, pattern_analysis)
        
        # 构建完整分析报告
        detailed_report = {
            'analysis_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_period': '2025年9月-11月',
            'signal_quality': quality_analysis,
            'signal_effectiveness': effectiveness_analysis,
            'price_patterns': pattern_analysis,
            'optimization_recommendations': recommendations,
            'summary': self._generate_summary(
                quality_analysis, effectiveness_analysis, pattern_analysis
            )
        }
        
        return detailed_report
    
    def _generate_summary(self, quality_analysis: Dict, effectiveness_analysis: Dict, pattern_analysis: Dict) -> str:
        """生成分析摘要
        
        Args:
            quality_analysis: 质量分析结果
            effectiveness_analysis: 有效性分析结果
            pattern_analysis: 模式分析结果
            
        Returns:
            分析摘要文本
        """
        summary = []
        
        # 信号质量摘要
        summary.append(f"信号质量分析：")
        summary.append(f"- 共识别{quality_analysis['total_signals']}个买入信号")
        strength_stats = quality_analysis['strength_stats']
        summary.append(f"- 信号强度均值：{strength_stats['mean']:.3f}，中位数：{strength_stats['median']:.3f}")
        summary.append(f"- 高强度信号(≥0.65)：{strength_stats['high_count']}个，中等强度信号(0.55-0.65)：{strength_stats['medium_count']}个")
        
        # 有效性摘要
        if effectiveness_analysis.get('has_trades', False):
            analysis = effectiveness_analysis['analysis']
            summary.append(f"\n交易有效性分析：")
            summary.append(f"- 总交易次数：{analysis['total_trades']}次，胜率：{analysis['win_rate']:.1f}%")
            summary.append(f"- 平均收益率：{analysis['avg_profit']:.2f}%，最大收益率：{analysis['max_profit']:.2f}%")
        
        # 模式分析摘要
        summary.append(f"\n价格模式分析：")
        time_gaps = pattern_analysis['time_gaps']
        summary.append(f"- 信号平均间隔：{time_gaps['mean_days']:.1f}天")
        
        # 核心结论
        if strength_stats['mean'] >= 0.6 and (not effectiveness_analysis.get('has_trades', False) or effectiveness_analysis['analysis']['win_rate'] >= 70):
            summary.append(f"\n核心结论：当前信号系统表现良好，建议保持并进行小幅优化。")
        else:
            summary.append(f"\n核心结论：信号系统需要进一步优化，重点关注信号强度和交易胜率。")
        
        return "\n".join(summary)
    
    def display_analysis_report(self, detailed_report: Dict):
        """显示分析报告
        
        Args:
            detailed_report: 详细分析报告
        """
        print("=" * 120)
        print(f"512660军工ETF买入信号深度分析报告")
        print(f"分析周期: {detailed_report['analysis_period']}")
        print(f"生成时间: {detailed_report['analysis_time']}")
        print("=" * 120)
        
        # 打印摘要
        print("\n📊 分析摘要:")
        print("-" * 80)
        print(detailed_report['summary'])
        
        # 信号质量详细分析
        print("\n🎯 信号质量详细分析:")
        print("-" * 80)
        quality = detailed_report['signal_quality']
        print(f"信号总数: {quality['total_signals']}个")
        print(f"信号强度统计:")
        print(f"  - 均值: {quality['strength_stats']['mean']:.3f}")
        print(f"  - 中位数: {quality['strength_stats']['median']:.3f}")
        print(f"  - 最小值: {quality['strength_stats']['min']:.3f}")
        print(f"  - 最大值: {quality['strength_stats']['max']:.3f}")
        print(f"  - 标准差: {quality['strength_stats']['std']:.3f}")
        
        print(f"\n信号强度分布:")
        print(f"  - 高强度信号(≥0.65): {quality['strength_stats']['high_count']}个")
        print(f"  - 中等强度信号(0.55-0.65): {quality['strength_stats']['medium_count']}个")
        print(f"  - 低强度信号(<0.55): {quality['strength_stats']['low_count']}个")
        
        print(f"\n月度信号分布:")
        print(f"  - 9月: {quality['monthly_distribution'][9]}个")
        print(f"  - 10月: {quality['monthly_distribution'][10]}个")
        print(f"  - 11月: {quality['monthly_distribution'][11]}个")
        
        # 交易有效性分析
        print("\n💰 交易有效性分析:")
        print("-" * 80)
        effectiveness = detailed_report['signal_effectiveness']
        if effectiveness.get('has_trades', False):
            analysis = effectiveness['analysis']
            print(f"总交易次数: {analysis['total_trades']}次")
            print(f"胜率: {analysis['win_rate']:.1f}%")
            print(f"平均收益率: {analysis['avg_profit']:.2f}%")
            print(f"最大收益率: {analysis['max_profit']:.2f}%")
            print(f"最小收益率: {analysis['min_profit']:.2f}%")
            print(f"收益率标准差: {analysis['profit_std']:.2f}%")
            
            print(f"\n按信号强度分组的平均收益:")
            for strength_range, avg_profit in analysis['strength_profit_correlation'].items():
                print(f"  - 强度{strength_range}: {avg_profit:.2f}%")
        else:
            print("无交易数据可供分析")
        
        # 价格模式分析
        print("\n📈 价格模式分析:")
        print("-" * 80)
        patterns = detailed_report['price_patterns']
        print(f"信号时间间隔统计:")
        print(f"  - 平均间隔: {patterns['time_gaps']['mean_days']:.1f}天")
        print(f"  - 中位数间隔: {patterns['time_gaps']['median_days']:.1f}天")
        print(f"  - 最小间隔: {patterns['time_gaps']['min_days']}天")
        print(f"  - 最大间隔: {patterns['time_gaps']['max_days']}天")
        
        # 优化建议
        print("\n💡 优化建议:")
        print("-" * 80)
        for i, recommendation in enumerate(detailed_report['optimization_recommendations'], 1):
            print(f"{i}. {recommendation}")
        
        print("\n" + "=" * 120)
        print("📊 图表生成信息:")
        print("已生成以下可视化图表:")
        print("1. 信号强度分布图")
        print("2. 月度信号分布图")
        print("3. 信号原因分布饼图")
        print("4. 交易收益率分析图")
        print("5. 信号时间间隔分析图")
        print("所有图表已保存至outputs/analysis/charts目录")
        print("=" * 120)
    
    def save_analysis_report(self, detailed_report: Dict):
        """保存分析报告到文件
        
        Args:
            detailed_report: 详细分析报告
        """
        output_file = f"{self.output_dir}/512660_buy_signal_detailed_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细分析报告已保存到: {output_file}")
        print(f"\n详细分析报告已保存到: {output_file}")
    
    def run_analysis(self):
        """运行完整的分析流程"""
        logger.info("开始对512660买入信号进行深度分析...")
        
        # 生成详细分析报告
        detailed_report = self.generate_detailed_analysis()
        
        if not detailed_report:
            logger.error("无法生成分析报告")
            return
        
        # 显示分析报告
        self.display_analysis_report(detailed_report)
        
        # 保存分析报告
        self.save_analysis_report(detailed_report)
        
        logger.info("分析完成!")


def main():
    """主函数"""
    analyzer = BuySignalDeepAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()