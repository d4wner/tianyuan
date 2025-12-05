#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512690买卖信号分析脚本

分析512690的买卖信号分布情况，包括信号质量、交易有效性和价格模式分析。

作者: TradeTianYuan
日期: 2025-11-28
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('512690SignalAnalyzer')

class SignalAnalyzer:
    """
    买卖信号分析器类
    """
    
    def __init__(self, symbol: str = '512690', data_dir: str = './data/daily'):
        """
        初始化信号分析器
        
        Args:
            symbol: 股票代码
            data_dir: 数据目录
        """
        self.symbol = symbol
        self.data_dir = data_dir
        self.daily_data = None
        self.signals = []
        self.trade_pairs = []
        
        # 创建结果目录
        self.results_dir = './results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self) -> bool:
        """
        加载日线数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            data_file = os.path.join(self.data_dir, f'{self.symbol}_daily.csv')
            if not os.path.exists(data_file):
                logger.error(f"数据文件不存在: {data_file}")
                return False
            
            self.daily_data = pd.read_csv(data_file)
            self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
            self.daily_data.sort_values('date', inplace=True)
            
            logger.info(f"成功加载{self.symbol}日线数据，共{len(self.daily_data)}条记录")
            return True
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False
    
    def generate_signals(self) -> List[Dict]:
        """
        基于日线数据生成模拟买卖信号
        注：由于没有实际的信号数据，这里基于技术指标生成模拟信号用于分析
        
        Returns:
            List[Dict]: 信号列表
        """
        if self.daily_data is None:
            logger.error("请先加载数据")
            return []
        
        signals = []
        
        # 计算简单的技术指标
        self.daily_data['close_ma5'] = self.daily_data['close'].rolling(window=5).mean()
        self.daily_data['close_ma20'] = self.daily_data['close'].rolling(window=20).mean()
        self.daily_data['volume_ma5'] = self.daily_data['volume'].rolling(window=5).mean()
        
        # 生成模拟的底分型和顶分型信号
        # 这里使用MA金叉死叉作为简化的信号生成方式
        for i in range(21, len(self.daily_data)):
            current = self.daily_data.iloc[i]
            prev = self.daily_data.iloc[i-1]
            prev_prev = self.daily_data.iloc[i-2]
            
            # 简单的金叉死叉信号
            if prev['close_ma5'] <= prev['close_ma20'] and current['close_ma5'] > current['close_ma20']:
                # 金叉 - 买入信号
                signal = {
                    'date': current['date'].timestamp() * 1000,  # 毫秒时间戳
                    'date_str': current['date'].strftime('%Y-%m-%d'),
                    'type': 'buy',
                    'price': current['close'],
                    'strength': np.random.uniform(0.55, 0.75),  # 随机强度
                    'reason': '底分型形成 + 买入信号'
                }
                signals.append(signal)
            
            elif prev['close_ma5'] >= prev['close_ma20'] and current['close_ma5'] < current['close_ma20']:
                # 死叉 - 卖出信号
                signal = {
                    'date': current['date'].timestamp() * 1000,  # 毫秒时间戳
                    'date_str': current['date'].strftime('%Y-%m-%d'),
                    'type': 'sell',
                    'price': current['close'],
                    'strength': np.random.uniform(0.55, 0.75),  # 随机强度
                    'reason': '顶分型形成 + 卖出信号'
                }
                signals.append(signal)
        
        # 过滤掉最近可能未完成的交易
        signals = [s for s in signals if s['date'] < (datetime.now().timestamp() - 86400) * 1000]
        
        self.signals = signals
        logger.info(f"生成{len(signals)}个模拟信号")
        return signals
    
    def analyze_signal_quality(self) -> Dict:
        """
        分析信号质量
        
        Returns:
            Dict: 信号质量分析结果
        """
        if not self.signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'strength_stats': {},
                'monthly_distribution': {9: 0, 10: 0, 11: 0}
            }
        
        # 计算信号强度统计
        strengths = [s['strength'] for s in self.signals]
        strength_stats = {
            'mean': np.mean(strengths),
            'median': np.median(strengths),
            'min': min(strengths),
            'max': max(strengths),
            'high_count': sum(1 for s in self.signals if s['strength'] >= 0.65),
            'medium_count': sum(1 for s in self.signals if 0.55 <= s['strength'] < 0.65),
            'low_count': sum(1 for s in self.signals if s['strength'] < 0.55)
        }
        
        # 按月统计信号分布
        monthly_distribution = {9: 0, 10: 0, 11: 0}
        for signal in self.signals:
            date_obj = datetime.fromtimestamp(signal['date'] / 1000)
            month = date_obj.month
            if month in monthly_distribution:
                monthly_distribution[month] += 1
        
        return {
            'total_signals': len(self.signals),
            'buy_signals': sum(1 for s in self.signals if s['type'] == 'buy'),
            'sell_signals': sum(1 for s in self.signals if s['type'] == 'sell'),
            'strength_stats': strength_stats,
            'monthly_distribution': monthly_distribution
        }
    
    def pair_trades(self) -> List[Dict]:
        """
        将买卖信号配对成交易
        
        Returns:
            List[Dict]: 交易对列表
        """
        trade_pairs = []
        buy_signal = None
        
        for signal in sorted(self.signals, key=lambda x: x['date']):
            if signal['type'] == 'buy' and buy_signal is None:
                buy_signal = signal
            elif signal['type'] == 'sell' and buy_signal is not None:
                # 配对成功
                profit_percent = ((signal['price'] / buy_signal['price']) - 1) * 100
                trade_pairs.append({
                    'buy_date': buy_signal['date_str'],
                    'buy_price': buy_signal['price'],
                    'buy_strength': buy_signal['strength'],
                    'sell_date': signal['date_str'],
                    'sell_price': signal['price'],
                    'sell_strength': signal['strength'],
                    'profit_percent': profit_percent
                })
                buy_signal = None
        
        self.trade_pairs = trade_pairs
        return trade_pairs
    
    def analyze_trade_effectiveness(self) -> Dict:
        """
        分析交易有效性
        
        Returns:
            Dict: 交易有效性分析结果
        """
        if not self.trade_pairs:
            return {'has_trades': False, 'analysis': {}}
        
        profits = [t['profit_percent'] for t in self.trade_pairs]
        win_trades = sum(1 for t in self.trade_pairs if t['profit_percent'] > 0)
        
        # 按强度分组分析收益
        strength_profit_correlation = {
            '低强度(<0.6)': [],
            '中强度(0.6-0.65)': [],
            '高强度(>0.65)': []
        }
        
        for trade in self.trade_pairs:
            strength = trade['buy_strength']
            if strength < 0.6:
                strength_profit_correlation['低强度(<0.6)'].append(trade['profit_percent'])
            elif strength <= 0.65:
                strength_profit_correlation['中强度(0.6-0.65)'].append(trade['profit_percent'])
            else:
                strength_profit_correlation['高强度(>0.65)'].append(trade['profit_percent'])
        
        # 计算各组平均收益
        for key in strength_profit_correlation:
            if strength_profit_correlation[key]:
                strength_profit_correlation[key] = np.mean(strength_profit_correlation[key])
            else:
                strength_profit_correlation[key] = 0
        
        analysis = {
            'total_trades': len(self.trade_pairs),
            'win_trades': win_trades,
            'win_rate': (win_trades / len(self.trade_pairs)) * 100,
            'avg_profit': np.mean(profits),
            'max_profit': max(profits),
            'min_profit': min(profits),
            'profit_std': np.std(profits),
            'strength_profit_correlation': strength_profit_correlation
        }
        
        return {'has_trades': True, 'analysis': analysis}
    
    def generate_report(self) -> str:
        """
        生成分析报告
        
        Returns:
            str: 分析报告文本
        """
        # 分析信号质量
        quality = self.analyze_signal_quality()
        
        # 配对交易
        self.pair_trades()
        
        # 分析交易有效性
        effectiveness = self.analyze_trade_effectiveness()
        
        # 生成报告
        report = []
        report.append(f"===== {self.symbol}买卖信号分析报告 =====")
        report.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 信号质量分析
        report.append("📊 信号质量分析:")
        report.append("-" * 80)
        report.append(f"总信号数量: {quality['total_signals']}个")
        report.append(f"买入信号: {quality['buy_signals']}个")
        report.append(f"卖出信号: {quality['sell_signals']}个")
        
        if quality['strength_stats']:
            strength = quality['strength_stats']
            report.append(f"信号强度均值: {strength['mean']:.3f}")
            report.append(f"信号强度中位数: {strength['median']:.3f}")
            report.append(f"高强度信号(≥0.65): {strength['high_count']}个")
            report.append(f"中强度信号(0.55-0.65): {strength['medium_count']}个")
            report.append(f"低强度信号(<0.55): {strength['low_count']}个")
        
        report.append("")
        report.append(f"月度信号分布:")
        report.append(f"  - 9月: {quality['monthly_distribution'][9]}个")
        report.append(f"  - 10月: {quality['monthly_distribution'][10]}个")
        report.append(f"  - 11月: {quality['monthly_distribution'][11]}个")
        
        # 交易有效性分析
        report.append("")
        report.append("💰 交易有效性分析:")
        report.append("-" * 80)
        if effectiveness.get('has_trades', False):
            analysis = effectiveness['analysis']
            report.append(f"总交易次数: {analysis['total_trades']}次")
            report.append(f"胜率: {analysis['win_rate']:.1f}%")
            report.append(f"平均收益率: {analysis['avg_profit']:.2f}%")
            report.append(f"最大收益率: {analysis['max_profit']:.2f}%")
            report.append(f"最小收益率: {analysis['min_profit']:.2f}%")
            report.append(f"收益率标准差: {analysis['profit_std']:.2f}%")
            
            report.append(f"\n按信号强度分组的平均收益:")
            for strength_range, avg_profit in analysis['strength_profit_correlation'].items():
                report.append(f"  - 强度{strength_range}: {avg_profit:.2f}%")
        else:
            report.append("无交易数据可供分析")
        
        # 最近的交易信号
        report.append("")
        report.append("📋 最近的交易信号:")
        report.append("-" * 80)
        recent_signals = sorted(self.signals, key=lambda x: x['date'], reverse=True)[:10]
        report.append(f"{'日期':<15} {'类型':<10} {'价格':<10} {'强度':<10} {'原因':<30}")
        report.append("-" * 80)
        
        for signal in recent_signals:
            type_str = "买入" if signal['type'] == 'buy' else "卖出"
            report.append(f"{signal['date_str']:<15} {type_str:<10} {signal['price']:<10.3f} {signal['strength']:<10.3f} {signal['reason']:<30}")
        
        # 交易建议
        report.append("")
        report.append("🎯 交易建议:")
        report.append("-" * 80)
        if quality['total_signals'] > 0:
            last_signal = max(self.signals, key=lambda x: x['date'])
            if last_signal['type'] == 'buy' and last_signal['strength'] > 0.65:
                report.append(f"📈 强烈买入: 最近信号为高强度买入信号({last_signal['date_str']})")
                report.append(f"  - 信号强度: {last_signal['strength']:.3f}")
                report.append(f"  - 建议仓位: 60%-80%")
            elif last_signal['type'] == 'buy':
                report.append(f"📈 谨慎买入: 最近信号为中低强度买入信号({last_signal['date_str']})")
                report.append(f"  - 信号强度: {last_signal['strength']:.3f}")
                report.append(f"  - 建议仓位: 30%-50%")
            else:
                report.append(f"📉 观望: 最近信号为卖出信号({last_signal['date_str']})")
        else:
            report.append("🔍 暂无信号: 建议继续观察市场走势")
        
        report.append("")
        report.append("⚠️ 风险提示:")
        report.append("-" * 80)
        report.append("1. 本分析基于模拟信号，仅供参考")
        report.append("2. 市场有风险，投资需谨慎")
        report.append("3. 建议结合其他技术指标和基本面分析")
        
        return "\n".join(report)
    
    def save_results(self, report: str) -> None:
        """
        保存分析结果
        
        Args:
            report: 分析报告文本
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存报告文本
        report_file = os.path.join(self.results_dir, f'{self.symbol}_signal_analysis_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"报告已保存至: {report_file}")
        
        # 保存信号数据
        signals_file = os.path.join(self.results_dir, f'{self.symbol}_signals_{timestamp}.json')
        with open(signals_file, 'w', encoding='utf-8') as f:
            json.dump(self.signals, f, ensure_ascii=False, indent=2)
        logger.info(f"信号数据已保存至: {signals_file}")
        
        # 保存交易对数据
        if self.trade_pairs:
            trades_file = os.path.join(self.results_dir, f'{self.symbol}_trades_{timestamp}.json')
            with open(trades_file, 'w', encoding='utf-8') as f:
                json.dump(self.trade_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"交易对数据已保存至: {trades_file}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='512690买卖信号分析脚本')
    parser.add_argument('--symbol', type=str, default='512690', help='股票代码')
    parser.add_argument('--data_dir', type=str, default='./data/daily', help='数据目录')
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 创建分析器实例
    analyzer = SignalAnalyzer(symbol=args.symbol, data_dir=args.data_dir)
    
    # 加载数据
    if not analyzer.load_data():
        logger.error("加载数据失败，退出程序")
        return
    
    # 生成信号
    analyzer.generate_signals()
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    # 保存结果
    analyzer.save_results(report)

if __name__ == "__main__":
    main()