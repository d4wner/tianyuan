#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512690日线三买信号分析脚本

基于修改后的三买判定规则，结合日线核心和周线级别前提条件，
正确分析酒ETF(512690)的三买信号。

作者: TradeTianYuan
日期: 2025-11-29
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们修改过的日线买点检测器
from src.daily_buy_signal_detector import BuySignalDetector
from src.weekly_trend_detector import WeeklyTrendDetector
from src.config import load_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('512690ThirdBuyAnalyzer')

class ThirdBuyAnalyzer:
    """
    日线三买信号分析器类
    """
    
    def __init__(self, symbol: str = '512690', data_dir: str = './data'):
        """
        初始化三买分析器
        
        Args:
            symbol: 股票代码
            data_dir: 数据目录
        """
        self.symbol = symbol
        self.data_dir = data_dir
        self.daily_data = None
        self.weekly_data = None
        self.config = None
        self.third_buy_signals = []
        self.trade_pairs = []
        
        # 创建检测器实例
        self.daily_detector = BuySignalDetector()
        self.weekly_detector = WeeklyTrendDetector()
        
        # 创建结果目录
        self.results_dir = './results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_config(self) -> bool:
        """
        加载ETF配置信息
        
        Returns:
            bool: 是否加载成功
        """
        try:
            config_file = os.path.join('config', 'etfs.yaml')
            self.config = load_config(config_file)
            logger.info(f"成功加载配置信息")
            return True
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            return False
    
    def load_daily_data(self) -> bool:
        """
        加载日线数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            data_file = os.path.join(self.data_dir, 'daily', f'{self.symbol}_daily.csv')
            if not os.path.exists(data_file):
                logger.error(f"日线数据文件不存在: {data_file}")
                return False
            
            self.daily_data = pd.read_csv(data_file)
            self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
            self.daily_data.sort_values('date', inplace=True)
            
            logger.info(f"成功加载{self.symbol}日线数据，共{len(self.daily_data)}条记录")
            return True
        except Exception as e:
            logger.error(f"加载日线数据失败: {str(e)}")
            return False
    
    def load_weekly_data(self) -> bool:
        """
        加载周线数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            data_file = os.path.join(self.data_dir, 'weekly', f'{self.symbol}_weekly.csv')
            if not os.path.exists(data_file):
                logger.error(f"周线数据文件不存在: {data_file}")
                return False
            
            self.weekly_data = pd.read_csv(data_file)
            self.weekly_data['date'] = pd.to_datetime(self.weekly_data['date'])
            self.weekly_data.sort_values('date', inplace=True)
            
            logger.info(f"成功加载{self.symbol}周线数据，共{len(self.weekly_data)}条记录")
            return True
        except Exception as e:
            logger.error(f"加载周线数据失败: {str(e)}")
            return False
    
    def get_weekly_trend_at_date(self, target_date: datetime) -> Dict:
        """
        获取指定日期的周线趋势状态
        
        Args:
            target_date: 目标日期
            
        Returns:
            Dict: 周线趋势信息
        """
        if self.weekly_data is None:
            return {"trend": "unknown", "strength": 0}
        
        # 找到目标日期之前最近的周线数据
        weekly_before_target = self.weekly_data[self.weekly_data['date'] <= target_date]
        if len(weekly_before_target) < 20:  # 需要足够的周线数据来判断趋势
            return {"trend": "unknown", "strength": 0}
        
        # 使用周线检测器判断趋势
        weekly_segment = weekly_before_target.tail(30).copy()
        trend_result = self.weekly_detector.detect_weekly_bullish_trend(weekly_segment)
        
        return trend_result
    
    def generate_third_buy_signals(self) -> List[Dict]:
        """
        基于修改后的三买判定规则生成信号
        严格按照四个硬性条件和周线前提条件
        
        Returns:
            List[Dict]: 三买信号列表
        """
        if self.daily_data is None or self.weekly_data is None:
            logger.error("请先加载日线和周线数据")
            return []
        
        third_buy_signals = []
        
        # 滚动窗口分析，确保有足够的数据来检测三买
        for window_end in range(60, len(self.daily_data)):
            # 提取当前窗口的数据
            window_data = self.daily_data.iloc[window_end-60:window_end].copy()
            current_date = window_data.iloc[-1]['date']
            
            # 获取当前日期的周线趋势（作为前提条件）
            weekly_trend = self.get_weekly_trend_at_date(current_date)
            
            # 周线前提条件：只在周线趋势向上或盘整时考虑三买
            if weekly_trend.get("trend") == "down":
                logger.debug(f"{current_date.strftime('%Y-%m-%d')} 周线趋势向下，跳过三买检测")
                continue
            
            # 使用修改后的三买检测方法
            is_third_buy, details = self.daily_detector.detect_daily_third_buy(window_data)
            
            if is_third_buy:
                # 添加周线前提条件信息
                signal = {
                    'date': current_date.timestamp() * 1000,  # 毫秒时间戳
                    'date_str': current_date.strftime('%Y-%m-%d'),
                    'type': 'third_buy',
                    'price': window_data.iloc[-1]['close'],
                    'strength': self.calculate_signal_strength(details, weekly_trend),
                    'weekly_trend': weekly_trend,
                    'details': details
                }
                third_buy_signals.append(signal)
                logger.info(f"在{current_date.strftime('%Y-%m-%d')} 检测到日线三买信号")
        
        # 去重：避免重复的信号（同一天不应该有多个三买信号）
        unique_signals = []
        seen_dates = set()
        for signal in third_buy_signals:
            if signal['date_str'] not in seen_dates:
                unique_signals.append(signal)
                seen_dates.add(signal['date_str'])
        
        self.third_buy_signals = unique_signals
        logger.info(f"成功生成{len(unique_signals)}个日线三买信号")
        return unique_signals
    
    def calculate_signal_strength(self, details: Dict, weekly_trend: Dict) -> float:
        """
        计算信号强度
        
        Args:
            details: 三买信号详情
            weekly_trend: 周线趋势信息
            
        Returns:
            float: 信号强度（0-1之间）
        """
        # 基础强度
        base_strength = 0.6  # 基础强度分
        
        # 中枢高度评分（中枢越明显，分数越高）
        central_height_pct = details['central_bank']['height_pct']
        if central_height_pct > 10:
            base_strength += 0.1
        elif central_height_pct > 7:
            base_strength += 0.05
        elif central_height_pct > 5:
            base_strength += 0.02
        
        # 突破强度评分
        if details['breakthrough']['consecutive_days'] > 2:
            base_strength += 0.05
        
        # 成交量评分
        if details['breakthrough']['volume_condition']:
            base_strength += 0.05
        
        # 周线趋势评分
        weekly_strength = weekly_trend.get('strength', 0)
        base_strength += weekly_strength * 0.2  # 周线趋势最多贡献0.2分
        
        # 确保强度在0-1之间
        return min(max(base_strength, 0.5), 1.0)
    
    def find_corresponding_sell_signals(self) -> List[Dict]:
        """
        为每个三买信号寻找对应的卖出信号
        
        Returns:
            List[Dict]: 交易对列表
        """
        trade_pairs = []
        
        for buy_signal in self.third_buy_signals:
            buy_date = pd.to_datetime(buy_signal['date_str'])
            buy_price = buy_signal['price']
            
            # 在日线数据中找到买入信号之后的数据
            after_buy_data = self.daily_data[self.daily_data['date'] > buy_date]
            if len(after_buy_data) == 0:
                continue
            
            # 寻找卖出信号（简化逻辑：设置10%止盈或-5%止损）
            stop_profit_price = buy_price * 1.10
            stop_loss_price = buy_price * 0.95
            max_hold_days = 60  # 最长持有60天
            
            sell_signal = None
            for i, row in after_buy_data.iterrows():
                # 检查是否达到止盈或止损条件
                if row['high'] >= stop_profit_price:
                    sell_signal = {
                        'date': row['date'].timestamp() * 1000,
                        'date_str': row['date'].strftime('%Y-%m-%d'),
                        'type': 'sell',
                        'price': stop_profit_price,
                        'reason': '达到10%止盈'
                    }
                    break
                elif row['low'] <= stop_loss_price:
                    sell_signal = {
                        'date': row['date'].timestamp() * 1000,
                        'date_str': row['date'].strftime('%Y-%m-%d'),
                        'type': 'sell',
                        'price': stop_loss_price,
                        'reason': '达到-5%止损'
                    }
                    break
                # 检查是否达到最长持有时间
                elif (row['date'] - buy_date).days >= max_hold_days:
                    sell_signal = {
                        'date': row['date'].timestamp() * 1000,
                        'date_str': row['date'].strftime('%Y-%m-%d'),
                        'type': 'sell',
                        'price': row['close'],
                        'reason': '达到最长持有时间60天'
                    }
                    break
            
            if sell_signal:
                # 计算收益率
                profit_percent = ((sell_signal['price'] / buy_price) - 1) * 100
                
                trade_pairs.append({
                    'buy_date': buy_signal['date_str'],
                    'buy_price': buy_price,
                    'buy_strength': buy_signal['strength'],
                    'sell_date': sell_signal['date_str'],
                    'sell_price': sell_signal['price'],
                    'sell_reason': sell_signal['reason'],
                    'profit_percent': profit_percent,
                    'hold_days': (pd.to_datetime(sell_signal['date_str']) - buy_date).days,
                    'weekly_trend_at_buy': buy_signal['weekly_trend']
                })
        
        self.trade_pairs = trade_pairs
        logger.info(f"成功配对{len(trade_pairs)}个交易")
        return trade_pairs
    
    def analyze_signal_quality(self) -> Dict:
        """
        分析三买信号质量
        
        Returns:
            Dict: 信号质量分析结果
        """
        if not self.third_buy_signals:
            return {
                'total_signals': 0,
                'monthly_distribution': {9: 0, 10: 0, 11: 0}
            }
        
        # 按月统计信号分布
        monthly_distribution = {9: 0, 10: 0, 11: 0}
        strengths = [s['strength'] for s in self.third_buy_signals]
        
        for signal in self.third_buy_signals:
            date_obj = datetime.fromtimestamp(signal['date'] / 1000)
            month = date_obj.month
            if month in monthly_distribution:
                monthly_distribution[month] += 1
        
        strength_stats = {
            'mean': np.mean(strengths) if strengths else 0,
            'median': np.median(strengths) if strengths else 0,
            'high_count': sum(1 for s in strengths if s >= 0.7),
            'medium_count': sum(1 for s in strengths if 0.6 <= s < 0.7),
            'low_count': sum(1 for s in strengths if s < 0.6)
        }
        
        return {
            'total_signals': len(self.third_buy_signals),
            'monthly_distribution': monthly_distribution,
            'strength_stats': strength_stats
        }
    
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
        
        # 按信号强度分组分析收益
        strength_profit_correlation = {
            '低强度(<0.6)': [],
            '中强度(0.6-0.7)': [],
            '高强度(>0.7)': []
        }
        
        for trade in self.trade_pairs:
            strength = trade['buy_strength']
            if strength < 0.6:
                strength_profit_correlation['低强度(<0.6)'].append(trade['profit_percent'])
            elif strength <= 0.7:
                strength_profit_correlation['中强度(0.6-0.7)'].append(trade['profit_percent'])
            else:
                strength_profit_correlation['高强度(>0.7)'].append(trade['profit_percent'])
        
        # 计算各组平均收益
        for key in strength_profit_correlation:
            if strength_profit_correlation[key]:
                strength_profit_correlation[key] = np.mean(strength_profit_correlation[key])
            else:
                strength_profit_correlation[key] = 0
        
        analysis = {
            'total_trades': len(self.trade_pairs),
            'win_trades': win_trades,
            'win_rate': (win_trades / len(self.trade_pairs)) * 100 if self.trade_pairs else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'min_profit': min(profits) if profits else 0,
            'profit_std': np.std(profits) if profits else 0,
            'avg_hold_days': np.mean([t['hold_days'] for t in self.trade_pairs]) if self.trade_pairs else 0,
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
        
        # 分析交易有效性
        effectiveness = self.analyze_trade_effectiveness()
        
        # 生成报告
        report = []
        report.append(f"===== {self.symbol}日线三买信号分析报告 =====")
        report.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"分析方法: 严格按照修改后的三买判定规则（四硬性条件+周线前提）")
        report.append("")
        
        # 信号质量分析
        report.append("📊 信号质量分析:")
        report.append("-" * 80)
        report.append(f"总三买信号数量: {quality['total_signals']}个")
        
        if quality.get('strength_stats'):
            strength = quality['strength_stats']
            report.append(f"信号强度均值: {strength['mean']:.3f}")
            report.append(f"信号强度中位数: {strength['median']:.3f}")
            report.append(f"高强度信号(≥0.7): {strength['high_count']}个")
            report.append(f"中强度信号(0.6-0.7): {strength['medium_count']}个")
            report.append(f"低强度信号(<0.6): {strength['low_count']}个")
        
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
            report.append(f"平均持有天数: {analysis['avg_hold_days']:.1f}天")
            
            report.append(f"\n按信号强度分组的平均收益:")
            for strength_range, avg_profit in analysis['strength_profit_correlation'].items():
                report.append(f"  - 强度{strength_range}: {avg_profit:.2f}%")
        else:
            report.append("无交易数据可供分析")
        
        # 最近的三买信号
        report.append("")
        report.append("📋 最近的三买信号:")
        report.append("-" * 80)
        recent_signals = sorted(self.third_buy_signals, key=lambda x: x['date'], reverse=True)[:5]
        report.append(f"{'日期':<15} {'价格':<10} {'强度':<10} {'周线趋势':<15} {'中枢高度':<10}")
        report.append("-" * 80)
        
        for signal in recent_signals:
            central_height = signal['details']['central_bank']['height_pct']
            weekly_trend = signal['weekly_trend'].get('trend', 'unknown')
            report.append(f"{signal['date_str']:<15} {signal['price']:<10.3f} {signal['strength']:<10.3f} {weekly_trend:<15} {central_height:<10.2f}%")
        
        # 交易建议
        report.append("")
        report.append("🎯 交易建议:")
        report.append("-" * 80)
        if quality['total_signals'] > 0:
            last_signal = max(self.third_buy_signals, key=lambda x: x['date'])
            last_signal_date = pd.to_datetime(last_signal['date_str'])
            days_since_last = (datetime.now() - last_signal_date).days
            
            if days_since_last <= 10:  # 最近10天内有信号
                if last_signal['strength'] > 0.7:
                    report.append(f"📈 强烈关注: 最近有高强度三买信号({last_signal['date_str']})")
                    report.append(f"  - 信号强度: {last_signal['strength']:.3f}")
                    report.append(f"  - 周线趋势: {last_signal['weekly_trend'].get('trend', 'unknown')}")
                    report.append(f"  - 建议: 结合当前市场环境考虑入场")
                else:
                    report.append(f"📊 谨慎关注: 最近有三买信号({last_signal['date_str']})")
                    report.append(f"  - 信号强度: {last_signal['strength']:.3f}")
                    report.append(f"  - 建议: 等待更明确的确认信号")
            else:
                report.append(f"🔍 观望: 最近三买信号已超过{days_since_last}天")
                report.append(f"  - 最后信号: {last_signal['date_str']} (强度: {last_signal['strength']:.3f})")
                report.append(f"  - 建议: 继续观察，等待新的三买信号形成")
        else:
            report.append("🔍 暂无三买信号: 建议继续观察市场走势")
        
        # 规则说明
        report.append("")
        report.append("📋 三买判定规则说明:")
        report.append("-" * 80)
        report.append("1. 中枢突破有效性（核心前提）：")
        report.append("   - 价格需连续2日收盘价≥中枢上沿×1.008（突破幅度≥0.8%）")
        report.append("   - 突破时的成交量需≥近5日均量的120%")
        report.append("2. 回抽有效性：突破后回抽的最低收盘价≥中枢上沿（严格不进中枢）")
        report.append("3. 背驰验证：回抽过程中30分钟级别形成底背驰")
        report.append("4. 中枢内信号排除：价格处于中枢区间内直接排除三买信号")
        report.append("5. 周线前提条件：只在周线趋势向上或盘整时考虑三买信号")
        
        report.append("")
        report.append("⚠️ 风险提示:")
        report.append("-" * 80)
        report.append("1. 本分析基于修改后的三买判定规则，严格遵循四硬性条件")
        report.append("2. 市场有风险，投资需谨慎")
        report.append("3. 建议结合多级别分析和风险控制策略")
        
        return "\n".join(report)
    
    def save_results(self, report: str) -> None:
        """
        保存分析结果
        
        Args:
            report: 分析报告文本
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存报告文本
        report_file = os.path.join(self.results_dir, f'{self.symbol}_third_buy_analysis_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"三买分析报告已保存至: {report_file}")
        
        # 保存三买信号数据
        signals_file = os.path.join(self.results_dir, f'{self.symbol}_third_buy_signals_{timestamp}.json')
        # 移除details中的numpy类型以避免JSON序列化问题
        serializable_signals = []
        for signal in self.third_buy_signals:
            serializable = signal.copy()
            # 转换numpy类型为Python原生类型
            if 'details' in serializable:
                if 'central_bank' in serializable['details']:
                    for key, value in serializable['details']['central_bank'].items():
                        if isinstance(value, (np.integer, np.floating)):
                            serializable['details']['central_bank'][key] = float(value)
            if isinstance(serializable['strength'], (np.integer, np.floating)):
                serializable['strength'] = float(serializable['strength'])
            serializable_signals.append(serializable)
        
        with open(signals_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_signals, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"三买信号数据已保存至: {signals_file}")
        
        # 保存交易对数据
        if self.trade_pairs:
            trades_file = os.path.join(self.results_dir, f'{self.symbol}_third_buy_trades_{timestamp}.json')
            # 转换numpy类型
            serializable_trades = []
            for trade in self.trade_pairs:
                serializable = trade.copy()
                for key, value in serializable.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable[key] = float(value)
                serializable_trades.append(serializable)
            
            with open(trades_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_trades, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"三买交易数据已保存至: {trades_file}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='512690日线三买信号分析脚本')
    parser.add_argument('--symbol', type=str, default='512690', help='股票代码')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 添加系统路径
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    args = parse_args()
    
    # 创建分析器实例
    analyzer = ThirdBuyAnalyzer(symbol=args.symbol, data_dir=args.data_dir)
    
    # 加载配置和数据
    if not analyzer.load_config():
        logger.error("加载配置失败，退出程序")
        return
    
    if not analyzer.load_daily_data():
        logger.error("加载日线数据失败，退出程序")
        return
    
    if not analyzer.load_weekly_data():
        logger.error("加载周线数据失败，退出程序")
        return
    
    # 生成三买信号
    logger.info("开始生成日线三买信号...")
    analyzer.generate_third_buy_signals()
    
    # 配对交易信号
    logger.info("开始配对对应的卖出信号...")
    analyzer.find_corresponding_sell_signals()
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    # 保存结果
    analyzer.save_results(report)
    
    logger.info("三买信号分析完成！")

if __name__ == "__main__":
    main()