#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交易信号统计分析脚本 - 区分核心(日线)和参考(分钟)级别信号"""

import json
import datetime
import yaml
import os
import pandas as pd
from typing import Dict, List, Tuple
from src.chanlun_daily_detector import ChanlunDailyDetector

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SignalStatistics')


class SignalStatisticsAnalyzer:
    """交易信号统计分析器"""
    
    def __init__(self, config_dir: str):
        """初始化统计分析器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = config_dir
        self.etfs_config = self._load_config('etfs.yaml')
        self.risk_rules = self._load_config('risk_rules.yaml')
        self.daily_detector = ChanlunDailyDetector()
        self.current_date = datetime.datetime.now()
        
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            配置字典
        """
        try:
            with open(os.path.join(self.config_dir, config_file), 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件 {config_file} 失败: {str(e)}")
            return {}
    
    def load_signals(self, signals_file: str) -> List[Dict]:
        """加载交易信号数据
        
        Args:
            signals_file: 信号文件路径
            
        Returns:
            信号列表
        """
        try:
            with open(signals_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载信号文件失败: {str(e)}")
            return []
    
    def parse_signal_timeframe(self, signal: Dict) -> Tuple[str, str]:
        """解析信号的时间周期
        
        Args:
            signal: 信号字典
            
        Returns:
            (周期类型, 具体周期)
        """
        # 检查是否有明确的周期信息
        if 'timeframe' in signal:
            timeframe = signal['timeframe'].lower()
            if 'day' in timeframe or '日线' in timeframe:
                return 'daily', '日线'
            elif '15min' in timeframe or '15分钟' in timeframe:
                return 'minute', '15分钟'
            elif '30min' in timeframe or '30分钟' in timeframe:
                return 'minute', '30分钟'
            elif '60min' in timeframe or '60分钟' in timeframe:
                return 'minute', '60分钟'
        
        # 检查原因字段
        if 'reason' in signal:
            reason = signal['reason'].lower()
            if '日线' in reason:
                return 'daily', '日线'
            elif '15分钟' in reason:
                return 'minute', '15分钟'
            elif '30分钟' in reason:
                return 'minute', '30分钟'
            elif '60分钟' in reason:
                return 'minute', '60分钟'
        
        # 默认假设为分钟级别（需要根据实际数据格式调整）
        logger.warning(f"无法确定信号周期: {signal.get('reason', '未知')}")
        return 'minute', '未知'
    
    def filter_core_daily_signals(self, signals: List[Dict]) -> List[Dict]:
        """筛选核心日线级别信号
        
        Args:
            signals: 所有信号列表
            
        Returns:
            核心日线级别信号列表
        """
        core_signals = []
        
        for signal in signals:
            timeframe_type, _ = self.parse_signal_timeframe(signal)
            # 只保留日线级别买入信号
            if timeframe_type == 'daily' and signal['type'] == 'buy':
                core_signals.append(signal)
        
        return core_signals
    
    def get_past_months_signals(self, signals: List[Dict], months: int = 3) -> List[Dict]:
        """获取过去几个月的信号
        
        Args:
            signals: 信号列表
            months: 月数，默认为3个月
            
        Returns:
            过滤后的信号列表
        """
        cutoff_date = self.current_date - datetime.timedelta(days=months*30)
        filtered_signals = []
        
        for signal in signals:
            # 转换时间戳
            signal_date = datetime.datetime.fromtimestamp(signal['date']/1000)
            if signal_date >= cutoff_date:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def calculate_core_statistics(self, signals: List[Dict]) -> Dict:
        """计算核心信号统计数据
        
        Args:
            signals: 核心日线级别信号列表
            
        Returns:
            统计结果字典
        """
        if not signals:
            return {
                'signal_count': 0,
                'average_strength': 0.0,
                'signals': []
            }
        
        # 计算平均强度
        strengths = [s['strength'] for s in signals]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        
        # 格式化信号数据
        formatted_signals = []
        for signal in signals:
            # 提取底分型和中枢信息（如果有）
            fractal_info = signal.get('fractal_info', {})
            central_info = signal.get('central_info', {})
            
            formatted_signal = {
                'signal_date': datetime.datetime.fromtimestamp(signal['date']/1000).strftime('%Y-%m-%d'),
                'central_upper_edge': central_info.get('high', 'N/A'),
                'central_lower_edge': central_info.get('low', 'N/A'),
                'fractal_data': {
                    'k2_high': fractal_info.get('k2_high', 'N/A'),
                    'k5_close': fractal_info.get('k5_close', 'N/A')
                } if fractal_info else {},
                'volume_ratio': signal.get('volume_ratio', 'N/A'),
                'signal_strength': signal['strength'],
                'meets_strategy': signal.get('meets_strategy', True)
            }
            formatted_signals.append(formatted_signal)
        
        return {
            'signal_count': len(signals),
            'average_strength': round(avg_strength, 4),
            'signals': formatted_signals
        }
    
    def calculate_minute_statistics(self, signals: List[Dict]) -> Dict:
        """计算分钟级别信号统计数据
        
        Args:
            signals: 所有信号列表
            
        Returns:
            统计结果字典
        """
        minute_signals = {}
        
        for signal in signals:
            timeframe_type, specific_timeframe = self.parse_signal_timeframe(signal)
            if timeframe_type == 'minute':
                if specific_timeframe not in minute_signals:
                    minute_signals[specific_timeframe] = []
                
                # 格式化分钟信号数据
                formatted_signal = {
                    'signal_time': datetime.datetime.fromtimestamp(signal['date']/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'signal_type': f"{specific_timeframe}参考"
                }
                minute_signals[specific_timeframe].append(formatted_signal)
        
        # 汇总统计
        statistics = {}
        for timeframe, signals in minute_signals.items():
            statistics[timeframe] = len(signals)
        
        return {
            'timeframe_counts': statistics,
            'total_minute_signals': sum(statistics.values()),
            'signals_by_timeframe': minute_signals
        }
    
    def analyze_daily_buy_condition(self, df: pd.DataFrame) -> Dict:
        """分析日线级别'创新低破中枢回抽'买点条件
        
        Args:
            df: 日线数据
            
        Returns:
            分析结果
        """
        return self.daily_detector.analyze_daily_buy_condition(df)
    
    def generate_statistics_report(self, signals_file: str) -> Dict:
        """生成完整的统计报告
        
        Args:
            signals_file: 信号文件路径
            
        Returns:
            完整统计报告
        """
        # 加载信号数据
        signals = self.load_signals(signals_file)
        
        # 获取过去3个月的信号
        recent_signals = self.get_past_months_signals(signals, months=3)
        
        # 筛选核心日线级别信号
        core_daily_signals = self.filter_core_daily_signals(recent_signals)
        
        # 计算核心统计数据
        core_statistics = self.calculate_core_statistics(core_daily_signals)
        
        # 计算分钟级别统计数据
        minute_statistics = self.calculate_minute_statistics(recent_signals)
        
        # 构建完整报告
        report = {
            'report_time': self.current_date.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics_period': '过去3个月',
            'core_statistics': core_statistics,
            'reference_statistics': minute_statistics,
            'summary': {
                'total_core_signals': core_statistics['signal_count'],
                'average_core_strength': core_statistics['average_strength'],
                'minute_signals_summary': minute_statistics['timeframe_counts']
            }
        }
        
        return report
    
    def display_report(self, report: Dict):
        """显示统计报告
        
        Args:
            report: 统计报告字典
        """
        print("=" * 80)
        print(f"交易信号统计报告 - {report['report_time']}")
        print(f"统计周期: {report['statistics_period']}")
        print("=" * 80)
        
        # 显示核心统计（日线级别）
        print("\n核心策略统计（仅日线级别）:")
        print("-" * 80)
        core_stats = report['core_statistics']
        print(f"核心信号数量: {core_stats['signal_count']}")
        print(f"核心信号平均强度: {core_stats['average_strength']:.4f}")
        
        if core_stats['signals']:
            print("\n核心信号详情:")
            print(f"{'触发日期':<15} {'中枢上沿':<10} {'中枢下沿':<10} {'量能比例':<10} {'信号强度':<10} {'满足策略':<10}")
            print("-" * 80)
            for signal in core_stats['signals']:
                meets = "True" if signal['meets_strategy'] else "False"
                print(f"{signal['signal_date']:<15} {signal['central_upper_edge']:<10} {signal['central_lower_edge']:<10} ")
                print(f"{signal['volume_ratio']:<10} {signal['signal_strength']:<10} {meets:<10}")
        
        # 显示参考统计（分钟级别）
        print("\n参考统计（分钟级别信号）:")
        print("-" * 80)
        minute_stats = report['reference_statistics']
        
        print("各分钟周期信号数量:")
        for timeframe, count in minute_stats['timeframe_counts'].items():
            print(f"  {timeframe}: {count}个 (非核心策略信号，短线参考)")
        
        print("\n-" * 80)
        print("重要说明:")
        print("1. 核心策略信号仅统计日线级别满足'创新低破中枢回抽'买点条件的信号")
        print("2. 分钟级别信号仅作短线参考，不纳入核心策略统计")
        print("3. 信号强度 = 背驰力度(30%) + 量能(40%) + 分型有效性(30%)")
        print("=" * 80)


def main():
    """主函数"""
    # 配置路径
    config_dir = '/Users/pingan/tools/trade/tianyuan/config'
    signals_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251124_120616.json'
    
    # 创建分析器实例
    analyzer = SignalStatisticsAnalyzer(config_dir)
    
    # 生成统计报告
    report = analyzer.generate_statistics_report(signals_file)
    
    # 显示报告
    analyzer.display_report(report)
    
    # 保存报告到文件
    output_file = f"/Users/pingan/tools/trade/tianyuan/outputs/statistics_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计报告已保存到: {output_file}")


if __name__ == "__main__":
    main()