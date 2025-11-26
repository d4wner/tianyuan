#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""信号失效条件检测模块 - 实现完整的交易信号有效性验证"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入依赖模块
from src.data_processor import DataProcessor
from src.daily_analyzer import DailyAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('signal_validator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('SignalValidator')

class SignalValidator:
    """交易信号有效性验证器"""
    
    def __init__(self, config: Dict[str, any] = None):
        """初始化信号验证器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.data_processor = DataProcessor(config)
        self.daily_analyzer = DailyAnalyzer()
        
        # 失效条件配置
        self.stop_loss_percent = self.config.get('stop_loss_percent', 5.0)  # 默认止损5%
        self.max_valid_days = self.config.get('max_valid_days', 15)  # 信号最长有效天数
        self.divergence_change_threshold = self.config.get('divergence_change_threshold', 0.3)  # 背驰变化阈值
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """解析时间戳字符串
        
        Args:
            timestamp_str: 时间戳字符串
            
        Returns:
            解析后的datetime对象
        """
        if not timestamp_str:
            return None
            
        try:
            # 尝试不同的时间格式
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%Y%m%d_%H%M%S",
                "%Y%m%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # 如果都失败，尝试自动解析
            return pd.to_datetime(timestamp_str).to_pydatetime()
            
        except Exception as e:
            logger.error(f"解析时间戳失败: {str(e)}")
            return None
    
    def check_price_breach(self, signal: Dict[str, any], latest_data: pd.DataFrame, 
                          breach_type: str = "stop_loss") -> Tuple[bool, str]:
        """检查价格突破条件
        
        Args:
            signal: 交易信号
            latest_data: 最新数据
            breach_type: 突破类型 ('stop_loss' 或 'take_profit')
            
        Returns:
            (是否突破, 原因描述)
        """
        if latest_data.empty:
            return False, "数据为空，无法检查价格突破"
        
        try:
            # 获取最新价格
            latest_price = latest_data.iloc[-1]['close']
            
            # 获取信号生成时的价格（简化处理，实际应该记录信号生成时的价格）
            # 这里假设使用最新数据的第一个价格作为信号生成时的价格
            signal_price = latest_data.iloc[0]['close']
            
            if breach_type == "stop_loss":
                # 计算止损价格
                stop_loss_price = signal_price * (1 - self.stop_loss_percent / 100)
                
                if latest_price < stop_loss_price:
                    return True, f"价格跌破止损位: 当前价 {latest_price:.2f}, 止损价 {stop_loss_price:.2f}"
            
            # 如果没有突破
            return False, "价格未突破止损位"
            
        except Exception as e:
            logger.error(f"检查价格突破失败: {str(e)}")
            return False, f"检查失败: {str(e)}"
    
    def check_daily_conditions(self, signal: Dict[str, any], latest_data: pd.DataFrame) -> Tuple[bool, str]:
        """检查日线买入条件是否仍然满足
        
        Args:
            signal: 交易信号
            latest_data: 最新日线数据
            
        Returns:
            (条件是否不再满足, 原因描述)
        """
        if latest_data.empty:
            return False, "数据为空，无法检查日线条件"
        
        try:
            # 重新分析最新的日线条件
            latest_daily_result = self.daily_analyzer.analyze_daily_conditions(latest_data)
            
            # 检查买入信号是否仍然存在
            if not latest_daily_result.get('buy_signal', False):
                # 获取不满足的具体原因
                reason = "日线买入条件不再满足"
                
                # 分析具体哪个条件不满足
                details = latest_daily_result.get('details', {})
                
                if not details.get('has_bottom_fractal', False):
                    reason += " - 底分型消失"
                elif not details.get('has_formed_pen', False):
                    reason += " - 笔结构不完整"
                elif not details.get('has_valid_zhongshu', False):
                    reason += " - 中枢结构变化"
                elif not details.get('has_divergence', False):
                    reason += " - 背驰信号消失"
                
                return True, reason
            
            return False, "日线买入条件仍然满足"
            
        except Exception as e:
            logger.error(f"检查日线条件失败: {str(e)}")
            return False, f"检查失败: {str(e)}"
    
    def check_time_expiry(self, signal: Dict[str, any]) -> Tuple[bool, str]:
        """检查信号是否因时间过期而失效
        
        Args:
            signal: 交易信号
            
        Returns:
            (是否过期, 原因描述)
        """
        try:
            # 获取信号生成时间
            timestamp_str = signal.get('timestamp', '')
            signal_time = self._parse_timestamp(timestamp_str)
            
            if not signal_time:
                return False, "无法确定信号生成时间"
            
            # 计算信号已存在的天数
            current_time = datetime.now()
            days_passed = (current_time - signal_time).days
            
            if days_passed >= self.max_valid_days:
                return True, f"信号已过期: 已超过 {self.max_valid_days} 天有效期"
            
            return False, f"信号未过期: 已存在 {days_passed} 天，有效期还剩 {self.max_valid_days - days_passed} 天"
            
        except Exception as e:
            logger.error(f"检查时间过期失败: {str(e)}")
            return False, f"检查失败: {str(e)}"
    
    def check_price_pattern_change(self, signal: Dict[str, any], latest_data: pd.DataFrame) -> Tuple[bool, str]:
        """检查价格形态是否发生变化
        
        Args:
            signal: 交易信号
            latest_data: 最新数据
            
        Returns:
            (形态是否变化, 原因描述)
        """
        if latest_data.empty:
            return False, "数据为空，无法检查价格形态"
        
        try:
            # 检测价格形态
            pattern_data = self.data_processor.detect_price_patterns(latest_data, "fractal")
            
            # 获取最近的分型
            recent_window = min(5, len(pattern_data))  # 检查最近5个交易日
            recent_data = pattern_data.tail(recent_window)
            
            # 检查是否出现新的顶分型
            new_top_fractals = recent_data[recent_data['top_fractal']].index.tolist()
            if new_top_fractals:
                return True, f"出现新的顶分型，可能反转下跌"
            
            # 检查是否跌破前底分型的低点
            if 'bottom_fractal' in pattern_data.columns:
                bottom_fractals = pattern_data[pattern_data['bottom_fractal']]
                if not bottom_fractals.empty:
                    # 获取最近的底分型
                    recent_bottom = bottom_fractals.iloc[-1]['low']
                    latest_low = latest_data.iloc[-1]['low']
                    
                    if latest_low < recent_bottom:
                        return True, "跌破最近底分型的低点，形态破坏"
            
            return False, "价格形态保持完好"
            
        except Exception as e:
            logger.error(f"检查价格形态变化失败: {str(e)}")
            return False, f"检查失败: {str(e)}"
    
    def check_divergence_change(self, signal: Dict[str, any], latest_data: pd.DataFrame) -> Tuple[bool, str]:
        """检查背驰状态是否发生变化
        
        Args:
            signal: 交易信号
            latest_data: 最新数据
            
        Returns:
            (背驰是否变化, 原因描述)
        """
        if latest_data.empty:
            return False, "数据为空，无法检查背驰变化"
        
        try:
            # 计算最新的MACD指标
            latest_macd = self.daily_analyzer._calculate_macd(latest_data)
            
            if latest_macd.empty:
                return False, "无法计算MACD指标"
            
            # 获取信号生成时的背驰强度
            original_divergence = signal.get('背驰强度', 0.0)
            
            # 分析最新的背驰情况
            latest_daily_result = self.daily_analyzer.analyze_daily_conditions(latest_data)
            current_divergence = latest_daily_result.get('divergence_strength', 0.0)
            
            # 检查背驰强度是否显著减弱
            divergence_change = abs(original_divergence - current_divergence)
            if divergence_change > self.divergence_change_threshold:
                if current_divergence < original_divergence:
                    return True, f"背驰强度减弱: 原值 {original_divergence:.2f}, 当前值 {current_divergence:.2f}"
            
            # 检查是否出现顶背离
            if 'has_top_divergence' in latest_daily_result and latest_daily_result['has_top_divergence']:
                return True, "出现顶背离信号，可能反转下跌"
            
            return False, "背驰状态保持稳定"
            
        except Exception as e:
            logger.error(f"检查背驰变化失败: {str(e)}")
            return False, f"检查失败: {str(e)}"
    
    def check_trend_reversal(self, signal: Dict[str, any], latest_data: pd.DataFrame) -> Tuple[bool, str]:
        """检查趋势是否发生反转
        
        Args:
            signal: 交易信号
            latest_data: 最新数据
            
        Returns:
            (趋势是否反转, 原因描述)
        """
        if latest_data.empty:
            return False, "数据为空，无法检查趋势反转"
        
        try:
            # 计算均线
            short_window = 5
            long_window = 20
            
            if len(latest_data) >= long_window:
                latest_data['short_ma'] = latest_data['close'].rolling(window=short_window).mean()
                latest_data['long_ma'] = latest_data['close'].rolling(window=long_window).mean()
                
                # 检查均线死叉
                if (latest_data['short_ma'].iloc[-2] > latest_data['long_ma'].iloc[-2] and 
                    latest_data['short_ma'].iloc[-1] <= latest_data['long_ma'].iloc[-1]):
                    return True, "短期均线下穿长期均线，出现死叉"
                
                # 检查连续下跌
                if len(latest_data) >= 5:
                    # 计算最近5天的涨跌幅
                    recent_changes = latest_data['close'].pct_change().tail(5)
                    consecutive_drops = sum(recent_changes < 0) >= 4
                    total_drop = recent_changes.sum() * 100
                    
                    if consecutive_drops and total_drop < -5:  # 连续下跌且累计跌幅超过5%
                        return True, f"连续下跌，累计跌幅 {total_drop:.2f}%"
            
            return False, "趋势未发生明显反转"
            
        except Exception as e:
            logger.error(f"检查趋势反转失败: {str(e)}")
            return False, f"检查失败: {str(e)}"
    
    def validate_signal(self, signal: Dict[str, any], latest_data: Optional[pd.DataFrame] = None, 
                       symbol: str = "sh512660") -> Dict[str, any]:
        """验证信号是否仍然有效
        
        Args:
            signal: 交易信号
            latest_data: 最新数据（可选）
            symbol: 股票代码
            
        Returns:
            更新后的信号，包含有效性信息
        """
        logger.info(f"开始验证信号有效性: {symbol}")
        
        # 深拷贝信号，避免修改原始数据
        validated_signal = signal.copy()
        
        # 初始化有效性标志和原因
        validated_signal['expired'] = False
        validated_signal['expiry_reason'] = "信号仍然有效"
        validated_signal['validation_results'] = {}
        
        # 如果未提供最新数据，自动获取
        if latest_data is None or latest_data.empty:
            logger.info("未提供最新数据，自动获取")
            latest_data = self.data_processor.get_daily_data(symbol, days=30)
            
            if latest_data.empty:
                validated_signal['expired'] = True
                validated_signal['expiry_reason'] = "无法获取最新数据，信号有效性无法验证"
                logger.error("无法获取最新数据进行验证")
                return validated_signal
        
        # 检查信号是否为买入信号，只有买入信号才需要验证有效性
        signal_type = signal.get('信号类型', '')
        if signal_type not in ['强烈买入', '买入', '谨慎买入', 'strong_buy', 'buy', 'weak_buy']:
            validated_signal['expired'] = True
            validated_signal['expiry_reason'] = "非买入信号，无需验证有效性"
            logger.info("非买入信号，无需验证")
            return validated_signal
        
        # 执行各项有效性检查
        checkers = [
            ("price_breach", self.check_price_breach, {}),
            ("daily_conditions", self.check_daily_conditions, {}),
            ("time_expiry", self.check_time_expiry, {}),
            ("price_pattern_change", self.check_price_pattern_change, {}),
            ("divergence_change", self.check_divergence_change, {}),
            ("trend_reversal", self.check_trend_reversal, {})
        ]
        
        # 任何一个条件满足就标记为失效
        for check_name, checker_func, kwargs in checkers:
            try:
                if check_name in ['daily_conditions', 'price_pattern_change', 'divergence_change', 'trend_reversal']:
                    expired, reason = checker_func(signal, latest_data, **kwargs)
                elif check_name == 'price_breach':
                    expired, reason = checker_func(signal, latest_data, breach_type="stop_loss", **kwargs)
                else:
                    expired, reason = checker_func(signal, **kwargs)
                
                validated_signal['validation_results'][check_name] = {
                    "expired": expired,
                    "reason": reason
                }
                
                # 如果检测到失效条件，立即标记并记录原因
                if expired:
                    validated_signal['expired'] = True
                    validated_signal['expiry_reason'] = reason
                    logger.warning(f"信号失效: {reason}")
                    # 一旦发现一个失效条件，就可以提前返回
                    break
                    
            except Exception as e:
                logger.error(f"执行 {check_name} 检查时出错: {str(e)}")
                validated_signal['validation_results'][check_name] = {
                    "expired": False,
                    "reason": f"检查失败: {str(e)}"
                }
        
        # 如果所有检查都通过，保持信号有效
        if not validated_signal['expired']:
            logger.info("信号验证通过，仍然有效")
        
        # 更新验证时间
        validated_signal['last_validation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return validated_signal
    
    def batch_validate_signals(self, signals: List[Dict[str, any]], 
                              symbol: str = "sh512660") -> List[Dict[str, any]]:
        """批量验证多个信号
        
        Args:
            signals: 信号列表
            symbol: 股票代码
            
        Returns:
            验证后的信号列表
        """
        logger.info(f"开始批量验证信号: {len(signals)} 个信号")
        
        # 获取一次最新数据，用于所有信号的验证
        latest_data = self.data_processor.get_daily_data(symbol, days=30)
        
        validated_signals = []
        expired_count = 0
        
        for signal in signals:
            validated_signal = self.validate_signal(signal, latest_data.copy(), symbol)
            validated_signals.append(validated_signal)
            
            if validated_signal.get('expired', False):
                expired_count += 1
        
        logger.info(f"批量验证完成: 共 {len(signals)} 个信号，其中 {expired_count} 个已失效")
        
        return validated_signals
    
    def generate_validation_report(self, signals: List[Dict[str, any]], 
                                  filename: Optional[str] = None) -> Dict[str, any]:
        """生成信号验证报告
        
        Args:
            signals: 验证后的信号列表
            filename: 报告文件名
            
        Returns:
            报告数据
        """
        report = {
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_signals": len(signals),
            "valid_signals": 0,
            "expired_signals": 0,
            "expiry_reasons": {},
            "detailed_results": []
        }
        
        for signal in signals:
            is_expired = signal.get('expired', False)
            reason = signal.get('expiry_reason', '未知')
            
            if is_expired:
                report['expired_signals'] += 1
                # 统计失效原因
                if reason in report['expiry_reasons']:
                    report['expiry_reasons'][reason] += 1
                else:
                    report['expiry_reasons'][reason] = 1
            else:
                report['valid_signals'] += 1
            
            # 记录详细结果
            detailed_result = {
                "signal_id": signal.get('股票代码', '') + "_" + signal.get('生成时间', '')[:10],
                "symbol": signal.get('股票代码', ''),
                "signal_type": signal.get('信号类型', ''),
                "generated_time": signal.get('生成时间', ''),
                "last_validation": signal.get('last_validation_time', ''),
                "expired": is_expired,
                "reason": reason
            }
            report['detailed_results'].append(detailed_result)
        
        logger.info(f"验证报告生成完成: 有效信号 {report['valid_signals']}, 失效信号 {report['expired_signals']}")
        
        # 如果提供了文件名，保存报告
        if filename:
            try:
                import json
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"验证报告已保存: {filename}")
            except Exception as e:
                logger.error(f"保存验证报告失败: {str(e)}")
        
        return report
    
    def update_stop_loss_threshold(self, new_threshold: float):
        """更新止损阈值
        
        Args:
            new_threshold: 新的止损百分比阈值
        """
        if new_threshold > 0 and new_threshold <= 50:  # 合理的止损范围
            self.stop_loss_percent = new_threshold
            logger.info(f"止损阈值已更新为: {new_threshold}%")
        else:
            logger.warning(f"无效的止损阈值: {new_threshold}，请提供0-50之间的值")
    
    def update_max_valid_days(self, new_days: int):
        """更新信号最大有效天数
        
        Args:
            new_days: 新的最大有效天数
        """
        if new_days > 0 and new_days <= 365:  # 合理的有效期范围
            self.max_valid_days = new_days
            logger.info(f"信号最大有效天数已更新为: {new_days} 天")
        else:
            logger.warning(f"无效的最大有效天数: {new_days}，请提供1-365之间的值")

def main():
    """测试信号验证器"""
    # 创建信号验证器实例
    validator = SignalValidator()
    
    # 创建测试信号
    test_signal = {
        "股票代码": "512660",
        "股票名称": "军工ETF",
        "信号类型": "强烈买入",
        "信号强度": 95,
        "置信度档位": "高置信",
        "周线确认": "高置信确认",
        "日线底分型数量": 3,
        "日线顶分型数量": 1,
        "背驰强度": 0.85,
        "满足条件数": 2,
        "总条件数": 3,
        "交易建议": "强烈买入",
        "是否失效": False,
        "失效原因": "",
        "生成时间": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 验证信号
    validated_signal = validator.validate_signal(test_signal)
    
    # 打印验证结果
    print("\n===== 信号验证结果 =====")
    print(f"信号是否失效: {'是' if validated_signal.get('expired', False) else '否'}")
    print(f"失效原因: {validated_signal.get('expiry_reason', '')}")
    print(f"最后验证时间: {validated_signal.get('last_validation_time', '')}")
    print("\n详细验证结果:")
    for check_name, result in validated_signal.get('validation_results', {}).items():
        print(f"  {check_name}: {'失效' if result['expired'] else '有效'} - {result['reason']}")
    print("=====================\n")

if __name__ == "__main__":
    main()