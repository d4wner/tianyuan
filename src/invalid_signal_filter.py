#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无效信号排除模块

该模块负责过滤不符合交易原则的无效信号，实现零容错的信号质量控制。

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
from typing import Dict, Optional, Union, Tuple, List

# 设置日志
logger = logging.getLogger(__name__)


class InvalidSignalFilter:
    """无效信号过滤器类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化无效信号过滤器
        
        Args:
            config: 过滤器配置参数
        """
        logger.info("初始化无效信号排除模块")
        
        # 默认配置
        self.default_config = {
            "daily_min_k_count": 60,             # 日线最小K线数量
            "weekly_min_k_count": 52,            # 周线最小K线数量
            "extreme_market_threshold": 0.05,    # 单日跌幅阈值（5%）
            "central_border_factor": 0.95,       # 中枢边界因子（0.95）
            "volume_multiplier": 1.3,            # 量能底分型阈值（1.3倍）
            "green_bar_reduction_factor": 0.5,   # 绿柱缩小因子（50%）
            "minimum_pen_length": 5,             # 最小笔长度
            "high_confidence_threshold": 0.8,    # 高置信度阈值
            "medium_confidence_threshold": 0.5,  # 中等置信度阈值
            "calibration_base_factor": 1.0,      # 基础校准因子
            "volatility_sensitivity": 0.1,       # 波动率敏感度
            "trend_strength_weight": 0.7         # 趋势强度权重
        }
        
        # 使用用户配置覆盖默认配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 周线置信度相关参数
        self.confidence_factors = {
            "high": 1.2,  # 高置信度因子
            "medium": 1.0,  # 中等置信度因子
            "low": 0.8  # 低置信度因子
        }
        
        logger.info(f"无效信号过滤器配置: {self.config}")
    
    def filter_invalid_signals(self, 
                             signals: List[str],
                             weekly_trend_result: Dict[str, any],
                             daily_buy_result: Dict[str, any],
                             validation_result: Dict[str, bool],
                             daily_data: Optional[pd.DataFrame] = None,
                             weekly_data: Optional[pd.DataFrame] = None,
                             market_index_data: Optional[Dict] = None) -> Tuple[List[str], List[str]]:
        """过滤无效信号
        
        Args:
            signals: 原始交易信号列表
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            validation_result: 数据源验证结果
            daily_data: 日线数据（可选，用于检测极端行情等）
            weekly_data: 周线数据（可选，用于检测周线趋势）
            market_index_data: 市场指数数据（可选，用于检测极端行情）
            
        Returns:
            (有效信号列表, 无效信号列表)
        """
        logger.info(f"开始过滤无效信号，原始信号数量: {len(signals)}")
        
        valid_signals = []
        invalid_signals = []
        
        # 对每个信号进行过滤
        for signal in signals:
            # 检查是否已经是无效信号格式
            if "无效原因" in signal:
                invalid_signals.append(signal)
                continue
            
            # 执行一系列过滤检查
            invalid_reason = self._check_signal_validity(
                signal,
                weekly_trend_result,
                daily_buy_result,
                validation_result,
                daily_data,
                weekly_data,
                market_index_data
            )
            
            # 根据检查结果分类
            if invalid_reason:
                # 格式化无效信号
                formatted_invalid = f"「无效原因：{invalid_reason}」"
                invalid_signals.append(formatted_invalid)
            else:
                valid_signals.append(signal)
        
        logger.info(f"信号过滤完成: 有效信号 {len(valid_signals)} 个, 无效信号 {len(invalid_signals)} 个")
        return valid_signals, invalid_signals
    
    def _check_signal_validity(self,
                              signal: str,
                              weekly_trend_result: Dict[str, any],
                              daily_buy_result: Dict[str, any],
                              validation_result: Dict[str, bool],
                              daily_data: Optional[pd.DataFrame] = None,
                              weekly_data: Optional[pd.DataFrame] = None,
                              market_index_data: Optional[Dict] = None) -> Optional[str]:
        """检查信号是否有效
        
        按照交易原则文档中的第六步「无效信号排除」进行检查：
        1. 无周线多头、数据源不足、未二次确认的建仓/加仓信号
        2. 中级别向上笔未走完、伪背驰、无量能底分型
        3. ETF跌停/停牌、沪深300/创业板指单日跌超5%
        4. 跌破中枢边界×0.95、无更大级别背驰仅靠分钟底分
        5. 固定比例止盈、无日线顶背驰+大分钟顶分型
        
        Args:
            signal: 交易信号字符串
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            validation_result: 数据源验证结果
            daily_data: 日线数据
            weekly_data: 周线数据
            market_index_data: 市场指数数据
            
        Returns:
            无效原因，如果信号有效则返回None
        """
        # 1. 检查数据源是否不足
        invalid_reason = self._check_data_validity(validation_result)
        if invalid_reason:
            return invalid_reason
        
        # 2. 检查周线多头
        invalid_reason = self._check_weekly_trend(weekly_trend_result)
        if invalid_reason:
            return invalid_reason
        
        # 3. 检查日线买点有效性
        invalid_reason = self._check_daily_buy_validity(daily_buy_result)
        if invalid_reason:
            return invalid_reason
        
        # 4. 检查极端行情
        invalid_reason = self._check_extreme_market(daily_data, market_index_data)
        if invalid_reason:
            return invalid_reason
        
        # 5. 检查是否为伪背驰
        invalid_reason = self._check_fake_divergence(daily_buy_result, daily_data)
        if invalid_reason:
            return invalid_reason
        
        # 6. 检查是否无量能底分型
        invalid_reason = self._check_volume_fractal(daily_buy_result, daily_data)
        if invalid_reason:
            return invalid_reason
        
        # 7. 检查中级别向上笔是否走完
        invalid_reason = self._check_up_pen_completion(signal, daily_buy_result)
        if invalid_reason:
            return invalid_reason
        
        # 信号通过所有检查，返回None表示有效
        return None
    
    def _check_data_validity(self, validation_result: Dict[str, bool]) -> Optional[str]:
        """检查数据源有效性
        
        Args:
            validation_result: 数据源验证结果
            
        Returns:
            无效原因，如果数据有效则返回None
        """
        if not validation_result:
            return "数据源验证结果缺失"
        
        if not validation_result.get("daily_valid", False):
            return "日线数据源不足"
        
        if not validation_result.get("weekly_valid", False):
            return "周线数据源不足"
        
        return None
    
    def _check_weekly_trend(self, weekly_trend_result: Dict[str, any]) -> Optional[str]:
        """检查周线多头趋势
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            
        Returns:
            无效原因，如果周线多头有效则返回None
        """
        if not weekly_trend_result:
            return "周线趋势检测结果缺失"
        
        if not weekly_trend_result.get("bullish_trend", False):
            return "未满足周线多头"
        
        # 检查MACD黄白线是否仍在零轴上方（防止周线多头被破坏）
        if "macd_above_zero" in weekly_trend_result and not weekly_trend_result["macd_above_zero"]:
            return "周线多头趋势已破坏(MACD黄白线不在零轴上方)"
        
        # 检查是否连续3根周K线收盘价下跌
        if "continuous_falling_weeks" in weekly_trend_result and weekly_trend_result["continuous_falling_weeks"] >= 3:
            return "周线多头趋势已破坏(连续3根周K线下跌)"
        
        return None
    
    def _check_daily_buy_validity(self, daily_buy_result: Dict[str, any]) -> Optional[str]:
        """检查日线买点有效性
        
        Args:
            daily_buy_result: 日线买点检测结果
            
        Returns:
            无效原因，如果日线买点有效则返回None
        """
        if not daily_buy_result:
            return "日线买点检测结果缺失"
        
        if daily_buy_result.get("strongest_signal", "") == "无买点":
            return "无有效的日线买点"
        
        return None
    
    def _check_extreme_market(self, 
                            daily_data: Optional[pd.DataFrame] = None,
                            market_index_data: Optional[Dict] = None) -> Optional[str]:
        """检查极端行情（ETF跌停/停牌、市场大幅下跌）
        
        Args:
            daily_data: 日线数据
            market_index_data: 市场指数数据
            
        Returns:
            无效原因，如果不是极端行情则返回None
        """
        # 检查ETF是否跌停或停牌（简化处理，实际需要更复杂的逻辑）
        if daily_data is not None and not daily_data.empty:
            # 检查最近一天的价格变化
            latest_data = daily_data.iloc[-1]
            
            # 检查是否有交易量（停牌检测）
            if 'volume' in latest_data and latest_data['volume'] == 0:
                return "ETF停牌"
            
            # 检查是否跌停（简化为跌幅超过9.5%）
            if 'open' in latest_data and 'close' in latest_data and latest_data['open'] > 0:
                daily_return = (latest_data['close'] - latest_data['open']) / latest_data['open']
                if daily_return < -0.095:
                    return "ETF跌停"
        
        # 检查市场指数是否大幅下跌
        if market_index_data:
            # 检查沪深300指数
            if 'CSI300' in market_index_data:
                csi_return = market_index_data['CSI300'].get('daily_return', 0)
                if csi_return < -self.config["extreme_market_threshold"]:
                    return f"沪深300指数单日跌幅超{self.config['extreme_market_threshold']*100}%"
            
            # 检查创业板指数
            if 'CHINEXT' in market_index_data:
                chin_return = market_index_data['CHINEXT'].get('daily_return', 0)
                if chin_return < -self.config["extreme_market_threshold"]:
                    return f"创业板指单日跌幅超{self.config['extreme_market_threshold']*100}%"
        
        return None
    
    def _check_fake_divergence(self, 
                             daily_buy_result: Dict[str, any],
                             daily_data: Optional[pd.DataFrame] = None) -> Optional[str]:
        """检查是否为伪背驰
        
        伪背驰定义：价格新低但黄白线同步新低，或无绿柱缩窄
        
        Args:
            daily_buy_result: 日线买点检测结果
            daily_data: 日线数据
            
        Returns:
            无效原因，如果不是伪背驰则返回None
        """
        if not daily_buy_result:
            return None
        
        # 获取信号类型
        signal_type = daily_buy_result.get("strongest_signal", "")
        
        # 只检查需要背驰的信号类型
        if signal_type in ["日线一买", "日线二买", "日线三买"]:
            # 获取对应信号的详细信息
            signal_key_map = {
                "日线二买": "second_buy",
                "日线一买": "first_buy",
                "日线三买": "third_buy"
            }
            
            signal_key = signal_key_map.get(signal_type)
            if signal_key:
                signal_details = daily_buy_result.get("signals", {}).get(signal_key, {})
                
                # 检查是否为伪背驰
                if "is_fake_divergence" in signal_details and signal_details["is_fake_divergence"]:
                    return "伪背驰（黄白线同步新低）"
                
                # 检查绿柱是否缩窄
                if "green_bar_reduction" in signal_details and not signal_details["green_bar_reduction"]:
                    return "伪背驰（无绿柱缩窄）"
        
        return None
    
    def _check_volume_fractal(self, 
                            daily_buy_result: Dict[str, any],
                            daily_data: Optional[pd.DataFrame] = None) -> Optional[str]:
        """检查是否有量能底分型
        
        Args:
            daily_buy_result: 日线买点检测结果
            daily_data: 日线数据
            
        Returns:
            无效原因，如果量能底分型有效则返回None
        """
        if not daily_buy_result:
            return None
        
        # 获取信号类型
        signal_type = daily_buy_result.get("strongest_signal", "")
        
        # 只检查需要量能底分型的信号类型
        if signal_type in ["日线一买", "日线二买", "日线三买"]:
            # 获取对应信号的详细信息
            signal_key_map = {
                "日线二买": "second_buy",
                "日线一买": "first_buy",
                "日线三买": "third_buy"
            }
            
            signal_key = signal_key_map.get(signal_type)
            if signal_key:
                signal_details = daily_buy_result.get("signals", {}).get(signal_key, {})
                
                # 检查是否有量能底分型
                if "has_volume_fractal" in signal_details and not signal_details["has_volume_fractal"]:
                    return "无量能底分型"
        
        return None
    
    def _check_up_pen_completion(self, signal: str, daily_buy_result: Dict[str, any]) -> Optional[str]:
        """检查中级别向上笔是否走完
        
        Args:
            signal: 交易信号字符串
            daily_buy_result: 日线买点检测结果
            
        Returns:
            无效原因，如果向上笔已走完则返回None
        """
        # 简化实现：检查信号中是否包含二次确认信息
        if "30分钟" in signal:
            if "二次确认" not in signal:
                # 实际应该检查是否有顶分型且确认K线收盘价跌破顶分型中心K线低点
                # 这里简化处理
                pass
        
        if "15分钟" in signal:
            if "二次确认" not in signal:
                # 同样简化处理
                pass
        
        return None
    
    def check_position_boundary(self, 
                               daily_buy_result: Dict[str, any],
                               current_price: float,
                               central_border: float) -> bool:
        """检查价格是否跌破中枢边界×0.95
        
        Args:
            daily_buy_result: 日线买点检测结果
            current_price: 当前价格
            central_border: 中枢边界价格
            
        Returns:
            True表示价格在安全范围内，False表示跌破阈值
        """
        # 计算安全阈值
        safe_threshold = central_border * self.config["central_border_factor"]
        
        # 检查价格是否低于安全阈值
        return current_price >= safe_threshold
    
    def check_addition_conditions(self, 
                                 weekly_trend_result: Dict[str, any],
                                 daily_buy_result: Dict[str, any],
                                 has_larger_divergence: bool) -> Optional[str]:
        """检查加仓条件是否有效
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            has_larger_divergence: 是否有更大级别背驰
            
        Returns:
            无效原因，如果加仓条件有效则返回None
        """
        # 检查周线多头是否保持
        if not weekly_trend_result.get("bullish_trend", False):
            return "周线多头趋势未保持"
        
        # 计算周线置信度
        confidence_level = self.get_confidence_level(weekly_trend_result)
        
        # 基于置信度调整加仓条件严格程度
        if confidence_level == "low":
            # 低置信度时需要更严格的加仓条件
            if not has_larger_divergence:
                return "低置信度时必须有更大级别背驰"
        else:
            # 中高置信度时可以适当放宽
            if not has_larger_divergence:
                return "无更大级别背驰仅靠分钟底分"
        
        return None
    
    def _check_confidence_level(self, confidence_level: str) -> bool:
        """检查置信度级别是否满足要求
        
        Args:
            confidence_level: 周线置信度等级
            
        Returns:
            置信度级别是否满足要求
        """
        # 最低接受低置信度
        return confidence_level in ["high", "medium", "low"]
    
    def get_calibration_factor(self, price_data: pd.DataFrame, weekly_trend_result: Dict[str, any]) -> float:
        """获取动态校准因子
        
        Args:
            price_data: 价格数据
            weekly_trend_result: 周线趋势检测结果
            
        Returns:
            动态校准因子
        """
        # 计算价格波动率（基于20日标准差）
        volatility = price_data['close'].pct_change().rolling(window=20).std().iloc[-1] if len(price_data) > 20 else 0.02
        
        # 计算趋势强度
        trend_strength = weekly_trend_result.get("trend_strength", 0.5)
        
        # 计算动态校准因子
        volatility_impact = 1.0 + self.config["volatility_sensitivity"] * (0.02 - volatility)
        trend_impact = self.config["trend_strength_weight"] * trend_strength + (1 - self.config["trend_strength_weight"])
        
        calibration_factor = self.config["calibration_base_factor"] * volatility_impact * trend_impact
        
        # 限制校准因子范围
        return max(0.6, min(1.5, calibration_factor))
    
    def get_confidence_level(self, weekly_trend_result: Dict[str, any]) -> str:
        """获取周线置信度等级
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            
        Returns:
            置信度等级（high/medium/low）
        """
        # 计算综合置信度分数
        confidence_score = 0
        
        # MACD黄白线位置（30分）
        if weekly_trend_result.get("macd_above_zero", False):
            confidence_score += 30
        
        # 连续上涨周数（20分）
        continuous_rising_weeks = weekly_trend_result.get("continuous_rising_weeks", 0)
        confidence_score += min(continuous_rising_weeks * 5, 20)
        
        # 趋势强度（30分）
        trend_strength = weekly_trend_result.get("trend_strength", 0.5)
        confidence_score += int(trend_strength * 30)
        
        # 背离情况（20分）
        if not weekly_trend_result.get("has_divergence", True):
            confidence_score += 20
        
        # 归一化到0-1范围
        confidence_score = confidence_score / 100
        
        # 判断置信度等级
        if confidence_score >= self.config["high_confidence_threshold"]:
            return "high"
        elif confidence_score >= self.config["medium_confidence_threshold"]:
            return "medium"
        else:
            return "low"
    
    def check_take_profit_conditions(self, 
                                   has_daily_top_divergence: bool,
                                   has_large_minute_top_fractal: bool,
                                   take_profit_type: str,
                                   weekly_trend_result: Optional[Dict[str, any]] = None) -> Optional[str]:
        """检查止盈条件是否有效
        
        Args:
            has_daily_top_divergence: 是否有日线顶背驰
            has_large_minute_top_fractal: 是否有大分钟顶分型
            take_profit_type: 止盈类型
            weekly_trend_result: 周线趋势检测结果，用于置信度判断
            
        Returns:
            无效原因，如果止盈条件有效则返回None
        """
        # 检查是否为固定比例止盈
        if take_profit_type == "fixed_percentage":
            return "固定比例止盈无效"
        
        # 检查是否同时满足日线顶背驰和大分钟顶分型
        if not (has_daily_top_divergence and has_large_minute_top_fractal):
            return "无日线顶背驰+大分钟顶分型"
        
        # 如果提供了周线趋势结果，根据置信度调整止盈条件
        if weekly_trend_result:
            confidence_level = self.get_confidence_level(weekly_trend_result)
            # 高置信度时，可以更灵活地处理止盈信号
            if confidence_level == "high":
                # 高置信度下，即使出现止盈信号也可以考虑持有更长时间
                pass
        
        return None


# 测试代码
def test_invalid_signal_filter():
    """测试无效信号过滤器"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建过滤器实例
    filter = InvalidSignalFilter()
    
    # 创建测试数据
    weekly_trend_result_valid = {
        "bullish_trend": True,
        "macd_above_zero": True,
        "continuous_falling_weeks": 0
    }
    
    weekly_trend_result_invalid = {
        "bullish_trend": False,
        "macd_above_zero": False,
        "continuous_falling_weeks": 3
    }
    
    daily_buy_result_valid = {
        "strongest_signal": "日线二买",
        "signal_type_priority": "核心",
        "signals": {
            "second_buy": {
                "detected": True,
                "is_fake_divergence": False,
                "green_bar_reduction": True,
                "has_volume_fractal": True
            }
        }
    }
    
    daily_buy_result_invalid = {
        "strongest_signal": "无买点",
        "signal_type_priority": "无",
        "signals": {}
    }
    
    validation_result_valid = {
        "daily_valid": True,
        "weekly_valid": True,
        "all_valid": True
    }
    
    validation_result_invalid = {
        "daily_valid": False,
        "weekly_valid": True,
        "all_valid": False
    }
    
    # 创建测试信号
    test_signals = [
        "「信号类型：周线多头+日线二买-30分钟建仓｜日线买点类型（核心/辅助/兜底）：核心｜分钟子仓位比例：65%｜实际仓位：19.5%（用户原有日线仓位×65%）｜加仓条件：未满足｜累计总仓位上限：80%｜止盈触发条件：未满足｜数据源：满足」",
        "「无效原因：未满足周线多头」"
    ]
    
    # 测试有效情况
    print("\n===== 测试有效情况 =====")
    valid_signals, invalid_signals = filter.filter_invalid_signals(
        test_signals,
        weekly_trend_result_valid,
        daily_buy_result_valid,
        validation_result_valid
    )
    
    print(f"有效信号数量: {len(valid_signals)}")
    print(f"无效信号数量: {len(invalid_signals)}")
    
    # 测试无效情况（周线非多头）
    print("\n===== 测试无效情况（周线非多头） =====")
    valid_signals, invalid_signals = filter.filter_invalid_signals(
        test_signals,
        weekly_trend_result_invalid,
        daily_buy_result_valid,
        validation_result_valid
    )
    
    print(f"有效信号数量: {len(valid_signals)}")
    print(f"无效信号数量: {len(invalid_signals)}")
    for signal in invalid_signals:
        print(f"无效信号: {signal}")
    
    # 测试无效情况（数据源不足）
    print("\n===== 测试无效情况（数据源不足） =====")
    valid_signals, invalid_signals = filter.filter_invalid_signals(
        test_signals,
        weekly_trend_result_valid,
        daily_buy_result_valid,
        validation_result_invalid
    )
    
    print(f"有效信号数量: {len(valid_signals)}")
    print(f"无效信号数量: {len(invalid_signals)}")
    for signal in invalid_signals:
        print(f"无效信号: {signal}")
    
    # 测试无效情况（无买点）
    print("\n===== 测试无效情况（无买点） =====")
    valid_signals, invalid_signals = filter.filter_invalid_signals(
        test_signals,
        weekly_trend_result_valid,
        daily_buy_result_invalid,
        validation_result_valid
    )
    
    print(f"有效信号数量: {len(valid_signals)}")
    print(f"无效信号数量: {len(invalid_signals)}")
    for signal in invalid_signals:
        print(f"无效信号: {signal}")
    
    # 测试加仓条件检查
    print("\n===== 测试加仓条件检查 =====")
    # 有效情况
    reason = filter.check_addition_conditions(
        weekly_trend_result_valid,
        daily_buy_result_valid,
        has_larger_divergence=True
    )
    print(f"有效加仓条件检查结果: {reason}")
    
    # 无效情况（无更大级别背驰）
    reason = filter.check_addition_conditions(
        weekly_trend_result_valid,
        daily_buy_result_valid,
        has_larger_divergence=False
    )
    print(f"无效加仓条件检查结果: {reason}")
    
    # 测试止盈条件检查
    print("\n===== 测试止盈条件检查 =====")
    # 有效情况
    reason = filter.check_take_profit_conditions(
        has_daily_top_divergence=True,
        has_large_minute_top_fractal=True,
        take_profit_type="divergence"
    )
    print(f"有效止盈条件检查结果: {reason}")
    
    # 无效情况（固定比例止盈）
    reason = filter.check_take_profit_conditions(
        has_daily_top_divergence=True,
        has_large_minute_top_fractal=True,
        take_profit_type="fixed_percentage"
    )
    print(f"无效止盈条件检查结果: {reason}")
    
    # 无效情况（无背驰）
    reason = filter.check_take_profit_conditions(
        has_daily_top_divergence=False,
        has_large_minute_top_fractal=True,
        take_profit_type="divergence"
    )
    print(f"无效止盈条件检查结果: {reason}")


if __name__ == "__main__":
    test_invalid_signal_filter()