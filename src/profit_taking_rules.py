#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
止盈规则模块

该模块负责实现止盈规则，仅绑定缠论级别信号，禁止固定比例止盈。

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union

# 设置日志
logger = logging.getLogger(__name__)


class ProfitTakingRules:
    """止盈规则处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化止盈规则处理器
        
        Args:
            config: 配置参数
        """
        logger.info("初始化止盈规则模块")
        
        # 默认配置
        self.default_config = {
            # 日线顶背驰判断参数
            "red_bar_reduction_factor": 0.5,  # 红柱缩小因子（50%）
            "rising_segment_min_length": 15,  # 上涨段最小长度
            
            # 分钟级别顶分型判断参数
            "min_tops_confirmation_bars": 2,  # 顶分型确认所需K线数
            "minute_bar_min_length": 5,       # 分钟K线最小数量
            
            # 止盈仓位比例配置
            "stop_profit_ratios": {
                "30min": 0.6,  # 30分钟顶分型止盈60%
                "15min": 0.4   # 15分钟顶分型止盈40%
            },
            
            # 止盈优先级配置（True表示需要更严格的确认）
            "strict_stop_profit_for": ["日线一买", "日线三买"],
            "moderate_stop_profit_for": ["日线二买"],  # 可适度放宽
            
            # 5分钟级别过滤（不触发止盈）
            "ignore_5min_tops": True,
            
            # MACD参数
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            
            # 周线置信度等级对应的止盈比例因子（与原比例相乘）
            "confidence_level_factors": {
                "极强": 0.7,  # 置信度越高，止盈比例越小
                "强": 0.8,
                "较强": 0.9,
                "中": 1.0,
                "较弱": 1.1,
                "弱": 1.2,
                "极弱": 1.3
            },
            
            # 无效信号排除配置
            "invalid_signal_conditions": {
                "min_profit_rate_for_stop_profit": 0.02,  # 最小盈利比例为2%
                "max_consecutive_loss_trades": 3,         # 最大连续亏损交易次数
                "min_position_ratio_for_stop_profit": 0.05  # 最小仓位比例为5%
            },
            
            # 动态校准参数
            "dynamic_calibration": {
                "enable": True,                        # 启用动态校准
                "price_volatility_factor": 0.2,        # 价格波动率影响因子
                "position_size_factor": 0.3,           # 仓位大小影响因子
                "market_condition_factor": 1.0         # 市场条件因子
            }
        }
        
        # 使用用户配置覆盖默认配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        logger.info(f"止盈规则配置: {self.config}")
    
    def check_stop_profit_conditions(self, 
                                  daily_data: pd.DataFrame,
                                  minute_data: Dict[str, pd.DataFrame],
                                  daily_signal_type: str,
                                  position_ratios: Dict[str, float],
                                  weekly_trend_result: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        """检查止盈触发条件
        
        止盈触发前提（需同时满足）：
        1. 日线级别形成顶背驰：日线上涨段价格新高+MACD黄白线不新高+红柱最大高度＜前一段50%
        2. 大分钟级别形态确认：30分钟/15分钟出现顶分型，且确认K线收盘价跌破分型中心低点
        
        Args:
            daily_data: 日线数据
            minute_data: 分钟级别数据字典，包含'30min'和'15min'数据
            daily_signal_type: 日线买点类型（日线二买/一买/三买/反抽）
            position_ratios: 仓位比例字典，包含各分钟级别的仓位
            
        Returns:
            止盈条件检查结果字典
        """
        logger.info(f"检查止盈触发条件，日线买点类型：{daily_signal_type}")
        
        # 初始化周线置信度信息
        confidence_level = "中"
        confidence_score = 0.5
        confidence_factor = 1.0
        
        # 如果提供了周线趋势结果，获取置信度信息
        if weekly_trend_result:
            confidence_level = weekly_trend_result.get("confidence_level", "中")
            confidence_score = weekly_trend_result.get("confidence_score", 0.5)
            # 获取置信度对应的止盈比例因子
            confidence_factor = self.config["confidence_level_factors"].get(
                confidence_level, 1.0
            )
            logger.info(f"使用周线置信度等级：{confidence_level}，对应的止盈因子：{confidence_factor}")
        
        result = {
            "should_take_profit": False,
            "daily_divergence_detected": False,
            "minute_confirmation": None,  # "30min" 或 "15min" 或 None
            "stop_profit_ratio": 0.0,
            "affected_positions": {},
            "divergence_info": {},
            "confirmation_info": {},
            "failing_reasons": [],
            "weekly_confidence_level": confidence_level,
            "weekly_confidence_factor": confidence_factor
        }
        
        # 1. 检查日线顶背驰
        daily_divergence_result = self.detect_daily_top_divergence(daily_data, daily_signal_type)
        result["daily_divergence_detected"] = daily_divergence_result.get("has_divergence", False)
        result["divergence_info"] = daily_divergence_result
        
        # 计算背驰强度分数
        divergence_strength_score = 0.0
        if result["daily_divergence_detected"]:
            # 尝试从日线顶背驰检测结果中获取背驰强度信息
            if "macd_values" in daily_divergence_result and "red_bar_heights" in daily_divergence_result:
                macd_values = daily_divergence_result["macd_values"]
                red_bar_heights = daily_divergence_result["red_bar_heights"]
                
                # 计算背驰强度分数（基于MACD峰值差异和红柱高度差异）
                if macd_values and "latest_peak" in macd_values and "previous_peak" in macd_values:
                    if macd_values["previous_peak"] > 0:  # 避免除以0
                        macd_diff_ratio = 1 - (macd_values["latest_peak"] / macd_values["previous_peak"])
                        divergence_strength_score += macd_diff_ratio * 0.5
                
                if red_bar_heights and "recent_max" in red_bar_heights and "earlier_max" in red_bar_heights:
                    if red_bar_heights["earlier_max"] > 0:  # 避免除以0
                        red_bar_diff_ratio = 1 - (red_bar_heights["recent_max"] / red_bar_heights["earlier_max"])
                        divergence_strength_score += red_bar_diff_ratio * 0.5
                
                # 限制在0-1范围内
                divergence_strength_score = min(1.0, max(0.0, divergence_strength_score))
        
        result["divergence_strength_score"] = divergence_strength_score
        
        if not result["daily_divergence_detected"]:
            reason = "未检测到日线顶背驰"
            result["failing_reasons"].append(reason)
            logger.warning(f"止盈条件1不满足：{reason}")
        else:
            logger.info("止盈条件1满足：检测到日线顶背驰")
        
        # 2. 检查大分钟级别形态确认
        # 优先检查30分钟顶分型
        minute_levels_to_check = ["30min", "15min"]
        confirmed_minute_level = None
        confirmation_info = {}
        
        for level in minute_levels_to_check:
            if level not in minute_data or minute_data[level] is None or minute_data[level].empty:
                logger.warning(f"{level}数据为空，跳过检查")
                continue
            
            # 检测顶分型确认
            top_fractal_result = self.detect_confirmed_top_fractal(minute_data[level], level)
            
            if top_fractal_result.get("has_confirmed_top", False):
                confirmed_minute_level = level
                confirmation_info = top_fractal_result
                logger.info(f"止盈条件2满足：{level}顶分型已确认")
                break
        
        result["minute_confirmation"] = confirmed_minute_level
        result["confirmation_info"] = confirmation_info
        
        if not confirmed_minute_level:
            reason = "未检测到大分钟级别顶分型确认"
            result["failing_reasons"].append(reason)
            logger.warning(f"止盈条件2不满足：{reason}")
        
        # 3. 综合判断是否触发止盈
        if result["daily_divergence_detected"] and result["minute_confirmation"]:
            # 根据分钟级别确定基础止盈比例
            base_stop_profit_ratio = self.config["stop_profit_ratios"].get(
                confirmed_minute_level, 0.0
            )
            
            # 应用周线置信度因子
            final_stop_profit_ratio = base_stop_profit_ratio * confidence_factor
            
            # 计算动态校准因子
            dynamic_factor = 1.0
            if self.config["dynamic_calibration"]["enable"]:
                # 根据价格波动率调整
                volatility_factor = self._calculate_price_volatility(daily_data)
                # 根据信号强度调整
                signal_strength = self._calculate_signal_strength(
                    result["daily_divergence_detected"], 
                    result["minute_confirmation"] is not None, 
                    divergence_strength_score > 0.7,  # 简化判断是否为强势背驰
                    divergence_strength_score
                )
                
                dynamic_factor = 1.0 + (volatility_factor - 0.5) * self.config["dynamic_calibration"]["price_volatility_factor"]
                dynamic_factor *= 0.8 + signal_strength * 0.2
                dynamic_factor *= self.config["dynamic_calibration"]["market_condition_factor"]
                
                # 限制动态因子在合理范围内
                dynamic_factor = max(0.7, min(1.5, dynamic_factor))
                logger.info(f"动态校准因子计算结果：{dynamic_factor:.2f}")
            
            # 应用动态校准因子
            final_stop_profit_ratio *= dynamic_factor
            
            # 限制止盈比例在合理范围内
            final_stop_profit_ratio = max(0.05, min(0.8, final_stop_profit_ratio))
            
            # 根据买点类型可能调整止盈条件严格度（仅对日线二买适度放宽）
            if daily_signal_type in self.config["moderate_stop_profit_for"]:
                logger.info(f"{daily_signal_type}止盈条件适度放宽")
            
            # 确定受影响的仓位
            affected_positions = {}
            if position_ratios:
                # 假设仓位字典中包含分钟级别的仓位信息
                for pos_type, ratio in position_ratios.items():
                    if pos_type.startswith(confirmed_minute_level.replace('min', '')):
                        affected_positions[pos_type] = ratio
            
            # 更新结果
            result["should_take_profit"] = True
            result["stop_profit_ratio"] = final_stop_profit_ratio
            result["base_stop_profit_ratio"] = base_stop_profit_ratio
            result["dynamic_calibration_factor"] = dynamic_factor
            result["affected_positions"] = affected_positions
            logger.info(f"止盈触发：日线顶背驰 + {confirmed_minute_level}顶分型，基础止盈比例：{base_stop_profit_ratio*100}%，应用因子后：{final_stop_profit_ratio*100}%")
        else:
            logger.warning(f"止盈未触发，失败原因：{', '.join(result['failing_reasons'])}")
        
        return result
    
    def detect_daily_top_divergence(self, 
                                  daily_data: pd.DataFrame,
                                  daily_signal_type: str) -> Dict[str, any]:
        """检测日线级别顶背驰
        
        日线顶背驰：日线上涨段价格新高+MACD黄白线不新高+红柱最大高度＜前一段50%
        
        Args:
            daily_data: 日线数据
            daily_signal_type: 日线买点类型（影响严格度）
            
        Returns:
            顶背驰检测结果字典
        """
        logger.info(f"检测日线级别顶背驰，日线买点类型：{daily_signal_type}")
        
        result = {
            "has_divergence": False,
            "reason": "",
            "price_higher": False,
            "macd_lower": False,
            "red_bar_reduced": False,
            "macd_values": {},
            "red_bar_heights": {}
        }
        
        if daily_data is None or daily_data.empty:
            result["reason"] = "日线数据为空"
            logger.warning(result["reason"])
            return result
        
        # 确保数据有足够的K线
        if len(daily_data) < self.config["rising_segment_min_length"]:
            result["reason"] = f"日线K线数量不足（{len(daily_data)} < {self.config['rising_segment_min_length']}）"
            logger.warning(result["reason"])
            return result
        
        # 计算MACD
        try:
            daily_data = self._calculate_macd(daily_data)
        except Exception as e:
            result["reason"] = f"计算MACD失败：{str(e)}"
            logger.error(result["reason"])
            return result
        
        # 简化实现：假设最近的数据就是上涨段
        # 实际应该识别上涨段和前一个上涨段
        
        # 获取最近部分数据
        recent_data = daily_data.tail(20).copy()  # 取最近20根K线作为分析区间
        
        # 检查价格是否创新高
        recent_high = recent_data["high"].max()
        historical_high = daily_data["high"].max()
        
        if recent_high == historical_high:
            result["price_higher"] = True
            logger.info(f"价格创新高：{recent_high}")
        else:
            result["reason"] = "价格未创新高"
            logger.info(result["reason"])
            return result
        
        # 查找价格新高对应的位置
        price_high_idx = recent_data["high"].idxmax()
        
        # 检查MACD是否未创新高
        # 简化：比较最近的MACD值和前一个高点的MACD值
        recent_macd_peaks = self._find_macd_peaks(recent_data, "macd")
        
        if len(recent_macd_peaks) >= 2:
            # 获取最近的两个MACD峰值
            latest_peak = recent_macd_peaks[-1]
            previous_peak = recent_macd_peaks[-2]
            
            if latest_peak[1] < previous_peak[1]:
                result["macd_lower"] = True
                logger.info(f"MACD未创新高：最近峰值{latest_peak[1]} < 前峰值{previous_peak[1]}")
                result["macd_values"] = {
                    "latest_peak": latest_peak[1],
                    "previous_peak": previous_peak[1]
                }
            else:
                result["reason"] = "MACD同步新高，无背驰"
                logger.info(result["reason"])
                return result
        else:
            result["reason"] = "MACD峰值不足，无法判断"
            logger.warning(result["reason"])
            return result
        
        # 检查红柱高度是否缩小
        # 简化：计算最近的红柱高度和前一个上涨段的红柱高度
        recent_red_bars = recent_data[recent_data["macd_hist"] > 0]["macd_hist"]
        
        if not recent_red_bars.empty:
            recent_max_red_bar = recent_red_bars.max()
            
            # 查找前一个上涨段的红柱
            # 简化：取前面部分数据
            earlier_data = daily_data.head(len(daily_data) - len(recent_data))
            earlier_red_bars = earlier_data[earlier_data["macd_hist"] > 0]["macd_hist"]
            
            if not earlier_red_bars.empty:
                earlier_max_red_bar = earlier_red_bars.max()
                
                if recent_max_red_bar < earlier_max_red_bar * self.config["red_bar_reduction_factor"]:
                    result["red_bar_reduced"] = True
                    logger.info(f"红柱高度缩小：当前最大高度{recent_max_red_bar} < 前一段最大高度{earlier_max_red_bar} × {self.config['red_bar_reduction_factor']}")
                    result["red_bar_heights"] = {
                        "recent_max": recent_max_red_bar,
                        "earlier_max": earlier_max_red_bar,
                        "threshold": earlier_max_red_bar * self.config["red_bar_reduction_factor"]
                    }
                else:
                    result["reason"] = "红柱高度未明显缩小"
                    logger.info(result["reason"])
                    return result
            else:
                result["reason"] = "前一段无红柱数据"
                logger.warning(result["reason"])
                return result
        else:
            result["reason"] = "最近无红柱数据"
            logger.warning(result["reason"])
            return result
        
        # 综合判断
        if result["price_higher"] and result["macd_lower"] and result["red_bar_reduced"]:
            result["has_divergence"] = True
            result["reason"] = "日线顶背驰确认"
            logger.info(result["reason"])
        
        return result
    
    def detect_confirmed_top_fractal(self, 
                                   minute_data: pd.DataFrame,
                                   minute_level: str) -> Dict[str, any]:
        """检测大分钟级别顶分型确认
        
        确认条件：顶分型且确认K线收盘价跌破分型中心K线低点
        
        Args:
            minute_data: 分钟级别数据
            minute_level: 分钟级别（"30min"或"15min"）
            
        Returns:
            顶分型确认结果字典
        """
        logger.info(f"检测{minute_level}级别顶分型确认")
        
        result = {
            "has_confirmed_top": False,
            "top_fractal_index": None,
            "confirmation_bar_index": None,
            "top_fractal_price": 0.0,
            "confirmation_price": 0.0,
            "reason": ""
        }
        
        if minute_data is None or minute_data.empty:
            result["reason"] = f"{minute_level}数据为空"
            logger.warning(result["reason"])
            return result
        
        # 确保有足够的K线
        if len(minute_data) < self.config["minute_bar_min_length"]:
            result["reason"] = f"{minute_level}K线数量不足（{len(minute_data)} < {self.config['minute_bar_min_length']}）"
            logger.warning(result["reason"])
            return result
        
        # 识别顶分型
        top_fractals = self._identify_top_fractal(minute_data)
        
        if not top_fractals:
            result["reason"] = f"{minute_level}未发现顶分型"
            logger.info(result["reason"])
            return result
        
        # 获取最近的顶分型
        last_top_idx = max(top_fractals)
        
        # 检查顶分型之后是否有足够的K线进行确认
        bars_after_top = len(minute_data) - last_top_idx - 1
        
        if bars_after_top < self.config["min_tops_confirmation_bars"]:
            result["reason"] = f"顶分型后K线数量不足，无法确认"
            logger.info(result["reason"])
            return result
        
        # 获取顶分型中心K线的低点
        top_center_low = minute_data.iloc[last_top_idx].get("low", 0)
        
        # 检查确认K线是否跌破顶分型中心K线低点
        confirmation_found = False
        confirmation_idx = -1
        confirmation_price = 0.0
        
        # 从顶分型的下一根K线开始检查
        for i in range(last_top_idx + 1, len(minute_data)):
            close_price = minute_data.iloc[i].get("close", 0)
            
            if close_price < top_center_low:
                confirmation_found = True
                confirmation_idx = i
                confirmation_price = close_price
                break
        
        if confirmation_found:
            # 确保确认K线是在顶分型后的有效范围内
            bars_from_top = confirmation_idx - last_top_idx
            
            if bars_from_top <= self.config["min_tops_confirmation_bars"] + 5:  # 允许5根K线的范围
                result["has_confirmed_top"] = True
                result["top_fractal_index"] = last_top_idx
                result["confirmation_bar_index"] = confirmation_idx
                result["top_fractal_price"] = top_center_low
                result["confirmation_price"] = confirmation_price
                result["reason"] = f"{minute_level}顶分型确认：确认K线收盘价{confirmation_price} < 顶分型中心低点{top_center_low}"
                logger.info(result["reason"])
            else:
                result["reason"] = "顶分型确认K线距离过远，视为无效"
                logger.info(result["reason"])
        else:
            result["reason"] = "顶分型后未出现确认K线"
            logger.info(result["reason"])
        
        return result
    
    def calculate_stop_profit_position(self, 
                                     stop_profit_result: Dict[str, any],
                                     current_positions: Dict[str, float]) -> Dict[str, float]:
        """计算止盈仓位
        
        Args:
            stop_profit_result: 止盈检查结果
            current_positions: 当前各类型仓位字典
            
        Returns:
            止盈仓位计算结果字典
        """
        logger.info("计算止盈仓位")
        
        result = {
            "stop_profit_amount": 0.0,
            "remaining_position": 0.0,
            "positions_to_reduce": {},
            "original_total": 0.0,
            "stop_profit_ratio_applied": 0.0,
            "valid": False
        }
        
        # 检查是否应该止盈
        if not stop_profit_result.get("should_take_profit", False):
            logger.warning("止盈条件不满足，无法计算止盈仓位")
            return result
        
        # 获取止盈比例和相关因子
        final_stop_profit_ratio = stop_profit_result.get("stop_profit_ratio", 0.0)
        base_stop_profit_ratio = stop_profit_result.get("base_stop_profit_ratio", 0.0)
        confidence_factor = stop_profit_result.get("weekly_confidence_factor", 1.0)
        dynamic_factor = stop_profit_result.get("dynamic_calibration_factor", 1.0)
        affected_positions = stop_profit_result.get("affected_positions", {})
        minute_level = stop_profit_result.get("minute_confirmation", "")
        
        # 计算当前总仓位
        original_total = sum(current_positions.values())
        result["original_total"] = original_total
        
        # 计算需要减少的仓位
        positions_to_reduce = {}
        total_to_reduce = 0.0
        
        if affected_positions:
            # 如果指定了受影响的仓位，只针对这些仓位止盈
            for pos_type, pos_ratio in affected_positions.items():
                reduction = pos_ratio * final_stop_profit_ratio
                positions_to_reduce[pos_type] = reduction
                total_to_reduce += reduction
        else:
            # 否则，对与分钟级别相关的所有仓位应用止盈比例
            for pos_type, pos_ratio in current_positions.items():
                if minute_level and minute_level.replace('min', '') in pos_type:
                    reduction = pos_ratio * final_stop_profit_ratio
                    positions_to_reduce[pos_type] = reduction
                    total_to_reduce += reduction
        
        # 确保止盈后剩余仓位不小于最小阈值
        min_remaining_position = self.config["invalid_signal_conditions"]["min_position_ratio_for_stop_profit"]
        max_stop_profit_amount = original_total - min_remaining_position
        total_to_reduce = min(total_to_reduce, max_stop_profit_amount)
        
        # 计算实际止盈比例
        actual_stop_profit_ratio = total_to_reduce / original_total if original_total > 0 else 0
        
        # 更新结果
        result["stop_profit_amount"] = total_to_reduce
        result["remaining_position"] = original_total - total_to_reduce
        result["positions_to_reduce"] = positions_to_reduce
        result["base_stop_profit_ratio"] = base_stop_profit_ratio
        result["final_stop_profit_ratio"] = final_stop_profit_ratio
        result["actual_stop_profit_ratio"] = actual_stop_profit_ratio
        result["weekly_confidence_factor"] = confidence_factor
        result["dynamic_calibration_factor"] = dynamic_factor
        result["valid"] = total_to_reduce > 0
        
        logger.info(f"止盈计算结果：止盈量={total_to_reduce*100}%, 剩余仓位={result['remaining_position']*100}%")
        logger.info(f"止盈比例详情：基础比例={base_stop_profit_ratio*100}%, 周线因子={confidence_factor}, 动态因子={dynamic_factor}, 实际应用比例={actual_stop_profit_ratio*100}%")
        
        # 打印详细的仓位减少信息
        if positions_to_reduce:
            for pos_type, reduction in positions_to_reduce.items():
                logger.info(f"  {pos_type}: 减少 {reduction*100}%")
        
        return result
    
    def generate_stop_profit_signal(self, 
                                  stop_profit_result: Dict[str, any],
                                  position_result: Dict[str, float]) -> Optional[Dict[str, any]]:
        """生成止盈交易信号
        
        Args:
            stop_profit_result: 止盈检查结果
            position_result: 止盈仓位计算结果
            
        Returns:
            止盈交易信号，如果不满足止盈条件则返回None
        """
        logger.info("生成止盈交易信号")
        
        # 检查是否应该止盈且计算有效
        if not stop_profit_result.get("should_take_profit", False) or not position_result.get("valid", False):
            logger.warning("止盈条件不满足或计算无效，不生成止盈信号")
            return None
        
        # 检查无效止盈条件
        invalid_conditions = self.check_invalid_stop_profit_conditions(
            stop_profit_result, position_result
        )
        
        if invalid_conditions.get("is_invalid", False):
            logger.warning(f"止盈信号无效：{invalid_conditions.get('invalid_reason', '')}")
            return None
        
        # 构造止盈信号
        signal = {
            "signal_type": "止盈",
            "trigger_reason": f"日线顶背驰+{stop_profit_result.get('minute_confirmation', '')}顶分型",
            "stop_profit_amount": position_result.get("stop_profit_amount", 0.0),
            "remaining_position": position_result.get("remaining_position", 0.0),
            "base_stop_profit_ratio": position_result.get("base_stop_profit_ratio", 0.0),
            "final_stop_profit_ratio": position_result.get("final_stop_profit_ratio", 0.0),
            "actual_stop_profit_ratio": position_result.get("actual_stop_profit_ratio", 0.0),
            "weekly_confidence_level": stop_profit_result.get("weekly_confidence_level", "中"),
            "weekly_confidence_factor": position_result.get("weekly_confidence_factor", 1.0),
            "dynamic_calibration_factor": position_result.get("dynamic_calibration_factor", 1.0),
            "affected_positions": position_result.get("positions_to_reduce", {}),
            "divergence_details": stop_profit_result.get("divergence_info", {}),
            "confirmation_details": stop_profit_result.get("confirmation_info", {}),
            "divergence_strength_score": stop_profit_result.get("divergence_strength_score", 0),
            "validity_check": {"is_valid": True, "checks": invalid_conditions.get("checks", {})}
        }
        
        logger.info(f"生成止盈信号：{signal['trigger_reason']}，止盈{signal['stop_profit_amount']*100}%")
        
        return signal
    
    def check_invalid_stop_profit_conditions(self, 
                                          stop_profit_result: Dict[str, any],
                                          position_result: Dict[str, float]) -> Dict[str, bool]:
        """检查无效的止盈条件
        
        Args:
            signal_type: 信号类型
            minute_level: 分钟级别
            
        Returns:
            无效条件检查结果
        """
        logger.info("检查无效止盈条件")
        
        checks = {}
        invalid_reasons = []
        
        # 计算当前盈利比例（假设通过价格信息计算）
        # 这里简化处理，实际应该从外部传入或通过数据计算
        current_profit_rate = 0.0
        
        # 检查1：盈利比例是否达到最小阈值
        min_profit_rate = self.config["invalid_signal_conditions"]["min_profit_rate_for_stop_profit"]
        profit_rate_check = current_profit_rate >= min_profit_rate or stop_profit_result.get("divergence_strength_score", 0) > 0.7
        
        checks["profit_rate_check"] = {
            "passed": profit_rate_check,
            "current_value": current_profit_rate,
            "threshold": min_profit_rate
        }
        
        if not profit_rate_check:
            invalid_reasons.append(f"盈利比例不足（当前：{current_profit_rate:.2%}，最小要求：{min_profit_rate:.2%}")
        
        # 检查2：止盈量是否过小
        min_position_ratio = self.config["invalid_signal_conditions"]["min_position_ratio_for_stop_profit"]
        stop_profit_amount = position_result.get("stop_profit_amount", 0)
        min_stop_profit_amount = min_position_ratio
        
        position_amount_check = stop_profit_amount >= min_stop_profit_amount
        
        checks["position_amount_check"] = {
            "passed": position_amount_check,
            "current_value": stop_profit_amount,
            "threshold": min_stop_profit_amount
        }
        
        if not position_amount_check:
            invalid_reasons.append(f"止盈量过小（当前：{stop_profit_amount:.2%}，最小要求：{min_stop_profit_amount:.2%}")
        
        # 检查3：是否有日线顶背驰
        has_divergence = stop_profit_result.get("daily_divergence_detected", False)
        checks["divergence_check"] = {
            "passed": has_divergence
        }
        
        if not has_divergence:
            invalid_reasons.append("无日线顶背驰")
        
        # 检查4：是否有分钟级别确认
        has_confirmation = stop_profit_result.get("minute_confirmation", None) is not None
        checks["confirmation_check"] = {
            "passed": has_confirmation
        }
        
        if not has_confirmation:
            invalid_reasons.append("无分钟级别确认")
        
        # 检查5：强势背驰的特殊处理
        is_strong_divergence = stop_profit_result.get("divergence_strength_score", 0) > 0.7
        if is_strong_divergence:
            # 对于强势背驰，可以适当放宽一些条件
            if f"盈利比例不足（当前：{current_profit_rate:.2%}，最小要求：{min_profit_rate:.2%}）" in invalid_reasons:
                invalid_reasons.remove(f"盈利比例不足（当前：{current_profit_rate:.2%}，最小要求：{min_profit_rate:.2%}")
            checks["profit_rate_check"]["passed"] = True
        
        is_invalid = len(invalid_reasons) > 0
        
        result = {
            "is_invalid": is_invalid,
            "invalid_reason": ", ".join(invalid_reasons) if is_invalid else "",
            "checks": checks
        }
        
        return result
    
    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            data: K线数据
            
        Returns:
            添加了MACD指标的数据
        """
        # 深拷贝数据以避免修改原始数据
        df = data.copy()
        
        # 计算EMA
        fast_ema = df['close'].ewm(span=self.config["macd_fast"], adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.config["macd_slow"], adjust=False).mean()
        
        # 计算MACD线（DIFF）
        df['macd'] = fast_ema - slow_ema
        
        # 计算信号线（DEA）
        df['macd_signal'] = df['macd'].ewm(span=self.config["macd_signal"], adjust=False).mean()
        
        # 计算柱状图（HIST）
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _identify_top_fractal(self, data: pd.DataFrame) -> List[int]:
        """识别顶分型（简化实现）
        
        Args:
            data: K线数据
            
        Returns:
            顶分型索引列表
        """
        top_fractals = []
        
        if len(data) < 3:
            return top_fractals
        
        # 遍历K线，寻找顶分型
        for i in range(1, len(data) - 1):
            # 顶分型定义：中间K线的最高价大于左右相邻K线的最高价
            if ('high' in data.columns and \
                data.iloc[i]['high'] > data.iloc[i-1]['high'] and \
                data.iloc[i]['high'] > data.iloc[i+1]['high']):
                top_fractals.append(i)
        
        return top_fractals
    
    def _find_macd_peaks(self, data: pd.DataFrame, column: str = "macd") -> List[Tuple[int, float]]:
        """查找MACD峰值
        
        Args:
            data: 包含MACD数据的数据框
            column: 要查找峰值的列名
            
        Returns:
            峰值索引和值的列表
        """
        peaks = []
        
        if data is None or data.empty or column not in data.columns:
            return peaks
    
    def _calculate_price_volatility(self, data: pd.DataFrame) -> float:
        """计算价格波动率作为动态校准因子的输入
        
        Args:
            data: K线数据
            
        Returns:
            标准化的波动率值（0-1）
        """
        # 使用最近20根K线计算波动率
        window = min(20, len(data))
        if window < 5:
            return 0.5  # 默认中等波动率
        
        # 计算日收益率
        returns = data['close'].pct_change().dropna()
        
        # 计算波动率（收益率的标准差）
        volatility = returns.iloc[-window:].std()
        
        # 标准化到0-1范围（假设波动率在0-5%之间）
        normalized_volatility = min(max(volatility / 0.05, 0), 1)
        
        return normalized_volatility
    
    def _calculate_signal_strength(self, 
                                 has_daily_top_divergence: bool,
                                 has_confirmed_top_fractal: bool,
                                 is_strong_divergence: bool,
                                 divergence_strength_score: float) -> float:
        """计算信号强度
        
        Args:
            has_daily_top_divergence: 是否有日线顶背驰
            has_confirmed_top_fractal: 是否有确认的顶分型
            is_strong_divergence: 是否是强势背驰
            divergence_strength_score: 背驰强度分数
            
        Returns:
            信号强度分数（0-1）
        """
        strength = 0.0
        
        # 基础分数
        if has_daily_top_divergence:
            strength += 0.4
        if has_confirmed_top_fractal:
            strength += 0.4
        
        # 强势背驰加成
        if is_strong_divergence:
            strength += 0.2
        
        # 背驰强度调整
        strength = min(1.0, strength * (0.8 + 0.4 * divergence_strength_score))
        
        return strength
        
        # 简化实现：查找局部最大值
        for i in range(1, len(data) - 1):
            if data.iloc[i][column] > data.iloc[i-1][column] and \
               data.iloc[i][column] > data.iloc[i+1][column]:
                peaks.append((i, data.iloc[i][column]))
        
        return peaks
    
    def _find_rising_segment(self, data: pd.DataFrame) -> pd.DataFrame:
        """识别上涨段（简化实现）
        
        Args:
            data: K线数据
            
        Returns:
            上涨段数据
        """
        # 简化实现：假设整个数据集都是上涨段
        # 实际应该基于趋势线或高低点分析来识别上涨段
        return data


# 测试代码
def test_profit_taking_rules():
    """测试止盈规则模块"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建规则处理器实例
    rules = ProfitTakingRules()
    
    # 创建模拟数据
    def create_mock_rising_data(levels=30, count=30):
        """创建模拟的上涨K线数据，包含顶背驰特征"""
        index = pd.date_range(start='2024-01-01', periods=count, freq=f'{levels}D')
        
        # 创建先涨后高位震荡的数据，最后出现新高
        # 第一部分：正常上涨
        first_part = np.linspace(100, 120, 15)
        # 第二部分：高位震荡，最后出现新高
        second_part = np.linspace(120, 125, 15)
        
        prices = np.concatenate([first_part, second_part])
        
        data = {
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(3000, 8000, count)
        }
        
        # 最后几根K线制造量价背离特征
        for i in range(count-5, count):
            data['volume'][i] = int(data['volume'][i] * 0.7)  # 成交量减少
        
        return pd.DataFrame(data, index=index)
    
    # 创建分钟级别的数据，包含顶分型
    def create_mock_minute_data_with_top(levels="30min", count=20):
        """创建包含顶分型的分钟级别数据"""
        index = pd.date_range(start='2024-06-01', periods=count, freq=f'{levels}')
        
        # 创建先涨后跌的数据，形成顶分型
        # 前半部分上涨
        first_part = np.linspace(125, 135, count//2)
        # 后半部分下跌
        second_part = np.linspace(135, 130, count - count//2)
        
        prices = np.concatenate([first_part, second_part])
        
        # 制造顶分型（第8-10根K线）
        if count >= 10:
            prices[8] = prices[9]  # 让第9根成为中心K线
            prices[9] = 137  # 中心K线最高价
            prices[10] = prices[9] - 0.5  # 右侧K线
        
        data = {
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 3000, count)
        }
        
        return pd.DataFrame(data, index=index)
    
    # 创建模拟数据
    mock_daily_data = create_mock_rising_data(levels=1, count=40)
    mock_30min_data = create_mock_minute_data_with_top(levels="30min", count=30)
    mock_15min_data = create_mock_minute_data_with_top(levels="15min", count=40)
    
    # 模拟分钟数据字典
    mock_minute_data = {
        "30min": mock_30min_data,
        "15min": mock_15min_data
    }
    
    # 模拟仓位
    mock_positions = {
        "30min_buy": 0.25,  # 30分钟建仓25%
        "15min_buy": 0.15   # 15分钟建仓15%
    }
    
    # 测试日线顶背驰检测
    print("\n===== 测试日线顶背驰检测 =====")
    
    divergence_result = rules.detect_daily_top_divergence(mock_daily_data, "日线二买")
    
    print(f"是否检测到顶背驰: {divergence_result['has_divergence']}")
    print(f"原因: {divergence_result['reason']}")
    print(f"价格创新高: {divergence_result['price_higher']}")
    print(f"MACD未创新高: {divergence_result['macd_lower']}")
    print(f"红柱高度缩小: {divergence_result['red_bar_reduced']}")
    print(f"MACD值: {divergence_result['macd_values']}")
    print(f"红柱高度: {divergence_result['red_bar_heights']}")
    
    # 测试分钟级别顶分型确认
    print("\n===== 测试分钟级别顶分型确认 =====")
    
    # 测试30分钟顶分型
    top_fractal_30min = rules.detect_confirmed_top_fractal(mock_30min_data, "30min")
    
    print(f"30分钟顶分型确认: {top_fractal_30min['has_confirmed_top']}")
    print(f"原因: {top_fractal_30min['reason']}")
    print(f"顶分型索引: {top_fractal_30min['top_fractal_index']}")
    print(f"确认K线索引: {top_fractal_30min['confirmation_bar_index']}")
    print(f"顶分型中心低点: {top_fractal_30min['top_fractal_price']}")
    print(f"确认K线收盘价: {top_fractal_30min['confirmation_price']}")
    
    # 测试止盈条件检查
    print("\n===== 测试止盈条件检查 =====")
    
    # 创建一个模拟的止盈结果（强制设为有效以测试流程）
    mock_stop_profit_result = {
        "should_take_profit": True,
        "daily_divergence_detected": True,
        "minute_confirmation": "30min",
        "stop_profit_ratio": 0.6,
        "affected_positions": mock_positions,
        "divergence_info": divergence_result,
        "confirmation_info": top_fractal_30min,
        "failing_reasons": []
    }
    
    # 计算止盈仓位
    print("\n===== 测试止盈仓位计算 =====")
    
    position_result = rules.calculate_stop_profit_position(
        mock_stop_profit_result,
        mock_positions
    )
    
    print(f"止盈计算有效: {position_result['valid']}")
    print(f"止盈量: {position_result['stop_profit_amount']*100}%")
    print(f"剩余仓位: {position_result['remaining_position']*100}%")
    print(f"应用的止盈比例: {position_result['stop_profit_ratio_applied']*100}%")
    print(f"原始总仓位: {position_result['original_total']*100}%")
    print(f"减少的仓位: {position_result['positions_to_reduce']}")
    
    # 测试生成止盈信号
    print("\n===== 测试生成止盈信号 =====")
    
    stop_profit_signal = rules.generate_stop_profit_signal(
        mock_stop_profit_result,
        position_result
    )
    
    if stop_profit_signal:
        print(f"止盈信号类型: {stop_profit_signal['signal_type']}")
        print(f"触发原因: {stop_profit_signal['trigger_reason']}")
        print(f"止盈量: {stop_profit_signal['stop_profit_amount']*100}%")
        print(f"剩余仓位: {stop_profit_signal['remaining_position']*100}%")
        print(f"止盈比例: {stop_profit_signal['stop_profit_ratio']*100}%")
        print(f"受影响的仓位: {stop_profit_signal['affected_positions']}")
    else:
        print("未生成止盈信号")
    
    # 测试无效止盈条件检查
    print("\n===== 测试无效止盈条件检查 =====")
    
    # 测试固定比例止盈（无效）
    invalid_fixed_ratio = rules.check_invalid_stop_profit_conditions(
        signal_type="固定比例止盈10%",
        minute_level="30min"
    )
    
    print(f"固定比例止盈是否无效: {invalid_fixed_ratio['is_invalid']}")
    print(f"无效原因: {invalid_fixed_ratio['invalid_reasons']}")
    
    # 测试仅5分钟顶分型（无效）
    invalid_5min = rules.check_invalid_stop_profit_conditions(
        signal_type="顶分型确认",
        minute_level="5min"
    )
    
    print(f"仅5分钟顶分型是否无效: {invalid_5min['is_invalid']}")
    print(f"无效原因: {invalid_5min['invalid_reasons']}")
    
    # 测试有效止盈条件
    valid_condition = rules.check_invalid_stop_profit_conditions(
        signal_type="日线顶背驰+30分钟顶分型",
        minute_level="30min"
    )
    
    print(f"有效止盈条件是否无效: {valid_condition['is_invalid']}")
    print(f"无效原因: {valid_condition['invalid_reasons']}")


if __name__ == "__main__":
    test_profit_taking_rules()