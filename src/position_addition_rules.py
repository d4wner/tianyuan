#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向下加仓规则模块

该模块负责实现向下加仓规则，包括加仓前提检查、仓位计算和优先级排序等功能。

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union

# 设置日志
logger = logging.getLogger(__name__)


class PositionAdditionRules:
    """向下加仓规则处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化向下加仓规则处理器
        
        Args:
            config: 配置参数
        """
        logger.info("初始化向下加仓规则模块")
        
        # 默认配置
        self.default_config = {
            # 中枢边界因子
            "central_border_factor": 0.95,      # 中枢边界×0.95
            
            # 周线多头趋势判断
            "weekly_bullish_macd_threshold": 0, # MACD黄白线必须在零轴上方
            "max_continuous_falling_weeks": 2,  # 最多允许连续2根周K线下跌
            
            # 背驰判断参数
            "green_bar_reduction_factor": 0.5,  # 绿柱缩小因子（50%）
            
            # 量能判断参数
            "volume_multiplier": 1.3,           # 量能底分型阈值（1.3倍）
            "volume_window_days": 5,            # 量能计算窗口
            
            # 加仓子仓位比例配置
            "week_div_30min_range": [0.3, 0.4],  # 周线底背驰+30分钟底分（30%-40%）
            "day_div_15min_range": [0.1, 0.2],   # 日线二次底背驰+15分钟底分（10%-20%）
            
            # 加仓优先级（由高到低）
            "addition_priority_order": ["日线二买", "日线一买", "日线三买", "日线破中枢反抽"],
            
            # 总仓位上限
            "max_total_position_ratio": 0.8,    # 80%
            
            # 周线置信度等级对应的加仓比例因子（与原比例相乘）
            "confidence_level_factors": {
                "极强": 1.2,
                "强": 1.1,
                "较强": 1.0,
                "中": 0.9,
                "较弱": 0.8,
                "弱": 0.7,
                "极弱": 0.5
            },
            
            # 背驰强度判断参数
            "strong_divergence_macd_ratio": 0.7,  # 强势背驰：MACD柱缩短至前一段的70%
            "strong_divergence_price_ratio": 0.03,  # 强势背驰：价格新低幅度小于3%
            
            # 分批加仓参数
            "enable_batch_addition": True,        # 启用分批加仓
            "batch_addition_count": 2,           # 分批加仓次数
            "batch_addition_interval_days": 1,   # 分批加仓间隔天数
        }
        
        # 使用用户配置覆盖默认配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        logger.info(f"向下加仓规则配置: {self.config}")
    
    def check_addition_preconditions(self, 
                                   weekly_trend_result: Dict[str, any],
                                   current_price: float,
                                   central_border: float,
                                   has_daily_secondary_divergence: bool,
                                   has_weekly_divergence: bool,
                                   has_minute_bottom_fractal: Dict[str, bool]) -> Dict[str, any]:
        """检查加仓前提条件
        
        加仓前提（需同时满足）：
        1. 趋势兜底：周线多头趋势未破坏
        2. 价格约束：未跌破中枢边界×0.95
        3. 背驰验证：日线二次底背驰或周线底背驰
        4. 形态触发：出现30分钟/15分钟放量底分型
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            current_price: 当前价格
            central_border: 中枢边界价格
            has_daily_secondary_divergence: 是否有日线二次底背驰
            has_weekly_divergence: 是否有周线底背驰
            has_minute_bottom_fractal: 分钟级别底分型情况
            
        Returns:
            前提检查结果字典
        """
        logger.info("检查加仓前提条件")
        
        result = {
            "all_preconditions_met": False,
            "weekly_trend_intact": False,
            "price_within_safe_range": False,
            "divergence_validated": False,
            "pattern_triggered": False,
            "failing_reasons": [],
            "recommended_minute_level": None,
            "divergence_type": None
        }
        
        # 1. 检查周线多头趋势是否未破坏
        weekly_bullish = weekly_trend_result.get("bullish_trend", False)
        macd_above_zero = weekly_trend_result.get("macd_above_zero", False)
        continuous_falling_weeks = weekly_trend_result.get("continuous_falling_weeks", 0)
        
        if weekly_bullish and macd_above_zero and continuous_falling_weeks <= self.config["max_continuous_falling_weeks"]:
            result["weekly_trend_intact"] = True
            logger.info("加仓前提条件1满足：周线多头趋势未破坏")
        else:
            reason = "周线多头趋势已破坏"
            result["failing_reasons"].append(reason)
            logger.warning(f"加仓前提条件1不满足：{reason}")
        
        # 2. 检查价格是否在安全范围内
        safe_threshold = central_border * self.config["central_border_factor"]
        if current_price >= safe_threshold:
            result["price_within_safe_range"] = True
            logger.info(f"加仓前提条件2满足：当前价格{current_price} >= 安全阈值{safe_threshold}")
        else:
            reason = f"价格跌破中枢边界×{self.config['central_border_factor']}"
            result["failing_reasons"].append(reason)
            logger.warning(f"加仓前提条件2不满足：{reason}")
        
        # 3. 检查背驰验证
        if has_daily_secondary_divergence or has_weekly_divergence:
            result["divergence_validated"] = True
            # 确定背驰类型
            if has_weekly_divergence:
                result["divergence_type"] = "周线底背驰"
            else:
                result["divergence_type"] = "日线二次底背驰"
            logger.info(f"加仓前提条件3满足：{result['divergence_type']}")
        else:
            reason = "无有效背驰验证"
            result["failing_reasons"].append(reason)
            logger.warning(f"加仓前提条件3不满足：{reason}")
        
        # 4. 检查形态触发
        if has_minute_bottom_fractal.get("30min", False):
            result["pattern_triggered"] = True
            result["recommended_minute_level"] = "30min"
            logger.info("加仓前提条件4满足：30分钟放量底分型")
        elif has_minute_bottom_fractal.get("15min", False):
            result["pattern_triggered"] = True
            result["recommended_minute_level"] = "15min"
            logger.info("加仓前提条件4满足：15分钟放量底分型")
        else:
            reason = "无分钟级别放量底分型"
            result["failing_reasons"].append(reason)
            logger.warning(f"加仓前提条件4不满足：{reason}")
        
        # 综合判断所有前提是否满足
        result["all_preconditions_met"] = (result["weekly_trend_intact"] and 
                                          result["price_within_safe_range"] and 
                                          result["divergence_validated"] and 
                                          result["pattern_triggered"])
        
        # 添加周线置信度信息
        result["weekly_confidence_level"] = weekly_trend_result.get("confidence_level", "中")
        result["weekly_confidence_score"] = weekly_trend_result.get("confidence_score", 0.5)
        
        if result["all_preconditions_met"]:
            logger.info(f"所有加仓前提条件均已满足，周线置信度等级：{result['weekly_confidence_level']}")
        else:
            logger.warning(f"加仓前提条件不满足，失败原因：{', '.join(result['failing_reasons'])}")
        
        return result
    
    def calculate_addition_position(self, 
                                  daily_signal_type: str,
                                  precondition_result: Dict[str, any],
                                  original_daily_position: float,
                                  current_total_position: float,
                                  daily_position_ratio_config: Dict[str, float]) -> Dict[str, float]:
        """计算加仓仓位
        
        Args:
            daily_signal_type: 日线买点类型（日线二买/一买/三买/反抽）
            precondition_result: 前提条件检查结果
            original_daily_position: 该日线买点的原有总仓位比例
            current_total_position: 当前累计总仓位比例
            daily_position_ratio_config: 用户配置的日线仓位比例
            
        Returns:
            加仓仓位计算结果字典
        """
        logger.info(f"计算加仓仓位：日线买点类型={daily_signal_type}, 原有仓位={original_daily_position*100}%, 当前总仓位={current_total_position*100}%")
        
        result = {
            "addition_position_ratio": 0.0,
            "post_addition_total": 0.0,
            "addition_sub_ratio": 0.0,
            "max_addition_available": 0.0,
            "allocation_method": "",
            "constraint_reason": None,
            "valid": False
        }
        
        # 检查前提条件是否满足
        if not precondition_result.get("all_preconditions_met", False):
            result["constraint_reason"] = "加仓前提条件未满足"
            logger.warning("加仓前提条件未满足，无法加仓")
            return result
        
        # 获取背驰类型、分钟级别和周线置信度信息
        divergence_type = precondition_result.get("divergence_type")
        minute_level = precondition_result.get("recommended_minute_level")
        confidence_level = precondition_result.get("weekly_confidence_level", "中")
        
        # 获取置信度对应的加仓比例因子
        confidence_factor = self.config["confidence_level_factors"].get(
            confidence_level, 1.0
        )
        logger.info(f"使用周线置信度等级：{confidence_level}，对应的比例因子：{confidence_factor}")
        
        # 根据背驰类型和分钟级别确定基础子仓位比例
        base_sub_ratio = 0.0
        allocation_method = ""
        
        if divergence_type == "周线底背驰" and minute_level == "30min":
            # 周线底背驰+30分钟底分（30%-40%）
            sub_range = self.config["week_div_30min_range"]
            base_sub_ratio = (sub_range[0] + sub_range[1]) / 2  # 取中间值
            allocation_method = f"周线底背驰+{minute_level}底分"
            logger.info(f"使用基础子仓位比例范围：{sub_range[0]*100}%-{sub_range[1]*100}%")
        elif divergence_type == "日线二次底背驰" and minute_level == "15min":
            # 日线二次底背驰+15分钟底分（10%-20%）
            sub_range = self.config["day_div_15min_range"]
            base_sub_ratio = (sub_range[0] + sub_range[1]) / 2  # 取中间值
            allocation_method = f"日线二次底背驰+{minute_level}底分"
            logger.info(f"使用基础子仓位比例范围：{sub_range[0]*100}%-{sub_range[1]*100}%")
        else:
            # 其他组合的默认处理
            # 周线底背驰+15分钟底分，使用较小的比例
            if divergence_type == "周线底背驰":
                sub_range = self.config["week_div_30min_range"]
                base_sub_ratio = sub_range[0]  # 取下限
                allocation_method = f"周线底背驰+{minute_level}底分（默认下限）"
            # 日线二次底背驰+30分钟底分，使用较大的比例
            elif divergence_type == "日线二次底背驰":
                sub_range = self.config["day_div_15min_range"]
                base_sub_ratio = sub_range[1]  # 取上限
                allocation_method = f"日线二次底背驰+{minute_level}底分（默认上限）"
            logger.info(f"使用默认基础子仓位比例：{base_sub_ratio*100}%")
        
        # 应用置信度因子计算最终子仓位比例
        sub_ratio = base_sub_ratio * confidence_factor
        # 限制子仓位比例在合理范围内（0.05-0.5）
        sub_ratio = max(min(sub_ratio, 0.5), 0.05)
        logger.info(f"应用置信度因子后的子仓位比例：{sub_ratio*100}%")
        
        # 计算理论上加仓量
        theoretical_addition = original_daily_position * sub_ratio
        
        # 计算剩余可加仓空间（不超过原有总仓位）
        remaining_from_original = original_daily_position - (original_daily_position * 0.5)  # 假设已用了50%的原始仓位
        max_addition_available = min(theoretical_addition, remaining_from_original)
        
        # 检查总仓位上限约束
        max_total_position = self.config["max_total_position_ratio"]
        available_for_total = max_total_position - current_total_position
        
        if available_for_total <= 0:
            result["constraint_reason"] = f"已达到总仓位上限{max_total_position*100}%"
            logger.warning(f"已达到总仓位上限{max_total_position*100}%，无法加仓")
            return result
        
        # 最终加仓量受限于原始仓位剩余空间和总仓位上限
        final_addition = min(max_addition_available, available_for_total)
        
        # 检查是否启用分批加仓
        batch_addition_plan = None
        if self.config["enable_batch_addition"] and final_addition > 0:
            batch_count = self.config["batch_addition_count"]
            batch_interval = self.config["batch_addition_interval_days"]
            
            # 计算每批加仓量（可以稍微不均分，第一批多一点）
            first_batch_ratio = 0.6  # 第一批占总加仓量的60%
            first_batch = final_addition * first_batch_ratio
            remaining_addition = final_addition - first_batch
            
            # 其余批次平均分配
            other_batches = remaining_addition / (batch_count - 1) if batch_count > 1 else 0
            
            batch_addition_plan = {
                "total_batches": batch_count,
                "batches": []
            }
            
            # 添加第一批
            batch_addition_plan["batches"].append({
                "batch_index": 1,
                "addition_amount": first_batch,
                "execution_day": 0  # 当天执行
            })
            
            # 添加后续批次
            for i in range(1, batch_count):
                batch_addition_plan["batches"].append({
                    "batch_index": i + 1,
                    "addition_amount": other_batches,
                    "execution_day": i * batch_interval  # 间隔执行
                })
            
            logger.info(f"启用分批加仓，共{batch_count}批次，间隔{batch_interval}天")
        
        # 更新结果
        result["addition_position_ratio"] = final_addition
        result["post_addition_total"] = current_total_position + final_addition
        result["addition_sub_ratio"] = sub_ratio
        result["max_addition_available"] = max_addition_available
        result["allocation_method"] = allocation_method
        result["valid"] = True
        result["weekly_confidence_factor"] = confidence_factor
        result["batch_addition_plan"] = batch_addition_plan
        
        logger.info(f"加仓计算结果：加仓比例={final_addition*100}%, 加仓后总仓位={result['post_addition_total']*100}%")
        
        # 如果加仓量为0或接近0，标记为无效
        if final_addition <= 0.001:  # 小于0.1%视为无效
            result["valid"] = False
            result["constraint_reason"] = "计算出的加仓量过小"
            logger.warning("计算出的加仓量过小，视为无效")
        
        return result
    
    def determine_addition_priority(self, 
                                  available_additions: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """确定加仓优先级
        
        加仓优先级：日线二买加仓优先级最高，其次是一买，最后是三买
        
        Args:
            available_additions: 可加仓的信号列表，每项包含日线买点类型和加仓计算结果
            
        Returns:
            按优先级排序的加仓列表
        """
        logger.info("确定加仓优先级")
        
        # 获取优先级顺序
        priority_order = self.config["addition_priority_order"]
        
        # 为每个可用加仓分配优先级分数
        for addition in available_additions:
            signal_type = addition.get("daily_signal_type", "")
            if signal_type in priority_order:
                addition["priority_score"] = priority_order.index(signal_type)
            else:
                addition["priority_score"] = len(priority_order)  # 默认最低优先级
        
        # 按优先级分数排序（分数越低优先级越高）
        sorted_additions = sorted(available_additions, key=lambda x: x.get("priority_score", 999))
        
        logger.info(f"加仓优先级排序结果：{[item.get('daily_signal_type') for item in sorted_additions]}")
        
        return sorted_additions
    
    def generate_addition_signal(self, 
                               sorted_additions: List[Dict[str, any]],
                               weekly_trend_result: Dict[str, any]) -> Optional[Dict[str, any]]:
        """生成最终的加仓信号
        
        Args:
            sorted_additions: 按优先级排序的加仓列表
            weekly_trend_result: 周线趋势检测结果
            
        Returns:
            最终的加仓交易信号，如果没有有效加仓则返回None
        """
        logger.info("生成加仓交易信号")
        
        # 遍历排序后的加仓列表，找到第一个有效的加仓
        for addition in sorted_additions:
            if addition.get("valid", False):
                # 获取相关信息
                daily_signal_type = addition.get("daily_signal_type", "")
                precondition_result = addition.get("precondition_result", {})
                position_result = addition.get("position_result", {})
                
                # 构造加仓信号
                signal = {
                    "signal_type": f"周线多头+{daily_signal_type}-{precondition_result.get('recommended_minute_level', '').replace('min', '')}分钟加仓",
                    "addition_position": position_result.get("addition_position_ratio", 0),
                    "post_addition_total": position_result.get("post_addition_total", 0),
                    "addition_sub_ratio": position_result.get("addition_sub_ratio", 0),
                    "allocation_method": position_result.get("allocation_method", ""),
                    "divergence_type": precondition_result.get("divergence_type", ""),
                    "minute_level": precondition_result.get("recommended_minute_level", ""),
                    "daily_signal_type": daily_signal_type,
                    "weekly_confidence_level": precondition_result.get("weekly_confidence_level", "中"),
                    "weekly_confidence_factor": position_result.get("weekly_confidence_factor", 1.0),
                    "batch_addition_plan": position_result.get("batch_addition_plan", None)
                }
                
                logger.info(f"生成加仓信号：{signal['signal_type']}")
                return signal
        
        logger.info("没有找到有效的加仓信号")
        return None
    
    def detect_daily_secondary_divergence(self, 
                                        daily_data: pd.DataFrame,
                                        original_buy_signal_index: int) -> bool:
        """检测日线二次底背驰
        
        日线二次底背驰规则同日线一买背驰
        
        Args:
            daily_data: 日线数据
            original_buy_signal_index: 原始买点的索引位置
            
        Returns:
            是否检测到日线二次底背驰
        """
        logger.info("检测日线二次底背驰")
        
        if daily_data is None or daily_data.empty:
            logger.warning("日线数据为空，无法检测二次底背驰")
            return False
        
        # 确保索引在有效范围内
        if original_buy_signal_index >= len(daily_data):
            logger.warning("原始买点索引超出数据范围")
            return False
        
        # 提取买点后的下跌段数据
        # 简化实现：查找买点后的新低
        post_buy_data = daily_data.iloc[original_buy_signal_index:]
        
        if post_buy_data.empty:
            logger.warning("买点后数据不足")
            return False
        
        # 查找买点后的新低
        buy_price = daily_data.iloc[original_buy_signal_index].get("low", 0)
        new_low_mask = post_buy_data["low"] < buy_price
        
        if not new_low_mask.any():
            logger.info("买点后未创新低，无二次底背驰")
            return False
        
        # 简化判断：实际需要计算MACD并判断背驰
        # 这里仅作为示例，返回True表示有二次底背驰
        logger.info("检测到日线二次底背驰")
        return True
    
    def detect_weekly_divergence(self, weekly_data: pd.DataFrame) -> bool:
        """检测周线底背驰
        
        周线底背驰：周线下跌段价格新低+黄白线不新低+绿柱缩窄
        
        Args:
            weekly_data: 周线数据
            
        Returns:
            是否检测到周线底背驰
        """
        logger.info("检测周线底背驰")
        
        if weekly_data is None or weekly_data.empty:
            logger.warning("周线数据为空，无法检测底背驰")
            return False
        
        # 简化实现：查找近期新低
        # 实际需要：
        # 1. 识别下跌段
        # 2. 计算MACD
        # 3. 判断价格新低但MACD不新低
        # 4. 判断绿柱缩窄
        
        # 检查是否有新低
        recent_low = weekly_data["low"].tail(10).min()
        is_recent_low = weekly_data["low"].iloc[-1] == recent_low
        
        if is_recent_low:
            logger.info("检测到周线底背驰")
            return True
        else:
            logger.info("未检测到周线底背驰")
            return False
    
    def detect_minute_bottom_fractal(self, 
                                  minute_data: Dict[str, pd.DataFrame],
                                  daily_data: pd.DataFrame) -> Dict[str, bool]:
        """检测分钟级别放量底分型
        
        Args:
            minute_data: 分钟级别数据字典，包含'30min'和'15min'数据
            daily_data: 日线数据（用于计算量能基准）
            
        Returns:
            分钟级别底分型检测结果字典
        """
        logger.info("检测分钟级别放量底分型")
        
        result = {
            "30min": False,
            "15min": False
        }
        
        # 计算量能基准
        avg_volume = None
        if daily_data is not None and not daily_data.empty and 'volume' in daily_data.columns:
            window_size = self.config["volume_window_days"]
            if len(daily_data) >= window_size:
                avg_volume = daily_data['volume'].tail(window_size).mean()
                logger.info(f"计算{window_size}日平均成交量：{avg_volume}")
        
        # 对每个分钟级别进行检测
        for level in ["30min", "15min"]:
            if level not in minute_data or minute_data[level] is None or minute_data[level].empty:
                logger.warning(f"{level}数据为空，跳过检测")
                continue
            
            data = minute_data[level]
            
            # 识别底分型（简化实现）
            bottom_fractals = self._identify_bottom_fractal(data)
            
            if not bottom_fractals:
                logger.info(f"{level}未发现底分型")
                continue
            
            # 获取最近的底分型
            last_bottom_idx = max(bottom_fractals)
            last_bottom = data.iloc[last_bottom_idx]
            
            # 检查量能
            has_volume_confirmation = False
            if 'volume' in last_bottom and avg_volume is not None:
                current_volume = last_bottom['volume']
                has_volume_confirmation = current_volume >= avg_volume * self.config["volume_multiplier"]
                logger.info(f"{level}底分型量能检查：当前成交量{current_volume} >= {avg_volume}*{self.config['volume_multiplier']} = {avg_volume*self.config['volume_multiplier']}？ {has_volume_confirmation}")
            
            result[level] = has_volume_confirmation
        
        return result
    
    def _identify_bottom_fractal(self, data: pd.DataFrame) -> List[int]:
        """识别底分型（简化实现）
        
        Args:
            data: K线数据
            
        Returns:
            底分型索引列表
        """
        bottom_fractals = []
        
        if len(data) < 3:
            return bottom_fractals
        
        # 遍历K线，寻找底分型
        for i in range(1, len(data) - 1):
            # 底分型定义：中间K线的最低价小于左右相邻K线的最低价
            if ('low' in data.columns and \
                data.iloc[i]['low'] < data.iloc[i-1]['low'] and \
                data.iloc[i]['low'] < data.iloc[i+1]['low']):
                bottom_fractals.append(i)
        
        return bottom_fractals
    
    def calculate_total_position_impact(self, 
                                      current_positions: Dict[str, float],
                                      new_addition: float) -> Dict[str, any]:
        """计算新增加仓对总仓位的影响
        
        Args:
            current_positions: 当前各类型仓位字典
            new_addition: 新增加仓量
            
        Returns:
            总仓位影响分析结果
        """
        # 计算当前总仓位
        current_total = sum(current_positions.values())
        
        # 计算加仓后总仓位
        new_total = current_total + new_addition
        
        # 获取总仓位上限
        max_total = self.config["max_total_position_ratio"]
        
        # 计算剩余可用空间
        remaining_space = max_total - new_total
        
        # 判断是否超过上限
        exceeds_limit = new_total > max_total
        
        return {
            "current_total_position": current_total,
            "new_total_position": new_total,
            "max_total_position": max_total,
            "remaining_space": remaining_space,
            "exceeds_limit": exceeds_limit,
            "can_add_more": remaining_space > 0.01  # 剩余空间大于1%视为可以继续加仓
        }


# 测试代码
def test_position_addition_rules():
    """测试向下加仓规则模块"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建规则处理器实例
    rules = PositionAdditionRules()
    
    # 创建模拟数据
    def create_mock_data(levels=30, count=20):
        """创建模拟的K线数据"""
        index = pd.date_range(start='2024-01-01', periods=count, freq=f'{levels}D')
        
        # 创建先涨后跌的数据
        prices = np.linspace(100, 110, 10)  # 上涨部分
        prices = np.concatenate([prices, np.linspace(110, 102, 10)])  # 回调部分
        
        data = {
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 5000, count)
        }
        
        return pd.DataFrame(data, index=index)
    
    # 创建模拟数据
    mock_daily_data = create_mock_data(levels=1, count=30)
    mock_weekly_data = create_mock_data(levels=7, count=20)
    
    # 创建分钟级别数据
    mock_minute_data = {
        "30min": create_mock_data(levels=0.5, count=50),  # 0.5小时=30分钟
        "15min": create_mock_data(levels=0.25, count=100)  # 0.25小时=15分钟
    }
    
    # 测试前提条件检查
    print("\n===== 测试加仓前提条件检查 =====")
    
    # 有效情况
    weekly_trend_result_valid = {
        "bullish_trend": True,
        "macd_above_zero": True,
        "continuous_falling_weeks": 1
    }
    
    # 无效情况（周线多头破坏）
    weekly_trend_result_invalid = {
        "bullish_trend": True,
        "macd_above_zero": False,
        "continuous_falling_weeks": 3
    }
    
    # 有效前提检查
    print("\n有效前提条件检查：")
    precondition_result = rules.check_addition_preconditions(
        weekly_trend_result=weekly_trend_result_valid,
        current_price=9.7,  # 假设中枢边界是10.0，安全阈值是9.5
        central_border=10.0,
        has_daily_secondary_divergence=False,
        has_weekly_divergence=True,
        has_minute_bottom_fractal={"30min": True, "15min": False}
    )
    
    print(f"所有前提满足: {precondition_result['all_preconditions_met']}")
    print(f"周线趋势完好: {precondition_result['weekly_trend_intact']}")
    print(f"价格在安全范围: {precondition_result['price_within_safe_range']}")
    print(f"背驰验证通过: {precondition_result['divergence_validated']}")
    print(f"形态触发: {precondition_result['pattern_triggered']}")
    print(f"失败原因: {precondition_result['failing_reasons']}")
    print(f"推荐分钟级别: {precondition_result['recommended_minute_level']}")
    print(f"背驰类型: {precondition_result['divergence_type']}")
    
    # 无效前提检查（价格跌破阈值）
    print("\n无效前提条件检查（价格跌破阈值）：")
    precondition_result_invalid_price = rules.check_addition_preconditions(
        weekly_trend_result=weekly_trend_result_valid,
        current_price=9.0,  # 低于安全阈值9.5
        central_border=10.0,
        has_daily_secondary_divergence=True,
        has_weekly_divergence=False,
        has_minute_bottom_fractal={"30min": False, "15min": True}
    )
    
    print(f"所有前提满足: {precondition_result_invalid_price['all_preconditions_met']}")
    print(f"失败原因: {precondition_result_invalid_price['failing_reasons']}")
    
    # 测试仓位计算
    print("\n===== 测试加仓仓位计算 =====")
    
    # 模拟用户配置的日线仓位
    daily_position_ratio_config = {
        "日线二买": 0.3,  # 30%
        "日线一买": 0.2,  # 20%
        "日线三买": 0.25  # 25%
    }
    
    # 有效仓位计算
    print("\n有效仓位计算：")
    position_result = rules.calculate_addition_position(
        daily_signal_type="日线二买",
        precondition_result=precondition_result,
        original_daily_position=daily_position_ratio_config["日线二买"],  # 30%
        current_total_position=0.4,  # 40%
        daily_position_ratio_config=daily_position_ratio_config
    )
    
    print(f"加仓有效: {position_result['valid']}")
    print(f"加仓比例: {position_result['addition_position_ratio']*100}%")
    print(f"加仓后总仓位: {position_result['post_addition_total']*100}%")
    print(f"子仓位比例: {position_result['addition_sub_ratio']*100}%")
    print(f"分配方式: {position_result['allocation_method']}")
    
    # 测试加仓优先级
    print("\n===== 测试加仓优先级 =====")
    
    # 创建多个可用加仓
    available_additions = [
        {
            "daily_signal_type": "日线一买",
            "precondition_result": precondition_result,
            "position_result": {
                "valid": True,
                "addition_position_ratio": 0.04,
                "post_addition_total": 0.44
            }
        },
        {
            "daily_signal_type": "日线二买",
            "precondition_result": precondition_result,
            "position_result": {
                "valid": True,
                "addition_position_ratio": 0.10,
                "post_addition_total": 0.50
            }
        },
        {
            "daily_signal_type": "日线三买",
            "precondition_result": precondition_result,
            "position_result": {
                "valid": True,
                "addition_position_ratio": 0.05,
                "post_addition_total": 0.45
            }
        }
    ]
    
    # 确定优先级
    sorted_additions = rules.determine_addition_priority(available_additions)
    
    print("按优先级排序的加仓列表：")
    for i, addition in enumerate(sorted_additions):
        print(f"{i+1}. {addition['daily_signal_type']} (优先级分数: {addition['priority_score']})")
    
    # 测试生成加仓信号
    print("\n===== 测试生成加仓信号 =====")
    
    signal = rules.generate_addition_signal(sorted_additions, weekly_trend_result_valid)
    
    if signal:
        print(f"加仓信号类型: {signal['signal_type']}")
        print(f"加仓比例: {signal['addition_position']*100}%")
        print(f"加仓后总仓位: {signal['post_addition_total']*100}%")
        print(f"子仓位比例: {signal['addition_sub_ratio']*100}%")
        print(f"背驰类型: {signal['divergence_type']}")
        print(f"分钟级别: {signal['minute_level']}")
    else:
        print("未生成加仓信号")
    
    # 测试底背驰检测
    print("\n===== 测试底背驰检测 =====")
    
    # 检测日线二次底背驰
    has_daily_div = rules.detect_daily_secondary_divergence(mock_daily_data, 10)
    print(f"日线二次底背驰: {has_daily_div}")
    
    # 检测周线底背驰
    has_weekly_div = rules.detect_weekly_divergence(mock_weekly_data)
    print(f"周线底背驰: {has_weekly_div}")
    
    # 测试分钟级别底分型检测
    print("\n===== 测试分钟级别底分型检测 =====")
    
    fractal_result = rules.detect_minute_bottom_fractal(mock_minute_data, mock_daily_data)
    print(f"30分钟底分型: {fractal_result['30min']}")
    print(f"15分钟底分型: {fractal_result['15min']}")


if __name__ == "__main__":
    test_position_addition_rules()