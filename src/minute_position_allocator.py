#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分钟级别子分仓系统模块

该模块负责实现分钟级别子仓位分配逻辑，包括30分钟/15分钟向上笔判定、回撤买点识别
以及根据日线买点类型分配子仓位比例，并与周线置信度等级关联。

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union

# 设置日志
logger = logging.getLogger(__name__)


class MinutePositionAllocator:
    """分钟级别子分仓系统"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化分钟级别子分仓系统
        
        Args:
            config: 配置参数
        """
        logger.info("初始化分钟级别子分仓系统")
        
        # 默认配置
        self.default_config = {
            # 30分钟向上笔配置
            "30min_min_k_count": 5,            # 最小K线数量
            "30min_top_fractal_confirm": 2,    # 顶分型确认K线数量
            "30min_primary_range": [0.6, 0.7], # 主优先级子仓位比例范围
            "30min_secondary_range": [0.4, 0.5], # 次优先级子仓位比例范围
            
            # 15分钟向上笔配置
            "15min_min_k_count": 5,            # 最小K线数量
            "15min_top_fractal_confirm": 2,    # 顶分型确认K线数量
            "15min_primary_range": [0.3, 0.4], # 主优先级子仓位比例范围
            "15min_secondary_range": [0.2, 0.3], # 次优先级子仓位比例范围
            
            # 小级别回撤买点配置
            "5min_min_k_count": 5,             # 5分钟最小K线数量
            "volume_multiplier_primary": 1.3,  # 主要量能阈值
            "volume_multiplier_secondary": 1.2, # 二买专属量能阈值（可放宽）
            "volume_window_days": 5,           # 量能计算窗口
            
            # 优先级配置
            "priority_order": ["日线二买", "日线三买", "日线一买"],  # 日线买点优先级
            "secondary_priority_order": ["日线一买", "日线三买", "日线二买"],  # 次要匹配顺序
            
            # 周线置信度相关配置
            "confidence_level_multipliers": {
                "HIGH": 1.1,         # 高置信度乘法因子
                "MEDIUM_HIGH": 1.0,  # 中高置信度乘法因子
                "MEDIUM": 0.95,      # 中置信度乘法因子
                "MEDIUM_LOW": 0.85,  # 中低置信度乘法因子
                "LOW": 0.75          # 低置信度乘法因子
            },
            
            # 回撤幅度阈值配置（用于确定安全买点）
            "safe_retracement_ranges": {
                "30min": [0.382, 0.618],  # 30分钟向上笔的安全回撤区间（黄金分割）
                "15min": [0.382, 0.618]   # 15分钟向上笔的安全回撤区间
            },
            
            # 建仓时间窗口配置
            "position_building_windows": {
                "30min": 5,  # 30分钟向上笔后的建仓时间窗口（小时数）
                "15min": 3   # 15分钟向上笔后的建仓时间窗口（小时数）
            }
        }
        
        # 使用用户配置覆盖默认配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        logger.info(f"分钟级别子分仓系统配置: {self.config}")
    
    def detect_minute_up_pen(self, 
                           minute_data: pd.DataFrame,
                           minute_level: str) -> Dict[str, any]:
        """检测分钟级别向上笔是否走完
        
        Args:
            minute_data: 分钟级别数据
            minute_level: 分钟级别，如'30min'或'15min'
            
        Returns:
            向上笔检测结果字典
        """
        logger.info(f"检测{minute_level}向上笔")
        
        result = {
            "up_pen_completed": False,
            "has_top_fractal": False,
            "confirmed": False,
            "signal": "",
            "last_top_fractal": None,
            "validation_reason": ""
        }
        
        # 根据分钟级别选择配置
        config_key_map = {
            "30min": {
                "min_k_count": "30min_min_k_count",
                "top_fractal_confirm": "30min_top_fractal_confirm"
            },
            "15min": {
                "min_k_count": "15min_min_k_count",
                "top_fractal_confirm": "15min_top_fractal_confirm"
            }
        }
        
        if minute_level not in config_key_map:
            logger.error(f"不支持的分钟级别: {minute_level}")
            result["validation_reason"] = f"不支持的分钟级别: {minute_level}"
            return result
        
        # 获取对应配置
        level_config = config_key_map[minute_level]
        min_k_count = self.config[level_config["min_k_count"]]
        confirm_k_count = self.config[level_config["top_fractal_confirm"]]
        
        # 验证数据
        if minute_data is None or minute_data.empty:
            logger.warning(f"{minute_level}数据为空")
            result["validation_reason"] = f"{minute_level}数据为空"
            return result
        
        # 去除包含关系（简化处理）
        cleaned_data = self._remove_inclusion(minute_data)
        
        # 检查K线数量是否满足要求
        if len(cleaned_data) < min_k_count:
            logger.info(f"{minute_level}K线数量不足，需要{min_k_count}根，当前{len(cleaned_data)}根")
            result["validation_reason"] = f"{minute_level}K线数量不足"
            return result
        
        # 识别顶分型
        top_fractal_indices = self._identify_top_fractal(cleaned_data)
        
        if not top_fractal_indices:
            logger.info(f"{minute_level}未发现顶分型")
            result["validation_reason"] = f"{minute_level}未发现顶分型"
            return result
        
        # 获取最近的顶分型
        last_top_idx = max(top_fractal_indices)
        last_top = cleaned_data.iloc[last_top_idx]
        result["has_top_fractal"] = True
        result["last_top_fractal"] = {
            "index": last_top_idx,
            "high": last_top.get("high", 0),
            "low": last_top.get("low", 0),
            "close": last_top.get("close", 0),
            "time": str(last_top.name) if isinstance(last_top.name, pd.Timestamp) else str(last_top.name)
        }
        
        # 检查确认K线
        # 顶分型后需要至少confirm_k_count根K线，且第confirm_k_count根K线收盘价低于顶分型中心K线低点
        center_k_low = last_top.get("low", 0)
        
        # 检查顶分型后是否有足够的确认K线
        if last_top_idx + confirm_k_count >= len(cleaned_data):
            logger.info(f"{minute_level}顶分型后确认K线不足")
            result["validation_reason"] = f"{minute_level}顶分型后确认K线不足"
            return result
        
        # 检查确认K线是否跌破顶分型中心K线低点
        confirm_k = cleaned_data.iloc[last_top_idx + confirm_k_count]
        confirm_close = confirm_k.get("close", 0)
        
        if confirm_close < center_k_low:
            logger.info(f"{minute_level}向上笔确认完成：顶分型确认K线收盘价{confirm_close}低于中心K线低点{center_k_low}")
            result["up_pen_completed"] = True
            result["confirmed"] = True
            result["signal"] = f"{minute_level}向上笔走完，可触发回撤买点"
        else:
            logger.info(f"{minute_level}向上笔未确认完成：顶分型确认K线收盘价{confirm_close}未低于中心K线低点{center_k_low}")
            result["validation_reason"] = f"{minute_level}向上笔未确认完成：确认K线未跌破顶分型中心低点"
        
        return result
    
    def detect_retracement_buy_points(self, 
                                   minute_data: pd.DataFrame,
                                   up_pen_level: str,
                                   daily_signal_type: str,
                                   daily_data: Optional[pd.DataFrame] = None,
                                   up_pen_result: Optional[Dict] = None,
                                   weekly_trend_result: Optional[Dict] = None) -> List[Dict]:
        """检测小级别回撤建仓买点（增强版）
        
        Args:
            minute_data: 分钟级别数据
            up_pen_level: 向上笔级别（'30min'或'15min'）
            daily_signal_type: 日线买点类型
            daily_data: 日线数据（用于量能计算）
            up_pen_result: 向上笔检测结果（包含向上笔的详细信息）
            weekly_trend_result: 周线趋势检测结果（用于获取置信度）
            
        Returns:
            回撤买点列表，按优先级排序
        """
        logger.info(f"检测{up_pen_level}向上笔后的回撤买点，日线信号类型：{daily_signal_type}")
        
        buy_points = []
        
        # 根据向上笔级别确定可接受的回撤买点级别
        retracement_levels = []
        if up_pen_level == "30min":
            retracement_levels = ["15min", "5min"]
        elif up_pen_level == "15min":
            retracement_levels = ["15min", "5min"]
        
        # 根据日线买点类型和周线置信度确定量能阈值
        volume_multiplier = self.config["volume_multiplier_primary"]
        if daily_signal_type == "日线二买":
            volume_multiplier = self.config["volume_multiplier_secondary"]  # 二买专属量能阈值可放宽
            logger.info(f"二买专属处理：量能阈值放宽至{volume_multiplier}倍")
        
        # 根据周线置信度调整量能阈值
        confidence_level = weekly_trend_result.get("confidence_level", "MEDIUM") if weekly_trend_result else "MEDIUM"
        if confidence_level in ["LOW", "MEDIUM_LOW"]:
            volume_multiplier *= 0.9  # 低置信度情况下可适当降低量能要求
            logger.info(f"低置信度情况调整：量能阈值降低至{volume_multiplier}倍")
        
        # 计算量能基准（如果有日线数据）
        avg_volume = None
        if daily_data is not None and not daily_data.empty and 'volume' in daily_data.columns:
            window_size = self.config["volume_window_days"]
            if len(daily_data) >= window_size:
                avg_volume = daily_data['volume'].tail(window_size).mean()
                logger.info(f"计算{window_size}日平均成交量：{avg_volume}")
        
        # 计算向上笔的高低点（用于判断回撤幅度）
        up_pen_high = None
        up_pen_low = None
        if up_pen_result and up_pen_result.get("last_top_fractal"):
            last_top = up_pen_result["last_top_fractal"]
            up_pen_high = last_top.get("high", 0)
            # 向上笔的低点需要在原始数据中查找（简化处理，实际应该从向上笔的起始点计算）
            if minute_data is not None and not minute_data.empty:
                up_pen_low = minute_data['low'].min()
            logger.info(f"向上笔高低点：高={up_pen_high}, 低={up_pen_low}")
        
        # 对每个可能的回撤级别进行检测
        for level in retracement_levels:
            # 重采样到目标级别
            level_data = self._resample_to_minute_level(minute_data, level)
            
            # 去除包含关系
            cleaned_data = self._remove_inclusion(level_data)
            
            # 识别底分型
            bottom_fractal_indices = self._identify_bottom_fractal(cleaned_data)
            
            if not bottom_fractal_indices:
                logger.info(f"{level}未发现底分型")
                continue
            
            # 只考虑最近的几个底分型（最近3个）
            recent_bottom_indices = sorted(bottom_fractal_indices, reverse=True)[:3]
            
            for idx in recent_bottom_indices:
                if idx >= len(cleaned_data):
                    continue
                    
                bottom = cleaned_data.iloc[idx]
                bottom_price = bottom.get("low", 0)
                
                # 计算回撤幅度
                retracement_ratio = None
                if up_pen_high and up_pen_low and up_pen_high > up_pen_low:
                    retracement_ratio = (up_pen_high - bottom_price) / (up_pen_high - up_pen_low)
                    logger.info(f"回撤幅度计算：({up_pen_high} - {bottom_price}) / ({up_pen_high} - {up_pen_low}) = {retracement_ratio:.3f}")
                
                # 检查回撤是否在安全区间
                is_in_safe_range = False
                if retracement_ratio is not None:
                    safe_range = self.config["safe_retracement_ranges"].get(up_pen_level, [0.3, 0.7])
                    is_in_safe_range = safe_range[0] <= retracement_ratio <= safe_range[1]
                    logger.info(f"回撤是否在安全区间{safe_range}：{is_in_safe_range}")
                
                # 检查量能
                has_volume_confirmation = False
                if 'volume' in bottom:
                    current_volume = bottom['volume']
                    if avg_volume is not None:
                        has_volume_confirmation = current_volume >= avg_volume * volume_multiplier
                        logger.info(f"{level}底分型量能检查：当前成交量{current_volume} >= {avg_volume}*{volume_multiplier} = {avg_volume*volume_multiplier}？ {has_volume_confirmation}")
                
                # 检查底分型后的确认
                has_confirmation = False
                if idx + 1 < len(cleaned_data):
                    confirm_bar = cleaned_data.iloc[idx + 1]
                    # 确认K线收盘价高于底分型最低价且成交量温和
                    has_confirmation = confirm_bar.get("close", 0) > bottom_price
                    logger.info(f"底分型确认：下一根K线收盘价{confirm_bar.get('close', 0)} > {bottom_price}？ {has_confirmation}")
                
                # 计算买点分数（综合评估）
                buy_point_score = 0
                if is_in_safe_range:
                    buy_point_score += 0.4
                if has_volume_confirmation:
                    buy_point_score += 0.3
                if has_confirmation:
                    buy_point_score += 0.2
                # 级别优先级
                if level == "15min":
                    buy_point_score += 0.1
                
                # 构造买点信息
                buy_point = {
                    "level": level,
                    "time": str(bottom.name) if isinstance(bottom.name, pd.Timestamp) else str(bottom.name),
                    "price": bottom.get("close", 0),
                    "low_price": bottom_price,
                    "has_volume_confirmation": has_volume_confirmation,
                    "is_in_safe_range": is_in_safe_range,
                    "has_confirmation": has_confirmation,
                    "retracement_ratio": retracement_ratio,
                    "score": buy_point_score,
                    "valid": is_in_safe_range and (has_volume_confirmation or has_confirmation)  # 有效买点条件
                }
                
                buy_points.append(buy_point)
                logger.info(f"发现{level}回撤买点：价格={buy_point['price']}, 回撤幅度={buy_point['retracement_ratio']:.3f}, 评分={buy_point_score:.3f}")
        
        # 按评分和级别优先级排序买点
        buy_points.sort(key=lambda x: (x["score"], 0 if x["level"] == "15min" else 1), reverse=True)
        
        return buy_points
    
    def calculate_risk_reward_ratio(self, daily_signal_type: str, up_pen_level: str, weekly_trend_result: Optional[Dict] = None) -> float:
        """计算风险收益比
        
        Args:
            daily_signal_type: 日线买点类型
            up_pen_level: 向上笔级别
            weekly_trend_result: 周线趋势检测结果
            
        Returns:
            风险收益比（>1表示收益大于风险，越大越好）
        """
        # 基础风险收益比
        base_rr_ratio = 2.0  # 默认2:1的风险收益比
        
        # 根据日线买点类型调整
        if daily_signal_type == "日线二买":
            base_rr_ratio = 2.5  # 二买风险收益比更高
        elif daily_signal_type == "日线三买":
            base_rr_ratio = 2.2  # 三买次之
        elif daily_signal_type == "日线一买":
            base_rr_ratio = 1.8  # 一买相对较低
        
        # 根据向上笔级别调整
        if up_pen_level == "30min":
            base_rr_ratio *= 1.1  # 30分钟级别更可靠
        elif up_pen_level == "15min":
            base_rr_ratio *= 0.95  # 15分钟级别稍弱
        
        # 根据周线趋势调整
        if weekly_trend_result:
            confidence_score = weekly_trend_result.get("confidence_score", 0.5)
            # 置信度越高，风险收益比越好
            base_rr_ratio *= (0.8 + 0.4 * confidence_score)
        
        return max(1.0, base_rr_ratio)  # 确保至少1:1
    
    def allocate_position(self, 
                        daily_signal_type: str,
                        up_pen_level: str,
                        daily_position_ratio: float,
                        weekly_trend_result: Optional[Dict] = None,
                        buy_point_score: float = 1.0,
                        volatility_level: Optional[str] = None,
                        volatility_value: Optional[float] = None) -> Dict[str, float]:
        """根据日线买点类型和分钟向上笔级别分配子仓位（增强版）
        
        Args:
            daily_signal_type: 日线买点类型（日线二买/一买/三买）
            up_pen_level: 向上笔级别（30min/15min）
            daily_position_ratio: 用户原有日线总仓位比例
            weekly_trend_result: 周线趋势检测结果（用于获取置信度）
            buy_point_score: 买点评分（0-1之间，用于微调仓位）
            volatility_level: 波动等级（高/中/低波动）
            volatility_value: 波动率数值（百分比）
            
        Returns:
            子仓位分配结果字典
        """
        logger.info(f"分配子仓位：日线信号={daily_signal_type}, 向上笔级别={up_pen_level}, 日线总仓位={daily_position_ratio*100}%, 买点评分={buy_point_score:.2f}, 波动等级={volatility_level}, 波动率={volatility_value}%")
        
        # 获取优先级顺序
        priority_order = self.config["priority_order"]
        
        # 确定是否为主优先级匹配
        is_primary_match = False
        if up_pen_level == "30min" and daily_signal_type == priority_order[0]:  # 二买+30分钟
            is_primary_match = True
        elif up_pen_level == "15min" and daily_signal_type in priority_order[1:]:  # 一买/三买+15分钟
            is_primary_match = True
        
        # 根据匹配优先级确定子仓位比例范围
        if up_pen_level == "30min":
            if is_primary_match:  # 二买+30分钟
                sub_range = self.config["30min_primary_range"]
            else:  # 非二买+30分钟
                sub_range = self.config["30min_secondary_range"]
        elif up_pen_level == "15min":
            if is_primary_match:  # 一买/三买+15分钟
                sub_range = self.config["15min_primary_range"]
            else:  # 二买+15分钟（作为补充）
                sub_range = self.config["15min_secondary_range"]
        else:
            logger.error(f"不支持的向上笔级别: {up_pen_level}")
            return {"error": "不支持的向上笔级别"}
        
        # 计算基础子仓位比例（根据买点评分在范围内动态调整）
        # 买点评分越高，仓位越靠近范围上限
        range_size = sub_range[1] - sub_range[0]
        sub_ratio = sub_range[0] + range_size * min(max(buy_point_score, 0.5), 1.0)  # 确保在0.5-1.0区间
        
        # 1. 根据周线置信度等级调整仓位
        confidence_multiplier = 1.0
        if weekly_trend_result:
            confidence_level = weekly_trend_result.get("confidence_level", "MEDIUM")
            confidence_score = weekly_trend_result.get("confidence_score", 0.5)
            confidence_multiplier = self.config["confidence_level_multipliers"].get(confidence_level, 1.0)
            
            # 添加总置信度加权（MACD背驰+顶底分型）
            total_confidence_weight = weekly_trend_result.get("weekly_confidence_details", {}).get("total_weight", 1.0)
            confidence_multiplier *= total_confidence_weight
            
            logger.info(f"应用周线置信度调整：置信度={confidence_level}, 置信度分数={confidence_score:.2f}, 总置信度加权={total_confidence_weight:.2f}, 乘法因子={confidence_multiplier:.2f}")
        
        # 2. 根据波动等级调整仓位
        volatility_multiplier = 1.0
        if volatility_level:
            if volatility_level == "高波动":
                volatility_multiplier = 0.85  # 高波动降低仓位
            elif volatility_level == "低波动":
                volatility_multiplier = 1.15  # 低波动增加仓位
            logger.info(f"应用波动等级调整：波动等级={volatility_level}, 乘法因子={volatility_multiplier}")
        elif volatility_value:
            # 如果有波动率数值，更精确地调整
            if volatility_value > 18.0:  # 高波动
                volatility_multiplier = max(0.7, 1.2 - (volatility_value - 18) * 0.02)
            elif volatility_value < 10.0:  # 低波动
                volatility_multiplier = min(1.3, 0.9 + (10 - volatility_value) * 0.025)
            logger.info(f"应用波动率调整：波动率={volatility_value:.2f}%, 乘法因子={volatility_multiplier:.2f}")
        
        # 3. 根据风险收益比调整仓位
        risk_reward_ratio = self.calculate_risk_reward_ratio(daily_signal_type, up_pen_level, weekly_trend_result)
        rr_multiplier = 1.0
        if risk_reward_ratio > 2.5:
            rr_multiplier = 1.1  # 高风险收益比增加仓位
        elif risk_reward_ratio < 1.5:
            rr_multiplier = 0.9  # 低风险收益比降低仓位
        logger.info(f"应用风险收益比调整：风险收益比={risk_reward_ratio:.2f}, 乘法因子={rr_multiplier}")
        
        # 综合所有调整因子
        adjusted_sub_ratio = sub_ratio * confidence_multiplier * volatility_multiplier * rr_multiplier
        
        # 确保调整后的比例在合理范围内（0.1-0.9）
        adjusted_sub_ratio = min(max(adjusted_sub_ratio, 0.1), 0.9)
        
        # 计算实际仓位
        actual_position = daily_position_ratio * adjusted_sub_ratio
        
        # 计算子仓位的子分仓比例（用于分批建仓）
        first_batch_ratio = 0.5  # 第一批建仓比例
        second_batch_ratio = 0.3  # 第二批建仓比例
        third_batch_ratio = 0.2  # 第三批建仓比例
        
        # 根据综合因子调整分批比例
        combined_multiplier = confidence_multiplier * volatility_multiplier * rr_multiplier
        if combined_multiplier > 1.2:  # 综合高置信度
            first_batch_ratio = 0.65
            second_batch_ratio = 0.25
            third_batch_ratio = 0.10
        elif combined_multiplier < 0.8:  # 综合低置信度
            first_batch_ratio = 0.35
            second_batch_ratio = 0.35
            third_batch_ratio = 0.30
        
        # 构造结果
        allocation_result = {
            "daily_signal_type": daily_signal_type,
            "up_pen_level": up_pen_level,
            "daily_position_ratio": daily_position_ratio,
            "base_sub_position_ratio": sub_ratio,
            "adjusted_sub_position_ratio": adjusted_sub_ratio,
            "actual_position_ratio": actual_position,
            "is_primary_match": is_primary_match,
            "confidence_multiplier": confidence_multiplier,
            "volatility_multiplier": volatility_multiplier,
            "risk_reward_ratio": risk_reward_ratio,
            "rr_multiplier": rr_multiplier,
            "combined_multiplier": combined_multiplier,
            "buy_point_score": buy_point_score,
            "volatility_level": volatility_level,
            "volatility_value": volatility_value,
            "batch_allocation": {
                "first_batch": actual_position * first_batch_ratio,
                "second_batch": actual_position * second_batch_ratio,
                "third_batch": actual_position * third_batch_ratio
            }
        }
        
        logger.info(f"子仓位分配结果：基础子比例={sub_ratio*100}%, 调整后子比例={adjusted_sub_ratio*100}%, ")
        logger.info(f"实际仓位={actual_position*100}%, 分批建仓：第一批={allocation_result['batch_allocation']['first_batch']*100}%, ")
        logger.info(f"第二批={allocation_result['batch_allocation']['second_batch']*100}%, 第三批={allocation_result['batch_allocation']['third_batch']*100}%")
        logger.info(f"综合调整因子：置信度={confidence_multiplier:.2f}, 波动={volatility_multiplier:.2f}, 风险收益={rr_multiplier:.2f}, 总因子={combined_multiplier:.2f}")
        
        return allocation_result
    
    def generate_primary_trading_signal(self, 
                                      weekly_trend_result: Dict[str, any],
                                      daily_buy_result: Dict[str, any],
                                      minute_analysis_results: Dict[str, any],
                                      daily_position_ratio: Dict[str, float],
                                      minute_data: Optional[pd.DataFrame] = None,
                                      daily_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """生成主要交易信号（增强版）
        
        根据周线多头、日线买点类型和分钟级分析结果，生成综合交易信号
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            minute_analysis_results: 分钟级别分析结果
            daily_position_ratio: 用户原有日线仓位比例配置
            minute_data: 分钟级别数据（用于检测回撤买点）
            daily_data: 日线数据（用于量能计算）
            
        Returns:
            交易信号字典，如果没有有效信号则返回None
        """
        logger.info("生成主要交易信号（增强版）")
        
        # 检查周线多头趋势
        if not weekly_trend_result.get("bullish_trend", False):
            logger.info("周线非多头，不生成交易信号")
            return None
        
        # 获取最强日线买点信号
        daily_signal_type = daily_buy_result.get("strongest_signal", "")
        if daily_signal_type == "无买点":
            logger.info("无有效的日线买点，不生成交易信号")
            return None
        
        # 获取日线买点优先级
        signal_type_priority = daily_buy_result.get("signal_type_priority", "")
        
        # 检查分钟级别分析结果
        best_minute_match = None
        best_allocation = None
        best_buy_points = None
        best_level = None
        
        # 按优先级检查分钟级别匹配情况
        priority_order = self.config["priority_order"]
        
        # 优先匹配30分钟向上笔（尤其是对二买）
        if "30min" in minute_analysis_results and minute_analysis_results["30min"].get("up_pen_completed", False):
            # 检查是否为二买
            if daily_signal_type == priority_order[0]:  # 二买+30分钟（最优组合）
                # 获取用户配置的日线仓位
                daily_ratio = daily_position_ratio.get(daily_signal_type, 0)
                
                # 检测回撤买点
                if minute_data is not None:
                    buy_points = self.detect_retracement_buy_points(
                        minute_data,
                        "30min",
                        daily_signal_type,
                        daily_data,
                        minute_analysis_results["30min"],
                        weekly_trend_result
                    )
                    
                    if buy_points and buy_points[0].get("valid", False):
                        # 使用最佳买点的评分进行仓位分配
                        best_buy_point = buy_points[0]
                        allocation = self.allocate_position(
                            daily_signal_type, 
                            "30min", 
                            daily_ratio,
                            weekly_trend_result,
                            best_buy_point.get("score", 1.0)
                        )
                        best_minute_match = minute_analysis_results["30min"]
                        best_allocation = allocation
                        best_buy_points = buy_points
                        best_level = "30min"
                        logger.info(f"找到最佳匹配：{daily_signal_type}+30分钟向上笔，最佳买点评分={best_buy_point.get('score', 1.0):.2f}")
        
        # 如果未找到最佳匹配，检查15分钟向上笔
        if best_minute_match is None and "15min" in minute_analysis_results and minute_analysis_results["15min"].get("up_pen_completed", False):
            # 获取用户配置的日线仓位
            daily_ratio = daily_position_ratio.get(daily_signal_type, 0)
            
            # 检测回撤买点
            if minute_data is not None:
                buy_points = self.detect_retracement_buy_points(
                    minute_data,
                    "15min",
                    daily_signal_type,
                    daily_data,
                    minute_analysis_results["15min"],
                    weekly_trend_result
                )
                
                if buy_points and buy_points[0].get("valid", False):
                    # 使用最佳买点的评分进行仓位分配
                    best_buy_point = buy_points[0]
                    allocation = self.allocate_position(
                        daily_signal_type, 
                        "15min", 
                        daily_ratio,
                        weekly_trend_result,
                        best_buy_point.get("score", 1.0)
                    )
                    best_minute_match = minute_analysis_results["15min"]
                    best_allocation = allocation
                    best_buy_points = buy_points
                    best_level = "15min"
                    logger.info(f"找到匹配：{daily_signal_type}+15分钟向上笔，最佳买点评分={best_buy_point.get('score', 1.0):.2f}")
        
        # 如果仍未找到匹配，尝试次要优先级
        if best_minute_match is None:
            secondary_order = self.config["secondary_priority_order"]
            for level in ["30min", "15min"]:
                if level in minute_analysis_results and minute_analysis_results[level].get("up_pen_completed", False):
                    daily_ratio = daily_position_ratio.get(daily_signal_type, 0)
                    
                    # 检测回撤买点
                    if minute_data is not None:
                        buy_points = self.detect_retracement_buy_points(
                            minute_data,
                            level,
                            daily_signal_type,
                            daily_data,
                            minute_analysis_results[level],
                            weekly_trend_result
                        )
                        
                        if buy_points and buy_points[0].get("valid", False):
                            best_buy_point = buy_points[0]
                            allocation = self.allocate_position(
                                daily_signal_type, 
                                level, 
                                daily_ratio,
                                weekly_trend_result,
                                best_buy_point.get("score", 1.0)
                            )
                            best_minute_match = minute_analysis_results[level]
                            best_allocation = allocation
                            best_buy_points = buy_points
                            best_level = level
                            logger.info(f"找到次要匹配：{daily_signal_type}+{level}向上笔，最佳买点评分={best_buy_point.get('score', 1.0):.2f}")
                            break
        
        # 如果仍未找到匹配，则不生成交易信号
        if best_minute_match is None or best_allocation is None or best_buy_points is None:
            logger.info("未找到有效的分钟级别匹配或回撤买点，不生成交易信号")
            return None
        
        # 获取最佳买点
        best_buy_point = best_buy_points[0]
        
        # 计算建仓时间窗口
        time_window = self.config["position_building_windows"].get(best_level, 3)  # 默认3小时
        
        # 获取周线置信度信息
        confidence_level = weekly_trend_result.get("confidence_level", "MEDIUM")
        component_scores = weekly_trend_result.get("component_scores", {})
        
        # 检测入场条件，添加用户要求的entry_window和best_price_range字段
        entry_conditions = self._detect_entry_conditions(
            daily_signal_type, 
            minute_data, 
            weekly_trend_result, 
            daily_buy_result,
            best_buy_point  # 传递最佳买点信息
        )
        
        # 构造交易信号
        signal = {
            "signal_type": f"周线多头{confidence_level}+{daily_signal_type}-{best_level.replace('min', '')}分钟建仓",
            "signal_type_priority": signal_type_priority,
            "confidence_level": confidence_level,
            "component_scores": component_scores,
            "base_sub_position_ratio": best_allocation.get("base_sub_position_ratio", 0),
            "adjusted_sub_position_ratio": best_allocation.get("adjusted_sub_position_ratio", 0),
            "actual_position": best_allocation["actual_position_ratio"],
            "actual_position_display": f"用户原有日线仓位×{best_allocation.get('adjusted_sub_position_ratio', 0)*100:.1f}%",
            "allocation_info": best_allocation,
            "minute_analysis": best_minute_match,
            "best_buy_point": best_buy_point,
            "position_building_time_window": time_window,
            "batch_allocation": best_allocation.get("batch_allocation", {}),
            "entry_window": entry_conditions.get("entry_window"),  # 添加入场时间窗口
            "best_price_range": entry_conditions.get("best_price_range"),  # 添加最佳价格区间
            "trading_recommendation": {
                "entry_price": best_buy_point.get("price", 0),
                "max_entry_price": best_buy_point.get("price", 0) * 1.01,  # 允许1%的滑点
                "batch_strategy": self._generate_batch_strategy(best_allocation, best_buy_point)
            }
        }
        
        logger.info(f"生成交易信号：{signal['signal_type']}")
        logger.info(f"建仓建议：入场价格={best_buy_point.get('price', 0)}, 分批策略={signal['trading_recommendation']['batch_strategy']}")
        
        return signal
    
    def _generate_batch_strategy(self, allocation: Dict, best_buy_point: Dict) -> str:
        """生成分批建仓策略
        
        Args:
            allocation: 仓位分配信息
            best_buy_point: 最佳买点信息
            
        Returns:
            分批建仓策略描述
        """
        batch_allocation = allocation.get("batch_allocation", {})
        first_batch = batch_allocation.get("first_batch", 0)
        second_batch = batch_allocation.get("second_batch", 0)
        third_batch = batch_allocation.get("third_batch", 0)
        
        # 根据买点评分和级别生成不同的分批策略
        buy_point_score = best_buy_point.get("score", 1.0)
        level = best_buy_point.get("level", "15min")
        
        if buy_point_score >= 0.8 and level == "15min":
            # 高质量买点，可以更积极建仓
            strategy = f"第一批{first_batch*100:.1f}%立即建仓，第二批{second_batch*100:.1f}%在价格回调至{best_buy_point.get('low_price', 0)*0.995:.2f}附近建仓，第三批{third_batch*100:.1f}%在更小级别回调时建仓"
        elif buy_point_score >= 0.6:
            # 中等质量买点，稳健建仓
            strategy = f"第一批{first_batch*100:.1f}%立即建仓，第二批{second_batch*100:.1f}%在价格回调至{best_buy_point.get('price', 0)*0.995:.2f}附近建仓，第三批{third_batch*100:.1f}%在确认上涨后建仓"
        else:
            # 较低质量买点，保守建仓
            strategy = f"第一批{first_batch*100:.1f}%轻仓试探，第二批{second_batch*100:.1f}%在确认突破后建仓，第三批{third_batch*100:.1f}%在回抽确认时建仓"
        
        return strategy

    def _calculate_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: K线数据
            fast: 快线参数
            slow: 慢线参数
            signal: 信号线参数
            
        Returns:
            包含MACD指标的数据帧
        """
        df = df.copy()
        df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        return df

    def _detect_5min_macd_golden_cross(self, five_min_data: pd.DataFrame) -> Tuple[bool, pd.Timestamp]:
        """检测5分钟MACD金叉
        
        Args:
            five_min_data: 5分钟级别数据
            
        Returns:
            (是否发生金叉, 金叉时间)
        """
        if len(five_min_data) < 30:  # 需要足够数据计算MACD
            return False, None
        
        five_min_data = self._calculate_macd(five_min_data)
        
        # 检测最后一根K线是否形成金叉
        last_cross = False
        cross_time = None
        
        for i in range(1, len(five_min_data)):
            if five_min_data['MACD_Hist'].iloc[i-1] <= 0 and five_min_data['MACD_Hist'].iloc[i] > 0:
                last_cross = True
                cross_time = five_min_data.index[i]
        
        return last_cross, cross_time

    def _detect_15min_bottom_fractal(self, fifteen_min_data: pd.DataFrame) -> bool:
        """检测15分钟底分型
        
        Args:
            fifteen_min_data: 15分钟级别数据
            
        Returns:
            是否存在底分型
        """
        if len(fifteen_min_data) < 5:  # 需要至少5根K线
            return False
        
        return self._identify_bottom_fractal(fifteen_min_data)

    def _detect_entry_conditions(self, daily_signal_type: str, minute_data: pd.DataFrame, 
                               weekly_trend_result: Dict, daily_buy_result: Dict, 
                               best_buy_point: Optional[Dict] = None) -> Dict:
        """检测不同日线信号的入场条件
        
        Args:
            daily_signal_type: 日线信号类型
            minute_data: 分钟级别数据
            weekly_trend_result: 周线趋势结果
            daily_buy_result: 日线买点结果
            best_buy_point: 最佳买点信息
            
        Returns:
            入场条件检测结果
        """
        # 重采样不同级别的数据
        fifteen_min_data = self._resample_to_minute_level(minute_data, '15min')
        thirty_min_data = self._resample_to_minute_level(minute_data, '30min')
        five_min_data = self._resample_to_minute_level(minute_data, '5min')
        
        result = {
            "entry_window": None,  # 入场时间窗口（分钟）
            "best_price_range": None,  # 最佳价格区间 (min, max)
            "signal_type": daily_signal_type,
            "conditions_met": False
        }
        
        current_time = minute_data.index[-1] if not minute_data.empty else pd.Timestamp.now()
        
        # 日线一买条件
        if daily_signal_type == "日线一买":
            # 检测15分钟底分型
            has_bottom_fractal = self._detect_15min_bottom_fractal(fifteen_min_data)
            
            # 检测5分钟MACD金叉
            has_golden_cross, cross_time = self._detect_5min_macd_golden_cross(five_min_data)
            
            if has_bottom_fractal and has_golden_cross and cross_time:
                # 计算入场窗口：金叉后10-20分钟
                entry_window_start = cross_time + pd.Timedelta(minutes=10)
                entry_window_end = cross_time + pd.Timedelta(minutes=20)
                
                # 检查当前时间是否在入场窗口内
                if entry_window_start <= current_time <= entry_window_end:
                    # 获取回撤低点
                    if best_buy_point is not None:
                        low_price = best_buy_point.get("low_price", fifteen_min_data['low'].min())
                    else:
                        low_price = fifteen_min_data['low'].min()
                      
                    # 计算最佳价格区间：回撤低点±0.3%
                    price_tolerance = low_price * 0.003
                    price_range = (low_price - price_tolerance, low_price + price_tolerance)
                    
                    result["entry_window"] = {
                        "start_time": entry_window_start,
                        "end_time": entry_window_end,
                        "window_minutes": 10
                    }
                    result["best_price_range"] = price_range
                    result["conditions_met"] = True
        
        # 破中枢反抽条件
        elif daily_signal_type == "破中枢反抽":
            # 获取反抽阈值
            retracement_threshold = daily_buy_result.get("retracement_threshold", thirty_min_data['close'].iloc[-1])
            
            # 检查30分钟是否站稳反抽阈值（最近2根K线收盘价≥阈值）
            thirty_min_close = thirty_min_data['close'].tail(2)
            if len(thirty_min_close) >= 2 and all(thirty_min_close >= retracement_threshold):
                # 检测15分钟MACD金叉
                fifteen_min_data = self._calculate_macd(fifteen_min_data)
                if len(fifteen_min_data) >= 2:
                    fifteen_min_macd_hist = fifteen_min_data['MACD_Hist'].tail(2)
                    if fifteen_min_macd_hist.iloc[0] <= 0 and fifteen_min_macd_hist.iloc[1] > 0:
                        # 检查5分钟收盘价是否≥反抽阈值
                        if not five_min_data.empty and five_min_data['close'].iloc[-1] >= retracement_threshold:
                            # 计算入场窗口：确认后15分钟内
                            entry_window_start = current_time
                            entry_window_end = current_time + pd.Timedelta(minutes=15)
                            
                            # 计算最佳价格区间：反抽阈值±0.5%
                            price_tolerance = retracement_threshold * 0.005
                            price_range = (retracement_threshold - price_tolerance, retracement_threshold + price_tolerance)
                            
                            result["entry_window"] = {
                                "start_time": entry_window_start,
                                "end_time": entry_window_end,
                                "window_minutes": 15
                            }
                            result["best_price_range"] = price_range
                            result["conditions_met"] = True
        
        # 日线二买条件
        elif daily_signal_type == "日线二买":
            # 入场窗口：30分钟向上笔确认后30分钟内
            entry_window_start = current_time
            entry_window_end = current_time + pd.Timedelta(minutes=30)
            
            # 获取向上笔起点价格
            if best_buy_point is not None:
                up_pen_start_price = best_buy_point.get("price", thirty_min_data['low'].iloc[0])
            else:
                up_pen_start_price = thirty_min_data['low'].iloc[0] if not thirty_min_data.empty else thirty_min_data['close'].iloc[-1]
            
            # 计算最佳价格区间：向上笔起点±0.4%
            price_tolerance = up_pen_start_price * 0.004
            price_range = (up_pen_start_price - price_tolerance, up_pen_start_price + price_tolerance)
            
            result["entry_window"] = {
                "start_time": entry_window_start,
                "end_time": entry_window_end,
                "window_minutes": 30
            }
            result["best_price_range"] = price_range
            result["conditions_met"] = True
        
        return result
    
    def _remove_inclusion(self, data: pd.DataFrame) -> pd.DataFrame:
        """去除K线包含关系（完善实现）
        
        Args:
            data: K线数据
            
        Returns:
            去除包含关系后的K线数据
        """
        logger.info("开始处理K线包含关系")
        
        # 复制数据，避免修改原始数据
        processed_data = data.copy()
        i = 1
        
        while i < len(processed_data) - 1:
            # 检查前一根K线与当前K线的包含关系
            prev_high = processed_data.iloc[i-1]['high']
            prev_low = processed_data.iloc[i-1]['low']
            curr_high = processed_data.iloc[i]['high']
            curr_low = processed_data.iloc[i]['low']
            
            # 判断是否存在包含关系
            has_inclusion = False
            if curr_high <= prev_high and curr_low >= prev_low:
                # 当前K线被前一根K线包含
                has_inclusion = True
                # 向上处理：取高点中的高点，低点中的高点
                if processed_data.iloc[i-2]['high'] > processed_data.iloc[i-1]['high']:
                    # 向上趋势，包含处理为：高取高，低取高
                    new_high = max(prev_high, curr_high)
                    new_low = max(prev_low, curr_low)
                else:
                    # 向下趋势，包含处理为：高取低，低取低
                    new_high = min(prev_high, curr_high)
                    new_low = min(prev_low, curr_low)
                
                # 更新前一根K线
                processed_data.iloc[i-1, processed_data.columns.get_loc('high')] = new_high
                processed_data.iloc[i-1, processed_data.columns.get_loc('low')] = new_low
                # 重新计算开盘价和收盘价（简化处理）
                processed_data.iloc[i-1, processed_data.columns.get_loc('open')] = processed_data.iloc[i-1]['open']
                processed_data.iloc[i-1, processed_data.columns.get_loc('close')] = processed_data.iloc[i]['close']
                
                # 删除当前K线
                processed_data = processed_data.drop(processed_data.index[i])
                i -= 1  # 回退一步，重新检查
            
            i += 1
            
        logger.info(f"包含关系处理完成，K线数量从{len(data)}减少到{len(processed_data)}")
        return processed_data
    
    def _identify_top_fractal(self, data: pd.DataFrame) -> List[int]:
        """识别顶分型
        
        Args:
            data: K线数据
            
        Returns:
            顶分型索引列表
        """
        # 这是一个简化的实现，实际的顶分型识别需要严格按照缠论定义
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
    
    def _identify_bottom_fractal(self, data: pd.DataFrame) -> List[int]:
        """识别底分型
        
        Args:
            data: K线数据
            
        Returns:
            底分型索引列表
        """
        # 这是一个简化的实现，实际的底分型识别需要严格按照缠论定义
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
    
    def _resample_to_minute_level(self, minute_data: pd.DataFrame, target_level: str) -> pd.DataFrame:
        """将分钟数据重采样到目标级别
        
        Args:
            minute_data: 分钟级别数据
            target_level: 目标分钟级别（15min/5min）
            
        Returns:
            重采样后的数据
        """
        logger.info(f"将分钟数据重采样到{target_level}级别")
        
        # 确保数据有时间索引
        if not isinstance(minute_data.index, pd.DatetimeIndex):
            # 尝试将'date'或'time'列转换为索引
            if 'date' in minute_data.columns:
                minute_data = minute_data.set_index('date')
            elif 'time' in minute_data.columns:
                minute_data = minute_data.set_index('time')
            else:
                logger.warning("无法确定时间列，返回原始数据")
                return minute_data
            
            # 确保索引是DatetimeIndex
            if not isinstance(minute_data.index, pd.DatetimeIndex):
                try:
                    minute_data.index = pd.to_datetime(minute_data.index)
                except:
                    logger.error("时间列转换失败，返回原始数据")
                    return minute_data
        
        # 根据目标级别确定重采样频率
        if target_level == "15min":
            freq = '15T'
        elif target_level == "5min":
            freq = '5T'
        elif target_level == "30min":
            freq = '30T'
        else:
            logger.error(f"不支持的目标级别：{target_level}，返回原始数据")
            return minute_data
        
        # 重采样
        try:
            resampled_data = minute_data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"重采样完成，数据行数从{len(minute_data)}减少到{len(resampled_data)}")
            return resampled_data
        except Exception as e:
            logger.error(f"重采样过程中出错：{str(e)}，返回原始数据")
            return minute_data


# 测试代码
def test_minute_position_allocator():
    """测试分钟级别子分仓系统"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建分配器实例
    allocator = MinutePositionAllocator()
    
    # 创建模拟的K线数据
    def create_mock_kline_data(levels=30, count=20):
        """创建模拟的K线数据"""
        index = pd.date_range(start='2024-01-01', periods=count, freq=f'{levels}T')
        
        # 创建向上笔的K线数据
        prices = np.linspace(100, 110, count-10)  # 上涨部分
        prices = np.concatenate([prices, np.linspace(110, 105, 10)])  # 回调部分
        
        data = {
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 5000, count)
        }
        
        return pd.DataFrame(data, index=index)
    
    # 测试向上笔检测
    print("\n===== 测试向上笔检测 =====")
    
    # 创建模拟数据
    mock_30min_data = create_mock_kline_data(levels=30, count=30)
    mock_15min_data = create_mock_kline_data(levels=15, count=50)
    
    # 检测30分钟向上笔
    print("\n检测30分钟向上笔：")
    result_30min = allocator.detect_minute_up_pen(mock_30min_data, "30min")
    print(f"向上笔是否完成: {result_30min['up_pen_completed']}")
    print(f"是否有顶分型: {result_30min['has_top_fractal']}")
    print(f"信号: {result_30min['signal']}")
    
    # 检测15分钟向上笔
    print("\n检测15分钟向上笔：")
    result_15min = allocator.detect_minute_up_pen(mock_15min_data, "15min")
    print(f"向上笔是否完成: {result_15min['up_pen_completed']}")
    print(f"是否有顶分型: {result_15min['has_top_fractal']}")
    print(f"信号: {result_15min['signal']}")
    
    # 测试子仓位分配
    print("\n===== 测试子仓位分配 =====")
    
    # 模拟用户配置的日线仓位
    daily_position_ratios = {
        "日线二买": 0.3,  # 30%
        "日线一买": 0.2,  # 20%
        "日线三买": 0.25  # 25%
    }
    
    # 测试二买+30分钟组合（最优）
    print("\n二买+30分钟组合（最优）：")
    allocation = allocator.allocate_position("日线二买", "30min", daily_position_ratios["日线二买"])
    print(f"子仓位比例: {allocation['sub_position_ratio']*100}%")
    print(f"实际仓位: {allocation['actual_position_ratio']*100}%")
    
    # 测试一买+15分钟组合
    print("\n一买+15分钟组合：")
    allocation = allocator.allocate_position("日线一买", "15min", daily_position_ratios["日线一买"])
    print(f"子仓位比例: {allocation['sub_position_ratio']*100}%")
    print(f"实际仓位: {allocation['actual_position_ratio']*100}%")
    
    # 测试二买+15分钟组合（作为补充）
    print("\n二买+15分钟组合（作为补充）：")
    allocation = allocator.allocate_position("日线二买", "15min", daily_position_ratios["日线二买"])
    print(f"子仓位比例: {allocation['sub_position_ratio']*100}%")
    print(f"实际仓位: {allocation['actual_position_ratio']*100}%")
    
    # 测试交易信号生成
    print("\n===== 测试交易信号生成 =====")
    
    # 模拟各模块结果
    weekly_trend_result = {
        "bullish_trend": True,
        "macd_above_zero": True,
        "continuous_falling_weeks": 0
    }
    
    daily_buy_result = {
        "strongest_signal": "日线二买",
        "signal_type_priority": "核心",
        "signals": {
            "second_buy": {"detected": True}
        }
    }
    
    minute_analysis_results = {
        "30min": result_30min,
        "15min": result_15min
    }
    
    # 生成交易信号
    signal = allocator.generate_primary_trading_signal(
        weekly_trend_result,
        daily_buy_result,
        minute_analysis_results,
        daily_position_ratios
    )
    
    if signal:
        print(f"交易信号类型: {signal['signal_type']}")
        print(f"日线买点类型优先级: {signal['signal_type_priority']}")
        print(f"分钟子仓位比例: {signal['sub_position_ratio']*100}%")
        print(f"实际仓位: {signal['actual_position_display']}")
    else:
        print("未生成交易信号")
    
    # 测试无效情况（周线非多头）
    print("\n测试无效情况（周线非多头）：")
    weekly_trend_result_invalid = {
        "bullish_trend": False,
        "macd_above_zero": False,
        "continuous_falling_weeks": 3
    }
    
    signal_invalid = allocator.generate_primary_trading_signal(
        weekly_trend_result_invalid,
        daily_buy_result,
        minute_analysis_results,
        daily_position_ratios
    )
    
    print(f"是否生成信号: {signal_invalid is not None}")


if __name__ == "__main__":
    test_minute_position_allocator()