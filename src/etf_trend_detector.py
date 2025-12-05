#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF通用横盘+向上笔判定规则实现

功能：
1. 横盘中枢自动识别（任意ETF通用）
2. ETF共性低波动适配规则（全品类适用）
3. 通用判定逻辑（所有ETF统一执行）
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ETFTrendDetector')

class ETFTrendDetector:
    """
    ETF通用趋势检测器
    实现横盘识别、中枢划分和向上笔判定的通用逻辑
    """
    
    # 配置参数 - 可在初始化时覆盖
    DEFAULT_CONFIG = {
        # 横盘检测配置
        'sideways_k_count': 20,           # 横盘判定的K线数量
        'sideways_amplitude_threshold': 15.0,  # 横盘振幅阈值(%)
        'sideways_consecutive_limit': 3,  # 连续突破限制
        
        # 向上笔幅度阈值(%)
        'up_leg_min_amplitude': {
            'daily': 3.0,                 # 日线向上笔最小涨幅
            '30min': 2.0,                 # 30分钟向上笔最小涨幅
            '15min': 1.5                  # 15分钟向上笔最小涨幅
        },
        
        # 突破有效性阈值(%)
        'breakout_threshold': {
            'daily': 0.5,                 # 日线突破中枢上沿最小幅度
            '30min': 0.3,                 # 30分钟突破中枢上沿最小幅度
            '15min': 0.2                  # 15分钟突破中枢上沿最小幅度
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化ETF趋势检测器
        
        Args:
            config: 自定义配置字典，可覆盖默认配置
        """
        # 合并配置
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        logger.info(f"ETF趋势检测器初始化完成，配置: {self.config}")
    
    def detect_sideways_market(self, df: pd.DataFrame, price_hold_days: int = None) -> Dict[str, Union[bool, Dict[str, float], str]]:
        """
        检测ETF是否处于横盘环境
        
        判定标准：
        1. 近20根日线K线振幅≤15%
        2. 高低点交替（无连续3根K线突破前高/前低）
        3. 可选：价格在中枢区间内持续≥price_hold_days个交易日（默认不启用）
        
        Args:
            df: 包含K线数据的DataFrame，需要包含'high', 'low', 'close'列
                并按时间顺序排序（最新数据在前或在后都可以，会自动处理）
            price_hold_days: 价格在中枢区间内持续的最少交易日数，None表示不启用时间维度判定
        
        Returns:
            Dict: 包含横盘检测结果的字典
                - is_sideways: 是否为横盘环境
                - amplitude: 计算的振幅(%)
                - center_range: 中枢区间{"lower": float, "upper": float}
                - reason: 判定理由说明
                - consecutive_high_breaks: 连续突破前高的最大次数
                - consecutive_low_breaks: 连续突破前低的最大次数
                - price_hold_days: 实际持续天数（如果启用了时间维度判定）
        """
        try:
            # 检查必要的列
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"数据缺少必要的列: {col}")
            
            # 获取最近N根K线数据（取最新的20根）
            df_recent = df.head(self.config['sideways_k_count']).copy()
            if len(df_recent) < self.config['sideways_k_count']:
                return {
                    'is_sideways': False,
                    'amplitude': 0.0,
                    'center_range': {'lower': 0.0, 'upper': 0.0},
                    'consecutive_high_breaks': 0,
                    'consecutive_low_breaks': 0,
                    'reason': f"数据不足，需要至少{self.config['sideways_k_count']}根K线"
                }
            
            # 计算振幅 - 严格按照需求：(最高价-最低价)/最低价 * 100%
            highest_high = df_recent['high'].max()
            lowest_low = df_recent['low'].min()
            amplitude = ((highest_high - lowest_low) / lowest_low) * 100
            logger.info(f"计算振幅: 最高={highest_high:.4f}, 最低={lowest_low:.4f}, 振幅={amplitude:.2f}%")
            
            # 计算中枢边界
            center_lower = df_recent['close'].min()  # 最低收盘价
            center_upper = df_recent['close'].max()  # 最高收盘价
            
            # 检测高低点交替（使用收盘价检测连续趋势）
            # 按时间顺序（从旧到新）处理
            df_sorted = df_recent.sort_index()
            closes = df_sorted['close'].values
            
            max_consecutive_high = 0
            max_consecutive_low = 0
            current_consecutive_high = 0
            current_consecutive_low = 0
            
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    current_consecutive_high += 1
                    current_consecutive_low = 0
                elif closes[i] < closes[i-1]:
                    current_consecutive_low += 1
                    current_consecutive_high = 0
                else:
                    current_consecutive_high = 0
                    current_consecutive_low = 0
                
                max_consecutive_high = max(max_consecutive_high, current_consecutive_high)
                max_consecutive_low = max(max_consecutive_low, current_consecutive_low)
            
            # 基础判定：振幅≤15%且无连续3根K线突破前高/前低
            base_is_sideways = (amplitude <= self.config['sideways_amplitude_threshold'] and 
                              max_consecutive_high < self.config['sideways_consecutive_limit'] and 
                              max_consecutive_low < self.config['sideways_consecutive_limit'])
            
            # 计算价格在中枢区间内的持续天数
            if base_is_sideways and price_hold_days is not None:
                # 按时间顺序（从旧到新）处理
                df_sorted = df_recent.sort_index()
                in_range_count = 0
                max_hold_days = 0
                
                for _, row in df_sorted.iterrows():
                    close = row['close']
                    if center_lower <= close <= center_upper:
                        in_range_count += 1
                        max_hold_days = max(max_hold_days, in_range_count)
                    else:
                        in_range_count = 0
                
                # 应用时间维度条件
                is_sideways = base_is_sideways and max_hold_days >= price_hold_days
                price_hold_days_actual = max_hold_days
            else:
                is_sideways = base_is_sideways
                price_hold_days_actual = None
            
            # 构建详细的判定理由
            reason_parts = []
            
            # 振幅信息
            if amplitude <= self.config['sideways_amplitude_threshold']:
                reason_parts.append(f"振幅: {amplitude:.2f}% ≤ {self.config['sideways_amplitude_threshold']}%")
            else:
                reason_parts.append(f"振幅: {amplitude:.2f}% > {self.config['sideways_amplitude_threshold']}%")
            
            # 连续突破信息
            if max_consecutive_high >= self.config['sideways_consecutive_limit']:
                reason_parts.append(f"存在连续上涨: 最多{max_consecutive_high}次")
            if max_consecutive_low >= self.config['sideways_consecutive_limit']:
                reason_parts.append(f"存在连续下跌: 最多{max_consecutive_low}次")
            
            if max_consecutive_high < self.config['sideways_consecutive_limit'] and max_consecutive_low < self.config['sideways_consecutive_limit']:
                reason_parts.append(f"高低点交替良好")
            
            # 时间维度信息
            if price_hold_days is not None:
                if price_hold_days_actual >= price_hold_days:
                    reason_parts.append(f"价格在中枢区间内持续天数: {price_hold_days_actual} ≥ {price_hold_days}天")
                else:
                    reason_parts.append(f"价格在中枢区间内持续天数: {price_hold_days_actual} < {price_hold_days}天")
            
            # 添加中枢区间信息
            reason_parts.append(f"中枢区间: [{center_lower:.4f}, {center_upper:.4f}]")
            
            # 返回详细结果
            result = {
                'is_sideways': is_sideways,
                'amplitude': amplitude,
                'center_range': {
                    'lower': center_lower,
                    'upper': center_upper
                },
                'consecutive_high_breaks': max_consecutive_high,
                'consecutive_low_breaks': max_consecutive_low,
                'max_consecutive_high_breaks': max_consecutive_high,  # 保持向后兼容
                'max_consecutive_low_breaks': max_consecutive_low,    # 保持向后兼容
                'price_hold_days': price_hold_days_actual,  # 新增：价格在中枢区间内的持续天数
                'reason': '; '.join(reason_parts)
            }
            
            logger.info(f"横盘检测结果: {'是' if is_sideways else '否'} - {result['reason']}")
            logger.info(f"  连续上涨: 最多{max_consecutive_high}次")
            logger.info(f"  连续下跌: 最多{max_consecutive_low}次")
            
            return result
            
        except Exception as e:
            logger.error(f"横盘检测失败: {str(e)}")
            return {
                'is_sideways': False,
                'amplitude': 0.0,
                'center_range': {'lower': 0.0, 'upper': 0.0},
                'consecutive_high_breaks': 0,
                'consecutive_low_breaks': 0,
                'reason': f"检测失败: {str(e)}"
            }
    
    def validate_up_leg_amplitude(self, start_price: float, current_price: float, 
                                timeframe: str = 'daily') -> Tuple[bool, float]:
        """
        验证向上笔幅度是否满足阈值要求
        
        ETF共性低波动适配规则：
        - 日线向上笔：涨幅≥3%
        - 30分钟向上笔：涨幅≥2%
        - 15分钟向上笔：涨幅≥1.5%
        
        Args:
            start_price: 起始价格（向上笔的起点价格）
            current_price: 当前价格（向上笔的终点价格）
            timeframe: 时间周期，可选 'daily', '30min', '15min'
        
        Returns:
            Tuple[bool, float]: (是否满足幅度要求, 实际涨幅百分比)
        """
        try:
            # 参数验证
            if start_price <= 0 or current_price <= 0:
                raise ValueError("价格必须为正数")
            
            # 计算涨幅百分比：(当前价格 - 起始价格) / 起始价格 * 100%
            amplitude = ((current_price - start_price) / start_price) * 100
            
            # 获取对应时间周期的阈值
            # 确保严格按照ETF共性低波动适配规则设置阈值
            min_amplitude = self.config['up_leg_min_amplitude'].get(timeframe, 3.0)
            
            # 明确记录各时间周期的阈值要求
            timeframe_threshold_info = {
                'daily': '日线向上笔，阈值≥3%',
                '30min': '30分钟向上笔，阈值≥2%',
                '15min': '15分钟向上笔，阈值≥1.5%'
            }
            
            # 判断是否满足幅度要求
            meets_requirement = amplitude >= min_amplitude
            
            # 详细日志记录
            threshold_desc = timeframe_threshold_info.get(timeframe, f'未知时间周期: {timeframe}')
            logger.info(f"向上笔幅度验证 - {threshold_desc}")
            logger.info(f"  起始价格: {start_price:.4f}, 当前价格: {current_price:.4f}")
            logger.info(f"  涨幅: {amplitude:.2f}%, 阈值: {min_amplitude}%, 满足: {meets_requirement}")
            
            return meets_requirement, amplitude
            
        except ValueError as ve:
            logger.error(f"向上笔幅度验证参数错误: {str(ve)}")
            return False, 0.0
        except Exception as e:
            logger.error(f"向上笔幅度验证失败: {str(e)}")
            return False, 0.0
    
    def calculate_up_leg_info(self, df: pd.DataFrame, start_index: int, end_index: int, 
                            timeframe: str = 'daily') -> Dict[str, Union[bool, float, str]]:
        """
        计算向上笔的详细信息，包括幅度、持续时间等
        
        Args:
            df: 包含K线数据的DataFrame
            start_index: 向上笔起始索引
            end_index: 向上笔结束索引
            timeframe: 时间周期
        
        Returns:
            Dict: 向上笔详细信息
        """
        try:
            # 参数验证
            if start_index < 0 or end_index >= len(df) or start_index > end_index:
                raise ValueError(f"索引范围无效: start={start_index}, end={end_index}, length={len(df)}")
            
            # 获取起始和结束价格
            start_price = df.iloc[start_index]['close']
            end_price = df.iloc[end_index]['close']
            
            # 计算涨幅
            amplitude_valid, amplitude = self.validate_up_leg_amplitude(
                start_price, end_price, timeframe
            )
            
            # 计算持续时间（K线数量）
            duration = end_index - start_index + 1
            
            # 获取对应时间周期的阈值描述
            threshold_map = {
                'daily': {'amplitude': 3.0, 'breakout': 0.5},
                '30min': {'amplitude': 2.0, 'breakout': 0.3},
                '15min': {'amplitude': 1.5, 'breakout': 0.2}
            }
            
            threshold_info = threshold_map.get(timeframe, {'amplitude': 3.0, 'breakout': 0.5})
            
            return {
                'valid': amplitude_valid,
                'amplitude': amplitude,
                'amplitude_threshold': threshold_info['amplitude'],
                'start_price': start_price,
                'end_price': end_price,
                'duration': duration,
                'timeframe': timeframe,
                'satisfied_conditions': [f"涨幅≥{threshold_info['amplitude']}%"] if amplitude_valid else []
            }
            
        except Exception as e:
            logger.error(f"计算向上笔信息失败: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'amplitude': 0.0,
                'amplitude_threshold': 0.0
            }
    
    def validate_breakout_effectiveness(self, price: float, center_upper: float, 
                                      timeframe: str = 'daily') -> Tuple[bool, float]:
        """
        验证突破中枢上沿的有效性
        
        突破有效性阈值（相对比例，适配任意价格的ETF）：
        - 日线：向上笔需突破对应级别中枢上沿≥0.5%
        - 30分钟：向上笔需突破对应级别中枢上沿≥0.3%
        - 15分钟：向上笔需突破对应级别中枢上沿≥0.2%
        避免"1分钱假突破"误判
        
        Args:
            price: 当前价格（用于判断突破幅度）
            center_upper: 中枢上沿价格
            timeframe: 时间周期，可选 'daily', '30min', '15min'
        
        Returns:
            Tuple[bool, float]: (突破是否有效, 突破幅度百分比)
        """
        try:
            # 参数验证
            if price <= 0 or center_upper <= 0:
                raise ValueError("价格必须为正数")
            
            # 计算突破幅度百分比：(当前价格 - 中枢上沿) / 中枢上沿 * 100%
            breakout_amplitude = ((price - center_upper) / center_upper) * 100
            
            # 获取对应时间周期的阈值
            # 确保严格按照突破有效性阈值要求设置
            min_breakout = self.config['breakout_threshold'].get(timeframe, 0.5)
            
            # 明确记录各时间周期的突破阈值要求
            timeframe_breakout_info = {
                'daily': '日线突破，需≥0.5%以避免假突破',
                '30min': '30分钟突破，需≥0.3%以避免假突破',
                '15min': '15分钟突破，需≥0.2%以避免假突破'
            }
            
            # 判断突破是否有效
            is_effective = breakout_amplitude >= min_breakout
            
            # 详细日志记录
            breakout_desc = timeframe_breakout_info.get(timeframe, f'未知时间周期: {timeframe}')
            logger.info(f"突破有效性验证 - {breakout_desc}")
            logger.info(f"  当前价格: {price:.4f}, 中枢上沿: {center_upper:.4f}")
            logger.info(f"  突破幅度: {breakout_amplitude:.2f}%, 阈值: {min_breakout}%, 有效: {is_effective}")
            
            # 记录绝对突破金额（元），帮助理解实际突破力度
            absolute_breakout = price - center_upper
            logger.info(f"  绝对突破金额: {absolute_breakout:.4f}元")
            
            return is_effective, breakout_amplitude
            
        except ValueError as ve:
            logger.error(f"突破有效性验证参数错误: {str(ve)}")
            return False, 0.0
        except Exception as e:
            logger.error(f"突破有效性验证失败: {str(e)}")
            return False, 0.0
    
    def detect_breakout_type(self, price: float, center_lower: float, center_upper: float, 
                           timeframe: str = 'daily') -> Dict[str, Union[bool, str, float]]:
        """
        检测价格相对于中枢的突破类型
        
        Args:
            price: 当前价格
            center_lower: 中枢下沿价格
            center_upper: 中枢上沿价格
            timeframe: 时间周期
        
        Returns:
            Dict: 突破类型和详细信息
        """
        try:
            # 参数验证
            if price <= 0 or center_lower <= 0 or center_upper <= 0:
                raise ValueError("价格必须为正数")
            if center_lower > center_upper:
                raise ValueError("中枢下沿不能大于中枢上沿")
            
            # 计算相对于中枢边界的位置
            distance_from_lower = ((price - center_lower) / center_lower) * 100
            distance_from_upper = ((price - center_upper) / center_upper) * 100
            
            # 检测突破有效性
            upper_breakout_valid, upper_breakout_amplitude = self.validate_breakout_effectiveness(
                price, center_upper, timeframe
            )
            
            # 判断突破类型
            if upper_breakout_valid:
                breakout_type = "有效向上突破"
                confidence = min(100, max(0, upper_breakout_amplitude * 10))  # 简单的置信度计算
            elif price > center_upper:
                breakout_type = "假突破"
                confidence = 0
            elif price < center_lower:
                breakout_type = "向下突破"
                confidence = 0
            else:
                breakout_type = "中枢内震荡"
                confidence = 0
            
            return {
                'breakout_type': breakout_type,
                'confidence': confidence,
                'price': price,
                'center_lower': center_lower,
                'center_upper': center_upper,
                'distance_from_lower': distance_from_lower,
                'distance_from_upper': distance_from_upper,
                'upper_breakout_valid': upper_breakout_valid,
                'upper_breakout_amplitude': upper_breakout_amplitude
            }
            
        except Exception as e:
            logger.error(f"突破类型检测失败: {str(e)}")
            return {
                'breakout_type': '未知',
                'error': str(e),
                'confidence': 0
            }
    
    def determine_up_leg_validity(self, df: pd.DataFrame, up_leg_start: int, 
                                up_leg_end: int, timeframe: str = 'daily') -> Dict:
        """
        综合判定向上笔是否有效
        
        通用判定逻辑（所有ETF统一执行）：
        1. 先自动检测是否处于横盘环境（按近20日数据计算）
        2. 若为横盘：仅当向上笔同时满足"突破中枢上沿+对应级别涨幅阈值"，才判定为有效向上笔
        3. 若为趋势行情（近20日振幅＞15%且有明确趋势）：保留原向上笔判定规则，同时叠加"幅度≥对应阈值"过滤微小波动
        
        Args:
            df: 包含K线数据的DataFrame
            up_leg_start: 向上笔起始索引
            up_leg_end: 向上笔结束索引
            timeframe: 时间周期，可选 'daily', '30min', '15min'
        
        Returns:
            Dict: 包含判定结果的详细信息
        """
        try:
            logger.info(f"开始通用判定流程 - 向上笔[{up_leg_start}:{up_leg_end}], 时间周期: {timeframe}")
            
            # 1. 检测是否处于横盘环境
            logger.info("步骤1: 执行横盘环境检测")
            sideways_result = self.detect_sideways_market(df)
            is_sideways = sideways_result['is_sideways']
            
            # 2. 验证向上笔索引范围
            logger.info("步骤2: 验证向上笔索引范围")
            if up_leg_start < 0 or up_leg_end >= len(df) or up_leg_start > up_leg_end:
                raise ValueError(f"向上笔索引超出范围: start={up_leg_start}, end={up_leg_end}, length={len(df)}")
            
            # 3. 获取向上笔相关价格和信息
            logger.info("步骤3: 获取向上笔价格信息")
            start_price = df.iloc[up_leg_start]['close']
            end_price = df.iloc[up_leg_end]['close']
            
            # 4. 计算向上笔详细信息
            up_leg_info = self.calculate_up_leg_info(df, up_leg_start, up_leg_end, timeframe)
            
            # 5. 根据市场环境进行不同的判定
            logger.info(f"步骤4: 根据市场环境[{sideways_result['reason']}]进行判定")
            satisfied_conditions = []
            failed_conditions = []
            
            if is_sideways:
                # 横盘环境: 需同时满足突破中枢上沿和幅度阈值
                market_type = "横盘环境"
                center_range = sideways_result['center_range']
                center_upper = center_range['upper']
                
                logger.info(f"横盘环境判定: 中枢区间=[{center_range['lower']:.4f}, {center_upper:.4f}]")
                
                # 检查突破有效性
                breakout_valid, breakout_amplitude = self.validate_breakout_effectiveness(
                    end_price, center_upper, timeframe
                )
                
                # 检查幅度有效性
                amplitude_valid = up_leg_info['valid']
                
                # 记录条件满足情况
                if amplitude_valid:
                    satisfied_conditions.append(f"涨幅≥{up_leg_info['amplitude_threshold']}% ({up_leg_info['amplitude']:.2f}%)")
                else:
                    failed_conditions.append(f"涨幅<{up_leg_info['amplitude_threshold']}% ({up_leg_info['amplitude']:.2f}%)")
                
                if breakout_valid:
                    satisfied_conditions.append(f"有效突破中枢上沿≥{self.config['breakout_threshold'][timeframe]}% ({breakout_amplitude:.2f}%)")
                else:
                    failed_conditions.append(f"未有效突破中枢上沿 ({breakout_amplitude:.2f}%)")
                
                # 横盘环境下，两个条件必须同时满足
                is_valid = amplitude_valid and breakout_valid
                
                if is_valid:
                    judgment = "横盘突破有效向上笔"
                else:
                    judgment = "横盘震荡波"
                    
                # 检测突破类型
                breakout_type_info = self.detect_breakout_type(
                    end_price, center_range['lower'], center_upper, timeframe
                )
                
            else:
                # 趋势行情: 只需满足幅度阈值过滤微小波动
                market_type = "趋势行情"
                amplitude_valid = up_leg_info['valid']
                
                # 记录条件满足情况
                if amplitude_valid:
                    satisfied_conditions.append(f"涨幅≥{up_leg_info['amplitude_threshold']}% ({up_leg_info['amplitude']:.2f}%)")
                else:
                    failed_conditions.append(f"涨幅<{up_leg_info['amplitude_threshold']}% ({up_leg_info['amplitude']:.2f}%)")
                
                # 趋势行情下，只需满足幅度阈值
                is_valid = amplitude_valid
                
                if is_valid:
                    judgment = "趋势有效向上笔"
                else:
                    judgment = "趋势微小波动"
                
                breakout_type_info = None
            
            # 6. 构建详细的返回结果
            result = {
                'is_valid_up_leg': is_valid,
                'market_type': market_type,
                'judgment': judgment,
                'timeframe': timeframe,
                'up_leg_range': {'start': up_leg_start, 'end': up_leg_end},
                'price_info': {
                    'start_price': start_price,
                    'end_price': end_price,
                    'amplitude': up_leg_info['amplitude']
                },
                'sideways_detection': sideways_result,
                'up_leg_info': up_leg_info,
                'satisfied_conditions': satisfied_conditions,
                'failed_conditions': failed_conditions,
                'decision_process': {
                    'step1': '横盘环境检测完成',
                    'step2': f'市场环境判定为: {market_type}',
                    'step3': f'条件验证: 满足{len(satisfied_conditions)}个条件，失败{len(failed_conditions)}个条件',
                    'step4': f'最终判定: {judgment}'
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 如果是横盘环境，添加突破相关信息
            if is_sideways:
                result['breakout_info'] = {
                    'valid': breakout_valid,
                    'amplitude': breakout_amplitude,
                    'threshold': self.config['breakout_threshold'].get(timeframe, 0.5),
                    'center_range': sideways_result['center_range'],
                    'breakout_type': breakout_type_info
                }
            
            # 7. 详细日志记录整个判定流程
            logger.info(f"向上笔有效性判定结果: {is_valid}")
            logger.info(f"  判定类型: {judgment}")
            logger.info(f"  满足条件: {', '.join(satisfied_conditions) if satisfied_conditions else '无'}")
            logger.info(f"  失败条件: {', '.join(failed_conditions) if failed_conditions else '无'}")
            
            return result
            
        except ValueError as ve:
            logger.error(f"向上笔有效性判定参数错误: {str(ve)}")
            return {
                'is_valid_up_leg': False,
                'error': str(ve),
                'market_type': "未知",
                'judgment': "参数错误",
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            logger.error(f"向上笔有效性判定失败: {str(e)}")
            return {
                'is_valid_up_leg': False,
                'error': str(e),
                'market_type': "未知",
                'judgment': "无法判定",
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    def batch_determine_up_legs(self, df: pd.DataFrame, up_legs: List[Dict[str, int]], 
                              timeframe: str = 'daily') -> List[Dict]:
        """
        批量判定多个向上笔的有效性
        
        Args:
            df: 包含K线数据的DataFrame
            up_legs: 向上笔列表，每个元素为{'start': int, 'end': int}
            timeframe: 时间周期
        
        Returns:
            List[Dict]: 各向上笔的判定结果
        """
        results = []
        
        for i, up_leg in enumerate(up_legs):
            try:
                logger.info(f"批量处理第{i+1}/{len(up_legs)}个向上笔")
                result = self.determine_up_leg_validity(
                    df, up_leg['start'], up_leg['end'], timeframe
                )
                results.append({
                    'index': i,
                    'up_leg': up_leg,
                    'result': result
                })
            except Exception as e:
                logger.error(f"批量处理向上笔{i+1}失败: {str(e)}")
                results.append({
                    'index': i,
                    'up_leg': up_leg,
                    'result': {'error': str(e), 'is_valid_up_leg': False}
                })
        
        # 统计结果
        valid_count = sum(1 for r in results if r['result'].get('is_valid_up_leg', False))
        logger.info(f"批量判定完成: 有效向上笔{valid_count}/{len(up_legs)}个")
        
        return results
    
    def analyze_trend_for_etf(self, df: pd.DataFrame, timeframe: str = 'daily') -> Dict:
        """
        对ETF进行完整的趋势分析
        
        Args:
            df: 包含K线数据的DataFrame
            timeframe: 时间周期
        
        Returns:
            Dict: 完整的趋势分析结果
        """
        try:
            # 1. 检测市场环境
            market_analysis = self.detect_sideways_market(df)
            
            # 2. 获取当前关键价格
            latest_close = df.iloc[0]['close']
            prev_close = df.iloc[1]['close'] if len(df) > 1 else latest_close
            
            # 3. 计算短期趋势
            short_term_trend = "向上" if latest_close > prev_close else "向下"
            
            # 4. 构建分析结果
            analysis = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'timeframe': timeframe,
                'market_type': "横盘环境" if market_analysis.get('is_sideways', False) else "趋势行情",  # 确保有market_type字段
                'is_valid_up_leg': False,  # 默认值，根据实际情况可调整
                'is_valid': market_analysis.get('is_sideways', False),  # 保持向后兼容
                'market_environment': {
                    'is_sideways': market_analysis.get('is_sideways', False),
                    'amplitude': market_analysis.get('amplitude', 0.0),
                    'reason': market_analysis.get('reason', '')
                },
                'price_info': {
                    'latest_close': latest_close,
                    'prev_close': prev_close,
                    'short_term_trend': short_term_trend
                },
                'center_info': market_analysis.get('center_range') if market_analysis.get('is_sideways', False) else None,
                'recommendations': self._generate_recommendations(market_analysis, short_term_trend)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"ETF趋势分析失败: {str(e)}")
            return {
                'error': str(e),
                'analysis_status': 'failed',
                'market_type': '未知'  # 确保即使出错也返回market_type
            }
    
    def _generate_recommendations(self, market_analysis: Dict, short_term_trend: str) -> List[str]:
        """
        根据市场分析生成建议（辅助方法）
        
        Args:
            market_analysis: 市场分析结果
            short_term_trend: 短期趋势
        
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 使用安全的字典访问方式
        is_sideways = market_analysis.get('is_sideways', False)
        
        if is_sideways:
            recommendations.append("当前处于横盘环境，关注中枢上沿突破机会")
            if short_term_trend == "向上":
                recommendations.append("短期向上，若突破中枢上沿且满足幅度要求，可考虑跟进")
            else:
                recommendations.append("短期向下，等待中枢下沿支撑确认")
        else:
            recommendations.append("当前处于趋势行情，跟随趋势操作")
            recommendations.append(f"短期{short_term_trend}，关注幅度是否满足阈值要求")
        
        return recommendations
    
    def batch_analyze_etfs(self, etf_data_dict: Dict[str, pd.DataFrame], 
                          timeframe: str = 'daily') -> Dict[str, Dict]:
        """
        批量分析多个ETF
        
        Args:
            etf_data_dict: {etf_code: df}的字典
            timeframe: 时间周期
        
        Returns:
            Dict[str, Dict]: 各ETF的分析结果
        """
        results = {}
        
        for etf_code, df in etf_data_dict.items():
            logger.info(f"开始分析ETF: {etf_code}")
            try:
                results[etf_code] = self.analyze_trend_for_etf(df, timeframe)
            except Exception as e:
                logger.error(f"分析ETF {etf_code} 失败: {str(e)}")
                results[etf_code] = {'error': str(e), 'analysis_status': 'failed'}
        
        return results

# 主函数示例
if __name__ == "__main__":
    # 示例用法
    print("ETF通用横盘+向上笔判定规则模块")
    print("请在其他模块中导入使用，例如：")
    print("from src.etf_trend_detector import ETFTrendDetector")
    print("\n示例代码：")
    print("""
    # 创建检测器实例
    detector = ETFTrendDetector()
    
    # 假设df是您的K线数据
    # sideways_result = detector.detect_sideways_market(df)
    # up_leg_validity = detector.determine_up_leg_validity(df, start_index, end_index, 'daily')
    """)