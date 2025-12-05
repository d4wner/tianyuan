#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日线买点判定模块（适配全波动ETF）

该模块实现日线级别的买点判定，支持自动适配低/中/高波动ETF，包括：
1. 日线二买（核心，确定性最高）
2. 日线一买（辅助）
3. 日线三买（辅助）
4. 日线破中枢反抽（兜底逻辑）

作者: TradeTianYuan
日期: 2024-01-20
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum
from datetime import datetime, timedelta

# 设置日志
logger = logging.getLogger(__name__)


class VolatilityLevel(Enum):
    """ETF波动等级枚举"""
    LOW = "低波动"      # ≤10%
    MEDIUM = "中波动"   # 10%-18%
    HIGH = "高波动"      # ＞18%

class BuySignalType(Enum):
    """日线买点类型枚举"""
    SECOND_BUY = "日线二买"  # 核心买点
    FIRST_BUY = "日线一买"   # 辅助买点
    THIRD_BUY = "日线三买"    # 辅助买点
    REVERSE_PULLBACK = "破中枢反抽"  # 兜底买点
    NONE = "无买点"


class BuySignalDetector:
    """日线买点检测器类（适配全波动ETF）"""
    
    # 动态参数映射表
    DYNAMIC_PARAMS = {
        VolatilityLevel.LOW: {
            'central_bank_amplitude': 0.05,       # 中枢振幅要求 ≥5%
            'break_central_threshold': 0.995,     # 破中枢阈值 = 中枢下沿×0.995
            'pullback_threshold': 1.005,          # 反抽阈值 = 中枢下沿×1.005
            'consecutive_days': '2连续',           # 连续2日达标
            'pullback_window': 5,                 # 反抽时间窗口 5个交易日
            'volume_threshold': 0.8,              # 量能验证阈值 ≥近5日均量80%
            'min30_sub_position_second_buy': 0.70, # 30分钟子仓位比例（二买）上限70%
            'min30_sub_position_second_buy_min': 0.65, # 30分钟子仓位比例（二买）下限65%
            'min15_sub_position_other': 0.30,      # 15分钟子仓位比例（一买/三买）上限30%
            'min15_sub_position_other_min': 0.20,  # 15分钟子仓位比例（一买/三买）下限20%
            'breakthrough_threshold_3rd': 1.005    # 三买突破阈值 ×1.005
        },
        VolatilityLevel.MEDIUM: {
            'central_bank_amplitude': 0.08,       # 中枢振幅要求 ≥8%
            'break_central_threshold': 0.99,      # 破中枢阈值 = 中枢下沿×0.99
            'pullback_threshold': 1.01,           # 反抽阈值 = 中枢下沿×1.01
            'consecutive_days': '2日内≥1日临界',  # 2日内≥1日达标+1日临界（阈值±0.002）
            'pullback_window': 6,                 # 反抽时间窗口 6个交易日
            'volume_threshold': 0.85,             # 量能验证阈值 ≥近5日均量85%
            'min30_sub_position_second_buy': 0.65, # 30分钟子仓位比例（二买）上限65%
            'min30_sub_position_second_buy_min': 0.60, # 30分钟子仓位比例（二买）下限60%
            'min15_sub_position_other': 0.35,      # 15分钟子仓位比例（一买/三买）上限35%
            'min15_sub_position_other_min': 0.25,  # 15分钟子仓位比例（一买/三买）下限25%
            'breakthrough_threshold_3rd': 1.01     # 三买突破阈值 ×1.01
        },
        VolatilityLevel.HIGH: {
            'central_bank_amplitude': 0.10,       # 中枢振幅要求 ≥10%
            'break_central_threshold': 0.985,     # 破中枢阈值 = 中枢下沿×0.985
            'pullback_threshold': 1.015,          # 反抽阈值 = 中枢下沿×1.015
            'consecutive_days': '2日内≥1日临界',  # 2日内≥1日达标+1日临界（阈值±0.002）
            'pullback_window': 7,                 # 反抽时间窗口 7个交易日
            'volume_threshold': 0.90,             # 量能验证阈值 ≥近5日均量90%
            'min30_sub_position_second_buy': 0.60, # 30分钟子仓位比例（二买）上限60%
            'min30_sub_position_second_buy_min': 0.55, # 30分钟子仓位比例（二买）下限55%
            'min15_sub_position_other': 0.40,      # 15分钟子仓位比例（一买/三买）上限40%
            'min15_sub_position_other_min': 0.30,  # 15分钟子仓位比例（一买/三买）下限30%
            'breakthrough_threshold_3rd': 1.015    # 三买突破阈值 ×1.015
        }
    }
    
    def __init__(self):
        # 初始化时添加logger
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("日线买点检测器初始化")
        # 初始化波动等级和参数
        self.volatility_level = None
        self.dynamic_params = None
        self.volatility_value = None  # 存储计算的波动率值
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 60) -> float:
        """计算ETF波动率
        
        Args:
            df: 日线数据框
            period: 计算周期，默认60天
            
        Returns:
            波动率百分比
        """
        if len(df) < period:
            self.logger.warning(f"数据不足{period}天，使用所有可用数据计算波动率")
            period = len(df)
        
        recent_data = df.tail(period)
        highest_high = recent_data['high'].max()
        lowest_low = recent_data['low'].min()
        avg_price = recent_data['close'].mean()
        
        volatility = ((highest_high - lowest_low) / avg_price) * 100
        self.logger.info(f"计算波动率: {(highest_high - lowest_low):.4f}/{avg_price:.4f} = {volatility:.2f}%")
        return volatility
    
    def determine_volatility_level(self, volatility: float) -> VolatilityLevel:
        """根据波动率确定ETF波动等级
        
        Args:
            volatility: 波动率百分比
            
        Returns:
            波动等级枚举值
        """
        if volatility <= 10.0:
            return VolatilityLevel.LOW
        elif volatility <= 18.0:
            return VolatilityLevel.MEDIUM
        else:
            return VolatilityLevel.HIGH
    
    def adapt_to_volatility(self, df: pd.DataFrame) -> Dict:
        """适配不同波动等级的ETF，设置动态参数
        
        Args:
            df: 日线数据框
            
        Returns:
            包含波动等级和动态参数的字典
        """
        # 计算波动率
        volatility = self.calculate_volatility(df)
        self.volatility_value = volatility
        
        # 确定波动等级
        level = self.determine_volatility_level(volatility)
        self.volatility_level = level
        
        # 设置动态参数
        self.dynamic_params = self.DYNAMIC_PARAMS[level]
        
        self.logger.info(f"ETF波动等级: {level.value} ({volatility:.2f}%)")
        self.logger.info(f"动态参数设置: {self.dynamic_params}")
        
        return {
            'volatility_level': level.value,
            'volatility_value': volatility,
            'dynamic_params': self.dynamic_params
        }
    
    def _process_kline_inclusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理K线包含关系（简化版）
        
        Args:
            df: 原始日线数据框
            
        Returns:
            处理包含关系后的K线数据框
        """
        df = df.copy()
        if len(df) < 3:
            return df  # 数据不足，无需处理
            
        processed = [df.iloc[0].copy(), df.iloc[1].copy()]
        
        for i in range(2, len(df)):
            current = df.iloc[i].copy()
            last = processed[-1]
            prev = processed[-2]
            
            # 简化的包含关系判断：只检查是否完全包含或被包含
            has_inclusion = (current['high'] <= last['high'] and current['low'] >= last['low']) or \
                           (current['high'] >= last['high'] and current['low'] <= last['low'])
            
            if not has_inclusion:
                # 无包含关系，直接添加
                processed.append(current)
                continue
            
            # 判断趋势（基于前两根K线）
            is_up = last['high'] > prev['high']  # 上涨趋势
            is_down = last['low'] < prev['low']  # 下跌趋势
            
            # 处理包含关系
            if is_up:
                # 上涨趋势：取高高
                new_k = last.copy()
                new_k['high'] = max(last['high'], current['high'])
                new_k['low'] = max(last['low'], current['low'])
                new_k['volume'] += current['volume']
            elif is_down:
                # 下跌趋势：取低低
                new_k = last.copy()
                new_k['high'] = min(last['high'], current['high'])
                new_k['low'] = min(last['low'], current['low'])
                new_k['volume'] += current['volume']
            else:
                # 横盘：取高低点范围
                new_k = last.copy()
                new_k['high'] = max(last['high'], current['high'])
                new_k['low'] = min(last['low'], current['low'])
                new_k['volume'] += current['volume']
            
            # 替换最后一根K线
            processed.pop()
            processed.append(new_k)
        
        # 转换回DataFrame
        result_df = pd.DataFrame(processed)
        
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result_df.columns and col in df.columns:
                result_df[col] = df[col].iloc[:len(result_df)]
        
        # 重新索引
        result_df.index = df.iloc[:len(result_df)].index  # 保持索引一致
        
        self.logger.info(f"K线包含处理前: {len(df)}根，处理后: {len(result_df)}根")
        return result_df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 数据框
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            添加了MACD指标的数据框
        """
        df = df.copy()
        
        # 计算EMA
        ema_fast = df['close'].ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False, min_periods=slow).mean()
        
        # 计算DIF线
        df['macd_diff'] = ema_fast - ema_slow
        
        # 计算DEA线（信号线）
        df['macd_dea'] = df['macd_diff'].ewm(span=signal, adjust=False, min_periods=signal).mean()
        
        # 计算MACD柱状图
        df['macd_hist'] = df['macd_diff'] - df['macd_dea']
        
        return df
    
    def _identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别顶底分型
        
        Args:
            df: 数据框
            
        Returns:
            添加了分型标识的数据框
        """
        df = df.copy()
        df['bottom_fractal'] = False
        df['top_fractal'] = False
        
        # 使用5根K线识别分型
        for i in range(2, len(df) - 2):
            # 底分型: 中间低，两侧高
            if (df.iloc[i]['low'] < df.iloc[i-2]['low'] and 
                df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                df.loc[df.index[i], 'bottom_fractal'] = True
            
            # 顶分型: 中间高，两侧低
            if (df.iloc[i]['high'] > df.iloc[i-2]['high'] and 
                df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                df.loc[df.index[i], 'top_fractal'] = True
        
        return df
    
    def detect_daily_first_buy(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """检测日线一买信号（支持全波动ETF适配）
        
        判定规则：
        ① 下跌段界定：日线完整下跌段≥动态参数K线数量，为独立下跌笔
        ② 核心背驰：价格创该下跌段内新低，但MACD黄白线未创该下跌段内新低
        ③ 辅助验证：当前绿柱最大高度＜前一个同级下跌段绿柱最大高度的动态阈值
        ④ 排除伪背驰：MACD黄白线处于零轴下方
        
        Args:
            df: 日线数据框
            
        Returns:
            (是否存在一买信号, 详细信息)
        """
        self.logger.info("检测日线一买信号...")
        
        # 数据源有效性校验
        if len(df) < 60:  # 需要至少60根K线进行分析
            self.logger.warning("日线数据不足60根，无法检测一买信号")
            return False, {
                "reason": "数据不足",
                "detail": f"需要至少60根日线K线，当前只有{len(df)}根",
                "data_source": "不足"
            }
        
        # 应用K线包含处理
        processed_df = self._process_kline_inclusion(df)
        
        # 计算MACD指标
        df_with_macd = self._calculate_macd(processed_df)
        
        # 获取动态参数
        min_downtrend_k_count = self.dynamic_params.get("min_downtrend_k_count", 10)
        green_hist_ratio_threshold = self.dynamic_params.get("green_hist_ratio_threshold", 0.7)
        price_divergence_threshold = self.dynamic_params.get("price_divergence_threshold", 1.03)
        green_trend_reduction_threshold = self.dynamic_params.get("green_trend_reduction_threshold", 0.8)
        
        # 1. 识别下跌段
        downtrend_segments = []
        current_segment = None
        consecutive_red_days = 0
        consecutive_green_days = 0
        
        for i in range(1, len(df_with_macd)):
            # 判断当日涨跌
            is_red = df_with_macd.iloc[i]['close'] < df_with_macd.iloc[i-1]['close']
            
            # 开始新的下跌段
            if is_red and current_segment is None:
                current_segment = {
                    'start_index': i,
                    'start_date': df_with_macd.iloc[i].get('date', f"第{i}天"),
                    'min_price': df_with_macd.iloc[i]['low'],
                    'min_price_date': df_with_macd.iloc[i].get('date', f"第{i}天"),
                    'min_diff': df_with_macd.iloc[i]['macd_diff'],
                    'min_diff_date': df_with_macd.iloc[i].get('date', f"第{i}天"),
                    'max_green_hist': 0,
                    'green_red_green_cycles': 0,
                    'has_consecutive_green_3': False,
                    'segment_length': 1
                }
                consecutive_red_days = 1
                consecutive_green_days = 0
            
            # 更新当前下跌段
            elif current_segment is not None:
                current_segment['segment_length'] += 1
                
                # 更新最低价和对应MACD值
                if df_with_macd.iloc[i]['low'] < current_segment['min_price']:
                    current_segment['min_price'] = df_with_macd.iloc[i]['low']
                    current_segment['min_price_date'] = df_with_macd.iloc[i].get('date', f"第{i}天")
                    current_segment['min_diff'] = df_with_macd.iloc[i]['macd_diff']
                    current_segment['min_diff_date'] = df_with_macd.iloc[i].get('date', f"第{i}天")
                
                # 更新最大绿柱高度
                if df_with_macd.iloc[i]['macd_hist'] < 0:  # 绿柱
                    current_segment['max_green_hist'] = min(current_segment['max_green_hist'], df_with_macd.iloc[i]['macd_hist'])
                
                # 统计连续红绿柱情况
                if is_red:
                    consecutive_red_days += 1
                    consecutive_green_days = 0
                else:
                    consecutive_green_days += 1
                    consecutive_red_days = 0
                    
                    # 检查是否有连续3根阳线
                    if consecutive_green_days >= 3:
                        current_segment['has_consecutive_green_3'] = True
                
                # 检测绿柱→红柱→绿柱的循环
                if i >= 2:
                    prev_prev_hist = df_with_macd.iloc[i-2]['macd_hist']
                    prev_hist = df_with_macd.iloc[i-1]['macd_hist']
                    curr_hist = df_with_macd.iloc[i]['macd_hist']
                    
                    # 绿→红→绿
                    if prev_prev_hist < 0 and prev_hist > 0 and curr_hist < 0:
                        current_segment['green_red_green_cycles'] += 1
                
                # 下跌段结束条件：连续3根阳线或MACD由负转正且持续2天
                if (consecutive_green_days >= 3 or 
                    (df_with_macd.iloc[i]['macd_diff'] > 0 and df_with_macd.iloc[i-1]['macd_diff'] > 0)):
                    
                    # 只有符合条件的下跌段才加入列表
                    if (current_segment['segment_length'] >= min_downtrend_k_count and 
                        not current_segment['has_consecutive_green_3']):
                        downtrend_segments.append(current_segment)
                    
                    current_segment = None
        
        # 如果最后一个下跌段未结束，也加入列表（如果符合条件）
        if (current_segment is not None and 
            current_segment['segment_length'] >= min_downtrend_k_count and 
            not current_segment['has_consecutive_green_3']):
            downtrend_segments.append(current_segment)
        
        self.logger.info(f"识别到{len(downtrend_segments)}个符合条件的下跌段")
        
        # 需要至少两个下跌段进行背驰比较
        if len(downtrend_segments) < 2:
            return False, {
                "reason": "下跌段不足",
                "detail": f"需要至少2个符合条件的下跌段，当前只有{len(downtrend_segments)}个",
                "data_source": "满足"
            }
        
        # 获取最近两个下跌段
        latest_segment = downtrend_segments[-1]
        previous_segment = downtrend_segments[-2]
        
        # 2. 检查核心背驰条件 - 使用动态价格新低阈值
        # 最新下跌段价格创新低，但MACD DIFF未创新低
        price_lower = latest_segment['min_price'] < previous_segment['min_price'] * price_divergence_threshold
        diff_higher = latest_segment['min_diff'] > previous_segment['min_diff']
        is_divergence = price_lower and diff_higher
        
        # 3. 检查辅助验证条件 - 使用动态绿柱阈值
        # 当前绿柱最大高度 < 前一个下跌段绿柱最大高度的动态阈值
        green_hist_condition = False
        if previous_segment['max_green_hist'] != 0:
            green_hist_condition = latest_segment['max_green_hist'] > previous_segment['max_green_hist'] * green_hist_ratio_threshold
        
        # 增加绿柱趋势检查：检查最近绿柱是否有缩小趋势
        green_trend_condition = False
        if len(df_with_macd) > latest_segment['start_index'] + 5:
            # 获取最近的MACD柱状图数据
            recent_hist = df_with_macd.iloc[latest_segment['start_index']:]['macd_hist'].values
            # 检查最后5个绿柱是否有明显缩小趋势
            green_hist = [h for h in recent_hist[-20:] if h < 0]  # 最近20个K线的绿柱
            if len(green_hist) >= 5:
                # 计算前半段和后半段的平均绿柱高度
                mid_idx = len(green_hist) // 2
                first_half_avg = abs(sum(green_hist[:mid_idx]) / mid_idx)
                second_half_avg = abs(sum(green_hist[mid_idx:]) / (len(green_hist) - mid_idx))
                # 如果后半段比前半段缩小至少动态阈值，视为绿柱有缩小趋势
                if second_half_avg < first_half_avg * green_trend_reduction_threshold:
                    green_trend_condition = True
        
        # 4. 检查伪背驰条件
        # MACD黄白线处于零轴下方
        current_diff = df_with_macd.iloc[-1]['macd_diff']
        current_dea = df_with_macd.iloc[-1]['macd_dea']
        below_zero_condition = current_diff < 0 and current_dea < 0
        
        # 伪背驰过滤：检查下跌段内是否只有连续绿柱，无红柱穿插
        segment_data = df_with_macd.iloc[latest_segment['start_index']:latest_segment['start_index'] + latest_segment['segment_length']]
        has_red_bars = len(segment_data[segment_data['macd_hist'] > 0]) > 0
        
        # 综合判定 - 使用动态条件：只要满足核心背驰和绿柱条件，就判定为一买
        # 绿柱条件满足任一即可：高度条件或趋势条件
        green_condition = green_hist_condition or green_trend_condition
        is_first_buy = (is_divergence and 
                       green_condition and 
                       below_zero_condition and
                       has_red_bars)  # 添加伪背驰过滤
        
        details = {
            "latest_segment": {
                "length": latest_segment['segment_length'],
                "min_price": latest_segment['min_price'],
                "min_price_date": latest_segment['min_price_date'],
                "min_diff": latest_segment['min_diff'],
                "max_green_hist": latest_segment['max_green_hist'],
                "cycles": latest_segment['green_red_green_cycles']
            },
            "previous_segment": {
                "length": previous_segment['segment_length'],
                "min_price": previous_segment['min_price'],
                "min_diff": previous_segment['min_diff'],
                "max_green_hist": previous_segment['max_green_hist']
            },
            "conditions": {
                "divergence": is_divergence,
                "green_hist": green_hist_condition,
                "green_trend": green_trend_condition,
                "below_zero": below_zero_condition,
                "has_red_bars": has_red_bars  # 添加伪背驰检查结果
            },
            "current_macd": {
                "diff": current_diff,
                "dea": current_dea
            },
            "volatility_level": self.volatility_level.value if self.volatility_level else "未设置",
            "data_source": "满足"
        }
        
        self.logger.info(f"日线一买信号检测结果: {'满足' if is_first_buy else '不满足'}")
        self.logger.info(f"核心背驰: {'满足' if is_divergence else '不满足'}")
        self.logger.info(f"绿柱条件: {'满足' if green_hist_condition else '不满足'}")
        self.logger.info(f"绿柱趋势: {'满足' if green_trend_condition else '不满足'}")
        self.logger.info(f"零轴下方: {'满足' if below_zero_condition else '不满足'}")
        self.logger.info(f"伪背驰过滤: {'满足' if has_red_bars else '不满足'}")
        
        return is_first_buy, details
    
    def detect_daily_second_buy(self, df: pd.DataFrame, first_buy_confirmed: bool = False, min30_df: pd.DataFrame = None) -> Tuple[bool, Dict]:
        """检测日线二买信号（支持全波动ETF适配）
        
        判定规则：
        ① 日线一买已确认（满足一买背驰规则）
        ② 一买后回调未创阶段新低，且未跌破一买对应的中枢下沿
        ③ 出现无包含底分型，底分型确认日成交量≥前5个交易日日均成交量×动态阈值
        ④ 回调过程中30分钟级别形成底背驰：30分钟下跌段价格新低+MACD黄白线不新低+绿柱最大高度＜前一段动态阈值
        
        Args:
            df: 日线数据框
            first_buy_confirmed: 是否已确认一买信号
            min30_df: 30分钟数据框（可选）
            
        Returns:
            (是否存在二买信号, 详细信息)
        """
        self.logger.info("检测日线二买信号...")
        
        # 数据源有效性校验
        if len(df) < 60:
            self.logger.warning("日线数据不足60根，无法检测二买信号")
            return False, {
                "reason": "数据不足",
                "detail": f"需要至少60根日线K线，当前只有{len(df)}根",
                "data_source": "不足"
            }
        
        # 应用K线包含处理
        processed_df = self._process_kline_inclusion(df)
        
        # 获取动态参数
        volume_threshold = self.dynamic_params.get("volume_threshold", 1.3)
        volume_relaxed_threshold = self.dynamic_params.get("volume_relaxed_threshold", 1.1)
        price_drop_threshold = self.dynamic_params.get("price_drop_threshold", 0.98)
        green_hist_ratio_threshold = self.dynamic_params.get("green_hist_ratio_threshold", 0.5)
        
        # 1. 确认一买信号
        first_buy = False
        first_buy_details = {}
        if not first_buy_confirmed:
            first_buy, first_buy_details = self.detect_daily_first_buy(processed_df)
            if not first_buy:
                return False, {
                    "reason": "一买未确认",
                    "detail": "日线二买需要先确认日线一买",
                    "data_source": "满足"
                }
        
        # 计算MACD指标和识别分型
        df_with_macd = self._calculate_macd(processed_df)
        
        # 2. 寻找最近的一买位置
        # 从一买信号详情获取一买位置，或基于最近低点定位
        first_buy_price = None
        first_buy_date = None
        recent_low_idx = None
        
        if first_buy_details and 'latest_segment' in first_buy_details:
            # 从一买信号详情中获取信息
            latest_segment = first_buy_details['latest_segment']
            first_buy_price = latest_segment.get('min_price')
            first_buy_date = latest_segment.get('min_price_date')
            # 尝试根据价格查找索引
            if first_buy_price:
                price_diff = abs(processed_df['low'] - first_buy_price)
                recent_low_idx = price_diff.idxmin()
        
        # 如果没有从一买详情获取到信息，使用简化方法
        if recent_low_idx is None:
            # 假设一买在最近的低点附近
            recent_low_idx = df_with_macd['low'].tail(30).idxmin()
            recent_low = df_with_macd.loc[recent_low_idx]
            first_buy_price = recent_low['low']
            first_buy_date = recent_low.get('date')
        
        # 3. 识别后续回调
        after_first_buy = df_with_macd[df_with_macd.index > recent_low_idx]
        if len(after_first_buy) < 10:  # 需要至少10根K线的回调
            return False, {
                "reason": "回调不充分",
                "detail": "一买后回调K线数量不足",
                "data_source": "满足"
            }
        
        # 4. 检查回调是否未创阶段新低 - 使用动态阈值
        callback_low = after_first_buy['low'].min()
        not_new_low = callback_low > first_buy_price * price_drop_threshold
        
        # 5. 识别一买对应的中枢下沿（简化处理）
        support_level = None
        if recent_low_idx > 10:
            # 尝试识别一买前的中枢
            before_first_buy_df = df_with_macd.iloc[max(0, recent_low_idx-50):recent_low_idx]
            support_level = before_first_buy_df['low'].min()
            
            # 检查回调是否跌破中枢下沿
            if callback_low < support_level * 0.99:
                self.logger.info(f"回调跌破中枢下沿，不满足二买条件")
                return False, {
                    "reason": "跌破中枢下沿",
                    "detail": f"回调最低价低于中枢下沿{support_level}的99%",
                    "data_source": "满足"
                }
        
        # 6. 检查是否出现无包含底分型
        has_bottom_fractal = False
        bottom_fractal_idx = -1
        bottom_fractal_price = None
        bottom_fractal_volume = 0
        
        # 使用分型识别函数，应用在处理过包含关系的K线上
        fractals = self._identify_fractals(processed_df)
        bottom_fractals = fractals.get('bottom', [])
        
        # 检查最近是否有底分型
        for i in range(len(bottom_fractals) - 1, -1, -1):
            fractal_idx = bottom_fractals[i]
            if fractal_idx > recent_low_idx:
                has_bottom_fractal = True
                bottom_fractal_idx = fractal_idx
                bottom_fractal_price = processed_df.iloc[fractal_idx]['close']
                bottom_fractal_volume = processed_df.iloc[fractal_idx]['volume']
                break
        
        if not has_bottom_fractal:
            self.logger.info("未检测到一买后的有效底分型，不满足二买条件")
            return False, {
                "reason": "无底分型",
                "detail": "一买后未出现有效的底分型",
                "data_source": "满足"
            }
        
        # 7. 检查成交量条件 - 使用动态阈值
        volume_condition = False
        
        # 获取前5个交易日的平均成交量
        if bottom_fractal_idx >= 5:
            prev_5_days = processed_df.iloc[max(0, bottom_fractal_idx-5):bottom_fractal_idx]
            if len(prev_5_days) >= 3:
                avg_volume = prev_5_days['volume'].mean()
                
                # 使用动态成交量阈值
                volume_condition = bottom_fractal_volume >= avg_volume * volume_threshold
                
                # 如果不满足，尝试使用放宽的阈值
                if not volume_condition and bottom_fractal_volume >= avg_volume * volume_relaxed_threshold:
                    volume_condition = True
                    self.logger.info(f"使用放宽的成交量阈值{volume_relaxed_threshold}倍")
        
        # 8. 30分钟级别底背驰验证
        macd_divergence = False
        min30_details = {"available": False}
        
        # 如果提供了30分钟数据，进行详细的底背驰验证
        if min30_df is not None and len(min30_df) >= 100:
            min30_details["available"] = True
            
            # 应用30分钟K线包含处理
            processed_min30_df = self._process_kline_inclusion(min30_df)
            
            # 计算30分钟MACD
            min30_with_macd = self._calculate_macd(processed_min30_df)
            
            # 识别最近的下跌段
            downtrend_segments = []
            current_segment = None
            
            for i in range(1, len(min30_with_macd)):
                is_red = min30_with_macd.iloc[i]['close'] < min30_with_macd.iloc[i-1]['close']
                
                if is_red and current_segment is None:
                    current_segment = {
                        'start_index': i,
                        'min_price': min30_with_macd.iloc[i]['low'],
                        'min_diff': min30_with_macd.iloc[i]['macd_diff'],
                        'max_green_hist': 0,
                        'length': 1
                    }
                elif current_segment is not None:
                    current_segment['length'] += 1
                    
                    if min30_with_macd.iloc[i]['low'] < current_segment['min_price']:
                        current_segment['min_price'] = min30_with_macd.iloc[i]['low']
                        current_segment['min_diff'] = min30_with_macd.iloc[i]['macd_diff']
                    
                    if min30_with_macd.iloc[i]['macd_hist'] < 0:
                        current_segment['max_green_hist'] = min(current_segment['max_green_hist'], min30_with_macd.iloc[i]['macd_hist'])
                    
                    # 下跌段结束条件：连续3根阳线
                    if not is_red and i >= 2 and not (min30_with_macd.iloc[i-1]['close'] < min30_with_macd.iloc[i-2]['close']):
                        if current_segment['length'] >= 5:  # 至少5根K线的下跌段
                            downtrend_segments.append(current_segment)
                        current_segment = None
            
            # 检查是否有足够的下跌段进行背驰比较
            if len(downtrend_segments) >= 2:
                latest_segment = downtrend_segments[-1]
                previous_segment = downtrend_segments[-2]
                
                # 价格新低
                price_lower = latest_segment['min_price'] < previous_segment['min_price']
                # MACD黄白线不新低
                diff_higher = latest_segment['min_diff'] > previous_segment['min_diff']
                # 绿柱缩小
                hist_cond = False
                if previous_segment['max_green_hist'] != 0:
                    hist_cond = latest_segment['max_green_hist'] > previous_segment['max_green_hist'] * green_hist_ratio_threshold
                
                macd_divergence = price_lower and diff_higher and hist_cond
                min30_details["divergence"] = macd_divergence
                min30_details["price_lower"] = price_lower
                min30_details["diff_higher"] = diff_higher
                min30_details["hist_cond"] = hist_cond
        else:
            # 如果没有30分钟数据，使用日线数据简化验证
            self.logger.info("未提供30分钟数据，使用日线MACD简化验证")
            recent_macd = df_with_macd.iloc[-20:].copy()
            
            recent_green_hist = recent_macd[recent_macd['macd_hist'] < 0]['macd_hist']
            
            if len(recent_green_hist) >= 5:
                mid_idx = len(recent_green_hist) // 2
                first_half_avg = abs(sum(recent_green_hist[:mid_idx]) / mid_idx)
                second_half_avg = abs(sum(recent_green_hist[mid_idx:]) / (len(recent_green_hist) - mid_idx))
                
                # 使用动态绿柱阈值
                if second_half_avg < first_half_avg * green_hist_ratio_threshold:
                    macd_divergence = True
        
        # 综合判定
        is_second_buy = (not_new_low and 
                        has_bottom_fractal and 
                        volume_condition and 
                        macd_divergence)
        
        details = {
            "first_buy": {
                "price": first_buy_price,
                "date": first_buy_date
            },
            "callback": {
                "low": callback_low,
                "not_new_low": not_new_low,
                "support_level": support_level
            },
            "fractal": {
                "has_bottom_fractal": has_bottom_fractal,
                "volume_condition": volume_condition,
                "volume": bottom_fractal_volume,
                "price": bottom_fractal_price
            },
            "macd": {
                "divergence": macd_divergence
            },
            "min30_details": min30_details,
            "volatility_level": self.volatility_level.value if self.volatility_level else "未设置",
            "data_source": "满足",
            "dynamic_params": {
                "volume_threshold": volume_threshold,
                "price_drop_threshold": price_drop_threshold,
                "green_hist_ratio_threshold": green_hist_ratio_threshold
            }
        }
        
        self.logger.info(f"日线二买信号检测结果: {'满足' if is_second_buy else '不满足'}")
        self.logger.info(f"底分型: {'满足' if has_bottom_fractal else '不满足'}")
        self.logger.info(f"成交量: {'满足' if volume_condition else '不满足'}")
        self.logger.info(f"MACD背驰: {'满足' if macd_divergence else '不满足'}")
        self.logger.info(f"波动等级: {self.volatility_level.value if self.volatility_level else '未设置'}")
        
        logger.info(f"日线二买信号检测结果: {'满足' if is_second_buy else '不满足'}")
        logger.info(f"未创新低: {'满足' if not_new_low else '不满足'}")
        logger.info(f"底分型: {'满足' if has_bottom_fractal else '不满足'}")
        logger.info(f"成交量: {'满足' if volume_condition else '不满足'}")
        logger.info(f"30分钟背驰: {'满足' if min30_divergence else '不满足'}")
        
        return is_second_buy, details
    
    def detect_daily_third_buy(self, df: pd.DataFrame, min30_df: pd.DataFrame = None) -> Tuple[bool, Dict]:
        """检测日线三买信号（支持全波动ETF适配）
        
        判定规则（必须同时满足，缺一不可）：
        1. 中枢突破有效性（核心前提）：
           - 价格需连续2日收盘价≥中枢上沿×动态突破阈值（根据波动等级自动适配）；
           - 突破时的成交量需≥近5日均量的动态阈值（确认突破量能）；
        2. 回抽有效性：
           突破后回抽的最低收盘价≥中枢上沿（回抽不进中枢）；
        3. 背驰验证：
           回抽过程中30分钟级别形成底背驰（价格新低+MACD黄白线不新低+绿柱高度＜前一段动态阈值）；
        4. 中枢内信号排除：
           若价格处于"中枢下沿-中枢上沿"区间内，直接排除三买信号，判定为"横盘震荡波"。
        
        Args:
            df: 日线数据框
            min30_df: 30分钟数据框（可选）
            
        Returns:
            (是否存在三买信号, 详细信息)
        """
        self.logger.info("检测日线三买信号...")
        
        # 数据源有效性校验
        if len(df) < 60:
            self.logger.warning("日线数据不足60根，无法检测三买信号")
            return False, {
                "reason": "数据不足",
                "detail": f"需要至少60根日线K线，当前只有{len(df)}根",
                "data_source": "不足"
            }
        
        # 应用K线包含处理
        processed_df = self._process_kline_inclusion(df)
        
        # 获取动态参数
        central_bank_amplitude_threshold = self.dynamic_params.get("central_bank_amplitude_threshold", 0.08)  # 中枢振幅阈值
        breakthrough_threshold_multiplier = self.dynamic_params.get("breakthrough_threshold_multiplier", 1.01)  # 突破阈值倍数
        volume_threshold = self.dynamic_params.get("volume_threshold", 1.2)  # 成交量阈值
        green_hist_ratio_threshold = self.dynamic_params.get("green_hist_ratio_threshold", 0.5)  # 绿柱比例阈值
        
        # 1. 识别中枢
        # 简化实现：使用分型识别中枢
        df_with_macd = self._calculate_macd(processed_df)
        
        # 2. 识别可能的中枢
        # 使用多窗口方法寻找中枢
        recent_data = df_with_macd.tail(60)  # 扩大范围提高识别率
        
        # 使用分型识别函数
        fractals = self._identify_fractals(processed_df)
        top_fractals = fractals.get('top', [])
        bottom_fractals = fractals.get('bottom', [])
        
        # 转换为处理过的df索引
        filtered_top_indices = [idx for idx in top_fractals if idx in recent_data.index]
        filtered_bottom_indices = [idx for idx in bottom_fractals if idx in recent_data.index]
        
        if len(filtered_top_indices) < 2 or len(filtered_bottom_indices) < 2:
            return False, {
                "reason": "中枢未形成",
                "detail": "未找到足够的顶底分型来确定中枢",
                "data_source": "满足"
            }
        
        # 获取最近的顶底分型
        recent_top_indices = sorted(filtered_top_indices)[-3:]
        recent_bottom_indices = sorted(filtered_bottom_indices)[-3:]
        
        # 计算中枢上沿和下沿
        central_bank_top = processed_df.iloc[recent_top_indices]['high'].min()  # 中枢上沿
        central_bank_bottom = processed_df.iloc[recent_bottom_indices]['low'].max()  # 中枢下沿
        
        # 计算中枢振幅并应用动态阈值
        central_bank_amplitude = (central_bank_top - central_bank_bottom) / central_bank_bottom
        
        # 检查是否形成合理中枢（根据波动等级使用动态阈值）
        if central_bank_amplitude < central_bank_amplitude_threshold:
            return False, {
                "reason": "中枢不合理",
                "detail": f"中枢振幅{central_bank_amplitude*100:.2f}%小于阈值{central_bank_amplitude_threshold*100:.2f}%，可能不是有效中枢",
                "data_source": "满足"
            }
        
        # 3. 中枢内信号排除：检查当前价格是否处于中枢区间内
        current_price = processed_df.iloc[-1]['close']
        in_central_range = central_bank_bottom <= current_price <= central_bank_top
        
        if in_central_range:
            return False, {
                "reason": "中枢内信号排除",
                "detail": f"当前价格{current_price}处于中枢区间[{central_bank_bottom}, {central_bank_top}]内，判定为横盘震荡波",
                "data_source": "满足"
            }
        
        # 4. 检查突破中枢上沿（使用动态阈值）
        latest_top_idx = max(recent_top_indices)
        after_central = df_with_macd[df_with_macd.index > latest_top_idx]
        
        if len(after_central) < 10:
            return False, {
                "reason": "突破不充分",
                "detail": "中枢形成后K线数量不足",
                "data_source": "满足"
            }
        
        # 计算动态突破阈值
        breakthrough_threshold = central_bank_top * breakthrough_threshold_multiplier
        
        # 检查是否有连续2天收盘价站稳突破阈值，且成交量满足条件
        consecutive_above = 0
        breakthrough_date = None
        has_volume_condition = False
        
        # 支持不同的连续达标要求
        consecutive_days_requirement = self.dynamic_params.get("consecutive_days_requirement", 2)
        
        # 对于中高波动ETF，支持2日内≥1日达标+1日临界
        if self.volatility_level in [VolatilityLevel.MEDIUM, VolatilityLevel.HIGH]:
            # 使用临界阈值进行判断
            critical_threshold = breakthrough_threshold * 0.998  # 临界值为阈值的99.8%
            days_above_threshold = 0
            days_above_critical = 0
            
            for i in range(5, len(after_central)):
                # 计算近5日均量
                recent_volumes = after_central.iloc[i-5:i]['volume']
                if len(recent_volumes) < 3:
                    continue
                    
                avg_volume_5d = recent_volumes.mean()
                
                # 检查价格条件
                close_price = after_central.iloc[i]['close']
                if close_price >= breakthrough_threshold:
                    days_above_threshold += 1
                elif close_price >= critical_threshold:
                    days_above_critical += 1
                
                # 检查成交量条件
                volume_condition_met = after_central.iloc[i]['volume'] >= avg_volume_5d * volume_threshold
                
                # 判断是否满足条件：≥1日达标+1日临界
                if days_above_threshold >= 1 and (days_above_threshold + days_above_critical) >= 2:
                    if volume_condition_met:
                        has_volume_condition = True
                        breakthrough_date = after_central.iloc[i].get('date')
                        break
                
                # 重置计数（如果连续两根K线都不满足临界条件）
                if i > 1 and after_central.iloc[i]['close'] < critical_threshold and after_central.iloc[i-1]['close'] < critical_threshold:
                    days_above_threshold = 0
                    days_above_critical = 0
            
            has_breakthrough = days_above_threshold >= 1 and (days_above_threshold + days_above_critical) >= 2 and has_volume_condition
        else:
            # 低波动ETF：严格要求连续达标
            for i in range(5, len(after_central)):  # 从第5根开始检查，确保有足够的成交量历史数据
                # 计算近5日均量
                recent_volumes = after_central.iloc[i-5:i]['volume']
                if len(recent_volumes) < 3:
                    continue
                    
                avg_volume_5d = recent_volumes.mean()
                
                # 检查价格突破阈值
                if after_central.iloc[i]['close'] >= breakthrough_threshold:
                    consecutive_above += 1
                    # 检查成交量条件
                    if after_central.iloc[i]['volume'] >= avg_volume_5d * volume_threshold:
                        has_volume_condition = True
                        if consecutive_above == 1:
                            breakthrough_date = after_central.iloc[i].get('date')
                else:
                    consecutive_above = 0
                    has_volume_condition = False
                
                if consecutive_above >= consecutive_days_requirement and has_volume_condition:
                    break
            
            has_breakthrough = consecutive_above >= consecutive_days_requirement and has_volume_condition
        
        # 5. 检查回抽有效性
        pullback_not_below = False
        pullback_min_close = None
        
        if has_breakthrough and breakthrough_date:
            # 找到突破日期的索引位置
            breakthrough_idx = None
            for i in range(len(after_central)):
                if after_central.iloc[i].get('date') == breakthrough_date:
                    breakthrough_idx = after_central.iloc[i].name
                    break
            
            if breakthrough_idx:
                after_breakthrough = after_central[after_central.index > breakthrough_idx]
                if len(after_breakthrough) >= 5:
                    # 检查回抽是否未跌破中枢上沿
                    pullback_min_close = after_breakthrough['close'].min()
                    
                    # 对于不同波动等级可能有不同的容差
                    tolerance = 0.0 if self.volatility_level == VolatilityLevel.LOW else 0.002
                    pullback_not_below = pullback_min_close >= central_bank_top * (1 - tolerance)
        
        # 6. 背驰验证：回抽过程中30分钟级别形成底背驰
        macd_divergence = False
        min30_details = {"available": False}
        
        # 如果提供了30分钟数据，进行详细的底背驰验证
        if min30_df is not None and len(min30_df) >= 100:
            min30_details["available"] = True
            
            # 应用30分钟K线包含处理
            processed_min30_df = self._process_kline_inclusion(min30_df)
            
            # 计算30分钟MACD
            min30_with_macd = self._calculate_macd(processed_min30_df)
            
            # 识别最近的下跌段（回抽段）
            downtrend_segments = []
            current_segment = None
            
            for i in range(1, len(min30_with_macd)):
                is_red = min30_with_macd.iloc[i]['close'] < min30_with_macd.iloc[i-1]['close']
                
                if is_red and current_segment is None:
                    current_segment = {
                        'start_index': i,
                        'min_price': min30_with_macd.iloc[i]['low'],
                        'min_diff': min30_with_macd.iloc[i]['macd_diff'],
                        'max_green_hist': 0,
                        'length': 1
                    }
                elif current_segment is not None:
                    current_segment['length'] += 1
                    
                    if min30_with_macd.iloc[i]['low'] < current_segment['min_price']:
                        current_segment['min_price'] = min30_with_macd.iloc[i]['low']
                        current_segment['min_diff'] = min30_with_macd.iloc[i]['macd_diff']
                    
                    if min30_with_macd.iloc[i]['macd_hist'] < 0:
                        current_segment['max_green_hist'] = min(current_segment['max_green_hist'], min30_with_macd.iloc[i]['macd_hist'])
                    
                    # 下跌段结束条件：连续3根阳线
                    if not is_red and i >= 2 and not (min30_with_macd.iloc[i-1]['close'] < min30_with_macd.iloc[i-2]['close']):
                        if current_segment['length'] >= 5:  # 至少5根K线的下跌段
                            downtrend_segments.append(current_segment)
                        current_segment = None
            
            # 检查是否有足够的下跌段进行背驰比较
            if len(downtrend_segments) >= 2:
                latest_segment = downtrend_segments[-1]
                previous_segment = downtrend_segments[-2]
                
                # 价格新低
                price_lower = latest_segment['min_price'] < previous_segment['min_price']
                # MACD黄白线不新低
                diff_higher = latest_segment['min_diff'] > previous_segment['min_diff']
                # 绿柱缩小
                hist_cond = False
                if previous_segment['max_green_hist'] != 0:
                    hist_cond = latest_segment['max_green_hist'] > previous_segment['max_green_hist'] * green_hist_ratio_threshold
                
                macd_divergence = price_lower and diff_higher and hist_cond
                min30_details["divergence"] = macd_divergence
                min30_details["price_lower"] = price_lower
                min30_details["diff_higher"] = diff_higher
                min30_details["hist_cond"] = hist_cond
        else:
            # 如果没有30分钟数据，使用日线数据简化验证
            self.logger.info("未提供30分钟数据，使用日线MACD简化验证")
            # 检查是否有底背驰迹象
            recent_macd = df_with_macd.iloc[-20:].copy()
            
            recent_green_hist = recent_macd[recent_macd['macd_hist'] < 0]['macd_hist']
            
            if len(recent_green_hist) >= 5:
                mid_idx = len(recent_green_hist) // 2
                first_half_avg = abs(sum(recent_green_hist[:mid_idx]) / mid_idx)
                second_half_avg = abs(sum(recent_green_hist[mid_idx:]) / (len(recent_green_hist) - mid_idx))
                
                # 使用动态绿柱阈值
                if second_half_avg < first_half_avg * green_hist_ratio_threshold:
                    macd_divergence = True
        
        # 综合判定（必须同时满足所有条件）
        is_third_buy = (has_breakthrough and 
                       pullback_not_below and 
                       macd_divergence)
        
        details = {
            "central_bank": {
                "top": central_bank_top,
                "bottom": central_bank_bottom,
                "height": central_bank_top - central_bank_bottom,
                "height_pct": central_bank_amplitude * 100,
                "required_amplitude_pct": central_bank_amplitude_threshold * 100
            },
            "breakthrough": {
                "has_breakthrough": has_breakthrough,
                "date": breakthrough_date,
                "consecutive_days": consecutive_above if self.volatility_level == VolatilityLevel.LOW else f"{days_above_threshold}+{days_above_critical}",
                "threshold": breakthrough_threshold,
                "volume_condition": has_volume_condition
            },
            "pullback": {
                "not_below": pullback_not_below,
                "min_close": pullback_min_close
            },
            "macd": {
                "divergence": macd_divergence
            },
            "min30_details": min30_details,
            "in_central_range": in_central_range,
            "current_price": current_price,
            "volatility_level": self.volatility_level.value if self.volatility_level else "未设置",
            "data_source": "满足",
            "dynamic_params": {
                "central_bank_amplitude_threshold": central_bank_amplitude_threshold,
                "breakthrough_threshold_multiplier": breakthrough_threshold_multiplier,
                "volume_threshold": volume_threshold,
                "green_hist_ratio_threshold": green_hist_ratio_threshold
            }
        }
        
        self.logger.info(f"日线三买信号检测结果: {'满足' if is_third_buy else '不满足'}")
        self.logger.info(f"中枢振幅: {central_bank_amplitude*100:.2f}%, 阈值: {central_bank_amplitude_threshold*100:.2f}%")
        self.logger.info(f"突破中枢有效性: {'满足' if has_breakthrough else '不满足'}")
        self.logger.info(f"回抽有效性: {'满足' if pullback_not_below else '不满足'}")
        self.logger.info(f"MACD背驰: {'满足' if macd_divergence else '不满足'}")
        self.logger.info(f"波动等级: {self.volatility_level.value if self.volatility_level else '未设置'}")
        
        # 兼容旧版本日志格式
        logger.info(f"日线三买信号检测结果: {'满足' if is_third_buy else '不满足'}")
        logger.info(f"突破中枢有效性: {'满足' if has_breakthrough else '不满足'}")
        logger.info(f"回抽有效性: {'满足' if pullback_not_below else '不满足'}")
        logger.info(f"30分钟背驰: {'满足' if macd_divergence else '不满足'}")
        logger.info(f"中枢内信号排除: {'触发' if in_central_range else '未触发'}")
        
        return is_third_buy, details
    
    def detect_daily_reverse_pullback(self, df: pd.DataFrame, min30_df: Optional[pd.DataFrame] = None, logger=None) -> Tuple[bool, Dict[str, any]]:
        """检测日线级别破中枢反抽买入信号 - 支持全波动ETF适配
        
        信号特征：
        1. 价格跌破中枢下沿
        2. 创阶段新低
        3. 形成企稳结构
        4. 价格站回中枢范围
        5. 成交量配合
        6. 30分钟级别背驰验证
        
        Args:
            df: 日线数据框
            min30_df: 30分钟级别数据框（可选）
            logger: 日志记录器
            
        Returns:
            (是否满足反抽信号, 详细信息字典)
        """
        if logger is None:
            logger = self.logger
            
        logger.info("开始检测日线破中枢反抽信号...")
        
        # 1. 数据源有效性校验
        if len(df) < 20:
            logger.warning("日线数据不足20天，跳过反抽信号检测")
            return False, {"error": "数据不足", "data_source": "不足"}
        
        # 2. 应用K线包含处理
        df_processed = df.copy()
        if hasattr(self, '_process_kline_inclusion'):
            df_processed = self._process_kline_inclusion(df_processed)
        
        # 3. 复制数据并计算MACD
        df_with_macd = df_processed.copy()
        df_with_macd = self._calculate_macd(df_with_macd)
        
        # 4. 根据波动等级设置动态参数
        # 中枢有效性优化：分波动等级振幅阈值（新增）
        if self.volatility_level == VolatilityLevel.LOW:
            # 低波动ETF参数
            central_bank_amplitude_threshold = 0.04  # 中枢振幅阈值4%（优化后：低波动≥4%）
            back_to_central_ratio = 0.95  # 站回中枢比例95%
            hist_back_to_central_ratio = 0.90  # 历史中枢站回比例90%
            stability_volatility_threshold = 3.0  # 企稳波动率阈值3%
            volume_threshold = 1.10  # 成交量放大阈值10%
            price_rise_threshold = 1.5  # 价格上涨阈值1.5%
            consecutive_above_days = 2  # 连续站回天数2天
        elif self.volatility_level == VolatilityLevel.MEDIUM:
            # 中波动ETF参数
            central_bank_amplitude_threshold = 0.06  # 中枢振幅阈值6%（优化后：中波动≥6%）
            back_to_central_ratio = 0.92  # 站回中枢比例92%
            hist_back_to_central_ratio = 0.88  # 历史中枢站回比例88%
            stability_volatility_threshold = 4.0  # 企稳波动率阈值4%
            volume_threshold = 1.08  # 成交量放大阈值8%
            price_rise_threshold = 1.2  # 价格上涨阈值1.2%
            consecutive_above_days = 2  # 连续站回天数2天
        else:
            # 高波动ETF参数
            central_bank_amplitude_threshold = 0.08  # 中枢振幅阈值8%（优化后：高波动≥8%）
            back_to_central_ratio = 0.90  # 站回中枢比例90%
            hist_back_to_central_ratio = 0.85  # 历史中枢站回比例85%
            stability_volatility_threshold = 5.0  # 企稳波动率阈值5%
            volume_threshold = 1.05  # 成交量放大阈值5%
            price_rise_threshold = 1.0  # 价格上涨阈值1%
            consecutive_above_days = 2  # 连续站回天数2天
        
        # 5. 使用多时间窗口寻找中枢（增加捕获概率）
        # 主要中枢：最近20-50天
        central_range_main = df_with_macd.iloc[-50:-20]
        central_bank_top_main = central_range_main['high'].max()
        central_bank_bottom_main = central_range_main['low'].min()
        central_bank_amplitude_main = (central_bank_top_main - central_bank_bottom_main) / central_bank_bottom_main
        
        # 备用中枢：最近15-35天（更短周期）
        if len(df_with_macd) >= 35:
            central_range_backup = df_with_macd.iloc[-35:-15]
            central_bank_top_backup = central_range_backup['high'].max()
            central_bank_bottom_backup = central_range_backup['low'].min()
            central_bank_amplitude_backup = (central_bank_top_backup - central_bank_bottom_backup) / central_bank_bottom_backup
        else:
            central_bank_top_backup = central_bank_top_main
            central_bank_bottom_backup = central_bank_bottom_main
            central_bank_amplitude_backup = central_bank_amplitude_main
        
        # 6. 历史重要中枢识别（增加对历史重要低点的识别）
        historical_central_bottom = None
        historical_central_top = None
        historical_central_amplitude = 0
        
        # 如果有足够的数据，寻找历史重要中枢（50-120天前）
        if len(df_with_macd) >= 120:
            # 寻找历史重要中枢区域
            historical_range = df_with_macd.iloc[-120:-50]
            # 识别历史中枢上下沿
            historical_central_top = historical_range['high'].max()
            historical_central_bottom = historical_range['low'].min()
            historical_central_amplitude = (historical_central_top - historical_central_bottom) / historical_central_bottom * 100
            
            logger.info(f"历史重要中枢分析（50-120天前）：")
            logger.info(f"  历史中枢上沿: {historical_central_top:.6f}")
            logger.info(f"  历史中枢下沿: {historical_central_bottom:.6f}")
            logger.info(f"  历史中枢振幅: {historical_central_amplitude:.2f}%")
            
            # 只保留有意义的历史中枢（根据波动等级动态调整波动率范围）
            min_hist_volatility = 2.0 if self.volatility_level == VolatilityLevel.LOW else 3.0
            max_hist_volatility = 12.0 if self.volatility_level == VolatilityLevel.LOW else 15.0
            if not (min_hist_volatility < historical_central_amplitude < max_hist_volatility):
                historical_central_bottom = None
                historical_central_top = None
                logger.info(f"  历史中枢波动性不符合要求（范围：{min_hist_volatility:.1f}%-{max_hist_volatility:.1f}%），忽略")
        else:
            logger.info("  数据不足120天，无法识别历史重要中枢")
        
        # 7. 检查是否跌破中枢下沿（同时检查主备用中枢和历史重要中枢）
        has_below_central = False
        has_below_historical = False  # 是否跌破历史重要中枢
        recent_low = None
        
        # 检查最近15天内是否跌破中枢下沿
        for i in range(len(df_with_macd) - 15, len(df_with_macd)):
            # 检查是否跌破当前中枢（只考虑有效中枢）
            if ((central_bank_amplitude_main >= central_bank_amplitude_threshold and 
                 df_with_macd.iloc[i]['close'] < central_bank_bottom_main) or 
                (central_bank_amplitude_backup >= central_bank_amplitude_threshold and 
                 df_with_macd.iloc[i]['close'] < central_bank_bottom_backup)):
                has_below_central = True
                if recent_low is None or df_with_macd.iloc[i]['low'] < recent_low:
                    recent_low = df_with_macd.iloc[i]['low']
            
            # 检查是否跌破历史重要中枢
            if historical_central_bottom and df_with_macd.iloc[i]['close'] < historical_central_bottom:
                has_below_historical = True
                if recent_low is None or df_with_macd.iloc[i]['low'] < recent_low:
                    recent_low = df_with_macd.iloc[i]['low']
        
        # 记录跌破历史中枢的情况
        if historical_central_bottom and has_below_historical:
            logger.info(f"检测到跌破历史重要中枢！历史中枢下沿: {historical_central_bottom:.6f}")
        
        # 合并判断条件：只要跌破任一中枢就算满足条件
        has_below_any_central = has_below_central or (historical_central_bottom and has_below_historical)
        
        # 8. 检查是否创最近新低（根据波动等级调整误差范围）
        is_recent_low_new = False
        if recent_low is not None:
            # 检查最近15天的低点
            recent_days = df_with_macd.iloc[-15:]
            error_margin = 1.05 if self.volatility_level == VolatilityLevel.HIGH else 1.03
            is_recent_low_new = recent_low <= recent_days['low'].min() * error_margin
        
        # 9. 企稳结构检查（根据波动等级调整波动率阈值）
        has_stability = False
        stability_count = 0
        stability_start = None
        
        # 方案1：连续2天收盘价不再创新低
        recent_close_prices = df_with_macd.tail(5)['close']
        if len(recent_close_prices) >= 3:
            # 最近2天的收盘价都高于或等于倒数第三天
            if recent_close_prices.iloc[-1] >= recent_close_prices.iloc[-3] and \
               recent_close_prices.iloc[-2] >= recent_close_prices.iloc[-3]:
                has_stability = True
                stability_count = 2
                stability_start = df_with_macd.index[-3]
        
        # 方案2：价格波动率
        if not has_stability:
            # 计算最近3-5天的价格波动率
            for window_size in range(3, min(6, len(df_with_macd))):
                recent_prices = df_with_macd.tail(window_size)['close']
                price_range = recent_prices.max() - recent_prices.min()
                price_mean = recent_prices.mean()
                volatility = price_range / price_mean * 100
                
                # 根据波动等级使用不同的波动率条件
                if volatility < stability_volatility_threshold:
                    has_stability = True
                    stability_start = df_with_macd.index[-window_size]
                    break
        
        # 10. 站回中枢的条件（根据波动等级调整站回比例）
        back_to_central = False
        back_to_historical = False  # 是否站回历史中枢
        
        # 检查最近5天内是否有2天收盘价站回中枢下沿的指定比例以上
        above_count = 0
        hist_above_count = 0
        consecutive_above_current = 0
        
        for i in range(len(df_with_macd) - 5, len(df_with_macd)):
            # 检查是否站回当前中枢
            current_central_condition = ((central_bank_amplitude_main >= central_bank_amplitude_threshold and 
                                         df_with_macd.iloc[i]['close'] > central_bank_bottom_main * back_to_central_ratio) or 
                                        (central_bank_amplitude_backup >= central_bank_amplitude_threshold and 
                                         df_with_macd.iloc[i]['close'] > central_bank_bottom_backup * back_to_central_ratio))
            
            if current_central_condition:
                above_count += 1
                consecutive_above_current += 1
            else:
                consecutive_above_current = 0
            
            # 检查是否站回历史重要中枢
            if historical_central_bottom and df_with_macd.iloc[i]['close'] > historical_central_bottom * hist_back_to_central_ratio:
                hist_above_count += 1
        
        # 根据波动等级判断站回条件
        if self.volatility_level == VolatilityLevel.LOW:
            # 低波动：严格要求连续站回
            back_to_central = consecutive_above_current >= consecutive_above_days
        else:
            # 中高波动：允许分散站回
            back_to_central = above_count >= consecutive_above_days
        
        # 历史中枢站回判定
        if historical_central_bottom and hist_above_count >= 1:
            back_to_historical = True
            logger.info(f"检测到站回历史中枢区域！")
        
        # 合并判断条件：站回任一中枢就算满足
        back_to_any_central = back_to_central or back_to_historical
        
        # 替代条件：检查最近3天是否有至少1天收盘价上涨超过指定阈值
        if not back_to_central and len(df_with_macd) >= 3:
            for i in range(1, 3):
                if i < len(df_with_macd):
                    daily_change = (df_with_macd.iloc[-i]['close'] - df_with_macd.iloc[-i-1]['close']) / df_with_macd.iloc[-i-1]['close'] * 100
                    if daily_change > price_rise_threshold:
                        back_to_central = True
                        back_to_any_central = True
                        break
        
        # 11. 成交量条件（根据波动等级调整要求）
        volume_condition = False
        
        # 条件1：最近3天中至少有1天成交量比前一天增加
        if len(df_with_macd) >= 4:
            for i in range(1, 3):
                if i < len(df_with_macd):
                    if df_with_macd.iloc[-i]['volume'] > df_with_macd.iloc[-i-1]['volume'] * volume_threshold:
                        volume_condition = True
                        break
        
        # 低波动ETF：成交量条件更严格，必须满足
        # 中高波动ETF：如果没有找到满足条件的成交量，直接设为True
        if not volume_condition and self.volatility_level != VolatilityLevel.LOW:
            volume_condition = True
        
        # 12. 30分钟级别背驰验证
        min30_details = {"validated": False, "has_divergence": False}
        if min30_df is not None and len(min30_df) > 60:
            min30_details["validated"] = True
            # 计算30分钟MACD
            min30_with_macd = min30_df.copy()
            min30_with_macd = self._calculate_macd(min30_with_macd, fast_period=12, slow_period=26, signal_period=9)
            
            # 检查最近30分钟K线是否有底背驰迹象
            recent_min30 = min30_with_macd.iloc[-30:].copy()
            recent_green_hist = recent_min30[recent_min30['macd_hist'] < 0]['macd_hist']
            
            if len(recent_green_hist) >= 5:
                mid_idx = len(recent_green_hist) // 2
                first_half_avg = abs(sum(recent_green_hist[:mid_idx]) / mid_idx)
                second_half_avg = abs(sum(recent_green_hist[mid_idx:]) / (len(recent_green_hist) - mid_idx))
                
                # 使用动态阈值：低波动0.8，中高波动0.7
                green_hist_ratio_threshold = 0.8 if self.volatility_level == VolatilityLevel.LOW else 0.7
                if second_half_avg < first_half_avg * green_hist_ratio_threshold:
                    min30_details["has_divergence"] = True
                    logger.info("30分钟级别检测到底背驰迹象，增强反抽信号强度")
        
        # 13. 综合判定逻辑
        # 核心条件：必须满足跌破中枢和企稳结构
        # 辅助条件：至少满足创新低、站回中枢、成交量条件中的1个
        
        # 使用合并后的中枢判断条件
        core_conditions = has_below_any_central and has_stability
        support_conditions = is_recent_low_new or back_to_any_central or volume_condition
        
        # 基础判定：核心条件必须满足，辅助条件至少满足1个
        is_reverse_pullback = core_conditions and support_conditions
        
        # 30分钟背驰增强：如果有30分钟背驰，可以放宽辅助条件
        if core_conditions and min30_details["validated"] and min30_details["has_divergence"]:
            # 即使辅助条件不满足，也可以触发信号
            if not is_reverse_pullback:
                logger.info("虽然常规辅助条件不满足，但30分钟级别背驰明显，触发反抽信号")
                is_reverse_pullback = True
        
        # 特殊处理历史中枢突破的情况
        # 如果跌破历史重要中枢后企稳，应给予更高优先级
        if historical_central_bottom and has_below_historical and has_stability:
            logger.info(f"特殊情况：跌破历史重要中枢后企稳，增强信号强度！")
            # 即使其他辅助条件不满足，也可以考虑触发信号
            if not is_reverse_pullback:
                logger.warning(f"虽然常规条件不满足，但跌破历史重要中枢后企稳，强制触发反抽信号")
                is_reverse_pullback = True
        
        # 14. 添加信号有效性标记（30分钟级强制验证）
        signal_validity = "有效信号"
        if is_reverse_pullback and not (min30_details["validated"] and min30_details["has_divergence"]):
            signal_validity = "弱信号（30分钟未验证）"
        elif not is_reverse_pullback:
            signal_validity = "无信号"
        
        # 构建详细信息
        details = {
            "central_bank": {
                "top_main": central_bank_top_main,
                "bottom_main": central_bank_bottom_main,
                "amplitude_main_pct": central_bank_amplitude_main * 100,
                "top_backup": central_bank_top_backup,
                "bottom_backup": central_bank_bottom_backup,
                "amplitude_backup_pct": central_bank_amplitude_backup * 100,
                "historical_top": historical_central_top,
                "historical_bottom": historical_central_bottom,
                "historical_amplitude_pct": historical_central_amplitude,
                "required_amplitude_pct": central_bank_amplitude_threshold * 100
            },
            "breakdown": {
                "has_below_central": has_below_central,
                "has_below_historical": has_below_historical,
                "has_below_any_central": has_below_any_central,
                "is_recent_low_new": is_recent_low_new,
                "recent_low": recent_low
            },
            "stability": {
                "has_stability": has_stability,
                "stability_count": stability_count,
                "stability_start": str(stability_start) if stability_start else None
            },
            "pullback": {
                "back_to_central": back_to_central,
                "back_to_historical": back_to_historical,
                "back_to_any_central": back_to_any_central,
                "volume_condition": volume_condition,
                "consecutive_above_current": consecutive_above_current,
                "above_count": above_count
            },
            "min30_details": min30_details,
            "core_conditions_met": core_conditions,
            "support_conditions_met": support_conditions,
            "current_price": df_with_macd.iloc[-1]['close'] if not df_with_macd.empty else None,
            "volatility_level": self.volatility_level.value if self.volatility_level else "未设置",
            "data_source": "满足",
            "signal_validity": signal_validity,  # 新增：信号有效性标记
            "dynamic_params": {
                "central_bank_amplitude_threshold": central_bank_amplitude_threshold,
                "back_to_central_ratio": back_to_central_ratio,
                "hist_back_to_central_ratio": hist_back_to_central_ratio,
                "stability_volatility_threshold": stability_volatility_threshold,
                "volume_threshold": volume_threshold,
                "price_rise_threshold": price_rise_threshold,
                "consecutive_above_days": consecutive_above_days
            }
        }
        
        logger.info(f"日线破中枢反抽信号检测结果: {'满足' if is_reverse_pullback else '不满足'}")
        logger.info(f"核心条件满足: {'满足' if core_conditions else '不满足'} (跌破任一中枢: {has_below_any_central}, 企稳结构: {has_stability})")
        logger.info(f"  - 跌破当前中枢: {has_below_central}")
        logger.info(f"  - 跌破历史中枢: {has_below_historical if historical_central_bottom else 'N/A'}")
        logger.info(f"辅助条件满足: {'满足' if support_conditions else '不满足'} (创新低: {is_recent_low_new}, 站回任一中枢: {back_to_any_central}, 成交量: {volume_condition})")
        logger.info(f"  - 站回当前中枢: {back_to_central}")
        logger.info(f"  - 站回历史中枢: {back_to_historical if historical_central_bottom else 'N/A'}")
        logger.info(f"波动等级: {self.volatility_level.value if self.volatility_level else '未设置'}")
        logger.info(f"30分钟背驰验证: {'有效' if min30_details['validated'] else '无效'} (底背驰: {min30_details['has_divergence']})")
        logger.info(f"信号有效性: {signal_validity}")  # 新增：信号有效性日志
        
        # 兼容旧版本日志格式
        logger.info(f"日线反抽信号检测结果: {'满足' if is_reverse_pullback else '不满足'}")
        
        return is_reverse_pullback, details
    
    def detect_buy_signals(self, df: pd.DataFrame, min30_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """检测所有日线买点信号
        
        按照优先级顺序：二买 > 一买 > 三买 > 反抽
        
        Args:
            df: 日线数据框
            min30_data: 30分钟级别数据框（可选）
            
        Returns:
            包含所有买点检测结果的字典
        """
        logger.info("开始检测所有日线买点信号...")
        
        # 按照优先级顺序检测买点
        # 1. 先检测一买，因为二买依赖一买
        first_buy, first_buy_details = self.detect_daily_first_buy(df)
        
        # 2. 检测二买（核心买点）
        second_buy, second_buy_details = self.detect_daily_second_buy(df, first_buy)
        
        # 3. 如果没有二买，检测一买（辅助买点）
        # 一买已经检测过了
        
        # 4. 检测三买（辅助买点）
        third_buy, third_buy_details = self.detect_daily_third_buy(df)
        
        # 5. 检测反抽（兜底买点）
        reverse_pullback, reverse_pullback_details = self.detect_daily_reverse_pullback(df)
        
        # 移除了强制触发反抽信号的代码
        
        # 确定最强买点
        if second_buy:
            strongest_signal = BuySignalType.SECOND_BUY
            signal_type_priority = "核心"
        elif first_buy:
            strongest_signal = BuySignalType.FIRST_BUY
            signal_type_priority = "辅助"
        elif third_buy:
            strongest_signal = BuySignalType.THIRD_BUY
            signal_type_priority = "辅助"
        elif reverse_pullback:
            strongest_signal = BuySignalType.REVERSE_PULLBACK
            signal_type_priority = "兜底"
        else:
            strongest_signal = BuySignalType.NONE
            signal_type_priority = "无"
        
        # 统计满足的买点数量，确保所有值都是布尔值
        second_buy = bool(second_buy)
        first_buy = bool(first_buy)
        third_buy = bool(third_buy)
        reverse_pullback = bool(reverse_pullback)
        satisfied_signals = sum([second_buy, first_buy, third_buy, reverse_pullback])
        
        # 收集各买点的失败原因（无买点细化）
        no_buy_reasons = []
        
        # 一买失败原因
        if not first_buy:
            first_buy_reason = "无一买："
            if "error" in first_buy_details:
                first_buy_reason += first_buy_details["error"]
            elif "has_bottom_divergence" in first_buy_details and not first_buy_details["has_bottom_divergence"]:
                first_buy_reason += "无MACD底背驰"
            elif "has_new_low" in first_buy_details and not first_buy_details["has_new_low"]:
                first_buy_reason += "价格未创新低"
            elif "has_bottom_candle" in first_buy_details and not first_buy_details["has_bottom_candle"]:
                first_buy_reason += "无底分型"
            elif "has_volume_expansion" in first_buy_details and not first_buy_details["has_volume_expansion"]:
                first_buy_reason += "无成交量放大"
            else:
                first_buy_reason += "未满足核心条件"
            no_buy_reasons.append(first_buy_reason)
        
        # 二买失败原因
        if not second_buy:
            second_buy_reason = "无二买："
            if "error" in second_buy_details:
                second_buy_reason += second_buy_details["error"]
            elif "has_first_buy" in second_buy_details and not second_buy_details["has_first_buy"]:
                second_buy_reason += "一买未确认"
            elif "has_callback" in second_buy_details and not second_buy_details["has_callback"]:
                second_buy_reason += "回调不充分"
            elif "has_rebound" in second_buy_details and not second_buy_details["has_rebound"]:
                second_buy_reason += "反弹力度不足"
            elif "callback_below_central" in second_buy_details and second_buy_details["callback_below_central"]:
                second_buy_reason += "回调跌破中枢"
            else:
                second_buy_reason += "未满足核心条件"
            no_buy_reasons.append(second_buy_reason)
        
        # 三买失败原因
        if not third_buy:
            third_buy_reason = "无三买："
            if "error" in third_buy_details:
                third_buy_reason += third_buy_details["error"]
            elif "has_central" in third_buy_details and not third_buy_details["has_central"]:
                third_buy_reason += "中枢未形成"
            elif "has_breakthrough" in third_buy_details and not third_buy_details["has_breakthrough"]:
                third_buy_reason += "中枢上沿未突破"
            elif "has_callback" in third_buy_details and not third_buy_details["has_callback"]:
                third_buy_reason += "回调不充分"
            elif "callback_below_central" in third_buy_details and third_buy_details["callback_below_central"]:
                third_buy_reason += "回调跌破中枢"
            else:
                third_buy_reason += "未满足核心条件"
            no_buy_reasons.append(third_buy_reason)
        
        # 反抽失败原因
        if not reverse_pullback:
            reverse_pullback_reason = "无反抽："
            if "error" in reverse_pullback_details:
                reverse_pullback_reason += reverse_pullback_details["error"]
            elif "core_conditions_met" in reverse_pullback_details and not reverse_pullback_details["core_conditions_met"]:
                if "has_below_any_central" in reverse_pullback_details and not reverse_pullback_details["has_below_any_central"]:
                    reverse_pullback_reason += "未跌破中枢"
                elif "has_stability" in reverse_pullback_details and not reverse_pullback_details["has_stability"]:
                    reverse_pullback_reason += "企稳结构未形成"
                else:
                    reverse_pullback_reason += "核心条件未满足"
            elif "support_conditions_met" in reverse_pullback_details and not reverse_pullback_details["support_conditions_met"]:
                reverse_pullback_reason += "辅助条件未满足"
            else:
                reverse_pullback_reason += "未满足核心条件"
            no_buy_reasons.append(reverse_pullback_reason)
        
        result = {
            "strongest_signal": strongest_signal.value,
            "signal_type_priority": signal_type_priority,
            "satisfied_signals_count": satisfied_signals,
            "no_buy_reasons": no_buy_reasons,  # 新增：无买点原因列表
            "signals": {
                "second_buy": {
                    "detected": second_buy,
                    "details": second_buy_details
                },
                "first_buy": {
                    "detected": first_buy,
                    "details": first_buy_details
                },
                "third_buy": {
                    "detected": third_buy,
                    "details": third_buy_details
                },
                "reverse_pullback": {
                    "detected": reverse_pullback,
                    "details": reverse_pullback_details
                }
            },
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"日线买点检测完成，最强信号: {strongest_signal.value}")
        logger.info(f"满足的买点数量: {satisfied_signals}/4")
        
        return result
    
    def generate_buy_signal_report(self, result: Dict[str, any]) -> str:
        """生成买点检测报告
        
        Args:
            result: 买点检测结果
            
        Returns:
            格式化的报告字符串
        """
        report_lines = ["===== 日线买点检测报告 ====="]
        
        # 总体结果
        report_lines.append(f"最强买点信号: {result['strongest_signal']}")
        report_lines.append(f"信号优先级: {result['signal_type_priority']}")
        report_lines.append(f"满足买点数量: {result['satisfied_signals_count']}/4")
        report_lines.append(f"检测时间: {result['timestamp']}")
        report_lines.append("")
        
        # 详细信号分析
        report_lines.append("【详细买点分析】")
        
        # 二买分析
        second_buy = result['signals']['second_buy']
        report_lines.append(f"1. 日线二买（核心）: {'✓ 满足' if second_buy['detected'] else '✗ 不满足'}")
        
        # 一买分析
        first_buy = result['signals']['first_buy']
        report_lines.append(f"2. 日线一买（辅助）: {'✓ 满足' if first_buy['detected'] else '✗ 不满足'}")
        
        # 三买分析
        third_buy = result['signals']['third_buy']
        report_lines.append(f"3. 日线三买（辅助）: {'✓ 满足' if third_buy['detected'] else '✗ 不满足'}")
        
        # 反抽分析
        reverse_pullback = result['signals']['reverse_pullback']
        report_lines.append(f"4. 破中枢反抽（兜底）: {'✓ 满足' if reverse_pullback['detected'] else '✗ 不满足'}")
        report_lines.append("")
        
        # 交易建议
        report_lines.append("【交易建议】")
        if result['strongest_signal'] == BuySignalType.SECOND_BUY.value:
            report_lines.append("✓ 日线二买（核心买点）确认，建议重点关注")
            report_lines.append("  - 优先匹配30分钟向上笔建仓（子仓位比例60%-70%）")
            report_lines.append("  - 加仓优先级最高")
        elif result['strongest_signal'] == BuySignalType.FIRST_BUY.value:
            report_lines.append("△ 日线一买（辅助买点）确认")
            report_lines.append("  - 可匹配15分钟向上笔建仓（子仓位比例20%-40%）")
            report_lines.append("  - 需谨慎，建议控制仓位")
        elif result['strongest_signal'] == BuySignalType.THIRD_BUY.value:
            report_lines.append("△ 日线三买（辅助买点）确认")
            report_lines.append("  - 可匹配15分钟向上笔建仓（子仓位比例20%-40%）")
            report_lines.append("  - 需确认突破有效性")
        elif result['strongest_signal'] == BuySignalType.REVERSE_PULLBACK.value:
            report_lines.append("! 破中枢反抽（兜底买点）确认")
            report_lines.append("  - 仅作为兜底策略，建议最小仓位试探")
            report_lines.append("  - 严格设置止损")
        else:
            report_lines.append("✗ 当前无明确日线买点信号")
            report_lines.append("  - 建议继续等待信号明确")
            report_lines.append("  - 优先关注周线多头趋势下的日线二买")
            
            # 新增：无买点原因明细
            if "no_buy_reasons" in result and result["no_buy_reasons"]:
                report_lines.append("")
                report_lines.append("【无买点原因明细】")
                for reason in result["no_buy_reasons"]:
                    report_lines.append(f"  • {reason}")
        
        report_lines.append("===================================")
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    # 测试用例
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建日线数据
    days = 80  # 80天数据
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # 创建先跌后涨的数据，模拟可能的买点
    base_price = 10.0
    prices = []
    current_price = base_price
    
    # 前40天：下跌趋势
    for i in range(40):
        change = -np.random.random() * 0.1 - 0.02  # 下跌
        current_price = max(base_price * 0.7, current_price + change)
        prices.append(current_price)
    
    # 中间20天：盘整
    for i in range(20):
        change = (np.random.random() - 0.5) * 0.1
        current_price += change
        prices.append(current_price)
    
    # 后20天：上涨趋势
    for i in range(20):
        change = np.random.random() * 0.1 + 0.02  # 上涨
        current_price += change
        prices.append(current_price)
    
    # 创建其他数据列
    daily_data = {
        'date': dates,
        'open': [p * (1 - np.random.random() * 0.01) for p in prices],
        'high': [p * (1 + np.random.random() * 0.02) for p in prices],
        'low': [p * (1 - np.random.random() * 0.02) for p in prices],
        'close': prices,
        'volume': [np.random.random() * 1000000 + 500000 for _ in range(days)]
    }
    
    daily_df = pd.DataFrame(daily_data)
    
    # 创建检测器并执行检测
    detector = BuySignalDetector()
    result = detector.detect_buy_signals(daily_df)
    
    # 打印报告
    print(detector.generate_buy_signal_report(result))