#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论核心计算器 - 完整修复版
修复了周线参数加载问题，增强配置验证与防御性编程
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any
import sys
import os
from datetime import datetime, timedelta

# 配置日志系统
logger = logging.getLogger('ChanlunCalculator')
logger.setLevel(logging.INFO)

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ChanlunCalculator:
    """缠论核心计算器 - 支持多时间级别指标计算与回测"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化计算器
        :param config: 全局配置字典，需包含chanlun子配置
        """
        self.config = config if config is not None else {}
        
        # ------------------------------
        # 1. 从chanlun子配置提取周线参数（关键修复）
        # ------------------------------
        chanlun_config = self.config.get('chanlun', {})
        self.weekly_fractal_sensitivity = chanlun_config.get('weekly_fractal_sensitivity', 3)
        self.weekly_pen_min_length = chanlun_config.get('weekly_pen_min_length', 5)
        self.weekly_central_min_length = chanlun_config.get('weekly_central_min_length', 5)
        
        # ------------------------------
        # 2. 日志验证参数加载（关键修复）
        # ------------------------------
        logger.info(
            f"周线参数加载完成 - "
            f"分型敏感度={self.weekly_fractal_sensitivity}, "
            f"笔最小长度={self.weekly_pen_min_length}, "
            f"中枢最小长度={self.weekly_central_min_length}"
        )
        
        # 基础缠论参数
        self.fractal_sensitivity = self.config.get('fractal_sensitivity', 3)
        self.pen_min_length = self.config.get('pen_min_length', 5)
        self.central_bank_min_bars = self.config.get('central_bank_min_bars', 5)
        self.ranging_threshold = self.config.get('ranging_threshold', 0.015)
        self.stop_loss_default = self.config.get('stop_loss_default', 0.03)
        
        # 止损策略配置
        stop_loss_config = self.config.get('risk_management', {}).get('stop_loss_settings', {})
        self.stop_loss_type = stop_loss_config.get('stop_loss_type', 'dynamic')
        self.stop_loss_atr_period = stop_loss_config.get('stop_loss_atr_period', 14)
        self.stop_loss_atr_multiplier = stop_loss_config.get('stop_loss_atr_multiplier', 2.0)
        
        # 时间级别专属参数
        self.minute_fractal_sensitivity = self.config.get('minute_fractal_sensitivity', 5)
        self.minute_pen_min_length = self.config.get('minute_pen_min_length', 10)
        
        # 数据验证配置
        self.date_validation_enabled = self.config.get('date_validation_enabled', True)
        self.min_data_points = self.config.get('min_data_points', 10)
        self.max_date_range_days = self.config.get('max_date_range_days', 365 * 5)  # 5年


    def _validate_input_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        """
        验证输入数据合法性
        :return: (是否有效, 错误信息)
        """
        try:
            # 检查数据是否为空
            if df is None or df.empty:
                return False, f"{timeframe}级别数据为空"
            
            # 检查数据点数
            if len(df) < self.min_data_points:
                return False, f"{timeframe}级别数据点数不足: {len(df)}条，至少需要{self.min_data_points}条"
            
            # 检查日期列
            if 'date' not in df.columns:
                return False, f"{timeframe}级别数据缺少日期列"
            
            # 转换日期格式
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_count = df['date'].isna().sum()
                if invalid_count > 0:
                    return False, f"{timeframe}级别数据包含{invalid_count}个无效日期"
            
            # 检查日期范围
            if len(df) >= 2:
                date_range = df['date'].max() - df['date'].min()
                if date_range.days > self.max_date_range_days:
                    return False, f"{timeframe}级别数据日期范围过大: {date_range.days}天，超过限制{self.max_date_range_days}天"
            
            # 检查必要列
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, f"{timeframe}级别数据缺失必要列: {missing_cols}"
            
            logger.info(f"{timeframe}级别数据验证通过: {len(df)}条记录")
            return True, "数据验证通过"
        
        except Exception as e:
            logger.error(f"数据验证异常: {str(e)}")
            return False, f"验证异常: {str(e)}"


    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算平均真实波幅(ATR)
        """
        if df.empty or len(df) < period:
            logger.warning(f"数据不足，无法计算ATR: {len(df)}条，需要{period}条")
            return pd.Series([0.0] * len(df), index=df.index)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=period).mean()
        
        logger.debug(f"ATR计算完成: 周期={period}, 数据点数={len(atr_series)}")
        return atr_series


    def calculate_dynamic_stoploss(self, df: pd.DataFrame, action: str) -> float:
        """
        计算动态止损价格
        """
        if df.empty:
            logger.warning("数据为空，无法计算动态止损")
            return 0.0
        
        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            atr_series = self.calculate_atr(df, self.stop_loss_atr_period)
            
            if atr_series.empty or len(atr_series) < self.stop_loss_atr_period:
                logger.warning("ATR数据不足，使用默认止损")
                return current_price * (1 - self.stop_loss_default)
            
            current_atr = atr_series.iloc[-1]
            
            if action == 'buy':
                stoploss = current_price - (current_atr * self.stop_loss_atr_multiplier)
                result = max(stoploss, 0.0)
                logger.debug(f"动态止损计算: 买入动作, 当前价={current_price:.2f}, ATR={current_atr:.4f}, 止损价={result:.2f}")
                return result
            elif action == 'sell':
                result = current_price + (current_atr * self.stop_loss_atr_multiplier)
                logger.debug(f"动态止损计算: 卖出动作, 当前价={current_price:.2f}, ATR={current_atr:.4f}, 止损价={result:.2f}")
                return result
            else:
                logger.warning(f"未知动作类型: {action}，使用默认止损")
                return current_price * (1 - self.stop_loss_default)
                
        except Exception as e:
            logger.error(f"计算动态止损失败: {str(e)}")
            return df.iloc[-1]['close'] * (1 - self.stop_loss_default)


    def calculate_signal_strength(self, df: pd.DataFrame) -> float:
        """
        计算信号强度
        """
        if df.empty or len(df) < 5:
            logger.warning("数据不足，信号强度设为默认值50")
            return 50.0
        
        try:
            recent = df.iloc[-5:]
            high_mean, low_mean, close_mean = recent['high'].mean(), recent['low'].mean(), recent['close'].mean()
            
            if close_mean == 0 or np.isnan(close_mean):
                return 50.0
                
            volatility = (high_mean - low_mean) / close_mean
            
            base_strength, volatility_factor = 50, 500 if volatility > 0.02 else 300
            strength = base_strength + volatility * volatility_factor
            
            # 缠论元素增强
            latest = df.iloc[-1]
            if latest.get('top_fractal', False): 
                strength -= 10
            if latest.get('bottom_fractal', False): 
                strength += 10
            if latest.get('pen_end', False):
                if latest.get('pen_type') == 'up': 
                    strength += 15
                else: 
                    strength -= 15
            if latest.get('segment_end', False):
                if latest.get('segment_type') == 'up': 
                    strength += 20
                else: 
                    strength -= 20
            if latest.get('central_bank', False):
                current_price = latest['close']
                central_high = latest.get('central_bank_high', current_price)
                central_low = latest.get('central_bank_low', current_price)
                
                if not np.isnan(central_high) and not np.isnan(central_low):
                    if current_price > central_high: 
                        strength += 25
                    elif current_price < central_low: 
                        strength -= 25
                    else: 
                        strength += 5
            
            return max(0.0, min(100.0, strength))
            
        except Exception as e:
            logger.error(f"计算信号强度失败: {str(e)}")
            return 50.0


    def calculate(self, df: pd.DataFrame, timeframe: str = 'daily') -> pd.DataFrame:
        """
        主计算函数 - 整合缠论元素计算
        :return: 带缠论标记的DataFrame
        """
        logger.info(f"开始计算{timeframe}级别缠论指标")
        
        # 应用时间级别参数
        if timeframe == 'weekly':
            fractal_sens = self.weekly_fractal_sensitivity
            pen_min = self.weekly_pen_min_length
            central_min = self.weekly_central_min_length
            logger.info(f"应用周线参数: 分型={fractal_sens}, 笔={pen_min}, 中枢={central_min}")
        elif timeframe == 'minute':
            fractal_sens = self.minute_fractal_sensitivity
            pen_min = self.minute_pen_min_length
            logger.info(f"应用分钟线参数: 分型={fractal_sens}, 笔={pen_min}")
        else:  # daily
            fractal_sens = self.fractal_sensitivity
            pen_min = self.pen_min_length
            logger.info(f"应用日线参数: 分型={fractal_sens}, 笔={pen_min}")
        
        try:
            # 数据预处理
            df = df.sort_values('date').reset_index(drop=True)
            
            # 初始化缠论列
            df['top_fractal'] = False
            df['bottom_fractal'] = False
            df['pen_type'] = None
            df['pen_start'] = False
            df['pen_end'] = False
            df['segment_type'] = None
            df['segment_start'] = False
            df['segment_end'] = False
            df['central_bank'] = False
            df['central_bank_high'] = np.nan
            df['central_bank_low'] = np.nan
            
            # 数据验证
            is_valid, error_msg = self._validate_input_data(df, timeframe)
            if not is_valid:
                logger.error(f"数据验证失败: {error_msg}")
                return df
            
            # 计算分型
            df = self._calculate_fractal(df, fractal_sens)
            
            # 计算笔
            df = self._calculate_pen(df, pen_min)
            
            # 计算线段
            df = self._calculate_segment(df)
            
            # 计算中枢
            df = self._calculate_central_bank(df, central_min if timeframe == 'weekly' else self.central_bank_min_bars)
            
            logger.info(f"缠论计算完成: 顶分型={df['top_fractal'].sum()}, 底分型={df['bottom_fractal'].sum()}, "
                       f"笔={df['pen_start'].sum()}, 线段={df['segment_start'].sum()}, 中枢={df['central_bank'].sum()}")
            return df
            
        except Exception as e:
            logger.error(f"计算失败: {str(e)}")
            return df


    def _calculate_fractal(self, df: pd.DataFrame, sens: int) -> pd.DataFrame:
        """
        计算顶底分型
        """
        if len(df) < 5:
            logger.warning("数据不足5条，无法计算分型")
            return df
        
        try:
            for i in range(sens, len(df) - sens):
                # 顶分型条件
                high_cond = (
                    df['high'].iloc[i] > df['high'].iloc[i-1] and
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and
                    df['high'].iloc[i] > df['high'].iloc[i+2]
                )
                
                # 底分型条件
                low_cond = (
                    df['low'].iloc[i] < df['low'].iloc[i-1] and
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and
                    df['low'].iloc[i] < df['low'].iloc[i+2]
                )
                
                if high_cond:
                    df.at[i, 'top_fractal'] = True
                if low_cond:
                    df.at[i, 'bottom_fractal'] = True
            
            logger.debug(f"分型计算完成: 顶={df['top_fractal'].sum()}, 底={df['bottom_fractal'].sum()}")
            return df
            
        except Exception as e:
            logger.error(f"分型计算失败: {str(e)}")
            return df


    def _calculate_pen(self, df: pd.DataFrame, min_len: int) -> pd.DataFrame:
        """
        计算笔
        """
        fractal_points = df.index[df['top_fractal'] | df['bottom_fractal']].tolist()
        if not fractal_points:
            logger.warning("无分型点，无法计算笔")
            return df
        
        try:
            pens = []
            current_pen = {
                'type': 'down' if df.loc[fractal_points[0], 'top_fractal'] else 'up',
                'start': fractal_points[0],
                'end': None
            }
            
            for i in range(1, len(fractal_points)):
                idx = fractal_points[i]
                last_idx = fractal_points[i-1]
                is_top = df.loc[idx, 'top_fractal']
                is_bottom = df.loc[idx, 'bottom_fractal']
                
                # 笔结束条件：长度达标且出现反向分型
                if current_pen['type'] == 'up' and is_top and (idx - current_pen['start']) >= min_len:
                    current_pen['end'] = last_idx
                    pens.append(current_pen)
                    current_pen = {'type': 'down', 'start': idx, 'end': None}
                elif current_pen['type'] == 'down' and is_bottom and (idx - current_pen['start']) >= min_len:
                    current_pen['end'] = last_idx
                    pens.append(current_pen)
                    current_pen = {'type': 'up', 'start': idx, 'end': None}
            
            # 处理最后一笔
            if current_pen['end'] is None and (len(df) - current_pen['start']) >= min_len:
                current_pen['end'] = len(df) - 1
                pens.append(current_pen)
            
            # 标记笔
            for pen in pens:
                df.at[pen['start'], 'pen_start'] = True
                df.at[pen['end'], 'pen_end'] = True
                df.at[pen['start'], 'pen_type'] = pen['type']
            
            logger.debug(f"笔计算完成: 共{len(pens)}笔")
            return df
            
        except Exception as e:
            logger.error(f"笔计算失败: {str(e)}")
            return df


    def _calculate_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算线段
        """
        pen_points = [i for i in range(len(df)) if df['pen_start'].iloc[i] or df['pen_end'].iloc[i]]
        if not pen_points:
            logger.warning("无笔点，无法计算线段")
            return df
        
        try:
            segments = []
            current_segment = {
                'type': df.loc[pen_points[0], 'pen_type'],
                'start': pen_points[0],
                'end': None
            }
            
            for i in range(1, len(pen_points)):
                idx = pen_points[i]
                last_idx = pen_points[i-1]
                current_pen_type = df.loc[idx, 'pen_type']
                
                if current_segment['type'] == 'up' and current_pen_type == 'down':
                    current_segment['end'] = last_idx
                    segments.append(current_segment)
                    current_segment = {'type': 'down', 'start': idx, 'end': None}
                elif current_segment['type'] == 'down' and current_pen_type == 'up':
                    current_segment['end'] = last_idx
                    segments.append(current_segment)
                    current_segment = {'type': 'up', 'start': idx, 'end': None}
            
            # 处理最后一线段
            if current_segment['end'] is None:
                current_segment['end'] = len(df) - 1
                segments.append(current_segment)
            
            # 标记线段
            for segment in segments:
                df.at[segment['start'], 'segment_start'] = True
                df.at[segment['end'], 'segment_end'] = True
                df.at[segment['start'], 'segment_type'] = segment['type']
            
            logger.debug(f"线段计算完成: 共{len(segments)}线段")
            return df
            
        except Exception as e:
            logger.error(f"线段计算失败: {str(e)}")
            return df


    def _calculate_central_bank(self, df: pd.DataFrame, min_bars: int) -> pd.DataFrame:
        """
        计算中枢
        """
        segment_points = [i for i in range(len(df)) if df['segment_start'].iloc[i] or df['segment_end'].iloc[i]]
        if len(segment_points) < 4:
            logger.warning("线段点不足4个，无法计算中枢")
            return df
        
        try:
            central_banks = []
            
            for i in range(3, len(segment_points)):
                p1, p2, p3, p4 = segment_points[i-3], segment_points[i-2], segment_points[i-1], segment_points[i]
                
                # 计算价格区间
                price_range1 = (min(df['low'].iloc[p1], df['low'].iloc[p2]), 
                               max(df['high'].iloc[p1], df['high'].iloc[p2]))
                price_range2 = (min(df['low'].iloc[p3], df['low'].iloc[p4]), 
                               max(df['high'].iloc[p3], df['high'].iloc[p4]))
                
                # 检查重叠
                overlap_low = max(price_range1[0], price_range2[0])
                overlap_high = min(price_range1[1], price_range2[1])
                
                if overlap_low < overlap_high and p4 - p1 >= min_bars:
                    central_banks.append({
                        "start": p1, "end": p4, "high": overlap_high, "low": overlap_low
                    })
            
            # 标记中枢
            for cb in central_banks:
                df.loc[cb['start']:cb['end'], 'central_bank'] = True
                df.loc[cb['start']:cb['end'], 'central_bank_high'] = cb['high']
                df.loc[cb['start']:cb['end'], 'central_bank_low'] = cb['low']
            
            logger.debug(f"中枢计算完成: 共{len(central_banks)}个中枢")
            return df
            
        except Exception as e:
            logger.error(f"中枢计算失败: {str(e)}")
            return df


    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        """
        if df.empty:
            return df
            
        try:
            # 初始化信号列
            df['action'] = 'hold'
            df['reason'] = ''
            df['strength'] = 0.0
            df['target_price'] = 0.0
            df['stoploss'] = 0.0
            df['stoploss_type'] = 'fixed'
            
            for i in range(len(df)):
                market_condition = self.determine_market_condition(df[:i+1])
                
                # 根据市场状况生成信号
                if market_condition in ['breakout_up', 'trending_up']:
                    df.at[i, 'action'] = 'buy'
                    df.at[i, 'reason'] = f'市场状况: {market_condition}'
                elif market_condition in ['breakout_down', 'trending_down']:
                    df.at[i, 'action'] = 'sell'
                    df.at[i, 'reason'] = f'市场状况: {market_condition}'
                elif market_condition == 'ranging':
                    current_price = df.iloc[i]['close']
                    if 'central_bank_high' in df.columns and not pd.isna(df.iloc[i].get('central_bank_high', np.nan)):
                        central_high = df.iloc[i]['central_bank_high']
                        central_low = df.iloc[i]['central_bank_low']
                        if current_price > (central_high + central_low) / 2:
                            df.at[i, 'action'] = 'sell'
                            df.at[i, 'reason'] = '震荡市上沿卖出'
                        else:
                            df.at[i, 'action'] = 'buy'
                            df.at[i, 'reason'] = '震荡市下沿买入'
                
                # 计算信号强度和相关价格
                df.at[i, 'strength'] = self.calculate_signal_strength(df[:i+1])
                df.at[i, 'target_price'] = self.calculate_target_price(df[:i+1], df.at[i, 'action'])
                df.at[i, 'stoploss'] = self.calculate_stoploss(df[:i+1], df.at[i, 'action'])
                df.at[i, 'stoploss_type'] = 'dynamic' if self.stop_loss_type == 'dynamic' else 'fixed'
            
            logger.info(f"交易信号生成完成: 买入信号={len(df[df['action']=='buy'])}, 卖出信号={len(df[df['action']=='sell'])}")
            return df
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {str(e)}")
            return df


    def determine_market_condition(self, df: pd.DataFrame) -> str:
        """
        确定市场状况
        """
        if df.empty:
            return 'unknown'
            
        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # 检查中枢状况
            if 'central_bank' in df.columns and latest.get('central_bank', False):
                central_high = latest.get('central_bank_high', current_price)
                central_low = latest.get('central_bank_low', current_price)
                
                if not np.isnan(central_high) and not np.isnan(central_low) and central_low > 0:
                    if current_price > central_high:
                        return 'breakout_up'
                    elif current_price < central_low:
                        return 'breakout_down'
                    else:
                        return 'ranging' if (central_high - central_low) / central_low < self.ranging_threshold else 'trending'
            
            # 检查线段方向
            if latest.get('segment_end', False):
                seg_type = latest.get('segment_type')
                return 'trending_up' if seg_type == 'up' else 'trending_down'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"市场状况判断失败: {str(e)}")
            return 'unknown'


    def calculate_target_price(self, df: pd.DataFrame, action: str) -> float:
        """
        计算目标价格
        """
        if df.empty:
            return 0.0
            
        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            
            if action == "buy":
                if 'central_bank_high' in df.columns and not pd.isna(latest.get('central_bank_high', np.nan)):
                    return max(current_price * 1.02, latest['central_bank_high'])
                return current_price * 1.02
            elif action == "sell":
                if 'central_bank_low' in df.columns and not pd.isna(latest.get('central_bank_low', np.nan)):
                    return min(current_price * 0.98, latest['central_bank_low'])
                return current_price * 0.98
            else:
                return current_price
            
        except Exception as e:
            logger.error(f"计算目标价格失败: {str(e)}")
            return df.iloc[-1]['close'] if not df.empty else 0.0


    def calculate_stoploss(self, df: pd.DataFrame, action: str) -> float:
        """
        计算止损价格
        """
        if df.empty:
            return 0.0
            
        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            
            if self.stop_loss_type == 'dynamic':
                return self.calculate_dynamic_stoploss(df, action)
            else:
                if action == "buy":
                    if 'central_bank_low' in df.columns and not pd.isna(latest.get('central_bank_low', np.nan)):
                        return min(current_price * (1 - self.stop_loss_default), latest['central_bank_low'])
                    return current_price * (1 - self.stop_loss_default)
                elif action == "sell":
                    if 'central_bank_high' in df.columns and not pd.isna(latest.get('central_bank_high', np.nan)):
                        return max(current_price * (1 + self.stop_loss_default), latest['central_bank_high'])
                    return current_price * (1 + self.stop_loss_default)
                else:
                    return current_price * (1 - self.stop_loss_default)
                    
        except Exception as e:
            logger.error(f"计算止损价格失败: {str(e)}")
            return df.iloc[-1]['close'] * (1 - self.stop_loss_default) if not df.empty else 0.0


    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000, timeframe: str = 'daily') -> Dict:
        """
        回测方法
        """
        logger.info(f"开始{timeframe}级别回测 - 初始资金: {initial_capital:,}")
        
        # 数据验证
        is_valid, error_msg = self._validate_input_data(df, timeframe)
        if not is_valid:
            logger.error(f"回测数据验证失败: {error_msg}")
            return self._create_error_result(initial_capital, error_msg)
        
        try:
            # 计算缠论指标
            result_df = self.calculate(df, timeframe)
            
            # 生成交易信号
            signal_df = self.generate_signals(result_df)
            
            portfolio_value, cash, trades = initial_capital, initial_capital, []
            positions, portfolio_values = {}, []
            
            for i in range(len(signal_df)):
                current_row = signal_df.iloc[i]
                current_price = current_row['close']
                action = current_row.get('action', 'hold')
                
                # 计算组合价值
                portfolio_value = cash + sum([shares * current_price for shares in positions.values()]) if positions else cash
                portfolio_values.append(portfolio_value)
                
                # 执行交易
                if action == 'buy' and cash > 0:
                    position_size = 0.1  # 10%仓位
                    buy_amount = cash * position_size
                    shares_to_buy = int(buy_amount / current_price) if current_price > 0 else 0
                    
                    if shares_to_buy > 0:
                        symbol = current_row.get('symbol', 'default')
                        positions[symbol] = positions.get(symbol, 0) + shares_to_buy
                        cash -= shares_to_buy * current_price
                        
                        trades.append({
                            'date': current_row.get('date', i), 
                            'action': 'buy', 
                            'price': current_price,
                            'shares': shares_to_buy, 
                            'value': shares_to_buy * current_price,
                            'portfolio_value': portfolio_value
                        })
                
                elif action == 'sell' and positions:
                    total_sell_value = 0
                    for symbol, shares in positions.items():
                        if shares > 0 and current_price > 0:
                            sell_value = shares * current_price
                            total_sell_value += sell_value
                            trades.append({
                                'date': current_row.get('date', i), 
                                'action': 'sell', 
                                'price': current_price,
                                'shares': shares, 
                                'value': sell_value, 
                                'portfolio_value': portfolio_value
                            })
                    
                    cash += total_sell_value
                    positions = {}
            
            # 计算回测指标
            final_value = portfolio_values[-1] if portfolio_values else initial_capital
            total_return = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0
            
            # 最大回撤
            max_drawdown, peak = 0, initial_capital
            for value in portfolio_values:
                if value > peak: 
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                if drawdown > max_drawdown: 
                    max_drawdown = drawdown
            
            # 夏普比率
            returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                    returns.append(ret)
            
            sharpe_ratio = np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0
            
            # 胜率
            profitable_trades, sell_trades = 0, [t for t in trades if t['action'] == 'sell']
            if sell_trades:
                for trade in sell_trades:
                    buy_trades = [t for t in trades if t['action'] == 'buy' and t.get('date', 0) <= trade.get('date', 0)]
                    if buy_trades and sum([t['shares'] for t in buy_trades]) > 0:
                        total_value = sum([t['value'] for t in buy_trades])
                        total_shares = sum([t['shares'] for t in buy_trades])
                        if total_shares > 0 and trade['price'] > (total_value / total_shares):
                            profitable_trades += 1
            
            win_rate = profitable_trades / len(sell_trades) if sell_trades else 0
            
            result = {
                'initial_capital': initial_capital, 
                'final_value': final_value,
                'return_percent': total_return * 100, 
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio, 
                'win_rate': win_rate * 100,
                'total_trades': len(trades), 
                'profitable_trades': profitable_trades,
                'trades': trades, 
                'portfolio_values': portfolio_values,
                'timeframe': timeframe, 
                'data_points': len(df),
                'success': True
            }
            
            logger.info(f"回测完成: 初始资金={initial_capital:,}, 最终价值={final_value:,.2f}, 回报率={total_return*100:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"回测执行异常: {str(e)}")
            return self._create_error_result(initial_capital, f"回测异常: {str(e)}")


    def _create_error_result(self, initial_capital: float, error_msg: str) -> Dict:
        """
        创建错误结果
        """
        return {
            'success': False,
            'error': error_msg,
            'initial_capital': initial_capital,
            'final_value': initial_capital,
            'return_percent': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profitable_trades': 0,
            'trades': [],
            'portfolio_values': [initial_capital],
            'data_points': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# 测试代码
if __name__ == "__main__":
    # 创建测试配置
    test_config = {
        'chanlun': {
            'weekly_fractal_sensitivity': 2,
            'weekly_pen_min_length': 3,
            'weekly_central_min_length': 3
        },
        'fractal_sensitivity': 3,
        'pen_min_length': 5,
        'stop_loss_default': 0.03
    }
    
    # 初始化计算器
    calculator = ChanlunCalculator(config=test_config)
    logger.info("计算器初始化完成")
    
    # 生成测试数据
    test_data = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=100, freq='D'),
        'open': np.random.rand(100) * 100 + 100,
        'high': np.random.rand(100) * 10 + 110,
        'low': np.random.rand(100) * 10 + 90,
        'close': np.random.rand(100) * 10 + 100,
        'volume': np.random.rand(100) * 1000000
    })
    
    # 测试日线级别计算
    print("=== 测试日线级别计算 ===")
    daily_result = calculator.calculate(test_data, 'daily')
    print(f"日线级别 - 顶分型数量: {daily_result['top_fractal'].sum()}")
    print(f"日线级别 - 笔数量: {daily_result['pen_start'].sum()}")
    
    # 测试周线级别计算
    print("\n=== 测试周线级别计算 ===")
    weekly_result = calculator.calculate(test_data, 'weekly')
    print(f"周线级别 - 顶分型数量: {weekly_result['top_fractal'].sum()}")
    print(f"周线级别 - 笔数量: {weekly_result['pen_start'].sum()}")
    
    # 测试信号生成
    print("\n=== 测试信号生成 ===")
    signals = calculator.generate_signals(test_data)
    print(f"买入信号数量: {len(signals[signals['action']=='buy'])}")
    print(f"卖出信号数量: {len(signals[signals['action']=='sell'])}")
    
    # 测试回测功能
    print("\n=== 测试回测功能 ===")
    backtest_result = calculator.backtest(test_data, 100000, 'daily')
    print(f"回测结果: 最终价值={backtest_result['final_value']:.2f}, 回报率={backtest_result['return_percent']:.2f}%")