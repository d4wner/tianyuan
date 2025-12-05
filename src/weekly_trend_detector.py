#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周线多头趋势判定模块
支持梯度置信度评估的周线趋势分析
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

# 设置日志
logger = logging.getLogger(__name__)


class WeeklyTrendDetector:
    """
    周线多头趋势检测器
    实现基于缠论的周线多头趋势梯度判定
    """
    
    def __init__(self):
        """
        初始化周线趋势检测器
        """
        # 配置参数
        self.LOOKBACK_PERIOD = 10  # 回溯周期
        self.MIN_RAISING_BARS_LOWER = 6  # 最低抬升K线数量
        self.MIN_RAISING_BARS_HIGH = 8  # 高置信度抬升K线数量
        self.MACD_ZERO_TOLERANCE = 0.02  # MACD零轴附近容忍度
        self.MACD_NEGATIVE_THRESHOLD = -0.05  # MACD负面阈值（排除条件）
        self.MAX_CONSECUTIVE_RED_SHRINK_LOW = 4  # 最大连续红柱缩小（中置信度）
        self.MAX_CONSECUTIVE_RED_SHRINK_HIGH = 3  # 最大连续红柱缩小（高置信度）
        self.MACD_FAST = 12  # MACD快线周期
        self.MACD_SLOW = 26  # MACD慢线周期
        self.MACD_SIGNAL = 9  # MACD信号线周期
        # 置信度等级配置
        self.confidence_thresholds = {
            "HIGH": 0.85,
            "MEDIUM_HIGH": 0.70,
            "MEDIUM": 0.55,
            "MEDIUM_LOW": 0.40,
            "LOW": 0.25,
            "UNCERTAIN": 0.10,
            "BEARISH": 0.00
        }
        # 评分权重配置
        self.weights = {
            "price_trend": 0.4,
            "macd": 0.3,
            "hist_stability": 0.3
        }
        # 过滤开关配置
        self.filter_enabled = {
            "macd_divergence": True,  # 背驰过滤开关（默认开启）
            "fractal": True  # 分型过滤开关（默认开启）
        }
        # 背驰过滤条件参数
        self.divergence_filter_params = {
            "central_border_tolerance": 0.05,  # 中枢边缘容忍度5%
            "volatility_thresholds": {  # 不同波动等级的成交量阈值
                "low": 0.8,  # 低波动：成交量≥近10周均值的80%
                "medium": 1.0,  # 中波动：成交量≥近10周均值
                "high": 1.2  # 高波动：成交量≥近10周均值的120%
            },
            "no_new_low_days": 3  # 3日未创新低
        }
        logger.info("周线多头趋势检测器初始化完成")
        logger.info(f"配置参数: 回溯周期={self.LOOKBACK_PERIOD}根周K")
        logger.info(f"过滤开关状态: 背驰过滤={self.filter_enabled['macd_divergence']}, 分型过滤={self.filter_enabled['fractal']}")
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 周线数据框
            
        Returns:
            添加了MACD指标的数据框
        """
        df = df.copy()
        
        # 计算EMA
        ema_fast = df['close'].ewm(span=self.MACD_FAST, adjust=False, min_periods=self.MACD_FAST).mean()
        ema_slow = df['close'].ewm(span=self.MACD_SLOW, adjust=False, min_periods=self.MACD_SLOW).mean()
        
        # 计算DIF线
        df['macd_diff'] = ema_fast - ema_slow
        
        # 计算DEA线（信号线）
        df['macd_dea'] = df['macd_diff'].ewm(span=self.MACD_SIGNAL, adjust=False, min_periods=self.MACD_SIGNAL).mean()
        
        # 计算MACD柱状图
        df['macd_hist'] = df['macd_diff'] - df['macd_dea']
        
        return df
    
    def _detect_central_border(self, df: pd.DataFrame) -> Tuple[float, float]:
        """检测有效中枢边缘（简化实现）
        
        Args:
            df: 周线数据框
            
        Returns:
            (中枢上沿, 中枢下沿)
        """
        recent_data = df.tail(20).copy()
        if len(recent_data) < 10:
            return 0, 0
        
        # 简化的中枢检测：使用最近10根K线的高低点范围作为中枢
        central_high = recent_data['high'].tail(10).max()
        central_low = recent_data['low'].tail(10).min()
        
        return central_high, central_low
    
    def _get_volatility_level(self, df: pd.DataFrame) -> str:
        """计算波动等级
        
        Args:
            df: 周线数据框
            
        Returns:
            波动等级（low/medium/high）
        """
        recent_data = df.tail(10).copy()
        if len(recent_data) < 5:
            return "medium"
        
        # 计算近10周的波动率（高低点差值百分比）
        volatility = ((recent_data['high'] - recent_data['low']) / recent_data['close'].shift(1)).mean()
        
        if volatility < 0.05:
            return "low"
        elif volatility < 0.10:
            return "medium"
        else:
            return "high"
    
    def _check_volume_threshold(self, df: pd.DataFrame, volatility_level: str) -> bool:
        """检查成交量是否满足波动等级对应的阈值
        
        Args:
            df: 周线数据框
            volatility_level: 波动等级
            
        Returns:
            是否满足成交量阈值
        """
        recent_data = df.tail(10).copy()
        if len(recent_data) < 10:
            return True  # 数据不足时默认满足
        
        # 计算近10周平均成交量
        avg_volume = recent_data['volume'].mean()
        current_volume = recent_data['volume'].iloc[-1]
        
        # 获取当前波动等级的成交量阈值
        threshold = self.divergence_filter_params['volatility_thresholds'][volatility_level]
        
        return current_volume >= avg_volume * threshold
    
    def _check_no_new_low(self, df: pd.DataFrame) -> bool:
        """检查是否满足3日未创新低
        
        Args:
            df: 周线数据框
            
        Returns:
            是否3日未创新低
        """
        recent_data = df.tail(5).copy()
        if len(recent_data) < 4:
            return True  # 数据不足时默认满足
        
        # 检查最近3天是否有创新低
        recent_low = recent_data['low'].iloc[-1]
        for i in range(1, self.divergence_filter_params['no_new_low_days'] + 1):
            if i >= len(recent_data):
                break
            if recent_data['low'].iloc[-i-1] < recent_low:
                return False
        
        return True
    
    def calc_weekly_macd_divergence_confidence(self, df: pd.DataFrame) -> Dict:
        """计算周线MACD顶底背驰置信度（新增功能，自动生效）
        
        Args:
            df: 周线数据框
            
        Returns:
            包含背驰类型和置信度分值的字典
        """
        logger.info("计算周线MACD顶底背驰置信度...")
        
        # 获取最近的K线数据
        recent_data = df.tail(10).copy()
        if len(recent_data) < 5:
            logger.warning("周线数据不足，无法计算MACD背驰")
            return {
                "weekly_macd_divergence_type": "无",
                "weekly_macd_divergence_confidence": 0
            }
        
        # 计算MACD指标（如果还没有）
        if 'macd_diff' not in df.columns:
            df = self._calculate_macd(df)
            recent_data = df.tail(10).copy()
        
        # 检查底背驰：价格新低 + MACD黄白线未新低 + 绿柱缩短≥30%
        price_lows = recent_data['low'].values
        macd_diffs = recent_data['macd_diff'].values
        macd_hists = recent_data['macd_hist'].values
        
        # 寻找价格和MACD的低点
        recent_price_low_idx = np.argmin(price_lows[-5:]) + 5
        recent_macd_diff_low_idx = np.argmin(macd_diffs[-5:]) + 5
        recent_macd_hist_low_idx = np.argmin(macd_hists[-5:]) + 5
        
        # 计算底背驰条件
        has_price_low = price_lows[recent_price_low_idx] < np.min(price_lows[:-5])
        has_macd_diff_higher = macd_diffs[recent_macd_diff_low_idx] > np.min(macd_diffs[:-5])
        
        # 计算绿柱缩短比例
        if macd_hists[-1] < 0 and macd_hists[-2] < 0:
            green_hist_shrink_ratio = abs(macd_hists[-1]) / abs(macd_hists[-2]) if abs(macd_hists[-2]) != 0 else 1
            has_green_hist_shrunk = green_hist_shrink_ratio <= 0.7  # 缩短≥30%
        else:
            has_green_hist_shrunk = False
        
        if has_price_low and has_macd_diff_higher and has_green_hist_shrunk:
            # 应用背驰过滤条件（如果开关开启）
            if self.filter_enabled['macd_divergence']:
                logger.info("应用背驰过滤条件...")
                
                # 1. 检查是否在有效中枢边缘
                central_high, central_low = self._detect_central_border(df)
                current_price = recent_data['close'].iloc[-1]
                tolerance = self.divergence_filter_params['central_border_tolerance']
                in_central_border = (central_low * (1 - tolerance) <= current_price <= central_high * (1 + tolerance))
                
                # 2. 检查成交量是否满足波动等级阈值
                volatility_level = self._get_volatility_level(df)
                volume_ok = self._check_volume_threshold(df, volatility_level)
                
                # 3. 检查3日未创新低
                no_new_low = self._check_no_new_low(df)
                
                logger.info(f"背驰过滤条件检查结果：中枢边缘={in_central_border}, 成交量阈值={volume_ok}, 3日未创新低={no_new_low}")
                
                # 只有所有过滤条件都满足才确认背驰
                if not (in_central_border and volume_ok and no_new_low):
                    logger.info("背驰过滤条件不满足，忽略背驰信号")
                    return {
                        "weekly_macd_divergence_type": "无",
                        "weekly_macd_divergence_confidence": 0
                    }
            
            logger.info("检测到底背驰，置信度加分+15")
            return {
                "weekly_macd_divergence_type": "底背驰",
                "weekly_macd_divergence_confidence": 15
            }
        
        # 检查顶背驰：价格新高 + MACD黄白线未新高 + 红柱缩短≥30%
        price_highs = recent_data['high'].values
        
        # 寻找价格和MACD的高点
        recent_price_high_idx = np.argmax(price_highs[-5:]) + 5
        recent_macd_diff_high_idx = np.argmax(macd_diffs[-5:]) + 5
        recent_macd_hist_high_idx = np.argmax(macd_hists[-5:]) + 5
        
        # 计算顶背驰条件
        has_price_high = price_highs[recent_price_high_idx] > np.max(price_highs[:-5])
        has_macd_diff_lower = macd_diffs[recent_macd_diff_high_idx] < np.max(macd_diffs[:-5])
        
        # 计算红柱缩短比例
        if macd_hists[-1] > 0 and macd_hists[-2] > 0:
            red_hist_shrink_ratio = macd_hists[-1] / macd_hists[-2] if macd_hists[-2] != 0 else 1
            has_red_hist_shrunk = red_hist_shrink_ratio <= 0.7  # 缩短≥30%
        else:
            has_red_hist_shrunk = False
        
        if has_price_high and has_macd_diff_lower and has_red_hist_shrunk:
            # 应用背驰过滤条件（如果开关开启）
            if self.filter_enabled['macd_divergence']:
                logger.info("应用背驰过滤条件...")
                
                # 1. 检查是否在有效中枢边缘
                central_high, central_low = self._detect_central_border(df)
                current_price = recent_data['close'].iloc[-1]
                tolerance = self.divergence_filter_params['central_border_tolerance']
                in_central_border = (central_low * (1 - tolerance) <= current_price <= central_high * (1 + tolerance))
                
                # 2. 检查成交量是否满足波动等级阈值
                volatility_level = self._get_volatility_level(df)
                volume_ok = self._check_volume_threshold(df, volatility_level)
                
                logger.info(f"背驰过滤条件检查结果：中枢边缘={in_central_border}, 成交量阈值={volume_ok}")
                
                # 顶背驰只需检查中枢边缘和成交量
                if not (in_central_border and volume_ok):
                    logger.info("背驰过滤条件不满足，忽略背驰信号")
                    return {
                        "weekly_macd_divergence_type": "无",
                        "weekly_macd_divergence_confidence": 0
                    }
            
            logger.info("检测到顶背驰，置信度减分-20")
            return {
                "weekly_macd_divergence_type": "顶背驰",
                "weekly_macd_divergence_confidence": -20
            }
        
        # 无背驰
        logger.info("未检测到MACD背驰，置信度加权0")
        return {
            "weekly_macd_divergence_type": "无",
            "weekly_macd_divergence_confidence": 0
        }
    
    def _calculate_candle_body_ratio(self, candle: pd.Series) -> float:
        """计算K线实体占比
        
        Args:
            candle: 单根K线数据
            
        Returns:
            实体占比（实体长度/整个K线长度）
        """
        body_length = abs(candle['close'] - candle['open'])
        candle_length = candle['high'] - candle['low']
        
        if candle_length == 0:
            return 0
            
        return body_length / candle_length
    
    def _get_avg_body_ratio(self, df: pd.DataFrame) -> float:
        """计算近10周的平均实体占比
        
        Args:
            df: 周线数据框
            
        Returns:
            近10周的平均实体占比
        """
        recent_data = df.tail(10).copy()
        if len(recent_data) < 5:
            return 0.3  # 默认值
        
        # 计算每根K线的实体占比
        body_ratios = []
        for _, candle in recent_data.iterrows():
            body_ratio = self._calculate_candle_body_ratio(candle)
            body_ratios.append(body_ratio)
            
        return np.mean(body_ratios)
    
    def _check_right_candle_change(self, right_candle: pd.Series, prev_candle: pd.Series) -> bool:
        """检查右侧K线的涨跌幅是否≥2%
        
        Args:
            right_candle: 右侧K线数据
            prev_candle: 前一根K线数据（分型的中间K线）
            
        Returns:
            是否满足涨跌幅≥2%
        """
        change_ratio = (right_candle['close'] - prev_candle['close']) / prev_candle['close']
        
        # 底分型右侧K线需要涨幅≥2%，顶分型右侧K线需要跌幅≥2%
        return abs(change_ratio) >= 0.02
    
    def calc_weekly_fractal_confidence(self, df: pd.DataFrame) -> Dict:
        """计算周线顶底分型置信度（新增功能，自动生效）
        
        Args:
            df: 周线数据框
            
        Returns:
            包含分型类型和置信度分值的字典
        """
        logger.info("计算周线顶底分型置信度...")
        
        # 获取最近的K线数据
        recent_data = df.tail(5).copy()
        if len(recent_data) < 3:
            logger.warning("周线数据不足，无法计算顶底分型")
            return {
                "weekly_fractal_type": "无",
                "weekly_fractal_confidence": 0
            }
        
        # 获取最近3根K线
        last3 = recent_data.tail(3)
        highs = last3['high'].values
        lows = last3['low'].values
        closes = last3['close'].values
        opens = last3['open'].values
        
        # 检查底分型：左阴 + 中低 + 右阳，中低为三周最低
        is_left_dark = closes[0] < opens[0]
        is_right_bright = closes[2] > opens[2]
        is_middle_lowest = lows[1] < min(lows[0], lows[2])
        is_middle_lowest_3w = lows[1] < min(df['low'].tail(5).values)  # 中低为五周最低
        
        if is_left_dark and is_right_bright and is_middle_lowest and is_middle_lowest_3w:
            # 应用分型过滤条件（如果开关开启）
            if self.filter_enabled['fractal']:
                logger.info("应用分型过滤条件...")
                
                # 获取中间K线数据
                middle_candle = last3.iloc[1]
                right_candle = last3.iloc[2]
                
                # 1. 检查实体占比是否≥近10周均值60%
                middle_body_ratio = self._calculate_candle_body_ratio(middle_candle)
                avg_body_ratio = self._get_avg_body_ratio(df)
                body_ratio_ok = middle_body_ratio >= avg_body_ratio * 0.6
                
                # 2. 检查右侧K线涨跌幅是否≥2%
                right_change_ok = self._check_right_candle_change(right_candle, middle_candle)
                
                logger.info(f"底分型过滤条件检查结果：实体占比={body_ratio_ok}({middle_body_ratio:.2f}≥{avg_body_ratio*0.6:.2f}), 右侧涨跌幅={right_change_ok}")
                
                # 只有所有过滤条件都满足才确认底分型
                if not (body_ratio_ok and right_change_ok):
                    logger.info("底分型过滤条件不满足，忽略分型信号")
                    return {
                        "weekly_fractal_type": "无",
                        "weekly_fractal_confidence": 0
                    }
            
            logger.info("检测到底分型，置信度加分+10")
            return {
                "weekly_fractal_type": "底分型",
                "weekly_fractal_confidence": 10
            }
        
        # 检查顶分型：左阳 + 中高 + 右阴，中高为三周最高
        is_left_bright = closes[0] > opens[0]
        is_right_dark = closes[2] < opens[2]
        is_middle_highest = highs[1] > max(highs[0], highs[2])
        is_middle_highest_3w = highs[1] > max(df['high'].tail(5).values)  # 中高为五周最高
        
        if is_left_bright and is_right_dark and is_middle_highest and is_middle_highest_3w:
            # 应用分型过滤条件（如果开关开启）
            if self.filter_enabled['fractal']:
                logger.info("应用分型过滤条件...")
                
                # 获取中间K线数据
                middle_candle = last3.iloc[1]
                right_candle = last3.iloc[2]
                
                # 1. 检查实体占比是否≥近10周均值60%
                middle_body_ratio = self._calculate_candle_body_ratio(middle_candle)
                avg_body_ratio = self._get_avg_body_ratio(df)
                body_ratio_ok = middle_body_ratio >= avg_body_ratio * 0.6
                
                # 2. 检查右侧K线涨跌幅是否≥2%
                right_change_ok = self._check_right_candle_change(right_candle, middle_candle)
                
                logger.info(f"顶分型过滤条件检查结果：实体占比={body_ratio_ok}({middle_body_ratio:.2f}≥{avg_body_ratio*0.6:.2f}), 右侧涨跌幅={right_change_ok}")
                
                # 只有所有过滤条件都满足才确认顶分型
                if not (body_ratio_ok and right_change_ok):
                    logger.info("顶分型过滤条件不满足，忽略分型信号")
                    return {
                        "weekly_fractal_type": "无",
                        "weekly_fractal_confidence": 0
                    }
            
            logger.info("检测到顶分型，置信度减分-15")
            return {
                "weekly_fractal_type": "顶分型",
                "weekly_fractal_confidence": -15
            }
        
        # 无分型
        logger.info("未检测到顶底分型，置信度加权0")
        return {
            "weekly_fractal_type": "无",
            "weekly_fractal_confidence": 0
        }
    
    def _check_price_rising(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """检查近10根周K线收盘价趋势
        
        Args:
            df: 周线数据框
            
        Returns:
            (是否满足底线条件, 详细信息)
        """
        logger.info(f"检查近{self.LOOKBACK_PERIOD}根周K线收盘价趋势...")
        
        # 获取最近LOOKBACK_PERIOD根周K线
        recent_data = df.tail(self.LOOKBACK_PERIOD).copy()
        if len(recent_data) < self.LOOKBACK_PERIOD:
            logger.warning(f"周K线数据不足{self.LOOKBACK_PERIOD}根，当前只有{len(recent_data)}根")
            return False, {
                "sufficient_data": False,
            "rising_weeks_count": 0,
            "rising_pairs": 0,
            "total_pairs": 0,
            "rising_percentage": 0,
            "reason": f"数据不足{self.LOOKBACK_PERIOD}根"
            }
        
        # 计算上升的相邻周数
        rising_pairs = 0
        total_pairs = self.LOOKBACK_PERIOD - 1
        
        # 计算上升的周数（与前一根相比收盘价上升的周数）
        rising_weeks_count = 0
        
        # 记录上升情况
        weekly_changes = []
        for i in range(1, len(recent_data)):
            prev_close = recent_data.iloc[i-1]['close']
            curr_close = recent_data.iloc[i]['close']
            is_rising = curr_close > prev_close
            
            weekly_changes.append({
                "date": recent_data.iloc[i]['date'],
                "prev_close": prev_close,
                "curr_close": curr_close,
                "change_pct": (curr_close - prev_close) / prev_close * 100,
                "is_rising": is_rising
            })
            
            if is_rising:
                rising_pairs += 1
                rising_weeks_count += 1
        
        # 计算上升百分比
        rising_percentage = (rising_pairs / total_pairs) * 100 if total_pairs > 0 else 0
        
        # 计算线性回归斜率
        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        slope = np.polyfit(x, y, 1)[0]
        has_positive_trend = slope > 0
        
        # 计算趋势强度
        trend_strength = (slope / recent_data['close'].mean()) * 100 if recent_data['close'].mean() > 0 else 0
        
        # 新条件：近10根周K线中≥6根收盘价呈抬升趋势
        # 这里rising_weeks_count是与前一周相比上升的周数，总共有9对相邻周
        # 我们需要调整为计算有多少根周K线的收盘价高于某个基准
        # 这里计算高于最近的低点的周数
        recent_lows = recent_data['low'].rolling(window=5, min_periods=1).min()
        weeks_above_recent_low = 0
        for i in range(1, len(recent_data)):
            if recent_data.iloc[i]['close'] > recent_lows.iloc[i-1]:
                weeks_above_recent_low += 1
        
        # 底线条件：近10根周K线中≥6根收盘价呈抬升趋势
        # 综合考虑：1)与前一周相比上升的周数 2)整体趋势为正 3)高于近期低点的周数
        is_price_rising = rising_weeks_count >= 6 and has_positive_trend and weeks_above_recent_low >= 6
        
        logger.info(f"价格趋势检查结果: {'满足' if is_price_rising else '不满足'} 底线条件")
        logger.info(f"与前一周相比上升的周数: {rising_weeks_count}周")
        logger.info(f"高于近期低点的周数: {weeks_above_recent_low}周")
        logger.info(f"整体趋势: {'上升' if has_positive_trend else '下降'}，趋势强度: {trend_strength:.4f}%")
        
        # 计算价格趋势评分（0-1之间）
        score = 0
        score += min(1.0, rising_weeks_count / 8) * 0.3  # 上升周数评分（最高8周得满分）
        score += min(1.0, trend_strength / 5) * 0.3     # 趋势强度评分（最高5%得满分）
        score += min(1.0, weeks_above_recent_low / 9) * 0.2  # 高于低点周数评分
        score += min(1.0, rising_percentage / 70) * 0.2  # 上升比例评分
        
        return is_price_rising, {
            "sufficient_data": True,
            "rising_weeks_count": rising_weeks_count,
            "rising_pairs": rising_pairs,
            "total_pairs": total_pairs,
            "rising_percentage": rising_percentage,
            "has_positive_trend": has_positive_trend,
            "trend_strength": trend_strength,
            "weeks_above_recent_low": weeks_above_recent_low,
            "weekly_changes": weekly_changes,
            "score": score,
            "reason": f"{'满足' if is_price_rising else '不满足'}近{self.LOOKBACK_PERIOD}根周K线≥6根收盘价呈抬升趋势条件"
        }
    
    def _check_macd_above_zero(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """检查MACD黄白线是否在零轴上方或附近
        
        Args:
            df: 包含MACD指标的周线数据框
            
        Returns:
            (是否满足底线条件, 详细信息)
        """
        logger.info("检查MACD黄白线是否在零轴上方或附近...")
        
        # 获取最近数据
        recent_data = df.tail(self.LOOKBACK_PERIOD).copy()
        if len(recent_data) < self.LOOKBACK_PERIOD:
            logger.warning(f"周K线数据不足{self.LOOKBACK_PERIOD}根，当前只有{len(recent_data)}根")
            return False, {
                "sufficient_data": False,
                "diff_above_zero_count": 0,
                "dea_above_zero_count": 0,
                "both_near_zero_count": 0,
                "current_diff": None,
                "current_dea": None,
                "reason": f"数据不足{self.LOOKBACK_PERIOD}根"
            }
        
        # 计算MACD指标
        if 'macd_diff' not in recent_data.columns or 'macd_dea' not in recent_data.columns:
            recent_data = self._calculate_macd(recent_data)
        
        # 零轴附近阈值
        zero_threshold = 0.02
        
        # 统计黄白线在零轴上方或附近的次数
        diff_above_zero = recent_data['macd_diff'] > 0
        dea_above_zero = recent_data['macd_dea'] > 0
        diff_near_zero = abs(recent_data['macd_diff']) <= zero_threshold
        dea_near_zero = abs(recent_data['macd_dea']) <= zero_threshold
        both_above_or_near_zero = (diff_above_zero | diff_near_zero) & (dea_above_zero | dea_near_zero)
        
        diff_above_zero_count = diff_above_zero.sum()
        dea_above_zero_count = dea_above_zero.sum()
        both_near_zero_count = both_above_or_near_zero.sum()
        
        # 获取最新的MACD值
        current_diff = recent_data.iloc[-1]['macd_diff']
        current_dea = recent_data.iloc[-1]['macd_dea']
        
        # 判断底线条件：
        # 最新的MACD黄白线在零轴上方或零轴附近±0.02区间
        is_above_or_near_zero = (current_diff > -zero_threshold and current_dea > -zero_threshold)
        
        logger.info(f"MACD黄白线检查结果: {'满足' if is_above_or_near_zero else '不满足'} 零轴上方或附近条件")
        logger.info(f"黄白线同时在零轴上方或附近的周数: {both_near_zero_count}/{self.LOOKBACK_PERIOD}")
        logger.info(f"最新MACD DIF: {current_diff:.4f}, DEA: {current_dea:.4f}")
        
        # 计算MACD金叉状态
        has_macd_gold_cross = False
        if len(df) >= 2:
            has_macd_gold_cross = (df['macd_diff'].iloc[-1] > df['macd_dea'].iloc[-1]) and \
                                 (df['macd_diff'].iloc[-2] <= df['macd_dea'].iloc[-2])
        
        # 计算MACD评分（0-1之间）
        score = 0
        both_above_zero_count = ((recent_data['macd_diff'] > 0) & (recent_data['macd_dea'] > 0)).sum()
        score += min(1.0, both_above_zero_count / 8) * 0.4  # 黄白线同时在零轴上方的周数
        score += min(1.0, diff_above_zero_count / 9) * 0.2  # DIF在零轴上方的周数
        score += min(1.0, current_diff / 0.1) * 0.15 if current_diff > 0 else 0  # 最新DIF强度
        score += 0.15 if has_macd_gold_cross else 0  # 金叉加分
        score += 0.1 if both_near_zero_count >= 8 else 0  # 零轴附近加分
        
        return is_above_or_near_zero, {
            "sufficient_data": True,
            "diff_above_zero_count": int(diff_above_zero_count),
            "dea_above_zero_count": int(dea_above_zero_count),
            "both_above_zero_count": int(both_above_zero_count),
            "both_near_zero_count": int(both_near_zero_count),
            "current_diff": current_diff,
            "current_dea": current_dea,
            "has_macd_gold_cross": has_macd_gold_cross,
            "score": score,
            "reason": f"{'满足' if is_above_or_near_zero else '不满足'}MACD黄白线在零轴上方或附近±0.02区间条件"
        }
    
    def _check_red_hist_not_decreasing(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """检查红柱未连续4根缩小
        
        Args:
            df: 包含MACD指标的周线数据框
            
        Returns:
            (是否满足底线条件, 详细信息)
        """
        logger.info("检查MACD红柱是否未连续4根缩小...")
        
        # 获取最近数据
        recent_data = df.tail(self.LOOKBACK_PERIOD).copy()
        if len(recent_data) < self.LOOKBACK_PERIOD:
            logger.warning(f"周K线数据不足{self.LOOKBACK_PERIOD}根，当前只有{len(recent_data)}根")
            return False, {
                "sufficient_data": False,
                "has_continuous_decrease": True,
                "max_continuous_decrease": 0,
                "current_hist": None,
                "reason": f"数据不足{self.LOOKBACK_PERIOD}根"
            }
        
        # 计算MACD指标
        if 'macd_hist' not in recent_data.columns:
            recent_data = self._calculate_macd(recent_data)
        
        # 分析红柱（正值）的连续性
        has_continuous_decrease = False
        max_continuous_decrease = 0
        current_continuous = 0
        red_hist_values = []
        
        for i in range(1, len(recent_data)):
            prev_hist = recent_data.iloc[i-1]['macd_hist']
            curr_hist = recent_data.iloc[i]['macd_hist']
            
            # 只关注红柱（正值）
            if prev_hist > 0 and curr_hist > 0:
                red_hist_values.append({
                    "date": recent_data.iloc[i]['date'],
                    "prev_hist": prev_hist,
                    "curr_hist": curr_hist,
                    "is_decreasing": curr_hist < prev_hist
                })
                
                if curr_hist < prev_hist:
                    current_continuous += 1
                    max_continuous_decrease = max(max_continuous_decrease, current_continuous)
                else:
                    current_continuous = 0
            else:
                current_continuous = 0  # 非红柱时重置计数
            
            # 检查是否出现连续4根红柱缩小
            if current_continuous >= 4:
                has_continuous_decrease = True
                break
        
        # 获取最新的红柱值
        current_hist = recent_data.iloc[-1]['macd_hist']
        
        # 判断条件：未出现连续4根红柱缩小
        is_not_decreasing = not has_continuous_decrease
        
        logger.info(f"红柱连续性检查结果: {'满足' if is_not_decreasing else '不满足'} 未连续4根缩小条件")
        logger.info(f"最大连续缩小周数: {max_continuous_decrease}")
        logger.info(f"最新MACD柱状图: {current_hist:.4f}")
        
        # 计算红柱周数和平均高度
        red_hist_weeks = (recent_data['macd_hist'] > 0).sum()
        red_hist_values_only = recent_data[recent_data['macd_hist'] > 0]['macd_hist']
        avg_red_hist = red_hist_values_only.mean() if len(red_hist_values_only) > 0 else 0
        
        # 计算红柱增长周数
        increasing_red_hist_weeks = 0
        for item in red_hist_values:
            if not item['is_decreasing']:
                increasing_red_hist_weeks += 1
        
        # 计算红柱稳定性评分（0-1之间）
        score = 0
        score += min(1.0, (10 - max_continuous_decrease) / 10) * 0.4  # 连续缩小越少越好
        score += min(1.0, red_hist_weeks / 9) * 0.2  # 红柱周数越多越好
        score += min(1.0, increasing_red_hist_weeks / 6) * 0.2  # 红柱增长周数
        score += min(1.0, current_hist / 0.1) * 0.1 if current_hist > 0 else 0  # 最新红柱高度
        score += min(1.0, avg_red_hist / 0.05) * 0.1 if avg_red_hist > 0 else 0  # 平均红柱高度
        
        return is_not_decreasing, {
            "sufficient_data": True,
            "has_continuous_decrease": has_continuous_decrease,
            "max_continuous_decrease": max_continuous_decrease,
            "current_hist": current_hist,
            "red_hist_values": red_hist_values,
            "red_hist_weeks": red_hist_weeks,
            "avg_red_hist": avg_red_hist,
            "increasing_red_hist_weeks": increasing_red_hist_weeks,
            "score": score,
            "reason": f"{'满足' if is_not_decreasing else '不满足'}红柱未连续4根缩小条件"
        }
    
    def _check_exclusion_rules(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """检查排除规则
        
        Args:
            df: 周线数据框
            
        Returns:
            (是否被排除, 排除原因)
        """
        # 获取最近LOOKBACK_PERIOD根周K线
        recent_data = df.tail(self.LOOKBACK_PERIOD).copy()
        
        # 规则1：检查周线收盘价是否创近10根周K新低
        recent_lows = recent_data['low']
        current_close = recent_data.iloc[-1]['close']
        is_new_low = current_close <= recent_lows.min()
        
        if is_new_low:
            reason = f"周线收盘价创近{self.LOOKBACK_PERIOD}根周K新低"
            logger.warning(f"触发排除规则: {reason}")
            return True, reason
        
        # 规则2：检查MACD黄白线是否跌破零轴下方0.05以上
        if 'macd_diff' not in recent_data.columns or 'macd_dea' not in recent_data.columns:
            recent_data = self._calculate_macd(recent_data)
        
        current_diff = recent_data.iloc[-1]['macd_diff']
        current_dea = recent_data.iloc[-1]['macd_dea']
        
        if current_diff < -0.05 and current_dea < -0.05:
            reason = "MACD黄白线跌破零轴下方0.05以上"
            logger.warning(f"触发排除规则: {reason}")
            return True, reason
        
        # 规则3：跌破20周均线
        if len(df) >= 20:
            df_with_ma = df.copy()
            df_with_ma['ma20'] = df_with_ma['close'].rolling(window=20).mean()
            current_price = df_with_ma['close'].iloc[-1]
            ma20 = df_with_ma['ma20'].iloc[-1]
            if current_price < ma20 * 0.98:  # 跌破20周均线2%
                reason = "跌破20周均线2%以上"
                logger.warning(f"触发排除规则: {reason}")
                return True, reason
        
        # 规则4：连续3根周K线收盘价下跌
        if len(df) >= 3:
            week1_close = df['close'].iloc[-3]
            week2_close = df['close'].iloc[-2]
            week3_close = df['close'].iloc[-1]
            if week3_close < week2_close and week2_close < week1_close:
                reason = "连续3根周K线收盘价下跌"
                logger.warning(f"触发排除规则: {reason}")
                return True, reason
        
        # 规则5：MACD柱状图持续为负
        if 'macd_hist' not in recent_data.columns:
            recent_data = self._calculate_macd(recent_data)
            
        if len(df) >= 6:
            recent_hist = df['macd_hist'].tail(6)
            if (recent_hist < 0).all():
                reason = "最近6根周K线MACD柱状图持续为负"
                logger.warning(f"触发排除规则: {reason}")
                return True, reason
        
        return False, ""
    
    def _determine_confidence_level(self, price_details: Dict, macd_details: Dict, hist_details: Dict) -> Tuple[str, float, float]:
        """确定置信度等级和仓位比例
        
        Args:
            price_details: 价格趋势详细信息
            macd_details: MACD详细信息
            hist_details: 柱状图详细信息
            
        Returns:
            (置信度等级, 二买30分钟子仓位比例, 一买/三买子仓位比例)
        """
        # 计算综合评分
        price_score = price_details.get('score', 0)
        macd_score = macd_details.get('score', 0)
        hist_score = hist_details.get('score', 0)
        
        # 应用权重计算加权总分
        weighted_score = (
            price_score * self.weights['price_trend'] +
            macd_score * self.weights['macd'] +
            hist_score * self.weights['hist_stability']
        )
        
        # 确定置信度等级
        if weighted_score >= self.confidence_thresholds['HIGH']:
            confidence_level = "HIGH"
            second_buy_ratio = 0.85  # 二买30分钟子仓位比例
            first_third_buy_ratio = 0.65  # 一买/三买子仓位比例
        elif weighted_score >= self.confidence_thresholds['MEDIUM_HIGH']:
            confidence_level = "MEDIUM_HIGH"
            second_buy_ratio = 0.70  # 二买30分钟子仓位比例
            first_third_buy_ratio = 0.50  # 一买/三买子仓位比例
        elif weighted_score >= self.confidence_thresholds['MEDIUM']:
            confidence_level = "MEDIUM"
            second_buy_ratio = 0.50  # 二买30分钟子仓位比例
            first_third_buy_ratio = 0.35  # 一买/三买子仓位比例
        elif weighted_score >= self.confidence_thresholds['MEDIUM_LOW']:
            confidence_level = "MEDIUM_LOW"
            second_buy_ratio = 0.30  # 二买30分钟子仓位比例
            first_third_buy_ratio = 0.20  # 一买/三买子仓位比例
        elif weighted_score >= self.confidence_thresholds['LOW']:
            confidence_level = "LOW"
            second_buy_ratio = 0.15  # 二买30分钟子仓位比例
            first_third_buy_ratio = 0.10  # 一买/三买子仓位比例
        elif weighted_score >= self.confidence_thresholds['UNCERTAIN']:
            confidence_level = "UNCERTAIN"
            second_buy_ratio = 0.05  # 极小仓位试探
            first_third_buy_ratio = 0.03  # 极小仓位试探
        else:
            confidence_level = "BEARISH"
            second_buy_ratio = 0.0  # 不建仓
            first_third_buy_ratio = 0.0  # 不建仓
        
        logger.info(f"综合评分: {weighted_score:.3f}, 置信度等级: {confidence_level}")
        
        return confidence_level, second_buy_ratio, first_third_buy_ratio
    
    def detect_weekly_bullish_trend(self, df: pd.DataFrame) -> Dict[str, any]:
        """检测周线多头趋势（梯度判定）
        
        Args:
            df: 周线数据框
            
        Returns:
            包含趋势检测结果的字典，带有置信度梯度
        """
        logger.info("开始检测周线多头趋势（梯度判定）...")
        
        # 初始化结果字典
        result = {
            "bullish_trend": False,
            "confidence": 0.0,
            "confidence_level": "UNCERTAIN",  # 初始化置信度等级
            "status": "pending",
            "reason": "",
            "satisfied_conditions_count": 0,
            "total_conditions_count": 3,
            "detailed_analysis": {},
            "position_ratios": {
                "second_buy_ratio": 0.0,
                "first_third_buy_ratio": 0.0
            },
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 检查数据框是否为空
        if df is None or df.empty:
            logger.error("周线数据为空")
            result["status"] = "数据为空"
            result["reason"] = "周线数据为空，无法进行趋势检测"
            result["detailed_analysis"] = {
                "price_condition": {"satisfied": False, "reason": "数据为空"},
                "macd_condition": {"satisfied": False, "reason": "数据为空"},
                "hist_condition": {"satisfied": False, "reason": "数据为空"}
            }
            return result
        
        # 确保数据包含必要的列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"周线数据缺少必要列: {missing_columns}")
            result["status"] = "数据不完整"
            result["reason"] = f"数据缺少必要列: {missing_columns}"
            result["detailed_analysis"] = {
                "price_condition": {"satisfied": False, "reason": f"缺少列: {missing_columns}"},
                "macd_condition": {"satisfied": False, "reason": f"缺少列: {missing_columns}"},
                "hist_condition": {"satisfied": False, "reason": f"缺少列: {missing_columns}"}
            }
            return result
        
        # 计算MACD指标（如果还没有）
        df_with_macd = self._calculate_macd(df)
        
        # 检查排除规则
        is_excluded, exclude_reason = self._check_exclusion_rules(df_with_macd)
        if is_excluded:
            result["status"] = "非多头（触发排除规则）"
            result["reason"] = exclude_reason
            result["detailed_analysis"] = {
                "exclusion_rule": {
                    "triggered": True,
                    "reason": exclude_reason
                }
            }
            return result
        
        # 检查三个底线条件
        price_result, price_details = self._check_price_rising(df_with_macd)
        macd_result, macd_details = self._check_macd_above_zero(df_with_macd)
        hist_result, hist_details = self._check_red_hist_not_decreasing(df_with_macd)
        
        # 计算满足的条件数
        satisfied_conditions = sum([price_result, macd_result, hist_result])
        result["satisfied_conditions_count"] = satisfied_conditions
        
        # 生成详细分析
        detailed_analysis = {
            "price_condition": {
                "satisfied": price_result,
                "details": price_details
            },
            "macd_condition": {
                "satisfied": macd_result,
                "details": macd_details
            },
            "hist_condition": {
                "satisfied": hist_result,
                "details": hist_details
            }
        }
        result["detailed_analysis"] = detailed_analysis
        
        # 确定置信度等级和仓位比例
        confidence_level, second_buy_ratio, first_third_buy_ratio = self._determine_confidence_level(
            price_details, macd_details, hist_details
        )
        
        # 计算综合置信度
        confidence = (satisfied_conditions / 3) * 100
        
        # 应用周线MACD顶底背驰和顶底分型置信度加权（新增功能，自动生效）
        macd_divergence_conf = self.calc_weekly_macd_divergence_confidence(df)
        fractal_conf = self.calc_weekly_fractal_confidence(df)
        
        # 计算总置信度加权分值
        total_confidence_weight = macd_divergence_conf["weekly_macd_divergence_confidence"] + fractal_conf["weekly_fractal_confidence"]
        
        # 将置信度加权应用到综合评分
        # 原基础评分范围是0-100，加权后可能超出范围，但不影响核心规则
        weighted_confidence = confidence + total_confidence_weight
        
        # 保留原有置信度作为基础评分
        result["base_confidence"] = confidence
        
        # 更新结果，添加置信度加权信息（兼容原数据结构）
        result["confidence_level"] = confidence_level
        result["confidence"] = weighted_confidence
        
        # 添加周线置信度加权详细信息（新增字段，不影响原有功能）
        result["weekly_confidence_details"] = {
            "base_confidence": confidence,
            "macd_divergence": macd_divergence_conf,
            "fractal": fractal_conf,
            "total_confidence_weight": total_confidence_weight,
            "weighted_confidence": weighted_confidence
        }
        
        # 添加评分信息到结果字典，方便直接访问
        result["component_scores"] = {
            "price_score": price_details.get('score', 0),
            "macd_score": macd_details.get('score', 0),
            "hist_score": hist_details.get('score', 0),
            "weighted_score": price_details.get('score', 0) * self.weights['price_trend'] + \
                            macd_details.get('score', 0) * self.weights['macd'] + \
                            hist_details.get('score', 0) * self.weights['hist_stability']
        }
        
        # 确定最终结果
        if confidence_level in ['HIGH', 'MEDIUM_HIGH', 'MEDIUM', 'MEDIUM_LOW', 'LOW']:
            result["bullish_trend"] = True
            result["status"] = f"周线多头（{confidence_level}）"
            result["reason"] = f"满足{satisfied_conditions}/3个多头底线条件，综合评分达到多头标准"
            result["position_ratios"] = {
                "second_buy_ratio": second_buy_ratio,
                "first_third_buy_ratio": first_third_buy_ratio
            }
        elif confidence_level == 'BEARISH':
            result["bullish_trend"] = False
            result["status"] = "非多头"
            result["reason"] = "周线趋势偏向空头，不符合多头条件"
        else:
            result["bullish_trend"] = False
            result["status"] = "非多头"
            result["reason"] = f"仅满足{satisfied_conditions}/3个多头底线条件，不符合周线多头定义"
        
        logger.info(f"周线多头趋势检测结果: {'多头' if result['bullish_trend'] else '非多头'}")
        logger.info(f"满足条件数: {satisfied_conditions}/3, 置信度: {confidence:.1f}%, 置信度等级: {confidence_level}")
        logger.info(f"状态: {result['status']}, 原因: {result['reason']}")
        if result['bullish_trend']:
            logger.info(f"推荐仓位比例 - 二买30分钟: {second_buy_ratio*100:.1f}%, 一买/三买: {first_third_buy_ratio*100:.1f}%")
        
        return result
    
    def generate_trend_report(self, result: Dict[str, any]) -> str:
        """生成趋势检测报告
        
        Args:
            result: 趋势检测结果
            
        Returns:
            格式化的报告字符串
        """
        report_lines = ["===== 周线多头趋势检测报告 ====="]
        
        # 总体结果
        report_lines.append(f"检测结果: {'✓ 周线多头' if result['bullish_trend'] else '✗ 非周线多头'}")
        report_lines.append(f"置信度: {result['confidence']:.1f}%")
        report_lines.append(f"置信度等级: {result['confidence_level']}")
        
        # 添加综合评分和组件评分（如果存在）
        price_score = 0
        macd_score = 0
        hist_score = 0
        
        if 'detailed_analysis' in result:
            if 'price_condition' in result['detailed_analysis'] and 'details' in result['detailed_analysis']['price_condition']:
                price_score = result['detailed_analysis']['price_condition']['details'].get('score', 0)
            if 'macd_condition' in result['detailed_analysis'] and 'details' in result['detailed_analysis']['macd_condition']:
                macd_score = result['detailed_analysis']['macd_condition']['details'].get('score', 0)
            if 'hist_condition' in result['detailed_analysis'] and 'details' in result['detailed_analysis']['hist_condition']:
                hist_score = result['detailed_analysis']['hist_condition']['details'].get('score', 0)
        
        # 计算加权总分
        weighted_score = price_score * self.weights['price_trend'] + \
                        macd_score * self.weights['macd'] + \
                        hist_score * self.weights['hist_stability']
        
        report_lines.append(f"综合评分: {weighted_score:.3f} (价格:{price_score:.3f} 权重0.4 | MACD:{macd_score:.3f} 权重0.3 | 红柱:{hist_score:.3f} 权重0.3)")
        report_lines.append(f"状态: {result['status']}")
        report_lines.append(f"原因: {result['reason']}")
        report_lines.append(f"满足条件: {result['satisfied_conditions_count']}/{result['total_conditions_count']}")
        report_lines.append(f"检测时间: {result['timestamp']}")
        
        # 添加周线置信度加权信息（新增，不影响原有格式）
        if 'weekly_confidence_details' in result:
            wc_details = result['weekly_confidence_details']
            report_lines.append("")
            report_lines.append("【周线置信度加权分析】")
            report_lines.append(f"基础置信度: {wc_details['base_confidence']:.1f}%")
            report_lines.append(f"MACD背驰类型: {wc_details['macd_divergence']['weekly_macd_divergence_type']}")
            report_lines.append(f"MACD背驰置信度加权: {wc_details['macd_divergence']['weekly_macd_divergence_confidence']:+d}%")
            report_lines.append(f"顶底分型类型: {wc_details['fractal']['weekly_fractal_type']}")
            report_lines.append(f"顶底分型置信度加权: {wc_details['fractal']['weekly_fractal_confidence']:+d}%")
            report_lines.append(f"总置信度加权: {wc_details['total_confidence_weight']:+d}%")
            report_lines.append(f"加权后置信度: {wc_details['weighted_confidence']:.1f}%")
        
        report_lines.append("")
        
        # 详细条件分析
        report_lines.append("【详细条件分析】")
        
        # 条件1：价格逐步抬升
        price = result['detailed_analysis']['price_condition']
        report_lines.append(f"1. 近{self.LOOKBACK_PERIOD}根周K线收盘价逐步抬升: {'✓ 通过' if price['satisfied'] else '✗ 未通过'}")
        if 'details' in price:
            details = price['details']
            if details.get('sufficient_data', False):
                report_lines.append(f"   - 上升周对数: {details['rising_pairs']}/{details['total_pairs']} ({details['rising_percentage']:.1f}%)")
                report_lines.append(f"   - 上升周数: {details['rising_weeks_count']}周")
                report_lines.append(f"   - 高于近期低点的周数: {details.get('weeks_above_recent_low', 0)}周")
                report_lines.append(f"   - 整体趋势: {'上升' if details['has_positive_trend'] else '下降'}")
                report_lines.append(f"   - 趋势强度: {details['trend_strength']:.4f}%")
                report_lines.append(f"   - 价格趋势评分: {details.get('score', 0):.3f}/1.0")
            else:
                report_lines.append(f"   - 原因: {details.get('reason', '数据不足')}")
        
        # 条件2：MACD黄白线在零轴上方或附近
        macd = result['detailed_analysis']['macd_condition']
        report_lines.append(f"2. MACD黄白线在零轴上方或附近: {'✓ 通过' if macd['satisfied'] else '✗ 未通过'}")
        if 'details' in macd:
            details = macd['details']
            if details.get('sufficient_data', False):
                report_lines.append(f"   - DIF在零轴上方周数: {details.get('diff_above_zero_count', 0)}周")
                report_lines.append(f"   - DEA在零轴上方周数: {details.get('dea_above_zero_count', 0)}周")
                report_lines.append(f"   - 黄白线同时在零轴上方的周数: {details.get('both_above_zero_count', 0)}周")
                report_lines.append(f"   - 黄白线同时在零轴附近的周数: {details.get('both_near_zero_count', 0)}/{self.LOOKBACK_PERIOD}")
                report_lines.append(f"   - 最新MACD DIF: {details.get('current_diff', 0):.4f}, DEA: {details.get('current_dea', 0):.4f}")
                report_lines.append(f"   - MACD金叉: {'是' if details.get('has_macd_gold_cross', False) else '否'}")
                report_lines.append(f"   - MACD评分: {details.get('score', 0):.3f}/1.0")
            else:
                report_lines.append(f"   - 原因: {details.get('reason', '数据不足')}")
        
        # 条件3：红柱未连续4根缩小
        hist = result['detailed_analysis']['hist_condition']
        report_lines.append(f"3. 红柱未连续4根缩小: {'✓ 通过' if hist['satisfied'] else '✗ 未通过'}")
        if 'details' in hist:
            details = hist['details']
            if details.get('sufficient_data', False):
                report_lines.append(f"   - 最大连续缩小周数: {details.get('max_continuous_decrease', 0)}")
                report_lines.append(f"   - 红柱出现周数: {details.get('red_hist_weeks', 0)}/{self.LOOKBACK_PERIOD}")
                report_lines.append(f"   - 红柱增长周数: {details.get('increasing_red_hist_weeks', 0)}周")
                report_lines.append(f"   - 平均红柱高度: {details.get('avg_red_hist', 0):.4f}")
                report_lines.append(f"   - 最新MACD柱状图: {details.get('current_hist', 0):.4f}")
                report_lines.append(f"   - 红柱稳定性评分: {details.get('score', 0):.3f}/1.0")
            else:
                report_lines.append(f"   - 原因: {details.get('reason', '数据不足')}")
        
        # 排除规则检查
        if 'exclusion_rule' in result['detailed_analysis']:
            exclusion = result['detailed_analysis']['exclusion_rule']
            if exclusion['triggered']:
                report_lines.append("")
                report_lines.append(f"触发排除规则: {exclusion['reason']}")
        
        report_lines.append("")
        
        # 交易建议和仓位配置
        report_lines.append("【交易建议】")
        if result['bullish_trend']:
            report_lines.append(f"✓ 周线多头趋势确认，置信度等级: {result['confidence_level']}")
            
            # 根据不同置信度等级提供差异化建议
            confidence_level = result.get('confidence_level', '')
            if confidence_level == "HIGH":
                report_lines.append("  - 策略建议: 强势多头，可积极布局，分批建仓")
                report_lines.append("  - 优先选择日线二买信号，加仓时机良好")
                report_lines.append("  - 可适当提高一买/三买仓位比例")
                report_lines.append("  - 可设置较宽松的止损")
            elif confidence_level == "MEDIUM_HIGH":
                report_lines.append("  - 策略建议: 较强多头，可积极参与，适度加仓")
                report_lines.append("  - 以日线二买为主，同时关注一买回调买入机会")
                report_lines.append("  - 设置合理止损")
            elif confidence_level == "MEDIUM":
                report_lines.append("  - 策略建议: 中性多头，谨慎参与，分批建仓")
                report_lines.append("  - 优先日线二买，可小仓位试探一买")
                report_lines.append("  - 严格设置止损")
            elif confidence_level == "MEDIUM_LOW":
                report_lines.append("  - 策略建议: 弱势多头，轻仓试探，密切关注趋势变化")
                report_lines.append("  - 以日线二买为主，极小仓位")
                report_lines.append("  - 极小仓位试探，做好止盈止损")
            elif confidence_level == "LOW":
                report_lines.append("  - 策略建议: 微弱多头，极小仓位试探，不宜重仓")
                report_lines.append("  - 仅考虑日线二买信号，极小仓位")
                report_lines.append("  - 严格控制风险，随时准备离场")
            else:
                report_lines.append("  - 重点关注日线二买信号（确定性最高）")
                report_lines.append("  - 日线一买/三买可作为辅助信号")
            
            report_lines.append("")
            report_lines.append("【推荐仓位比例】")
            report_lines.append(f"  - 日线二买对应30分钟子仓位比例: {result['position_ratios']['second_buy_ratio']*100:.1f}%")
            report_lines.append(f"  - 日线一买/三买对应子仓位比例: {result['position_ratios']['first_third_buy_ratio']*100:.1f}%")
        else:
            if 'exclusion_rule' in result['detailed_analysis'] and result['detailed_analysis']['exclusion_rule'].get('triggered', False):
                exclusion_reason = result['detailed_analysis']['exclusion_rule'].get('reason', '')
                report_lines.append(f"✗ 触发排除规则: {exclusion_reason}")
                report_lines.append("  - 建议: 保持观望，等待排除条件解除后再考虑")
            else:
                report_lines.append("✗ 周线非多头趋势，建议谨慎交易")
                report_lines.append("  - 需等待周线多头趋势确认后再考虑建仓")
                report_lines.append("  - 暂不建议加仓操作")
        
        report_lines.append("===================================")
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    # 测试用例
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建周线数据
    weeks = 12  # 12周数据
    dates = [datetime.now() - timedelta(weeks=i) for i in range(weeks)]
    dates.reverse()
    
    # 创建上升趋势的收盘价
    base_price = 100
    weekly_increase = 0.5  # 每周平均上涨0.5元
    volatility = 2.0  # 波动范围
    
    close_prices = []
    current_price = base_price
    for i in range(weeks):
        # 添加一些随机性，但确保整体趋势向上
        change = np.random.normal(weekly_increase, volatility)
        current_price = max(base_price * 0.9, current_price + change)
        close_prices.append(current_price)
    
    # 创建其他数据列
    weekly_data = {
        'date': dates,
        'open': [p * (1 - np.random.random() * 0.01) for p in close_prices],
        'high': [p * (1 + np.random.random() * 0.02) for p in close_prices],
        'low': [p * (1 - np.random.random() * 0.02) for p in close_prices],
        'close': close_prices,
        'volume': [np.random.random() * 10000000 for _ in range(weeks)]
    }
    
    weekly_df = pd.DataFrame(weekly_data)
    
    # 创建检测器并执行检测
    detector = WeeklyTrendDetector()
    result = detector.detect_weekly_bullish_trend(weekly_df)
    
    # 打印报告
    print(detector.generate_trend_report(result))