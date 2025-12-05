#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""缠论日线级别买点检测器 - 实现特殊一买条件"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('ChanlunDailyDetector')
logger.setLevel(logging.INFO)


class ChanlunDailyDetector:
    """缠论日线级别买点检测器，实现创新低破中枢回抽一买条件"""
    
    def __init__(self, min_pen_length: int = 3, divergence_threshold: float = 0.15):
        """初始化日线检测器
        
        Args:
            min_pen_length: 笔的最小K线数量
            divergence_threshold: 背驰阈值（降低阈值提高敏感度）
        """
        self.min_pen_length = min_pen_length
        self.divergence_threshold = divergence_threshold
        # 优化参数配置
        self.fractal_sensitivity = 0.7  # 降低敏感度，严格底分型确认条件
        self.volume_threshold = 1.5     # 提高量能阈值，强化量能配合要求
        self.price_change_threshold = 1.0  # 价格变化百分比阈值
        self.central_overlap_ratio = 0.5  # 新增中枢重叠比例要求
        # MACD背驰检测优化参数
        self.macd_lookback_period = 25  # 背驰检测的回溯周期
        self.soft_divergence_threshold = 0.1  # 软背驰阈值（较低）
        self.hard_divergence_threshold = 0.3  # 硬背驰阈值（较高）
        self.macd_area_weight = 0.6  # MACD面积在背驰强度中的权重
        self.macd_peaks_weight = 0.4  # MACD峰值在背驰强度中的权重
    
    def identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别顶底分型（优化版：增加灵活性）
        
        Args:
            df: 日线数据
            
        Returns:
            添加了顶底分型标记的数据框
        """
        df = df.copy()
        df['top_fractal'] = False
        df['bottom_fractal'] = False
        df['fractal_strength'] = 0.0  # 新增分型强度字段
        
        # 检查数据量是否足够识别分型
        if len(df) < 3:
            logger.warning(f"数据量不足，无法识别分型：只有{len(df)}行数据")
            return df
        
        # 使用灵活的分型识别方法
        for i in range(1, len(df) - 1):
            # 标准顶分型（中间高）
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+1]['high']):
                # 检查是否有更强的确认（如果有更多数据）
                strength = 0.5  # 基础强度
                if i >= 2 and i <= len(df) - 3:
                    if (df.iloc[i]['high'] > df.iloc[i-2]['high'] and 
                        df.iloc[i]['high'] > df.iloc[i+2]['high']):
                        strength = 1.0  # 完全分型
                df.at[df.index[i], 'top_fractal'] = True
                df.at[df.index[i], 'fractal_strength'] = strength
            
            # 标准底分型（中间低）
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+1]['low']):
                # 检查是否有更强的确认
                strength = 0.5  # 基础强度
                if i >= 2 and i <= len(df) - 3:
                    if (df.iloc[i]['low'] < df.iloc[i-2]['low'] and 
                        df.iloc[i]['low'] < df.iloc[i+2]['low']):
                        strength = 1.0  # 完全分型
                df.at[df.index[i], 'bottom_fractal'] = True
                df.at[df.index[i], 'fractal_strength'] = strength
        
        logger.info(f"识别到顶分型: {df['top_fractal'].sum()}个, 底分型: {df['bottom_fractal'].sum()}个")
        return df
    
    def identify_pens(self, df: pd.DataFrame) -> List[Dict]:
        """识别笔
        
        Args:
            df: 包含顶底分型的数据框
            
        Returns:
            笔的列表，每个笔包含起点、终点、类型等信息
        """
        # 检查数据框是否为空或数据量是否足够
        if df.empty or len(df) < 2:
            logger.warning(f"数据量不足，无法识别笔：只有{len(df)}行数据")
            return []
        
        # 提取所有顶底分型的位置索引
        top_fractals_pos = np.where(df['top_fractal'])[0].tolist()
        bottom_fractals_pos = np.where(df['bottom_fractal'])[0].tolist()
        
        logger.info(f"识别到的顶分型位置: {top_fractals_pos}")
        logger.info(f"识别到的底分型位置: {bottom_fractals_pos}")
        
        # 合并并排序所有分型
        all_fractals = []
        all_fractals.extend([(pos, 'top') for pos in top_fractals_pos])
        all_fractals.extend([(pos, 'bottom') for pos in bottom_fractals_pos])
        all_fractals.sort(key=lambda x: x[0])
        
        pens = []
        if len(all_fractals) < 2:
            logger.warning("分型数量不足，无法识别笔")
            return pens
        
        # 识别笔（顶底分型交替）
        current_fractal = all_fractals[0]
        for next_fractal in all_fractals[1:]:
            # 顶底交替
            if current_fractal[1] != next_fractal[1]:
                start_pos, start_type = current_fractal
                end_pos, end_type = next_fractal
                
                # 检查位置索引是否有效
                if start_pos >= len(df) or end_pos >= len(df):
                    logger.warning(f"无效的分型位置索引: start_pos={start_pos}, end_pos={end_pos}, df_length={len(df)}")
                    continue
                
                # 检查笔的长度
                if end_pos - start_pos >= self.min_pen_length:
                    # 确定笔的类型
                    pen_type = 'up' if end_type == 'top' else 'down'
                    
                    # 计算涨幅/跌幅
                    try:
                        start_price = df.iloc[start_pos]['low'] if start_type == 'bottom' else df.iloc[start_pos]['high']
                        end_price = df.iloc[end_pos]['high'] if end_type == 'top' else df.iloc[end_pos]['low']
                        price_change_pct = ((end_price - start_price) / start_price) * 100
                        
                        pen = {
                            'start_idx': start_pos,
                            'end_idx': end_pos,
                            'start_date': df.iloc[start_pos]['date'] if 'date' in df.columns else None,
                            'end_date': df.iloc[end_pos]['date'] if 'date' in df.columns else None,
                            'type': pen_type,
                            'start_price': start_price,
                            'end_price': end_price,
                            'price_change_pct': price_change_pct,
                            'length': end_pos - start_pos + 1
                        }
                        pens.append(pen)
                        current_fractal = next_fractal
                    except IndexError as e:
                        logger.error(f"获取笔数据时索引错误: {str(e)}")
                        continue
        
        return pens
    
    def identify_central_banks(self, pens: List[Dict]) -> List[Dict]:
        """识别中枢（优化版：增加中枢重叠比例要求）
        
        Args:
            pens: 笔的列表
            
        Returns:
            中枢列表
        """
        central_banks = []
        
        if len(pens) < 3:
            return central_banks
        
        # 寻找至少3笔组成的中枢
        for i in range(len(pens) - 2):
            # 检查是否有重叠区间
            pen1, pen2, pen3 = pens[i], pens[i+1], pens[i+2]
            
            # 中枢需要有至少3笔，且至少包含一个向上笔和一个向下笔
            if pen1['type'] == pen3['type'] and pen1['type'] != pen2['type']:
                # 计算价格区间
                all_highs = [pen1['end_price'] if pen1['type'] == 'up' else pen1['start_price'],
                             pen2['end_price'] if pen2['type'] == 'up' else pen2['start_price'],
                             pen3['end_price'] if pen3['type'] == 'up' else pen3['start_price']]
                all_lows = [pen1['start_price'] if pen1['type'] == 'up' else pen1['end_price'],
                            pen2['start_price'] if pen2['type'] == 'up' else pen2['end_price'],
                            pen3['start_price'] if pen3['type'] == 'up' else pen3['end_price']]
                
                high = min(all_highs)
                low = max(all_lows)
                
                # 确保有重叠区间
                if high > low:
                    # 计算各笔的价格范围
                    pen1_high = pen1['end_price'] if pen1['type'] == 'up' else pen1['start_price']
                    pen1_low = pen1['start_price'] if pen1['type'] == 'up' else pen1['end_price']
                    pen1_range = pen1_high - pen1_low
                    
                    pen2_high = pen2['end_price'] if pen2['type'] == 'up' else pen2['start_price']
                    pen2_low = pen2['start_price'] if pen2['type'] == 'up' else pen2['end_price']
                    pen2_range = pen2_high - pen2_low
                    
                    pen3_high = pen3['end_price'] if pen3['type'] == 'up' else pen3['start_price']
                    pen3_low = pen3['start_price'] if pen3['type'] == 'up' else pen3['end_price']
                    pen3_range = pen3_high - pen3_low
                    
                    # 计算中枢重叠比例
                    central_range = high - low
                    max_pen_range = max(pen1_range, pen2_range, pen3_range)
                    overlap_ratio = central_range / max_pen_range if max_pen_range > 0 else 0
                    
                    # 新增中枢重叠比例要求
                    if overlap_ratio >= self.central_overlap_ratio:
                        central_bank = {
                            'start_idx': pen1['start_idx'],
                            'end_idx': pen3['end_idx'],
                            'start_date': pen1['start_date'],
                            'end_date': pen3['end_date'],
                            'high': high,
                            'low': low,
                            'middle': (high + low) / 2,
                            'range_pct': ((high - low) / low) * 100,
                            'pen_count': 3,
                            'overlap_ratio': overlap_ratio  # 记录重叠比例
                        }
                        central_banks.append(central_bank)
                        logger.info(f"识别到有效中枢: 上沿={high:.2f}, 下沿={low:.2f}, 重叠比例={overlap_ratio:.2f}")
        
        return central_banks
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 日线数据
            
        Returns:
            添加了MACD指标的数据框
        """
        df = df.copy()
        # 计算EMA
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        # 计算MACD线
        df['macd_line'] = df['ema12'] - df['ema26']
        # 计算信号线
        df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        # 计算柱状图
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        
        return df
    
    def calculate_macd_area(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """计算MACD柱状图面积（优化版：加权面积计算）
        
        Args:
            df: 包含MACD数据的数据框
            start_idx: 起始索引
            end_idx: 结束索引
            
        Returns:
            MACD柱状图面积（加权计算）
        """
        if 'macd_hist' not in df.columns:
            return 0.0
        
        # 确保索引有效
        if start_idx < 0 or end_idx >= len(df) or start_idx > end_idx:
            return 0.0
        
        # 获取区间内的MACD柱状图数据
        hist_data = df.iloc[start_idx:end_idx+1]['macd_hist']
        
        # 计算加权面积（最近的数据赋予更高权重）
        weights = np.linspace(0.5, 1.0, len(hist_data))
        weighted_area = abs((hist_data * weights).sum())
        
        return weighted_area
    
    def find_macd_peaks(self, df: pd.DataFrame, start_idx: int, end_idx: int, is_bullish: bool = False) -> List[Tuple[int, float]]:
        """寻找MACD柱状图的峰值点
        
        Args:
            df: 包含MACD数据的数据框
            start_idx: 起始索引
            end_idx: 结束索引
            is_bullish: 是否寻找看涨峰值（红柱）
            
        Returns:
            峰值点列表 [(索引, 值)]
        """
        if 'macd_hist' not in df.columns:
            return []
        
        peaks = []
        
        # 根据看涨/看跌选择合适的比较函数
        if is_bullish:
            # 看涨时寻找红柱峰值（正值）
            for i in range(start_idx + 1, end_idx):
                if (df.iloc[i]['macd_hist'] > df.iloc[i-1]['macd_hist'] and 
                    df.iloc[i]['macd_hist'] > df.iloc[i+1]['macd_hist'] and 
                    df.iloc[i]['macd_hist'] > 0):
                    peaks.append((i, df.iloc[i]['macd_hist']))
        else:
            # 看跌时寻找绿柱峰值（负值的绝对值最大）
            for i in range(start_idx + 1, end_idx):
                if (df.iloc[i]['macd_hist'] < df.iloc[i-1]['macd_hist'] and 
                    df.iloc[i]['macd_hist'] < df.iloc[i+1]['macd_hist'] and 
                    df.iloc[i]['macd_hist'] < 0):
                    peaks.append((i, df.iloc[i]['macd_hist']))
        
        return peaks
    
    def detect_divergence(self, df: pd.DataFrame, pens: List[Dict]) -> Tuple[bool, float]:
        """检测下跌背驰（增强版：多种背驰检测方法组合）
        
        Args:
            df: 日线数据
            pens: 笔的列表
            
        Returns:
            (是否背驰, 背驰强度)
        """
        # 计算MACD指标
        df = self.calculate_macd(df)
        
        # 记录所有检测方法的结果
        results = []
        
        # 方法1: 基于笔的背驰检测
        result1 = self._detect_divergence_by_pens(df, pens)
        results.append(result1)
        if result1[0]:
            logger.info(f"通过笔分析检测到背驰，强度: {result1[1]:.4f}")
        
        # 方法2: 基于价格和MACD柱状图的直接比较
        result2 = self._detect_direct_divergence(df)
        results.append(result2)
        if result2[0]:
            logger.info(f"通过直接比较检测到背驰，强度: {result2[1]:.4f}")
        
        # 方法3: 基于MACD柱状图峰值的背驰检测（新增）
        result3 = self._detect_divergence_by_macd_peaks(df)
        results.append(result3)
        if result3[0]:
            logger.info(f"通过MACD峰值检测到背驰，强度: {result3[1]:.4f}")
        
        # 方法4: 基于价格结构的多级背驰检测（新增）
        result4 = self._detect_multi_level_divergence(df)
        results.append(result4)
        if result4[0]:
            logger.info(f"通过多级结构检测到背驰，强度: {result4[1]:.4f}")
        
        # 找出强度最大的背驰信号
        valid_results = [(strength, method_idx) for method_idx, (is_valid, strength) in enumerate(results) if is_valid]
        
        if valid_results:
            max_strength, max_idx = max(valid_results)
            logger.info(f"最终背驰判定: 方法{max_idx+1}，强度: {max_strength:.4f}")
            return True, max_strength
        
        return False, 0.0
    
    def _detect_divergence_by_macd_peaks(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """基于MACD柱状图峰值的背驰检测（新增方法）
        
        Args:
            df: 包含MACD数据的数据框
            
        Returns:
            (是否背驰, 背驰强度)
        """
        if 'macd_hist' not in df.columns:
            return False, 0.0
        
        # 获取最近的MACD数据
        recent_df = df.tail(self.macd_lookback_period).copy()
        
        # 寻找绿柱峰值（看跌背驰）
        green_peaks = self.find_macd_peaks(recent_df, 0, len(recent_df) - 1, is_bullish=False)
        
        if len(green_peaks) < 2:
            return False, 0.0
        
        # 获取最近两个绿柱峰值
        peaks_sorted = sorted(green_peaks, key=lambda x: x[0])  # 按时间排序
        peak1_idx, peak1_val = peaks_sorted[-2]  # 前一个峰值
        peak2_idx, peak2_val = peaks_sorted[-1]  # 最近峰值
        
        # 获取对应位置的价格低点
        price1_low = recent_df.iloc[peak1_idx]['low']
        price2_low = recent_df.iloc[peak2_idx]['low']
        
        # 检查是否价格创新低，但MACD绿柱峰值未创新低（绝对值变小）
        if price2_low < price1_low and peak2_val > peak1_val:
            # 计算背驰强度
            price_diff_pct = abs((price2_low - price1_low) / price1_low) * 100
            macd_diff_pct = abs((peak2_val - peak1_val) / peak1_val) * 100 if peak1_val != 0 else 0
            
            # 计算时间间隔权重（时间间隔越大，权重越高）
            time_weight = (peak2_idx - peak1_idx) / len(recent_df)
            
            strength = (0.6 * price_diff_pct + 0.4 * macd_diff_pct) * time_weight
            
            logger.debug(f"MACD峰值背驰检测: 价格新低={price2_low:.3f}<{price1_low:.3f}, MACD峰值={peak2_val:.3f}>{peak1_val:.3f}, 强度={strength:.4f}")
            
            if strength >= self.divergence_threshold:
                return True, min(strength / 10, 1.0)  # 归一化
        
        return False, 0.0
    
    def _detect_multi_level_divergence(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """基于价格结构的多级背驰检测（新增方法）
        
        Args:
            df: 包含MACD数据的数据框
            
        Returns:
            (是否背驰, 背驰强度)
        """
        if 'macd_hist' not in df.columns or len(df) < 30:
            return False, 0.0
        
        # 获取最近数据
        recent_df = df.tail(30).copy()
        
        # 寻找最近的两个明显低点区域
        low_regions = []
        
        # 滑动窗口寻找低点区域
        window_size = 3
        for i in range(window_size, len(recent_df) - window_size):
            # 检查是否是局部低点区域
            if (recent_df.iloc[i]['low'] < recent_df.iloc[i-window_size:i]['low'].min() and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i+1:i+window_size+1]['low'].min()):
                # 记录这个低点区域的信息
                region_start = max(0, i - window_size)
                region_end = min(len(recent_df) - 1, i + window_size)
                region_low = recent_df.iloc[region_start:region_end+1]['low'].min()
                region_macd = recent_df.iloc[region_start:region_end+1]['macd_hist'].mean()
                
                low_regions.append({
                    'center_idx': i,
                    'start_idx': region_start,
                    'end_idx': region_end,
                    'low': region_low,
                    'macd': region_macd,
                    'date': recent_df.iloc[i]['date'] if 'date' in recent_df.columns else None
                })
        
        if len(low_regions) < 2:
            return False, 0.0
        
        # 按时间排序，取最后两个低点区域
        low_regions_sorted = sorted(low_regions, key=lambda x: x['center_idx'])
        prev_region = low_regions_sorted[-2]
        recent_region = low_regions_sorted[-1]
        
        # 检查是否价格创新低，但MACD未创新低
        if recent_region['low'] < prev_region['low'] and recent_region['macd'] > prev_region['macd']:
            # 计算区域间的背驰强度
            price_diff_pct = abs((recent_region['low'] - prev_region['low']) / prev_region['low']) * 100
            macd_diff_pct = abs((recent_region['macd'] - prev_region['macd']) / prev_region['macd']) * 100 if prev_region['macd'] != 0 else 0
            
            # 计算区域完整性得分
            prev_region_score = (prev_region['end_idx'] - prev_region['start_idx'] + 1) / (2 * window_size + 1)
            recent_region_score = (recent_region['end_idx'] - recent_region['start_idx'] + 1) / (2 * window_size + 1)
            
            # 综合强度
            strength = 0.5 * (price_diff_pct + macd_diff_pct) * 0.5 * (prev_region_score + recent_region_score)
            
            logger.debug(f"多级背驰检测: 前区域低点={prev_region['low']:.3f}, MACD={prev_region['macd']:.3f}; 近区域低点={recent_region['low']:.3f}, MACD={recent_region['macd']:.3f}; 强度={strength:.4f}")
            
            if strength >= self.soft_divergence_threshold:
                return True, min(strength / 15, 1.0)  # 归一化
        
        return False, 0.0
    
    def _detect_divergence_by_pens(self, df: pd.DataFrame, pens: List[Dict]) -> Tuple[bool, float]:
        """基于笔的背驰检测（优化版：多笔比较和梯度阈值）"""
        # 过滤出向下笔
        down_pens = [pen for pen in pens if pen['type'] == 'down']
        if len(down_pens) < 2:
            logger.debug(f"向下笔数量不足，无法基于笔检测背驰: {len(down_pens)}个")
            return False, 0.0
        
        # 按时间顺序排列向下笔（最旧到最新）
        down_pens_sorted = sorted(down_pens, key=lambda x: x['end_idx'])
        
        # 获取最近两个向下笔
        if len(down_pens_sorted) >= 2:
            # 前一个向下笔（更早的）
            prev_down_pen = down_pens_sorted[-2]
            # 最近的向下笔
            recent_down_pen = down_pens_sorted[-1]
            
            # 检查价格是否创新低
            if recent_down_pen['end_price'] >= prev_down_pen['end_price']:
                # 即使没有创新低，也可以检查是否有潜在的盘整背驰
                price_ratio = recent_down_pen['end_price'] / prev_down_pen['end_price']
                if price_ratio > 0.97:  # 价格接近，但可能有背驰
                    logger.debug(f"价格接近创新低（比例: {price_ratio:.3f}），检查盘整背驰")
                else:
                    logger.debug("未满足价格创新低条件")
                    return False, 0.0
            
            # 计算MACD绿柱面积（使用优化的加权面积计算）
            prev_area = self.calculate_macd_area(df, prev_down_pen['start_idx'], prev_down_pen['end_idx'])
            recent_area = self.calculate_macd_area(df, recent_down_pen['start_idx'], recent_down_pen['end_idx'])
            
            logger.debug(f"前向下笔MACD面积: {prev_area:.6f}, 最近向下笔MACD面积: {recent_area:.6f}")
            
            # 检查MACD面积是否减小
            if prev_area <= 0:
                logger.debug("前向下笔MACD面积计算失败")
                return False, 0.0
            
            area_ratio = recent_area / prev_area
            
            # 梯度阈值判断
            if area_ratio >= 0.9:  # 面积几乎没有减小
                logger.debug("未满足MACD面积减小条件")
                return False, 0.0
            
            # 计算背驰强度
            # 1. 价格背驰分量
            if prev_down_pen['price_change_pct'] != 0:
                price_divergence = abs(recent_down_pen['price_change_pct'] / prev_down_pen['price_change_pct'])
            else:
                price_divergence = 0
            
            # 2. MACD面积背驰分量
            macd_area_divergence = (prev_area - recent_area) / prev_area if prev_area > 0 else 0
            
            # 3. MACD峰值背驰分量（新增）
            prev_macd_peaks = self.find_macd_peaks(df, prev_down_pen['start_idx'], prev_down_pen['end_idx'], is_bullish=False)
            recent_macd_peaks = self.find_macd_peaks(df, recent_down_pen['start_idx'], recent_down_pen['end_idx'], is_bullish=False)
            
            macd_peak_divergence = 0
            if prev_macd_peaks and recent_macd_peaks:
                # 获取每个笔中的最低MACD值（绿柱峰值）
                prev_min_macd = min([peak[1] for peak in prev_macd_peaks])
                recent_min_macd = min([peak[1] for peak in recent_macd_peaks])
                
                if prev_min_macd != 0:
                    # 绿柱峰值比较（负值比较）
                    macd_peak_divergence = (prev_min_macd - recent_min_macd) / abs(prev_min_macd)
            
            # 加权计算综合背驰强度
            divergence_strength = (0.3 * price_divergence + 
                                  self.macd_area_weight * macd_area_divergence + 
                                  self.macd_peaks_weight * macd_peak_divergence)
            
            # 根据面积比例调整阈值要求
            if area_ratio < 0.5:  # 面积明显减小，使用较低阈值
                effective_threshold = self.soft_divergence_threshold
            else:  # 面积减小不明显，使用较高阈值
                effective_threshold = self.divergence_threshold
            
            if divergence_strength >= effective_threshold:
                return True, min(divergence_strength, 1.0)
        
        return False, 0.0
    
    def _detect_direct_divergence(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """直接基于价格和MACD柱状图的背驰检测（优化版：多周期低点检测）"""
        if 'macd_hist' not in df.columns:
            return False, 0.0
        
        # 使用更长的回溯周期
        recent_df = df.tail(self.macd_lookback_period).copy()
        
        # 方法1: 寻找精确低点（单日低点）
        precise_low_points = []
        for i in range(1, len(recent_df) - 1):
            if (recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low']):
                # 记录低点信息，包括前后几天的MACD数据
                start_idx = max(0, i - 1)
                end_idx = min(len(recent_df) - 1, i + 1)
                avg_macd = recent_df.iloc[start_idx:end_idx+1]['macd_hist'].mean()
                precise_low_points.append((i, recent_df.iloc[i]['low'], avg_macd))
        
        # 方法2: 寻找区域低点（多日低点区域）
        window_low_points = []
        window_size = 3
        for i in range(window_size, len(recent_df) - window_size):
            window = recent_df.iloc[i-window_size:i+window_size+1]
            if recent_df.iloc[i]['low'] == window['low'].min():
                window_avg_macd = window['macd_hist'].mean()
                window_low_points.append((i, recent_df.iloc[i]['low'], window_avg_macd))
        
        # 合并两种方法的结果
        all_low_points = precise_low_points + window_low_points
        
        # 去重并按索引排序
        unique_low_points = {}
        for idx, price, macd in all_low_points:
            if idx not in unique_low_points or price < unique_low_points[idx][0]:
                unique_low_points[idx] = (price, macd)
        
        sorted_low_points = [(idx, price, macd) for idx, (price, macd) in 
                           sorted(unique_low_points.items(), key=lambda x: x[0])]
        
        if len(sorted_low_points) < 2:
            logger.debug(f"低点数量不足，无法直接检测背驰: {len(sorted_low_points)}个")
            return False, 0.0
        
        # 检查所有可能的低点对组合
        best_strength = 0
        has_divergence = False
        
        # 检查最近的3个低点组合
        for i in range(max(0, len(sorted_low_points) - 3), len(sorted_low_points) - 1):
            for j in range(i + 1, len(sorted_low_points)):
                low1_idx, low1_price, low1_macd = sorted_low_points[i]
                low2_idx, low2_price, low2_macd = sorted_low_points[j]
                
                # 确保时间间隔合理（至少3个交易日）
                if low2_idx - low1_idx < 3:
                    continue
                
                # 检查背驰条件
                if low2_price < low1_price * 0.98:  # 价格创新低（允许2%误差）
                    # 情况1: MACD明显背离（柱状图变浅）
                    if low2_macd > low1_macd * 0.8:  # 绿柱变浅
                        # 计算背驰强度
                        price_diff_pct = abs((low2_price - low1_price) / low1_price) * 100
                        
                        # 计算MACD差异百分比
                        if low1_macd != 0:
                            macd_diff_pct = abs((low2_macd - low1_macd) / low1_macd) * 100
                        else:
                            macd_diff_pct = 0
                        
                        # 计算时间权重
                        time_weight = min((low2_idx - low1_idx) / len(recent_df), 1.0)
                        
                        # 综合背驰强度
                        strength = 0.4 * price_diff_pct + 0.6 * macd_diff_pct
                        strength *= time_weight  # 时间间隔越大，权重越高
                        
                        logger.debug(f"直接检测组合[{i},{j}]: 低点1价格={low1_price:.3f}, MACD={low1_macd:.3f}; 低点2价格={low2_price:.3f}, MACD={low2_macd:.3f}; 强度={strength:.4f}")
                        
                        if strength > best_strength:
                            best_strength = strength
                            
                            # 根据强度判断是否构成背驰
                            if strength >= self.soft_divergence_threshold * 10:  # 调整阈值比例
                                has_divergence = True
        
        if has_divergence:
            return True, min(best_strength / 20, 1.0)  # 归一化
        
        return False, 0.0
    
    def detect_bottom_fractal_confirmation(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """检测底分型确认（优化版：严格底分型确认条件）
        
        Args:
            df: 包含顶底分型的数据框
            
        Returns:
            (是否确认, 底分型信息)
        """
        if 'bottom_fractal' not in df.columns or len(df) < 3:
            return False, {}
        
        # 找到最近的底分型
        bottom_fractal_indices = np.where(df['bottom_fractal'])[0]
        if len(bottom_fractal_indices) == 0:
            return False, {}
        
        # 获取最后一个底分型的位置
        last_bottom_idx = bottom_fractal_indices[-1]
        
        # 确保底分型后有足够的K线进行确认（至少两根K线）
        if last_bottom_idx >= len(df) - 2:
            return False, {}
        
        # 底分型K线索引（严格定义）
        k3_idx = last_bottom_idx  # 底分型中间K线
        k4_idx = last_bottom_idx + 1  # 确认K线1
        k5_idx = last_bottom_idx + 2  # 确认K线2
        
        # 获取基本数据
        k3_low = df.iloc[k3_idx]['low']
        k3_close = df.iloc[k3_idx]['close']
        k4_close = df.iloc[k4_idx]['close']
        k4_open = df.iloc[k4_idx]['open']
        k5_close = df.iloc[k5_idx]['close']
        k5_open = df.iloc[k5_idx]['open']
        
        # 获取分型强度
        fractal_strength = df.iloc[k3_idx]['fractal_strength'] if 'fractal_strength' in df.columns else 0.5
        
        # 严格的底分型确认条件
        # 条件1: 分型后一天收盘价高于分型当天收盘价
        confirmation_condition1 = k4_close > k3_close
        # 条件2: 分型后第二天收盘价高于第一天收盘价（连续上涨确认）
        confirmation_condition2 = k5_close > k4_close
        # 条件3: 分型后K线形成阳线
        confirmation_condition3 = k4_close > k4_open
        # 条件4: 分型当天最低价是前后三根K线的最低点
        is_true_low = True
        for i in range(1, 4):
            if k3_idx - i >= 0 and df.iloc[k3_idx - i]['low'] < k3_low:
                is_true_low = False
                break
            if k3_idx + i < len(df) and df.iloc[k3_idx + i]['low'] < k3_low:
                is_true_low = False
                break
        
        # 严格确认：所有条件必须满足
        is_confirmed = confirmation_condition1 and confirmation_condition2 and confirmation_condition3 and is_true_low
        
        # 获取更多K线信息
        k1_idx = k3_idx - 2 if k3_idx - 2 >= 0 else 0
        k2_idx = k3_idx - 1 if k3_idx - 1 >= 0 else 0
        
        bottom_fractal_info = {
            'k1_idx': k1_idx,
            'k2_idx': k2_idx,
            'k3_idx': k3_idx,
            'k4_idx': k4_idx,
            'k5_idx': k5_idx,
            'k1_date': df.iloc[k1_idx]['date'] if 'date' in df.columns else None,
            'k3_date': df.iloc[k3_idx]['date'] if 'date' in df.columns else None,
            'k4_date': df.iloc[k4_idx]['date'] if 'date' in df.columns else None,
            'k5_date': df.iloc[k5_idx]['date'] if 'date' in df.columns else None,
            'k3_low': k3_low,
            'k3_close': k3_close,
            'k4_close': k4_close,
            'k4_open': k4_open,
            'k5_close': k5_close,
            'k5_open': k5_open,
            'fractal_strength': fractal_strength,
            'confirmation_condition1': confirmation_condition1,
            'confirmation_condition2': confirmation_condition2,
            'confirmation_condition3': confirmation_condition3,
            'is_true_low': is_true_low,
            'is_confirmed': is_confirmed
        }
        
        logger.info(f"底分型确认分析: 分型强度={fractal_strength:.2f}, 条件1={confirmation_condition1}, 条件2={confirmation_condition2}, 条件3={confirmation_condition3}, 是否真底={is_true_low}, 最终结果={is_confirmed}")
        return is_confirmed, bottom_fractal_info
    
    def check_volume_condition(self, df: pd.DataFrame, confirm_date_idx: int) -> Tuple[bool, float]:
        """检查量能放大条件（优化版：强化量能配合要求）
        
        Args:
            df: 日线数据
            confirm_date_idx: 确认日的索引
            
        Returns:
            (是否满足, 量能放大比例)
        """
        if 'volume' not in df.columns or confirm_date_idx < 5:
            # 要求更多历史数据
            return False, 0.0
        
        # 获取确认日成交量
        confirm_volume = df.iloc[confirm_date_idx]['volume']
        
        # 计算前5日平均成交量（延长周期以确保稳定性）
        lookback_days = 5
        avg_volume = df.iloc[confirm_date_idx-lookback_days:confirm_date_idx]['volume'].mean()
        
        # 计算前10日平均成交量（更长期的对比）
        long_lookback_days = min(10, confirm_date_idx)
        long_avg_volume = df.iloc[confirm_date_idx-long_lookback_days:confirm_date_idx]['volume'].mean()
        
        # 计算前一个交易日成交量
        prev_volume = df.iloc[confirm_date_idx-1]['volume'] if confirm_date_idx > 0 else avg_volume
        
        # 计算量能放大比例
        if avg_volume > 0 and long_avg_volume > 0:
            volume_ratio = confirm_volume / avg_volume
            long_volume_ratio = confirm_volume / long_avg_volume
            daily_increase = confirm_volume / prev_volume if prev_volume > 0 else 0
            
            # 价格上涨确认
            price_rising = False
            if confirm_date_idx > 0:
                current_close = df.iloc[confirm_date_idx]['close']
                prev_close = df.iloc[confirm_date_idx-1]['close']
                price_rising = current_close > prev_close
            
            # 严格的量能条件：两个条件必须同时满足
            # 1. 相对于短期平均和长期平均都有明显放大
            # 2. 必须伴随价格上涨
            is_met = (volume_ratio >= self.volume_threshold) and \
                     (long_volume_ratio >= 1.3) and \
                     price_rising
            
            # 特别强的量能放大可以稍微放宽价格条件
            if (volume_ratio >= self.volume_threshold * 1.3) and \
               (long_volume_ratio >= 1.5):
                # 允许价格小幅下跌但跌幅不超过1%
                price_change_pct = ((current_close - prev_close) / prev_close) * 100
                is_met = abs(price_change_pct) <= 1.0
            
            logger.info(f"量能验证: 确认日成交量={confirm_volume}, 前{lookback_days}日平均={avg_volume:.2f}, 比例={volume_ratio:.2f}, 长期比例={long_volume_ratio:.2f}, 日增幅={daily_increase:.2f}, 价格上涨={price_rising}, 结果={is_met}")
            return is_met, volume_ratio
        
        return False, 0.0
    
    def detect_inno_low_break_central_first_buy(self, df: pd.DataFrame, central_banks: List[Dict], pens: List[Dict]) -> Dict:
        """
        检测创新低破中枢回抽一买
        
        Args:
            df: 日线数据
            central_banks: 中枢列表
            pens: 笔的列表
            
        Returns:
            包含买点检测结果的字典
        """
        # 初始化结果
        result = {
            'has_valid_buy_point': False,
            'signal_type': 'inno_low_break_central_first_buy',  # 新增信号类型标识
            'signal_subtype': '',  # 新增信号子类型
            'central_bank_conditions_met': False,
            'break_central_conditions_met': False,
            'divergence_conditions_met': False,
            'fractal_confirmation_met': False,
            'volume_condition_met': False,
            'central_bank_info': None,
            'break_info': None,
            'divergence_info': None,
            'fractal_info': None,
            'volume_info': None,
            'signal_strength': 0.0,
            'signal_date': None
        }
        
        # 1. 检查是否有有效的中枢
        if not central_banks:
            logger.info("未找到有效的中枢")
            return result
        
        # 获取最近的中枢
        recent_central_bank = max(central_banks, key=lambda x: x['end_idx'])
        result['central_bank_info'] = recent_central_bank
        result['central_bank_conditions_met'] = True
        
        logger.info(f"找到最近中枢: 上沿={recent_central_bank['high']:.2f}, 下沿={recent_central_bank['low']:.2f}")
        
        # 2. 检查是否跌破中枢下沿并创新低
        # 找到中枢之后的K线
        post_central_df = df.iloc[recent_central_bank['end_idx']:].copy()
        
        # 检查是否有收盘价跌破中枢下沿
        below_central = post_central_df['close'] < recent_central_bank['low']
        if not below_central.any():
            logger.info("未检测到跌破中枢下沿的情况")
            return result
        
        # 找到第一个跌破中枢的位置
        first_break_idx = post_central_df.index[below_central.idxmax()]
        
        # 检查跌破后的最低收盘价是否创中枢震荡阶段的新低
        # 中枢震荡阶段的最低价
        central_period_df = df.iloc[recent_central_bank['start_idx']:recent_central_bank['end_idx']].copy()
        central_low = central_period_df['low'].min() if not central_period_df.empty else 0
        
        # 跌破后的数据
        after_break_df = df.iloc[first_break_idx:].copy()
        after_break_low = after_break_df['low'].min() if not after_break_df.empty else 0
        
        if after_break_low >= central_low:
            logger.info(f"未满足创新低条件: 跌破后最低={after_break_low:.2f}, 中枢震荡阶段最低={central_low:.2f}")
            return result
        
        result['break_central_conditions_met'] = True
        result['break_info'] = {
            'first_break_idx': first_break_idx,
            'first_break_date': df.iloc[first_break_idx]['date'] if 'date' in df.columns else None,
            'central_low': central_low,
            'after_break_low': after_break_low
        }
        
        logger.info(f"满足破中枢创新低条件: 跌破后最低={after_break_low:.2f}, 中枢震荡阶段最低={central_low:.2f}")
        
        # 3. 检测下跌背驰
        has_divergence, divergence_strength = self.detect_divergence(df, pens)
        result['divergence_conditions_met'] = has_divergence
        result['divergence_info'] = {
            'has_divergence': has_divergence,
            'divergence_strength': divergence_strength
        }
        
        # 4. 检测底分型确认
        is_fractal_confirmed, fractal_info = self.detect_bottom_fractal_confirmation(df)
        result['fractal_confirmation_met'] = is_fractal_confirmed
        result['fractal_info'] = fractal_info
        
        # 5. 检查量能放大条件
        if is_fractal_confirmed and fractal_info:
            k5_idx = fractal_info.get('k5_idx')
            if k5_idx is not None and k5_idx < len(df):
                volume_met, volume_ratio = self.check_volume_condition(df, k5_idx)
                result['volume_condition_met'] = volume_met
                result['volume_info'] = {
                    'volume_met': volume_met,
                    'volume_ratio': volume_ratio
                }
        
        # 6. 计算信号强度
        conditions_met = sum([
            result['central_bank_conditions_met'],
            result['break_central_conditions_met'],
            result['divergence_conditions_met'],
            result['fractal_confirmation_met'],
            result['volume_condition_met']
        ])
        
        # 计算背驰力度得分（30%）
        divergence_score = result['divergence_info'].get('divergence_strength', 0) * 0.3
        
        # 计算量能得分（40%）
        volume_score = min(result['volume_info'].get('volume_ratio', 0) / 2, 1.0) * 0.4 if result['volume_info'] else 0
        
        # 计算分型有效性得分（30%）
        fractal_score = 0.3 if result['fractal_confirmation_met'] else 0
        
        # 综合信号强度
        signal_strength = divergence_score + volume_score + fractal_score
        result['signal_strength'] = signal_strength
        
        # 7. 确定是否满足完整买点条件（优化版：使用加权计分方式）
        # 核心条件必须全部满足：中枢存在、跌破中枢、底分型确认
        core_conditions_met = all([
            result['central_bank_conditions_met'],
            result['break_central_conditions_met'],
            result['fractal_confirmation_met']
        ])
        
        # 辅助条件可部分满足：背驰、量能
        support_conditions_count = sum([
            result['divergence_conditions_met'],
            result['volume_condition_met']
        ])
        
        # 确定买点有效性
        if core_conditions_met:
            # 核心条件满足时，至少需要1个辅助条件满足
            result['has_valid_buy_point'] = support_conditions_count >= 1
            
            # 增强信号强度计算
            if result['divergence_info']:
                divergence_bonus = result['divergence_info'].get('divergence_strength', 0) * 0.2
                result['signal_strength'] += divergence_bonus
            
            # 满足所有条件给予额外强度奖励
            if support_conditions_count == 2:
                result['signal_strength'] = min(result['signal_strength'] + 0.1, 1.0)
                
            # 设置信号子类型
            if result['divergence_conditions_met'] and result['volume_condition_met']:
                result['signal_subtype'] = 'strong'
            elif result['divergence_conditions_met']:
                result['signal_subtype'] = 'divergence'
            elif result['volume_condition_met']:
                result['signal_subtype'] = 'volume'
        else:
            result['has_valid_buy_point'] = False
        
        logger.info(f"买点条件评估: 核心条件={core_conditions_met}, 辅助条件满足数={support_conditions_count}/2, 最终结果={result['has_valid_buy_point']}, 信号强度={result['signal_strength']:.4f}")
        
        # 设置信号日期为底分型确认日
        if result['fractal_info'] and result['fractal_info'].get('k5_date'):
            result['signal_date'] = result['fractal_info'].get('k5_date')
        
        logger.info(f"创新低破中枢回抽一买检测完成: 是否满足条件={result['has_valid_buy_point']}, 信号强度={signal_strength:.4f}")
        
        return result
    
    def analyze_daily_buy_condition(self, df: pd.DataFrame) -> Dict:
        """分析日线级别买点条件（支持多种买入信号类型的差异化处理）
        
        Args:
            df: 日线数据
            
        Returns:
            包含完整分析结果的字典
        """
        try:
            if df.empty:
                logger.error("日线数据为空，无法分析")
                return {
                    'success': False,
                    'error': '日线数据为空'
                }
            
            # 复制数据以避免修改原始数据
            df_copy = df.copy()
            
            # 确保日期列存在并转换为datetime
            if 'date' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
                df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # 1. 识别顶底分型
            df_copy = self.identify_fractals(df_copy)
            
            # 2. 识别笔
            pens = self.identify_pens(df_copy)
            
            # 3. 识别中枢
            central_banks = self.identify_central_banks(pens)
            
            # 4. 检测创新低破中枢回抽一买
            buy_point_result = self.detect_inno_low_break_central_first_buy(df_copy, central_banks, pens)
            
            # 初始化信号列表
            signals = []
            primary_signal = None
            
            # 如果存在特殊一买信号，添加到信号列表
            if buy_point_result['has_valid_buy_point']:
                signal = {
                    'type': buy_point_result['signal_type'],
                    'subtype': buy_point_result['signal_subtype'],
                    'date': buy_point_result['signal_date'],
                    'strength': buy_point_result['signal_strength'],
                    'explanation': f"日线级别创新低破中枢回抽一买（{buy_point_result['signal_subtype']}）",
                    'details': {
                        'central_bank': buy_point_result['central_bank_info'],
                        'break_info': buy_point_result['break_info'],
                        'divergence_info': buy_point_result['divergence_info'],
                        'fractal_info': buy_point_result['fractal_info'],
                        'volume_info': buy_point_result['volume_info']
                    }
                }
                signals.append(signal)
                primary_signal = signal
            
            result = {
                'success': True,
                'has_valid_buy_point': buy_point_result['has_valid_buy_point'],
                'signal_strength': buy_point_result['signal_strength'],
                'signal_date': buy_point_result['signal_date'],
                'primary_signal': primary_signal,
                'all_signals': signals,
                'conditions': {
                    'central_bank_met': buy_point_result['central_bank_conditions_met'],
                    'break_central_met': buy_point_result['break_central_conditions_met'],
                    'divergence_met': buy_point_result['divergence_conditions_met'],
                    'fractal_confirmation_met': buy_point_result['fractal_confirmation_met'],
                    'volume_condition_met': buy_point_result['volume_condition_met']
                },
                'details': {
                    'central_bank': buy_point_result['central_bank_info'],
                    'break_info': buy_point_result['break_info'],
                    'divergence_info': buy_point_result['divergence_info'],
                    'fractal_info': buy_point_result['fractal_info'],
                    'volume_info': buy_point_result['volume_info'],
                    'pens': pens[-5:] if pens else [],  # 返回最近5个笔
                    'central_banks': central_banks
                }
            }
            
            # 添加信号统计信息
            if signals:
                signal_types = {signal['type'] for signal in signals}
                signal_subtypes = {signal['subtype'] for signal in signals}
                result['signal_statistics'] = {
                    'total_signals': len(signals),
                    'signal_types': list(signal_types),
                    'signal_subtypes': list(signal_subtypes),
                    'average_strength': sum(s['strength'] for s in signals) / len(signals)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"日线级别买点分析失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# 测试代码
if __name__ == "__main__":
    # 设置日志
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # 创建示例数据进行测试
    dates = pd.date_range(start='2025-10-01', periods=30)
    data = {
        'date': dates,
        'open': range(100, 130),
        'high': [x + 2 for x in range(100, 130)],
        'low': [x - 2 for x in range(100, 130)],
        'close': range(101, 131),
        'volume': [x * 100 for x in range(30)]
    }
    df = pd.DataFrame(data)
    
    # 手动添加一些底分型标记用于测试
    df.loc[10, 'low'] = 95  # 人为制造一个底分型
    df.loc[11, 'low'] = 96
    df.loc[12, 'low'] = 94  # 底分型低点
    df.loc[13, 'low'] = 95
    df.loc[14, 'low'] = 96
    
    detector = ChanlunDailyDetector()
    result = detector.analyze_daily_buy_condition(df)
    print("日线级别买点分析结果:")
    print(result)