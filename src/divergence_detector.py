#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
背驰检测模块

该模块提供各种背驰检测的方法，包括：
1. 向下笔背驰检测
2. 绿柱区域背驰检测
3. 通用背驰强度计算

作者: TradeTianYuan
日期: 2025-11-25
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# 设置日志
logger = logging.getLogger(__name__)


class DivergenceDetector:
    """背驰检测器类，提供各种背驰检测方法"""
    
    def __init__(self, divergence_threshold: float = 0.3):
        """初始化背驰检测器
        
        Args:
            divergence_threshold: 背驰阈值，用于判断是否存在背驰
        """
        self.divergence_threshold = divergence_threshold
        logger.info(f"背驰检测器初始化: 阈值={divergence_threshold}")
    
    def detect_down_pens_divergence(self, df: pd.DataFrame, pens: List[Dict]) -> Tuple[bool, float]:
        """检测向下笔之间的背驰（底背驰）
        
        Args:
            df: 日线数据框
            pens: 笔数据列表
            
        Returns:
            (是否存在背驰, 背驰强度)
        """
        logger.info("开始检测向下笔背驰...")
        
        # 过滤出向下笔
        down_pens = [pen for pen in pens if pen['type'] == 'down']
        if len(down_pens) < 2:
            logger.info(f"向下笔数量不足，无法检测背驰。当前向下笔数量: {len(down_pens)}")
            return False, 0.0
        
        # 获取最近两个向下笔
        recent_down_pens = sorted(down_pens, key=lambda x: x['end_date'], reverse=True)[:2]
        if len(recent_down_pens) < 2:
            logger.info("最近向下笔数量不足，无法检测背驰")
            return False, 0.0
        
        # 第一个向下笔（更近的）
        first_pen = recent_down_pens[0]
        # 第二个向下笔（更早的）
        second_pen = recent_down_pens[1]
        
        # 确保按时间顺序排列
        if first_pen['start_date'] < second_pen['start_date']:
            first_pen, second_pen = second_pen, first_pen
        
        logger.info(f"第一个向下笔: {first_pen['start_date'].strftime('%Y-%m-%d')} 至 {first_pen['end_date'].strftime('%Y-%m-%d')}")
        logger.info(f"第二个向下笔: {second_pen['start_date'].strftime('%Y-%m-%d')} 至 {second_pen['end_date'].strftime('%Y-%m-%d')}")
        
        # 计算价格跌幅
        first_pen_price_change = (first_pen['end_price'] - first_pen['start_price']) / first_pen['start_price']
        second_pen_price_change = (second_pen['end_price'] - second_pen['start_price']) / second_pen['start_price']
        
        logger.info(f"第一个向下笔价格变化率: {first_pen_price_change:.4f}")
        logger.info(f"第二个向下笔价格变化率: {second_pen_price_change:.4f}")
        
        # 计算MACD柱面积
        first_pen_macd_area = self._calculate_macd_area_for_pen(df, first_pen)
        second_pen_macd_area = self._calculate_macd_area_for_pen(df, second_pen)
        
        logger.info(f"第一个向下笔MACD面积: {first_pen_macd_area:.6f}")
        logger.info(f"第二个向下笔MACD面积: {second_pen_macd_area:.6f}")
        
        # 检查是否存在底背驰条件
        # 1. 价格创新低
        if first_pen['end_price'] >= second_pen['end_price']:
            logger.info("不满足价格创新低条件，未检测到底背驰")
            return False, 0.0
        
        # 2. MACD面积减小（绿柱面积减小表示背驰）
        if second_pen_macd_area <= 0 or first_pen_macd_area >= second_pen_macd_area:
            logger.info("不满足MACD面积减小条件，未检测到底背驰")
            return False, 0.0
        
        # 计算背驰强度
        # 价格创新低的程度（跌幅更大）
        price_divergence = abs(first_pen_price_change) / abs(second_pen_price_change) if second_pen_price_change != 0 else 0
        
        # MACD面积减小的程度
        macd_divergence = 1 - (first_pen_macd_area / second_pen_macd_area) if second_pen_macd_area != 0 else 0
        
        # 综合背驰强度
        divergence_strength = 0.4 * price_divergence + 0.6 * macd_divergence
        
        logger.info(f"价格背驰度: {price_divergence:.4f}, MACD背驰度: {macd_divergence:.4f}")
        logger.info(f"计算得到的背驰强度: {divergence_strength:.4f}")
        
        # 判断是否达到背驰阈值
        if divergence_strength >= self.divergence_threshold:
            logger.info(f"检测到底背驰! 背驰强度={divergence_strength:.4f}，超过阈值{self.divergence_threshold}")
            return True, min(divergence_strength, 1.0)
        else:
            logger.info(f"背驰强度不足，未达到背驰阈值。当前强度: {divergence_strength:.4f}，阈值: {self.divergence_threshold}")
            return False, divergence_strength
    
    def detect_green_zones_divergence(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """检测MACD绿柱区域之间的背驰
        
        Args:
            df: 包含MACD数据的数据框
            
        Returns:
            (是否存在背驰, 背驰强度)
        """
        logger.info("开始检测绿柱区域背驰...")
        
        # 验证数据
        if 'date' not in df.columns or 'macd_hist' not in df.columns or 'close' not in df.columns:
            logger.error("数据框缺少必要的列")
            return False, 0.0
        
        # 分离9月份和10-11月份的数据
        sep_data = df[df['date'].dt.month == 9].copy()
        oct_nov_data = df[df['date'].dt.month.isin([10, 11])].copy()
        
        if sep_data.empty or oct_nov_data.empty:
            logger.warning("9月份或10-11月份数据为空，无法检测背驰")
            return False, 0.0
        
        # 找出绿柱区域（MACD柱状图为负的区域）
        sep_green_periods = self._identify_green_zones(sep_data)
        oct_nov_green_periods = self._identify_green_zones(oct_nov_data)
        
        logger.info(f"9月份绿柱区域数量: {len(sep_green_periods)}")
        logger.info(f"10-11月份绿柱区域数量: {len(oct_nov_green_periods)}")
        
        # 计算9月份绿柱总面积
        sep_total_area = 0
        for period in sep_green_periods:
            sep_total_area += self._calculate_macd_area(sep_data, period['start_date'], period['end_date'])
        
        # 计算10-11月份绿柱总面积
        oct_nov_total_area = 0
        for period in oct_nov_green_periods:
            oct_nov_total_area += self._calculate_macd_area(oct_nov_data, period['start_date'], period['end_date'])
        
        logger.info(f"9月份绿柱总面积: {sep_total_area:.6f}")
        logger.info(f"10-11月份绿柱总面积: {oct_nov_total_area:.6f}")
        
        # 获取价格低点
        sep_low = sep_data['close'].min()
        oct_nov_low = oct_nov_data['close'].min()
        
        logger.info(f"9月份最低收盘价: {sep_low:.2f}")
        logger.info(f"10-11月份最低收盘价: {oct_nov_low:.2f}")
        
        # 检查底背驰条件
        if oct_nov_low < sep_low and sep_total_area > 0 and oct_nov_total_area < sep_total_area:
            # 计算背驰强度
            price_divergence = (sep_low - oct_nov_low) / sep_low
            macd_divergence = (sep_total_area - oct_nov_total_area) / sep_total_area
            
            # 综合背驰强度
            divergence_strength = 0.5 * price_divergence + 0.5 * macd_divergence
            
            logger.info(f"价格背驰度: {price_divergence:.4f}, MACD背驰度: {macd_divergence:.4f}")
            logger.info(f"绿柱区域背驰检测: 检测到底背驰! 价格创新低但MACD绿柱面积减小, 背驰强度={divergence_strength:.4f}")
            return True, min(divergence_strength, 1.0)
        
        logger.info("绿柱区域背驰检测: 未检测到背驰条件")
        return False, 0.0
    
    def _identify_green_zones(self, df: pd.DataFrame) -> List[Dict]:
        """识别MACD绿柱区域（柱状图为负的区域）
        
        Args:
            df: 包含MACD数据的数据框
            
        Returns:
            绿柱区域列表
        """
        green_periods = []
        in_green_zone = False
        current_period = None
        
        for idx, row in df.iterrows():
            if row['macd_hist'] < 0:  # 绿柱
                if not in_green_zone:
                    # 开始一个新的绿柱区域
                    current_period = {
                        'start': idx,
                        'start_date': row['date'],
                        'end': idx,
                        'end_date': row['date']
                    }
                    in_green_zone = True
                else:
                    # 延续当前绿柱区域
                    current_period['end'] = idx
                    current_period['end_date'] = row['date']
            else:
                if in_green_zone:
                    # 结束当前绿柱区域
                    green_periods.append(current_period)
                    in_green_zone = False
        
        # 处理最后一个绿柱区域（如果数据结束时仍在绿柱区域）
        if in_green_zone and current_period:
            green_periods.append(current_period)
        
        return green_periods
    
    def _calculate_macd_area(self, df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """计算MACD柱状图面积
        
        Args:
            df: 包含MACD的数据框
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            MACD柱状图面积
        """
        if 'macd_hist' not in df.columns or 'date' not in df.columns:
            return 0.0
        
        # 使用日期范围过滤数据，避免索引问题
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        period_data = df[mask]
        
        # 计算面积（使用绝对值确保绿柱面积为正）
        if not period_data.empty:
            area = abs(period_data['macd_hist'].sum())
            return area
        return 0.0
    
    def _calculate_macd_area_for_pen(self, df: pd.DataFrame, pen: Dict) -> float:
        """计算笔对应的MACD柱状图面积
        
        Args:
            df: 包含MACD的数据框
            pen: 笔数据字典
            
        Returns:
            MACD柱状图面积
        """
        return self._calculate_macd_area(df, pen['start_date'], pen['end_date'])
    
    def calculate_divergence_strength(self, price_change1: float, price_change2: float, 
                                     macd_area1: float, macd_area2: float) -> float:
        """计算通用背驰强度
        
        Args:
            price_change1: 第一个周期的价格变化率
            price_change2: 第二个周期的价格变化率
            macd_area1: 第一个周期的MACD面积
            macd_area2: 第二个周期的MACD面积
            
        Returns:
            背驰强度 (0-1)
        """
        # 计算价格背驰度
        price_divergence = 0
        if price_change2 != 0:
            price_divergence = abs(price_change1) / abs(price_change2)
        
        # 计算MACD背驰度
        macd_divergence = 0
        if macd_area2 > 0:
            macd_divergence = 1 - (macd_area1 / macd_area2)
        
        # 综合背驰强度
        divergence_strength = 0.4 * price_divergence + 0.6 * macd_divergence
        
        return min(max(divergence_strength, 0), 1.0)  # 限制在0-1范围内


# 如果直接运行此模块，进行简单测试
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建检测器实例
    detector = DivergenceDetector()
    logger.info("背驰检测器模块测试完成")