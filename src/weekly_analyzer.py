#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""周线分析模块 - 实现周线分档条件检测"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('WeeklyAnalyzer')
logger.setLevel(logging.INFO)

class WeeklyAnalyzer:
    """周线分析器，实现周线分档条件检测"""
    
    def __init__(self, ma_period: int = 5):
        """初始化周线分析器
        
        Args:
            ma_period: 移动平均线周期，默认为5周线
        """
        self.ma_period = ma_period
    
    def calculate_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算移动平均线
        
        Args:
            df: 周线数据
            period: 周期
            
        Returns:
            移动平均线Series
        """
        return df['close'].rolling(window=period).mean()
    
    def calculate_ma_slope(self, ma_series: pd.Series, window: int = 3) -> pd.Series:
        """计算移动平均线斜率
        
        Args:
            ma_series: 移动平均线数据
            window: 计算斜率的窗口大小
            
        Returns:
            斜率Series，值为百分比变化
        """
        # 计算简单差值
        ma_diff = ma_series.diff(1)
        # 使用前值计算百分比变化作为斜率
        slope = (ma_diff / ma_series.shift(1)) * 100
        # 平滑处理
        return slope.rolling(window=window).mean()
    
    def detect_high_confidence_weekly(self, df: pd.DataFrame) -> bool:
        """检测高置信档周线条件
        
        条件：周线收盘价在MA5周线上方，且MA5周线向上（MA5周线的斜率为正）
        
        Args:
            df: 周线数据
            
        Returns:
            是否满足高置信档条件
        """
        if len(df) < self.ma_period + 3:
            logger.warning(f"数据量不足，无法计算高置信档条件: {len(df)}条数据")
            return False
        
        # 计算MA5
        df['ma5'] = self.calculate_ma(df, self.ma_period)
        # 计算MA5斜率
        df['ma5_slope'] = self.calculate_ma_slope(df['ma5'])
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 条件1: 收盘价在MA5上方
        condition1 = latest['close'] > latest['ma5']
        # 条件2: MA5斜率为正
        condition2 = latest['ma5_slope'] > 0
        
        result = condition1 and condition2
        logger.info(f"高置信档条件检测: 收盘价在MA5上方={condition1}, MA5斜率为正={condition2}, 结果={result}")
        return result
    
    def detect_medium_confidence_weekly(self, df: pd.DataFrame) -> bool:
        """检测中置信档周线条件
        
        条件：周线收盘价在MA5周线下方，但不超过5%，且MA5周线斜率为0或略微向下（-0.5%至0%）
        
        Args:
            df: 周线数据
            
        Returns:
            是否满足中置信档条件
        """
        if len(df) < self.ma_period + 3:
            logger.warning(f"数据量不足，无法计算中置信档条件: {len(df)}条数据")
            return False
        
        # 计算MA5
        df['ma5'] = self.calculate_ma(df, self.ma_period)
        # 计算MA5斜率
        df['ma5_slope'] = self.calculate_ma_slope(df['ma5'])
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 条件1: 收盘价在MA5下方，但不超过5%
        price_diff_pct = ((latest['close'] - latest['ma5']) / latest['ma5']) * 100
        condition1 = -5 <= price_diff_pct <= 0
        # 条件2: MA5斜率为0或略微向下（-0.5%至0%）
        condition2 = -0.5 <= latest['ma5_slope'] <= 0
        
        result = condition1 and condition2
        logger.info(f"中置信档条件检测: 收盘价在MA5下方且不超过5%={condition1}, MA5斜率为0或略微向下={condition2}, 结果={result}")
        return result
    
    def analyze_weekly_condition(self, df: pd.DataFrame) -> Dict[str, any]:
        """分析周线条件，返回详细结果
        
        Args:
            df: 周线数据
            
        Returns:
            包含分析结果的字典
        """
        if df.empty:
            logger.error("周线数据为空，无法分析")
            return {
                'success': False,
                'error': '周线数据为空',
                'weekly_level': 'unknown',
                'confidence_score': 0.0
            }
        
        try:
            # 复制数据以避免修改原始数据
            df_copy = df.copy()
            
            # 计算必要指标
            df_copy['ma5'] = self.calculate_ma(df_copy, self.ma_period)
            df_copy['ma5_slope'] = self.calculate_ma_slope(df_copy['ma5'])
            
            latest = df_copy.iloc[-1]
            price_diff_pct = ((latest['close'] - latest['ma5']) / latest['ma5']) * 100 if not pd.isna(latest['ma5']) else 0
            
            # 检测分档条件
            is_high_confidence = self.detect_high_confidence_weekly(df_copy)
            is_medium_confidence = not is_high_confidence and self.detect_medium_confidence_weekly(df_copy)
            
            # 确定周线级别和置信度分数
            if is_high_confidence:
                weekly_level = 'high'
                confidence_score = 1.0
            elif is_medium_confidence:
                weekly_level = 'medium'
                confidence_score = 0.7
            else:
                weekly_level = 'low'
                confidence_score = 0.3
            
            result = {
                'success': True,
                'weekly_level': weekly_level,
                'confidence_score': confidence_score,
                'latest_data': {
                    'date': latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else str(latest['date']),
                    'close': latest['close'],
                    'ma5': float(latest['ma5']) if not pd.isna(latest['ma5']) else None,
                    'ma5_slope': float(latest['ma5_slope']) if not pd.isna(latest['ma5_slope']) else None,
                    'price_vs_ma5_pct': float(price_diff_pct)
                },
                'conditions': {
                    'price_above_ma5': latest['close'] > latest['ma5'] if not pd.isna(latest['ma5']) else False,
                    'ma5_slope_positive': latest['ma5_slope'] > 0 if not pd.isna(latest['ma5_slope']) else False,
                    'price_below_ma5_but_within_5pct': -5 <= price_diff_pct <= 0,
                    'ma5_slope_flat_or_slightly_down': -0.5 <= latest['ma5_slope'] <= 0 if not pd.isna(latest['ma5_slope']) else False
                }
            }
            
            logger.info(f"周线条件分析完成: 级别={weekly_level}, 置信度={confidence_score}")
            return result
            
        except Exception as e:
            logger.error(f"周线条件分析失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'weekly_level': 'unknown',
                'confidence_score': 0.0
            }

# 测试代码
if __name__ == "__main__":
    # 设置日志
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # 创建示例数据进行测试
    dates = pd.date_range(start='2025-09-01', periods=10, freq='W')
    data = {
        'date': dates,
        'open': range(100, 110),
        'high': range(102, 112),
        'low': range(98, 108),
        'close': [101, 103, 105, 107, 109, 111, 113, 115, 117, 119],  # 上升趋势
        'volume': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }
    df = pd.DataFrame(data)
    
    analyzer = WeeklyAnalyzer()
    result = analyzer.analyze_weekly_condition(df)
    print("\n高置信档测试结果:")
    print(result)
    
    # 测试中置信档
    data['close'] = [101, 103, 105, 107, 109, 108, 107, 106, 105, 104]  # 轻微下降
    df2 = pd.DataFrame(data)
    result2 = analyzer.analyze_weekly_condition(df2)
    print("\n中置信档测试结果:")
    print(result2)