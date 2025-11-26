#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日线分析模块 - 实现日线核心条件检测"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('DailyAnalyzer')
logger.setLevel(logging.INFO)

class DailyAnalyzer:
    """日线分析器，实现日线核心条件检测"""
    
    def __init__(self, min_pen_length: int = 5, divergence_threshold: float = 0.3):
        """初始化日线分析器
        
        Args:
            min_pen_length: 笔的最小K线数量（缠论标准为至少5根K线）
            divergence_threshold: 背驰阈值
        """
        self.min_pen_length = min_pen_length
        self.divergence_threshold = divergence_threshold
    
    def identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别顶底分型
        
        Args:
            df: 日线数据
            
        Returns:
            添加了顶底分型标记的数据框
        """
        df = df.copy()
        df['top_fractal'] = False
        df['bottom_fractal'] = False
        
        # 检查数据量是否足够识别分型
        if len(df) < 5:
            logger.warning(f"数据量不足，无法识别分型：只有{len(df)}行数据")
            return df
        
        for i in range(2, len(df) - 2):
            # 顶分型：中间K线最高价高于两侧各两根K线的最高价
            if (df.iloc[i]['high'] > df.iloc[i-2]['high'] and 
                df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                df.at[df.index[i], 'top_fractal'] = True
            
            # 底分型：中间K线最低价低于两侧各两根K线的最低价
            if (df.iloc[i]['low'] < df.iloc[i-2]['low'] and 
                df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                df.at[df.index[i], 'bottom_fractal'] = True
        
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
        
        # 提取所有顶底分型的位置索引（使用numpy.where来获取位置索引）
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
                            'start_date': df.iloc[start_pos]['date'],
                            'end_date': df.iloc[end_pos]['date'],
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
        """识别中枢
        
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
                    central_bank = {
                        'start_idx': pen1['start_idx'],
                        'end_idx': pen3['end_idx'],
                        'start_date': pen1['start_date'],
                        'end_date': pen3['end_date'],
                        'high': high,
                        'low': low,
                        'middle': (high + low) / 2,
                        'range_pct': ((high - low) / low) * 100,
                        'pen_count': 3
                    }
                    central_banks.append(central_bank)
        
        return central_banks
    
    def detect_divergence(self, df: pd.DataFrame, pens: List[Dict]) -> Tuple[bool, float]:
        """检测背驰
        
        Args:
            df: 日线数据
            pens: 笔的列表
            
        Returns:
            (是否背驰, 背驰强度)
        """
        # 计算MACD指标用于背驰检测
        df = self._calculate_macd(df)
        
        # 打印所有笔的详细信息，特别是11月21-23日相关的笔
        logger.info("\n=== 所有笔的详细信息 ===")
        for i, pen in enumerate(pens):
            pen_length = pen['end_idx'] - pen['start_idx'] + 1
            logger.info(f"笔{i+1}: 类型={pen['type']}, 起始价格={pen['start_price']}, 结束价格={pen['end_price']}, "
                      f"涨幅={pen['price_change_pct']:.2f}%, 长度={pen_length}根K线, "
                      f"起始日期={pen['start_date']}, 结束日期={pen['end_date']}")
        
        # 使用DivergenceDetector类检测背驰
        from src.divergence_detector import DivergenceDetector
        detector = DivergenceDetector(self.divergence_threshold)
        
        # 方法1: 传统基于向下笔的背驰检测
        down_pens_divergence, down_pens_strength = detector.detect_down_pens_divergence(df, pens)
        
        # 方法2: 基于9-11月MACD绿柱区域的非连续背驰检测
        green_zones_divergence, green_zones_strength = detector.detect_green_zones_divergence(df)
        
        logger.info(f"\n背驰检测汇总: 向下笔背驰={down_pens_divergence}, 绿柱区域背驰={green_zones_divergence}")
        
        # 只要有一个方法检测到背驰，就认为存在背驰
        if down_pens_divergence or green_zones_divergence:
            # 返回较大的背驰强度
            max_strength = max(down_pens_strength, green_zones_strength)
            logger.info(f"最终背驰检测结果: 存在背驰, 最大背驰强度={max_strength:.4f}")
            return True, max_strength
        
        logger.info("最终背驰检测结果: 不存在背驰")
        return False, 0.0
    
    # _detect_down_pens_divergence方法已移至DivergenceDetector类中实现
    
    # _detect_green_zones_divergence方法已移至DivergenceDetector类中实现
    
    def _analyze_green_zones_comparison(self, macd_period: pd.DataFrame, green_periods: List[Dict]):
        """分析9月份与10-11月份绿柱区域的对比"""
        logger.info("\n=== 9月份与10-11月份绿柱区域对比分析 ===")
        
        # 分离9月份和10-11月份的绿柱区域
        sep_green_periods = []
        oct_nov_green_periods = []
        
        for period in green_periods:
            if period['start_date'].month == 9:
                sep_green_periods.append(period)
            elif period['start_date'].month in [10, 11]:
                oct_nov_green_periods.append(period)
        
        # 计算9月份绿柱统计
        sep_total_length = 0
        sep_total_area = 0
        for period in sep_green_periods:
            # 使用日期范围计算正确的长度
            mask = (macd_period['date'] >= period['start_date']) & (macd_period['date'] <= period['end_date'])
            sep_total_length += len(macd_period[mask])
            # 使用DivergenceDetector的MACD面积计算方法
            from src.divergence_detector import DivergenceDetector
            detector = DivergenceDetector(self.divergence_threshold)
            sep_total_area += detector._calculate_macd_area(macd_period, period['start_date'], period['end_date'])
        
        # 计算10-11月份绿柱统计
        oct_nov_total_length = 0
        oct_nov_total_area = 0
        for period in oct_nov_green_periods:
            # 使用日期范围计算正确的长度
            mask = (macd_period['date'] >= period['start_date']) & (macd_period['date'] <= period['end_date'])
            oct_nov_total_length += len(macd_period[mask])
            # 使用DivergenceDetector的MACD面积计算方法
            from src.divergence_detector import DivergenceDetector
            detector = DivergenceDetector(self.divergence_threshold)
            oct_nov_total_area += detector._calculate_macd_area(macd_period, period['start_date'], period['end_date'])
        
        logger.info(f"9月份绿柱区域: {len(sep_green_periods)}个区域, 总天数={sep_total_length}, 总面积={sep_total_area:.6f}")
        logger.info(f"10-11月份绿柱区域: {len(oct_nov_green_periods)}个区域, 总天数={oct_nov_total_length}, 总面积={oct_nov_total_area:.6f}")
        
        # 计算比例和差异
        if sep_total_area > 0:
            area_ratio = oct_nov_total_area / sep_total_area
            logger.info(f"10-11月/9月绿柱面积比: {area_ratio:.4f}")
            if area_ratio < 1:
                logger.info(f"注意: 10-11月份绿柱总面积比9月份小{(1 - area_ratio) * 100:.2f}%")
        
        # 分析是否符合背驰条件
        sep_prices = macd_period[macd_period['date'].dt.month == 9]['close']
        oct_nov_prices = macd_period[macd_period['date'].dt.month.isin([10, 11])]['close']
        
        if not sep_prices.empty and not oct_nov_prices.empty:
            sep_low = sep_prices.min()
            oct_nov_low = oct_nov_prices.min()
            
            logger.info(f"9月份最低收盘价: {sep_low:.2f}")
            logger.info(f"10-11月份最低收盘价: {oct_nov_low:.2f}")
            
            if oct_nov_low < sep_low and oct_nov_total_area < sep_total_area:
                logger.info("⚠️ 符合底背驰条件: 价格创新低但MACD绿柱面积减小")
            else:
                logger.info("不符合底背驰条件")
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # 输出9-11月的MACD数据，特别是绿柱
        if 'date' in df.columns:
            macd_period = df[(df['date'] >= pd.to_datetime('2025-09-01')) & (df['date'] <= pd.to_datetime('2025-11-30'))]
            logger.info("\n=== 9-11月MACD数据详情 ===")
            for idx, row in macd_period.iterrows():
                if not np.isnan(row['macd_hist']):
                    color = '绿柱' if row['macd_hist'] < 0 else '红柱' if row['macd_hist'] > 0 else '零'
                    if color == '绿柱':
                        logger.info(f"日期: {row['date'].strftime('%Y-%m-%d')}, MACD柱状: {row['macd_hist']:.6f} ({color})")
            
            # 统计绿柱区域
            green_periods = []
            current_period = None
            for idx, row in macd_period.iterrows():
                if not np.isnan(row['macd_hist']) and row['macd_hist'] < 0:
                    if current_period is None:
                        current_period = {'start': idx, 'end': idx, 'start_date': row['date'], 'end_date': row['date']}
                    else:
                        current_period['end'] = idx
                        current_period['end_date'] = row['date']
                else:
                    if current_period is not None:
                        green_periods.append(current_period)
                        current_period = None
            if current_period is not None:
                green_periods.append(current_period)
            
            logger.info("\n=== 9-11月绿柱区域统计 ===")
            for i, period in enumerate(green_periods, 1):
                start_date = period['start_date']
                end_date = period['end_date']
                # 重新计算正确的长度（使用日期数量）
                mask = (macd_period['date'] >= start_date) & (macd_period['date'] <= end_date)
                length = len(macd_period[mask])
                # 使用DivergenceDetector的MACD面积计算方法
                from src.divergence_detector import DivergenceDetector
                detector = DivergenceDetector(self.divergence_threshold)
                total_area = detector._calculate_macd_area(macd_period, start_date, end_date)
                logger.info(f"绿柱区域{i}: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}, 长度: {length}天, 总面积: {total_area:.6f}")
            
            # 对比9月份与10-11月份绿柱区域
            self._analyze_green_zones_comparison(macd_period, green_periods)
        
        return df
    
    # MACD面积计算方法已移至DivergenceDetector类中，在调用处直接使用该类的方法
    
    def check_recent_up_pen(self, pens: List[Dict]) -> bool:
        """检查最近的向上笔是否满足条件（涨幅不低于0.5%）
        
        Args:
            pens: 笔的列表
            
        Returns:
            是否满足条件
        """
        # 找到最近的向上笔
        up_pens = [p for p in pens if p['type'] == 'up']
        if not up_pens:
            return False
        
        recent_up_pen = up_pens[-1]
        condition = recent_up_pen['price_change_pct'] >= 0.5
        logger.info(f"向上笔条件检查: 涨幅={recent_up_pen['price_change_pct']:.2f}%, 结果={condition}")
        return condition
    
    def check_recent_down_pen(self, pens: List[Dict]) -> bool:
        """检查最近的向下笔是否满足条件
        
        注意：回调比例阈值（不超过前一个向上笔涨幅的30%）是自定义条件，不是缠论标准
        缠论判断笔/走势的核心是K线结构、高低点重叠性和级别联动，而非幅度比例
        
        Args:
            pens: 笔的列表
            
        Returns:
            是否满足条件
        """
        # 需要至少2笔，且最近一笔是向下笔，前一笔是向上笔
        if len(pens) < 2:
            logger.info("向下笔条件检查: 笔数量不足2笔，返回False")
            return False
        
        recent_pen = pens[-1]
        logger.info(f"向下笔条件检查: 最近一笔类型={recent_pen['type']}, 起始日期={recent_pen['start_date']}, 结束日期={recent_pen['end_date']}")
        if recent_pen['type'] != 'down':
            logger.info(f"向下笔条件检查: 最近一笔不是向下笔(type={recent_pen['type']})，返回False")
            return False
        
        # 找到最近的向上笔
        up_pens = [p for p in pens if p['type'] == 'up']
        if not up_pens:
            logger.info("向下笔条件检查: 没有找到向上笔，返回False")
            return False
        
        recent_up_pen = up_pens[-1]
        recent_down_pen = pens[-1]
        
        # 计算回调比例
        callback_ratio = abs(recent_down_pen['price_change_pct'] / recent_up_pen['price_change_pct']) if recent_up_pen['price_change_pct'] != 0 else 0
        condition = callback_ratio <= 0.3
        
        # 添加详细日志，输出日期信息
        up_start_date = recent_up_pen.get('start_date', '未知')
        up_end_date = recent_up_pen.get('end_date', '未知')
        down_start_date = recent_down_pen.get('start_date', '未知')
        down_end_date = recent_down_pen.get('end_date', '未知')
        
        logger.info(f"向下笔条件详细分析：")
        logger.info(f"  最近向上笔：开始日期={up_start_date}, 结束日期={up_end_date}, 涨幅={recent_up_pen['price_change_pct']:.2f}%")
        logger.info(f"  最近向下笔：开始日期={down_start_date}, 结束日期={down_end_date}, 跌幅={recent_down_pen['price_change_pct']:.2f}%")
        logger.info(f"  回调比例={callback_ratio:.2f}, 结果={condition}")
        
        return condition
    
    def check_recent_bottom_fractal(self, df: pd.DataFrame) -> bool:
        """检查最近是否有底分型
        
        Args:
            df: 日线数据
            
        Returns:
            是否有最近的底分型
        """
        # 检查最近10根K线是否有底分型
        recent_df = df.tail(min(10, len(df)))
        has_bottom_fractal = recent_df['bottom_fractal'].any()
        logger.info(f"底分型检查: 最近10根K线是否有底分型={has_bottom_fractal}")
        return has_bottom_fractal
    
    def analyze_daily_conditions(self, df: pd.DataFrame) -> Dict[str, any]:
        """分析日线核心条件
        
        Args:
            df: 日线数据
            
        Returns:
            包含分析结果的字典
        """
        if df.empty:
            logger.error("日线数据为空，无法分析")
            return {
                'success': False,
                'error': '日线数据为空',
                'conditions_met': 0,
                'total_conditions': 5
            }
        
        try:
            df_copy = df.copy()
            
            # 1. 识别顶底分型
            df_copy = self.identify_fractals(df_copy)
            
            # 识别笔
            pens = self.identify_pens(df_copy)
            
            # 记录所有笔的详细信息
            logger.info("\n=== 所有笔的详细信息 ===")
            for i, pen in enumerate(pens):
                # 计算笔的长度（K线数量）
                pen_length = pen['end_idx'] - pen['start_idx'] + 1
                logger.info(f"笔{i+1}: 类型={pen['type']}, 起始日期={pen['start_date']}, 结束日期={pen['end_date']}, "
                          f"涨幅={pen['price_change_pct']:.2f}%, 长度={pen_length}根K线")
            
            # 3. 识别中枢
            central_banks = self.identify_central_banks(pens)
            
            # 4. 检测背驰
            has_divergence, divergence_strength = self.detect_divergence(df_copy, pens)
            
            # 5. 检查各项条件
            conditions = {
                'recent_up_pen': self.check_recent_up_pen(pens),
                'recent_down_pen': self.check_recent_down_pen(pens),
                'has_central_bank': len(central_banks) > 0,
                'has_divergence': has_divergence,
                'has_bottom_fractal': self.check_recent_bottom_fractal(df_copy)
            }
            
            # 统计满足的条件数
            conditions_met = sum(conditions.values())
            total_conditions = len(conditions)
            
            # 判断是否满足买入信号条件（至少满足4个条件）
            buy_signal = conditions_met >= 4
            
            result = {
                'success': True,
                'conditions_met': conditions_met,
                'total_conditions': total_conditions,
                'buy_signal': buy_signal,
                'conditions': conditions,
                'divergence_strength': divergence_strength,
                'pen_count': len(pens),
                'central_bank_count': len(central_banks),
                'details': {
                    'pens': pens[-5:] if pens else [],  # 返回最近5个笔
                    'central_banks': central_banks[-2:] if central_banks else [],  # 返回最近2个中枢
                    'recent_fractals': {
                        'top_count': df_copy.tail(20)['top_fractal'].sum(),
                        'bottom_count': df_copy.tail(20)['bottom_fractal'].sum()
                    }
                }
            }
            
            logger.info(f"日线条件分析完成: 满足{conditions_met}/{total_conditions}个条件, 买入信号={buy_signal}")
            return result
            
        except Exception as e:
            logger.error(f"日线条件分析失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'conditions_met': 0,
                'total_conditions': 5
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
    
    analyzer = DailyAnalyzer()
    result = analyzer.analyze_daily_conditions(df)
    print("日线条件分析结果:")
    print(result)