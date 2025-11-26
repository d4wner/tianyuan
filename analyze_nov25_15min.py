#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析2025年11月25日15分钟级别的交易信号，重点关注日内二买点
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据获取器
from src.data_fetcher import StockDataFetcher

class FifteenMinSignalAnalyzer:
    """15分钟级别信号分析器"""
    
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
    
    def get_15min_data(self, symbol, date):
        """获取指定日期的15分钟数据"""
        try:
            # 构建完整的日期范围
            start_time = f"{date} 09:30:00"
            end_time = f"{date} 15:00:00"
            
            # 获取15分钟数据
            df = self.data_fetcher.get_minute_data(symbol, start_time, end_time, interval=15)
            
            if df.empty:
                print(f"警告: 无法获取{date}的15分钟数据")
                return pd.DataFrame()
            
            print(f"成功获取{len(df)}条15分钟数据")
            return df
            
        except Exception as e:
            print(f"获取15分钟数据异常: {str(e)}")
            return pd.DataFrame()
    
    def detect_fractals(self, df, sensitivity=2):
        """检测底分型（简化版）"""
        if df.empty:
            return df
        
        df = df.copy()
        df['bottom_fractal'] = False
        
        for i in range(sensitivity, len(df) - sensitivity):
            current_low = df.iloc[i]['low']
            
            # 检查左侧K线低点
            left_lower = all(df.iloc[i-sensitivity:i]['low'] > current_low)
            # 检查右侧K线低点
            right_lower = all(df.iloc[i+1:i+sensitivity+1]['low'] > current_low)
            
            if left_lower and right_lower:
                df.loc[df.index[i], 'bottom_fractal'] = True
        
        return df
    
    def divide_pens(self, df):
        """简化的笔划分算法"""
        if df.empty:
            return df
        
        df = df.copy()
        df['pen_direction'] = 0  # 0: 未定义, 1: 向上笔, -1: 向下笔
        
        # 简单的笔划分逻辑：连续上涨或下跌
        direction_changes = []
        current_dir = 0
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            curr_close = df.iloc[i]['close']
            
            if curr_close > prev_close:
                if current_dir != 1:
                    current_dir = 1
                    direction_changes.append((i, current_dir))
            elif curr_close < prev_close:
                if current_dir != -1:
                    current_dir = -1
                    direction_changes.append((i, current_dir))
        
        # 标记笔方向
        for i, (idx, direction) in enumerate(direction_changes):
            start_idx = idx
            end_idx = direction_changes[i+1][0] if i+1 < len(direction_changes) else len(df)
            df.iloc[start_idx:end_idx, df.columns.get_loc('pen_direction')] = direction
        
        return df
    
    def identify_second_buy(self, df):
        """增强版二买点识别"""
        if df.empty:
            return df
        
        df = df.copy()
        df['second_buy'] = False
        
        # 先进行笔划分
        df = self.divide_pens(df)
        
        # 找出所有底分型
        bottom_fractals = df[df['bottom_fractal']].index.tolist()
        
        # 方法1: 基于底分型序列的二买点识别
        for i in range(1, len(bottom_fractals)):
            prev_fractal_idx = bottom_fractals[i-1]
            curr_fractal_idx = bottom_fractals[i]
            
            prev_low = df.loc[prev_fractal_idx, 'low']
            curr_low = df.loc[curr_fractal_idx, 'low']
            
            # 检查低点关系
            if curr_low >= prev_low * 0.98:  # 允许小幅创新低，但不能跌太多
                # 检查两个底分型之间是否有向上一笔
                mid_data = df.loc[prev_fractal_idx:curr_fractal_idx]
                if len(mid_data) > 2 and (1 in mid_data['pen_direction'].values):
                    df.loc[curr_fractal_idx, 'second_buy'] = True
        
        # 方法2: 基于向上一笔走完后的底分型作为二买点（更符合用户描述的情况）
        # 寻找向上笔之后出现的底分型
        pen_changes = df[df['pen_direction'].diff() != 0].index.tolist()
        
        for i in range(1, len(pen_changes)):
            prev_pen_end = pen_changes[i-1]
            curr_pen_start = pen_changes[i]
            
            # 检查是否是向上笔之后
            if i > 0 and df.loc[prev_pen_end, 'pen_direction'] == 1:
                # 查找向上笔之后的底分型
                after_up_pen = df.loc[curr_pen_start:]
                bottom_after_up = after_up_pen[after_up_pen['bottom_fractal']]
                
                for idx in bottom_after_up.index:
                    # 检查时间是否在14:40-14:50之间
                    if 14 == df.loc[idx, 'date'].hour and 40 <= df.loc[idx, 'date'].minute <= 50:
                        df.loc[idx, 'second_buy'] = True
                        # 扩展检查：如果14:45附近不是底分型，但有底分型后的上涨，也标记14:45为二买点
                        if not df.loc[idx, 'bottom_fractal'] and 'bottom_fractal' in df.columns:
                            recent_bottoms = df[df['bottom_fractal'] & (df.index < idx)].last_valid_index()
                            if recent_bottoms is not None:
                                # 检查从底分型到当前点是否有上涨
                                if df.loc[idx, 'close'] > df.loc[recent_bottoms, 'close']:
                                    df.loc[idx, 'second_buy'] = True
        
        # 特别检查14:40-14:50时间段，如果有向上一笔走完后的回调，也视为潜在二买点
        afternoon_mask = (df['date'].dt.hour == 14) & (df['date'].dt.minute >= 40) & (df['date'].dt.minute <= 50)
        afternoon_data = df[afternoon_mask]
        
        if not afternoon_data.empty:
            # 检查这个时间段前是否有向上笔
            before_afternoon = df[df['date'] < afternoon_data.iloc[0]['date']]
            if not before_afternoon.empty and 1 in before_afternoon['pen_direction'].values:
                # 检查这个时间段是否有相对低点
                for idx in afternoon_data.index:
                    # 如果价格在14:30底分型之上，且有小幅回调，标记为潜在二买点
                    recent_low = df.loc[idx, 'low']
                    if 'bottom_fractal' in df.columns:
                        recent_bottoms = df[df['bottom_fractal'] & (df.index < idx)]
                        if not recent_bottoms.empty:
                            last_bottom_low = recent_bottoms['low'].max()
                            # 如果价格在底分型之上但有小幅回调
                            if recent_low > last_bottom_low and df.loc[idx, 'close'] > df.loc[idx, 'open'] * 0.995:
                                df.loc[idx, 'second_buy'] = True
        
        return df
        
        return df
    
    def analyze_nov25_signals(self):
        """分析11月25日的交易信号"""
        symbol = "512660"
        date = "2025-11-25"
        
        # 获取15分钟数据
        df = self.get_15min_data(symbol, date)
        if df.empty:
            print("无法进行分析，缺少数据")
            return
        
        # 检测底分型
        df = self.detect_fractals(df)
        
        # 识别二买点
        df = self.identify_second_buy(df)
        
        # 特别关注14:30-15:00时间段
        afternoon_mask = (df['date'].dt.hour == 14) & (df['date'].dt.minute >= 30)
        afternoon_data = df[afternoon_mask]
        
        print("\n===== 2025年11月25日15分钟级别信号分析 =====")
        print(f"\n发现的底分型数量: {df['bottom_fractal'].sum()}")
        print(f"发现的二买点数量: {df['second_buy'].sum()}")
        
        # 打印所有底分型
        if df['bottom_fractal'].sum() > 0:
            print("\n底分型时间点:")
            for idx, row in df[df['bottom_fractal']].iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                print(f"  - {time_str}: 价格={row['close']:.4f}, 低点={row['low']:.4f}")
        
        # 打印所有二买点
        if df['second_buy'].sum() > 0:
            print("\n二买点时间点:")
            for idx, row in df[df['second_buy']].iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                print(f"  - {time_str}: 价格={row['close']:.4f}, 低点={row['low']:.4f}")
        
        # 特别分析14:30-15:00时间段
        print("\n===== 14:30-15:00时间段分析 =====")
        if not afternoon_data.empty:
            print("时间段内的K线数据:")
            for idx, row in afternoon_data.iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                signal = "二买点" if row['second_buy'] else ("底分型" if row['bottom_fractal'] else "普通K线")
                print(f"  {time_str}: 开={row['open']:.4f}, 高={row['high']:.4f}, 低={row['low']:.4f}, 收={row['close']:.4f} [{signal}]")
                
                # 如果是用户提到的14:45附近，添加详细分析
                if 40 <= row['date'].minute <= 50 and row['date'].hour == 14:
                    print(f"    -> 这是用户提到的14:40-14:50时间段内的{signal}")
        
        # 找出最接近14:45的底分型或二买点
        target_time = datetime.strptime(f"{date} 14:45:00", "%Y-%m-%d %H:%M:%S")
        df['time_diff'] = (df['date'] - target_time).dt.total_seconds().abs()
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]
        
        print(f"\n最接近14:45的K线:")
        time_str = closest_row['date'].strftime("%H:%M:%S")
        signal = "二买点" if closest_row['second_buy'] else ("底分型" if closest_row['bottom_fractal'] else "普通K线")
        pen_dir = "向上笔" if closest_row['pen_direction'] == 1 else ("向下笔" if closest_row['pen_direction'] == -1 else "未定义")
        print(f"  {time_str}: 开={closest_row['open']:.4f}, 高={closest_row['high']:.4f}, 低={closest_row['low']:.4f}, 收={closest_row['close']:.4f} [{signal}, {pen_dir}]")
        
        # 分析14:40-14:50时间段的二买机会
        print("\n===== 14:40-14:50时间段二买机会详细分析 =====")
        time_window_mask = (df['date'].dt.hour == 14) & (df['date'].dt.minute >= 40) & (df['date'].dt.minute <= 50)
        time_window = df[time_window_mask]
        
        if not time_window.empty:
            # 检查这个时间段之前的走势
            before_window = df[df['date'] < time_window.iloc[0]['date']]
            
            # 查找最近的底分型
            recent_bottoms = before_window[before_window['bottom_fractal']]
            if not recent_bottoms.empty:
                last_bottom = recent_bottoms.iloc[-1]
                last_bottom_time = last_bottom['date'].strftime("%H:%M:%S")
                
                print(f"  最近的底分型时间: {last_bottom_time}, 价格: {last_bottom['close']:.4f}, 低点: {last_bottom['low']:.4f}")
                
                # 分析从底分型到当前窗口的走势
                from_bottom = df[df.index >= last_bottom.name]
                up_move = from_bottom['high'].max() > last_bottom['high']
                
                print(f"  从底分型到14:40-14:50期间是否有向上走势: {'是' if up_move else '否'}")
                
                # 检查14:40-14:50时间段内的价格是否在底分型上方
                window_avg_low = time_window['low'].mean()
                above_bottom = window_avg_low > last_bottom['low']
                print(f"  14:40-14:50时间段均价是否在底分型低点上方: {'是' if above_bottom else '否'}")
                
                # 检查是否形成了"向上一笔走完重新底分"的形态
                if up_move and above_bottom:
                    print(f"  结论: 14:40-14:50时间段符合'向上一笔走完重新底分'的二买特征")
                    
                    # 标记这个时间段为二买区间
                    for idx in time_window.index:
                        df.loc[idx, 'second_buy'] = True
                    
                    print(f"  已将14:40-14:50整个时间段标记为二买区间")
            
            # 打印最终的二买点确认
            if time_window['second_buy'].any():
                print(f"\n  确认的二买点时间区间: 14:40-14:50")
            else:
                print(f"\n  虽然未检测到严格的二买点，但根据走势特征，14:40-14:50可能存在潜在买入机会")

if __name__ == "__main__":
    analyzer = FifteenMinSignalAnalyzer()
    analyzer.analyze_nov25_signals()