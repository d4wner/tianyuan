#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析2025年11月24日15分钟级别的交易信号，重点关注日内买入点
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
        df['top_fractal'] = False
        
        # 检测底分型
        for i in range(sensitivity, len(df) - sensitivity):
            current_low = df.iloc[i]['low']
            # 检查左侧K线低点
            left_lower = all(df.iloc[i-sensitivity:i]['low'] > current_low)
            # 检查右侧K线低点
            right_lower = all(df.iloc[i+1:i+sensitivity+1]['low'] > current_low)
            
            if left_lower and right_lower:
                df.loc[df.index[i], 'bottom_fractal'] = True
        
        # 检测顶分型
        for i in range(sensitivity, len(df) - sensitivity):
            current_high = df.iloc[i]['high']
            # 检查左侧K线高点
            left_higher = all(df.iloc[i-sensitivity:i]['high'] < current_high)
            # 检查右侧K线高点
            right_higher = all(df.iloc[i+1:i+sensitivity+1]['high'] < current_high)
            
            if left_higher and right_higher:
                df.loc[df.index[i], 'top_fractal'] = True
        
        return df
    
    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD指标"""
        if df.empty:
            return df
        
        df = df.copy()
        # 计算指数移动平均线
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        df['macd'] = df['ema_fast'] - df['ema_slow']
        # 计算信号线
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        # 计算柱状图
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def identify_buy_signals(self, df):
        """识别买入信号"""
        if df.empty:
            return df
        
        df = df.copy()
        df['buy_signal'] = False
        
        # 1. 底分型买入信号
        df.loc[df['bottom_fractal'], 'buy_signal'] = True
        
        # 2. MACD金叉买入信号
        df['macd_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df.loc[df['macd_cross'], 'buy_signal'] = True
        
        # 3. MACD柱状图由负转正或加深
        df['hist_improve'] = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'] < 0)
        df.loc[df['hist_improve'], 'buy_signal'] = True
        
        # 4. 价格突破前高
        df['price_break'] = df['close'] > df['high'].rolling(window=3).max().shift(1)
        df.loc[df['price_break'], 'buy_signal'] = True
        
        return df
    
    def analyze_nov24_signals(self):
        """分析11月24日的交易信号"""
        symbol = "512660"
        date = "2025-11-24"
        
        # 获取15分钟数据
        df = self.get_15min_data(symbol, date)
        if df.empty:
            print("无法进行分析，缺少数据")
            return
        
        # 检测分型
        df = self.detect_fractals(df)
        
        # 计算MACD
        df = self.calculate_macd(df)
        
        # 识别买入信号
        df = self.identify_buy_signals(df)
        
        print("\n===== 2025年11月24日15分钟级别信号分析 =====")
        print(f"\n发现的底分型数量: {df['bottom_fractal'].sum()}")
        print(f"发现的顶分型数量: {df['top_fractal'].sum()}")
        print(f"发现的买入信号数量: {df['buy_signal'].sum()}")
        
        # 打印所有买入信号
        if df['buy_signal'].sum() > 0:
            print("\n买入信号时间点:")
            for idx, row in df[df['buy_signal']].iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                
                # 确定信号类型
                signals = []
                if row['bottom_fractal']:
                    signals.append("底分型")
                if row['macd_cross']:
                    signals.append("MACD金叉")
                if row['hist_improve']:
                    signals.append("MACD柱状图改善")
                if row['price_break']:
                    signals.append("价格突破")
                
                signal_str = ", ".join(signals)
                print(f"  - {time_str}: 价格={row['close']:.4f}, 低点={row['low']:.4f}, 信号类型: {signal_str}")
        
        # 分析全天走势阶段
        print("\n===== 全天走势阶段分析 =====")
        # 分时段分析
        time_periods = [
            (9, 30, 10, 30, "早盘第一小时"),
            (10, 30, 11, 30, "早盘第二小时"),
            (13, 00, 14, 00, "下午第一小时"),
            (14, 00, 15, 00, "尾盘小时")
        ]
        
        best_buy_period = None
        best_buy_score = 0
        
        for start_hour, start_min, end_hour, end_min, period_name in time_periods:
            mask = (
                (df['date'].dt.hour > start_hour) | 
                ((df['date'].dt.hour == start_hour) & (df['date'].dt.minute >= start_min))
            ) & (
                (df['date'].dt.hour < end_hour) | 
                ((df['date'].dt.hour == end_hour) & (df['date'].dt.minute < end_min))
            )
            
            period_data = df[mask]
            if not period_data.empty:
                buy_signals_in_period = period_data['buy_signal'].sum()
                price_change = period_data['close'].iloc[-1] - period_data['close'].iloc[0]
                
                # 计算买入信号得分
                buy_score = buy_signals_in_period * 3 + (price_change > 0) * 2
                
                print(f"\n{period_name} ({start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}):")
                print(f"  买入信号数量: {buy_signals_in_period}")
                print(f"  价格变动: {price_change:.4f}")
                print(f"  买入适宜度评分: {buy_score}")
                
                # 更新最佳买入时段
                if buy_score > best_buy_score:
                    best_buy_score = buy_score
                    best_buy_period = period_name
        
        # 输出最佳买入时段
        if best_buy_period:
            print(f"\n最佳买入时段: {best_buy_period} (得分: {best_buy_score})")
        
        # 分析14:00-15:00时段的详细信号（重点关注尾盘）
        print("\n===== 尾盘(14:00-15:00)详细信号分析 =====")
        afternoon_mask = (df['date'].dt.hour >= 14) & (df['date'].dt.hour < 15)
        afternoon_data = df[afternoon_mask]
        
        if not afternoon_data.empty:
            print("尾盘K线数据:")
            for idx, row in afternoon_data.iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                signal = "买入信号" if row['buy_signal'] else "普通K线"
                print(f"  {time_str}: 开={row['open']:.4f}, 高={row['high']:.4f}, 低={row['low']:.4f}, 收={row['close']:.4f} [{signal}], MACD柱状图={row['macd_hist']:.6f}")
        
        # 检查是否存在日内背驰信号
        print("\n===== 日内背驰信号检查 =====")
        # 寻找价格创新低但MACD未创新低的情况
        df['price_low_innovate'] = df['low'] < df['low'].rolling(window=5).min().shift(1)
        df['macd_low_innovate'] = df['macd'] < df['macd'].rolling(window=5).min().shift(1)
        df['divergence'] = df['price_low_innovate'] & (~df['macd_low_innovate'])
        
        if df['divergence'].any():
            print("发现日内背驰信号:")
            for idx, row in df[df['divergence']].iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                print(f"  - {time_str}: 价格创新低但MACD未创新低")
        else:
            print("未发现明确的日内背驰信号")
        
        # 总结最佳买入时机
        print("\n===== 最佳买入时机总结 =====")
        # 优先考虑同时具备多种买入信号的时间点
        df['signal_count'] = df[['bottom_fractal', 'macd_cross', 'hist_improve', 'price_break']].sum(axis=1)
        
        if df['signal_count'].max() > 0:
            best_buy_signals = df[df['signal_count'] == df['signal_count'].max()]
            
            print(f"最强烈的买入信号出现在:")
            for idx, row in best_buy_signals.iterrows():
                time_str = row['date'].strftime("%H:%M:%S")
                print(f"  - {time_str}: {int(row['signal_count'])}个信号共振")
        
        # 最终建议买入时间段
        if df['buy_signal'].any():
            first_buy = df[df['buy_signal']].iloc[0]
            last_buy = df[df['buy_signal']].iloc[-1]
            
            first_time = first_buy['date'].strftime("%H:%M")
            last_time = last_buy['date'].strftime("%H:%M")
            
            print(f"\n建议买入时间区间: {first_time}-{last_time}")
        
        # 检查日线级别的背驰是否有对应的日内体现
        if 'divergence' in df.columns and df['divergence'].any():
            print("\n注意: 日内存在背驰信号，与日线级别的背驰相呼应")

if __name__ == "__main__":
    analyzer = FifteenMinSignalAnalyzer()
    analyzer.analyze_nov24_signals()