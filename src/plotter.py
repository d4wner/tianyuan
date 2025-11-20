# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
from datetime import datetime

logger = logging.getLogger('ChanlunPlotter')

class ChanlunPlotter:
    """缠论图表绘制器"""
    
    def __init__(self, config=None):
        """
        初始化绘制器
        :param config: 绘图配置
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'outputs/plots')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("缠论绘图器初始化完成")
    
    def plot(self, df, symbol):
        """
        绘制缠论图表
        :param df: 包含缠论指标的DataFrame
        :param symbol: 股票代码
        """
        if df.empty:
            logger.warning("数据为空，无法绘图")
            return
            
        try:
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制价格曲线
            plt.plot(df['date'], df['close'], label='Close Price', color='blue')
            
            # 标记分型
            if 'top_fractal' in df.columns:
                top_fractals = df[df['top_fractal']]
                plt.scatter(top_fractals['date'], top_fractals['high'], 
                           marker='v', color='red', label='Top Fractal')
            
            if 'bottom_fractal' in df.columns:
                bottom_fractals = df[df['bottom_fractal']]
                plt.scatter(bottom_fractals['date'], bottom_fractals['low'], 
                           marker='^', color='green', label='Bottom Fractal')
            
            # 标记笔
            if 'pen_start' in df.columns:
                pen_starts = df[df['pen_start']]
                plt.scatter(pen_starts['date'], pen_starts['close'], 
                           marker='o', color='purple', label='Pen Start')
            
            if 'pen_end' in df.columns:
                pen_ends = df[df['pen_end']]
                plt.scatter(pen_ends['date'], pen_ends['close'], 
                           marker='s', color='orange', label='Pen End')
            
            # 标记中枢
            if 'central_bank' in df.columns:
                central_banks = df[df['central_bank']]
                for idx, row in central_banks.iterrows():
                    plt.axvspan(row['date'], row['date'] + pd.Timedelta(days=1), 
                               alpha=0.2, color='gray')
            
            # 设置图表属性
            plt.title(f"缠论分析 - {symbol}")
            plt.xlabel("日期")
            plt.ylabel("价格")
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            logger.info(f"图表已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}")
    
    def plot_minute(self, df, symbol, period):
        """
        绘制分钟数据图表
        :param df: 分钟数据DataFrame
        :param symbol: 股票代码
        :param period: 分钟周期
        """
        if df.empty:
            logger.warning("分钟数据为空，无法绘图")
            return
            
        try:
            # 创建图表
            plt.figure(figsize=(15, 8))
            
            # 绘制价格曲线
            plt.plot(df['date'], df['close'], label='Close Price', color='blue')
            
            # 设置图表属性
            plt.title(f"{symbol} {period}分钟图")
            plt.xlabel("时间")
            plt.ylabel("价格")
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{period}_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            logger.info(f"分钟图表已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"分钟数据绘图失败: {str(e)}")