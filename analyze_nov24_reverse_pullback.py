#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门分析11月24-25日军工ETF(512660)的反抽信号问题
验证为什么系统没有识别到明显的破中枢反抽机会
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta

# 设置日志，同时输出到文件
log_file = 'nov24_reverse_pullback_analysis.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('nov24_analysis')
logger.info(f"分析开始，日志将保存到: {log_file}")

class ReversePullbackAnalyzer:
    """反抽信号分析器"""
    
    def __init__(self, file_path):
        # 先读取看看实际的列名
        temp_df = pd.read_csv(file_path, nrows=1)
        logger.info(f"CSV文件列名: {list(temp_df.columns)}")
        
        # 直接读取并设置第一列为索引
        self.df = pd.read_csv(file_path)
        # 假设第一列是日期，直接处理
        first_col = self.df.columns[0]
        self.df['date'] = pd.to_datetime(self.df[first_col])
        self.df.set_index('date', inplace=True)
        logger.info(f"加载数据成功，共{len(self.df)}条记录")
        logger.info(f"日期范围: {self.df.index.min()} 至 {self.df.index.max()}")
    
    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD指标"""
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        return df
    
    def analyze_september_central_bank(self):
        """分析9月初的中枢"""
        # 获取9月初的数据（9月1日-9月15日）
        sept_data = self.df[(self.df.index >= '2025-09-01') & (self.df.index <= '2025-09-15')]
        
        if len(sept_data) == 0:
            logger.error("未找到9月初数据")
            return None
        
        central_top = sept_data['high'].max()
        central_bottom = sept_data['low'].min()
        central_mid = (central_top + central_bottom) / 2
        
        logger.info(f"9月初中枢分析：")
        logger.info(f"  日期范围: 2025-09-01 至 2025-09-15")
        logger.info(f"  中枢上沿: {central_top:.6f}")
        logger.info(f"  中枢下沿: {central_bottom:.6f}")
        logger.info(f"  中枢中点: {central_mid:.6f}")
        
        # 特别关注9月4日
        sep4_data = self.df[self.df.index == '2025-09-04']
        if not sep4_data.empty:
            logger.info(f"9月4日数据：")
            logger.info(f"  开盘价: {sep4_data['open'].iloc[0]:.6f}")
            logger.info(f"  最高价: {sep4_data['high'].iloc[0]:.6f}")
            logger.info(f"  最低价: {sep4_data['low'].iloc[0]:.6f}")
            logger.info(f"  收盘价: {sep4_data['close'].iloc[0]:.6f}")
        
        return central_top, central_bottom
    
    def analyze_november_breakdown(self, central_bottom):
        """分析11月的跌破情况"""
        # 获取11月数据
        nov_data = self.df[(self.df.index >= '2025-11-01') & (self.df.index <= '2025-11-30')]
        
        if len(nov_data) == 0:
            logger.error("未找到11月数据")
            return None
        
        # 检查跌破中枢的情况
        below_central = nov_data[nov_data['close'] < central_bottom]
        logger.info(f"11月跌破中枢({central_bottom:.6f})的天数: {len(below_central)}")
        
        for idx, row in below_central.iterrows():
            logger.info(f"  {idx.strftime('%Y-%m-%d')}: 收盘价={row['close']:.6f}, 最低价={row['low']:.6f}, 跌破幅度={(central_bottom - row['close']) / central_bottom * 100:.2f}%")
        
        # 特别关注11月20-25日
        critical_period = nov_data[(nov_data.index >= '2025-11-20') & (nov_data.index <= '2025-11-25')]
        logger.info(f"\n11月20-25日关键时期数据：")
        for idx, row in critical_period.iterrows():
            is_below = row['close'] < central_bottom
            logger.info(f"  {idx.strftime('%Y-%m-%d')}: 最高价={row['high']:.6f}, 最低价={row['low']:.6f}, 收盘价={row['close']:.6f}, 成交量={row['volume']:,}, 是否跌破中枢: {is_below}")
        
        # 检查是否创新低
        recent_low = nov_data['low'].min()
        recent_low_date = nov_data[nov_data['low'] == recent_low].index[0]
        logger.info(f"\n11月最低点位：")
        logger.info(f"  日期: {recent_low_date.strftime('%Y-%m-%d')}")
        logger.info(f"  最低点: {recent_low:.6f}")
        logger.info(f"  低于9月4日低点{(central_bottom - recent_low) / central_bottom * 100:.2f}%")
        
        return recent_low, recent_low_date
    
    def simulate_reverse_pullback_detection(self, central_top, central_bottom):
        """模拟系统反抽信号检测逻辑"""
        logger.info("\n===== 模拟系统反抽信号检测逻辑 =====")
        
        # 获取包含11月24-25日的足够数据
        analysis_data = self.df[(self.df.index >= '2025-10-01') & (self.df.index <= '2025-11-25')]
        analysis_data = self.calculate_macd(analysis_data)
        
        # 模拟系统检测的中枢识别（最近20-50天）
        if len(analysis_data) >= 50:
            central_range_main = analysis_data.iloc[-50:-20]
            central_bank_top_main = central_range_main['high'].max()
            central_bank_bottom_main = central_range_main['low'].min()
            logger.info(f"系统识别的主中枢（最近20-50天）：")
            logger.info(f"  中枢上沿: {central_bank_top_main:.6f}")
            logger.info(f"  中枢下沿: {central_bank_bottom_main:.6f}")
        else:
            logger.warning("数据不足50天，无法模拟主中枢识别")
        
        # 检查系统中枢与实际9月中枢的差异
        logger.info(f"\n系统中枢与实际9月中枢对比：")
        logger.info(f"  实际9月中枢下沿: {central_bottom:.6f}")
        if len(analysis_data) >= 50:
            logger.info(f"  系统识别中枢下沿: {central_bank_bottom_main:.6f}")
            logger.info(f"  差异: {(central_bottom - central_bank_bottom_main):.6f} ({(central_bottom - central_bank_bottom_main) / central_bottom * 100:.2f}%)")
        
        # 检测11月24日是否满足反抽信号条件
        nov24_data = analysis_data[analysis_data.index == '2025-11-24']
        if not nov24_data.empty:
            logger.info(f"\n11月24日反抽信号条件检查：")
            
            # 1. 检查是否跌破中枢
            has_below_central = nov24_data['close'].iloc[0] < central_bottom
            logger.info(f"  1. 跌破实际9月中枢: {has_below_central}")
            
            # 2. 检查是否企稳
            recent_5days = analysis_data.tail(5)
            if len(recent_5days) >= 3:
                stability_check = (recent_5days['close'].iloc[-1] >= recent_5days['close'].iloc[-3] and 
                                 recent_5days['close'].iloc[-2] >= recent_5days['close'].iloc[-3])
                logger.info(f"  2. 企稳结构（最近2天收盘价不低于倒数第三天）: {stability_check}")
            
            # 3. 检查是否站回中枢
            back_to_central = nov24_data['close'].iloc[0] > central_bottom * 0.90
            logger.info(f"  3. 站回中枢90%以上: {back_to_central}")
            logger.info(f"    当前价: {nov24_data['close'].iloc[0]:.6f}, 中枢90%: {central_bottom * 0.90:.6f}")
            
            # 4. MACD分析
            if 'macd_hist' in recent_5days.columns:
                recent_macd = recent_5days['macd_hist'].iloc[-3:]
                macd_improving = all(recent_macd[i] >= recent_macd[i-1] for i in range(1, len(recent_macd)))
                logger.info(f"  4. MACD柱状图是否改善: {macd_improving}")
                logger.info(f"    最近MACD柱状图: {recent_macd.values}")
        
        # 模拟超宽松条件检查
        logger.info(f"\n超宽松条件检查：")
        recent_3days = analysis_data.tail(3)
        if len(recent_3days) >= 3:
            # 条件1：最近2天连续上涨
            cond1 = (recent_3days['close'].iloc[-1] > recent_3days['close'].iloc[-2] and 
                   recent_3days['close'].iloc[-2] > recent_3days['close'].iloc[-3])
            logger.info(f"  1. 最近2天连续上涨: {cond1}")
            
            # 条件2：最近1天大幅上涨
            daily_change = (recent_3days['close'].iloc[-1] - recent_3days['close'].iloc[-2]) / recent_3days['close'].iloc[-2] * 100
            cond2 = daily_change > 2.0
            logger.info(f"  2. 最近1天大幅上涨(>{2.0}%): {cond2} ({daily_change:.2f}%)")
    
    def analyze_system_issues(self):
        """分析系统可能存在的问题"""
        logger.info("\n===== 系统问题分析 =====")
        
        # 1. 中枢识别窗口问题
        logger.info("1. 中枢识别窗口问题：")
        logger.info("   - 系统使用最近20-50天作为主中枢识别窗口")
        logger.info("   - 9月初的中枢可能已超出这个窗口范围")
        logger.info("   - 导致系统无法正确识别历史重要中枢")
        
        # 2. 信号优先级问题
        logger.info("\n2. 信号优先级问题：")
        logger.info("   - 反抽信号是'兜底买点'，优先级最低")
        logger.info("   - 系统可能因为其他信号逻辑干扰而忽略了反抽信号")
        
        # 3. 代码逻辑问题
        logger.info("\n3. 可能的代码逻辑问题：")
        logger.info("   - 注释显示'移除了调试用的强制返回True逻辑'")
        logger.info("   - 可能之前有更宽松的逻辑来捕获反抽信号")
        
        # 4. 改进建议
        logger.info("\n4. 改进建议：")
        logger.info("   - 增加历史中枢识别机制，不仅依赖最近窗口")
        logger.info("   - 对于明显跌破重要历史低点的情况，应特别处理")
        logger.info("   - 优化反抽信号检测逻辑，使其更敏感于重要的价格突破")
        logger.info("   - 考虑将重要历史低点作为特殊的参考点位")

def main():
    """主函数"""
    try:
        # 数据文件路径
        data_file = '/Users/pingan/tools/trade/tianyuan/data/512660_daily_data.csv'
        
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在: {data_file}")
            return
        
        # 初始化分析器
        analyzer = ReversePullbackAnalyzer(data_file)
        
        # 分析9月初中枢
        central_top, central_bottom = analyzer.analyze_september_central_bank()
        
        # 分析11月跌破情况
        analyzer.analyze_november_breakdown(central_bottom)
        
        # 模拟系统反抽信号检测
        analyzer.simulate_reverse_pullback_detection(central_top, central_bottom)
        
        # 分析系统问题
        analyzer.analyze_system_issues()
        
        logger.info("\n分析完成！请查看详细日志文件: nov24_reverse_pullback_analysis.log")
        
        # 输出关键结论到控制台
        print("\n===== 关键结论摘要 =====")
        print("1. 中枢识别窗口问题：系统使用最近20-50天窗口，导致9月初中枢可能未被正确识别")
        print("2. 信号优先级问题：反抽信号作为'兜底买点'，优先级最低")
        print("3. 代码逻辑问题：可能移除了更宽松的反抽信号捕获逻辑")
        print("4. 建议改进：增加历史中枢识别，特别处理重要历史低点的突破")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()