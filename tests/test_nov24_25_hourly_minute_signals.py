import sys
import os
import logging
import pandas as pd
import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import StockDataFetcher
from hourly_signal_detector import HourlySignalDetector
from daily_buy_signal_detector import BuySignalDetector

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def test_nov24_25_hourly_minute_signals():
    """检测512660在11月24日和25日的小时/分钟级别买入信号"""
    logging.info("开始检测军工ETF(512660)11月24-25日的小时/分钟级别买入信号")
    
    # 初始化检测器和数据获取器
    data_fetcher = StockDataFetcher()
    hourly_detector = HourlySignalDetector()
    daily_detector = BuySignalDetector()
    
    # 定义测试日期范围
    start_date = '2025-11-20'
    end_date = '2025-11-26'
    test_dates = ['2025-11-24', '2025-11-25']
    
    # 获取日线数据作为基础
    daily_data = data_fetcher.get_daily_data('512660', start_date=start_date, end_date=end_date)
    
    if daily_data is None or len(daily_data) == 0:
        logging.error("未获取到日线数据")
        return
    
    logging.info(f"成功获取日线数据，共{len(daily_data)}条记录")
    
    # 获取小时线数据
    hourly_data = data_fetcher.get_hourly_data('512660', start_date=start_date, end_date=end_date)
    
    if hourly_data is None or len(hourly_data) == 0:
        logging.error("未获取到小时线数据")
        return
    
    logging.info(f"成功获取小时线数据，共{len(hourly_data)}条记录")
    
    # 获取分钟线数据
    minute_data = data_fetcher.get_minute_data('512660', start_date=start_date, end_date=end_date)
    
    if minute_data is None or len(minute_data) == 0:
        logging.error("未获取到分钟线数据")
        return
    
    logging.info(f"成功获取分钟线数据，共{len(minute_data)}条记录")
    
    # 按日期分别测试
    for test_date in test_dates:
        logging.info(f"\n{'='*60}")
        logging.info(f"测试日期：{test_date}")
        logging.info(f"{'='*60}")
        
        # 获取当天的日线数据
        daily_on_date = daily_data[daily_data['date'] == pd.to_datetime(test_date)]
        if not daily_on_date.empty:
            logging.info(f"日线数据 - 开盘: {daily_on_date.iloc[0]['open']:.3f}, 收盘: {daily_on_date.iloc[0]['close']:.3f}, \
                     最低: {daily_on_date.iloc[0]['low']:.3f}, 最高: {daily_on_date.iloc[0]['high']:.3f}")
        
        # 检测日线信号
        daily_df_subset = daily_data[daily_data['date'] <= pd.to_datetime(test_date)]
        daily_signal, daily_info = daily_detector.detect_daily_first_buy(daily_df_subset)
        if daily_signal:
            logging.info("✅ 日线级别一买信号：是")
            logging.info(f"   信号详情: {daily_info}")
        else:
            logging.info("❌ 日线级别一买信号：否")
        
        # 检测小时线信号
        logging.info("\n小时线级别信号检测:")
        hourly_on_date = hourly_data[hourly_data['date'].dt.date == pd.to_datetime(test_date).date()]
        
        if hourly_on_date.empty:
            logging.info("   无小时线数据")
        else:
            logging.info(f"   当天小时线数据条数: {len(hourly_on_date)}")
            
            # 遍历每小时检测信号
            hourly_buy_times = []
            
            # 整体检测底分型
            hourly_df_with_fractal = hourly_detector.detect_hourly_bottom_fractal(hourly_on_date)
            
            # 查找有底分型标记的行
            bottom_fractal_signals = hourly_df_with_fractal[hourly_df_with_fractal['hourly_bottom_fractal'] == True]
            
            if not bottom_fractal_signals.empty:
                for _, signal_row in bottom_fractal_signals.iterrows():
                    signal_time = signal_row['date'].strftime('%H:%M')
                    price = signal_row['close']
                    hourly_buy_times.append((signal_time, price))
                    
                    logging.info(f"   ⏰ {signal_time} - 小时线底分型信号，价格: {price:.3f}")
            
            if not hourly_buy_times:
                logging.info("   🚫 当天无小时线买入信号")
        
        # 检测分钟线信号
        logging.info("\n分钟线级别信号检测:")
        minute_on_date = minute_data[minute_data['date'].dt.date == pd.to_datetime(test_date).date()]
        
        if minute_on_date.empty:
            logging.info("   无分钟线数据")
        else:
            logging.info(f"   当天分钟线数据条数: {len(minute_on_date)}")
            
            # 按30分钟周期检测信号（每30分钟聚合一次）
            minute_on_date['30min_interval'] = minute_on_date['date'].dt.floor('30min')
            
            # 聚合为30分钟K线
            thirty_min_data = minute_on_date.groupby('30min_interval').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            logging.info(f"   30分钟K线数据条数: {len(thirty_min_data)}")
            
            # 遍历每30分钟检测信号
            thirty_min_buy_times = []
            for i in range(1, len(thirty_min_data)):
                thirty_min_df_subset = thirty_min_data.iloc[:i+1]
                
                # 简单的底分型检测逻辑（中间低，两边高）
                if i >= 2:
                    current = thirty_min_df_subset.iloc[-1]
                    prev = thirty_min_df_subset.iloc[-2]
                    prev_prev = thirty_min_df_subset.iloc[-3]
                    
                    if prev['low'] < current['low'] and prev['low'] < prev_prev['low']:
                        signal_time = prev['30min_interval'].strftime('%H:%M')
                        price = prev['close']
                        thirty_min_buy_times.append((signal_time, price))
                        
                        logging.info(f"   ⏰ {signal_time} - 30分钟底分型信号，价格: {price:.3f}")
            
            if not thirty_min_buy_times:
                logging.info("   🚫 当天无30分钟买入信号")
        
    logging.info("\n" + "="*60)
    logging.info("测试完成！")

if __name__ == "__main__":
    test_nov24_25_hourly_minute_signals()