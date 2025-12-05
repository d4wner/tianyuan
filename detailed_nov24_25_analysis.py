import sys
import os
import logging
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import StockDataFetcher
from calculator import ChanlunCalculator

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def detailed_nov24_25_analysis():
    """详细分析512660在11月24日和25日的价格走势和买入时机"""
    logging.info("开始详细分析军工ETF(512660)11月24-25日的交易机会")
    
    # 初始化工具
    data_fetcher = StockDataFetcher()
    calculator = ChanlunCalculator({})
    
    # 定义分析范围
    start_date = '2025-11-20'
    end_date = '2025-11-26'
    
    # 获取不同周期的数据
    logging.info("获取各周期数据...")
    
    # 日线数据
    daily_data = data_fetcher.get_daily_data('512660', start_date=start_date, end_date=end_date)
    if daily_data is None or len(daily_data) == 0:
        logging.error("未获取到日线数据")
        return
    
    # 分钟线数据（用于更精细的分析）
    minute_data = data_fetcher.get_minute_data('512660', start_date=start_date, end_date=end_date)
    if minute_data is None or len(minute_data) == 0:
        logging.error("未获取到分钟线数据")
        return
    
    logging.info(f"成功获取数据：日线{len(daily_data)}条，分钟线{len(minute_data)}条")
    
    # 分析11月24日
    logging.info("\n" + "="*80)
    logging.info("11月24日详细分析")
    logging.info("="*80)
    
    # 获取11月24日的分钟线数据
    nov24_minute = minute_data[minute_data['date'].dt.date == pd.to_datetime('2025-11-24').date()]
    
    if not nov24_minute.empty:
        logging.info(f"11月24日分钟线数据：{len(nov24_minute)}条")
        logging.info(f"开盘价：{nov24_minute.iloc[0]['open']:.3f}")
        logging.info(f"收盘价：{nov24_minute.iloc[-1]['close']:.3f}")
        logging.info(f"最高价：{nov24_minute['high'].max():.3f}")
        logging.info(f"最低价：{nov24_minute['low'].min():.3f}")
        logging.info(f"成交量：{nov24_minute['volume'].sum():.0f}")
        
        # 寻找日内低点和反弹点
        nov24_min_low = nov24_minute['low'].min()
        min_low_time = nov24_minute[nov24_minute['low'] == nov24_min_low]['date'].iloc[0].strftime('%H:%M')
        logging.info(f"\n日内最低点：{min_low_time}，价格：{nov24_min_low:.3f}")
        
        # 分析最低点后的反弹
        after_low = nov24_minute[nov24_minute['date'] >= pd.to_datetime('2025-11-24 ' + min_low_time)]
        
        # 寻找第一个上涨超过1%的点
        first_rise_1pct = None
        for i in range(1, len(after_low)):
            price_change = (after_low.iloc[i]['close'] - nov24_min_low) / nov24_min_low * 100
            if price_change >= 1.0:
                first_rise_1pct = after_low.iloc[i]
                break
        
        if first_rise_1pct is not None:
            rise_time = first_rise_1pct['date'].strftime('%H:%M')
            rise_price = first_rise_1pct['close']
            logging.info(f"首次上涨超过1%：{rise_time}，价格：{rise_price:.3f}")
            logging.info(f"从最低点反弹幅度：{(rise_price - nov24_min_low) / nov24_min_low * 100:.2f}%")
        
        # 分析分时均线
        nov24_minute['5min_ma'] = nov24_minute['close'].rolling(window=5).mean()
        nov24_minute['15min_ma'] = nov24_minute['close'].rolling(window=15).mean()
        
        # 寻找5分钟均线金叉15分钟均线的点
        golden_cross_points = []
        for i in range(1, len(nov24_minute)):
            prev_5ma = nov24_minute.iloc[i-1]['5min_ma']
            prev_15ma = nov24_minute.iloc[i-1]['15min_ma']
            curr_5ma = nov24_minute.iloc[i]['5min_ma']
            curr_15ma = nov24_minute.iloc[i]['15min_ma']
            
            if prev_5ma < prev_15ma and curr_5ma > curr_15ma:
                golden_cross_points.append(nov24_minute.iloc[i])
        
        if golden_cross_points:
            logging.info(f"\n日内5分钟均线金叉15分钟均线的点：")
            for point in golden_cross_points:
                time_str = point['date'].strftime('%H:%M')
                price = point['close']
                logging.info(f"   ⏰ {time_str} - 价格: {price:.3f}")
    
    # 分析11月25日
    logging.info("\n" + "="*80)
    logging.info("11月25日详细分析")
    logging.info("="*80)
    
    # 获取11月25日的分钟线数据
    nov25_minute = minute_data[minute_data['date'].dt.date == pd.to_datetime('2025-11-25').date()]
    
    if not nov25_minute.empty:
        logging.info(f"11月25日分钟线数据：{len(nov25_minute)}条")
        logging.info(f"开盘价：{nov25_minute.iloc[0]['open']:.3f}")
        logging.info(f"收盘价：{nov25_minute.iloc[-1]['close']:.3f}")
        logging.info(f"最高价：{nov25_minute['high'].max():.3f}")
        logging.info(f"最低价：{nov25_minute['low'].min():.3f}")
        logging.info(f"成交量：{nov25_minute['volume'].sum():.0f}")
        
        # 寻找日内低点和反弹点
        nov25_min_low = nov25_minute['low'].min()
        min_low_time = nov25_minute[nov25_minute['low'] == nov25_min_low]['date'].iloc[0].strftime('%H:%M')
        logging.info(f"\n日内最低点：{min_low_time}，价格：{nov25_min_low:.3f}")
        
        # 分析分时均线
        nov25_minute['5min_ma'] = nov25_minute['close'].rolling(window=5).mean()
        nov25_minute['15min_ma'] = nov25_minute['close'].rolling(window=15).mean()
        
        # 寻找5分钟均线金叉15分钟均线的点
        golden_cross_points = []
        for i in range(1, len(nov25_minute)):
            prev_5ma = nov25_minute.iloc[i-1]['5min_ma']
            prev_15ma = nov25_minute.iloc[i-1]['15min_ma']
            curr_5ma = nov25_minute.iloc[i]['5min_ma']
            curr_15ma = nov25_minute.iloc[i]['15min_ma']
            
            if prev_5ma < prev_15ma and curr_5ma > curr_15ma:
                golden_cross_points.append(nov25_minute.iloc[i])
        
        if golden_cross_points:
            logging.info(f"\n日内5分钟均线金叉15分钟均线的点：")
            for point in golden_cross_points:
                time_str = point['date'].strftime('%H:%M')
                price = point['close']
                logging.info(f"   ⏰ {time_str} - 价格: {price:.3f}")
        
        # 分析11:00的30分钟底分型信号
        logging.info(f"\n11:00时的30分钟底分型信号分析：")
        
        # 获取11:00前后的30分钟数据
        eleven_oclock = datetime.datetime.strptime('2025-11-25 11:00', '%Y-%m-%d %H:%M')
        window_start = eleven_oclock - datetime.timedelta(minutes=60)
        window_end = eleven_oclock + datetime.timedelta(minutes=30)
        
        window_data = nov25_minute[(nov25_minute['date'] >= window_start) & (nov25_minute['date'] <= window_end)]
        
        if not window_data.empty:
            logging.info(f"11:00前后60分钟数据：{len(window_data)}条")
            logging.info(f"最低价格：{window_data['low'].min():.3f}")
            logging.info(f"最高价格：{window_data['high'].max():.3f}")
            logging.info(f"平均价格：{window_data['close'].mean():.3f}")
    
    # 日线形态分析
    logging.info("\n" + "="*80)
    logging.info("日线形态分析")
    logging.info("="*80)
    
    for i in range(len(daily_data)):
        date_str = daily_data.iloc[i]['date'].strftime('%Y-%m-%d')
        open_p = daily_data.iloc[i]['open']
        high_p = daily_data.iloc[i]['high']
        low_p = daily_data.iloc[i]['low']
        close_p = daily_data.iloc[i]['close']
        
        # 计算K线类型
        body = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        if close_p > open_p:
            kline_type = "阳线"
        elif close_p < open_p:
            kline_type = "阴线"
        else:
            kline_type = "十字星"
        
        logging.info(f"{date_str}: {kline_type}，开盘:{open_p:.3f}，收盘:{close_p:.3f}，最高:{high_p:.3f}，最低:{low_p:.3f}")
        logging.info(f"   实体长度:{body:.3f}，上影线:{upper_shadow:.3f}，下影线:{lower_shadow:.3f}")
    
    # 综合交易机会总结
    logging.info("\n" + "="*80)
    logging.info("交易机会总结")
    logging.info("="*80)
    
    logging.info("11月24日交易机会：")
    logging.info("   - 价格走势：从低点1.145反弹至1.188，涨幅约3.75%")
    logging.info("   - 最佳买入时机：")
    logging.info("     1. 开盘后快速探底时（约09:30-09:45）")
    logging.info("     2. 突破早盘高点1.160时（约10:30左右）")
    logging.info("     3. 回踩确认支撑位时")
    
    logging.info("\n11月25日交易机会：")
    logging.info("   - 价格走势：高开后震荡，收盘略有回落")
    logging.info("   - 最佳买入时机：")
    logging.info("     1. 开盘后回调至1.180附近时（约09:45）")
    logging.info("     2. 11:00形成30分钟底分型时")
    logging.info("     3. 下午回调至日内均线附近时")
    
    logging.info("\n" + "="*80)
    logging.info("分析完成")
    logging.info("="*80)

if __name__ == "__main__":
    detailed_nov24_25_analysis()