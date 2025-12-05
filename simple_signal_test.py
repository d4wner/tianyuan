import sys
import os
import logging
import pandas as pd
import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from daily_buy_signal_detector import BuySignalDetector
from data_fetcher import StockDataFetcher

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """简化的信号测试"""
    logging.info("开始测试军工ETF(512660)2025年的破中枢反抽信号")
    
    # 初始化检测器和数据获取器
    detector = BuySignalDetector()
    data_fetcher = StockDataFetcher()
    
    # 获取2025年的日线数据
    df = data_fetcher.get_daily_data('512660', start_date='2025-01-01', end_date='2025-12-31')
    
    if df is None or len(df) == 0:
        logging.error("未获取到数据")
        return
    
    logging.info(f"成功获取数据，共{len(df)}条记录")
    logging.info(f"数据日期范围：{df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    
    # 获取所有日期，按时间排序
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    dates.sort()
    
    # 测试每个日期
    for i, test_date_str in enumerate(dates):
        # 确保有足够的历史数据
        if i < 60:  # 至少需要60根K线
            continue
        
        try:
            # 获取到测试日期的数据
            df_subset = df[df['date'] <= pd.to_datetime(test_date_str)]
            
            logging.info(f"\n测试日期：{test_date_str}")
            
            # 检测破中枢反抽信号
            signal, info = detector.detect_daily_reverse_pullback(df_subset)
            
            if signal:
                logging.info("✅ 检测到破中枢反抽信号！")
                logging.info(f"   信号有效性: {info.get('signal_validity', '未知')}")
                logging.info(f"   当前价格: {info.get('current_price', '未知')}")
                logging.info(f"   波动等级: {info.get('volatility_level', '未设置')}")
                logging.info(f"   数据源: {info.get('data_source', '未知')}")
                # 输出动态参数
                dynamic_params = info.get('dynamic_params', {})
                if dynamic_params:
                    logging.info(f"   动态参数: {dynamic_params}")
            else:
                logging.info("❌ 未检测到破中枢反抽信号")
                
        except Exception as e:
            logging.error(f"❌ 测试日期 {test_date_str} 时发生错误：{str(e)}")
            logging.exception(e)
    
    logging.info("\n测试完成！")

if __name__ == "__main__":
    main()