import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from daily_buy_signal_detector import BuySignalDetector
from data_fetcher import StockDataFetcher
import pandas as pd
import logging
import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/test_2025_signals.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 测试函数
def test_all_2025_signals():
    """测试2025年所有交易信号的检测"""
    try:
        # 初始化数据获取器和信号检测器
        data_fetcher = StockDataFetcher()
        detector = BuySignalDetector()
        
        # 获取2025年的所有历史数据
        logging.info("开始获取2025年军工ETF(512660)的历史数据...")
        df = data_fetcher.get_daily_data('512660', start_date='2025-01-01', end_date='2025-12-31')
        
        if df is None or len(df) == 0:
            logging.error("未获取到数据")
            return False
        
        logging.info(f"成功获取数据，共{len(df)}条记录")
        logging.info(f"数据日期范围：{df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
        
        # 遍历2025年的每一天，检查是否能正确检测信号
        success_count = 0
        error_count = 0
        processed_dates = []
        
        logging.info("开始测试2025年每日的信号检测...")
        
        # 获取所有日期
        all_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        
        for test_date in all_dates:
            test_date_str = test_date.strftime('%Y-%m-%d')
            
            try:
                # 确保有足够的历史数据
                df_subset = df[df['date'] <= test_date_str]
                if len(df_subset) < 60:  # 至少需要60根K线
                    continue
                
                # 检测所有类型的信号
                logging.info(f"\n测试日期：{test_date_str}")
                
                # 检测一买信号
                first_buy_signal, first_buy_info = detector.detect_daily_first_buy(df_subset)
                
                # 检测二买信号
                second_buy_signal, second_buy_info = detector.detect_daily_second_buy(df_subset, first_buy_signal)
                
                # 检测三买信号
                third_buy_signal, third_buy_info = detector.detect_daily_third_buy(df_subset)
                
                # 检测破中枢反抽信号
                reverse_pullback_signal, reverse_pullback_info = detector.detect_daily_reverse_pullback(df_subset)
                
                # 汇总信号
                has_signal = any([reverse_pullback_signal, first_buy_signal, second_buy_signal, third_buy_signal])
                
                if has_signal:
                    logging.info(f"✅ 检测到信号：破中枢反抽={reverse_pullback_signal}, 一买={first_buy_signal}, 二买={second_buy_signal}, 三买={third_buy_signal}")
                else:
                    logging.info(f"❌ 未检测到信号")
                
                success_count += 1
                processed_dates.append(test_date_str)
                
            except Exception as e:
                logging.error(f"❌ 测试日期 {test_date_str} 时发生错误：{str(e)}")
                logging.exception(e)
                error_count += 1
                
                # 如果错误连续超过5次，可能存在系统性问题，需要停止
                if error_count >= 5:
                    logging.error("连续发生5次错误，测试终止")
                    return False
        
        logging.info("\n" + "="*60)
        logging.info("测试完成！")
        logging.info(f"总测试日期数：{len(all_dates)}")
        logging.info(f"有效测试日期数：{len(processed_dates)}")
        logging.info(f"成功测试数：{success_count}")
        logging.info(f"错误测试数：{error_count}")
        logging.info(f"成功率：{success_count / len(processed_dates) * 100:.2f}%")
        
        return error_count == 0
        
    except Exception as e:
        logging.error(f"测试过程中发生严重错误：{str(e)}")
        logging.exception(e)
        return False

if __name__ == "__main__":
    print("开始全面测试2025年军工ETF(512660)的交易信号检测...")
    print("详细日志将保存到 logs/test_2025_signals.log")
    
    success = test_all_2025_signals()
    
    if success:
        print("\n✅ 所有测试通过！2025年的信号检测功能正常")
    else:
        print("\n❌ 测试失败！请查看日志获取详细信息")
        sys.exit(1)