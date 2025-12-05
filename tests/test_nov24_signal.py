import pandas as pd
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('/Users/pingan/tools/trade/tianyuan'))

from src.daily_buy_signal_detector import BuySignalDetector
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_nov24_signal')

# 加载数据
df = pd.read_csv('/Users/pingan/tools/trade/tianyuan/data/daily/512660_daily.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("=== 512660 11月24日破中枢反抽信号测试 ===")
print(f"数据总行数: {len(df)}")
print(f"最新日期: {df.index[-1]}")
print(f"11月21日数据: {df.loc['2025-11-21']}")
print(f"11月24日数据: {df.loc['2025-11-24']}")

# 创建检测器实例
detector = BuySignalDetector()
detector.logger = logger  # 设置日志

# 测试破中枢反抽信号
try:
    print("\n=== 设置波动等级和动态参数 ===")
    detector.adapt_to_volatility(df)
    print(f"波动等级: {detector.volatility_level}")
    print(f"波动率: {detector.volatility_value:.2f}%")
    print(f"动态参数: {detector.dynamic_params}")
    
    print("\n=== 开始检测破中枢反抽信号 ===")
    is_reverse_pullback, details = detector.detect_daily_reverse_pullback(df)
    
    print(f"\n检测结果: {'✓ 存在' if is_reverse_pullback else '✗ 不存在'}")
    
    # 打印关键信息
    print("\n=== 中枢信息 ===")
    print(f"主中枢: {details['central_bank']['bottom_main']:.4f} - {details['central_bank']['top_main']:.4f}")
    print(f"主中枢振幅: {details['central_bank']['amplitude_main_pct']:.2f}%")
    print(f"备用中枢: {details['central_bank']['bottom_backup']:.4f} - {details['central_bank']['top_backup']:.4f}")
    print(f"备用中枢振幅: {details['central_bank']['amplitude_backup_pct']:.2f}%")
    print(f"要求振幅阈值: {details['central_bank']['required_amplitude_pct']:.2f}%")
    
    print("\n=== 跌破中枢情况 ===")
    print(f"跌破当前中枢: {details['breakdown']['has_below_central']}")
    print(f"跌破历史中枢: {details['breakdown']['has_below_historical']}")
    print(f"跌破任一中枢: {details['breakdown']['has_below_any_central']}")
    print(f"创最近新低: {details['breakdown']['is_recent_low_new']}")
    print(f"最近低点: {details['breakdown']['recent_low']:.4f}")
    
    print("\n=== 企稳情况 ===")
    print(f"企稳结构: {details['stability']['has_stability']}")
    print(f"企稳天数: {details['stability']['stability_count']}")
    print(f"企稳开始日期: {details['stability']['stability_start']}")
    
    print("\n=== 站回中枢情况 ===")
    print(f"站回当前中枢: {details['pullback']['back_to_central']}")
    print(f"站回历史中枢: {details['pullback']['back_to_historical']}")
    print(f"站回任一中枢: {details['pullback']['back_to_any_central']}")
    print(f"成交量条件: {details['pullback']['volume_condition']}")
    print(f"连续站回天数: {details['pullback']['consecutive_above_current']}")
    
    print("\n=== 核心条件 ===")
    print(f"核心条件满足: {details['core_conditions_met']}")
    print(f"辅助条件满足: {details['support_conditions_met']}")
    print(f"信号有效性: {details['signal_validity']}")
    
    print("\n=== 动态参数 ===")
    for param, value in details['dynamic_params'].items():
        print(f"{param}: {value}")
        
    # 检查11月24日是否满足站回中枢条件
    print("\n=== 11月24日详细检查 ===")
    nov24_close = df.loc['2025-11-24']['close']
    central_bottom = max(details['central_bank']['bottom_main'], details['central_bank']['bottom_backup'])
    print(f"11月24日收盘价: {nov24_close:.4f}")
    print(f"中枢下沿: {central_bottom:.4f}")
    print(f"收盘价是否大于中枢下沿: {nov24_close > central_bottom}")
    print(f"收盘价与中枢下沿的比例: {(nov24_close / central_bottom - 1) * 100:.2f}%")
    
    # 检查最近15天的最低价
    print("\n=== 最近15天最低价检查 ===")
    recent_15_low = df.tail(15)['low'].min()
    print(f"最近15天最低价: {recent_15_low:.4f}")
    print(f"11月21日最低价: {df.loc['2025-11-21']['low']:.4f}")
    print(f"是否为最近新低: {recent_15_low == df.loc['2025-11-21']['low']}")
    
except Exception as e:
    print(f"\n检测过程中出错: {e}")
    import traceback
    traceback.print_exc()