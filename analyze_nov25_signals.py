import json
import datetime

# 加载信号文件
def analyze_signals(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        signals = json.load(f)
    
    print(f"总共有 {len(signals)} 条信号记录")
    
    # 转换时间戳并显示
    for i, signal in enumerate(signals):
        timestamp = signal['date']
        dt = datetime.datetime.fromtimestamp(timestamp / 1000)
        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 检查是否是2025年11月24日或25日的信号
        if (dt.year == 2025 and dt.month == 11 and 
            (dt.day == 24 or dt.day == 25)):
            print(f"信号 {i+1}: {date_str}, 类型: {signal['type']}, 价格: {signal['price']}, ")
            print(f"  强度: {signal['strength']}, 原因: {signal['reason']}")
    
    # 查找11月25日14:45左右的信号
    target_date = datetime.datetime(2025, 11, 25, 14, 45, 0)
    print(f"\n查找 {target_date.strftime('%Y-%m-%d %H:%M')} 左右的信号:")
    
    for i, signal in enumerate(signals):
        timestamp = signal['date']
        dt = datetime.datetime.fromtimestamp(timestamp / 1000)
        
        # 计算与目标时间的时间差（分钟）
        time_diff = abs((dt - target_date).total_seconds() / 60)
        
        if time_diff <= 30 and dt.year == 2025 and dt.month == 11 and dt.day == 25:
            print(f"信号 {i+1}: {dt.strftime('%Y-%m-%d %H:%M:%S')}, 类型: {signal['type']}, 价格: {signal['price']}")
            print(f"  强度: {signal['strength']}, 原因: {signal['reason']}, 与目标时间差: {time_diff:.2f} 分钟")

# 分析11月25日的信号文件
print("分析11月25日的信号文件:")
analyze_signals('/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251125_084914.json')

# 分析11月24日的信号文件
print("\n分析11月24日的信号文件:")
analyze_signals('/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251124_120616.json')