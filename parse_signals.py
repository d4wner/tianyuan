import json
import datetime

# 打开并读取JSON文件
with open('outputs/analysis/512660_buy_signal_detailed_analysis_20251205_083602.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取并显示信号信息
signals = data['signal_quality']['signals']
print('2025年军工ETF(512660)交易信号详情：')
print('-' * 60)
for signal in signals:
    date = datetime.datetime.fromtimestamp(signal['date'] / 1000).strftime('%Y-%m-%d')
    price = signal['price']
    strength = signal['strength']
    reason = signal['reason']
    print(f"日期: {date} | 价格: {price} | 强度: {strength} | 原因: {reason}")

print('-' * 60)
print(f'总信号数: {len(signals)}')