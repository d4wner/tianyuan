import json
import datetime
import pandas as pd

# 读取交易信号数据
with open('outputs/exports/sh512660_signals_20251124_120616.json', 'r') as f:
    signals = json.load(f)

# 转换时间戳为日期
for signal in signals:
    signal['date_str'] = datetime.datetime.fromtimestamp(signal['date']/1000).strftime('%Y-%m-%d')

# 分析交易对（买入后跟随的卖出信号）
trade_pairs = []
current_buy = None

for signal in signals:
    if signal['type'] == 'buy':
        current_buy = signal
    elif signal['type'] == 'sell' and current_buy:
        # 计算收益
        profit_percent = ((signal['price'] - current_buy['price']) / current_buy['price']) * 100
        trade_pairs.append({
            'buy_date': current_buy['date_str'],
            'buy_price': current_buy['price'],
            'sell_date': signal['date_str'],
            'sell_price': signal['price'],
            'profit_percent': profit_percent,
            'buy_strength': current_buy['strength'],
            'sell_strength': signal['strength']
        })
        current_buy = None  # 重置以寻找下一个买入信号

# 显示交易对和盈利情况
print("交易信号分析结果 (2024-2025年)")
print("=" * 80)
print(f"总交易次数: {len(trade_pairs)}")
print()
print("交易详情:")
print("-" * 80)
print(f"{'买入日期':<15} {'卖出日期':<15} {'买入价':<10} {'卖出价':<10} {'收益率(%)':<15} {'买入强度':<10} {'卖出强度':<10}")
print("-" * 80)

winning_trades = 0
losing_trades = 0
total_profit = 0

for i, trade in enumerate(trade_pairs, 1):
    if trade['profit_percent'] > 0:
        winning_trades += 1
        profit_mark = "✓"
    else:
        losing_trades += 1
        profit_mark = "✗"
    
    total_profit += trade['profit_percent']
    
    print(f"{trade['buy_date']:<15} {trade['sell_date']:<15} {trade['buy_price']:<10.3f} {trade['sell_price']:<10.3f} {profit_mark} {trade['profit_percent']:>10.2f}% {trade['buy_strength']:<10.3f} {trade['sell_strength']:<10.3f}")

# 计算总收益和胜率
win_rate = (winning_trades / len(trade_pairs)) * 100 if trade_pairs else 0

print("-" * 80)
print(f"盈利交易: {winning_trades}")
print(f"亏损交易: {losing_trades}")
print(f"胜率: {win_rate:.2f}%")
print(f"总收益率: {total_profit:.2f}%")
print()

# 显示最近几次交易信号
print("最近的交易信号:")
print("-" * 80)
recent_signals = sorted(signals, key=lambda x: x['date'], reverse=True)[:10]
print(f"{'日期':<15} {'类型':<10} {'价格':<10} {'强度':<10} {'原因':<30}")
print("-" * 80)

for signal in recent_signals:
    type_str = "买入" if signal['type'] == 'buy' else "卖出"
    print(f"{signal['date_str']:<15} {type_str:<10} {signal['price']:<10.3f} {signal['strength']:<10.3f} {signal['reason']:<30}")

# 查找最大盈利的交易
if trade_pairs:
    max_profit_trade = max(trade_pairs, key=lambda x: x['profit_percent'])
    print()
    print("最大盈利交易:")
    print(f"  买入日期: {max_profit_trade['buy_date']}")
    print(f"  卖出日期: {max_profit_trade['sell_date']}")
    print(f"  买入价: {max_profit_trade['buy_price']}")
    print(f"  卖出价: {max_profit_trade['sell_price']}")
    print(f"  收益率: {max_profit_trade['profit_percent']:.2f}%")