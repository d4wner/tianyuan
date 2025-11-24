import json
import datetime
import yaml
import pandas as pd

# 读取交易信号数据
with open('outputs/exports/sh512660_signals_20251124_120616.json', 'r') as f:
    signals = json.load(f)

# 读取配置文件（使用绝对路径）
import os

config_dir = '/Users/pingan/tools/trade/tianyuan/config'

with open(os.path.join(config_dir, 'etfs.yaml'), 'r') as f:
    etfs_config = yaml.safe_load(f)

with open(os.path.join(config_dir, 'risk_rules.yaml'), 'r') as f:
    risk_rules = yaml.safe_load(f)

with open(os.path.join(config_dir, 'system.yaml'), 'r') as f:
    system_config = yaml.safe_load(f)

# 调试输出：查看etfs_config的结构
print("调试信息：etfs_config的键")
print(list(etfs_config.keys()))
print()

# 获取512660的配置信息
security_code = '512660'
security_config = None

# 遍历所有类别查找512660
for category in etfs_config:
    if category == 'global':
        continue
    print(f"检查类别: {category}, 类型: {type(etfs_config[category])}")
    if isinstance(etfs_config[category], dict):
        print(f"  该类别下的ETF列表: {list(etfs_config[category].keys())}")
        if security_code in etfs_config[category]:
            security_config = etfs_config[category][security_code]
            print(f"  找到512660的配置!")
            break

if not security_config:
    print(f"未找到 {security_code} 的配置信息")
    print("正在使用默认配置...")
    # 使用默认配置，避免脚本退出
    security_config = {
        'position_limit': 0.3,
        'type': 'sector',
        'name': '军工ETF'
    }

# 获取该证券类型的最大仓位限制
position_limit = security_config['position_limit']  # 0.3
print(f"证券类型: {security_config['type']}")
print(f"最大仓位限制: {position_limit * 100}%")
print()

# 根据信号强度推断缠论级别
def infer_chanlun_level(signal_type, strength):
    if signal_type == 'buy':
        if strength >= 0.4:  # 高信号强度
            return '二买'
        elif strength >= 0.2:  # 中等信号强度
            return '一买'
        else:  # 低信号强度
            return '三买'
    else:  # sell
        if strength >= 0.4:  # 高信号强度
            return '二卖'
        elif strength >= 0.2:  # 中等信号强度
            return '一卖'
        else:  # 低信号强度
            return '三卖'

# 根据缠论级别和信号强度计算建议仓位
def calculate_suggested_position(signal_type, level, strength):
    # 获取对应级别的仓位范围
    if signal_type == 'buy':
        if level == '一买':
            position_range = risk_rules['buy_point_rules']['first_buy']['position_size']  # [0.1, 0.15]
        elif level == '二买':
            position_range = risk_rules['buy_point_rules']['second_buy']['position_size']  # [0.4, 0.5]
        else:  # 三买
            position_range = risk_rules['buy_point_rules']['third_buy']['position_size']  # [0.2, 0.25]
    else:  # sell
        if level == '一卖':
            return -risk_rules['sell_point_rules']['first_sell']['position_adjust']  # 0.5
        elif level == '二卖':
            return -risk_rules['sell_point_rules']['second_sell']['position_adjust']  # 0.3
        else:  # 三卖
            return -risk_rules['sell_point_rules']['third_sell']['position_adjust']  # 0.2
    
    # 根据信号强度在范围内插值
    min_pos, max_pos = position_range
    position = min_pos + (max_pos - min_pos) * ((strength - 0.1) / 0.9)  # 假设strength在0.1-1.0之间
    
    # 确保不超过该证券的最大仓位限制
    return min(position, position_limit)

# 转换时间戳为日期，并添加缠论级别和建议仓位信息
for signal in signals:
    signal['date_str'] = datetime.datetime.fromtimestamp(signal['date']/1000).strftime('%Y-%m-%d')
    signal['chanlun_level'] = infer_chanlun_level(signal['type'], signal['strength'])
    signal['suggested_position'] = calculate_suggested_position(signal['type'], signal['chanlun_level'], signal['strength'])

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
            'buy_level': current_buy['chanlun_level'],
            'buy_strength': current_buy['strength'],
            'suggested_buy_position': current_buy['suggested_position'],
            'sell_date': signal['date_str'],
            'sell_price': signal['price'],
            'sell_level': signal['chanlun_level'],
            'sell_strength': signal['strength'],
            'suggested_sell_position': signal['suggested_position'],
            'profit_percent': profit_percent
        })
        current_buy = None  # 重置以寻找下一个买入信号

# 显示带有缠论级别和仓位建议的交易信号分析结果
print("512660 (军工ETF) 交易信号分析结果 (2024-2025年)")
print("=" * 100)
print(f"总交易次数: {len(trade_pairs)}")
print()
print("交易详情 (包含缠论级别和建议仓位):")
print("-" * 100)
print(f"{'买入日期':<12} {'买入级别':<8} {'买入价':<10} {'建议仓位':<10} {'卖出日期':<12} {'卖出级别':<8} {'卖出价':<10} {'收益率(%)':<12} {'信号强度'}")
print("-" * 100)

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
    
    print(f"{trade['buy_date']:<12} {trade['buy_level']:<8} {trade['buy_price']:<10.3f} {trade['suggested_buy_position']*100:>6.1f}% {trade['sell_date']:<12} {trade['sell_level']:<8} {trade['sell_price']:<10.3f} {profit_mark} {trade['profit_percent']:>8.2f}% 买入:{trade['buy_strength']:.3f}/卖出:{trade['sell_strength']:.3f}")

# 计算总收益和胜率
win_rate = (winning_trades / len(trade_pairs)) * 100 if trade_pairs else 0

print("-" * 100)
print(f"盈利交易: {winning_trades}")
print(f"亏损交易: {losing_trades}")
print(f"胜率: {win_rate:.2f}%")
print(f"总收益率: {total_profit:.2f}%")
print()

# 显示最近几次交易信号（包含缠论级别和仓位建议）
print("最近的交易信号 (包含缠论级别和仓位建议):")
print("-" * 100)
recent_signals = sorted(signals, key=lambda x: x['date'], reverse=True)[:10]
print(f"{'日期':<12} {'类型':<8} {'级别':<8} {'价格':<10} {'强度':<10} {'建议仓位':<10} {'原因':<30}")
print("-" * 100)

for signal in recent_signals:
    type_str = "买入" if signal['type'] == 'buy' else "卖出"
    pos_str = f"{signal['suggested_position']*100:.1f}%" if signal['type'] == 'buy' else f"-{signal['suggested_position']*100:.1f}%"
    print(f"{signal['date_str']:<12} {type_str:<8} {signal['chanlun_level']:<8} {signal['price']:<10.3f} {signal['strength']:<10.3f} {pos_str:<10} {signal['reason']:<30}")

# 查找最大盈利的交易
if trade_pairs:
    max_profit_trade = max(trade_pairs, key=lambda x: x['profit_percent'])
    print()
    print("最大盈利交易:")
    print(f"  买入日期: {max_profit_trade['buy_date']}")
    print(f"  买入级别: {max_profit_trade['buy_level']}")
    print(f"  买入价: {max_profit_trade['buy_price']}")
    print(f"  建议仓位: {max_profit_trade['suggested_buy_position']*100:.1f}%")
    print(f"  卖出日期: {max_profit_trade['sell_date']}")
    print(f"  卖出级别: {max_profit_trade['sell_level']}")
    print(f"  卖出价: {max_profit_trade['sell_price']}")
    print(f"  收益率: {max_profit_trade['profit_percent']:.2f}%")

# 统计不同缠论级别信号的表现
buy_level_stats = {}
sell_level_stats = {}

for trade in trade_pairs:
    # 买入级别统计
    if trade['buy_level'] not in buy_level_stats:
        buy_level_stats[trade['buy_level']] = {'count': 0, 'profit_sum': 0, 'wins': 0}
    
    buy_level_stats[trade['buy_level']]['count'] += 1
    buy_level_stats[trade['buy_level']]['profit_sum'] += trade['profit_percent']
    if trade['profit_percent'] > 0:
        buy_level_stats[trade['buy_level']]['wins'] += 1
    
    # 卖出级别统计
    if trade['sell_level'] not in sell_level_stats:
        sell_level_stats[trade['sell_level']] = {'count': 0, 'profit_sum': 0, 'wins': 0}
    
    sell_level_stats[trade['sell_level']]['count'] += 1
    sell_level_stats[trade['sell_level']]['profit_sum'] += trade['profit_percent']
    if trade['profit_percent'] > 0:
        sell_level_stats[trade['sell_level']]['wins'] += 1

print()
print("不同缠论级别买入信号的表现统计:")
print("-" * 80)
print(f"{'级别':<8} {'次数':<8} {'胜率':<10} {'平均收益率':<12}")
print("-" * 80)

for level in ['一买', '二买', '三买']:
    if level in buy_level_stats:
        stats = buy_level_stats[level]
        avg_profit = stats['profit_sum'] / stats['count']
        win_rate = (stats['wins'] / stats['count']) * 100
        print(f"{level:<8} {stats['count']:<8} {win_rate:>6.1f}% {avg_profit:>10.2f}%")