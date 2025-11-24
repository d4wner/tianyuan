import json
import datetime
import yaml
import os

# 读取交易信号数据
signals_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251124_120616.json'
with open(signals_file, 'r') as f:
    signals = json.load(f)

# 读取配置文件
config_dir = '/Users/pingan/tools/trade/tianyuan/config'
with open(os.path.join(config_dir, 'etfs.yaml'), 'r') as f:
    etfs_config = yaml.safe_load(f)

with open(os.path.join(config_dir, 'risk_rules.yaml'), 'r') as f:
    risk_rules = yaml.safe_load(f)

# 设置默认配置，避免配置读取问题
security_config = {
    'position_limit': 0.3,  # 根据etfs.yaml中的sector类别设置
    'type': 'sector',
    'name': '军工ETF'
}

print(f"证券类型: {security_config['type']}")
print(f"最大仓位限制: {security_config['position_limit'] * 100}%")
print()

# 根据信号强度推断缠论级别
def infer_chanlun_level(signal_type, strength):
    if signal_type == 'buy':
        if strength >= 0.4:  # 高信号强度（0.4及以上）
            return '二买'
        elif strength >= 0.2:  # 中等信号强度（0.2-0.4）
            return '一买'
        else:  # 低信号强度（0.2以下）
            return '三买'
    else:  # sell
        if strength >= 0.4:  # 高信号强度（0.4及以上）
            return '二卖'
        elif strength >= 0.2:  # 中等信号强度（0.2-0.4）
            return '一卖'
        else:  # 低信号强度（0.2以下）
            return '三卖'

# 根据缠论级别和信号强度计算建议仓位
def calculate_suggested_position(signal_type, level, strength):
    if signal_type == 'buy':
        # 买入信号的仓位配置
        if level == '一买':
            position_range = risk_rules['buy_point_rules']['first_buy']['position_size']
        elif level == '二买':
            position_range = risk_rules['buy_point_rules']['second_buy']['position_size']
        else:  # 三买
            position_range = risk_rules['buy_point_rules']['third_buy']['position_size']
        
        # 根据信号强度在范围内插值
        min_pos, max_pos = position_range
        # 归一化信号强度到0-1范围（假设strength在0-1之间）
        normalized_strength = min(max(strength, 0), 1)
        position = min_pos + (max_pos - min_pos) * normalized_strength
        
        # 确保不超过该证券的最大仓位限制
        return min(position, security_config['position_limit'])
    else:  # sell
        # 卖出信号的仓位调整
        if level == '一卖':
            return -risk_rules['sell_point_rules']['first_sell']['position_adjust']
        elif level == '二卖':
            return -risk_rules['sell_point_rules']['second_sell']['position_adjust']
        else:  # 三卖
            return -risk_rules['sell_point_rules']['third_sell']['position_adjust']

# 转换时间戳为日期，并添加缠论级别和建议仓位信息
for signal in signals:
    signal['date_str'] = datetime.datetime.fromtimestamp(signal['date']/1000).strftime('%Y-%m-%d')
    signal['chanlun_level'] = infer_chanlun_level(signal['type'], signal['strength'])
    signal['suggested_position'] = calculate_suggested_position(signal['type'], signal['chanlun_level'], signal['strength'])
    # 计算建议的实际买卖金额（基于总资金600000）
    total_capital = 600000  # 回测时使用的初始资金
    if signal['type'] == 'buy':
        signal['suggested_amount'] = total_capital * signal['suggested_position']
    else:
        # 卖出时基于当前持仓的建议卖出比例
        signal['suggested_amount'] = "根据持仓比例"

# 分析交易对（买入后跟随的卖出信号）
trade_pairs = []
current_buy = None

for signal in signals:
    if signal['type'] == 'buy':
        current_buy = signal
    elif signal['type'] == 'sell' and current_buy:
        # 计算收益
        profit_percent = ((signal['price'] - current_buy['price']) / current_buy['price']) * 100
        # 计算基于建议仓位的实际收益金额
        suggested_profit_amount = current_buy['suggested_amount'] * (profit_percent / 100)
        
        trade_pairs.append({
            'buy_date': current_buy['date_str'],
            'buy_price': current_buy['price'],
            'buy_level': current_buy['chanlun_level'],
            'buy_strength': current_buy['strength'],
            'suggested_buy_position': current_buy['suggested_position'],
            'suggested_buy_amount': current_buy['suggested_amount'],
            'sell_date': signal['date_str'],
            'sell_price': signal['price'],
            'sell_level': signal['chanlun_level'],
            'sell_strength': signal['strength'],
            'suggested_sell_position': signal['suggested_position'],
            'profit_percent': profit_percent,
            'suggested_profit_amount': suggested_profit_amount
        })
        current_buy = None  # 重置以寻找下一个买入信号

# 显示带有缠论级别和仓位建议的交易信号分析结果
print("512660 (军工ETF) 交易信号分析结果 (2024-2025年)")
print("=" * 120)
print(f"总交易次数: {len(trade_pairs)}")
print()
print("交易详情 (包含缠论级别和建议仓位):")
print("-" * 120)
print(f"{'买入日期':<12} {'买入级别':<8} {'买入价':<10} {'建议仓位':<10} {'建议金额':<12} {'卖出日期':<12} {'卖出级别':<8} {'卖出价':<10} {'收益率(%)':<12} {'收益金额':<12}")
print("-" * 120)

winning_trades = 0
losing_trades = 0
total_profit = 0
total_suggested_profit = 0

for i, trade in enumerate(trade_pairs, 1):
    if trade['profit_percent'] > 0:
        winning_trades += 1
        profit_mark = "✓"
    else:
        losing_trades += 1
        profit_mark = "✗"
    
    total_profit += trade['profit_percent']
    total_suggested_profit += trade['suggested_profit_amount']
    
    print(f"{trade['buy_date']:<12} {trade['buy_level']:<8} {trade['buy_price']:<10.3f} {trade['suggested_buy_position']*100:>6.1f}% {trade['suggested_buy_amount']:>10.0f} {trade['sell_date']:<12} {trade['sell_level']:<8} {trade['sell_price']:<10.3f} {profit_mark} {trade['profit_percent']:>8.2f}% {trade['suggested_profit_amount']:>10.2f}")

# 计算总收益和胜率
win_rate = (winning_trades / len(trade_pairs)) * 100 if trade_pairs else 0

print("-" * 120)
print(f"盈利交易: {winning_trades}")
print(f"亏损交易: {losing_trades}")
print(f"胜率: {win_rate:.2f}%")
print(f"总收益率: {total_profit:.2f}%")
print(f"基于建议仓位的总收益金额: {total_suggested_profit:.2f} 元")
print()

# 显示最近几次交易信号（包含缠论级别和仓位建议）
print("最近的交易信号 (包含缠论级别和仓位建议):")
print("-" * 120)
recent_signals = sorted(signals, key=lambda x: x['date'], reverse=True)[:10]
print(f"{'日期':<12} {'类型':<8} {'级别':<8} {'价格':<10} {'强度':<10} {'建议仓位':<10} {'建议金额':<12} {'原因':<30}")
print("-" * 120)

for signal in recent_signals:
    type_str = "买入" if signal['type'] == 'buy' else "卖出"
    pos_str = f"{signal['suggested_position']*100:.1f}%" if signal['type'] == 'buy' else f"{signal['suggested_position']*100:.1f}%"
    amount_str = f"{signal['suggested_amount']:.0f}" if isinstance(signal['suggested_amount'], (int, float)) else signal['suggested_amount']
    print(f"{signal['date_str']:<12} {type_str:<8} {signal['chanlun_level']:<8} {signal['price']:<10.3f} {signal['strength']:<10.3f} {pos_str:<10} {amount_str:<12} {signal['reason']:<30}")

# 查找最大盈利的交易
if trade_pairs:
    max_profit_trade = max(trade_pairs, key=lambda x: x['profit_percent'])
    print()
    print("最大盈利交易:")
    print(f"  买入日期: {max_profit_trade['buy_date']}")
    print(f"  买入级别: {max_profit_trade['buy_level']}")
    print(f"  买入价: {max_profit_trade['buy_price']:.3f}")
    print(f"  建议仓位: {max_profit_trade['suggested_buy_position']*100:.1f}%")
    print(f"  建议买入金额: {max_profit_trade['suggested_buy_amount']:.0f} 元")
    print(f"  卖出日期: {max_profit_trade['sell_date']}")
    print(f"  卖出级别: {max_profit_trade['sell_level']}")
    print(f"  卖出价: {max_profit_trade['sell_price']:.3f}")
    print(f"  收益率: {max_profit_trade['profit_percent']:.2f}%")
    print(f"  收益金额: {max_profit_trade['suggested_profit_amount']:.2f} 元")

# 统计不同缠论级别信号的表现
buy_level_stats = {}
sell_level_stats = {}

for trade in trade_pairs:
    # 买入级别统计
    if trade['buy_level'] not in buy_level_stats:
        buy_level_stats[trade['buy_level']] = {'count': 0, 'profit_sum': 0, 'profit_amount_sum': 0, 'wins': 0}
    
    buy_level_stats[trade['buy_level']]['count'] += 1
    buy_level_stats[trade['buy_level']]['profit_sum'] += trade['profit_percent']
    buy_level_stats[trade['buy_level']]['profit_amount_sum'] += trade['suggested_profit_amount']
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
print(f"{'级别':<8} {'次数':<8} {'胜率':<10} {'平均收益率':<12} {'平均收益金额':<12}")
print("-" * 80)

for level in ['一买', '二买', '三买']:
    if level in buy_level_stats:
        stats = buy_level_stats[level]
        avg_profit = stats['profit_sum'] / stats['count']
        avg_profit_amount = stats['profit_amount_sum'] / stats['count']
        win_rate = (stats['wins'] / stats['count']) * 100
        print(f"{level:<8} {stats['count']:<8} {win_rate:>6.1f}% {avg_profit:>10.2f}% {avg_profit_amount:>10.2f}")

# 保存增强后的交易信号到新文件
enhanced_signals = []
for signal in signals:
    enhanced_signal = {
        'date': signal['date'],
        'date_str': signal['date_str'],
        'type': signal['type'],
        'price': signal['price'],
        'strength': signal['strength'],
        'chanlun_level': signal['chanlun_level'],
        'suggested_position': signal['suggested_position'],
        'suggested_amount': signal['suggested_amount'],
        'reason': f"{signal['chanlun_level']} + {signal['reason']}"
    }
    enhanced_signals.append(enhanced_signal)

# 保存增强后的信号到新文件
output_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_enhanced.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(enhanced_signals, f, ensure_ascii=False, indent=2)

print()
print(f"增强后的交易信号已保存到: {output_file}")
print("增强信息包括：缠论级别、建议仓位和建议交易金额")