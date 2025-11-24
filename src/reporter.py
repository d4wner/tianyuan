#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import pandas as pd
from datetime import datetime, timedelta
from .utils import (
    get_last_trading_day, 
    calculate_max_drawdown, 
    calculate_sharpe_ratio, 
    calculate_sortino_ratio,
    format_number,
    parse_date
)

# 直接在当前文件定义日期常量（替代导入，打破循环）
DATE_FORMAT = "%Y-%m-%d"
DATE_FORMAT_ALT = "%Y%m%d"

def parse_date(date_str):
    """支持多种日期格式解析"""
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except:
        try:
            return datetime.strptime(date_str, DATE_FORMAT_ALT)
        except:
            raise ValueError(f"不支持的日期格式：{date_str}，请使用{DATE_FORMAT}或{DATE_FORMAT_ALT}")

logger = logging.getLogger('Reporter')

def generate_pre_market_report(symbols, api, calculator, start_date=None, end_date=None):
    # 函数内容保持不变
    """生成盘前报告"""
    logger.info("生成盘前市场状态报告")
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "type": "pre_market",
        "symbols": [],
        "market_analysis": {}
    }
    
    if not end_date:
        end_date = get_last_trading_day().strftime("%Y%m%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    logger.info(f"使用日期范围: {start_date} 至 {end_date}")
    
    for symbol in symbols:
        logger.info(f"处理股票: {symbol}")
        df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.warning(f"股票 {symbol} 获取数据为空")
            continue
        
        result = calculator.calculate(df)
        market_condition = calculator.determine_market_condition(result)
        
        report['symbols'].append({
            "symbol": symbol,
            "last_close": df.iloc[-1]['close'],
            "market_condition": market_condition,
            "key_levels": {
                "support": calculator.calculate_stoploss(result),
                "resistance": calculator.calculate_target_price(result, 'buy')
            },
            "recent_trend": calculator.analyze_trend(result, period=5),
            "volume_analysis": calculator.analyze_volume(result)
        })
    
    if report['symbols']:
        condition_counts = {}
        for s in report['symbols']:
            cond = s['market_condition']
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
        
        if condition_counts.get('trending_up', 0) + condition_counts.get('breakout_up', 0) > \
           condition_counts.get('trending_down', 0) + condition_counts.get('breakout_down', 0):
            overall_trend = "up"
        elif condition_counts.get('trending_down', 0) + condition_counts.get('breakout_down', 0) > 0:
            overall_trend = "down"
        else:
            overall_trend = "sideways"
        
        volatile_count = sum(1 for s in report['symbols'] if s['market_condition'] in ['volatile', 'breakout_up', 'breakout_down'])
        risk_level = "high" if volatile_count / len(report['symbols']) > 0.5 else "medium"
        
        if overall_trend == "up":
            recommendation = "buy" if risk_level == "medium" else "cautious_buy"
        elif overall_trend == "down":
            recommendation = "sell" if risk_level == "medium" else "cautious_sell"
        else:
            recommendation = "hold"
    else:
        overall_trend = "unknown"
        risk_level = "unknown"
        recommendation = "no_data"
    
    report['market_analysis'] = {
        "overall_trend": overall_trend,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "condition_distribution": condition_counts if report['symbols'] else {}
    }
    
    logger.info(f"盘前报告生成完成: {len(report['symbols'])}只股票")
    return report

def generate_daily_report(symbols, api, calculator, start_date=None, end_date=None):
    """生成盘后日报"""
    logger.info("生成每日报告")
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "type": "daily",
        "symbols": [],
        "trades": [],
        "performance": {},
        "market_summary": {}
    }
    
    if not end_date:
        end_date = get_last_trading_day().strftime("%Y%m%d")
    if not start_date:
        start_date = end_date
    
    logger.info(f"使用日期范围: {start_date} 至 {end_date}")
    
    if symbols:
        market_index = symbols[0]
        market_df = api.get_daily_data(market_index, start_date=start_date, end_date=end_date)
        if not market_df.empty:
            report['market_summary'] = {
                "index": market_index,
                "open": market_df.iloc[0]['open'],
                "close": market_df.iloc[-1]['close'],
                "high": market_df['high'].max(),
                "low": market_df['low'].min(),
                "volume": market_df['volume'].sum(),
                "change": (market_df.iloc[-1]['close'] - market_df.iloc[0]['open']) / market_df.iloc[0]['open'] * 100
            }
    
    for symbol in symbols:
        logger.info(f"处理股票: {symbol}")
        df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.warning(f"股票 {symbol} 获取数据为空")
            continue
        
        result = calculator.calculate(df)
        daily_signals = calculator.detect_signals(result.tail(5))
        
        report['symbols'].append({
            "symbol": symbol,
            "open": df.iloc[0]['open'],
            "close": df.iloc[-1]['close'],
            "high": df['high'].max(),
            "low": df['low'].min(),
            "volume": df['volume'].sum(),
            "change": (df.iloc[-1]['close'] - df.iloc[0]['open']) / df.iloc[0]['open'] * 100,
            "market_condition": calculator.determine_market_condition(result),
            "signals": daily_signals,
            "key_levels": {
                "support": calculator.calculate_stoploss(result),
                "resistance": calculator.calculate_target_price(result, 'buy')
            }
        })
    
    report['trades'] = [
        {"symbol": s["symbol"], 
         "time": f"{datetime.now().hour}:{datetime.now().minute}:00", 
         "action": "buy" if s["change"] > 0 else "sell", 
         "price": s["close"], 
         "shares": 1000,
         "reason": "daily_signal" if s["signals"] else "trend_following"
        } for s in report['symbols'][:2]
    ]
    
    # 修复②：无交易时收益显示优化
    trade_count = len(report['trades'])
    if trade_count > 0 and report['symbols']:
        total_investment = sum(t["price"] * t["shares"] for t in report['trades'])
        current_value = sum(t["price"] * t["shares"] * (1 + s["change"]/100) 
                          for t, s in zip(report['trades'], report['symbols'][:len(report['trades'])]))
        
        total_return = ((current_value - total_investment) / total_investment) * 100
        max_drawdown = calculate_max_drawdown([t["price"] for t in report['trades']])
        winning_trades = sum(1 for t, s in zip(report['trades'], report['symbols'][:len(report['trades'])]) if s["change"] > 0)
        win_rate = (winning_trades / len(report['trades'])) * 100 if report['trades'] else 0
        
        report['performance'] = {
            "total_return": round(total_return, 2),
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(report['trades']),
            "total_investment": round(total_investment, 2),
            "current_value": round(current_value, 2)
        }
    else:
        # 无交易时收益明确设为0
        total_return = 0.0
        logger.info("无交易记录，收益为0%")
        report['performance'] = {
            "total_return": total_return,
            "max_drawdown": 0,
            "win_rate": 0,
            "total_trades": 0,
            "total_investment": 0,
            "current_value": 0
        }
    
    logger.info(f"日报生成完成: {len(report['symbols'])}只股票")
    return report

def generate_backtest_report(symbol, strategy_name, start_date, end_date, 
                            initial_capital, final_capital, trades, performance,
                            timeframe='daily', benchmark_return=0):
    """生成回测报告"""
    logger.info(f"生成回测报告: {symbol} ({start_date}至{end_date})")
    
    try:
        # 使用修复后的parse_date函数处理日期
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        duration_days = (end_dt - start_dt).days
    except ValueError as e:
        logger.warning(f"日期格式解析失败，无法计算持续时间: {e}")
        duration_days = 0
    
    # 修复②：无交易时收益显示优化
    trade_count = len(trades)
    if trade_count == 0:
        total_return = 0.0  # 无交易时收益为0%
        logger.info("无交易记录，收益为0%")
    else:
        total_return = (final_capital - initial_capital) / initial_capital * 100
    
    annualized_return = 0
    if duration_days > 0 and trade_count > 0:  # 只有有交易且有有效时长时才计算年化收益
        years = duration_days / 365.25
        annualized_return = (pow((final_capital / initial_capital), 1/years) - 1) * 100
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.profit > 0)
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = sum(t.profit for t in trades if t.profit > 0)
    total_loss = sum(abs(t.profit) for t in trades if t.profit <= 0)
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    holding_days = [t.holding_days for t in trades if t.holding_days is not None]
    avg_holding_days = sum(holding_days) / len(holding_days) if holding_days else 0
    
    consecutive_wins = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for t in trades:
        if t.profit > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "backtest",
        "symbol": symbol,
        "strategy": strategy_name,
        "timeframe": timeframe,
        "date_range": {
            "start": start_date,
            "end": end_date,
            "duration_days": duration_days,
            "duration_years": round(duration_days / 365.25, 2) if duration_days > 0 else 0
        },
        "capital": {
            "initial": round(initial_capital, 2),
            "final": round(final_capital, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized_return, 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "excess_return_pct": round(total_return - benchmark_return, 2) if benchmark_return else 0
        },
        "performance": {
            **performance,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "average_holding_days": round(avg_holding_days, 1),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "sharpe_ratio": round(calculate_sharpe_ratio(performance.get('returns', [])), 2),
            "sortino_ratio": round(calculate_sortino_ratio(performance.get('returns', [])), 2)
        },
        "trade_summary": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "average_profit_per_trade": round(total_profit / winning_trades, 2) if winning_trades > 0 else 0,
            "average_loss_per_trade": round(total_loss / losing_trades, 2) if losing_trades > 0 else 0,
            "profit_factor": round(profit_factor, 2)
        },
        "trades": [{
            "trade_id": t.trade_id,
            "date": t.date.strftime("%Y-%m-%d") if hasattr(t.date, 'strftime') else str(t.date),
            "type": t.type,
            "price": round(t.price, 2),
            "shares": t.shares,
            "profit": round(t.profit, 2),
            "holding_days": t.holding_days,
            "signal_source": t.signal_source,
            "notes": t.notes
        } for t in trades]
    }
    
    logger.info(f"回测报告生成完成: 总收益 {total_return:.2f}%")
    return report

def generate_multiple_backtest_report(reports):
    """生成多策略/多标的回测汇总报告"""
    logger.info(f"生成多回测汇总报告，共{len(reports)}个回测结果")
    
    if not reports:
        logger.warning("没有回测报告可汇总")
        return {"error": "没有回测报告可汇总", "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    # 修复②：无交易时收益显示优化
    total_initial = sum(r['capital']['initial'] for r in reports)
    total_final = sum(r['capital']['final'] for r in reports)
    
    # 计算总交易数
    total_trades_across_reports = sum(r['trade_summary']['total_trades'] for r in reports)
    
    if total_trades_across_reports == 0:
        total_return = 0.0  # 所有报告都无交易时总收益为0%
        logger.info("所有回测报告均无交易记录，总收益为0%")
    else:
        total_return = (total_final - total_initial) / total_initial * 100
    
    try:
        # 使用修复后的parse_date函数处理日期
        earliest_start = min(parse_date(r['date_range']['start']) for r in reports)
        latest_end = max(parse_date(r['date_range']['end']) for r in reports)
        total_duration_days = (latest_end - earliest_start).days
    except ValueError as e:
        logger.warning(f"日期格式解析失败，使用平均持续时间: {e}")
        total_duration_days = sum(r['date_range']['duration_days'] for r in reports) / len(reports)
    
    strategy_stats = {}
    for report in reports:
        strategy = report['strategy']
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                'count': 0,
                'total_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'total_initial': 0,
                'total_final': 0,
                'total_trades': 0
            }
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['total_return'] += report['capital']['total_return_pct']
        strategy_stats[strategy]['win_rate'] += report['performance']['win_rate_pct']
        strategy_stats[strategy]['max_drawdown'] += report['performance']['max_drawdown']
        strategy_stats[strategy]['sharpe_ratio'] += report['performance'].get('sharpe_ratio', 0)
        strategy_stats[strategy]['profit_factor'] += report['performance']['profit_factor']
        strategy_stats[strategy]['total_initial'] += report['capital']['initial']
        strategy_stats[strategy]['total_final'] += report['capital']['final']
        strategy_stats[strategy]['total_trades'] += report['trade_summary']['total_trades']
    
    for strategy in strategy_stats:
        count = strategy_stats[strategy]['count']
        strategy_stats[strategy]['avg_return'] = round(strategy_stats[strategy]['total_return'] / count, 2)
        strategy_stats[strategy]['avg_win_rate'] = round(strategy_stats[strategy]['win_rate'] / count, 2)
        strategy_stats[strategy]['avg_max_drawdown'] = round(strategy_stats[strategy]['max_drawdown'] / count, 2)
        strategy_stats[strategy]['avg_sharpe_ratio'] = round(strategy_stats[strategy]['sharpe_ratio'] / count, 2)
        strategy_stats[strategy]['avg_profit_factor'] = round(strategy_stats[strategy]['profit_factor'] / count, 2)
        
        # 修复②：策略无交易时收益处理
        if strategy_stats[strategy]['total_trades'] == 0:
            strategy_stats[strategy]['total_return_pct'] = 0.0
        else:
            strategy_stats[strategy]['total_return_pct'] = round(
                (strategy_stats[strategy]['total_final'] - strategy_stats[strategy]['total_initial']) / 
                strategy_stats[strategy]['total_initial'] * 100, 2
            )
    
    symbol_stats = {}
    for report in reports:
        symbol = report['symbol']
        if symbol not in symbol_stats:
            symbol_stats[symbol] = {
                'count': 0,
                'total_return': 0,
                'best_strategy': '',
                'best_return': -float('inf'),
                'worst_strategy': '',
                'worst_return': float('inf'),
                'total_trades': 0
            }
        symbol_stats[symbol]['count'] += 1
        symbol_stats[symbol]['total_return'] += report['capital']['total_return_pct']
        symbol_stats[symbol]['total_trades'] += report['trade_summary']['total_trades']
        
        if report['capital']['total_return_pct'] > symbol_stats[symbol]['best_return']:
            symbol_stats[symbol]['best_return'] = report['capital']['total_return_pct']
            symbol_stats[symbol]['best_strategy'] = report['strategy']
        if report['capital']['total_return_pct'] < symbol_stats[symbol]['worst_return']:
            symbol_stats[symbol]['worst_return'] = report['capital']['total_return_pct']
            symbol_stats[symbol]['worst_strategy'] = report['strategy']
    
    for symbol in symbol_stats:
        count = symbol_stats[symbol]['count']
        # 修复②：标的无交易时收益处理
        if symbol_stats[symbol]['total_trades'] == 0:
            symbol_stats[symbol]['avg_return'] = 0.0
        else:
            symbol_stats[symbol]['avg_return'] = round(symbol_stats[symbol]['total_return'] / count, 2)
    
    best_strategy = None
    best_strategy_return = -float('inf')
    worst_strategy = None
    worst_strategy_return = float('inf')
    
    for strategy, stats in strategy_stats.items():
        if stats['total_return_pct'] > best_strategy_return:
            best_strategy = strategy
            best_strategy_return = stats['total_return_pct']
        if stats['total_return_pct'] < worst_strategy_return:
            worst_strategy = strategy
            worst_strategy_return = stats['total_return_pct']
    
    timeframe_distribution = {}
    for report in reports:
        tf = report['timeframe']
        timeframe_distribution[tf] = timeframe_distribution.get(tf, 0) + 1
    
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "backtest_summary",
        "total_reports": len(reports),
        "date_range": {
            "earliest_start": earliest_start.strftime("%Y-%m-%d") if 'earliest_start' in locals() else "",
            "latest_end": latest_end.strftime("%Y-%m-%d") if 'latest_end' in locals() else "",
            "total_duration_days": total_duration_days
        },
        "overall_performance": {
            "total_initial_capital": round(total_initial, 2),
            "total_final_capital": round(total_final, 2),
            "total_return_pct": round(total_return, 2),
            "average_annualized_return_pct": round(
                sum(r['capital']['annualized_return_pct'] for r in reports) / len(reports), 2
            ),
            "average_max_drawdown_pct": round(
                sum(r['performance']['max_drawdown'] for r in reports) / len(reports), 2
            ),
            "average_sharpe_ratio": round(
                sum(r['performance'].get('sharpe_ratio', 0) for r in reports) / len(reports), 2
            )
        },
        "strategy_stats": strategy_stats,
        "symbol_stats": symbol_stats,
        "timeframe_distribution": timeframe_distribution,
        "best_strategy": {
            "name": best_strategy,
            "return_pct": best_strategy_return
        } if best_strategy else None,
        "worst_strategy": {
            "name": worst_strategy,
            "return_pct": worst_strategy_return
        } if worst_strategy else None,
        "trade_summary": {
            "total_trades": total_trades_across_reports,
            "total_winning_trades": sum(r['trade_summary']['winning_trades'] for r in reports),
            "total_losing_trades": sum(r['trade_summary']['losing_trades'] for r in reports),
            "overall_win_rate_pct": round(
                (sum(r['trade_summary']['winning_trades'] for r in reports) / 
                 sum(r['trade_summary']['total_trades'] for r in reports)) * 100, 2
                ) if sum(r['trade_summary']['total_trades'] for r in reports) > 0 else 0,
            "total_profit": round(sum(r['trade_summary']['total_profit'] for r in reports), 2),
            "total_loss": round(sum(r['trade_summary']['total_loss'] for r in reports), 2),
            "overall_profit_factor": round(
                sum(r['trade_summary']['total_profit'] for r in reports) / 
                sum(r['trade_summary']['total_loss'] for r in reports), 2
                ) if sum(r['trade_summary']['total_loss'] for r in reports) > 0 else 0
        }
    }
    
    logger.info(f"多回测汇总报告生成完成，总收益: {total_return:.2f}%")
    return summary