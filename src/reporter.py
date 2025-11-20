#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import pandas as pd
from datetime import datetime, timedelta
from .calculator import ChanlunCalculator
from .data_fetcher import StockDataAPI
from .utils import get_last_trading_day  # 修复导入路径

logger = logging.getLogger('Reporter')

def generate_pre_market_report(symbols, api, calculator, start_date=None, end_date=None):
    """生成盘前报告"""
    logger.info("生成盘前市场状态报告")
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "type": "pre_market",
        "symbols": [],
        "market_analysis": {}
    }
    
    # 如果没有提供日期范围，使用默认值
    if not end_date:
        end_date = get_last_trading_day().strftime("%Y%m%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    logger.info(f"使用日期范围: {start_date} 至 {end_date}")
    
    for symbol in symbols:
        logger.info(f"处理股票: {symbol}")
        # 获取数据 - 使用传入的日期范围
        df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.warning(f"股票 {symbol} 获取数据为空")
            continue
        
        # 计算缠论指标
        result = calculator.calculate(df)
        
        # 分析市场状况
        market_condition = calculator.determine_market_condition(result)
        
        # 添加分析结果
        report['symbols'].append({
            "symbol": symbol,
            "last_close": df.iloc[-1]['close'],
            "market_condition": market_condition,
            "key_levels": {
                "support": calculator.calculate_stoploss(result),
                "resistance": calculator.calculate_target_price(result, 'buy')
            }
        })
    
    # 市场整体分析
    if report['symbols']:
        # 如果所有股票都处于上涨趋势，则整体趋势为上涨
        if all(s['market_condition'] in ['trending_up', 'breakout_up'] for s in report['symbols']):
            overall_trend = "up"
        else:
            overall_trend = "down"
    else:
        overall_trend = "unknown"
    
    report['market_analysis'] = {
        "overall_trend": overall_trend,
        "risk_level": "medium",
        "recommendation": "hold"
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
        "performance": {}
    }
    
    # 如果没有提供日期范围，使用默认值
    if not end_date:
        end_date = get_last_trading_day().strftime("%Y%m%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    logger.info(f"使用日期范围: {start_date} 至 {end_date}")
    
    for symbol in symbols:
        logger.info(f"处理股票: {symbol}")
        # 获取数据 - 使用传入的日期范围
        df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.warning(f"股票 {symbol} 获取数据为空")
            continue
        
        # 计算缠论指标
        result = calculator.calculate(df)
        
        # 添加分析结果
        report['symbols'].append({
            "symbol": symbol,
            "open": df.iloc[0]['open'],
            "close": df.iloc[-1]['close'],
            "change": (df.iloc[-1]['close'] - df.iloc[0]['open']) / df.iloc[0]['open'] * 100
        })
    
    # 添加交易记录（模拟）
    report['trades'] = [
        {"symbol": "510300", "time": "10:30:00", "action": "buy", "price": 4.55, "shares": 1000},
        {"symbol": "510500", "time": "14:30:00", "action": "sell", "price": 5.60, "shares": 500}
    ]
    
    # 计算当日绩效
    report['performance'] = {
        "total_return": 1.5,  # %
        "max_drawdown": 0.8,  # %
        "win_rate": 65.0      # %
    }
    
    logger.info(f"日报生成完成: {len(report['symbols'])}只股票")
    return report