#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论股票分析系统主程序 - 完整修复版
修复内容：股票代码类型错误、日期格式处理、参数传递逻辑、周线数据处理等
"""

import argparse
import logging
import os
import shutil
import sys
import time
import pandas as pd
import yaml
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 修复导入错误：确保正确引用数据获取器
from src.data_fetcher import StockDataFetcher as StockDataAPI
from src.config import load_config, save_config
from src.calculator import ChanlunCalculator
from src.monitor import ChanlunMonitor
from src.backtester import BacktestEngine
from src.plotter import ChanlunPlotter
from src.exporter import ChanlunExporter
from src.reporter import generate_pre_market_report, generate_daily_report
from src.notifier import DingdingNotifier
from src.utils import (
    get_last_trading_day, is_trading_hour, get_valid_date_range_str,
    format_date, parse_date, get_date_range
)

# 配置日志
logger = logging.getLogger('ChanlunSystem')
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def setup_logger(logfile=None):
    """配置日志系统，支持文件输出"""
    if logfile:
        # 确保日志目录存在
        log_dir = os.path.dirname(logfile)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_etf_config(config_path: str = 'config/etfs.yaml') -> dict:
    """加载ETF配置文件，包含市场前缀等信息"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"成功加载ETF配置，共{len(config)}个类别")
        return config
    except FileNotFoundError:
        logger.warning(f"ETF配置文件未找到: {config_path}，使用默认配置")
        return {}
    except Exception as e:
        logger.error(f"加载ETF配置失败: {str(e)}")
        return {}

def get_market_prefix(symbol: str, etf_config: dict) -> str:
    """根据ETF配置获取市场前缀（sh/sz）"""
    try:
        # 遍历所有类别（除了global）
        for category, etfs in etf_config.items():
            if category == 'global' or not isinstance(etfs, dict):
                continue
                
            # 检查股票代码是否在该类别中
            if symbol in etfs:
                etf_info = etfs[symbol]
                if 'market' in etf_info:
                    return etf_info['market']
        
        # 如果没有找到配置，使用默认逻辑
        if symbol.startswith(('5', '6', '9', '7')):  # 补充沪市代码前缀识别
            return 'sh'
        else:
            return 'sz'
            
    except Exception as e:
        logger.warning(f"获取市场前缀失败: {str(e)}，使用默认逻辑")
        if symbol.startswith(('5', '6', '9', '7')):
            return 'sh'
        else:
            return 'sz'

def adjust_symbol_format(symbol: str, etf_config: dict) -> str:
    """
    调整股票代码格式，确保添加正确的市场前缀
    修复：添加类型检查，防止非字符串类型输入
    """
    # 核心修复：确保输入为字符串类型
    if not isinstance(symbol, str):
        raise TypeError(f"股票代码必须是字符串，实际为{type(symbol)}")
    
    # 移除可能的前缀，避免重复添加
    if symbol.startswith(('sh', 'sz')):
        return symbol
    
    market_prefix = get_market_prefix(symbol, etf_config)
    full_symbol = f"{market_prefix}{symbol}"
    logger.info(f"股票代码格式调整: {symbol} -> {full_symbol}")
    return full_symbol

def evaluate_market_status(api, calculator, symbols: List[str]) -> Dict[str, Any]:
    """
    评估市场整体状态
    :param api: 数据API实例
    :param calculator: 缠论计算器实例
    :param symbols: 股票代码列表
    :return: 市场状态评估报告
    """
    logger.info("===== 市场状态评估 =====")
    
    status_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 修复时间格式错误
        "overall_status": "unknown",
        "symbols_status": {},
        "trending_count": 0,
        "ranging_count": 0,
        "declining_count": 0,
        "breakout_count": 0,
        "recommendation": "暂无建议"
    }
    
    # 评估每只股票
    for symbol in symbols:
        try:
            # 获取日线数据（30天）
            start_date, end_date = get_valid_date_range_str(30)
            df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date, force_refresh=True)
            
            if df.empty:
                logger.warning(f"股票 {symbol} 获取数据为空")
                continue
            
            # 计算缠论指标
            result = calculator.calculate(df)
            
            # 获取市场状态
            market_condition = calculator.determine_market_condition(result)
            
            # 记录状态
            status_report["symbols_status"][symbol] = {
                "condition": market_condition,
                "price": df.iloc[-1]['close'],
                "volume": df.iloc[-1]['volume'],
                "signal_strength": calculate_signal_strength(result)
            }
            
            # 统计状态数量
            if 'trending' in market_condition:
                status_report["trending_count"] += 1
            elif 'ranging' in market_condition:
                status_report["ranging_count"] += 1
            elif 'declining' in market_condition or 'breakout_down' in market_condition:
                status_report["declining_count"] += 1
            elif 'breakout' in market_condition:
                status_report["breakout_count"] += 1
                
        except Exception as e:
            logger.error(f"评估股票 {symbol} 状态失败: {str(e)}", exc_info=True)
            status_report["symbols_status"][symbol] = {
                "condition": "error",
                "error": str(e)
            }
    
    # 确定整体市场状态
    total_symbols = len(symbols)
    if total_symbols == 0:
        status_report["recommendation"] = "未提供股票代码"
        return status_report
        
    if status_report["trending_count"] / total_symbols > 0.6:
        status_report["overall_status"] = "trending_up"
        status_report["recommendation"] = "市场处于上升趋势，建议逢低买入"
    elif status_report["ranging_count"] / total_symbols > 0.6:
        status_report["overall_status"] = "ranging"
        status_report["recommendation"] = "市场处于震荡整理，建议高抛低吸"
    elif status_report["declining_count"] / total_symbols > 0.6:
        status_report["overall_status"] = "trending_down"
        status_report["recommendation"] = "市场处于下降趋势，建议谨慎操作"
    elif status_report["breakout_count"] / total_symbols > 0.4:
        status_report["overall_status"] = "breakout"
        status_report["recommendation"] = "市场出现突破信号，建议密切关注"
    
    return status_report

def calculate_signal_strength(result_df: pd.DataFrame) -> int:
    """
    计算信号强度（0-100）
    :param result_df: 包含缠论指标的DataFrame
    :return: 信号强度值
    """
    if result_df.empty:
        return 0
    
    latest = result_df.iloc[-1]
    strength = 50  # 基准强度
    
    # 根据分型调整强度
    if latest.get('top_fractal', False):
        strength -= 10
    if latest.get('bottom_fractal', False):
        strength += 10
    
    # 根据笔调整强度
    if latest.get('pen_end', False):
        if latest.get('pen_type') == 'up':
            strength += 15
        else:
            strength -= 15
    
    # 根据线段调整强度
    if latest.get('segment_end', False):
        if latest.get('segment_type') == 'up':
            strength += 20
        else:
            strength -= 20
    
    # 根据中枢调整强度
    if latest.get('central_bank', False):
        current_price = latest['close']
        central_high = latest.get('central_bank_high', current_price)
        central_low = latest.get('central_bank_low', current_price)
        
        if current_price > central_high:
            strength += 25
        elif current_price < central_low:
            strength -= 25
        else:
            strength += 5  # 在中枢内略微偏多
    
    # 修复：添加对背驰的强度调整
    # 支持多种背驰标记格式
    divergence = latest.get('divergence', '')
    if divergence in ['bull', 'bullish', 'bottom']:
        strength += 20  # 底背驰增加信号强度
    elif divergence in ['bear', 'bearish', 'top']:
        strength -= 20  # 顶背驰减少信号强度
    
    return max(0, min(100, strength))  # 限制在0-100范围内

def validate_date_format(date_str: str, format: str = "%Y%m%d") -> bool:
    """验证日期格式是否符合要求"""
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False

def run_monitor_mode(config: Dict[str, Any], symbols: List[str], interval: int = 300):
    """运行监控模式，定时扫描市场信号"""
    logger.info("===== 监控模式 =====")
    
    # 初始化组件
    api = StockDataAPI(config.get('data_fetcher', {}))
    calculator = ChanlunCalculator(config.get('chanlun', {}))
    notifier = DingdingNotifier(config.get('dingding', {}))
    
    # 正确初始化监控器（修复：使用正确的参数顺序和配置）
    monitor = ChanlunMonitor(
        system_config=config,
        api=api,
        calculator=calculator,
        notifier=notifier
    )
    
    # 覆盖监控间隔（如果提供）
    if interval > 0:
        monitor.interval = interval
    
    # 添加所有监控股票
    for symbol in symbols:
        monitor.add_symbol(symbol)
    
    # 确保输出目录存在
    os.makedirs("outputs/signals", exist_ok=True)
    
    try:
        # 调用监控器的start方法，使用其内部的循环逻辑
        monitor.start()
            
    except KeyboardInterrupt:
        logger.info("监控模式被用户中断")
    except Exception as e:
        logger.error(f"监控模式运行出错: {str(e)}", exc_info=True)

def run_scan_once_mode(config: Dict[str, Any], symbols: List[str], plot: bool = False, export: bool = False):
    """单次扫描模式，扫描一次后退出"""
    logger.info("===== 单次扫描模式 =====")
    
    # 初始化组件
    api = StockDataAPI(config.get('data_fetcher', {}))
    calculator = ChanlunCalculator(config.get('chanlun', {}))
    plotter = ChanlunPlotter(config.get('plotter', {})) if plot else None
    exporter = ChanlunExporter(config.get('exporter', {})) if export else None
    
    # 确保输出目录存在
    if plot:
        os.makedirs("outputs/plots", exist_ok=True)
    if export:
        os.makedirs("outputs/exports", exist_ok=True)
    
    # 扫描所有股票
    scan_results = {}
    for symbol in symbols:
        try:
            logger.info(f"扫描股票: {symbol}")
            
            # 获取日线数据
            start_date, end_date = get_valid_date_range_str(180)  # 半年数据
            df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date, force_refresh=True)
            
            if df.empty:
                logger.warning(f"股票 {symbol} 日线数据为空")
                continue
            
            # 计算缠论指标
            result = calculator.calculate(df)
            # 从结果中获取信号
            signals = [item for item in result['data'] if item.get('signal') in ['buy', 'sell']]
            # 将结果数据转换为DataFrame
            result_df = pd.DataFrame(result['data'])
            
            # 记录结果
            # 转换signals中的Timestamp对象为字符串
            serializable_signals = []
            for signal in signals:
                serializable_signal = signal.copy()
                # 检查并转换date字段
                if 'date' in serializable_signal and hasattr(serializable_signal['date'], 'strftime'):
                    serializable_signal['date'] = serializable_signal['date'].strftime('%Y-%m-%d')
                # 转换其他可能的datetime字段
                for key, value in serializable_signal.items():
                    if hasattr(value, 'strftime'):
                        serializable_signal[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                serializable_signals.append(serializable_signal)
            
            scan_results[symbol] = {
                "signals": serializable_signals,
                "latest_price": df.iloc[-1]['close'],
                "signal_strength": calculate_signal_strength(result_df),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"股票 {symbol} 扫描完成，信号: {signals}")
            
            # 绘图
            if plot and plotter:
                plotter.plot(result_df, symbol)
            
            # 导出数据
            if export and exporter:
                exporter.export(result_df, symbol, "daily", config.get('output_format', 'csv'))
        
        except Exception as e:
            logger.error(f"扫描股票 {symbol} 时出错: {str(e)}", exc_info=True)
            scan_results[symbol] = {"error": str(e)}
    
    # 保存扫描结果
    scan_filename = f"outputs/scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(scan_filename, 'w', encoding='utf-8') as f:
        json.dump(scan_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"单次扫描完成，结果已保存至: {scan_filename}")
    return scan_results

def run_pre_market_mode(config: Dict[str, Any], symbols: List[str]):
    """盘前报告模式，生成盘前分析报告"""
    logger.info("===== 盘前报告模式 =====")
    
    # 初始化组件
    api = StockDataAPI(config.get('data_fetcher', {}))
    calculator = ChanlunCalculator(config.get('chanlun', {}))
    notifier = DingdingNotifier(config.get('dingding', {}))
    
    # 确保输出目录存在
    os.makedirs("outputs/reports", exist_ok=True)
    
    try:
        # 获取上一个交易日
        last_trading_day = get_last_trading_day()
        logger.info(f"基于上一交易日数据生成报告: {last_trading_day}")
        
        # 生成盘前报告
        report = generate_pre_market_report(
            symbols=symbols,
            api=api,
            calculator=calculator,
            start_date=format_date(parse_date(last_trading_day) - timedelta(days=60)),
            end_date=last_trading_day
        )
        
        # 保存报告
        report_filename = f"pre_market_report_{datetime.now().strftime('%Y%m%d')}.json"
        report_path = os.path.join("outputs/reports", report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"盘前报告已保存: {report_path}")
        
        # 发送钉钉通知
        notifier.send_report_notification("盘前报告", report)
        
        return report
    
    except Exception as e:
        logger.error(f"生成盘前报告失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

def run_daily_report_mode(config: Dict[str, Any], symbols: List[str]):
    """盘后日报模式，生成每日交易报告"""
    logger.info("===== 盘后日报模式 =====")
    
    # 初始化组件
    api = StockDataAPI(config.get('data_fetcher', {}))
    calculator = ChanlunCalculator(config.get('chanlun', {}))
    notifier = DingdingNotifier(config.get('dingding', {}))
    
    # 确保输出目录存在
    os.makedirs("outputs/reports", exist_ok=True)
    
    try:
        # 获取当前交易日
        today = datetime.now().strftime("%Y%m%d")
        if not validate_trading_date(today):
            logger.warning(f"{today} 不是交易日，使用上一交易日数据")
            today = get_last_trading_day()
        
        # 生成盘后日报
        report = generate_daily_report(
            symbols=symbols,
            api=api,
            calculator=calculator,
            start_date=format_date(parse_date(today) - timedelta(days=30)),
            end_date=today
        )
        
        # 保存报告
        report_filename = f"daily_report_{today}.json"
        report_path = os.path.join("outputs/reports", report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"盘后日报已保存: {report_path}")
        
        # 发送钉钉通知
        notifier.send_report_notification("盘后日报", report)
        
        return report
    
    except Exception as e:
        logger.error(f"生成盘后日报失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

def run_weekly_report_mode(config: Dict[str, Any], symbols: List[str]):
    """周报告模式，生成每周分析报告"""
    logger.info("===== 周报告模式 =====")
    
    # 初始化组件
    api = StockDataAPI(config.get('data_fetcher', {}))
    calculator = ChanlunCalculator(config.get('chanlun', {}))
    notifier = DingdingNotifier(config.get('dingding', {}))
    
    # 确保输出目录存在
    os.makedirs("outputs/reports/weekly", exist_ok=True)
    
    try:
        # 获取本周范围
        end_date = get_last_trading_day()
        start_date = format_date(parse_date(end_date) - timedelta(days=30))  # 近30天数据
        
        # 生成周报告
        report = generate_weekly_report(
            symbols=symbols,
            api=api,
            calculator=calculator,
            start_date=start_date,
            end_date=end_date
        )
        
        # 保存报告
        week_str = datetime.strptime(end_date, "%Y%m%d").strftime("%Y%W")
        report_filename = f"weekly_report_{week_str}.json"
        report_path = os.path.join("outputs/reports/weekly", report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"周报告已保存: {report_path}")
        
        # 发送钉钉通知
        notifier.send_report_notification("周报告", report)
        
        return report
    
    except Exception as e:
        logger.error(f"生成周报告失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

def run_backtest_mode(config: Dict[str, Any], symbols: List[str], args):
    """回测模式，执行策略回测"""
    logger.info("===== 回测模式 =====")
    
    # 验证参数
    if not symbols:
        logger.error("回测模式必须指定股票代码（--symbols）")
        return
    
    # 验证日期格式
    if not validate_date_format(args.start_date):
        logger.error(f"开始日期格式错误: {args.start_date}，必须为YYYYMMDD")
        return
    if not validate_date_format(args.end_date):
        logger.error(f"结束日期格式错误: {args.end_date}，必须为YYYYMMDD")
        return
    
    # 转换日期格式为YYYY-MM-DD（内部使用格式）
    start_date = f"{args.start_date[:4]}-{args.start_date[4:6]}-{args.start_date[6:8]}"
    end_date = f"{args.end_date[:4]}-{args.end_date[4:6]}-{args.end_date[6:8]}"  # 正确的日期格式转换
    
    # 初始化回测器
    try:
        engine = BacktestEngine(config)
        logger.info("回测引擎初始化成功")
    except Exception as e:
        logger.error(f"回测引擎初始化失败: {str(e)}", exc_info=True)
        return
    
    # 处理股票代码（确保格式正确）
    etf_config = load_etf_config()
    try:
        adjusted_symbols = [adjust_symbol_format(symbol, etf_config) for symbol in symbols]
    except TypeError as e:
        logger.error(f"股票代码处理失败: {str(e)}")
        return
    
    # 执行回测（逐个处理股票）
    backtest_results = {}
    for symbol in adjusted_symbols:
        try:
            logger.info(f"开始回测: {symbol} ({args.timeframe}线)")
            logger.info(f"日期范围: {start_date} 至 {end_date}")
            
            # 创建回测参数对象
            from src.backtester import BacktestParams
            params = BacktestParams(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=args.timeframe,
                initial_capital=args.capital,
                enable_short=hasattr(args, 'enable_short') and args.enable_short
            )
            
            # 调用回测引擎
            result = engine.run_backtest(params)
            
            # 处理回测结果
            if result and result.success:
                logger.info(f"回测完成: {symbol}，总收益: {result.return_percent:.2f}%")
                backtest_results[symbol] = {
                    'success': True,
                    'return_percent': result.return_percent,
                    'data': getattr(result, 'data', None),
                    'final_capital': result.final_capital,
                    'max_drawdown': result.max_drawdown,
                    'trade_count': result.trade_count,
                    'win_rate': result.win_rate
                }
                
                # 绘图
                if args.plot and hasattr(result, 'data'):
                    plotter = ChanlunPlotter(config.get('plotter', {}))
                    plotter.plot(result.data, symbol)
                
                # 导出数据
                if args.export and hasattr(result, 'data'):
                    exporter = ChanlunExporter(config.get('exporter', {}))
                    exporter.export(result.data, symbol, args.timeframe, args.output_format)
            else:
                error_msg = result.error if result else '未知错误'
                logger.error(f"回测失败: {symbol}，原因: {error_msg}")
                backtest_results[symbol] = {"error": error_msg}
        
        except Exception as e:
            logger.error(f"处理{symbol}回测时出错: {str(e)}", exc_info=True)
            backtest_results[symbol] = {"error": str(e)}
    
    # 保存回测结果
    if backtest_results:
        result_filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join("outputs/backtest", result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(backtest_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"回测结果已保存至: {result_path}")
    
    return backtest_results

def main():
    """主函数，解析参数并执行相应模式"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='缠论股票分析系统')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-s', '--symbols', nargs='*', help='股票代码列表（必填）')
    parser.add_argument('-i', '--interval', type=int, default=300, help='监控间隔(秒)，默认300秒')
    parser.add_argument('-d', '--debug', action='store_true', help='调试模式（输出详细日志）')
    parser.add_argument('-b', '--backtest', action='store_true', help='回测模式（与--mode backtest等效）')
    parser.add_argument('-p', '--plot', action='store_true', help='绘图模式（与其他模式配合使用）')
    parser.add_argument('-e', '--export', action='store_true', help='导出模式（与其他模式配合使用）')
    parser.add_argument('-c', '--count', action='store_true', help='统计机会次数')
    parser.add_argument('-l', '--logfile', help='日志文件路径')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)', default='20200101')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)', default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--period', type=str, help='统计周期(如6m表示6个月)', default='6m')
    parser.add_argument('--daemon', action='store_true', help='以守护进程模式运行')
    parser.add_argument('--capital', type=float, default=100000, help='回测初始资金，默认100000')
    parser.add_argument('--timeframe', choices=['weekly', 'daily', 'minute'], 
                        default='daily', help='时间级别（回测/分析用）')
    
    # 运行模式参数
    parser.add_argument('--mode', choices=[
        'daily', 'minute', 'pre_market', 'intraday', 
        'post_market', 'scan_once', 'monitor', 'status', 'backtest', 'weekly_report'
    ], default='daily', help='运行模式')
    
    # 分钟数据相关参数
    parser.add_argument('--minute_period', default='5m', help='分钟周期，如5m/15m/30m')
    parser.add_argument('--minute_days', type=int, default=3, help='分钟数据天数，默认3天')
    
    # 导出格式参数
    parser.add_argument('--output_format', choices=['csv', 'json', 'xlsx'], 
                        default='csv', help='导出格式: csv/json/xlsx')
    
    args = parser.parse_args()
    
    # 配置日志
    if args.debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    if args.logfile:
        setup_logger(args.logfile)
    
    # 加载系统配置
    try:
        config = load_config('config/system.yaml')
        logger.info("配置文件加载成功: config/system.yaml")
    except Exception as e:
        logger.error(f"配置文件加载失败: {str(e)}")
        return
    
    # 处理股票代码（如果未提供，尝试从配置中获取）
    etf_config = load_etf_config()
    symbols = args.symbols or config.get('default_symbols', [])
    
    if not symbols:
        logger.error("未提供股票代码，请使用--symbols参数指定")
        return
    
    # 调整股票代码格式
    try:
        adjusted_symbols = [adjust_symbol_format(symbol, etf_config) for symbol in symbols]
    except TypeError as e:
        logger.error(f"股票代码处理失败: {str(e)}")
        return
    
    # 根据模式执行相应功能
    if args.mode == 'backtest' or args.backtest:
        # 强制设置为回测模式
        run_backtest_mode(config, adjusted_symbols, args)
    
    elif args.mode == 'status':
        # 市场状态评估模式
        api = StockDataAPI(config.get('data_fetcher', {}))
        calculator = ChanlunCalculator(config.get('chanlun', {}))
        report = evaluate_market_status(api, calculator, adjusted_symbols)
        logger.info(f"市场状态评估结果: {report['overall_status']}")
        logger.info(f"操作建议: {report['recommendation']}")
        
        # 保存状态报告
        status_filename = f"market_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join("outputs/status", status_filename), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    elif args.mode == 'monitor':
        # 监控模式
        run_monitor_mode(config, adjusted_symbols, args.interval)
    
    elif args.mode == 'scan_once':
        # 单次扫描模式
        run_scan_once_mode(config, adjusted_symbols, args.plot, args.export)
    
    elif args.mode == 'pre_market':
        # 盘前报告模式
        run_pre_market_mode(config, adjusted_symbols)
    
    elif args.mode == 'post_market' or args.mode == 'daily':
        # 盘后日报模式
        run_daily_report_mode(config, adjusted_symbols)
    
    elif args.mode == 'weekly_report':
        # 周报告模式
        run_weekly_report_mode(config, adjusted_symbols)
    
    elif args.mode == 'minute':
        # 分钟线分析模式
        logger.info("===== 分钟线分析模式 =====")
        api = StockDataAPI(config.get('data_fetcher', {}))
        calculator = ChanlunCalculator(config.get('chanlun', {}))
        plotter = ChanlunPlotter(config.get('plotter', {})) if args.plot else None
        
        for symbol in adjusted_symbols:
            try:
                df = api.get_minute_data(
                    symbol, 
                    period=args.minute_period,
                    days=args.minute_days
                )
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 分钟线数据为空")
                    continue
                
                result = calculator.calculate(df)
                signals = calculator.detect_signals(result)
                logger.info(f"{symbol} ({args.minute_period}) 信号: {signals}")
                
                if args.plot and plotter:
                    plotter.plot(result, symbol)
                
                if args.export:
                    exporter = ChanlunExporter(config.get('exporter', {}))
                    exporter.export(result, symbol, args.minute_period, args.output_format)
            
            except Exception as e:
                logger.error(f"处理{symbol}分钟线数据时出错: {str(e)}", exc_info=True)
    
    elif args.mode == 'intraday':
        # 盘中模式（实时更新）
        logger.info("===== 盘中模式 =====")
        if not is_trading_hour():
            logger.warning("当前非交易时间，盘中模式仅在交易时间有效")
        
        run_monitor_mode(config, adjusted_symbols, interval=60)  # 缩短监控间隔至1分钟
    
    else:
        logger.warning(f"未知模式: {args.mode}")

if __name__ == "__main__":
    main()