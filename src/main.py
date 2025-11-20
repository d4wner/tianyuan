#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论股票分析系统主程序 - 完整修复版
修复时间参数传递和周线处理逻辑问题
"""

import argparse
import logging
import os
import shutil
import sys
import time
import pandas as pd
import yaml
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 修复导入错误：使用StockDataFetcher并重命名为StockDataAPI
from src.data_fetcher import StockDataFetcher as StockDataAPI
from src.config import load_config, save_config
from src.calculator import ChanlunCalculator
from src.monitor import ChanlunMonitor
from src.backtester import ChanlunBacktester
from src.plotter import ChanlunPlotter
from src.exporter import ChanlunExporter
from src.reporter import generate_pre_market_report, generate_daily_report
from src.notifier import DingdingNotifier
from src.utils import get_last_trading_day, is_trading_hour, get_valid_date_range_str

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
    """配置日志系统"""
    if logfile:
        # 创建文件处理器
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_etf_config(config_path: str = 'config/etfs.yaml') -> dict:
    """加载ETF配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载ETF配置，共{len(config)}个类别")
        return config
    except Exception as e:
        logger.error(f"加载ETF配置失败: {str(e)}")
        return {}

def get_market_prefix(symbol: str, etf_config: dict) -> str:
    """根据ETF配置获取市场前缀"""
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
        if symbol.startswith(('5', '6', '9')):
            return 'sh'
        else:
            return 'sz'
            
    except Exception as e:
        logger.warning(f"获取市场前缀失败: {str(e)}，使用默认逻辑")
        if symbol.startswith(('5', '6', '9')):
            return 'sh'
        else:
            return 'sz'

def adjust_symbol_format(symbol: str, etf_config: dict) -> str:
    """调整股票代码格式，添加市场前缀"""
    market_prefix = get_market_prefix(symbol, etf_config)
    full_symbol = f"{market_prefix}{symbol}"
    logger.info(f"股票代码格式调整: {symbol} -> {full_symbol}")
    return full_symbol

def evaluate_market_status(api, calculator, symbols):
    """
    评估市场状态
    :param api: 数据API
    :param calculator: 缠论计算器
    :param symbols: 股票代码列表
    :return: 市场状态评估结果
    """
    logger.info("===== 市场状态评估 =====")
    
    status_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%:%S"),
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
            # 获取日线数据
            start_date, end_date = get_valid_date_range_str(30)
            df = api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
            
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
            logger.error(f"评估股票 {symbol} 状态失败: {str(e)}")
            status_report["symbols_status"][symbol] = {
                "condition": "error",
                "error": str(e)
            }
    
    # 确定整体市场状态
    total_symbols = len(symbols)
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

def calculate_signal_strength(result_df):
    """
    计算信号强度
    :param result_df: 包含缠论指标的DataFrame
    :return: 信号强度(0-100)
    """
    if result_df.empty:
        return 0
    
    latest = result_df.iloc[-1]
    strength = 50  # 基准强度
    
    # 根据分型增加强度
    if latest.get('top_fractal', False):
        strength -= 10
    if latest.get('bottom_fractal', False):
        strength += 10
    
    # 根据笔增加强度
    if latest.get('pen_end', False):
        if latest.get('pen_type') == 'up':
            strength += 15
        else:
            strength -= 15
    
    # 根据线段增加强度
    if latest.get('segment_end', False):
        if latest.get('segment_type') == 'up':
            strength += 20
        else:
            strength -= 20
    
    # 根据中枢增加强度
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
    
    return max(0, min(100, strength))  # 限制在0-100范围内

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='缠论股票分析系统')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-s', '--symbols', nargs='*', help='股票代码列表')
    parser.add_argument('-i', '--interval', type=int, help='监控间隔(秒)')
    parser.add_argument('-d', '--debug', action='store_true', help='调试模式')
    parser.add_argument('-b', '--backtest', action='store_true', help='回测模式')
    parser.add_argument('-p', '--plot', action='store_true', help='绘图模式')
    parser.add_argument('-e', '--export', action='store_true', help='导出模式')
    parser.add_argument('-c', '--count', action='store_true', help='统计机会次数')
    parser.add_argument('-l', '--logfile', help='日志文件路径')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)', default='20200101')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)', default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--period', type=str, help='统计周期(如6m表示6个月)', default='6m')
    parser.add_argument('--daemon', action='store_true', help='以守护进程模式运行')
    
    # 添加分钟数据相关参数
    parser.add_argument('--mode', choices=['daily', 'minute', 'pre_market', 'intraday', 'post_market', 'scan_once', 'monitor', 'status', 'backtest'], 
                        default='daily', help='运行模式: daily(日线), minute(分钟线), pre_market(盘前), intraday(盘中), post_market(盘后), scan_once(单次扫描), monitor(监控模式), status(状态评估), backtest(回测模式)')
    parser.add_argument('--minute_period', default='5m', help='分钟周期')
    parser.add_argument('--minute_days', type=int, default=3, help='分钟数据天数')
    
    # 添加导出格式参数
    parser.add_argument('--output_format', choices=['csv', 'json', 'xlsx'], 
                        default='csv', help='导出格式: csv/json/xlsx')
    
    # 添加新参数
    parser.add_argument('--exceed_position', action='store_true', help='测试仓位超限场景')
    parser.add_argument('--clean_cache', action='store_true', help='清理数据缓存')
    parser.add_argument('--skip_network_check', action='store_true', help='跳过网络检查')
    parser.add_argument('--test_notification', action='store_true', help='测试钉钉通知功能')
    parser.add_argument('--notification_template', choices=['signal', 'error', 'alert'], 
                        default='signal', help='通知模板: signal(信号), error(错误), alert(警报)')
    # 新增状态评估参数
    parser.add_argument('--status', action='store_true', help='评估市场状态')
    
    # 新增时间级别参数
    parser.add_argument('--timeframe', choices=['daily', 'weekly', 'minute'], 
                        default='daily', help='K线级别: daily(日线), weekly(周线), minute(分钟线)')

    # 添加ETF参数作为symbols的别名
    parser.add_argument('--etf', nargs='*', help='ETF代码列表（与--symbols相同）')
    
    # 添加报告级别参数
    parser.add_argument('--report_level', choices=['simple', 'detailed'], default='simple', help='报告级别: simple(简单), detailed(详细)')

    args = parser.parse_args()
    
    # 如果提供了--etf参数，将其值赋给--symbols
    if args.etf and not args.symbols:
        args.symbols = args.etf
    
    # 设置默认日志文件
    if not args.logfile:
        # 创建日志目录
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        args.logfile = os.path.join(log_dir, 'system.log')
    
    # 配置日志
    setup_logger(args.logfile)
    
    try:
        # 加载系统配置
        system_config = load_config()
        
        # 加载ETF配置
        etf_config = load_etf_config()
        
        # 清理缓存
        if args.clean_cache:
            logger.info("===== 清理缓存 =====")
            cache_dirs = [
                'data/daily',
                'data/minute',
                'data/signals',
                'outputs/backtest',
                'outputs/plots',
                'outputs/reports'
            ]
            
            for dir in cache_dirs:
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                    os.makedirs(dir)
                    logger.info(f"已清理: {dir}")
            
            logger.info("缓存清理完成")
            return
        
        # 仓位超限测试
        if args.exceed_position:
            logger.warning("===== 仓位超限测试 =====")
            # 强制设置仓位超限
            system_config['risk_management']['max_total_position'] = 1.5
            system_config['risk_management']['max_single_position'] = 0.6
            save_config(system_config)
            logger.warning("已设置仓位超限参数")
        
        # 更新配置
        if args.interval:
            system_config['monitoring']['interval'] = args.interval
            save_config(system_config)
        
        # 创建API实例
        api = StockDataAPI(
            max_retries=system_config.get('data_fetcher', {}).get('max_retries', 3),
            timeout=system_config.get('data_fetcher', {}).get('timeout', 10)
        )
        
        # 创建计算器
        calculator = ChanlunCalculator(
            config=system_config.get('chanlun', {})
        )
        
        # 创建通知器
        notifier = DingdingNotifier()
        
        # 钉钉通知测试
        if args.test_notification:
            logger.info("===== 钉钉通知测试 =====")
            
            # 根据选择的模板发送测试通知
            if args.notification_template == 'signal':
                logger.info("发送交易信号测试通知")
                signal_details = {
                    "signal_type": "buy",
                    "price": 4.55,
                    "target_price": 4.85,
                    "stoploss": 4.40,
                    "position_size": 0.3,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "strategy": "缠论底分型突破",
                    "confidence": 0.85
                }
                symbol = args.symbols[0] if args.symbols else "510300"
                notifier.send_signal(symbol, signal_details)
            elif args.notification_template == 'error':
                logger.info("发送错误通知测试")
                error_msg = "系统测试错误: 钉钉通知功能验证"
                notifier.send_error(error_msg)
            elif args.notification_template == 'alert':
                logger.info("发送风险警报测试通知")
                alert_details = {
                    "alert_type": "止损触发",
                    "price": 4.40,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": "价格跌破关键支撑位",
                    "suggestion": "立即平仓"
                }
                symbol = args.symbols[0] if args.symbols else "510300"
                notifier.send_alert(symbol, alert_details)
            
            logger.info("钉钉通知测试完成，请检查钉钉是否收到消息")
            return
        
        # 新增状态评估模式
        if args.status or args.mode == 'status':
            logger.info("===== 市场状态评估模式 =====")
            
            # 如果没有指定股票代码，从配置文件中加载
            if not args.symbols:
                try:
                    # 获取所有ETF代码
                    all_etfs = []
                    for category in etf_config:
                        if category != 'global' and isinstance(etf_config[category], dict):
                            all_etfs.extend(list(etf_config[category].keys()))
                    
                    # 确保所有股票代码都是字符串类型
                    args.symbols = [str(symbol) for symbol in all_etfs]
                    logger.info(f"从配置文件加载 {len(args.symbols)} 只ETF")
                    
                except Exception as e:
                    logger.error(f"加载ETF配置异常: {str(e)}")
                    sys.exit(1)
            
            # 评估市场状态
            status_report = evaluate_market_status(api, calculator, args.symbols)
            
            # 打印评估结果
            logger.info(f"市场状态评估完成 - 时间: {status_report['timestamp']}")
            logger.info(f"整体市场状态: {status_report['overall_status']}")
            logger.info(f"趋势股数量: {status_report['trending_count']}")
            logger.info(f"震荡股数量: {status_report['ranging_count']}")
            logger.info(f"下跌股数量: {status_report['declining_count']}")
            logger.info(f"突破股数量: {status_report['breakout_count']}")
            logger.info(f"操作建议: {status_report['recommendation']}")
            
            # 打印每只股票的状态
            logger.info("===== 个股状态详情 =====")
            for symbol, status in status_report['symbols_status'].items():
                if 'error' in status:
                    logger.info(f"{symbol}: 错误 - {status['error']}")
                else:
                    logger.info(f"{symbol}: {status['condition']}, 价格: {status['price']}, 量能: {status['volume']}, 信号强度: {status['signal_strength']}")
            
            return
        
        # 如果没有指定股票代码，从配置文件中加载
        if not args.symbols:
            try:
                # 获取所有ETF代码
                all_etfs = []
                for category in etf_config:
                    if category != 'global' and isinstance(etf_config[category], dict):
                        all_etfs.extend(list(etf_config[category].keys()))
                
                # 确保所有股票代码都是字符串类型
                args.symbols = [str(symbol) for symbol in all_etfs]
                logger.info(f"从配置文件加载 {len(args.symbols)} 只ETF")
                
            except Exception as e:
                logger.error(f"加载ETF配置异常: {str(e)}")
                sys.exit(1)
        else:
            # 确保命令行传入的股票代码也是字符串类型
            args.symbols = [str(symbol) for symbol in args.symbols]
        
        # 机会统计模式
        if args.count:
            logger.info("===== 机会统计模式 =====")
            # 解析统计周期
            period_months = int(args.period.rstrip('m'))
            
            for symbol in args.symbols:
                logger.info(f"统计股票: {symbol} 周期: {period_months}个月")
                
                # 获取有效的日期范围字符串
                start_date_str, end_date_str = get_valid_date_range_str(period_months*30)
                logger.info(f"使用日期范围: {start_date_str} 至 {end_date_str}")
                
                # 调整股票代码格式
                full_symbol = adjust_symbol_format(symbol, etf_config)
                
                # 获取数据
                df = api.get_daily_data(full_symbol, start_date=start_date_str, end_date=end_date_str)
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 获取数据为空，跳过统计")
                    continue
                
                # 计算缠论指标
                result = calculator.calculate(df)
                
                # 统计机会次数
                if 'signal' in result.columns:
                    opportunities = len(result[result['signal'] == 'buy'])
                else:
                    # 使用底分型作为机会指标
                    logger.warning("数据中无信号列，使用分型作为机会指标")
                    opportunities = len(result[result['bottom_fractal'] == True])
                
                logger.info(f"股票 {symbol} 在过去 {period_months} 个月共有 {opportunities} 次交易机会")
            
            logger.info("机会统计完成")
            return
        
        # 盘前模式
        if args.mode == 'pre_market':
            logger.info("===== 盘前模式 =====")
            if not args.symbols:
                logger.error("盘前模式需要指定股票代码")
                return
                
            # 使用有效的日期范围字符串
            start_date_str, end_date_str = get_valid_date_range_str(30)
            logger.info(f"使用日期范围: {start_date_str} 至 {end_date_str}")
            
            # 使用正式的generate_pre_market_report函数
            report = generate_pre_market_report(args.symbols, api, calculator, start_date_str, end_date_str)
            logger.info(f"盘前报告生成完成: {report}")
            return
        
        # 盘中模式
        if args.mode == 'intraday':
            logger.info("===== 盘中模式 =====")
            if not args.symbols:
                logger.error("盘中模式需要指定股票代码")
                return
                
            monitor = ChanlunMonitor(
                system_config=system_config,
                api=api,
                calculator=calculator,
                notifier=notifier
            )
            
            # 添加监控股票
            for symbol in args.symbols:
                monitor.add_symbol(symbol)
            
            # 启动监控
            monitor.start()
            return
        
        # 盘后模式
        if args.mode == 'post_market':
            logger.info("===== 盘后模式 =====")
            if not args.symbols:
                logger.error("盘后模式需要指定股票代码")
                return
                
            # 使用有效的日期范围
            start_date_str, end_date_str = get_valid_date_range_str(30)
            logger.info(f"使用日期范围: {start_date_str} 至 {end_date_str}")
            
            # 使用正式的generate_daily_report函数
            report = generate_daily_report(args.symbols, api, calculator, start_date_str, end_date_str)
            logger.info(f"盘后报告生成完成: {report}")
            return
        
        # 单次扫描模式
        if args.mode == 'scan_once':
            logger.info("===== 单次扫描模式 =====")
            if not args.symbols:
                logger.error("单次扫描模式需要指定股票代码")
                return
                
            for symbol in args.symbols:
                # 双重确保股票代码是字符串类型
                symbol = str(symbol)
                logger.info(f"扫描股票: {symbol}")
                
                # 调整股票代码格式
                full_symbol = adjust_symbol_format(symbol, etf_config)
                
                # 获取分钟数据
                df = api.get_minute_data(full_symbol, period=args.minute_period, days=args.minute_days)
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 获取数据为空，跳过扫描")
                    continue
                
                # 计算缠论指标
                result = calculator.calculate(df)
                
                # 检测交易信号
                if 'signal' in result.columns and 'buy' in result['signal'].values:
                    logger.info(f"检测到交易信号: {symbol}")
                    # 发送钉钉通知
                    signal_details = result[result['signal'] == 'buy'].iloc[-1].to_dict()
                    notifier.send_signal(symbol, signal_details)
                else:
                    logger.info(f"未检测到交易信号: {symbol}")
            
            logger.info("扫描完成，进程退出")
            return
        
        # 监控模式
        if args.mode == 'monitor':
            logger.info("===== 监控模式 =====")
            if not args.symbols:
                logger.error("监控模式需要指定股票代码")
                return
                
            monitor = ChanlunMonitor(
                system_config=system_config,
                api=api,
                calculator=calculator,
                notifier=notifier
            )
            
            # 添加监控股票
            for symbol in args.symbols:
                monitor.add_symbol(symbol)
            
            # 启动监控
            monitor.start()
            return
        
        # 回测模式
        if args.backtest or args.mode == 'backtest':
            logger.info("===== 回测模式 =====")
            if not args.symbols:
                logger.error("回测模式需要指定股票代码")
                return
                
            backtester = ChanlunBacktester(
                api=api,
                calculator=calculator,
                config=system_config.get('backtest', {})
            )
            
            for symbol in args.symbols:
                logger.info(f"回测股票: {symbol}")
                
                # 使用有效的日期范围字符串
                start_date_str, end_date_str = get_valid_date_range_str(30)
                logger.info(f"{args.timeframe}线数据回测: {symbol} {start_date_str} 至 {end_date_str}")
                
                # 调整股票代码格式
                full_symbol = adjust_symbol_format(symbol, etf_config)
                
                # 根据时间级别获取数据
                if args.timeframe == 'daily':
                    df = api.get_daily_data(full_symbol, start_date=start_date_str, end_date=end_date_str)
                elif args.timeframe == 'weekly':
                    # 使用周线数据
                    df = api.get_weekly_data(full_symbol, start_date=start_date_str, end_date=end_date_str)
                elif args.timeframe == 'minute':
                    # 使用分钟数据
                    df = api.get_minute_data(full_symbol, period=args.minute_period, days=args.minute_days)
                else:
                    logger.error(f"不支持的时间级别: {args.timeframe}")
                    continue
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 获取数据为空，跳过回测")
                    continue
                
                # 执行回测 - 修复：传递timeframe参数和时间参数
                backtest_result = backtester.run(
                    df, 
                    timeframe=args.timeframe,
                    start=start_date_str,  # 新增：传递开始日期
                    end=end_date_str      # 新增：传递结束日期
                )
                logger.info(f"回测结果: {backtest_result}")
            
            logger.info("回测完成")
            return
        
        # 调试模式
        if args.debug:
            logger.info("===== 调试模式 =====")
            if not args.symbols:
                logger.error("调试模式需要指定股票代码")
                return
                
            symbol = args.symbols[0]
            logger.info(f"处理股票: {symbol}")
            
            # 调整股票代码格式
            full_symbol = adjust_symbol_format(symbol, etf_config)
            
            if args.mode == 'minute':
                logger.info(f"请求分钟数据: {symbol} {args.minute_period} 最近 {args.minute_days} 天")
                df = api.get_minute_data(full_symbol, period=args.minute_period, days=args.minute_days)
            else:
                # 获取有效日期范围字符串
                start_date_str, end_date_str = get_valid_date_range_str(30)
                logger.info(f"请求日线数据: {symbol} {start_date_str} 至 {end_date_str}")
                df = api.get_daily_data(full_symbol, start_date=start_date_str, end_date=end_date_str)
            
            if df.empty:
                logger.warning(f"股票 {symbol} 获取数据为空")
            else:
                logger.info(f"获取数据成功: {len(df)} 条")
                # 计算缠论指标
                result = calculator.calculate(df)
                logger.info(f"缠论计算结果: {len(result)} 条")
                # 打印前几行
                logger.info(f"\n{result.head()}")
            
            logger.info("调试模式完成")
            return
        
        # 绘图模式
        if args.plot:
            logger.info("===== 绘图模式 =====")
            if not args.symbols:
                logger.error("绘图模式需要指定股票代码")
                return
                
            plotter = ChanlunPlotter(
                config=system_config.get('plot', {})
            )
            
            for symbol in args.symbols:
                logger.info(f"绘制股票: {symbol}")
                
                # 调整股票代码格式
                full_symbol = adjust_symbol_format(symbol, etf_config)
                
                if args.mode == 'minute':
                    logger.info(f"分钟数据绘图: {symbol} {args.minute_period} 最近 {args.minute_days} 天")
                    df = api.get_minute_data(full_symbol, period=args.minute_period, days=args.minute_days)
                else:
                    # 获取有效日期范围字符串
                    start_date_str, end_date_str = get_valid_date_range_str(30)
                    logger.info(f"日线数据绘图: {symbol} {start_date_str} 至 {end_date_str}")
                    df = api.get_daily_data(full_symbol, start_date=start_date_str, end_date=end_date_str)
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 获取数据为空，跳过绘图")
                    continue
                
                # 计算缠论指标
                result = calculator.calculate(df)
                
                # 绘制图表
                plotter.plot(result, symbol)
            
            logger.info("绘图完成")
            return
        
        # 导出模式
        if args.export:
            logger.info("===== 导出模式 =====")
            if not args.symbols:
                logger.error("导出模式需要指定股票代码")
                return
                
            exporter = ChanlunExporter(
                config=system_config.get('export', {})
            )
            exporter.output_format = args.output_format
            
            for symbol in args.symbols:
                logger.info(f"导出股票: {symbol} 格式: {args.output_format}")
                
                # 调整股票代码格式
                full_symbol = adjust_symbol_format(symbol, etf_config)
                
                if args.mode == 'minute':
                    logger.info(f"分钟数据导出: {symbol} {args.minute_period} 最近 {args.minute_days} 天")
                    df = api.get_minute_data(full_symbol, period=args.minute_period, days=args.minute_days)
                else:
                    # 获取有效日期范围字符串
                    start_date_str, end_date_str = get_valid_date_range_str(30)
                    logger.info(f"日线数据导出: {symbol} {start_date_str} 至 {end_date_str}")
                    df = api.get_daily_data(full_symbol, start_date=start_date_str, end_date=end_date_str)
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 获取数据为空，跳过导出")
                    continue
                
                # 计算缠论指标
                result = calculator.calculate(df)
                
                # 导出数据
                exporter.export(result, symbol)
            
            logger.info(f"导出完成: 格式={args.output_format}")
            return
        
        # 监控模式
        logger.info("===== 监控模式 =====")
        if not args.symbols:
            logger.error("监控模式需要指定股票代码")
            return
            
        monitor = ChanlunMonitor(
            system_config=system_config,
            api=api,
            calculator=calculator,
            notifier=notifier
        )
        
        # 添加监控股票
        for symbol in args.symbols:
            monitor.add_symbol(symbol)
        
        # 启动监控
        monitor.start()
        
    except Exception as e:
        logger.critical(f"系统崩溃: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        
        # 发送钉钉通知
        try:
            # 创建新的通知器实例
            notifier = DingdingNotifier()
            notifier.send_error(f"系统崩溃: {str(e)}")
        except Exception as notifier_error:
            logger.error(f"发送钉钉通知失败: {str(notifier_error)}")
        
        # 退出程序
        sys.exit(1)

if __name__ == "__main__":
    main()