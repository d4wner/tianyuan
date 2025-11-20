#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论回测系统 - 完整修复版
仅移除EmailNotifier相关逻辑，保留所有回测核心功能
原始行数1127行，调整后保持功能完整且行数接近
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

# 忽略警告
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入依赖模块（移除EmailNotifier导入）
try:
    from src.config import load_config
    from src.data_fetcher import StockDataAPI
    from src.calculator import ChanlunCalculator
    from src.notifier import DingdingNotifier  # 仅保留钉钉通知器
    from src.utils import (
        get_last_trading_day, 
        is_trading_hour, 
        get_valid_date_range_str,
        validate_date_format,
        convert_date_format,
        calculate_max_drawdown,
        calculate_sharpe_ratio,
        calculate_sortino_ratio
    )
    from src.reporter import generate_pre_market_report, generate_daily_report
    from src.exporter import ChanlunExporter
    from src.plotter import ChanlunPlotter
except ImportError as e:
    logging.error(f"导入依赖模块失败: {str(e)}")
    raise

# 配置日志（保持原始配置，不冗余）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'logs', 'backtest.log'), encoding='utf-8')
    ]
)
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
logger = logging.getLogger('ChanlunBacktest')

# 全局常量定义（保留原始常量，无新增）
VALID_TIMEFRAMES = ['daily', 'weekly', 'minute', '60m', '30m', '15m', '5m']
MIN_DATA_POINTS = 10
MAX_RETRY_COUNT = 3
DATE_FORMAT = '%Y-%m-%d'
DATE_FORMAT_ALT = '%Y%m%d'
REPORT_DIR = os.path.join(project_root, 'reports')
CHART_DIR = os.path.join(project_root, 'charts')
EXPORT_DIR = os.path.join(project_root, 'exports')

# 数据类定义（保留原始结构，无修改）
@dataclass
class BacktestParams:
    """回测参数数据类"""
    symbol: str
    start_date: str
    end_date: str
    timeframe: str = 'daily'
    initial_capital: float = 100000.0
    transaction_cost: float = 0.0005
    slippage: float = 0.001
    stop_loss_ratio: float = 0.05
    take_profit_ratio: float = 0.1
    min_holding_days: int = 1
    max_holding_days: int = 90

@dataclass
class TradeRecord:
    """交易记录数据类"""
    date: datetime
    type: str  # 'buy' 或 'sell'
    price: float
    shares: int
    cost: float
    capital: float
    position: int
    profit: float = 0.0
    holding_days: int = 0
    signal_source: str = ''
    notes: str = ''

@dataclass
class BacktestResult:
    """回测结果数据类"""
    success: bool = False
    error: str = ''
    params: Optional[BacktestParams] = None
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    return_percent: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    trade_records: List[TradeRecord] = field(default_factory=list)
    daily_equity: List[Dict[str, Any]] = field(default_factory=list)
    actual_date_range: Dict[str, str] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    charts: Dict[str, Any] = field(default_factory=dict)

class BacktestEngine:
    """缠论回测引擎核心类 - 仅移除EmailNotifier相关逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化回测引擎"""
        self.config = config
        self.data_api = None
        self.calculator = None
        self.notifier = None  # 仅保留钉钉通知器
        self.plotter = None
        self.exporter = None
        self.performance_metrics = {}
        self.trade_records = []
        self.historical_data = {}
        self.daily_equity = []
        self.backtest_params = None
        
        self._initialize_components()
        self._create_output_directories()
        logger.info("缠论回测引擎初始化完成（仅保留钉钉通知）")
    
    def _initialize_components(self):
        """初始化所有组件（移除EmailNotifier相关）"""
        logger.info("初始化回测组件...")
        
        # 初始化数据API
        try:
            data_fetcher_config = self.config.get('data_fetcher', {})
            self.data_api = StockDataAPI(
                max_retries=data_fetcher_config.get('max_retries', MAX_RETRY_COUNT),
                timeout=data_fetcher_config.get('timeout', 10),
            )
            logger.info("数据API初始化成功")
        except Exception as e:
            logger.error(f"数据API初始化失败: {str(e)}")
            raise
        
        # 初始化缠论计算器
        try:
            chanlun_config = self.config.get('chanlun', {})
            self.calculator = ChanlunCalculator(
                fractal_sensitivity=chanlun_config.get('fractal_sensitivity', 2),
                min_bi_length=chanlun_config.get('min_bi_length', 3),
                min_xd_length=chanlun_config.get('min_xd_length', 3),
                zs_range_ratio=chanlun_config.get('zs_range_ratio', 0.01),
                signal_threshold=chanlun_config.get('signal_threshold', 0.8)
            )
            logger.info("缠论计算器初始化成功")
        except Exception as e:
            logger.error(f"缠论计算器初始化失败: {str(e)}")
            raise
        
        # 初始化钉钉通知器（移除EmailNotifier相关代码）
        try:
            if self.config.get('notifications', {}).get('enabled', False):
                self.notifier = DingdingNotifier(
                    webhook=self.config['notifications'].get('dingding_webhook'),
                    secret=self.config['notifications'].get('dingding_secret'),
                    at_mobiles=self.config['notifications'].get('at_mobiles', []),
                    at_all=self.config['notifications'].get('at_all', False)
                )
                if self.notifier.test_connection():
                    logger.info("钉钉通知器初始化并测试成功")
                else:
                    logger.warning("钉钉通知器初始化成功，但连接测试失败")
            else:
                logger.info("通知功能未启用")
        except Exception as e:
            logger.error(f"钉钉通知器初始化失败: {str(e)}")
            raise
        
        # 初始化绘图器
        try:
            plotter_config = self.config.get('plotter', {})
            self.plotter = ChanlunPlotter(
                figsize=plotter_config.get('figsize', (16, 12)),
                dpi=plotter_config.get('dpi', 150),
                style=plotter_config.get('style', 'seaborn-v0_8-darkgrid'),
                font_size=plotter_config.get('font_size', 10),
                chinese_font=plotter_config.get('chinese_font', 'SimHei')
            )
            logger.info("绘图器初始化成功")
        except Exception as e:
            logger.error(f"绘图器初始化失败: {str(e)}")
            raise
        
        # 初始化数据导出器
        try:
            exporter_config = self.config.get('exporter', {})
            self.exporter = ChanlunExporter(
                format=exporter_config.get('format', ['csv', 'excel', 'json']),
                encoding=exporter_config.get('encoding', 'utf-8'),
                decimal_places=exporter_config.get('decimal_places', 4)
            )
            logger.info("数据导出器初始化成功")
        except Exception as e:
            logger.error(f"数据导出器初始化失败: {str(e)}")
            raise
    
    def _create_output_directories(self):
        """创建所有输出目录"""
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            os.makedirs(CHART_DIR, exist_ok=True)
            os.makedirs(EXPORT_DIR, exist_ok=True)
            logger.info(f"输出目录创建完成: {REPORT_DIR}, {CHART_DIR}, {EXPORT_DIR}")
        except Exception as e:
            logger.error(f"创建输出目录失败: {str(e)}")
            raise
    
    def run_comprehensive_backtest(self, 
                                  symbol: str, 
                                  start_date: str, 
                                  end_date: str, 
                                  timeframe: str = 'daily',
                                  initial_capital: float = 100000.0,
                                  transaction_cost: float = 0.0005,
                                  slippage: float = 0.001) -> BacktestResult:
        """运行全面回测（保留所有核心功能）"""
        result = BacktestResult()
        result.initial_capital = initial_capital
        result.final_capital = initial_capital
        
        logger.info("=" * 80)
        logger.info("开始执行全面缠论回测")
        logger.info(f"股票代码: {symbol} | 时间级别: {timeframe} | 日期范围: {start_date} ~ {end_date}")
        logger.info(f"初始资金: {initial_capital:.2f} 元 | 交易成本: {transaction_cost:.4f} | 滑点率: {slippage:.4f}")
        logger.info("=" * 80)
        
        try:
            # 1. 参数验证
            validation_result = self._validate_backtest_params(symbol, start_date, end_date, timeframe)
            if not validation_result['success']:
                error_msg = f"参数验证失败: {validation_result['error']}"
                logger.error(error_msg)
                result.success = False
                result.error = error_msg
                self._send_error_notification(error_msg)
                return result
            
            formatted_start = validation_result['formatted_start']
            formatted_end = validation_result['formatted_end']
            logger.info(f"格式化后日期范围: {formatted_start} ~ {formatted_end}")
            
            # 初始化回测参数
            self.backtest_params = BacktestParams(
                symbol=symbol,
                start_date=formatted_start,
                end_date=formatted_end,
                timeframe=timeframe,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=slippage
            )
            result.params = self.backtest_params
            
            # 2. 数据获取
            data_result = self._acquire_and_validate_data(
                symbol=symbol,
                start_date=formatted_start,
                end_date=formatted_end,
                timeframe=timeframe
            )
            if not data_result['success']:
                error_msg = f"数据获取失败: {data_result['error']}"
                logger.error(error_msg)
                result.success = False
                result.error = error_msg
                self._send_error_notification(error_msg)
                return result
            
            df = data_result['data']
            actual_start = df['date'].min().strftime(DATE_FORMAT)
            actual_end = df['date'].max().strftime(DATE_FORMAT)
            logger.info(f"数据获取成功: {len(df)} 条记录 | 实际日期范围: {actual_start} ~ {actual_end}")
            
            result.actual_date_range = {
                'requested_start': formatted_start,
                'requested_end': formatted_end,
                'actual_start': actual_start,
                'actual_end': actual_end,
                'data_count': len(df)
            }
            self.historical_data[symbol] = df
            
            # 3. 缠论计算
            calculation_result = self._perform_chanlun_calculations(df, timeframe)
            if not calculation_result['success']:
                error_msg = f"缠论计算失败: {calculation_result['error']}"
                logger.error(error_msg)
                result.success = False
                result.error = error_msg
                self._send_error_notification(error_msg)
                return result
            
            calculated_df = calculation_result['data']
            logger.info("缠论计算完成: 分型、笔、线段、中枢、买卖点已全部计算")
            
            # 4. 回测执行
            backtest_exec_result = self._execute_backtest_logic(calculated_df)
            if not backtest_exec_result['success']:
                error_msg = f"回测执行失败: {backtest_exec_result['error']}"
                logger.error(error_msg)
                result.success = False
                result.error = error_msg
                self._send_error_notification(error_msg)
                return result
            
            # 更新回测结果
            result.success = True
            result.final_capital = backtest_exec_result['final_capital']
            result.total_return = backtest_exec_result['total_return']
            result.return_percent = backtest_exec_result['return_percent']
            result.annual_return = backtest_exec_result['annual_return']
            result.max_drawdown = backtest_exec_result['max_drawdown']
            result.sharpe_ratio = backtest_exec_result['sharpe_ratio']
            result.sortino_ratio = backtest_exec_result['sortino_ratio']
            result.trade_count = backtest_exec_result['trade_count']
            result.buy_count = backtest_exec_result['buy_count']
            result.sell_count = backtest_exec_result['sell_count']
            result.winning_trades = backtest_exec_result['winning_trades']
            result.losing_trades = backtest_exec_result['losing_trades']
            result.win_rate = backtest_exec_result['win_rate']
            result.average_profit = backtest_exec_result['average_profit']
            result.average_loss = backtest_exec_result['average_loss']
            result.profit_factor = backtest_exec_result['profit_factor']
            result.trade_records = backtest_exec_result['trade_records']
            result.daily_equity = backtest_exec_result['daily_equity']
            
            # 5. 报告生成
            report_result = self._generate_comprehensive_reports(result)
            result.report = report_result
            logger.info(f"报告生成完成: {report_result['report_path']}")
            
            # 6. 图表生成
            chart_result = self._generate_detailed_charts(result, calculated_df)
            result.charts = chart_result
            logger.info(f"图表生成完成: {chart_result['chart_dir']}")
            
            # 7. 数据导出
            export_result = self._export_backtest_results(result)
            logger.info(f"数据导出完成: {export_result['export_dir']}")
            
            # 8. 发送钉钉通知（移除邮件通知）
            if self.config.get('notifications', {}).get('enabled', False) and self.notifier:
                self._send_backtest_complete_notification(result)
            
            # 输出最终结果
            logger.info("=" * 80)
            logger.info("回测完成！最终结果摘要")
            logger.info(f"初始资金: {initial_capital:.2f} 元 | 最终资金: {result.final_capital:.2f} 元")
            logger.info(f"总回报: {result.total_return:.2f} 元 ({result.return_percent:.2f}%) | 年化回报: {result.annual_return:.2f}%")
            logger.info(f"最大回撤: {result.max_drawdown:.2f}% | 夏普比率: {result.sharpe_ratio:.2f} | 索提诺比率: {result.sortino_ratio:.2f}")
            logger.info(f"交易次数: {result.trade_count} | 胜率: {result.win_rate:.2f}% | 盈利因子: {result.profit_factor:.2f}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            error_msg = f"回测过程异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.success = False
            result.error = error_msg
            self._send_error_notification(error_msg)
            return result
    
    def _validate_backtest_params(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> Dict[str, Any]:
        """验证回测参数有效性"""
        logger.info("验证回测参数...")
        
        # 验证股票代码
        if not symbol or len(symbol) < 3 or len(symbol) > 10:
            return {'success': False, 'error': f"无效股票代码: {symbol}（长度3-10字符）"}
        
        # 验证时间级别
        if timeframe not in VALID_TIMEFRAMES:
            return {'success': False, 'error': f"不支持的时间级别: {timeframe}，支持：{VALID_TIMEFRAMES}"}
        
        # 验证并格式化日期
        try:
            # 验证开始日期
            if not validate_date_format(start_date):
                start_date = convert_date_format(start_date)
                if not validate_date_format(start_date):
                    return {'success': False, 'error': f"无效开始日期: {start_date}（支持YYYY-MM-DD/YYYYMMDD）"}
            
            # 验证结束日期
            if not validate_date_format(end_date):
                end_date = convert_date_format(end_date)
                if not validate_date_format(end_date):
                    return {'success': False, 'error': f"无效结束日期: {end_date}（支持YYYY-MM-DD/YYYYMMDD）"}
            
            # 验证日期顺序
            start_dt = datetime.strptime(start_date, DATE_FORMAT)
            end_dt = datetime.strptime(end_date, DATE_FORMAT)
            if start_dt >= end_dt:
                return {'success': False, 'error': f"开始日期 {start_date} 不能晚于结束日期 {end_date}"}
            
            # 验证日期范围
            max_date_range = timedelta(days=365*10)
            if end_dt - start_dt > max_date_range:
                return {'success': False, 'error': "日期范围超过10年，请缩小范围"}
            
            # 验证日期不超过当前
            current_dt = datetime.now()
            if end_dt > current_dt:
                return {'success': False, 'error': f"结束日期 {end_date} 不能超过当前日期 {current_dt.strftime(DATE_FORMAT)}"}
        
        except Exception as e:
            return {'success': False, 'error': f"日期验证失败: {str(e)}"}
        
        # 防御性检查：避免传入DataFrame
        try:
            self._validate_symbol_not_dataframe(symbol)
        except ValueError as e:
            return {'success': False, 'error': str(e)}
        
        logger.info("参数验证通过")
        return {
            'success': True,
            'formatted_start': start_date,
            'formatted_end': end_date
        }
    
    def _validate_symbol_not_dataframe(self, symbol: Any):
        """防御性检查：确保symbol不是DataFrame"""
        if symbol is None:
            raise ValueError("股票代码不能为None")
        
        if isinstance(symbol, (pd.DataFrame, pd.Series)):
            logger.error("DataFrame/Series被当作股票代码传递")
            raise ValueError("无效股票代码类型：不能是DataFrame/Series")
        
        symbol_str = str(symbol)
        if len(symbol_str) > 100:
            logger.error(f"疑似DataFrame传入: {symbol_str[:100]}...")
            raise ValueError("无效股票代码：字符串过长（疑似DataFrame）")
        
        dataframe_indicators = ['DataFrame', 'Series', 'open', 'high', 'low', 'close', 'volume']
        if any(ind in symbol_str for ind in dataframe_indicators):
            logger.error(f"检测到DataFrame特征: {symbol_str[:200]}")
            raise ValueError("无效股票代码：包含DataFrame特征关键词")
    
    def _acquire_and_validate_data(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> Dict[str, Any]:
        """获取并验证数据（修复日期处理逻辑）"""
        logger.info(f"获取 {symbol} {timeframe} 级别数据...")
        
        try:
            # 根据时间级别获取数据
            if timeframe == 'daily':
                df = self.data_api.get_daily_data(symbol, start_date, end_date)
            elif timeframe == 'weekly':
                df = self.data_api.get_weekly_data(symbol, start_date, end_date)
            elif timeframe == 'minute':
                df = self.data_api.get_minute_data(symbol, interval='1m', limit=1440)
            elif timeframe == '5m':
                df = self.data_api.get_minute_data(symbol, interval='5m', limit=288)
            elif timeframe == '15m':
                df = self.data_api.get_minute_data(symbol, interval='15m', limit=96)
            elif timeframe == '30m':
                df = self.data_api.get_minute_data(symbol, interval='30m', limit=48)
            elif timeframe == '60m':
                df = self.data_api.get_minute_data(symbol, interval='60m', limit=24)
            else:
                return {'success': False, 'error': f'不支持的时间级别: {timeframe}'}
            
            # 数据有效性校验
            if df.empty:
                return {'success': False, 'error': f'未获取到 {symbol} 的数据'}
            if len(df) < MIN_DATA_POINTS:
                return {'success': False, 'error': f'数据不足 {MIN_DATA_POINTS} 条（当前 {len(df)} 条）'}
            
            # 检查必要列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {'success': False, 'error': f'缺失必要列: {missing_columns}（必需：{required_columns}）'}
            
            # 日期处理
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                if df.empty:
                    return {'success': False, 'error': '日期格式全部无效'}
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                df = df.drop(columns=['timestamp'])
            else:
                return {'success': False, 'error': '缺少日期列（date/timestamp）'}
            
            # 数据清洗
            df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            start_dt = datetime.strptime(start_date, DATE_FORMAT)
            end_dt = datetime.strptime(end_date, DATE_FORMAT)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            # 价格和成交量校验
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=price_columns)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df.dropna(subset=['volume']).query('volume >= 0')
            
            if len(df) < MIN_DATA_POINTS:
                return {'success': False, 'error': f'清洗后数据不足 {MIN_DATA_POINTS} 条'}
            
            logger.info(f"数据获取完成：{len(df)} 条有效记录")
            return {'success': True, 'data': df}
            
        except Exception as e:
            logger.error(f"数据获取异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'数据获取失败: {str(e)}'}
    
    def _perform_chanlun_calculations(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """执行完整缠论计算（分型、笔、线段、中枢、买卖点）"""
        logger.info("执行缠论计算...")
        
        try:
            calculated_df = df.copy()
            
            # 根据时间级别调整参数
            if timeframe in ['weekly', 'daily']:
                self.calculator.set_params(fractal_sensitivity=2, min_bi_length=3, min_xd_length=3, zs_range_ratio=0.015)
            elif timeframe in ['60m', '30m']:
                self.calculator.set_params(fractal_sensitivity=1, min_bi_length=2, min_xd_length=2, zs_range_ratio=0.01)
            else:
                self.calculator.set_params(fractal_sensitivity=1, min_bi_length=2, min_xd_length=2, zs_range_ratio=0.008)
            
            # 1. 计算分型
            calculated_df = self.calculator.calculate_fx(calculated_df)
            logger.info(f"分型计算完成：顶分型 {calculated_df['top_fractal'].sum()} 个，底分型 {calculated_df['bottom_fractal'].sum()} 个")
            
            # 2. 计算笔
            calculated_df = self.calculator.calculate_bi(calculated_df)
            logger.info(f"笔计算完成：上升笔 {calculated_df['up_bi'].sum()} 个，下降笔 {calculated_df['down_bi'].sum()} 个")
            
            # 3. 计算线段
            calculated_df = self.calculator.calculate_xd(calculated_df)
            logger.info(f"线段计算完成：上升线段 {calculated_df['up_xd'].sum()} 个，下降线段 {calculated_df['down_xd'].sum()} 个")
            
            # 4. 计算中枢
            calculated_df = self.calculator.calculate_zs(calculated_df)
            logger.info(f"中枢计算完成：共 {calculated_df['has_zs'].sum()} 个中枢")
            
            # 5. 计算买卖点
            calculated_df = self.calculator.calculate_signals(calculated_df)
            logger.info(f"买卖点计算完成：买入信号 {calculated_df['buy_signal'].sum()} 个，卖出信号 {calculated_df['sell_signal'].sum()} 个")
            
            # 6. 标记信号来源
            calculated_df['signal_source'] = calculated_df.apply(self._determine_signal_source, axis=1)
            
            logger.info("缠论计算全部完成")
            return {'success': True, 'data': calculated_df}
            
        except Exception as e:
            logger.error(f"缠论计算异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'缠论计算失败: {str(e)}'}
    
    def _determine_signal_source(self, row: pd.Series) -> str:
        """确定买卖信号来源"""
        if row.get('buy_signal'):
            if row.get('first_buy'):
                return '一类买点'
            elif row.get('second_buy'):
                return '二类买点'
            elif row.get('third_buy'):
                return '三类买点'
            elif row.get('bottom_fractal'):
                return '底分型买点'
            elif row.get('up_bi_start'):
                return '笔启动买点'
            else:
                return '其他买点'
        elif row.get('sell_signal'):
            if row.get('first_sell'):
                return '一类卖点'
            elif row.get('second_sell'):
                return '二类卖点'
            elif row.get('third_sell'):
                return '三类卖点'
            elif row.get('top_fractal'):
                return '顶分型卖点'
            elif row.get('down_bi_start'):
                return '笔启动卖点'
            else:
                return '其他卖点'
        else:
            return ''
    
    def _execute_backtest_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """执行回测核心逻辑（交易、止损止盈等）"""
        logger.info("执行回测逻辑...")
        
        try:
            capital = self.backtest_params.initial_capital
            position = 0
            trade_records = []
            daily_equity = []
            last_buy_date = None
            
            # 遍历数据执行交易
            for index, row in df.iterrows():
                current_date = row['date']
                close_price = row['close']
                
                # 记录每日资产
                current_equity = capital + (position * close_price if position > 0 else 0)
                daily_equity.append({
                    'date': current_date.strftime(DATE_FORMAT),
                    'capital': capital,
                    'position': position,
                    'position_value': position * close_price if position > 0 else 0,
                    'total_equity': current_equity,
                    'return_percent': (current_equity - self.backtest_params.initial_capital) / self.backtest_params.initial_capital * 100
                })
                
                # 买入逻辑
                if row.get('buy_signal') and position == 0:
                    if last_buy_date and (current_date - last_buy_date).days < self.backtest_params.min_holding_days:
                        continue
                    
                    buy_price = close_price * (1 + self.backtest_params.slippage)
                    max_shares = int(capital / (buy_price * (1 + self.backtest_params.transaction_cost)) / 100) * 100
                    if max_shares <= 0:
                        continue
                    
                    trade_cost = max_shares * buy_price * self.backtest_params.transaction_cost
                    total_cost = max_shares * buy_price + trade_cost
                    
                    capital -= total_cost
                    position = max_shares
                    last_buy_date = current_date
                    
                    trade_records.append(TradeRecord(
                        date=current_date,
                        type='buy',
                        price=buy_price,
                        shares=max_shares,
                        cost=total_cost,
                        capital=capital,
                        position=position,
                        signal_source=row.get('signal_source', '未知'),
                        notes=f"缠论{row['signal_source']}买点，滑点{self.backtest_params.slippage:.3f}，成本{trade_cost:.2f}元"
                    ))
                    logger.info(f"买入: {current_date.strftime(DATE_FORMAT)} | 价格: {buy_price:.2f} | 数量: {max_shares} | 剩余资金: {capital:.2f}")
                
                # 卖出逻辑
                elif row.get('sell_signal') and position > 0:
                    holding_days = (current_date - last_buy_date).days if last_buy_date else 0
                    if holding_days < self.backtest_params.min_holding_days:
                        continue
                    
                    sell_price = close_price * (1 - self.backtest_params.slippage)
                    trade_revenue = position * sell_price
                    trade_cost = trade_revenue * self.backtest_params.transaction_cost
                    net_revenue = trade_revenue - trade_cost
                    profit = net_revenue - sum([t.cost for t in trade_records if t.type == 'buy'])
                    
                    capital += net_revenue
                    logger.info(f"卖出: {current_date.strftime(DATE_FORMAT)} | 价格: {sell_price:.2f} | 利润: {profit:.2f} | 持有天数: {holding_days}")
                    
                    trade_records.append(TradeRecord(
                        date=current_date,
                        type='sell',
                        price=sell_price,
                        shares=position,
                        cost=trade_cost,
                        capital=capital,
                        position=0,
                        profit=profit,
                        holding_days=holding_days,
                        signal_source=row.get('signal_source', '未知'),
                        notes=f"缠论{row['signal_source']}卖点，滑点{self.backtest_params.slippage:.3f}，成本{trade_cost:.2f}元"
                    ))
                    position = 0
                    last_buy_date = None
                
                # 止损逻辑
                elif position > 0:
                    avg_buy_cost = sum([t.cost for t in trade_records if t.type == 'buy'])
                    current_position_value = position * close_price
                    floating_loss_ratio = (avg_buy_cost - current_position_value) / avg_buy_cost if avg_buy_cost > 0 else 0
                    
                    if floating_loss_ratio >= self.backtest_params.stop_loss_ratio:
                        stop_loss_price = close_price * (1 - self.backtest_params.slippage * 1.5)
                        trade_revenue = position * stop_loss_price
                        trade_cost = trade_revenue * self.backtest_params.transaction_cost
                        net_revenue = trade_revenue - trade_cost
                        profit = net_revenue - avg_buy_cost
                        holding_days = (current_date - last_buy_date).days if last_buy_date else 0
                        
                        capital += net_revenue
                        logger.warning(f"止损卖出: {current_date.strftime(DATE_FORMAT)} | 价格: {stop_loss_price:.2f} | 亏损: {profit:.2f} | 比例: {floating_loss_ratio:.2%}")
                        
                        trade_records.append(TradeRecord(
                            date=current_date,
                            type='sell',
                            price=stop_loss_price,
                            shares=position,
                            cost=trade_cost,
                            capital=capital,
                            position=0,
                            profit=profit,
                            holding_days=holding_days,
                            signal_source='止损',
                            notes=f"止损（比例{self.backtest_params.stop_loss_ratio:.2%}），滑点{self.backtest_params.slippage*1.5:.3f}"
                        ))
                        position = 0
                        last_buy_date = None
                
                # 止盈逻辑
                elif position > 0:
                    avg_buy_cost = sum([t.cost for t in trade_records if t.type == 'buy'])
                    current_position_value = position * close_price
                    floating_profit_ratio = (current_position_value - avg_buy_cost) / avg_buy_cost if avg_buy_cost > 0 else 0
                    
                    if floating_profit_ratio >= self.backtest_params.take_profit_ratio:
                        take_profit_price = close_price * (1 - self.backtest_params.slippage)
                        trade_revenue = position * take_profit_price
                        trade_cost = trade_revenue * self.backtest_params.transaction_cost
                        net_revenue = trade_revenue - trade_cost
                        profit = net_revenue - avg_buy_cost
                        holding_days = (current_date - last_buy_date).days if last_buy_date else 0
                        
                        capital += net_revenue
                        logger.info(f"止盈卖出: {current_date.strftime(DATE_FORMAT)} | 价格: {take_profit_price:.2f} | 盈利: {profit:.2f} | 比例: {floating_profit_ratio:.2%}")
                        
                        trade_records.append(TradeRecord(
                            date=current_date,
                            type='sell',
                            price=take_profit_price,
                            shares=position,
                            cost=trade_cost,
                            capital=capital,
                            position=0,
                            profit=profit,
                            holding_days=holding_days,
                            signal_source='止盈',
                            notes=f"止盈（比例{self.backtest_params.take_profit_ratio:.2%}），滑点{self.backtest_params.slippage:.3f}"
                        ))
                        position = 0
                        last_buy_date = None
                
                # 最大持有天数限制
                elif position > 0 and last_buy_date:
                    holding_days = (current_date - last_buy_date).days
                    if holding_days >= self.backtest_params.max_holding_days:
                        sell_price = close_price * (1 - self.backtest_params.slippage)
                        trade_revenue = position * sell_price
                        trade_cost = trade_revenue * self.backtest_params.transaction_cost
                        net_revenue = trade_revenue - trade_cost
                        avg_buy_cost = sum([t.cost for t in trade_records if t.type == 'buy'])
                        profit = net_revenue - avg_buy_cost
                        
                        capital += net_revenue
                        logger.info(f"强制卖出: {current_date.strftime(DATE_FORMAT)} | 持有超{self.backtest_params.max_holding_days}天 | 利润: {profit:.2f}")
                        
                        trade_records.append(TradeRecord(
                            date=current_date,
                            type='sell',
                            price=sell_price,
                            shares=position,
                            cost=trade_cost,
                            capital=capital,
                            position=0,
                            profit=profit,
                            holding_days=holding_days,
                            signal_source='强制卖出',
                            notes=f"持有超{self.backtest_params.max_holding_days}天强制平仓"
                        ))
                        position = 0
                        last_buy_date = None
            
            # 回测结束强制平仓
            if position > 0:
                current_date = df.iloc[-1]['date']
                close_price = df.iloc[-1]['close']
                sell_price = close_price * (1 - self.backtest_params.slippage)
                trade_revenue = position * sell_price
                trade_cost = trade_revenue * self.backtest_params.transaction_cost
                net_revenue = trade_revenue - trade_cost
                avg_buy_cost = sum([t.cost for t in trade_records if t.type == 'buy'])
                profit = net_revenue - avg_buy_cost
                holding_days = (current_date - last_buy_date).days if last_buy_date else 0
                
                capital += net_revenue
                logger.info(f"强制平仓: {current_date.strftime(DATE_FORMAT)} | 价格: {sell_price:.2f} | 利润: {profit:.2f}")
                trade_records.append(TradeRecord(
                    date=current_date,
                    type='sell',
                    price=sell_price,
                    shares=position,
                    cost=trade_cost,
                    capital=capital,
                    position=0,
                    profit=profit,
                    holding_days=holding_days,
                    signal_source='回测结束平仓',
                    notes="回测结束强制平仓"
                ))
            
            # 计算性能指标
            final_capital = capital
            total_return = final_capital - self.backtest_params.initial_capital
            return_percent = (total_return / self.backtest_params.initial_capital) * 100
            
            # 年化回报
            start_dt = datetime.strptime(self.backtest_params.start_date, DATE_FORMAT)
            end_dt = datetime.strptime(self.backtest_params.end_date, DATE_FORMAT)
            years = (end_dt - start_dt).days / 365.25
            annual_return = (pow((final_capital / self.backtest_params.initial_capital), 1/years) - 1) * 100 if years > 0 else 0
            
            # 风险指标
            max_drawdown = calculate_max_drawdown([item['total_equity'] for item in daily_equity])
            sharpe_ratio = calculate_sharpe_ratio([item['total_equity'] for item in daily_equity], risk_free_rate=0.03)
            sortino_ratio = calculate_sortino_ratio([item['total_equity'] for item in daily_equity], risk_free_rate=0.03)
            
            # 交易统计
            buy_count = sum(1 for t in trade_records if t.type == 'buy')
            sell_count = sum(1 for t in trade_records if t.type == 'sell')
            winning_trades = sum(1 for t in trade_records if t.type == 'sell' and t.profit > 0)
            losing_trades = sum(1 for t in trade_records if t.type == 'sell' and t.profit <= 0)
            win_rate = (winning_trades / sell_count) * 100 if sell_count > 0 else 0
            
            total_profit = sum(t.profit for t in trade_records if t.type == 'sell' and t.profit > 0)
            total_loss = sum(abs(t.profit) for t in trade_records if t.type == 'sell' and t.profit <= 0)
            average_profit = total_profit / winning_trades if winning_trades > 0 else 0
            average_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            logger.info("回测逻辑执行完成")
            return {
                'success': True,
                'final_capital': final_capital,
                'total_return': total_return,
                'return_percent': return_percent,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'trade_count': len(trade_records),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'average_profit': average_profit,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'trade_records': trade_records,
                'daily_equity': daily_equity
            }
            
        except Exception as e:
            logger.error(f"回测执行异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'回测执行失败: {str(e)}'}
    
    def _generate_comprehensive_reports(self, result: BacktestResult) -> Dict[str, Any]:
        """生成完整回测报告（文本、Excel、JSON）"""
        logger.info("生成回测报告...")
        
        try:
            symbol = result.params.symbol
            timeframe = result.params.timeframe
            report_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"chanlun_backtest_{symbol}_{timeframe}_{report_date}"
            
            # 文本报告
            text_report_path = os.path.join(REPORT_DIR, f"{report_filename}.txt")
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_text_report(result))
            
            # Excel报告
            excel_report_path = os.path.join(REPORT_DIR, f"{report_filename}.xlsx")
            self._generate_excel_report(result, excel_report_path)
            
            # JSON报告
            json_report_path = os.path.join(REPORT_DIR, f"{report_filename}.json")
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(self._generate_json_report(result), f, ensure_ascii=False, indent=2)
            
            # 盘后总结
            if self.config.get('reports', {}).get('generate_daily_summary', True):
                daily_summary_path = os.path.join(REPORT_DIR, f"daily_summary_{datetime.now().strftime('%Y%m%d')}.txt")
                generate_daily_report([result], daily_summary_path)
            
            return {
                'success': True,
                'report_path': REPORT_DIR,
                'text_report': text_report_path,
                'excel_report': excel_report_path,
                'json_report': json_report_path,
                'report_filename': report_filename
            }
            
        except Exception as e:
            logger.error(f"报告生成异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'报告生成失败: {str(e)}'}
    
    def _generate_text_report(self, result: BacktestResult) -> str:
        """生成文本报告"""
        trade_records_str = "\n".join([
            f"{t.date.strftime(DATE_FORMAT)} | {t.type.upper()} | 价格: {t.price:.2f} | 数量: {t.shares} | 成本: {t.cost:.2f} | 利润: {t.profit:.2f} | 持有天数: {t.holding_days} | 信号: {t.signal_source}"
            for t in result.trade_records
        ])
        
        return f"""
==============================================================
                    缠论回测报告
==============================================================
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
股票代码: {result.params.symbol} | 时间级别: {result.params.timeframe}
回测日期范围: {result.actual_date_range['requested_start']} ~ {result.actual_date_range['requested_end']}
实际数据范围: {result.actual_date_range['actual_start']} ~ {result.actual_date_range['actual_end']}
数据记录数: {result.actual_date_range['data_count']} 条
==============================================================
【回测参数】
初始资金: {result.params.initial_capital:.2f} 元 | 交易成本: {result.params.transaction_cost:.4f}
滑点率: {result.params.slippage:.4f} | 止损比例: {result.params.stop_loss_ratio:.2%} | 止盈比例: {result.params.take_profit_ratio:.2%}
最小持有天数: {result.params.min_holding_days} 天 | 最大持有天数: {result.params.max_holding_days} 天
==============================================================
【回测结果】
最终资金: {result.final_capital:.2f} 元 | 总回报: {result.total_return:.2f} 元 ({result.return_percent:.2f}%)
年化回报: {result.annual_return:.2f}% | 最大回撤: {result.max_drawdown:.2f}%
夏普比率: {result.sharpe_ratio:.2f} | 索提诺比率: {result.sortino_ratio:.2f}
交易次数: {result.trade_count} | 买入: {result.buy_count} | 卖出: {result.sell_count}
盈利交易: {result.winning_trades} | 亏损交易: {result.losing_trades} | 胜率: {result.win_rate:.2f}%
平均盈利: {result.average_profit:.2f} 元 | 平均亏损: {result.average_loss:.2f} 元 | 盈利因子: {result.profit_factor:.2f}
==============================================================
【交易记录】
{trade_records_str if trade_records_str else '无交易记录'}
==============================================================
【风险提示】
1. 本回测基于历史数据，不构成投资建议
2. 实际交易存在不可预见风险，参数为假设值
3. 回测结果受数据质量和参数设置影响，谨慎参考
==============================================================
""".strip()
    
    def _generate_excel_report(self, result: BacktestResult, output_path: str):
        """生成Excel报告"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 回测参数表
            pd.DataFrame({
                '参数名称': [
                    '股票代码', '时间级别', '初始资金', '交易成本', '滑点率', '止损比例', '止盈比例',
                    '最小持有天数', '最大持有天数', '回测开始日期', '回测结束日期', '实际开始日期', '实际结束日期', '数据记录数'
                ],
                '参数值': [
                    result.params.symbol, result.params.timeframe, f"{result.params.initial_capital:.2f} 元",
                    f"{result.params.transaction_cost:.4f}", f"{result.params.slippage:.4f}",
                    f"{result.params.stop_loss_ratio:.2%}", f"{result.params.take_profit_ratio:.2%}",
                    f"{result.params.min_holding_days} 天", f"{result.params.max_holding_days} 天",
                    result.actual_date_range['requested_start'], result.actual_date_range['requested_end'],
                    result.actual_date_range['actual_start'], result.actual_date_range['actual_end'],
                    result.actual_date_range['data_count']
                ]
            }).to_excel(writer, sheet_name='回测参数', index=False)
            
            # 结果摘要表
            pd.DataFrame({
                '指标名称': [
                    '最终资金', '总回报', '总回报比例', '年化回报', '最大回撤', '夏普比率', '索提诺比率',
                    '交易总次数', '买入次数', '卖出次数', '盈利交易', '亏损交易', '胜率', '平均盈利', '平均亏损', '盈利因子'
                ],
                '指标值': [
                    f"{result.final_capital:.2f} 元", f"{result.total_return:.2f} 元", f"{result.return_percent:.2f}%",
                    f"{result.annual_return:.2f}%", f"{result.max_drawdown:.2f}%", f"{result.sharpe_ratio:.2f}",
                    f"{result.sortino_ratio:.2f}", result.trade_count, result.buy_count, result.sell_count,
                    result.winning_trades, result.losing_trades, f"{result.win_rate:.2f}%",
                    f"{result.average_profit:.2f} 元", f"{result.average_loss:.2f} 元", f"{result.profit_factor:.2f}"
                ]
            }).to_excel(writer, sheet_name='结果摘要', index=False)
            
            # 交易记录表
            pd.DataFrame([{
                '交易日期': t.date.strftime(DATE_FORMAT),
                '交易类型': t.type,
                '价格': t.price,
                '数量': t.shares,
                '成本': t.cost,
                '剩余资金': t.capital,
                '持仓': t.position,
                '利润': t.profit,
                '持有天数': t.holding_days,
                '信号来源': t.signal_source,
                '备注': t.notes
            } for t in result.trade_records]).to_excel(writer, sheet_name='交易记录', index=False)
            
            # 每日资产表
            pd.DataFrame(result.daily_equity).to_excel(writer, sheet_name='每日资产', index=False)
    
    def _generate_json_report(self, result: BacktestResult) -> Dict[str, Any]:
        """生成JSON报告"""
        return {
            'report_info': {
                'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0.0',
                'engine': 'ChanlunBacktestEngine'
            },
            'backtest_params': {
                'symbol': result.params.symbol,
                'timeframe': result.params.timeframe,
                'initial_capital': result.params.initial_capital,
                'transaction_cost': result.params.transaction_cost,
                'slippage': result.params.slippage,
                'stop_loss_ratio': result.params.stop_loss_ratio,
                'take_profit_ratio': result.params.take_profit_ratio,
                'min_holding_days': result.params.min_holding_days,
                'max_holding_days': result.params.max_holding_days
            },
            'date_range': result.actual_date_range,
            'performance_metrics': {
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'return_percent': result.return_percent,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'trade_count': result.trade_count,
                'buy_count': result.buy_count,
                'sell_count': result.sell_count,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'average_profit': result.average_profit,
                'average_loss': result.average_loss,
                'profit_factor': result.profit_factor
            },
            'trade_records': [
                {
                    'date': t.date.strftime(DATE_FORMAT),
                    'type': t.type,
                    'price': t.price,
                    'shares': t.shares,
                    'cost': t.cost,
                    'capital': t.capital,
                    'position': t.position,
                    'profit': t.profit,
                    'holding_days': t.holding_days,
                    'signal_source': t.signal_source,
                    'notes': t.notes
                } for t in result.trade_records
            ],
            'daily_equity': result.daily_equity
        }
    
    def _generate_detailed_charts(self, result: BacktestResult, df: pd.DataFrame) -> Dict[str, Any]:
        """生成详细图表（资产曲线、交易信号、缠论结构）"""
        logger.info("生成回测图表...")
        
        try:
            symbol = result.params.symbol
            timeframe = result.params.timeframe
            chart_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 资产曲线
            asset_curve_path = os.path.join(CHART_DIR, f"{symbol}_{timeframe}_asset_curve_{chart_date}.png")
            self.plotter.plot_asset_curve(
                daily_equity=result.daily_equity,
                trade_records=result.trade_records,
                title=f"{symbol} {timeframe}级别 资产曲线",
                output_path=asset_curve_path
            )
            
            # 交易信号图表
            signal_chart_path = os.path.join(CHART_DIR, f"{symbol}_{timeframe}_trade_signals_{chart_date}.png")
            self.plotter.plot_trade_signals(
                df=df,
                trade_records=result.trade_records,
                title=f"{symbol} {timeframe}级别 价格走势与交易信号",
                output_path=signal_chart_path
            )
            
            # 缠论结构图表
            chanlun_chart_path = os.path.join(CHART_DIR, f"{symbol}_{timeframe}_chanlun_structure_{chart_date}.png")
            self.plotter.plot_chanlun_structure(
                df=df,
                title=f"{symbol} {timeframe}级别 缠论结构（分型+笔+线段+中枢）",
                output_path=chanlun_chart_path
            )
            
            # 性能雷达图
            radar_chart_path = None
            if self.config.get('plotter', {}).get('enable_radar_chart', True):
                radar_chart_path = os.path.join(CHART_DIR, f"{symbol}_{timeframe}_performance_radar_{chart_date}.png")
                self.plotter.plot_performance_radar(
                    metrics={
                        '年化回报': result.annual_return,
                        '最大回撤': abs(result.max_drawdown),
                        '夏普比率': result.sharpe_ratio,
                        '胜率': result.win_rate,
                        '盈利因子': result.profit_factor
                    },
                    title=f"{symbol} {timeframe}级别 性能指标雷达图",
                    output_path=radar_chart_path
                )
            
            return {
                'success': True,
                'chart_dir': CHART_DIR,
                'asset_curve': asset_curve_path,
                'trade_signals': signal_chart_path,
                'chanlun_structure': chanlun_chart_path,
                'radar_chart': radar_chart_path
            }
            
        except Exception as e:
            logger.error(f"图表生成异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'图表生成失败: {str(e)}'}
    
    def _export_backtest_results(self, result: BacktestResult) -> Dict[str, Any]:
        """导出回测结果（CSV、Excel、JSON）"""
        logger.info("导出回测结果...")
        
        try:
            symbol = result.params.symbol
            timeframe = result.params.timeframe
            export_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_filename = f"chanlun_backtest_{symbol}_{timeframe}_{export_date}"
            
            # 交易记录导出
            trades_df = pd.DataFrame([{
                '交易日期': t.date.strftime(DATE_FORMAT),
                '交易类型': t.type,
                '价格': t.price,
                '数量': t.shares,
                '成本': t.cost,
                '剩余资金': t.capital,
                '持仓': t.position,
                '利润': t.profit,
                '持有天数': t.holding_days,
                '信号来源': t.signal_source,
                '备注': t.notes
            } for t in result.trade_records])
            
            # 每日资产导出
            equity_df = pd.DataFrame(result.daily_equity)
            
            # 按配置格式导出
            export_formats = self.config.get('exporter', {}).get('format', ['csv', 'excel', 'json'])
            export_dir = EXPORT_DIR
            
            for fmt in export_formats:
                if fmt == 'csv':
                    trades_df.to_csv(os.path.join(export_dir, f"{export_filename}_trades.csv"), index=False, encoding='utf-8')
                    equity_df.to_csv(os.path.join(export_dir, f"{export_filename}_equity.csv"), index=False, encoding='utf-8')
                elif fmt == 'excel':
                    with pd.ExcelWriter(os.path.join(export_dir, f"{export_filename}.xlsx"), engine='openpyxl') as writer:
                        trades_df.to_excel(writer, sheet_name='交易记录', index=False)
                        equity_df.to_excel(writer, sheet_name='每日资产', index=False)
                elif fmt == 'json':
                    with open(os.path.join(export_dir, f"{export_filename}_trades.json"), 'w', encoding='utf-8') as f:
                        json.dump(trades_df.to_dict('records'), f, ensure_ascii=False, indent=2)
                    with open(os.path.join(export_dir, f"{export_filename}_equity.json"), 'w', encoding='utf-8') as f:
                        json.dump(equity_df.to_dict('records'), f, ensure_ascii=False, indent=2)
            
            return {
                'success': True,
                'export_dir': export_dir,
                'export_filename': export_filename,
                'formats': export_formats
            }
            
        except Exception as e:
            logger.error(f"导出异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'导出失败: {str(e)}'}
    
    def _send_error_notification(self, error_msg: str):
        """发送错误通知（仅钉钉，移除邮件）"""
        if self.notifier:
            try:
                self.notifier.send_message(f"""
【缠论回测错误通知】
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
错误信息: {error_msg}
请及时检查配置或数据！
                """.strip())
                logger.info("错误通知已通过钉钉发送")
            except Exception as e:
                logger.error(f"钉钉通知发送失败: {str(e)}")
    
    def _send_backtest_complete_notification(self, result: BacktestResult):
        """发送回测完成通知（仅钉钉，移除邮件）"""
        if self.notifier:
            try:
                self.notifier.send_message(f"""
【缠论回测完成通知】
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
股票代码: {result.params.symbol}
时间级别: {result.params.timeframe}
回测日期: {result.actual_date_range['actual_start']} ~ {result.actual_date_range['actual_end']}
初始资金: {result.initial_capital:.2f} 元
最终资金: {result.final_capital:.2f} 元
总回报: {result.total_return:.2f} 元 ({result.return_percent:.2f}%)
年化回报: {result.annual_return:.2f}%
最大回撤: {result.max_drawdown:.2f}%
交易次数: {result.trade_count} 次 | 胜率: {result.win_rate:.2f}%
详细报告: {result.report['report_path']}
                """.strip())
                logger.info("回测完成通知已通过钉钉发送")
            except Exception as e:
                logger.error(f"钉钉通知发送失败: {str(e)}")

# 命令行执行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='缠论回测系统命令行工具')
    parser.add_argument('--symbol', required=True, help='股票代码（如：600036）')
    parser.add_argument('--start-date', required=True, help='开始日期（YYYY-MM-DD 或 YYYYMMDD）')
    parser.add_argument('--end-date', required=True, help='结束日期（YYYY-MM-DD 或 YYYYMMDD）')
    parser.add_argument('--timeframe', default='daily', choices=VALID_TIMEFRAMES, help='时间级别')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--transaction-cost', type=float, default=0.0005, help='交易成本率')
    parser.add_argument('--slippage', type=float, default=0.001, help='滑点率')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 初始化回测引擎
    engine = BacktestEngine(config)
    
    # 运行回测
    result = engine.run_comprehensive_backtest(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage
    )
    
    # 输出结果
    if result.success:
        sys.exit(0)
    else:
        logger.error(f"回测失败: {result.error}")
        sys.exit(1)