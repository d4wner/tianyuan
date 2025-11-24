#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论回测系统 - 完整功能版
包含所有核心回测逻辑、交易执行、绩效分析、报告生成等功能
原始功能完整保留，未删减关键模块
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

# 导入依赖模块
try:
    from src.config import (
        load_config, get_backtest_config, get_chanlun_config,
        get_risk_management_config, get_data_paths, ensure_data_directories
    )
    from src.data_fetcher import StockDataFetcher
    from src.calculator import ChanlunCalculator
    from src.notifier import DingdingNotifier
    from src.utils import (
        get_last_trading_day, 
        is_trading_hour, 
        get_valid_date_range_str,
        validate_date_format,
        convert_date_format,
        calculate_max_drawdown,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        format_number,
        get_timeframe_multiplier,
        merge_dataframes,
        parse_date,
        format_date,
        #1123
        DATE_FORMAT,
        DATE_FORMAT_ALT,
        TIME_FORMAT,
        DATETIME_FORMAT,
    )
    from src.reporter import (
        generate_pre_market_report, 
        generate_daily_report, 
        generate_backtest_report,
        generate_multiple_backtest_report
    )
    from src.exporter import ChanlunExporter
    from src.plotter import ChanlunPlotter
except ImportError as e:
    logging.error(f"导入依赖模块失败: {str(e)}")
    raise

# 配置日志
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

# 全局常量定义
VALID_TIMEFRAMES = ['daily', 'weekly', 'monthly', 'minute', '60m', '30m', '15m', '5m', '1m']
MIN_DATA_POINTS = 30  # 最小数据点数量
MAX_RETRY_COUNT = 5
#DATE_FORMAT = '%Y-%m-%d'
#DATE_FORMAT_ALT = '%Y%m%d'
#TIME_FORMAT = '%H:%M:%S'
#DATETIME_FORMAT = f'{DATE_FORMAT} {TIME_FORMAT}'

# 从配置获取路径（修复：使用配置管理的路径）
data_paths = get_data_paths()
REPORT_DIR = data_paths['reports']
CHART_DIR = data_paths['plots']
EXPORT_DIR = data_paths['exports']
CACHE_DIR = os.path.join(project_root, 'cache')

# 确保所有数据目录存在
ensure_data_directories()

# 数据类定义
@dataclass
class BacktestParams:
    """回测参数数据类"""
    symbol: str
    start_date: str
    end_date: str
    timeframe: str = 'daily'
    # 初始资金会从config/system.yaml配置文件中读取，这里的默认值仅在配置未指定时使用
    initial_capital: float = 100000.0
    transaction_cost: float = 0.0005  # 交易成本比例
    slippage: float = 0.001  # 滑点比例
    stop_loss_ratio: float = 0.05  # 止损比例
    take_profit_ratio: float = 0.1  # 止盈比例
    min_holding_days: int = 1  # 最小持有天数
    max_holding_days: int = 90  # 最大持有天数
    signal_strength_threshold: float = 0.1  # 信号强度阈值，与calculator.py保持一致
    enable_short: bool = False  # 是否允许做空
    capital_allocation: float = 1.0  # 资金分配比例
    trailing_stop: bool = False  # 是否启用移动止损

    def __post_init__(self):
        """参数验证与格式化"""
        # 验证股票代码格式
        if not isinstance(self.symbol, str) or len(self.symbol) < 6:
            raise ValueError(f"无效的股票代码: {self.symbol}")
        
        # 验证日期格式并统一
        if not validate_date_format(self.start_date):
            raise ValueError(f"无效的开始日期格式: {self.start_date}, 应为{DATE_FORMAT}或{DATE_FORMAT_ALT}")
        if not validate_date_format(self.end_date):
            raise ValueError(f"无效的结束日期格式: {self.end_date}, 应为{DATE_FORMAT}或{DATE_FORMAT_ALT}")
            
        # 转换为标准日期格式（修复：确保转换后是字符串类型，兼容报告生成）
        self.start_date = convert_date_format(self.start_date, input_format='auto', output_format=DATE_FORMAT)
        self.end_date = convert_date_format(self.end_date, input_format='auto', output_format=DATE_FORMAT)
        
        # 验证时间级别
        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"无效的时间级别: {self.timeframe}, 可选值: {VALID_TIMEFRAMES}")
        
        # 验证数值参数
        if self.initial_capital <= 0:
            raise ValueError("初始资金必须大于0")
        if self.transaction_cost < 0 or self.transaction_cost > 0.1:
            raise ValueError("交易成本比例必须在0到0.1之间")
        if self.stop_loss_ratio <= 0 or self.stop_loss_ratio > 0.5:
            raise ValueError("止损比例必须在0到0.5之间")


@dataclass
class TradeRecord:
    """交易记录数据类"""
    date: datetime
    type: str  # 'buy' 或 'sell', 'short', 'cover'
    price: float
    shares: int
    cost: float  # 交易成本
    capital: float  # 交易后资金
    position: int  # 持仓数量
    profit: float = 0.0  # 利润
    holding_days: int = 0  # 持有天数
    signal_source: str = ''  # 信号来源
    notes: str = ''  # 备注
    trade_id: str = ''  # 交易ID
    stop_loss_price: Optional[float] = None  # 止损价格
    take_profit_price: Optional[float] = None  # 止盈价格

    def __post_init__(self):
        """确保日期是datetime类型（修复：兼容更多日期格式）"""
        if isinstance(self.date, str):
            try:
                self.date = datetime.strptime(self.date, DATETIME_FORMAT)
            except ValueError:
                try:
                    self.date = datetime.strptime(self.date, DATE_FORMAT)
                except ValueError:
                    self.date = parse_date(self.date)  # 使用工具类解析日期


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
    short_count: int = 0
    cover_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_holding_period: float = 0.0
    trade_records: List[TradeRecord] = field(default_factory=list)
    daily_equity: List[Dict[str, Any]] = field(default_factory=list)
    actual_date_range: Dict[str, str] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    charts: Dict[str, Any] = field(default_factory=dict)
    signal_analysis: Dict[str, Any] = field(default_factory=dict)  # 信号分析
    benchmark_return: float = 0.0  # 基准收益


class BacktestEngine:
    """缠论回测引擎核心类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化回测引擎"""
        self.config = config or load_config('config/system.yaml')
        self.data_api = None
        self.calculator = None
        self.notifier = None
        self.plotter = None
        self.exporter = None
        self.performance_metrics = {}
        self.trade_records = []
        self.historical_data = {}  # 存储不同时间级别的历史数据
        self.daily_equity = []
        self.backtest_params = None
        self.position = 0  # 当前持仓
        self.current_capital = 0.0  # 当前资金
        self.current_date = None  # 当前日期
        self.benchmark_data = None  # 基准数据
        self.signal_data = {}  # 信号数据
        
        # 初始化组件
        self._initialize_components()
        self._create_output_directories()
        logger.info("缠论回测引擎初始化完成")
    
    def _initialize_components(self):
        """初始化所有组件"""
        logger.info("初始化回测组件...")
        
        # 初始化数据API
        try:
            data_fetcher_config = self.config.get('data_fetcher', {})
            self.data_api = StockDataFetcher(
                max_retries=data_fetcher_config.get('max_retries', MAX_RETRY_COUNT),
                timeout=data_fetcher_config.get('timeout', 10),
            )
            logger.info("数据API初始化成功")
        except Exception as e:
            logger.error(f"数据API初始化失败: {str(e)}")
            raise
        
        # 初始化缠论计算器（修复：传入初始资金参数）
        try:
            chanlun_config = get_chanlun_config()
            calculator_config = {
                'chanlun': chanlun_config,
                'risk_management': get_risk_management_config()
            }
            # 优先从配置文件读取初始资金，然后再考虑backtest_params中的值
            config_initial_capital = self.config.get('system', {}).get('backtest', {}).get('initial_capital', 100000.0)
            initial_capital = self.backtest_params.initial_capital if self.backtest_params else config_initial_capital
            self.calculator = ChanlunCalculator(
                calculator_config,
                initial_capital=initial_capital
            )
            logger.info("缠论计算器初始化成功")
        except Exception as e:
            logger.error(f"缠论计算器初始化失败: {str(e)}")
            raise
        
        # 初始化钉钉通知器
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
        
        # 初始化绘图器（修复：确保配置传递正确，兼容plotter.py）
        try:
            plotter_config = self.config.get('plotter', {})
            # 强制设置输出目录为CHART_DIR，确保图表保存路径统一
            plotter_config['output_dir'] = CHART_DIR
            self.plotter = ChanlunPlotter(
                config=plotter_config
            )
            logger.info("绘图器初始化成功")
        except Exception as e:
            logger.error(f"绘图器初始化失败: {str(e)}")
            raise
        
        # 初始化数据导出器
        try:
            exporter_config = self.config.get('exporter', {})
            self.exporter = ChanlunExporter(
                config=exporter_config
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
            os.makedirs(CACHE_DIR, exist_ok=True)
            logger.info("输出目录创建/验证成功")
        except Exception as e:
            logger.error(f"创建输出目录失败: {str(e)}")
            raise
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                             start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史数据
        :param symbol: 股票代码
        :param timeframe: 时间级别
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 历史数据DataFrame
        """
        try:
            logger.info(f"获取 {symbol} {timeframe} 数据: {start_date} 至 {end_date}")
            
            # 声明实际日期变量，默认使用传入的日期
            actual_start = start_date
            actual_end = end_date
            
            # 根据时间级别获取数据（修改部分：接收实际日期范围）
            if timeframe == 'daily':
                # 改为接收三元组返回值
                data, actual_start, actual_end = self.data_api.get_daily_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            elif timeframe == 'weekly':
                # 改为接收三元组返回值
                data, actual_start, actual_end = self.data_api.get_weekly_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            elif timeframe == 'monthly':
                # 改为接收三元组返回值（假设get_monthly_data已修改）
                data, actual_start, actual_end = self.data_api.get_monthly_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            elif timeframe in ['minute', '60m', '30m', '15m', '5m', '1m']:
                # 改为接收三元组返回值（假设get_minute_data已修改）
                data, actual_start, actual_end = self.data_api.get_minute_data(
                    symbol=symbol,
                    frequency=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                raise ValueError(f"不支持的时间级别: {timeframe}")
            
            # 核心修改：更新回测参数中的实际日期范围
            if hasattr(self, 'backtest_params'):
                self.backtest_params.start_date = actual_start
                self.backtest_params.end_date = actual_end
                logger.info(f"回测日期范围已调整为实际数据范围: {actual_start} 至 {actual_end}")
            
            # 以下为你原有的逻辑，保持不变
            # 数据验证
            if data.empty:
                raise ValueError(f"未获取到 {symbol} 的数据")
            
            if len(data) < MIN_DATA_POINTS:
                logger.warning(f"数据点数量不足 {MIN_DATA_POINTS}, 实际: {len(data)}")
            
            # 确保日期列格式正确
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce').dropna()
                data = data.sort_values('date').reset_index(drop=True)
            
            logger.info(f"成功获取 {symbol} 数据，共 {len(data)} 条记录")
            return data
        
        except Exception as e:
            logger.error(f"获取历史数据失败: {str(e)}")
            raise
    
    def calculate_chanlun_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        计算缠论指标
        :param data: 原始数据
        :param timeframe: 时间级别
        :return: 包含缠论指标的数据
        """
        try:
            logger.info(f"计算缠论指标，时间级别: {timeframe}")
            
            # 调用缠论计算器
            result = self.calculator.calculate(df=data)
            result_df = pd.DataFrame(result['data'])
            
            # 验证计算结果
            if result_df.empty:
                raise ValueError("缠论指标计算结果为空")
                
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 
                              'top_fractal', 'bottom_fractal', 'pen_type', 'signal', 'signal_strength']
            for col in required_columns:
                if col not in result_df.columns:
                    raise ValueError(f"缠论计算结果缺少必要列: {col}")
            
            logger.info("缠论指标计算完成")
            return result_df
        
        except Exception as e:
            logger.error(f"计算缠论指标失败: {str(e)}")
            raise
    
    def generate_trading_signals(self, indicator_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        生成交易信号（修复：使用计算器生成的信号强度，解决信号传递失败问题）
        :param indicator_data: 包含缠论指标的数据
        :return: 交易信号列表
        """
        signals = []
        
        try:
            logger.info("开始生成交易信号")
            # 直接使用0-1区间阈值，与calculator.py保持一致
            threshold = self.backtest_params.signal_strength_threshold
            
            # 遍历每一行数据寻找交易信号
            for i in range(1, len(indicator_data)):
                current = indicator_data.iloc[i]
                prev = indicator_data.iloc[i-1]
                date = current['date']
                
                # 直接使用计算器生成的0-1区间信号强度
                signal_strength = current.get('signal_strength', 0)
                
                # 买入信号: 底分型形成且有买入信号
                if current['bottom_fractal'] and current['signal'] == 'buy':
                    signals.append({
                        'date': date,
                        'type': 'buy',
                        'price': current['close'],
                        'strength': signal_strength,
                        'reason': '底分型形成 + 买入信号'
                    })
                    logger.debug(f"生成买入信号: {date} 强度: {signal_strength}")
                
                # 卖出信号: 顶分型形成且有卖出信号
                if current['top_fractal'] and current['signal'] == 'sell':
                    signals.append({
                        'date': date,
                        'type': 'sell',
                        'price': current['close'],
                        'strength': signal_strength,
                        'reason': '顶分型形成 + 卖出信号'
                    })
                    logger.debug(f"生成卖出信号: {date} 强度: {signal_strength}")
                
                # 做空信号 (如果允许)
                if self.backtest_params.enable_short and current['top_fractal'] and current['signal'] == 'short':
                    signals.append({
                        'date': date,
                        'type': 'short',
                        'price': current['close'],
                        'strength': signal_strength,
                        'reason': '顶分型形成 + 做空信号'
                    })
                    logger.debug(f"生成做空信号: {date} 强度: {signal_strength}")
                
                # 平仓信号 (如果允许做空)
                if self.backtest_params.enable_short and current['bottom_fractal'] and current['signal'] == 'cover':
                    signals.append({
                        'date': date,
                        'type': 'cover',
                        'price': current['close'],
                        'strength': signal_strength,
                        'reason': '底分型形成 + 平仓信号'
                    })
                    logger.debug(f"生成平仓信号: {date} 强度: {signal_strength}")
            
            logger.info(f"共生成 {len(signals)} 个交易信号")
            return signals
        
        except Exception as e:
            logger.error(f"生成交易信号失败: {str(e)}")
            raise
    
    def execute_trades(self, signals: List[Dict[str, Any]], data: pd.DataFrame) -> List[TradeRecord]:
        """执行交易信号并生成交易记录"""
        trade_records = []
        self.position = 0
        self.current_capital = self.backtest_params.initial_capital
        last_trade_date = None
        
        try:
            logger.info(f"开始执行交易，接收到{len(signals)}个信号")
            
            for signal in signals:
                # 计算持仓天数限制
                if last_trade_date and self.position != 0:
                    days_diff = (signal['date'] - last_trade_date).days
                    # 放宽限制，只对连续买入/卖出做限制，不阻止正常的买入后卖出
                    if (signal['type'] == 'buy' and self.position > 0) or \
                       (signal['type'] == 'sell' and self.position <= 0) or \
                       (signal['type'] == 'short' and self.position < 0) or \
                       (signal['type'] == 'cover' and self.position >= 0):
                        if days_diff < self.backtest_params.min_holding_days:
                            logger.debug(f"未满足最小持有天数 {self.backtest_params.min_holding_days}，跳过信号")
                            continue
                
                # 计算可交易数量
                price = signal['price'] * (1 + self.backtest_params.slippage)  # 加入滑点
                max_shares = int((self.current_capital * self.backtest_params.capital_allocation) / price)
                if max_shares <= 0:
                    logger.debug("资金不足，无法交易")
                    continue
                
                logger.debug(f"处理信号: {signal['type']} {signal['date']} 价格: {price} 当前持仓: {self.position}")
                
                # 执行买入
                if signal['type'] == 'buy' and self.position <= 0:
                    shares = max_shares
                    cost = shares * price
                    transaction_cost = cost * self.backtest_params.transaction_cost
                    total_cost = cost + transaction_cost
                    
                    if total_cost > self.current_capital:
                        logger.debug("资金不足，调整交易数量")
                        shares = int((self.current_capital * self.backtest_params.capital_allocation) / 
                                    (price * (1 + self.backtest_params.transaction_cost)))
                        if shares <= 0:
                            continue
                        cost = shares * price
                        transaction_cost = cost * self.backtest_params.transaction_cost
                        total_cost = cost + transaction_cost
                    
                    self.current_capital -= total_cost
                    self.position += shares
                    
                    # 计算止损止盈价格
                    stop_loss_price = price * (1 - self.backtest_params.stop_loss_ratio)
                    take_profit_price = price * (1 + self.backtest_params.take_profit_ratio)
                    
                    # 生成交易ID
                    trade_id = f"TRADE_{len(trade_records) + 1}_{signal['type'].upper()}_{signal['date'].strftime('%Y%m%d%H%M%S')}"
                    
                    trade = TradeRecord(
                        date=signal['date'],
                        type='buy',
                        price=price,
                        shares=shares,
                        cost=transaction_cost,
                        capital=self.current_capital,
                        position=self.position,
                        signal_source=f"强度: {signal['strength']:.2f}",
                        notes=signal['reason'],
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        trade_id=trade_id
                    )
                    trade_records.append(trade)
                    last_trade_date = signal['date']
                    logger.info(f"执行买入: {signal['date']} {shares}股 @ {price:.2f}")
                
                # 执行卖出
                elif signal['type'] == 'sell' and self.position > 0:
                    shares = self.position  # 全仓卖出
                    revenue = shares * price
                    transaction_cost = revenue * self.backtest_params.transaction_cost
                    net_revenue = revenue - transaction_cost
                    
                    # 计算利润（修复：精准计算买入成本）
                    last_buys = [t for t in trade_records if t.type == 'buy' and t.position > 0]
                    if not last_buys:
                        logger.warning("无匹配的买入记录，跳过卖出信号")
                        continue
                    last_buy = last_buys[-1]
                    # 按买入时的成本比例计算
                    cost_basis = (last_buy.price * shares) + (last_buy.cost * (shares / last_buy.shares))
                    profit = net_revenue - cost_basis
                    
                    self.current_capital += net_revenue
                    self.position -= shares
                    
                    # 计算持有天数
                    holding_days = (signal['date'] - last_buy.date).days
                    
                    # 生成交易ID
                    trade_id = f"TRADE_{len(trade_records) + 1}_{signal['type'].upper()}_{signal['date'].strftime('%Y%m%d%H%M%S')}"
                    
                    trade = TradeRecord(
                        date=signal['date'],
                        type='sell',
                        price=price,
                        shares=shares,
                        cost=transaction_cost,
                        capital=self.current_capital,
                        position=self.position,
                        profit=profit,
                        holding_days=holding_days,
                        signal_source=f"强度: {signal['strength']:.2f}",
                        notes=signal['reason'],
                        trade_id=trade_id
                    )
                    trade_records.append(trade)
                    last_trade_date = signal['date']
                    logger.info(f"执行卖出: {signal['date']} {shares}股 @ {price:.2f}, 利润: {profit:.2f}")
                
                # 执行做空（如果允许）
                elif self.backtest_params.enable_short and signal['type'] == 'short' and self.position <= 0:
                    shares = max_shares
                    # 做空时的保证金（简化：按100%保证金计算）
                    margin = shares * price * 1.0
                    transaction_cost = margin * self.backtest_params.transaction_cost
                    
                    if margin + transaction_cost > self.current_capital:
                        logger.debug("保证金不足，调整做空数量")
                        shares = int((self.current_capital * self.backtest_params.capital_allocation) / 
                                    (price * (1 + self.backtest_params.transaction_cost)))
                        if shares <= 0:
                            continue
                        margin = shares * price * 1.0
                        transaction_cost = margin * self.backtest_params.transaction_cost
                    
                    self.current_capital -= (margin + transaction_cost)
                    self.position -= shares  # 做空持仓为负数
                    
                    stop_loss_price = price * (1 + self.backtest_params.stop_loss_ratio)  # 做空止损：价格上涨
                    take_profit_price = price * (1 - self.backtest_params.take_profit_ratio)  # 做空止盈：价格下跌
                    
                    trade_id = f"TRADE_{len(trade_records) + 1}_{signal['type'].upper()}_{signal['date'].strftime('%Y%m%d%H%M%S')}"
                    
                    trade = TradeRecord(
                        date=signal['date'],
                        type='short',
                        price=price,
                        shares=shares,
                        cost=transaction_cost,
                        capital=self.current_capital,
                        position=self.position,
                        signal_source=f"强度: {signal['strength']:.2f}",
                        notes=signal['reason'],
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        trade_id=trade_id
                    )
                    trade_records.append(trade)
                    last_trade_date = signal['date']
                    logger.info(f"执行做空: {signal['date']} {shares}股 @ {price:.2f}")
                
                # 执行平仓（如果允许做空）
                elif self.backtest_params.enable_short and signal['type'] == 'cover' and self.position < 0:
                    shares = abs(self.position)  # 全仓平仓
                    revenue = shares * price
                    transaction_cost = revenue * self.backtest_params.transaction_cost
                    net_revenue = revenue - transaction_cost
                    
                    # 计算做空利润
                    last_shorts = [t for t in trade_records if t.type == 'short' and t.position < 0]
                    if not last_shorts:
                        logger.warning("无匹配的做空记录，跳过平仓信号")
                        continue
                    last_short = last_shorts[-1]
                    # 做空利润 = 卖出价 - 买入平仓价 - 交易成本
                    profit = (last_short.price * shares) - (price * shares) - transaction_cost - last_short.cost
                    
                    # 返还保证金
                    margin_return = last_short.price * shares * 1.0
                    self.current_capital += (net_revenue + margin_return)
                    self.position += shares  # 平仓后持仓归0
                    
                    holding_days = (signal['date'] - last_short.date).days
                    
                    trade_id = f"TRADE_{len(trade_records) + 1}_{signal['type'].upper()}_{signal['date'].strftime('%Y%m%d%H%M%S')}"
                    
                    trade = TradeRecord(
                        date=signal['date'],
                        type='cover',
                        price=price,
                        shares=shares,
                        cost=transaction_cost,
                        capital=self.current_capital,
                        position=self.position,
                        profit=profit,
                        holding_days=holding_days,
                        signal_source=f"强度: {signal['strength']:.2f}",
                        notes=signal['reason'],
                        trade_id=trade_id
                    )
                    trade_records.append(trade)
                    last_trade_date = signal['date']
                    logger.info(f"执行平仓: {signal['date']} {shares}股 @ {price:.2f}, 利润: {profit:.2f}")
            
            logger.info(f"交易执行完成，共 {len(trade_records)} 笔交易")
            return trade_records
        
        except Exception as e:
            logger.error(f"执行交易失败: {str(e)}")
            raise
    
    def calculate_performance(self, trade_records: List[TradeRecord], data: pd.DataFrame) -> Dict[str, Any]:
        """计算回测绩效指标"""
        if not trade_records:
            logger.warning("无交易记录，无法计算绩效指标")
            return {}
        
        try:
            logger.info("开始计算绩效指标")
            
            # 基础指标
            initial_capital = self.backtest_params.initial_capital
            
            # 修复：计算最终资金时考虑持仓股票市值
            if not trade_records:
                final_capital = initial_capital
            else:
                # 获取最后一笔交易记录
                last_trade = trade_records[-1]
                
                # 检查是否有未平仓的持仓
                open_position = False
                position_shares = 0
                position_avg_price = 0
                
                # 计算当前持仓情况
                for trade in trade_records:
                    if trade.type == 'buy':
                        position_shares += trade.shares
                        position_avg_price = ((position_avg_price * (position_shares - trade.shares)) + 
                                            (trade.price * trade.shares)) / position_shares
                    elif trade.type == 'sell':
                        position_shares -= trade.shares
                    elif trade.type == 'short':
                        position_shares -= trade.shares  # 做空为负
                        position_avg_price = ((abs(position_avg_price) * (abs(position_shares) - trade.shares)) + 
                                            (trade.price * trade.shares)) / abs(position_shares)
                    elif trade.type == 'cover':
                        position_shares += trade.shares  # 平空为正
                
                # 如果有持仓，计算持仓市值
                if position_shares != 0:
                    # 获取最后一天的收盘价作为估算价格
                    last_close_price = data.iloc[-1]['close'] if not data.empty else position_avg_price
                    position_value = abs(position_shares) * last_close_price
                    
                    # 最终资金 = 剩余资金 + 持仓市值
                    final_capital = last_trade.capital + position_value
                    logger.info(f"计算最终资金：剩余资金={last_trade.capital:.2f}, 持仓市值={position_value:.2f}, 最终资金={final_capital:.2f}")
                else:
                    # 无持仓，直接使用最后交易后的资金
                    final_capital = last_trade.capital
            
            total_return = final_capital - initial_capital
            return_percent = (total_return / initial_capital) * 100
            
            # 计算回测持续时间（修复：用于年化收益计算）
            start_date = pd.to_datetime(self.backtest_params.start_date)
            end_date = pd.to_datetime(self.backtest_params.end_date)
            total_days = (end_date - start_date).days
            years = total_days / 365.25  # 按自然年计算
            annual_return = (pow((final_capital / initial_capital), 1/years) - 1) * 100 if years > 0 else 0
            
            # 交易统计
            trade_count = len(trade_records)
            buy_count = sum(1 for t in trade_records if t.type == 'buy')
            sell_count = sum(1 for t in trade_records if t.type == 'sell')
            short_count = sum(1 for t in trade_records if t.type == 'short')
            cover_count = sum(1 for t in trade_records if t.type == 'cover')
            
            # 盈利统计
            winning_trades = sum(1 for t in trade_records if t.profit > 0)
            losing_trades = sum(1 for t in trade_records if t.profit < 0)
            win_rate = winning_trades / trade_count if trade_count > 0 else 0
            
            # 收益指标
            total_profit = sum(t.profit for t in trade_records if t.profit > 0)
            total_loss = sum(abs(t.profit) for t in trade_records if t.profit < 0)
            average_profit = total_profit / winning_trades if winning_trades > 0 else 0
            average_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            # 最大连续盈亏
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_wins = 0
            current_losses = 0
            
            for t in trade_records:
                if t.profit > 0:
                    current_wins += 1
                    current_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                elif t.profit < 0:
                    current_losses += 1
                    current_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)
            
            # 持有期统计
            holding_periods = [t.holding_days for t in trade_records if t.holding_days > 0]
            average_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0
            
            # 计算最大回撤（精准版）
            daily_equity = []
            current_equity = initial_capital
            peak_equity = initial_capital
            max_drawdown = 0.0
            drawdown_dates = []
            drawdown_values = []
            
            # 跟踪当前持仓情况
            position_shares = 0
            position_avg_price = 0
            available_capital = initial_capital
            
            # 按实际交易日计算权益曲线
            for date_idx, date_row in data.iterrows():
                date = date_row['date']
                current_price = date_row['close']
                
                # 更新当日权益
                daily_trades = [t for t in trade_records if t.date.date() == date.date()]
                for trade in daily_trades:
                    if trade.type == 'buy':
                        # 更新持仓信息
                        position_shares += trade.shares
                        position_avg_price = ((position_avg_price * (position_shares - trade.shares)) + 
                                            (trade.price * trade.shares)) / position_shares
                        available_capital = trade.capital
                    elif trade.type == 'sell':
                        # 更新持仓信息
                        position_shares -= trade.shares
                        available_capital = trade.capital
                    elif trade.type == 'short':
                        # 更新持仓信息
                        position_shares -= trade.shares  # 做空为负
                        position_avg_price = ((abs(position_avg_price) * (abs(position_shares) - trade.shares)) + 
                                            (trade.price * trade.shares)) / abs(position_shares) if abs(position_shares) > 0 else 0
                        available_capital = trade.capital
                    elif trade.type == 'cover':
                        # 更新持仓信息
                        position_shares += trade.shares  # 平空为正
                        available_capital = trade.capital
                
                # 计算当日总权益（可用资金 + 持仓市值）
                if position_shares > 0:  # 多头持仓
                    position_value = position_shares * current_price
                    current_equity = available_capital + position_value
                elif position_shares < 0:  # 空头持仓
                    # 空头持仓的市值计算为：保证金 + (开仓价 - 当前价) * 持仓数量
                    margin = abs(position_shares) * position_avg_price
                    position_value = margin + (position_avg_price - current_price) * abs(position_shares)
                    current_equity = available_capital + position_value
                else:  # 空仓
                    current_equity = available_capital
                
                # 防止权益出现异常值
                if current_equity <= 0:
                    logger.warning(f"异常权益值: {current_equity} 在 {date}")
                    # 使用前一天的权益值作为安全保障
                    if daily_equity:
                        current_equity = daily_equity[-1]['equity']
                    else:
                        current_equity = initial_capital * 0.1  # 至少保留10%的初始资金
                
                daily_equity.append({
                    'date': date,
                    'equity': current_equity
                })
                
                # 更新最大回撤
                peak_equity = max(peak_equity, current_equity)
                # 确保peak_equity不为0，避免除零错误
                if peak_equity > 0:
                    drawdown = (peak_equity - current_equity) / peak_equity
                    # 限制最大回撤计算的合理性，避免极端异常值
                    if drawdown > 1.0:  # 如果回撤超过100%，视为异常
                        logger.warning(f"异常回撤值: {drawdown * 100:.2f}% 在 {date}")
                        drawdown = min(drawdown, 1.0)  # 最多计为100%回撤
                    drawdown_dates.append(date)
                    drawdown_values.append(drawdown)
                    max_drawdown = max(max_drawdown, drawdown)
                else:
                    drawdown = 0
                    drawdown_dates.append(date)
                    drawdown_values.append(drawdown)
            
            # 计算夏普比率（优化：更稳健的计算方法）
            equity_values = pd.Series([e['equity'] for e in daily_equity])
            # 计算日收益率
            returns = equity_values.pct_change().dropna()
            
            # 处理极端收益率值
            if len(returns) > 0:
                # 移除异常值（超过3个标准差的值）
                returns_clean = returns[abs(returns - returns.mean()) <= 3 * returns.std()]
                if len(returns_clean) == 0:  # 如果移除所有值，则使用原始数据
                    returns_clean = returns
            else:
                returns_clean = returns
            
            risk_free_rate = 0.03  # 年化无风险利率
            daily_risk_free = risk_free_rate / 252
            
            if returns_clean.std() != 0:
                # 计算超额收益
                excess_returns = returns_clean - daily_risk_free
                # 使用超额收益计算夏普比率
                sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / returns_clean.std())
                # 确保计算结果合理
                if abs(sharpe_ratio) > 10:  # 限制极端值
                    sharpe_ratio = np.sign(sharpe_ratio) * 10
            else:
                sharpe_ratio = 0
            
            logger.debug(f"夏普比率计算 - 平均收益: {returns_clean.mean():.6f}, 标准差: {returns_clean.std():.6f}, 结果: {sharpe_ratio:.2f}")
            
            # 计算索提诺比率（只考虑下行风险，优化处理）
            downside_returns = returns_clean[returns_clean < daily_risk_free]  # 使用相对无风险利率的下行风险
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                sortino_ratio = np.sqrt(252) * ((returns_clean.mean() - daily_risk_free) / downside_returns.std())
                # 确保计算结果合理
                if abs(sortino_ratio) > 10:
                    sortino_ratio = np.sign(sortino_ratio) * 10
            else:
                sortino_ratio = 0
            
            # 基准收益（使用期间涨跌幅）
            benchmark_return = 0.0
            if not data.empty:
                start_price = data.iloc[0]['close']
                end_price = data.iloc[-1]['close']
                benchmark_return = ((end_price - start_price) / start_price) * 100
            
            # 整理绩效指标
            performance = {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'return_percent': return_percent,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'trade_count': trade_count,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'short_count': short_count,
                'cover_count': cover_count,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate * 100,
                'average_profit': average_profit,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'average_holding_period': average_holding_period,
                'daily_equity': daily_equity,
                'benchmark_return': benchmark_return
            }
            
            logger.info("绩效指标计算完成")
            return performance
        
        except Exception as e:
            logger.error(f"计算绩效指标失败: {str(e)}")
            raise

    def run_backtest(self, params: BacktestParams) -> BacktestResult:
        """
        运行回测
        :param params: 回测参数
        :return: 回测结果
        """
        try:
            logger.info(f"开始回测: {params.symbol} {params.timeframe} {params.start_date}至{params.end_date}")
            
            # 从配置文件读取初始资金并应用到params
            config_initial_capital = self.config.get('system', {}).get('backtest', {}).get('initial_capital', 100000.0)
            # 仅当用户未通过命令行参数指定初始资金时，才使用配置文件中的值
            if params.initial_capital == 100000.0:  # 默认值
                params.initial_capital = config_initial_capital
                logger.info(f"应用配置文件中的初始资金: {config_initial_capital:.2f}元")
            
            self.backtest_params = params
            
            # 重新初始化计算器以确保参数生效（确保初始资金参数正确传递）
            self._initialize_components()
            
            # 获取历史数据
            historical_data = self.fetch_historical_data(
                symbol=params.symbol,
                timeframe=params.timeframe,
                start_date=params.start_date,
                end_date=params.end_date
            )
            self.historical_data[params.timeframe] = historical_data
            
            # 计算缠论指标
            indicator_data = self.calculate_chanlun_indicators(
                data=historical_data,
                timeframe=params.timeframe
            )
            
            # 生成交易信号
            signals = self.generate_trading_signals(indicator_data)
            self.signal_data[params.timeframe] = signals
            
            # 执行交易
            trade_records = self.execute_trades(signals, historical_data)
            self.trade_records = trade_records
            
            # 计算绩效指标
            performance = self.calculate_performance(trade_records, historical_data)
            
            # 生成信号分析
            signal_analysis = {
                'total_signals': len(signals),
                'buy_signals': sum(1 for s in signals if s['type'] == 'buy'),
                'sell_signals': sum(1 for s in signals if s['type'] == 'sell'),
                'short_signals': sum(1 for s in signals if s['type'] == 'short'),
                'cover_signals': sum(1 for s in signals if s['type'] == 'cover'),
                'average_strength': np.mean([s['strength'] for s in signals]) if signals else 0
            }
            
            # 生成图表
            charts = {}
            try:
                # 使用现有的plot_price_with_signals方法代替不存在的plot_backtest_results方法
                chart_path = self.plotter.plot_price_with_signals(
                    df=historical_data,
                    symbol=params.symbol,
                    trade_records=trade_records
                )
                charts['backtest_summary'] = chart_path
                logger.info(f"回测图表生成成功: {chart_path}")
            except Exception as e:
                logger.warning(f"生成回测图表失败: {str(e)}")
            
            # 生成报告
            report = generate_backtest_report(
                symbol=params.symbol,
                strategy_name='ChanlunStrategy',
                start_date=params.start_date,
                end_date=params.end_date,
                initial_capital=params.initial_capital,
                final_capital=performance['final_capital'] if performance else params.initial_capital,
                trades=trade_records,
                performance=performance,
                timeframe=params.timeframe
            )
            
            # 导出数据
            try:
                # 使用现有的export_signals方法导出交易信号
                if signals:
                    signals_path = self.exporter.export_signals(
                        signals=signals,
                        symbol=params.symbol
                    )
                    logger.info(f"交易信号导出成功: {signals_path}")
                
                # 使用export方法导出历史数据
                if not historical_data.empty:
                    data_path = self.exporter.export(
                        df=historical_data,
                        symbol=params.symbol,
                        export_type='backtest_data'
                    )
                    logger.info(f"历史数据导出成功: {data_path}")
                
                # 导出性能指标
                if performance:
                    perf_df = pd.DataFrame([performance])
                    perf_path = self.exporter.export(
                        df=perf_df,
                        symbol=params.symbol,
                        export_type='performance'
                    )
                    logger.info(f"性能指标导出成功: {perf_path}")
                    
            except Exception as e:
                logger.warning(f"导出回测结果失败: {str(e)}")
            
            # 整理回测结果
            result = BacktestResult(
                success=True,
                params=params,
                initial_capital=performance.get('initial_capital', 0),
                final_capital=performance.get('final_capital', 0),
                total_return=performance.get('total_return', 0),
                return_percent=performance.get('return_percent', 0),
                annual_return=performance.get('annual_return', 0),
                max_drawdown=performance.get('max_drawdown', 0),
                sharpe_ratio=performance.get('sharpe_ratio', 0),
                sortino_ratio=performance.get('sortino_ratio', 0),
                trade_count=performance.get('trade_count', 0),
                buy_count=performance.get('buy_count', 0),
                sell_count=performance.get('sell_count', 0),
                short_count=performance.get('short_count', 0),
                cover_count=performance.get('cover_count', 0),
                winning_trades=performance.get('winning_trades', 0),
                losing_trades=performance.get('losing_trades', 0),
                win_rate=performance.get('win_rate', 0),
                average_profit=performance.get('average_profit', 0),
                average_loss=performance.get('average_loss', 0),
                profit_factor=performance.get('profit_factor', 0),
                max_consecutive_wins=performance.get('max_consecutive_wins', 0),
                max_consecutive_losses=performance.get('max_consecutive_losses', 0),
                average_holding_period=performance.get('average_holding_period', 0),
                trade_records=trade_records,
                daily_equity=performance.get('daily_equity', []),
                actual_date_range={
                    'start': historical_data['date'].min().strftime(DATE_FORMAT),
                    'end': historical_data['date'].max().strftime(DATE_FORMAT)
                },
                report=report,
                charts=charts,
                signal_analysis=signal_analysis,
                benchmark_return=performance.get('benchmark_return', 0)
            )
            
            logger.info(f"回测完成: {params.symbol} 最终资金: {result.final_capital:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"回测失败: {str(e)}", exc_info=True)
            return BacktestResult(
                success=False,
                error=str(e),
                params=params
            )

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='缠论回测系统')
    parser.add_argument('--symbol', type=str, required=True, help='股票代码')
    parser.add_argument('--start-date', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='daily', help='时间级别 (daily, weekly, monthly, 60m, 30m, 15m, 5m, 1m)')
    parser.add_argument('--initial-capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--enable-short', action='store_true', help='允许做空')
    
    args = parser.parse_args()
    
    # 运行回测
    try:
        params = BacktestParams(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            initial_capital=args.initial_capital,
            enable_short=args.enable_short
        )
        
        engine = BacktestEngine()
        result = engine.run_backtest(params)
        
        if result.success:
            print(f"回测成功！最终资金: {result.final_capital:.2f}")
            print(f"总收益率: {result.return_percent:.2f}%")
            print(f"最大回撤: {result.max_drawdown:.2f}%")
            print(f"交易次数: {result.trade_count}")
            print(f"胜率: {result.win_rate:.2f}%")
        else:
            print(f"回测失败: {result.error}")
    except Exception as e:
        print(f"执行回测时出错: {str(e)}")
        sys.exit(1)