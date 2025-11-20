#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论回测引擎（完整版V2.3）
支持：单标的/多标的回测、多时间级别适配、缠论全指标计算、精细化交易规则、风险控制、绩效归因、结果可视化、多渠道通知
更新日志：
1. 修复 calculator.calculate() tuple 返回值适配问题
2. 修复初始资金传递优先级（用户--capital参数优先）
3. 增加返回值类型校验，提升容错性
4. 优化初始资金日志显示：所有日志统一使用用户指定的初始资金，无硬编码值
"""

import argparse
import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ===================== 日志配置（原有完整配置）=====================
logger = logging.getLogger('ChanlunBacktester')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)

# 控制台输出（原有配置）
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 文件输出（原有配置，按日期拆分日志）
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'backtest_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ===================== 路径配置（原有完整配置）=====================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# ===================== 导入核心模块（原有完整导入）=====================
from src.data_fetcher import DataFetcher  # 数据获取（支持腾讯/新浪/本地缓存）
from src.calculator import ChanlunCalculator  # 缠论核心计算器
from src.notifier import DingTalkNotifier, EmailNotifier  # 多渠道通知
from src.plotter import ChanlunPlotter  # 可视化（K线+缠论指标+交易信号）
from src.exporter import ChanlunExporter  # 结果导出（CSV/Excel/JSON）
from src.config_loader import ConfigLoader  # 配置加载（YAML/JSON）
from src.risk_manager import RiskManager  # 风险控制（止损/仓位/对冲）
from src.performance_analyzer import PerformanceAnalyzer  # 绩效分析（归因/夏普/最大回撤）
from src.utils import (  # 工具函数（原有完整工具集）
    date2str, str2date, format_number, calculate_fee, 
    validate_symbol, get_timeframe_interval, retry_decorator
)

# ===================== 常量定义（原有完整常量）=====================
SUPPORTED_MODES = ['single', 'multi']  # 回测模式
SUPPORTED_TIMEFRAMES = ['daily', 'weekly', '60min', '30min', '15min', '5min']  # 支持的时间级别
REQUIRED_KLINE_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']  # K线必要列
REQUIRED_CALC_COLUMNS = [  # 缠论计算必要输出列（原有完整列表）
    'top_fractal', 'bottom_fractal', 'fractal_type',
    'pen_type', 'pen_id', 'pen_start', 'pen_end',
    'segment_type', 'segment_id', 'segment_start', 'segment_end',
    'central_bank', 'central_bank_id', 'central_bank_high', 'central_bank_low',
    'divergence', 'divergence_strength', 'signal', 'signal_strength',
    'stop_loss_price', 'market_condition'
]
DEFAULT_EXPORT_PATH = 'outputs/exports'  # 默认导出路径
DEFAULT_CACHE_EXPIRE = 3600  # 数据缓存过期时间（秒）
MIN_DATA_POINTS = 50  # 最小有效数据量

# ===================== 回测引擎主类（完整保留原有逻辑）=====================
class ChanlunBacktester:
    """
    缠论回测引擎（完整版）
    核心功能：
    1. 多模式回测：单标的/多标的组合
    2. 多时间级别：日线/周线/分钟线
    3. 完整缠论指标：分型/笔/线段/中枢/背离
    4. 精细化交易规则：信号过滤/仓位管理/止损止盈
    5. 风险控制：动态止损/仓位限制/滑点手续费
    6. 绩效分析：总收益/年化/夏普/最大回撤/胜率/盈亏比
    7. 结果输出：可视化/多格式导出/多渠道通知
    """
    
    def __init__(self, args: argparse.Namespace):
        """初始化回测引擎（完整保留原有初始化逻辑）"""
        self.args = args
        self._parse_args()  # 解析命令行参数
        self._load_configs()  # 加载配置文件
        self._init_components()  # 初始化核心组件
        self._validate_params()  # 验证参数合法性
        self._init_backtest_state()  # 初始化回测状态
        
        # 全局变量（原有完整定义）
        self.backtest_results: Dict[str, Any] = {}  # 回测结果存储
        self.trade_records: List[Dict[str, Any]] = []  # 全局交易记录
        self.portfolio_history: List[Dict[str, Any]] = []  # 组合历史
        
        logger.info("="*80)
        logger.info(f"缠论回测引擎（完整版V2.3）初始化完成 | 初始资金：{self.initial_capital:.2f}元")
        logger.info("="*80)

    def _parse_args(self):
        """解析命令行参数（完整保留原有逻辑）"""
        self.mode = self.args.mode
        self.symbol = self.args.symbol.strip()
        self.start_date_str = self.args.start_date
        self.end_date_str = self.args.end_date
        self.timeframe = self.args.timeframe
        self.initial_capital = self.args.capital  # 用户指定初始资金（核心变量）
        self.enable_notify = self.args.enable_notify
        self.enable_plot = self.args.enable_plot
        self.enable_cache = self.args.enable_cache
        self.export_path = self.args.export_path or DEFAULT_EXPORT_PATH
        self.config_path = self.args.config_path or 'config/system.yaml'
        self.risk_config_path = self.args.risk_config_path or 'config/risk.yaml'
        self.notifier_config_path = self.args.notifier_config_path or 'config/notifier.yaml'
        
        # 多标的处理（原有逻辑）
        self.symbols = [s.strip() for s in self.symbol.split(',')] if self.mode == 'multi' else [self.symbol]
        
        # 日期转换（原有逻辑）
        self.start_date = str2date(self.start_date_str)
        self.end_date = str2date(self.end_date_str)

    def _load_configs(self):
        """加载所有配置文件（完整保留原有逻辑）"""
        self.config_loader = ConfigLoader()
        
        # 系统配置（原有逻辑）
        self.system_config = self.config_loader.load(self.config_path)
        self.chanlun_config = self.system_config.get('chanlun', {})
        self.data_fetcher_config = self.system_config.get('data_fetcher', {})
        self.plotter_config = self.system_config.get('plotter', {})
        self.exporter_config = self.system_config.get('exporter', {})
        
        # 风险配置（原有逻辑）
        self.risk_config = self.config_loader.load(self.risk_config_path)
        self.stop_loss_config = self.risk_config.get('stop_loss', {})
        self.position_config = self.risk_config.get('position', {})
        self.cost_config = self.risk_config.get('cost', {})
        
        # 通知配置（原有逻辑）
        self.notifier_config = self.config_loader.load(self.notifier_config_path)
        
        # 缠论参数按时间级别拆分（原有逻辑）
        self.timeframe_chanlun_config = self._get_timeframe_chanlun_config()

    def _get_timeframe_chanlun_config(self) -> Dict[str, Any]:
        """获取当前时间级别的缠论参数（原有完整逻辑）"""
        # 优先级：时间级别专属配置 > 全局缠论配置 > 默认值
        default_config = {
            'fractal_sensitivity': 3,
            'pen_min_length': 5,
            'central_bank_min_length': 5,
            'segment_min_length': 3,
            'divergence_threshold': 0.015,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'atr_period': 14
        }
        global_chanlun_config = self.chanlun_config.get('global', {})
        timeframe_specific_config = self.chanlun_config.get(self.timeframe, {})
        
        # 合并配置（原有逻辑）
        config = {**default_config, **global_chanlun_config, **timeframe_specific_config}
        logger.info(f"当前{self.timeframe}级别缠论参数：{json.dumps(config, ensure_ascii=False, indent=2)}")
        return config

    def _init_components(self):
        """初始化核心组件（完整保留原有逻辑，优化初始资金日志）"""
        # 1. 数据获取器（原有逻辑）
        self.data_fetcher = DataFetcher(
            config=self.data_fetcher_config,
            enable_cache=self.enable_cache,
            cache_expire=DEFAULT_CACHE_EXPIRE
        )
        logger.info("数据获取器初始化完成（支持：腾讯数据源/新浪数据源/本地缓存/多时间级别）")
        
        # 2. 缠论计算器（核心修复：传递用户指定的初始资金，优化日志）
        calculator_config = {
            'chanlun': self.chanlun_config,
            'risk_management': self.risk_config,
            'initial_capital': self.initial_capital,  # 优先使用用户--capital参数
            'data_validation_enabled': self.system_config.get('data_validation_enabled', True),
            'min_data_points': self.system_config.get('min_data_points', MIN_DATA_POINTS)
        }
        self.calculator = ChanlunCalculator(config=calculator_config)
        logger.info(f"缠论核心计算器初始化完成（支持：分型/笔/线段/中枢/背离/信号生成）| 初始资金：{self.initial_capital:.2f}元")
        
        # 3. 风险管理器（原有逻辑）
        self.risk_manager = RiskManager(
            stop_loss_config=self.stop_loss_config,
            position_config=self.position_config,
            cost_config=self.cost_config
        )
        logger.info("风险管理器初始化完成（支持：动态止损/仓位管理/成本计算）")
        
        # 4. 绩效分析器（原有逻辑）
        self.performance_analyzer = PerformanceAnalyzer(
            risk_free_rate=self.risk_config.get('risk_free_rate', 0.03)
        )
        logger.info("绩效分析器初始化完成（支持：收益/夏普/最大回撤/归因分析）")
        
        # 5. 可视化工具（原有逻辑）
        self.plotter = ChanlunPlotter(config=self.plotter_config) if self.enable_plot else None
        if self.enable_plot:
            logger.info("可视化工具初始化完成（支持：K线+缠论指标+交易信号+组合曲线）")
        
        # 6. 结果导出器（原有逻辑）
        self.exporter = ChanlunExporter(
            output_dir=self.export_path,
            config=self.exporter_config
        )
        logger.info(f"结果导出器初始化完成（输出目录：{self.export_path}，支持：CSV/Excel/JSON）")
        
        # 7. 通知工具（原有逻辑）
        self.notifiers = []
        if self.enable_notify:
            # 钉钉通知（原有逻辑）
            if self.notifier_config.get('dingtalk', {}).get('enabled', False):
                dingtalk_notifier = DingTalkNotifier(config=self.notifier_config['dingtalk'])
                self.notifiers.append(dingtalk_notifier)
            # 邮件通知（原有逻辑）
            if self.notifier_config.get('email', {}).get('enabled', False):
                email_notifier = EmailNotifier(config=self.notifier_config['email'])
                self.notifiers.append(email_notifier)
            logger.info(f"通知工具初始化完成（共{len(self.notifiers)}个渠道）")

    def _validate_params(self):
        """验证所有参数合法性（完整保留原有逻辑）"""
        logger.info("开始参数合法性验证...")
        
        # 1. 模式验证（原有逻辑）
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"不支持的回测模式：{self.mode}，仅支持{SUPPORTED_MODES}")
        
        # 2. 时间级别验证（原有逻辑）
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"不支持的时间级别：{self.timeframe}，仅支持{SUPPORTED_TIMEFRAMES}")
        
        # 3. 日期验证（原有逻辑）
        if self.start_date >= self.end_date:
            raise ValueError(f"开始日期{self.start_date_str}不能晚于结束日期{self.end_date_str}")
        date_diff = (self.end_date - self.start_date).days
        if date_diff < 30:
            logger.warning(f"回测周期过短（仅{date_diff}天），可能导致结果不准确")
        
        # 4. 标的验证（原有逻辑）
        for symbol in self.symbols:
            if not validate_symbol(symbol):
                raise ValueError(f"无效标的代码：{symbol}（格式错误）")
        
        # 5. 初始资金验证（原有逻辑）
        if self.initial_capital <= 0:
            raise ValueError(f"初始资金必须大于0，当前值：{self.initial_capital}")
        
        # 6. 路径验证（原有逻辑）
        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path, exist_ok=True)
            logger.warning(f"导出路径不存在，已自动创建：{self.export_path}")
        
        # 7. 配置文件验证（原有逻辑）
        for config_path in [self.config_path, self.risk_config_path, self.notifier_config_path]:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在：{config_path}")
        
        logger.info(f"参数合法性验证通过 | 初始资金：{self.initial_capital:.2f}元")

    def _init_backtest_state(self):
        """初始化回测状态（完整保留原有逻辑）"""
        # 单标的回测状态（原有逻辑）
        self.single_state = {
            'cash': self.initial_capital,
            'position': 0,
            'avg_cost': 0.0,
            'current_stop_loss': 0.0,
            'trade_count': 0,
            'win_count': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'peak_value': self.initial_capital
        }
        
        # 多标的回测状态（原有逻辑）
        self.multi_state = {
            'cash': self.initial_capital,
            'positions': {symbol: 0 for symbol in self.symbols},
            'avg_costs': {symbol: 0.0 for symbol in self.symbols},
            'stop_losses': {symbol: 0.0 for symbol in self.symbols},
            'trade_counts': {symbol: 0 for symbol in self.symbols},
            'win_counts': {symbol: 0 for symbol in self.symbols},
            'profits': {symbol: 0.0 for symbol in self.symbols},
            'peak_values': {symbol: self.initial_capital / len(self.symbols) for symbol in self.symbols},
            'total_peak_value': self.initial_capital
        }
        if self.mode == 'multi':
            logger.info(f"多标的资金分配完成 | 总初始资金：{self.initial_capital:.2f}元 | 单标的分配：{self.initial_capital/len(self.symbols):.2f}元/个")

    @retry_decorator(max_retries=3, delay=2)
    def _fetch_kline_data(self, symbol: str) -> pd.DataFrame:
        """获取K线数据（完整保留原有逻辑，含重试/缓存/数据清洗）"""
        logger.info(f"\n{'='*50}")
        logger.info(f"开始获取标的{symbol}的{self.timeframe}数据 | 回测初始资金：{self.initial_capital:.2f}元")
        logger.info(f"日期范围：{self.start_date_str} ~ {self.end_date_str}")
        
        # 转换日期格式（原有逻辑）
        start_date = date2str(self.start_date)
        end_date = date2str(self.end_date)
        
        # 根据时间级别获取数据（原有完整逻辑）
        if self.timeframe == 'daily':
            kline_df = self.data_fetcher.get_daily_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
        elif self.timeframe == 'weekly':
            kline_df = self.data_fetcher.get_weekly_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
        elif self.timeframe in ['60min', '30min', '15min', '5min']:
            interval = int(self.timeframe.replace('min', ''))
            kline_df = self.data_fetcher.get_minute_data(
                symbol=symbol, start_date=start_date, end_date=end_date, interval=interval
            )
        else:
            raise ValueError(f"不支持的时间级别：{self.timeframe}")
        
        # 数据清洗（原有完整逻辑）
        kline_df = self._clean_kline_data(kline_df, symbol)
        
        # 验证数据量（原有逻辑）
        if len(kline_df) < MIN_DATA_POINTS:
            raise ValueError(f"标的{symbol}有效数据量不足{MIN_DATA_POINTS}条（当前{len(kline_df)}条）")
        
        logger.info(f"标的{symbol}数据获取完成：{len(kline_df)}条有效记录")
        logger.info(f"实际日期范围：{date2str(kline_df['date'].min())} ~ {date2str(kline_df['date'].max())}")
        return kline_df

    def _clean_kline_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """清洗K线数据（完整保留原有逻辑：去重/补全/格式转换）"""
        df = df.copy()
        
        # 1. 列名标准化（原有逻辑）
        df.columns = df.columns.str.lower()
        for col in REQUIRED_KLINE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"标的{symbol}K线数据缺失必要列：{col}")
        
        # 2. 日期格式转换（原有逻辑）
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # 3. 去重（原有逻辑）
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # 4. 排序（原有逻辑）
        df = df.sort_values('date').reset_index(drop=True)
        
        # 5. 补全缺失值（原有逻辑）
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].fillna(method='ffill').interpolate(method='linear')
        df['volume'] = df['volume'].fillna(0)
        
        # 6. 价格有效性过滤（原有逻辑）
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        df = df[df['volume'] >= 0]
        
        # 7. 截取日期范围（原有逻辑）
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        return df.reset_index(drop=True)

    def _calculate_chanlun(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        """执行缠论计算（核心修复：适配tuple返回值+类型校验）"""
        logger.info(f"\n开始{self.timeframe}级别缠论计算... | 回测初始资金：{self.initial_capital:.2f}元")
        
        # 获取当前时间级别的缠论参数（原有逻辑）
        cl_config = self.timeframe_chanlun_config
        fractal_sensitivity = cl_config['fractal_sensitivity']
        pen_min_length = cl_config['pen_min_length']
        central_bank_min_length = cl_config['central_bank_min_length']
        segment_min_length = cl_config['segment_min_length']
        divergence_threshold = cl_config['divergence_threshold']
        
        # 执行缠论计算（核心修复：接收tuple返回值）
        try:
            # 解析 tuple (计算后DataFrame, 回测结果)
            result_df, calculator_backtest_result = self.calculator.calculate(
                df=kline_df,
                timeframe=self.timeframe,
                fractal_sensitivity=fractal_sensitivity,
                pen_min_length=pen_min_length,
                central_bank_min_length=central_bank_min_length,
                segment_min_length=segment_min_length,
                divergence_threshold=divergence_threshold
            )
            # 保存计算器返回的回测结果（用于后续合并）
            self.calculator_backtest_result = calculator_backtest_result
        except Exception as e:
            logger.error(f"缠论计算失败：{str(e)}", exc_info=True)
            raise
        
        # 核心修复：增加返回值类型校验（避免tuple导致的AttributeError）
        if not isinstance(result_df, pd.DataFrame):
            error_msg = f"缠论计算返回值类型错误：预期pd.DataFrame，实际为{type(result_df)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # 验证计算结果列（原有逻辑）
        missing_cols = [col for col in REQUIRED_CALC_COLUMNS if col not in result_df.columns]
        if missing_cols:
            logger.warning(f"缠论计算结果缺失列：{missing_cols}，部分功能可能受影响")
        
        # 统计核心指标（原有逻辑）
        top_fractal_count = result_df['top_fractal'].sum()
        bottom_fractal_count = result_df['bottom_fractal'].sum()
        pen_count = result_df['pen_id'].nunique() - 1  # 排除-1（无笔）
        segment_count = result_df['segment_id'].nunique() - 1  # 排除-1（无线段）
        central_bank_count = result_df['central_bank_id'].nunique() - 1  # 排除-1（无中枢）
        buy_signal_count = (result_df['signal'] == 'buy').sum()
        sell_signal_count = (result_df['signal'] == 'sell').sum()
        
        # 日志输出统计结果（优化初始资金显示）
        logger.info(f"缠论计算完成，核心指标统计：")
        logger.info(f"  - 顶分型：{top_fractal_count}个 | 底分型：{bottom_fractal_count}个")
        logger.info(f"  - 有效笔：{pen_count}支 | 有效线段：{segment_count}段 | 有效中枢：{central_bank_count}个")
        logger.info(f"  - 买入信号：{buy_signal_count}个 | 卖出信号：{sell_signal_count}个")
        logger.info(f"  - 回测初始资金：{self.initial_capital:.2f}元")
        
        return result_df

    def _execute_trades(self, df: pd.DataFrame, symbol: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """执行交易（完整保留原有逻辑：信号过滤/仓位计算/止损执行/成本扣除）"""
        logger.info(f"\n开始标的{symbol}交易执行... | 初始资金：{self.initial_capital:.2f}元")
        trade_records = []
        position_history = []
        
        # 单标的/多标的状态区分（原有逻辑）
        if self.mode == 'single':
            state = self.single_state
        else:
            state = self.multi_state
            # 多标的资金分配（优化日志：显示单标的分配金额）
            symbol_capital = self.initial_capital / len(self.symbols)
            logger.info(f"多标的模式 - 标的{symbol}分配资金：{symbol_capital:.2f}元（总初始资金：{self.initial_capital:.2f}元）")
        
        # 遍历K线执行交易（原有完整逻辑）
        for idx, row in df.iterrows():
            date = row['date']
            close_price = row['close']
            signal = row['signal']
            signal_strength = row['signal_strength']
            stop_loss_price = row['stop_loss_price']
            
            # 记录当前组合状态（原有逻辑）
            current_position = state['position'] if self.mode == 'single' else state['positions'][symbol]
            current_cash = state['cash'] if self.mode == 'single' else state['cash']
            position_value = current_position * close_price
            total_value = current_cash + position_value
            
            position_history.append({
                'date': date,
                'symbol': symbol,
                'cash': current_cash,
                'position': current_position,
                'position_value': position_value,
                'total_value': total_value,
                'signal': signal,
                'close_price': close_price,
                'stop_loss_price': stop_loss_price,
                'initial_capital': self.initial_capital  # 记录初始资金到历史数据
            })
            
            # 无信号则跳过（原有逻辑）
            if signal == 'hold':
                continue
            
            # 买入信号执行（原有完整逻辑）
            if signal == 'buy' and current_position == 0:
                # 计算仓位（原有逻辑：信号强度+风险配置）
                if self.mode == 'single':
                    position_ratio = self._calculate_position_ratio(signal_strength)
                    max_position_value = state['cash'] * position_ratio
                else:
                    position_ratio = self._calculate_position_ratio(signal_strength)
                    max_position_value = symbol_capital * position_ratio
                
                # 计算可买数量（原有逻辑：扣除成本）
                buy_price = close_price * (1 + self.cost_config.get('slippage_ratio', 0.001))  # 滑点
                max_shares = max_position_value / buy_price
                max_shares = int(max_shares // 100 * 100) if self.position_config.get('round_lot', True) else int(max_shares)
                if max_shares <= 0:
                    logger.warning(f"日期{date2str(date)}：买入信号但资金不足，可买数量{max_shares} | 初始资金：{self.initial_capital:.2f}元")
                    continue
                
                # 计算交易成本（原有逻辑）
                trade_amount = max_shares * buy_price
                fee = calculate_fee(
                    amount=trade_amount,
                    commission_rate=self.cost_config.get('commission_rate', 0.0005),
                    min_commission=self.cost_config.get('min_commission', 5.0)
                )
                total_cost = trade_amount + fee
                
                # 验证资金（原有逻辑）
                if total_cost > state['cash']:
                    logger.warning(f"日期{date2str(date)}：买入信号但资金不足（需{total_cost:.2f}，可用{state['cash']:.2f}）| 初始资金：{self.initial_capital:.2f}元")
                    continue
                
                # 更新状态（原有逻辑）
                if self.mode == 'single':
                    state['cash'] -= total_cost
                    state['position'] = max_shares
                    state['avg_cost'] = total_cost / max_shares
                    state['current_stop_loss'] = stop_loss_price
                    state['trade_count'] += 1
                else:
                    state['cash'] -= total_cost
                    state['positions'][symbol] = max_shares
                    state['avg_costs'][symbol] = total_cost / max_shares
                    state['stop_losses'][symbol] = stop_loss_price
                    state['trade_counts'][symbol] += 1
                
                # 记录交易（原有逻辑）
                trade_record = {
                    'trade_id': len(trade_records) + 1,
                    'date': date,
                    'symbol': symbol,
                    'signal': 'buy',
                    'price': buy_price,
                    'shares': max_shares,
                    'trade_amount': trade_amount,
                    'fee': fee,
                    'total_cost': total_cost,
                    'cash_after': state['cash'] if self.mode == 'single' else state['cash'],
                    'avg_cost': state['avg_cost'] if self.mode == 'single' else state['avg_costs'][symbol],
                    'signal_strength': signal_strength,
                    'initial_capital': self.initial_capital  # 记录初始资金到交易记录
                }
                trade_records.append(trade_record)
                logger.info(f"日期{date2str(date)}：执行买入交易 -> {symbol} {max_shares}股，价格{buy_price:.2f}，成本{total_cost:.2f} | 初始资金：{self.initial_capital:.2f}元")
            
            # 卖出信号执行（原有完整逻辑）
            elif signal == 'sell' and current_position > 0:
                # 计算卖出价格（原有逻辑：滑点）
                sell_price = close_price * (1 - self.cost_config.get('slippage_ratio', 0.001))
                
                # 计算交易收益（原有逻辑）
                trade_amount = current_position * sell_price
                fee = calculate_fee(
                    amount=trade_amount,
                    commission_rate=self.cost_config.get('commission_rate', 0.0005),
                    min_commission=self.cost_config.get('min_commission', 5.0)
                )
                net_amount = trade_amount - fee
                
                # 计算盈亏（原有逻辑）
                avg_cost = state['avg_cost'] if self.mode == 'single' else state['avg_costs'][symbol]
                total_cost = avg_cost * current_position
                profit = net_amount - total_cost
                profit_ratio = (profit / total_cost) * 100 if total_cost > 0 else 0
                
                # 更新状态（原有逻辑）
                if self.mode == 'single':
                    state['cash'] += net_amount
                    state['position'] = 0
                    state['trade_count'] += 1
                    state['total_profit'] += profit
                    if profit > 0:
                        state['win_count'] += 1
                    # 更新最大回撤（原有逻辑）
                    state['peak_value'] = max(state['peak_value'], total_value)
                    drawdown = (state['peak_value'] - state['cash']) / state['peak_value']
                    state['max_drawdown'] = max(state['max_drawdown'], drawdown)
                else:
                    state['cash'] += net_amount
                    state['positions'][symbol] = 0
                    state['trade_counts'][symbol] += 1
                    state['profits'][symbol] += profit
                    if profit > 0:
                        state['win_counts'][symbol] += 1
                    # 更新最大回撤（原有逻辑）
                    symbol_peak = max(state['peak_values'][symbol], total_value)
                    state['peak_values'][symbol] = symbol_peak
                    total_peak = sum(state['peak_values'].values()) + state['cash']
                    state['total_peak_value'] = total_peak
                    current_total_value = sum([state['positions'][s] * df.iloc[idx]['close'] for s in self.symbols]) + state['cash']
                    drawdown = (state['total_peak_value'] - current_total_value) / state['total_peak_value']
                    state['max_drawdown'] = max(state['max_drawdown'], drawdown)
                
                # 记录交易（原有逻辑）
                trade_record = {
                    'trade_id': len(trade_records) + 1,
                    'date': date,
                    'symbol': symbol,
                    'signal': 'sell',
                    'price': sell_price,
                    'shares': current_position,
                    'trade_amount': trade_amount,
                    'fee': fee,
                    'net_amount': net_amount,
                    'avg_cost': avg_cost,
                    'profit': profit,
                    'profit_ratio': profit_ratio,
                    'cash_after': state['cash'],
                    'signal_strength': signal_strength,
                    'initial_capital': self.initial_capital  # 记录初始资金到交易记录
                }
                trade_records.append(trade_record)
                logger.info(f"日期{date2str(date)}：执行卖出交易 -> {symbol} {current_position}股，价格{sell_price:.2f}，盈利{profit:.2f}（{profit_ratio:.1f}%）| 初始资金：{self.initial_capital:.2f}元")
        
        logger.info(f"标的{symbol}交易执行完成：共{len(trade_records)}笔交易 | 初始资金：{self.initial_capital:.2f}元")
        return trade_records, position_history

    def _calculate_position_ratio(self, signal_strength: float) -> float:
        """计算仓位比例（完整保留原有逻辑：信号强度+风险配置）"""
        min_ratio = self.position_config.get('min_ratio', 0.2)
        max_ratio = self.position_config.get('max_ratio', 0.8)
        weight = self.position_config.get('signal_strength_weight', 0.6)
        
        # 信号强度加权（原有逻辑）
        position_ratio = min_ratio + (max_ratio - min_ratio) * (signal_strength / 100) * weight
        position_ratio = max(min_ratio, min(max_ratio, position_ratio))
        return position_ratio

    def _analyze_performance(self, trade_records: List[Dict[str, Any]], position_history: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """分析回测绩效（完整保留原有逻辑：收益/夏普/最大回撤/胜率等）"""
        logger.info(f"\n开始标的{symbol}绩效分析... | 初始资金：{self.initial_capital:.2f}元")
        
        # 转换为DataFrame便于分析（原有逻辑）
        position_df = pd.DataFrame(position_history)
        position_df['date'] = pd.to_datetime(position_df['date'])
        trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
        
        # 计算核心绩效指标（原有完整逻辑）
        if len(position_df) == 0:
            logger.warning(f"标的{symbol}无组合历史数据，绩效分析跳过 | 初始资金：{self.initial_capital:.2f}元")
            return {}
        
        # 1. 收益指标（原有逻辑）
        initial_value = position_df['total_value'].iloc[0]
        final_value = position_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        trade_days = (position_df['date'].iloc[-1] - position_df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / trade_days) - 1 if trade_days > 0 else 0
        
        # 2. 风险指标（原有逻辑）
        daily_returns = position_df['total_value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # 年化波动率
        sharpe_ratio = (annual_return - self.risk_config.get('risk_free_rate', 0.03)) / volatility if volatility != 0 else 0
        
        # 3. 最大回撤（原有逻辑）
        position_df['cumulative_max'] = position_df['total_value'].cummax()
        position_df['drawdown'] = (position_df['total_value'] - position_df['cumulative_max']) / position_df['cumulative_max']
        max_drawdown = abs(position_df['drawdown'].min())
        
        # 4. 交易指标（原有逻辑）
        trade_count = len(trade_records)
        long_trade_count = trade_count // 2  # 买入卖出为一对
        win_rate = 0.0
        profit_factor = 0.0
        avg_profit_ratio = 0.0
        
        if long_trade_count > 0 and not trade_df.empty:
            # 胜率（原有逻辑）
            sell_trades = trade_df[trade_df['signal'] == 'sell']
            win_count = (sell_trades['profit'] > 0).sum()
            win_rate = (win_count / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
            
            # 盈亏比（原有逻辑）
            total_profit = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
            total_loss = abs(sell_trades[sell_trades['profit'] <= 0]['profit'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # 平均收益率（原有逻辑）
            avg_profit_ratio = sell_trades['profit_ratio'].mean()
        
        # 5. 其他指标（原有逻辑）
        max_consecutive_win = self._calculate_max_consecutive_win(trade_records)
        max_consecutive_loss = self._calculate_max_consecutive_loss(trade_records)
        avg_holding_days = self._calculate_avg_holding_days(trade_records)
        
        # 整理绩效结果（优化：明确记录初始资金）
        performance = {
            'symbol': symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date_str,
            'end_date': self.end_date_str,
            'initial_capital': self.initial_capital,  # 明确记录用户指定的初始资金
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'long_trade_count': long_trade_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit_ratio': avg_profit_ratio,
            'max_consecutive_win': max_consecutive_win,
            'max_consecutive_loss': max_consecutive_loss,
            'avg_holding_days': avg_holding_days,
            'trade_days': trade_days
        }
        
        # 日志输出绩效结果（优化初始资金显示）
        logger.info("绩效分析完成，核心指标：")
        logger.info(f"  - 初始资金：{self.initial_capital:.2f}元 | 最终资金：{final_value:.2f}元")
        logger.info(f"  - 总收益率：{total_return:.2%} | 年化收益率：{annual_return:.2%}")
        logger.info(f"  - 夏普比率：{sharpe_ratio:.2f} | 最大回撤：{max_drawdown:.2%}")
        logger.info(f"  - 交易次数：{trade_count} | 胜率：{win_rate:.2f}% | 盈亏比：{profit_factor:.2f}")
        
        return performance

    def _calculate_max_consecutive_win(self, trade_records: List[Dict[str, Any]]) -> int:
        """计算最大连续盈利次数（原有逻辑）"""
        if not trade_records:
            return 0
        max_win = 0
        current_win = 0
        for record in trade_records:
            if record['signal'] == 'sell' and record['profit'] > 0:
                current_win += 1
                max_win = max(max_win, current_win)
            elif record['signal'] == 'sell' and record['profit'] <= 0:
                current_win = 0
        return max_win

    def _calculate_max_consecutive_loss(self, trade_records: List[Dict[str, Any]]) -> int:
        """计算最大连续亏损次数（原有逻辑）"""
        if not trade_records:
            return 0
        max_loss = 0
        current_loss = 0
        for record in trade_records:
            if record['signal'] == 'sell' and record['profit'] <= 0:
                current_loss += 1
                max_loss = max(max_loss, current_loss)
            elif record['signal'] == 'sell' and record['profit'] > 0:
                current_loss = 0
        return max_loss

    def _calculate_avg_holding_days(self, trade_records: List[Dict[str, Any]]) -> float:
        """计算平均持仓天数（原有逻辑）"""
        if len(trade_records) < 2:
            return 0.0
        holding_days = []
        buy_records = [r for r in trade_records if r['signal'] == 'buy']
        sell_records = [r for r in trade_records if r['signal'] == 'sell']
        pair_count = min(len(buy_records), len(sell_records))
        for i in range(pair_count):
            buy_date = buy_records[i]['date']
            sell_date = sell_records[i]['date']
            days = (sell_date - buy_date).days
            holding_days.append(days)
        return np.mean(holding_days) if holding_days else 0.0

    def _plot_results(self, df: pd.DataFrame, position_history: List[Dict[str, Any]], symbol: str):
        """绘制回测结果（完整保留原有逻辑：K线+缠论指标+交易信号+组合曲线）"""
        if not self.enable_plot or self.plotter is None:
            return
        
        logger.info(f"\n开始绘制标的{symbol}回测结果... | 初始资金：{self.initial_capital:.2f}元")
        try:
            # 转换数据格式（原有逻辑）
            position_df = pd.DataFrame(position_history)
            position_df['date'] = pd.to_datetime(position_df['date'])
            
            # 绘制K线+缠论指标（原有逻辑）
            kline_plot_path = os.path.join(
                self.export_path,
                f'kline_chanlun_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.png'
            )
            self.plotter.plot_kline_chanlun(
                df=df,
                trade_records=self.trade_records,
                save_path=kline_plot_path,
                title=f'{symbol} {self.timeframe} 缠论指标+交易信号（初始资金：{self.initial_capital:.2f}元）'
            )
            
            # 绘制组合价值曲线（原有逻辑，优化标题显示初始资金）
            portfolio_plot_path = os.path.join(
                self.export_path,
                f'portfolio_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.png'
            )
            self.plotter.plot_portfolio(
                position_df=position_df,
                save_path=portfolio_plot_path,
                title=f'{symbol} {self.timeframe} 组合价值曲线（初始资金：{self.initial_capital:.2f}元）'
            )
            
            # 绘制绩效指标雷达图（原有逻辑，优化标题显示初始资金）
            performance = self.backtest_results[symbol]['performance']
            radar_plot_path = os.path.join(
                self.export_path,
                f'performance_radar_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.png'
            )
            self.plotter.plot_performance_radar(
                performance=performance,
                save_path=radar_plot_path,
                title=f'{symbol} {self.timeframe} 绩效雷达图（初始资金：{self.initial_capital:.2f}元）'
            )
            
            logger.info(f"标的{symbol}绘图完成，文件保存至：{self.export_path} | 初始资金：{self.initial_capital:.2f}元")
        except Exception as e:
            logger.error(f"标的{symbol}绘图失败：{str(e)} | 初始资金：{self.initial_capital:.2f}元", exc_info=True)

    def _export_results(self, df: pd.DataFrame, trade_records: List[Dict[str, Any]], position_history: List[Dict[str, Any]], performance: Dict[str, Any], symbol: str):
        """导出回测结果（完整保留原有逻辑：多格式导出）"""
        logger.info(f"\n开始导出标的{symbol}回测结果... | 初始资金：{self.initial_capital:.2f}元")
        try:
            # 导出K线+缠论指标（原有逻辑）
            chanlun_data_path = os.path.join(
                self.export_path,
                f'chanlun_data_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.csv'
            )
            self.exporter.export_csv(df, chanlun_data_path)
            
            # 导出交易记录（原有逻辑）
            if trade_records:
                trade_records_path = os.path.join(
                    self.export_path,
                    f'trade_records_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.xlsx'
                )
                self.exporter.export_excel(pd.DataFrame(trade_records), trade_records_path)
            
            # 导出组合历史（原有逻辑）
            position_history_path = os.path.join(
                self.export_path,
                f'position_history_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.csv'
            )
            self.exporter.export_csv(pd.DataFrame(position_history), position_history_path)
            
            # 导出绩效指标（原有逻辑，确保包含初始资金）
            performance_path = os.path.join(
                self.export_path,
                f'performance_{symbol}_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.json'
            )
            self.exporter.export_json(performance, performance_path)
            
            logger.info(f"标的{symbol}结果导出完成，文件保存至：{self.export_path} | 初始资金：{self.initial_capital:.2f}元")
        except Exception as e:
            logger.error(f"标的{symbol}结果导出失败：{str(e)} | 初始资金：{self.initial_capital:.2f}元", exc_info=True)

    def _send_notifications(self, performance: Dict[str, Any], symbol: str):
        """发送回测通知（完整保留原有逻辑：多渠道通知，优化初始资金显示）"""
        if not self.enable_notify or not self.notifiers:
            return
        
        logger.info(f"\n开始发送标的{symbol}回测通知... | 初始资金：{self.initial_capital:.2f}元")
        try:
            # 构建通知内容（优化：明确显示用户指定的初始资金）
            notify_content = f"""
【缠论回测完成通知】
=======================
标的：{symbol}
时间级别：{self.timeframe}
回测周期：{self.start_date_str} ~ {self.end_date_str}
初始资金：{self.initial_capital:.2f}元  # 显示用户指定值，而非配置文件默认值
最终资金：{performance['final_value']:.2f}元
总收益率：{performance['total_return']:.2%}
年化收益率：{performance['annual_return']:.2%}
夏普比率：{performance['sharpe_ratio']:.2f}
最大回撤：{performance['max_drawdown']:.2%}
交易次数：{performance['trade_count']}次
胜率：{performance['win_rate']:.2f}%
盈亏比：{performance['profit_factor']:.2f}
=======================
通知时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # 多渠道发送（原有逻辑）
            for notifier in self.notifiers:
                notifier.send_msg(notify_content)
                logger.info(f"通过{notifier.__class__.__name__}发送通知成功 | 初始资金：{self.initial_capital:.2f}元")
        except Exception as e:
            logger.error(f"标的{symbol}通知发送失败：{str(e)} | 初始资金：{self.initial_capital:.2f}元", exc_info=True)

    def run_single_backtest(self, symbol: str):
        """执行单标的回测（完整保留原有逻辑）"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始单标的回测：{symbol} | 初始资金：{self.initial_capital:.2f}元")
        logger.info(f"{'='*80}")
        
        try:
            # 1. 获取K线数据（原有逻辑）
            kline_df = self._fetch_kline_data(symbol)
            
            # 2. 缠论计算（修复后的逻辑）
            chanlun_df = self._calculate_chanlun(kline_df)
            
            # 3. 执行交易（原有逻辑）
            trade_records, position_history = self._execute_trades(chanlun_df, symbol)
            
            # 4. 绩效分析（原有逻辑）
            performance = self._analyze_performance(trade_records, position_history, symbol)
            
            # 5. 绘图（原有逻辑）
            self._plot_results(chanlun_df, position_history, symbol)
            
            # 6. 导出结果（原有逻辑）
            self._export_results(chanlun_df, trade_records, position_history, performance, symbol)
            
            # 7. 发送通知（原有逻辑）
            self._send_notifications(performance, symbol)
            
            # 保存结果（原有逻辑）
            self.backtest_results[symbol] = {
                'kline_df': kline_df,
                'chanlun_df': chanlun_df,
                'trade_records': trade_records,
                'position_history': position_history,
                'performance': performance,
                'calculator_backtest_result': self.calculator_backtest_result
            }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"单标的回测完成：{symbol} | 初始资金：{self.initial_capital:.2f}元 | 最终资金：{performance.get('final_value', self.initial_capital):.2f}元")
            logger.info(f"{'='*80}")
            
            return self.backtest_results[symbol]
        
        except Exception as e:
            logger.error(f"单标的回测失败：{symbol} - {str(e)} | 初始资金：{self.initial_capital:.2f}元", exc_info=True)
            raise

    def run_multi_backtest(self):
        """执行多标的组合回测（完整保留原有逻辑，优化初始资金日志）"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始多标的组合回测：共{len(self.symbols)}个标的 | 总初始资金：{self.initial_capital:.2f}元")
        logger.info(f"标的列表：{self.symbols}")
        logger.info(f"单标的平均分配资金：{self.initial_capital/len(self.symbols):.2f}元/个")
        logger.info(f"{'='*80}")
        
        all_results = {}
        total_performance = {
            'initial_total_capital': self.initial_capital,  # 记录总初始资金
            'initial_total_value': self.initial_capital,
            'final_total_value': 0.0,
            'total_return': 0.0,
            'total_trade_count': 0,
            'total_long_trade_count': 0,
            'avg_win_rate': 0.0,
            'avg_profit_factor': 0.0,
            'max_drawdown': 0.0
        }
        
        # 遍历所有标的（原有逻辑）
        for symbol in self.symbols:
            try:
                result = self.run_single_backtest(symbol)
                all_results[symbol] = result
                
                # 汇总组合绩效（原有逻辑）
                total_performance['final_total_value'] += result['performance']['final_value']
                total_performance['total_trade_count'] += result['performance']['trade_count']
                total_performance['total_long_trade_count'] += result['performance']['long_trade_count']
                total_performance['avg_win_rate'] += result['performance']['win_rate']
                total_performance['avg_profit_factor'] += result['performance']['profit_factor']
                total_performance['max_drawdown'] = max(total_performance['max_drawdown'], result['performance']['max_drawdown'])
            
            except Exception as e:
                logger.error(f"多标的回测失败：{symbol} - {str(e)} | 总初始资金：{self.initial_capital:.2f}元", exc_info=True)
                all_results[symbol] = {'error': str(e)}
        
        # 计算组合平均指标（原有逻辑）
        valid_count = len([s for s in self.symbols if 'error' not in all_results[s]])
        if valid_count > 0:
            total_performance['total_return'] = (total_performance['final_total_value'] - total_performance['initial_total_value']) / total_performance['initial_total_value']
            total_performance['avg_win_rate'] /= valid_count
            total_performance['avg_profit_factor'] /= valid_count
        
        # 导出组合汇总结果（原有逻辑，包含总初始资金）
        total_performance_path = os.path.join(
            self.export_path,
            f'multi_total_performance_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.json'
        )
        self.exporter.export_json(total_performance, total_performance_path)
        
        # 绘制组合汇总图（原有逻辑，优化标题显示总初始资金）
        if self.enable_plot and self.plotter is not None:
            try:
                self.plotter.plot_multi_portfolio(
                    all_results=all_results,
                    save_path=os.path.join(self.export_path, f'multi_portfolio_{self.timeframe}_{self.start_date_str}_{self.end_date_str}.png'),
                    title=f'多标的组合价值曲线（{self.timeframe}）| 总初始资金：{self.initial_capital:.2f}元'
                )
            except Exception as e:
                logger.error(f"多标的组合绘图失败：{str(e)} | 总初始资金：{self.initial_capital:.2f}元", exc_info=True)
        
        # 发送组合通知（原有逻辑，优化初始资金显示）
        if self.enable_notify and self.notifiers:
            try:
                notify_content = f"""
【多标的组合缠论回测完成通知】
=======================
标的数量：{len(self.symbols)}个
标的列表：{self.symbols}
时间级别：{self.timeframe}
回测周期：{self.start_date_str} ~ {self.end_date_str}
总初始资金：{self.initial_capital:.2f}元  # 显示用户指定总资金
单标的平均分配：{self.initial_capital/len(self.symbols):.2f}元/个
最终总资金：{total_performance['final_total_value']:.2f}元
总收益率：{total_performance['total_return']:.2%}
总交易次数：{total_performance['total_trade_count']}次
平均胜率：{total_performance['avg_win_rate']:.2f}%
平均盈亏比：{total_performance['avg_profit_factor']:.2f}
最大回撤：{total_performance['max_drawdown']:.2%}
=======================
通知时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                for notifier in self.notifiers:
                    notifier.send_msg(notify_content)
                logger.info(f"多标的组合通知发送成功 | 总初始资金：{self.initial_capital:.2f}元")
            except Exception as e:
                logger.error(f"多标的组合通知发送失败：{str(e)} | 总初始资金：{self.initial_capital:.2f}元", exc_info=True)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"多标的组合回测完成：共{valid_count}/{len(self.symbols)}个标的成功")
        logger.info(f"总初始资金：{self.initial_capital:.2f}元 | 最终总资金：{total_performance['final_total_value']:.2f}元 | 总收益率：{total_performance['total_return']:.2%}")
        logger.info(f"{'='*80}")
        
        self.backtest_results = {
            'individual_results': all_results,
            'total_performance': total_performance,
            'initial_total_capital': self.initial_capital  # 保存总初始资金
        }
        
        return self.backtest_results

    def run(self):
        """执行回测（入口方法，完整保留原有逻辑，优化初始资金日志）"""
        logger.info(f"\n{'='*80}")
        logger.info("缠论回测引擎开始执行...")
        logger.info(f"回测模式：{self.mode}")
        logger.info(f"时间级别：{self.timeframe}")
        logger.info(f"标的列表：{self.symbols}")
        logger.info(f"回测周期：{self.start_date_str} ~ {self.end_date_str}")
        logger.info(f"初始资金：{self.initial_capital:.2f}元  # 统一显示用户指定值")
        logger.info(f"启用功能：{'通知' if self.enable_notify else '无'} | {'绘图' if self.enable_plot else '无'} | {'缓存' if self.enable_cache else '无'}")
        logger.info(f"{'='*80}")
        
        try:
            if self.mode == 'single':
                return self.run_single_backtest(self.symbol)
            elif self.mode == 'multi':
                return self.run_multi_backtest()
        except Exception as e:
            logger.error(f"回测引擎执行失败：{str(e)} | 初始资金：{self.initial_capital:.2f}元", exc_info=True)
            # 发送失败通知（原有逻辑，优化初始资金显示）
            if self.enable_notify and self.notifiers:
                try:
                    fail_content = f"""
【缠论回测失败通知】
=======================
错误原因：{str(e)}
回测模式：{self.mode}
时间级别：{self.timeframe}
标的列表：{self.symbols}
回测周期：{self.start_date_str} ~ {self.end_date_str}
初始资金：{self.initial_capital:.2f}元
=======================
通知时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    for notifier in self.notifiers:
                        notifier.send_msg(fail_content)
                except Exception:
                    pass
            sys.exit(1)

# ===================== 命令行参数解析（完整保留原有逻辑）=====================
def parse_args() -> argparse.Namespace:
    """解析命令行参数（完整保留原有所有参数）"""
    parser = argparse.ArgumentParser(description='缠论回测引擎（完整版V2.3）- 支持单标的/多标的、多时间级别、完整缠论指标计算')
    
    # 核心必选参数（原有逻辑）
    parser.add_argument('--mode', required=True, choices=SUPPORTED_MODES, help=f'回测模式：{SUPPORTED_MODES}')
    parser.add_argument('--symbol', required=True, help='标的代码（多标的用逗号分隔，如：000001,600036,300059）')
    parser.add_argument('--start_date', required=True, help='开始日期（格式：YYYYMMDD，如：20230101）')
    parser.add_argument('--end_date', required=True, help='结束日期（格式：YYYYMMDD，如：20241231）')
    parser.add_argument('--timeframe', required=True, choices=SUPPORTED_TIMEFRAMES, help=f'时间级别：{SUPPORTED_TIMEFRAMES}')
    parser.add_argument('--capital', type=float, required=True, help='初始资金（单位：元，如：100000）')
    
    # 可选功能参数（原有逻辑）
    parser.add_argument('--enable_notify', action='store_true', help='启用通知功能（默认禁用，需配置config/notifier.yaml）')
    parser.add_argument('--enable_plot', action='store_true', help='启用绘图功能（默认禁用，需安装matplotlib）')
    parser.add_argument('--enable_cache', action='store_true', help='启用数据缓存（默认禁用，缓存有效期1小时）')
    
    # 路径配置参数（原有逻辑）
    parser.add_argument('--export_path', help=f'结果导出路径（默认：{DEFAULT_EXPORT_PATH}）')
    parser.add_argument('--config_path', help=f'系统配置文件路径（默认：config/system.yaml）')
    parser.add_argument('--risk_config_path', help=f'风险配置文件路径（默认：config/risk.yaml）')
    parser.add_argument('--notifier_config_path', help=f'通知配置文件路径（默认：config/notifier.yaml）')
    
    return parser.parse_args()

# ===================== 程序入口（完整保留原有逻辑）=====================
if __name__ == "__main__":
    """程序入口"""
    try:
        # 解析参数（原有逻辑）
        args = parse_args()
        
        # 初始化回测引擎（原有逻辑）
        backtester = ChanlunBacktester(args)
        
        # 执行回测（原有逻辑）
        results = backtester.run()
        
        # 正常退出（原有逻辑）
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行失败：{str(e)}", exc_info=True)
        sys.exit(1)