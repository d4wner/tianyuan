#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论回测系统 - 修复版本
修复了日期范围不正确、符号验证、导入错误和配置KeyError问题
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入依赖模块（彻底移除validate_trading_date）
try:
    from src.config import load_config, save_config
    from src.data_fetcher import StockDataAPI, DataFetchError
    from src.calculator import ChanlunCalculator
    from src.notifier import DingdingNotifier
    from src.utils import get_last_trading_day, is_trading_hour, get_valid_date_range_str
    from src.reporter import generate_pre_market_report, generate_daily_report
    from src.exporter import ChanlunExporter
    from src.plotter import ChanlunPlotter
except ImportError as e:
    logging.error(f"导入依赖模块失败: {e}")
    raise

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('ChanlunBacktest')

class BacktestEngine:
    """缠论回测引擎核心类 - 修复日期范围和符号验证问题"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化回测引擎
        :param config: 系统配置
        """
        self.config = config
        self.data_api = None
        self.calculator = None
        self.notifier = None
        self.plotter = None
        self.exporter = None
        
        self._initialize_components()
        logger.info("缠论回测引擎初始化完成")
    
    def _initialize_components(self):
        """初始化所有组件（配置访问使用get方法）"""
        # 初始化数据API
        data_fetcher_config = self.config.get('data_fetcher', {})
        self.data_api = StockDataAPI(
            max_retries=data_fetcher_config.get('max_retries', 3),
            timeout=data_fetcher_config.get('timeout', 10)
        )
        
        # 初始化计算器
        chanlun_config = self.config.get('chanlun', {})
        self.calculator = ChanlunCalculator(chanlun_config)
        
        # 初始化通知器（兼容无notifications配置）
        self.notifier = DingdingNotifier(self.config)
        
        # 初始化绘图器
        plotter_config = self.config.get('plotter', {})
        self.plotter = ChanlunPlotter(plotter_config)
        
        # 初始化数据导出器
        self.exporter = ChanlunExporter(self.config.get('exporter', {}))
    
    def run_comprehensive_backtest(self, symbol: str, start_date: str, end_date: str, 
                                  timeframe: str = 'weekly', initial_capital: float = 100000) -> Dict[str, Any]:
        """
        运行全面回测 - 修复日期范围问题
        :param symbol: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param timeframe: 时间级别
        :param initial_capital: 初始资金
        :return: 完整回测结果
        """
        # 增强日志：记录原始日期参数
        logger.info(f"开始全面回测: {symbol} {timeframe}")
        logger.info(f"用户指定日期范围: {start_date} 至 {end_date}")
        logger.info(f"初始资金: {initial_capital}")
        
        try:
            # 防御性检查：验证symbol不是DataFrame
            self._validate_symbol_not_dataframe(symbol)
            
            # 1. 数据获取阶段 - 修复日期处理
            data_result = self._acquire_and_validate_data(symbol, start_date, end_date, timeframe)
            if not data_result['success']:
                error_msg = f"数据获取失败: {data_result['error']}"
                logger.error(error_msg)
                return self._create_error_result(initial_capital, error_msg)
            
            df = data_result['data']
            actual_start = df['date'].min().strftime('%Y-%m-%d')
            actual_end = df['date'].max().strftime('%Y-%m-%d')
            logger.info(f"数据获取成功: {len(df)}条记录, 实际日期范围: {actual_start} 至 {actual_end}")
            
            # 2. 缠论计算阶段
            calculation_result = self._perform_chanlun_calculation(df, timeframe)
            if not calculation_result['success']:
                error_msg = f"缠论计算失败: {calculation_result['error']}"
                logger.error(error_msg)
                return self._create_error_result(initial_capital, error_msg)
            
            calculated_df = calculation_result['data']
            
            # 3. 回测执行阶段
            backtest_result = self._execute_backtest(calculated_df, initial_capital, timeframe)
            if not backtest_result['success']:
                error_msg = f"回测执行失败: {backtest_result['error']}"
                logger.error(error_msg)
                return self._create_error_result(initial_capital, error_msg)
            
            result = backtest_result['data']
            
            # 4. 报告生成阶段
            report_result = self._generate_comprehensive_report(result, symbol, timeframe)
            result['report'] = report_result
            
            # 5. 图表生成阶段（兼容无plotter配置）
            if self.config.get('plotter', {}).get('enabled', False):
                chart_result = self._generate_detailed_charts(result, symbol, timeframe)
            else:
                chart_result = {'success': False, 'error': '图表生成未启用'}
            result['charts'] = chart_result
            
            # 6. 通知发送阶段（兼容无notifications配置）
            if self.config.get('notifications', {}).get('enabled', False):
                self._send_notifications(result, symbol, timeframe)
            
            # 记录实际使用的日期范围
            result['actual_date_range'] = {
                'start': actual_start,
                'end': actual_end,
                'requested_start': start_date,
                'requested_end': end_date
            }
            
            logger.info(f"全面回测完成: 总回报{result.get('return_percent', 0):.2f}%, 实际日期范围: {actual_start} 至 {actual_end}")
            return result
            
        except Exception as e:
            error_msg = f"回测过程异常: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(initial_capital, error_msg)
    
    def _validate_symbol_not_dataframe(self, symbol: Any):
        """
        优化的符号验证方法：确保symbol是有效的股票代码而非DataFrame或其他无效类型
        核心优化：更精准的类型检测、更友好的错误提示、更全面的格式验证
        :param symbol: 要检查的符号
        """
        # 检查是否为None
        if symbol is None:
            logger.critical("股票代码参数为None，无法执行回测")
            raise ValueError("股票代码不能为None")
        
        # 直接检查是否为Pandas DataFrame或Series（核心修复点）
        if isinstance(symbol, pd.DataFrame):
            logger.critical(f"检测到Pandas DataFrame作为股票代码，数据形状: {symbol.shape}")
            raise ValueError("无效股票代码类型: 不能将DataFrame对象作为股票代码传递")
        elif isinstance(symbol, pd.Series):
            logger.critical(f"检测到Pandas Series作为股票代码，数据长度: {len(symbol)}")
            raise ValueError("无效股票代码类型: 不能将Series对象作为股票代码传递")
        
        # 转换为字符串并清理（处理非字符串输入）
        try:
            symbol_str = str(symbol).strip()
        except Exception as e:
            logger.critical(f"无法将股票代码转换为字符串，输入类型: {type(symbol)}，错误: {str(e)}")
            raise ValueError(f"股票代码格式无效，无法转换为字符串: {str(e)}")
        
        # 检查字符串长度是否合理（正常股票代码不会超过20字符）
        if len(symbol_str) > 20:
            logger.critical(f"股票代码过长({len(symbol_str)}字符)，疑似无效输入: {symbol_str[:50]}...")
            raise ValueError(f"股票代码过长（超过20字符），可能是误传的DataFrame/Series字符串表示")
        
        # 检查是否包含DataFrame相关特征关键词（不区分大小写，精准匹配）
        dataframe_indicators = [
            'DataFrame', 'Series', 'open', 'high', 'low', 'close', 'volume', 'date',
            'timestamp', 'adj_close', 'amount', 'turnover', 'pe', 'pb'
        ]
        matched_indicators = [ind for ind in dataframe_indicators if ind.lower() in symbol_str.lower()]
        if matched_indicators:
            logger.critical(f"股票代码包含DataFrame特征关键词: {matched_indicators}，输入值: {symbol_str[:200]}")
            raise ValueError(f"无效股票代码: 包含'{matched_indicators[0]}'等数据列名或Pandas对象关键词")
        
        # 检查是否为有效的股票代码格式（支持A股、港股、美股常见格式）
        pattern = r'^([a-zA-Z]{2})?(\d{5,6}|\w{1,5})(\.[A-Za-z]{2})?$'
        if not re.match(pattern, symbol_str):
            logger.warning(
                f"股票代码格式不标准: {symbol_str}\n"
                f"建议格式：\n"
                f"- A股: 000001 / sh000001 / 000001.SH\n"
                f"- 港股: HK00700 / 00700.HK\n"
                f"- 美股: AAPL / AAPL.US"
            )
    
    def _acquire_and_validate_data(self, symbol: str, start_date: str, end_date: str, 
                                  timeframe: str) -> Dict[str, Any]:
        """
        获取并验证数据 - 修复日期范围问题
        :return: 包含成功状态和数据的结果字典
        """
        try:
            logger.info(f"数据获取阶段 - 符号: {symbol}, 时间级别: {timeframe}")
            logger.info(f"请求日期范围: {start_date} 至 {end_date}")
            
            # 简单日期格式验证（不依赖外部函数）
            def parse_simple_date(date_str: str) -> datetime:
                """简单日期解析（支持YYYYMMDD和YYYY-MM-DD）"""
                date_str = str(date_str).strip()
                try:
                    if len(date_str) == 8 and date_str.isdigit():
                        return datetime.strptime(date_str, '%Y%m%d')
                    else:
                        return datetime.strptime(date_str, '%Y-%m-%d')
                except Exception:
                    raise ValueError(f"日期格式错误: {date_str}，支持YYYYMMDD或YYYY-MM-DD")
            
            # 解析并验证日期范围
            try:
                start_dt = parse_simple_date(start_date)
                end_dt = parse_simple_date(end_date)
            except ValueError as e:
                logger.error(f"日期解析失败: {str(e)}")
                return {'success': False, 'error': str(e)}
            
            if start_dt >= end_dt:
                logger.error(f"日期范围无效: 开始日期{start_date} >= 结束日期{end_date}")
                return {'success': False, 'error': '开始日期不能大于等于结束日期'}
            
            # 限制最大回测周期
            max_days = self.config.get('backtest', {}).get('max_period_days', 365*5)
            if (end_dt - start_dt).days > max_days:
                logger.warning(f"回测周期过长（{max_days}天限制），自动截断为最近{max_days}天")
                start_dt = end_dt - timedelta(days=max_days)
            
            # 格式化为YYYY-MM-DD（适配数据源）
            start_date_str = start_dt.strftime('%Y-%m-%d')
            end_date_str = end_dt.strftime('%Y-%m-%d')
            
            # 根据时间级别获取数据
            if timeframe == 'weekly':
                df = self.data_api.get_weekly_data(symbol, start_date_str, end_date_str)
            elif timeframe == 'daily':
                df = self.data_api.get_daily_data(symbol, start_date_str, end_date_str)
            elif timeframe == 'minute':
                minute_days = self.config.get('data_fetcher', {}).get('minute_days', 30)
                df = self.data_api.get_minute_data(symbol, '5m', minute_days)
            else:
                return {'success': False, 'error': f'不支持的时间级别: {timeframe}，支持weekly/daily/minute'}
            
            # 验证数据质量
            if df.empty:
                logger.warning(f"获取的数据为空 - 符号: {symbol}, 日期范围: {start_date_str}至{end_date_str}")
                return {'success': False, 'error': '获取的数据为空，请检查股票代码或日期范围'}
            
            if len(df) < 10:
                logger.warning(f"数据点数不足: {len(df)}条（至少需要10个数据点）")
                return {'success': False, 'error': f'数据点数不足，仅获取到{len(df)}条，至少需要10个数据点'}
            
            # 检查必要列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"缺失必要数据列: {missing_columns}")
                return {'success': False, 'error': f'缺失必要数据列: {missing_columns}，必须包含open/high/low/close/volume'}
            
            # 处理日期列
            if 'date' not in df.columns:
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    df = df.rename(columns={'timestamp': 'datetime'})
                    logger.info("数据列转换：timestamp -> datetime，新增date列（日期）")
                else:
                    logger.error("数据中没有日期列（date）或时间戳列（timestamp）")
                    return {'success': False, 'error': '数据中没有日期列或时间戳列，无法进行回测'}
            
            # 数据排序和去重
            df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
            df = df.reset_index(drop=True)
            
            logger.info(f"数据预处理完成: {len(df)}条有效记录")
            return {'success': True, 'data': df}
            
        except DataFetchError as e:
            logger.error(f"数据获取失败（数据源异常）: {str(e)}")
            return {'success': False, 'error': f'数据源异常: {str(e)}'}
        except Exception as e:
            logger.error(f"数据获取过程异常: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _perform_chanlun_calculation(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """执行缠论计算（包含分型、笔、线段、中枢识别）"""
        try:
            logger.info(f"开始缠论计算 - 时间级别: {timeframe}，数据量: {len(df)}条")
            
            # 根据时间级别加载对应的缠论参数
            chanlun_params = self.config.get('chanlun', {}).get(timeframe, {})
            if not chanlun_params:
                chanlun_params = self.config.get('chanlun', {}).get('default', {})
                logger.warning(f"未配置{timeframe}级别缠论参数，使用默认参数: {chanlun_params}")
            
            # 执行缠论计算
            result_df = self.calculator.calculate(
                df,
                timeframe=timeframe,
                fractal_sensitivity=chanlun_params.get('fractal_sensitivity', 3),
                pen_min_length=chanlun_params.get('pen_min_length', 5),
                segment_min_length=chanlun_params.get('segment_min_length', 3),
                central_bank_min_length=chanlun_params.get('central_bank_min_length', 5)
            )
            
            # 验证计算结果
            required_calc_columns = ['top_fractal', 'bottom_fractal', 'pen_type', 'segment_type', 'central_bank']
            missing_calc_cols = [col for col in required_calc_columns if col not in result_df.columns]
            if missing_calc_cols:
                logger.warning(f"缠论计算缺失部分列: {missing_calc_cols}")
            
            logger.info("缠论计算完成")
            return {'success': True, 'data': result_df}
        except Exception as e:
            logger.error(f"缠论计算失败: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _execute_backtest(self, df: pd.DataFrame, initial_capital: float, timeframe: str) -> Dict[str, Any]:
        """执行回测逻辑（基于缠论信号的交易策略）"""
        try:
            logger.info(f"开始回测执行 - 初始资金: {initial_capital:.2f}，时间级别: {timeframe}")
            
            # 初始化回测参数（全部使用get方法，避免KeyError）
            backtest_config = self.config.get('backtest', {})
            backtest_params = {
                'initial_capital': initial_capital,
                'slippage': backtest_config.get('slippage', 0.001),  # 滑点率 0.1%
                'transaction_cost': backtest_config.get('transaction_cost', 0.0003),  # 交易成本 0.03%
                'max_position': backtest_config.get('max_single_position', 0.5),  # 单只股票最大仓位 50%
                'stop_loss_ratio': backtest_config.get('stop_loss_ratio', 0.05),  # 止损比例 5%
                'take_profit_ratio': backtest_config.get('take_profit_ratio', 0.1),  # 止盈比例 10%
                'signal_type': backtest_config.get('signal_type', 'pen_segment_central_bank'),
                'min_holding_period': backtest_config.get('min_holding_period', 1)
            }
            
            # 调用计算器的回测方法
            result = self.calculator.backtest(df, backtest_params, timeframe)
            
            # 验证回测结果完整性
            required_result_fields = [
                'equity_curve', 'drawdown', 'return_percent', 'max_drawdown',
                'sharpe_ratio', 'win_rate', 'total_trades', 'profit_factor',
                'volatility', 'downside_risk', 'sortino_ratio', 'calmar_ratio',
                'avg_holding_period', 'max_holding_period', 'monthly_trades',
                'trades', 'price_data'
            ]
            missing_fields = [field for field in required_result_fields if field not in result]
            if missing_fields:
                logger.warning(f"回测结果缺失部分字段: {missing_fields}")
                # 补充缺失字段的默认值
                for field in missing_fields:
                    if field == 'equity_curve':
                        result[field] = pd.Series([initial_capital] * len(df), index=df.index)
                    elif field == 'drawdown':
                        result[field] = pd.Series([0.0] * len(df), index=df.index)
                    elif field.endswith('_percent'):
                        result[field] = 0.0
                    elif field.endswith('_ratio'):
                        result[field] = 0.0
                    elif field.endswith('_trades'):
                        result[field] = 0
                    elif field in ['trades', 'price_data']:
                        result[field] = pd.DataFrame() if field != 'trades' else []
            
            logger.info(
                f"回测执行完成 - 总交易次数: {result.get('total_trades', 0)}, "
                f"总回报: {result.get('return_percent', 0):.2f}%, "
                f"最大回撤: {result.get('max_drawdown', 0):.2f}%"
            )
            return {'success': True, 'data': result}
        except Exception as e:
            logger.error(f"回测执行失败: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _generate_comprehensive_report(self, result: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        """生成综合回测报告（包含性能、风险、交易活动分析）"""
        try:
            logger.info(f"生成综合回测报告 - 股票: {symbol}, 时间级别: {timeframe}")
            
            # 提取核心指标
            performance = {
                'return_percent': round(result.get('return_percent', 0), 2),
                'max_drawdown': round(result.get('max_drawdown', 0), 2),
                'sharpe_ratio': round(result.get('sharpe_ratio', 0), 2),
                'win_rate': round(result.get('win_rate', 0) * 100, 2),
                'total_trades': result.get('total_trades', 0),
                'profit_factor': round(result.get('profit_factor', 0), 2),
                'expectancy': round(result.get('expectancy', 0), 2),
                'avg_profit_per_trade': round(result.get('avg_profit_per_trade', 0), 2),
                'avg_loss_per_trade': round(result.get('avg_loss_per_trade', 0), 2)
            }
            
            risk_metrics = {
                'volatility': round(result.get('volatility', 0) * 100, 2),
                'downside_risk': round(result.get('downside_risk', 0) * 100, 2),
                'sortino_ratio': round(result.get('sortino_ratio', 0), 2),
                'calmar_ratio': round(result.get('calmar_ratio', 0), 2),
                'value_at_risk': round(result.get('value_at_risk', 0), 2),
                'conditional_value_at_risk': round(result.get('conditional_value_at_risk', 0), 2)
            }
            
            trading_activity = {
                'avg_holding_period': round(result.get('avg_holding_period', 0), 1),
                'max_holding_period': result.get('max_holding_period', 0),
                'min_holding_period': result.get('min_holding_period', 0),
                'monthly_trades': result.get('monthly_trades', {}),
                'win_streak': result.get('win_streak', 0),
                'lose_streak': result.get('lose_streak', 0),
                'long_trades_count': result.get('long_trades_count', 0),
                'short_trades_count': result.get('short_trades_count', 0)
            }
            
            # 生成报告主体
            report = {
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': result['actual_date_range'].get('start', 'N/A'),
                    'end_date': result['actual_date_range'].get('end', 'N/A'),
                    'initial_capital': result.get('initial_capital', 100000),
                    'final_capital': round(result.get('final_value', result.get('initial_capital', 100000)), 2),
                    'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'performance': performance,
                'risk_metrics': risk_metrics,
                'trading_activity': trading_activity,
                'strategy_params': self.config.get('backtest', {}),
                'chanlun_params': self.config.get('chanlun', {}).get(timeframe, self.config.get('chanlun', {}).get('default', {})),
                'summary': self._generate_report_summary(performance, risk_metrics)
            }
            
            # 导出报告（如果启用）
            exporter_config = self.config.get('exporter', {})
            if exporter_config.get('enabled', False):
                export_formats = exporter_config.get('formats', ['json', 'csv'])
                export_path = self.exporter.export_report(
                    report, 
                    symbol=symbol, 
                    timeframe=timeframe,
                    formats=export_formats,
                    output_dir=exporter_config.get('output_dir', 'outputs/reports')
                )
                report['export_info'] = {
                    'path': export_path,
                    'formats': export_formats
                }
                logger.info(f"回测报告已导出至: {export_path}")
            
            logger.info("综合回测报告生成完成")
            return report
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}", exc_info=True)
            return {'error': str(e), 'partial_report': {}}
    
    def _generate_report_summary(self, performance: Dict[str, Any], risk_metrics: Dict[str, Any]) -> str:
        """生成报告摘要（自然语言描述）"""
        try:
            return_percent = performance['return_percent']
            max_drawdown = performance['max_drawdown']
            win_rate = performance['win_rate']
            total_trades = performance['total_trades']
            sharpe_ratio = performance['sharpe_ratio']
            
            summary_parts = []
            
            # 收益总结
            if return_percent > 50:
                summary_parts.append(f"总回报率{return_percent}%，表现优秀")
            elif return_percent > 10:
                summary_parts.append(f"总回报率{return_percent}%，表现良好")
            elif return_percent > 0:
                summary_parts.append(f"总回报率{return_percent}%，表现一般")
            else:
                summary_parts.append(f"总回报率{return_percent}%，表现不佳")
            
            # 风险总结
            if max_drawdown < 10:
                summary_parts.append(f"最大回撤{max_drawdown}%，风险控制优秀")
            elif max_drawdown < 20:
                summary_parts.append(f"最大回撤{max_drawdown}%，风险控制良好")
            else:
                summary_parts.append(f"最大回撤{max_drawdown}%，风险较高")
            
            # 交易频率总结
            if total_trades == 0:
                summary_parts.append("未产生任何交易")
            elif total_trades < 10:
                summary_parts.append(f"共执行{total_trades}笔交易，交易频率较低")
            elif total_trades < 50:
                summary_parts.append(f"共执行{total_trades}笔交易，交易频率适中")
            else:
                summary_parts.append(f"共执行{total_trades}笔交易，交易频率较高")
            
            # 胜率总结
            if win_rate > 60:
                summary_parts.append(f"胜率{win_rate}%，策略准确性较高")
            elif win_rate > 50:
                summary_parts.append(f"胜率{win_rate}%，策略准确性良好")
            else:
                summary_parts.append(f"胜率{win_rate}%，策略准确性一般")
            
            # 夏普比率总结
            if sharpe_ratio > 2:
                summary_parts.append(f"夏普比率{sharpe_ratio}，风险调整后收益优秀")
            elif sharpe_ratio > 1:
                summary_parts.append(f"夏普比率{sharpe_ratio}，风险调整后收益良好")
            else:
                summary_parts.append(f"夏普比率{sharpe_ratio}，风险调整后收益一般")
            
            return "，".join(summary_parts) + "。"
        except Exception as e:
            logger.error(f"生成报告摘要失败: {str(e)}")
            return "报告摘要生成失败，详细数据请查看完整报告。"
    
    def _generate_detailed_charts(self, result: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        """生成详细的回测图表（资金曲线、最大回撤、交易信号、缠论结构）"""
        try:
            logger.info(f"生成回测图表 - 股票: {symbol}, 时间级别: {timeframe}")
            
            # 创建图表保存目录
            chart_config = self.config.get('plotter', {})
            base_chart_dir = chart_config.get('output_dir', 'outputs/charts')
            chart_dir = os.path.join(base_chart_dir, timeframe, symbol)
            os.makedirs(chart_dir, exist_ok=True)
            
            # 1. 资金曲线图表
            equity_curve_path = self.plotter.plot_equity_curve(
                equity_curve=result['equity_curve'],
                benchmark_curve=result.get('benchmark_curve'),
                save_path=os.path.join(chart_dir, f'{symbol}_equity_curve.png'),
                title=f'{symbol} {timeframe} 资金曲线',
                xlabel='日期',
                ylabel='资产价值（元）'
            )
            
            # 2. 最大回撤图表
            drawdown_path = self.plotter.plot_drawdown(
                drawdown=result['drawdown'],
                save_path=os.path.join(chart_dir, f'{symbol}_drawdown.png'),
                title=f'{symbol} {timeframe} 最大回撤',
                xlabel='日期',
                ylabel='回撤比例（%）'
            )
            
            # 3. 交易信号图表（价格+信号+仓位）
            signals_path = self.plotter.plot_signals(
                price_data=result['price_data'],
                trades=result['trades'],
                positions=result.get('positions'),
                save_path=os.path.join(chart_dir, f'{symbol}_trading_signals.png'),
                title=f'{symbol} {timeframe} 交易信号',
                xlabel='日期',
                ylabel='价格（元）'
            )
            
            # 4. 缠论结构图表（K线+分型+笔+线段+中枢）
            chanlun_path = self.plotter.plot_chanlun_structure(
                price_data=result['price_data'],
                save_path=os.path.join(chart_dir, f'{symbol}_chanlun_structure.png'),
                title=f'{symbol} {timeframe} 缠论结构',
                xlabel='日期',
                ylabel='价格（元）'
            )
            
            # 5. 性能指标雷达图
            radar_path = self.plotter.plot_performance_radar(
                performance=result['report']['performance'],
                risk_metrics=result['report']['risk_metrics'],
                save_path=os.path.join(chart_dir, f'{symbol}_performance_radar.png'),
                title=f'{symbol} {timeframe} 性能雷达图'
            )
            
            logger.info(f"回测图表生成完成，保存目录: {chart_dir}")
            return {
                'success': True,
                'chart_dir': chart_dir,
                'equity_curve_path': equity_curve_path,
                'drawdown_path': drawdown_path,
                'signals_path': signals_path,
                'chanlun_structure_path': chanlun_path,
                'performance_radar_path': radar_path
            }
        except Exception as e:
            logger.error(f"图表生成失败: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e), 'chart_dir': None}
    
    def _send_notifications(self, result: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """发送回测结果通知（钉钉）"""
        try:
            logger.info(f"发送回测结果通知 - 股票: {symbol}")
            
            # 提取核心信息
            return_percent = result.get('return_percent', 0)
            max_drawdown = result.get('max_drawdown', 0)
            total_trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0) * 100
            actual_date_range = result.get('actual_date_range', {})
            summary = result['report'].get('summary', '')
            
            # 构建通知内容
            content = (
                f"📊 缠论回测结果通知\n"
                f"=======================\n"
                f"股票代码: {symbol}\n"
                f"时间级别: {timeframe}\n"
                f"日期范围: {actual_date_range.get('start', 'N/A')} 至 {actual_date_range.get('end', 'N/A')}\n"
                f"初始资金: {result.get('initial_capital', 100000):,.2f}元\n"
                f"最终资金: {result.get('final_value', result.get('initial_capital', 100000)):,.2f}元\n"
                f"总回报率: {return_percent:.2f}%\n"
                f"最大回撤: {max_drawdown:.2f}%\n"
                f"交易次数: {total_trades}次\n"
                f"胜率: {win_rate:.2f}%\n"
                f"夏普比率: {result.get('sharpe_ratio', 0):.2f}\n"
                f"=======================\n"
                f"📝 总结: {summary}\n"
                f"📁 详细报告: {result['report'].get('export_info', {}).get('path', '未导出')}"
            )
            
            # 发送文本通知
            self.notifier.send_text(content)
            
            # 发送图表（如果生成成功）
            if result.get('charts', {}).get('success', False):
                chart_paths = [
                    result['charts']['equity_curve_path'],
                    result['charts']['signals_path'],
                    result['charts']['chanlun_structure_path']
                ]
                # 过滤不存在的图表路径
                valid_chart_paths = [path for path in chart_paths if path and os.path.exists(path)]
                if valid_chart_paths:
                    self.notifier.send_images(valid_chart_paths)
                    logger.info(f"已发送{len(valid_chart_paths)}张图表到钉钉")
            
            logger.info("回测结果通知发送完成")
        except Exception as e:
            logger.error(f"通知发送失败: {str(e)}", exc_info=True)
    
    def _create_error_result(self, initial_capital: float, error_msg: str) -> Dict[str, Any]:
        """创建错误结果对象（统一错误返回格式）"""
        return {
            'success': False,
            'error': error_msg,
            'initial_capital': initial_capital,
            'final_value': initial_capital,
            'return_percent': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'volatility': 0.0,
            'downside_risk': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'avg_holding_period': 0.0,
            'max_holding_period': 0,
            'monthly_trades': {},
            'trades': [],
            'price_data': pd.DataFrame(),
            'equity_curve': pd.Series(),
            'drawdown': pd.Series(),
            'actual_date_range': {},
            'report': {'error': error_msg, 'partial_report': {}},
            'charts': {'success': False, 'error': error_msg}
        }
    
    def batch_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                      timeframe: str = 'weekly', initial_capital: float = 100000) -> Dict[str, Any]:
        """批量回测多个股票"""
        logger.info(f"开始批量回测 - 标的数量: {len(symbols)}, 时间级别: {timeframe}, 初始资金: {initial_capital:.2f}元")
        
        # 初始化批量回测结果
        batch_results = {
            'metadata': {
                'batch_id': datetime.now().strftime('%Y%m%d%H%M%S'),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe,
                'initial_capital_per_symbol': initial_capital,
                'total_symbols': len(symbols),
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': None
            },
            'success_count': 0,
            'fail_count': 0,
            'results': {},
            'summary': {
                'avg_return': 0.0,
                'median_return': 0.0,
                'max_return': -float('inf'),
                'min_return': float('inf'),
                'best_symbol': None,
                'worst_symbol': None,
                'avg_max_drawdown': 0.0,
                'avg_win_rate': 0.0,
                'avg_trades_count': 0.0,
                'profitable_symbols_count': 0,
                'profitable_ratio': 0.0
            }
        }
        
        # 逐个执行回测
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n===== 批量回测进度: {i}/{len(symbols)} - 股票: {symbol} =====")
            try:
                # 执行单只股票回测
                single_result = self.run_comprehensive_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    initial_capital=initial_capital
                )
                
                batch_results['results'][symbol] = single_result
                
                # 统计成功/失败
                if single_result.get('success', False):
                    batch_results['success_count'] += 1
                    
                    # 提取关键指标用于汇总
                    return_percent = single_result.get('return_percent', 0)
                    max_drawdown = single_result.get('max_drawdown', 0)
                    win_rate = single_result.get('win_rate', 0)
                    total_trades = single_result.get('total_trades', 0)
                    
                    # 更新汇总统计
                    batch_results['summary']['avg_return'] += return_percent
                    batch_results['summary']['avg_max_drawdown'] += max_drawdown
                    batch_results['summary']['avg_win_rate'] += win_rate
                    batch_results['summary']['avg_trades_count'] += total_trades
                    
                    # 更新最值
                    if return_percent > batch_results['summary']['max_return']:
                        batch_results['summary']['max_return'] = return_percent
                        batch_results['summary']['best_symbol'] = symbol
                    if return_percent < batch_results['summary']['min_return']:
                        batch_results['summary']['min_return'] = return_percent
                        batch_results['summary']['worst_symbol'] = symbol
                    # 统计盈利标的
                    if return_percent > 0:
                        batch_results['summary']['profitable_symbols_count'] += 1
                else:
                    batch_results['fail_count'] += 1
                    logger.error(f"批量回测 {symbol} 失败: {single_result.get('error', '未知错误')}")
            
            except Exception as e:
                logger.error(f"批量回测 {symbol} 异常: {str(e)}", exc_info=True)
                batch_results['results'][symbol] = {
                    'success': False,
                    'error': str(e),
                    'initial_capital': initial_capital,
                    'final_value': initial_capital
                }
                batch_results['fail_count'] += 1
        
        # 计算平均指标
        total_success = batch_results['success_count']
        if total_success > 0:
            batch_results['summary']['avg_return'] /= total_success
            batch_results['summary']['avg_max_drawdown'] /= total_success
            batch_results['summary']['avg_win_rate'] /= total_success
            batch_results['summary']['avg_trades_count'] /= total_success
            batch_results['summary']['profitable_ratio'] = (batch_results['summary']['profitable_symbols_count'] / total_success) * 100
        
        # 计算中位数回报
        return_list = [
            res.get('return_percent', 0) 
            for res in batch_results['results'].values() 
            if res.get('success', False)
        ]
        if return_list:
            batch_results['summary']['median_return'] = np.median(return_list)
        else:
            batch_results['summary']['median_return'] = 0.0
        
        # 补充结束时间
        batch_results['metadata']['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存批量回测结果
        batch_report_dir = self.config.get('exporter', {}).get('batch_report_dir', 'outputs/reports/batch')
        os.makedirs(batch_report_dir, exist_ok=True)
        batch_report_path = os.path.join(
            batch_report_dir,
            f'batch_backtest_{timeframe}_{batch_results["metadata"]["batch_id"]}.json'
        )
        with open(batch_report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        batch_results['report_path'] = batch_report_path
        
        # 发送批量回测摘要通知
        if self.config.get('notifications', {}).get('enabled', False):
            self._send_batch_backtest_notification(batch_results)
        
        logger.info(
            f"\n===== 批量回测完成 =====\n"
            f"总标的数: {len(symbols)}\n"
            f"成功: {batch_results['success_count']}个\n"
            f"失败: {batch_results['fail_count']}个\n"
            f"平均回报率: {batch_results['summary']['avg_return']:.2f}%\n"
            f"最高回报率: {batch_results['summary']['max_return']:.2f}% ({batch_results['summary']['best_symbol']})\n"
            f"最低回报率: {batch_results['summary']['min_return']:.2f}% ({batch_results['summary']['worst_symbol']})\n"
            f"盈利标的比例: {batch_results['summary']['profitable_ratio']:.2f}%\n"
            f"报告保存路径: {batch_report_path}"
        )
        
        return batch_results
    
    def _send_batch_backtest_notification(self, batch_results: Dict[str, Any]) -> None:
        """发送批量回测摘要通知"""
        try:
            summary = batch_results['summary']
            metadata = batch_results['metadata']
            
            content = (
                f"📊 批量缠论回测完成通知\n"
                f"=======================\n"
                f"批量ID: {metadata['batch_id']}\n"
                f"标的数量: {metadata['total_symbols']}个\n"
                f"时间级别: {metadata['timeframe']}\n"
                f"日期范围: {metadata['start_date']} 至 {metadata['end_date']}\n"
                f"执行时间: {metadata['start_time']} - {metadata['end_time']}\n"
                f"=======================\n"
                f"✅ 成功: {batch_results['success_count']}个\n"
                f"❌ 失败: {batch_results['fail_count']}个\n"
                f"📈 平均回报率: {summary['avg_return']:.2f}%\n"
                f"📊 中位数回报率: {summary['median_return']:.2f}%\n"
                f"🏆 最佳标的: {summary['best_symbol']} ({summary['max_return']:.2f}%)\n"
                f"⚠️  最差标的: {summary['worst_symbol']} ({summary['min_return']:.2f}%)\n"
                f"💰 盈利标的比例: {summary['profitable_ratio']:.2f}%\n"
                f"📊 平均胜率: {summary['avg_win_rate']*100:.2f}%\n"
                f"=======================\n"
                f"📁 详细报告: {batch_results['report_path']}"
            )
            
            self.notifier.send_text(content)
            logger.info("批量回测摘要通知发送完成")
        except Exception as e:
            logger.error(f"批量回测通知发送失败: {str(e)}", exc_info=True)
    
    def optimize_parameters(self, symbol: str, start_date: str, end_date: str, 
                           param_ranges: Dict[str, List[Any]], timeframe: str = 'daily') -> Dict[str, Any]:
        """参数优化（网格搜索）"""
        logger.info(f"开始参数优化 - 股票: {symbol}, 时间级别: {timeframe}")
        logger.info(f"参数搜索空间: {param_ranges}")
        
        from itertools import product
        import time
        
        # 验证股票代码
        self._validate_symbol_not_dataframe(symbol)
        
        # 获取并验证数据（避免重复获取）
        data_result = self._acquire_and_validate_data(symbol, start_date, end_date, timeframe)
        if not data_result['success']:
            error_msg = f"参数优化失败: 数据获取失败 - {data_result['error']}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        df = data_result['data']
        logger.info(f"参数优化数据准备完成: {len(df)}条记录")
        
        # 执行缠论计算（基础计算，参数优化时仅调整策略参数）
        calculation_result = self._perform_chanlun_calculation(df, timeframe)
        if not calculation_result['success']:
            error_msg = f"参数优化失败: 缠论计算失败 - {calculation_result['error']}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        calculated_df = calculation_result['data']
        
        # 生成参数组合（网格搜索）
        param_names = list(param_ranges.keys())
        param_combinations = product(*param_ranges.values())
        total_combinations = np.prod([len(range_list) for range_list in param_ranges.values()])
        logger.info(f"参数组合总数: {total_combinations}个")
        
        # 初始化优化结果
        best_result = None
        best_params = None
        best_score = -float('inf')
        optimization_results = []
        score_metric = self.config.get('optimization', {}).get('score_metric', 'sharpe_ratio')
        
        # 遍历所有参数组合
        for i, params in enumerate(param_combinations, 1):
            param_dict = dict(zip(param_names, params))
            logger.info(f"测试参数组合 {i}/{total_combinations}: {param_dict}")
            
            try:
                start_time = time.time()
                
                # 临时修改回测参数
                backtest_config = self.config.get('backtest', {}).copy()
                backtest_config.update(param_dict)
                
                # 执行回测
                backtest_result = self._execute_backtest(
                    calculated_df,
                    initial_capital=self.config.get('optimization', {}).get('initial_capital', 100000),
                    timeframe=timeframe
                )
                
                if not backtest_result['success']:
                    logger.warning(f"参数组合 {param_dict} 回测失败: {backtest_result['error']}")
                    continue
                
                result = backtest_result['data']
                result['parameters'] = param_dict
                result['test_duration'] = round(time.time() - start_time, 2)
                
                # 计算评分（根据目标指标）
                if score_metric == 'sharpe_ratio':
                    score = result.get('sharpe_ratio', 0)
                elif score_metric == 'return_percent':
                    score = result.get('return_percent', 0) / max(result.get('max_drawdown', 1), 0.01)
                elif score_metric == 'profit_factor':
                    score = result.get('profit_factor', 0)
                elif score_metric == 'win_rate':
                    score = result.get('win_rate', 0)
                else:
                    score = result.get('sharpe_ratio', 0)
                
                result['score'] = score
                optimization_results.append(result)
                
                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_params = param_dict
                    logger.info(f"更新最佳参数: {best_params}, 最佳评分: {best_score:.2f}")
            
            except Exception as e:
                logger.error(f"参数组合 {param_dict} 测试失败: {str(e)}", exc_info=True)
                continue
        
        # 验证优化结果
        if best_result is None:
            error_msg = "所有参数组合测试失败，未找到有效参数"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # 保存优化结果
        opt_config = self.config.get('optimization', {})
        opt_output_dir = opt_config.get('output_dir', 'outputs/optimization')
        os.makedirs(opt_output_dir, exist_ok=True)
        
        opt_result_path = os.path.join(
            opt_output_dir,
            f'{symbol}_{timeframe}_param_optimization_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
        )
        
        with open(opt_result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'score_metric': score_metric,
                    'total_combinations': total_combinations,
                    'success_combinations': len(optimization_results),
                    'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'best_parameters': best_params,
                'best_result': best_result,
                'all_results': optimization_results,
                'param_ranges': param_ranges
            }, f, ensure_ascii=False, indent=2)
        
        # 发送优化结果通知
        if self.config.get('notifications', {}).get('enabled', False):
            self._send_parameter_optimization_notification(best_params, best_result, symbol, timeframe)
        
        logger.info(
            f"参数优化完成 - 最佳参数: {best_params}\n"
            f"最佳评分: {best_score:.2f} ({score_metric})\n"
            f"回测回报率: {best_result.get('return_percent', 0):.2f}%\n"
            f"最大回撤: {best_result.get('max_drawdown', 0):.2f}%\n"
            f"结果保存路径: {opt_result_path}"
        )
        
        return {
            'success': True,
            'best_parameters': best_params,
            'best_result': best_result,
            'best_score': best_score,
            'all_results': optimization_results,
            'optimization_path': opt_result_path,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'score_metric': score_metric,
                'total_combinations': total_combinations
            }
        }
    
    def _send_parameter_optimization_notification(self, best_params: Dict[str, Any], 
                                                best_result: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """发送参数优化结果通知"""
        try:
            content = (
                f"🔧 缠论参数优化完成通知\n"
                f"=======================\n"
                f"股票代码: {symbol}\n"
                f"时间级别: {timeframe}\n"
                f"优化目标: {self.config.get('optimization', {}).get('score_metric', 'sharpe_ratio')}\n"
                f"=======================\n"
                f"🏆 最佳参数:\n"
            )
            # 格式化参数输出
            for param_name, param_value in best_params.items():
                content += f"  • {param_name}: {param_value}\n"
            
            content += (
                f"=======================\n"
                f"📊 回测性能:\n"
                f"  • 总回报率: {best_result.get('return_percent', 0):.2f}%\n"
                f"  • 最大回撤: {best_result.get('max_drawdown', 0):.2f}%\n"
                f"  • 夏普比率: {best_result.get('sharpe_ratio', 0):.2f}\n"
                f"  • 胜率: {best_result.get('win_rate', 0)*100:.2f}%\n"
                f"  • 交易次数: {best_result.get('total_trades', 0)}次\n"
                f"=======================\n"
                f"📁 详细结果: {self.config.get('optimization', {}).get('output_dir', 'outputs/optimization')}"
            )
            
            self.notifier.send_text(content)
            logger.info("参数优化结果通知发送完成")
        except Exception as e:
            logger.error(f"参数优化通知发送失败: {str(e)}", exc_info=True)

class ChanlunBacktester:
    """缠论回测器外层包装类（提供统一调用接口）"""
    
    def __init__(self, config_path: str = 'config/system.yaml'):
        """初始化缠论回测器"""
        self.config = load_config(config_path)
        self.engine = BacktestEngine(self.config)
        logger.info("ChanlunBacktester 初始化完成")
    
    def run(self, symbol: str, start_date: str, end_date: str, timeframe: str = 'weekly', 
           initial_capital: float = 100000) -> Dict[str, Any]:
        """运行单只股票回测"""
        self.engine._validate_symbol_not_dataframe(symbol)
        return self.engine.run_comprehensive_backtest(symbol, start_date, end_date, timeframe, initial_capital)
    
    def run_batch(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = 'weekly', 
                 initial_capital: float = 100000) -> Dict[str, Any]:
        """运行批量回测"""
        return self.engine.batch_backtest(symbols, start_date, end_date, timeframe, initial_capital)
    
    def optimize_params(self, symbol: str, start_date: str, end_date: str, param_ranges: Dict[str, List[Any]], 
                       timeframe: str = 'daily') -> Dict[str, Any]:
        """运行参数优化"""
        return self.engine.optimize_parameters(symbol, start_date, end_date, param_ranges, timeframe)

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='缠论回测系统 - 命令行工具')
    
    # 基础参数
    parser.add_argument('-c', '--config', default='config/system.yaml', help='配置文件路径')
    parser.add_argument('-m', '--mode', required=True, choices=['single', 'batch', 'optimize'], help='运行模式：single(单只)/batch(批量)/optimize(参数优化)')
    parser.add_argument('-s', '--symbol', help='股票代码（single/optimize模式必填）')
    parser.add_argument('-S', '--symbols', nargs='+', help='股票代码列表（batch模式必填）')
    parser.add_argument('-t', '--timeframe', default='daily', choices=['weekly', 'daily', 'minute'], help='时间级别')
    parser.add_argument('--start_date', required=True, help='开始日期（YYYYMMDD或YYYY-MM-DD）')
    parser.add_argument('--end_date', required=True, help='结束日期（YYYYMMDD或YYYY-MM-DD）')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    
    # 参数优化相关参数
    parser.add_argument('--param_ranges', type=str, help='参数范围JSON字符串（optimize模式必填）')
    parser.add_argument('--score_metric', default='sharpe_ratio', choices=['sharpe_ratio', 'return_percent', 'profit_factor', 'win_rate'], help='优化目标指标')
    
    # 输出相关参数
    parser.add_argument('--output_dir', default='outputs', help='输出目录')
    parser.add_argument('--enable_notify', action='store_true', help='启用钉钉通知')
    parser.add_argument('--enable_plot', action='store_true', help='启用图表生成')
    parser.add_argument('--debug', action='store_true', help='调试模式（输出详细日志）')
    
    args = parser.parse_args()
    
    # 调试模式配置
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 初始化回测器
    try:
        backtester = ChanlunBacktester(args.config)
    except Exception as e:
        logger.critical(f"回测器初始化失败: {str(e)}")
        sys.exit(1)
    
    # 调整配置（使用setdefault避免KeyError）
    # 初始化notifications配置（如果不存在）
    if 'notifications' not in backtester.config:
        backtester.config['notifications'] = {}
    backtester.config['notifications']['enabled'] = args.enable_notify
    
    # 初始化plotter配置（如果不存在）
    if 'plotter' not in backtester.config:
        backtester.config['plotter'] = {}
    backtester.config['plotter']['enabled'] = args.enable_plot
    
    # 初始化exporter配置（如果不存在）
    if 'exporter' not in backtester.config:
        backtester.config['exporter'] = {}
    backtester.config['exporter']['output_dir'] = args.output_dir
    
    # 初始化optimization配置（如果不存在）
    if args.mode == 'optimize' and 'optimization' not in backtester.config:
        backtester.config['optimization'] = {}
    if args.mode == 'optimize':
        backtester.config['optimization']['score_metric'] = args.score_metric
    
    # 根据模式执行
    try:
        if args.mode == 'single':
            # 单只股票回测
            if not args.symbol:
                logger.error("single模式必须指定--symbol参数")
                sys.exit(1)
            
            result = backtester.run(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                timeframe=args.timeframe,
                initial_capital=args.capital
            )
            
            # 输出结果摘要
            if result.get('success', False):
                logger.info("\n" + "="*50)
                logger.info("单只股票回测结果摘要")
                logger.info("="*50)
                logger.info(f"股票代码: {args.symbol}")
                logger.info(f"总回报率: {result.get('return_percent', 0):.2f}%")
                logger.info(f"最大回撤: {result.get('max_drawdown', 0):.2f}%")
                logger.info(f"交易次数: {result.get('total_trades', 0)}次")
                logger.info(f"胜率: {result.get('win_rate', 0)*100:.2f}%")
                logger.info(f"夏普比率: {result.get('sharpe_ratio', 0):.2f}")
                logger.info(f"报告路径: {result['report'].get('export_info', {}).get('path', '未导出')}")
                logger.info(f"图表路径: {result['charts'].get('chart_dir', '未生成')}")
                logger.info("="*50)
            else:
                logger.error(f"回测失败: {result.get('error', '未知错误')}")
                sys.exit(1)
        
        elif args.mode == 'batch':
            # 批量回测
            if not args.symbols:
                logger.error("batch模式必须指定--symbols参数")
                sys.exit(1)
            
            result = backtester.run_batch(
                symbols=args.symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                timeframe=args.timeframe,
                initial_capital=args.capital
            )
            
            # 输出批量结果摘要
            logger.info("\n" + "="*50)
            logger.info("批量回测结果摘要")
            logger.info("="*50)
            logger.info(f"总标的数: {len(args.symbols)}")
            logger.info(f"成功: {result['success_count']}个")
            logger.info(f"失败: {result['fail_count']}个")
            logger.info(f"平均回报率: {result['summary']['avg_return']:.2f}%")
            logger.info(f"最佳标的: {result['summary']['best_symbol']} ({result['summary']['max_return']:.2f}%)")
            logger.info(f"最差标的: {result['summary']['worst_symbol']} ({result['summary']['min_return']:.2f}%)")
            logger.info(f"盈利标的比例: {result['summary']['profitable_ratio']:.2f}%")
            logger.info(f"批量报告路径: {result['report_path']}")
            logger.info("="*50)
        
        elif args.mode == 'optimize':
            # 参数优化
            if not args.symbol:
                logger.error("optimize模式必须指定--symbol参数")
                sys.exit(1)
            if not args.param_ranges:
                logger.error("optimize模式必须指定--param_ranges参数（JSON字符串）")
                sys.exit(1)
            
            # 解析参数范围
            try:
                param_ranges = json.loads(args.param_ranges)
            except json.JSONDecodeError as e:
                logger.error(f"param_ranges解析失败: {str(e)}")
                sys.exit(1)
            
            result = backtester.optimize_params(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                param_ranges=param_ranges,
                timeframe=args.timeframe
            )
            
            # 输出优化结果摘要
            if result.get('success', False):
                logger.info("\n" + "="*50)
                logger.info("参数优化结果摘要")
                logger.info("="*50)
                logger.info(f"股票代码: {args.symbol}")
                logger.info(f"优化目标: {args.score_metric}")
                logger.info(f"最佳参数: {result['best_parameters']}")
                logger.info(f"最佳评分: {result['best_score']:.2f}")
                logger.info(f"回测回报率: {result['best_result'].get('return_percent', 0):.2f}%")
                logger.info(f"最大回撤: {result['best_result'].get('max_drawdown', 0):.2f}%")
                logger.info(f"优化结果路径: {result['optimization_path']}")
                logger.info("="*50)
            else:
                logger.error(f"参数优化失败: {result.get('error', '未知错误')}")
                sys.exit(1)
        
        logger.info("程序执行完成")
        sys.exit(0)
        
    except Exception as e:
        logger.critical(f"程序执行异常: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()