#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论回测系统 - 修复版本
修复了日期范围不正确和符号验证问题
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
from typing import Dict, List, Optional, Tuple, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入依赖模块
try:
    from src.config import load_config
    from src.data_fetcher import StockDataAPI
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
        """初始化所有组件"""
        # 初始化数据API
        data_fetcher_config = self.config.get('data_fetcher', {})
        self.data_api = StockDataAPI(
            max_retries=data_fetcher_config.get('max_retries', 3),
            timeout=data_fetcher_config.get('timeout', 10)
        )
        
        # 初始化计算器
        chanlun_config = self.config.get('chanlun', {})
        self.calculator = ChanlunCalculator(chanlun_config)
        
        # 初始化通知器
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
        # 🔧🔧🔧🔧🔧🔧 增强日志：记录原始日期参数
        logger.info(f"开始全面回测: {symbol} {timeframe}")
        logger.info(f"用户指定日期范围: {start_date} 至 {end_date}")
        logger.info(f"初始资金: {initial_capital}")
        
        try:
            # 🔧🔧🔧🔧🔧🔧 防御性检查：验证symbol不是DataFrame
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
            
            # 5. 图表生成阶段
            chart_result = self._generate_detailed_charts(result, symbol, timeframe)
            result['charts'] = chart_result
            
            # 6. 通知发送阶段
            if self.config.get('notifications', {}).get('enabled', False):
                self._send_notifications(result, symbol, timeframe)
            
            # 🔧🔧🔧🔧🔧🔧 记录实际使用的日期范围
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
        防御性检查：确保symbol不是DataFrame或其他无效类型
        :param symbol: 要检查的符号
        """
        # 🔧🔧🔧🔧🔧🔧 修复：防止DataFrame被当作symbol传递
        if symbol is None:
            raise ValueError("股票代码不能为None")
        
        # 检查是否为DataFrame或其他复杂对象
        symbol_str = str(symbol)
        if len(symbol_str) > 100:  # 正常股票代码不会超过20字符
            logger.error(f"疑似DataFrame被当作股票代码传递: {symbol_str[:100]}...")
            raise ValueError(f"无效股票代码类型: 疑似DataFrame对象")
        
        # 检查DataFrame特征关键词
        dataframe_indicators = ['DataFrame', 'Series', 'open', 'high', 'low', 'close', 'volume', 'date']
        if any(indicator in symbol_str for indicator in dataframe_indicators):
            logger.error(f"检测到DataFrame特征在股票代码中: {symbol_str[:200]}")
            raise ValueError(f"无效股票代码: 检测到DataFrame特征")
    
    def _acquire_and_validate_data(self, symbol: str, start_date: str, end_date: str, 
                                  timeframe: str) -> Dict[str, Any]:
        """
        获取并验证数据 - 修复日期范围问题
        :return: 包含成功状态和数据的结果字典
        """
        try:
            # 🔧🔧🔧🔧🔧🔧 增强日志：记录日期参数
            logger.info(f"数据获取阶段 - 符号: {symbol}, 时间级别: {timeframe}")
            logger.info(f"请求日期范围: {start_date} 至 {end_date}")
            
            # 根据时间级别获取数据
            if timeframe == 'weekly':
                df = self.data_api.get_weekly_data(symbol, start_date, end_date)
            elif timeframe == 'daily':
                df = self.data_api.get_daily_data(symbol, start_date, end_date)
            elif timeframe == 'minute':
                df = self.data_api.get_minute_data(symbol, '5m', 30)
            else:
                return {'success': False, 'error': f'不支持的时间级别: {timeframe}'}
            
            # 验证数据质量
            if df.empty:
                logger.warning("获取的数据为空")
                return {'success': False, 'error': '获取的数据为空'}
            
            if len(df) < 10:
                logger.warning(f"数据点数不足: {len(df)}条，至少需要10个数据点")
                return {'success': False, 'error': '数据点数不足，至少需要10个数据点'}
            
            # 检查必要列
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"缺失必要列: {missing_columns}")
                return {'success': False, 'error': f'缺失必要列: {missing_columns}'}
            
            # 🔧🔧🔧🔧🔧🔧 修复：优先使用数据源返回的日期信息
            if 'date' not in df.columns:
                if 'timestamp' in df.columns:
                    df = df.rename(columns={'timestamp': 'date'})
                    logger.info("使用timestamp列作为日期列")
                else:
                    # 🔧 关键修复：不创建可能错误的日期范围，直接返回错误
                    logger.error("数据缺少日期列，无法进行时间序列分析")
                    return {'success': False, 'error': '数据缺少日期列，无法进行时间序列分析'}
            
            # 🔧 安全处理日期列
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # 记录实际数据日期范围
                actual_start = df['date'].min()
                actual_end = df['date'].max()
                days_range = (actual_end - actual_start).days
                
                logger.info(f"实际数据日期范围: {actual_start.strftime('%Y-%m-%d')} 至 {actual_end.strftime('%Y-%m-%d')}")
                logger.info(f"数据点数: {len(df)}条, 时间跨度: {days_range}天")
                
                # 🔧 检查日期范围是否符合预期
                expected_start = pd.to_datetime(start_date)
                expected_end = pd.to_datetime(end_date)
                
                if actual_start > expected_start or actual_end < expected_end:
                    logger.warning(f"数据日期范围不完整: 预期{expected_start.strftime('%Y-%m-%d')}~{expected_end.strftime('%Y-%m-%d')}, 实际{actual_start.strftime('%Y-%m-%d')}~{actual_end.strftime('%Y-%m-%d')}")
                
            except Exception as e:
                logger.error(f"日期处理异常: {str(e)}")
                return {'success': False, 'error': f'日期处理异常: {str(e)}'}
            
            return {'success': True, 'data': df}
            
        except Exception as e:
            logger.error(f"数据获取异常: {str(e)}")
            return {'success': False, 'error': f'数据获取异常: {str(e)}'}
    
    def _perform_chanlun_calculation(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        执行缠论计算 - 增强符号验证
        :return: 包含成功状态和计算结果的结果字典
        """
        try:
            # 🔧🔧🔧🔧🔧🔧 防御性检查：确保没有DataFrame被错误传递
            if hasattr(df, 'symbol') and not isinstance(df.symbol, str):
                logger.warning(f"检测到非字符串symbol: {type(df.symbol)}")
            
            # 设置时间级别参数
            self.calculator.set_timeframe_params(timeframe)
            
            # 计算缠论指标
            calculated_df = self.calculator.calculate(df, timeframe)
            
            # 验证计算结果
            if calculated_df.empty:
                return {'success': False, 'error': '缠论计算结果为空'}
            
            # 检查是否生成了必要的缠论列
            chanlun_columns = ['top_fractal', 'bottom_fractal', 'pen_type', 'central_bank']
            generated_columns = [col for col in chanlun_columns if col in calculated_df.columns]
            if len(generated_columns) < 2:
                logger.warning(f"生成的缠论指标较少: {generated_columns}")
            
            return {'success': True, 'data': calculated_df}
            
        except Exception as e:
            return {'success': False, 'error': f'缠论计算异常: {str(e)}'}
    
    def _execute_backtest(self, df: pd.DataFrame, initial_capital: float, 
                         timeframe: str) -> Dict[str, Any]:
        """
        执行回测
        :return: 包含成功状态和回测结果的结果字典
        """
        try:
            # 使用计算器的回测功能
            result = self.calculator.backtest(df, initial_capital, timeframe)
            
            # 验证回测结果
            if not result or 'final_value' not in result:
                return {'success': False, 'error': '回测结果无效'}
            
            # 添加额外指标
            result = self._enhance_backtest_metrics(result)
            
            return {'success': True, 'data': result}
            
        except Exception as e:
            return {'success': False, 'error': f'回测执行异常: {str(e)}'}
    
    def _enhance_backtest_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强回测指标计算
        """
        # 计算年化波动率
        if 'portfolio_values' in result and len(result['portfolio_values']) > 1:
            returns = []
            for i in range(1, len(result['portfolio_values'])):
                ret = (result['portfolio_values'][i] - result['portfolio_values'][i-1]) / result['portfolio_values'][i-1]
                returns.append(ret)
            
            if returns:
                result['volatility'] = np.std(returns) * np.sqrt(252) * 100  # 年化波动率
                result['max_return'] = max(returns) * 100 if returns else 0
                result['min_return'] = min(returns) * 100 if returns else 0
        
        # 计算盈亏比
        if 'trades' in result:
            profitable_trades = [t for t in result['trades'] if t.get('profit', 0) > 0]
            loss_trades = [t for t in result['trades'] if t.get('profit', 0) < 0]
            
            if loss_trades:
                avg_profit = np.mean([t.get('profit', 0) for t in profitable_trades]) if profitable_trades else 0
                avg_loss = abs(np.mean([t.get('profit', 0) for t in loss_trades])) if loss_trades else 0
                result['profit_loss_ratio'] = avg_profit / avg_loss if avg_loss > 0 else float('inf')
            else:
                result['profit_loss_ratio'] = float('inf')
        
        # 计算交易频率
        if 'data_points' in result and 'total_trades' in result:
            result['trade_frequency'] = result['total_trades'] / result['data_points'] * 100 if result['data_points'] > 0 else 0
        
        return result
    
    def _generate_comprehensive_report(self, result: Dict[str, Any], symbol: str, 
                                      timeframe: str) -> Dict[str, Any]:
        """
        生成全面报告
        """
        try:
            report_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance_metrics': self._extract_performance_metrics(result),
                'trade_analysis': self._analyze_trades(result),
                'risk_metrics': self._calculate_risk_metrics(result),
                'summary': self._generate_summary(result, symbol, timeframe),
                # 🔧🔧🔧🔧🔧🔧 新增：记录实际日期范围
                'date_range_info': result.get('actual_date_range', {})
            }
            
            # 保存报告
            report_filename = f"backtest_report_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("outputs/reports", exist_ok=True)
            with open(f"outputs/reports/{report_filename}", 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"报告已保存: outputs/reports/{report_filename}")
            return report_data
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return {'error': str(e)}
    
    def _extract_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取性能指标"""
        return {
            'initial_capital': result.get('initial_capital', 0),
            'final_value': result.get('final_value', 0),
            'total_return': result.get('return_percent', 0),
            'annual_return': result.get('annual_return', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'win_rate': result.get('win_rate', 0),
            'total_trades': result.get('total_trades', 0),
            'profitable_trades': result.get('profitable_trades', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'volatility': result.get('volatility', 0),
            'profit_loss_ratio': result.get('profit_loss_ratio', 0),
            'trade_frequency': result.get('trade_frequency', 0)
        }
    
    def _analyze_trades(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """分析交易记录"""
        trades = result.get('trades', [])
        if not trades:
            return {'total_trades': 0}
        
        buy_trades = [t for t in trades if t.get('action') == 'buy']
        sell_trades = [t for t in trades if t.get('action') == 'sell']
        
        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'avg_profit': np.mean([t.get('profit', 0) for t in sell_trades]) if sell_trades else 0,
            'avg_holding_period': self._calculate_avg_holding_period(trades),
            'consecutive_wins': self._calculate_consecutive_wins(sell_trades),
            'consecutive_losses': self._calculate_consecutive_losses(sell_trades)
        }
    
    def _calculate_avg_holding_period(self, trades: List[Dict]) -> float:
        """计算平均持仓周期"""
        holding_periods = []
        buy_dates = {}
        
        for trade in trades:
            if trade['action'] == 'buy':
                buy_dates[trade.get('symbol', 'default')] = trade.get('date')
            elif trade['action'] == 'sell':
                buy_date = buy_dates.get(trade.get('symbol', 'default'))
                if buy_date and hasattr(buy_date, '__sub__'):
                    holding_period = (trade.get('date') - buy_date).days
                    holding_periods.append(holding_period)
        
        return np.mean(holding_periods) if holding_periods else 0
    
    def _calculate_consecutive_wins(self, sell_trades: List[Dict]) -> int:
        """计算连续盈利次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sell_trades:
            if trade.get('profit', 0) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, sell_trades: List[Dict]) -> int:
        """计算连续亏损次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sell_trades:
            if trade.get('profit', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_risk_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """计算风险指标"""
        portfolio_values = result.get('portfolio_values', [])
        if len(portfolio_values) < 2:
            return {}
        
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        if not returns:
            return {}
        
        return {
            'var_95': np.percentile(returns, 5) * 100,  # 95% VaR
            'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]) * 100 if returns else 0,
            'downside_deviation': np.std([r for r in returns if r < 0]) * np.sqrt(252) * 100 if [r for r in returns if r < 0] else 0,
            'ulcer_index': self._calculate_ulcer_index(portfolio_values),
            'calmar_ratio': result.get('annual_return', 0) / result.get('max_drawdown', 1) if result.get('max_drawdown', 0) > 0 else 0
        }
    
    def _calculate_ulcer_index(self, portfolio_values: List[float]) -> float:
        """计算溃疡指数"""
        if len(portfolio_values) < 2:
            return 0
        
        peak = portfolio_values[0]
        drawdowns_squared = []
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns_squared.append(drawdown ** 2)
        
        return np.sqrt(np.mean(drawdowns_squared)) * 100 if drawdowns_squared else 0
    
    def _generate_summary(self, result: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        """生成总结"""
        performance = result.get('return_percent', 0)
        risk = result.get('max_drawdown', 0)
        
        if performance > 20 and risk < 10:
            rating = '优秀'
            recommendation = '策略表现优异，建议继续使用'
        elif performance > 10 and risk < 15:
            rating = '良好'
            recommendation = '策略表现良好，可考虑优化'
        elif performance > 0:
            rating = '一般'
            recommendation = '策略有待优化，建议调整参数'
        else:
            rating = '较差'
            recommendation = '策略需要重大调整或放弃'
        
        return {
            'rating': rating,
            'recommendation': recommendation,
            'strengths': self._identify_strengths(result),
            'weaknesses': self._identify_weaknesses(result),
            'improvement_suggestions': self._generate_improvement_suggestions(result)
        }
    
    def _identify_strengths(self, result: Dict[str, Any]) -> List[str]:
        """识别优势"""
        strengths = []
        
        if result.get('win_rate', 0) > 60:
            strengths.append('高胜率')
        if result.get('profit_loss_ratio', 0) > 2:
            strengths.append('良好的盈亏比')
        if result.get('max_drawdown', 0) < 10:
            strengths.append('低回撤')
        if result.get('sharpe_ratio', 0) > 1:
            strengths.append('优异的夏普比率')
        if result.get('annual_return', 0) > 15:
            strengths.append('高年化收益')
        
        return strengths if strengths else ['需进一步优化']
    
    def _identify_weaknesses(self, result: Dict[str, Any]) -> List[str]:
        """识别劣势"""
        weaknesses = []
        
        if result.get('win_rate', 0) < 40:
            weaknesses.append('胜率偏低')
        if result.get('profit_loss_ratio', 0) < 1:
            weaknesses.append('盈亏比不理想')
        if result.get('max_drawdown', 0) > 20:
            weaknesses.append('回撤较大')
        if result.get('sharpe_ratio', 0) < 0.5:
            weaknesses.append('风险调整后收益不佳')
        if result.get('annual_return', 0) < 5:
            weaknesses.append('收益水平较低')
        
        return weaknesses if weaknesses else ['无明显劣势']
    
    def _generate_improvement_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if result.get('win_rate', 0) < 50:
            suggestions.append('优化入场时机，提高信号质量')
        if result.get('profit_loss_ratio', 0) < 1.5:
            suggestions.append('调整止损止盈策略，改善风险收益比')
        if result.get('max_drawdown', 0) > 15:
            suggestions.append('加强风险控制，降低单次交易仓位')
        if result.get('trade_frequency', 0) > 30:
            suggestions.append('减少交易频率，降低交易成本')
        if result.get('volatility', 0) > 20:
            suggestions.append('考虑增加过滤条件，降低组合波动')
        
        return suggestions if suggestions else ['当前策略参数较为合理']
    
    def _generate_detailed_charts(self, result: Dict[str, Any], symbol: str, 
                                 timeframe: str) -> Dict[str, Any]:
        """
        生成详细图表
        """
        try:
            chart_files = []
            
            # 1. 组合价值曲线
            if 'portfolio_values' in result:
                fig1 = plt.figure(figsize=(12, 8))
                plt.plot(result['portfolio_values'])
                plt.title(f'{symbol} {timeframe}回测 - 组合价值曲线')
                plt.xlabel('时间')
                plt.ylabel('组合价值')
                chart1_file = f"portfolio_growth_{symbol}_{timeframe}.png"
                plt.savefig(f"outputs/plots/{chart1_file}", dpi=300, bbox_inches='tight')
                chart_files.append(chart1_file)
                plt.close(fig1)
            
            # 2. 回撤曲线
            if 'portfolio_values' in result:
                fig2 = plt.figure(figsize=(12, 8))
                portfolio_values = result['portfolio_values']
                peak = portfolio_values[0]
                drawdowns = []
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak * 100
                    drawdowns.append(drawdown)
                
                plt.plot(drawdowns)
                plt.title(f'{symbol} {timeframe}回测 - 回撤曲线')
                plt.xlabel('时间')
                plt.ylabel('回撤百分比 (%)')
                chart2_file = f"drawdown_{symbol}_{timeframe}.png"
                plt.savefig(f"outputs/plots/{chart2_file}", dpi=300, bbox_inches='tight')
                chart_files.append(chart2_file)
                plt.close(fig2)
            
            logger.info(f"生成{len(chart_files)}张图表")
            return {'chart_files': chart_files, 'success': True}
            
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
            return {'error': str(e), 'success': False}
    
    def _send_notifications(self, result: Dict[str, Any], symbol: str, timeframe: str):
        """发送通知"""
        try:
            # 生成通知内容
            performance = result.get('return_percent', 0)
            drawdown = result.get('max_drawdown', 0)
            win_rate = result.get('win_rate', 0)
            
            # 🔧🔧🔧🔧🔧🔧 包含实际日期范围信息
            actual_range = result.get('actual_date_range', {})
            actual_start = actual_range.get('start', '未知')
            actual_end = actual_range.get('end', '未知')
            
            message = (
                f"回测完成通知\n"
                f"标的: {symbol} ({timeframe})\n"
                f"实际日期范围: {actual_start} 至 {actual_end}\n"
                f"总回报: {performance:.2f}%\n"
                f"最大回撤: {drawdown:.2f}%\n"
                f"胜率: {win_rate:.2f}%\n"
                f"交易次数: {result.get('total_trades', 0)}\n"
                f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # 发送钉钉通知
            self.notifier.send_signal(symbol, {
                'action': 'report',
                'message': message,
                'performance': performance,
                'risk_level': 'low' if drawdown < 10 else 'medium' if drawdown < 20 else 'high'
            })
            
            logger.info("回测完成通知已发送")
            
        except Exception as e:
            logger.error(f"发送通知失败: {e}")
    
    def _create_error_result(self, initial_capital: float, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'success': False,
            'error': error_msg,
            'initial_capital': initial_capital,
            'final_value': initial_capital,
            'return_percent': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profitable_trades': 0,
            'trades': [],
            'portfolio_values': [initial_capital],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

class ChanlunBacktester:
    """兼容层 - 保持原有接口"""
    
    def __init__(self, api, calculator, config=None):
        self.engine = BacktestEngine(config or {})
        self.api = api
        self.calculator = calculator
    
    def run(self, symbol, timeframe, start, end, initial_capital=100000):
        """
        修改后的run方法：根据symbol、timeframe、start、end参数获取数据并执行回测
        :param symbol: 股票代码
        :param timeframe: 时间级别（weekly/daily/minute）
        :param start: 开始日期
        :param end: 结束日期
        :param initial_capital: 初始资金，默认100000
        :return: 回测结果
        """
        # 🔧🔧🔧🔧🔧🔧 防御性检查
        self.engine._validate_symbol_not_dataframe(symbol)
        
        # 根据timeframe获取数据
        if timeframe == 'weekly':
            df = self.api.get_weekly_data(symbol, start, end)
        elif timeframe == 'daily':
            df = self.api.get_daily_data(symbol, start, end)
        elif timeframe == 'minute':
            df = self.api.get_minute_data(symbol, '5m', 30)
        else:
            raise ValueError(f"不支持的时间级别: {timeframe}")
        
        # 调用计算器的回测功能
        return self.calculator.backtest(df, initial_capital, timeframe)
    
    def run_backtest(self, symbol, start_date, end_date, timeframe, initial_capital=100000):
        """兼容原有run_backtest方法"""
        return self.engine.run_comprehensive_backtest(symbol, start_date, end_date, timeframe, initial_capital)

def main():
    """主函数 - 保持原有逻辑"""
    parser = argparse.ArgumentParser(description='缠论回测系统')
    parser.add_argument('--mode', choices=['backtest', 'realtime', 'weekly_scan', 'pre_market', 'daily_report'], 
                       default='backtest', help='运行模式')
    parser.add_argument('--timeframe', choices=['daily', 'weekly', 'minute'], 
                       default='weekly', help='时间级别')
    parser.add_argument('--start', type=str, default='2023-11-08', 
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-11-08', 
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--etf', type=str, default='sh510300', 
                       help='ETF代码')
    parser.add_argument('--capital', type=float, default=100000,
                       help='初始资金')
    parser.add_argument('--report_level', choices=['basic', 'detailed'], 
                       default='detailed', help='报告详细程度')
    
    args = parser.parse_args()
    
    # 加载配置
    system_config = load_config()
    
    # 使用新引擎
    engine = BacktestEngine(system_config)
    
    if args.mode == 'backtest':
        result = engine.run_comprehensive_backtest(
            symbol=args.etf,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe,
            initial_capital=args.capital
        )
        
        if result.get('success', False):
            logger.info("回测完成")
            # 输出实际日期范围信息
            actual_range = result.get('actual_date_range', {})
            if actual_range:
                logger.info(f"实际使用的日期范围: {actual_range.get('start')} 至 {actual_range.get('end')}")
        else:
            logger.error(f"回测失败: {result.get('error', '未知错误')}")
    
    elif args.mode == 'pre_market':
        """盘前报告模式"""
        logger.info("生成盘前报告")
        
        # 确保输出目录存在
        os.makedirs("outputs/reports", exist_ok=True)
        
        # 生成盘前报告
        report = generate_pre_market_report(
            symbols=[args.etf],
            api=engine.data_api,
            calculator=engine.calculator,
            start_date=args.start,
            end_date=args.end
        )
        
        # 保存报告
        report_filename = f"pre_market_report_{args.etf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"outputs/reports/{report_filename}", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"盘前报告已保存: outputs/reports/{report_filename}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    elif args.mode == 'daily_report':
        """盘后日报模式"""
        logger.info("生成盘后日报")
        
        # 确保输出目录存在
        os.makedirs("outputs/reports", exist_ok=True)
        
        # 生成盘后日报
        report = generate_daily_report(
            symbols=[args.etf],
            api=engine.data_api,
            calculator=engine.calculator,
            start_date=args.start,
            end_date=args.end
        )
        
        # 保存报告
        report_filename = f"daily_report_{args.etf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"outputs/reports/{report_filename}", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"盘后日报已保存: outputs/reports/{report_filename}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    else:
        logger.info(f"{args.mode}模式暂未实现")

if __name__ == "__main__":
    main()