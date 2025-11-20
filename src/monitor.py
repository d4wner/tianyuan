#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论实时监控模块 - 完整修复版（无模拟数据污染）
包含所有功能：实时监控、周线扫描、风险控制、信号生成等
已移除所有降级逻辑和模拟数据代码
"""

import sys
import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 修复导入路径问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入依赖模块（不再使用降级逻辑）
from src.config import load_config
from src.data_fetcher import StockDataAPI
from src.calculator import ChanlunCalculator
from src.notifier import DingdingNotifier
from src.utils import get_last_trading_day, is_trading_hour, get_valid_date_range_str

logger = logging.getLogger('ChanlunMonitor')

class RiskEngine:
    """风控引擎 - 内聚所有风控逻辑"""
    
    def __init__(self, config):
        """
        初始化风控引擎
        :param config: 风控配置
        """
        self.config = config
        logger.info("风控引擎初始化完成")
    
    def check_position_risk(self, symbol, position_size, available_cash, total_capital):
        """
        检查仓位风险
        :param symbol: 股票代码
        :param position_size: 计划仓位比例
        :param available_cash: 可用资金
        :param total_capital: 总资金
        :return: 是否通过风控
        """
        # 检查单标的风险暴露
        max_single = self.config.get('max_single_position', 0.4)
        if position_size > max_single:
            logger.warning(f"{symbol} 仓位{position_size}超过单标的上限{max_single}")
            return False
        
        # 检查总仓位风险
        total_position = 1.0 - (available_cash / total_capital)
        max_total = self.config.get('max_total_position', 0.7)
        if total_position + position_size > max_total:
            logger.warning(f"总仓位{total_position+position_size}超过上限{max_total}")
            return False
        
        return True
    
    def check_market_risk(self, market_data):
        """
        检查市场风险
        :param market_data: 市场数据
        :return: 风险等级
        """
        if market_data.empty:
            return "UNKNOWN"
            
        # 计算市场波动率
        volatility = (market_data['high'].iloc[-1] - market_data['low'].iloc[-1]) / market_data['close'].iloc[-1]
        
        if volatility > 0.05:
            return "HIGH_VOLATILITY"
        elif volatility > 0.03:
            return "MEDIUM_VOLATILITY"
        else:
            return "LOW_VOLATILITY"

class ChanlunMonitor:
    """缠论实时监控模块 - 支持周线扫描"""
    
    def __init__(self, system_config, api, calculator, notifier):
        """
        初始化监控器
        :param system_config: 系统配置
        :param api: 数据API
        :param calculator: 缠论计算器
        :param notifier: 通知器
        """
        self.config = system_config
        self.api = api
        self.calculator = calculator
        self.notifier = notifier
        self.symbols = []
        self.position = {}
        self.total_capital = 600000  # 预设总资金60万
        self.available_cash = self.total_capital
        self.interval = self.config.get('monitoring', {}).get('interval', 10)
        self.minute_period = self.config.get('monitoring', {}).get('minute_period', '5m')
        self.minute_days = self.config.get('monitoring', {}).get('minute_days', 3)
        self.last_signals = {}  # 修复：确保初始化
        
        # 初始化风控引擎
        risk_config = self.config.get('risk_management', {})
        self.risk_engine = RiskEngine(risk_config)
        
        # 加载持仓
        self.load_positions()
        logger.info("缠论监控器初始化完成")
    
    def add_symbol(self, symbol):
        """
        添加监控股票
        :param symbol: 股票代码
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"添加监控股票: {symbol}")
    
    def load_positions(self):
        """
        加载持仓信息
        """
        try:
            # 从 data/signals/ 目录加载持仓文件
            positions_path = self.config.get('data_storage', {}).get('positions_path', 'data/signals/positions.json')
            
            # 确保目录存在
            os.makedirs(os.path.dirname(positions_path), exist_ok=True)
            
            if os.path.exists(positions_path):
                # 从文件加载持仓
                with open(positions_path, 'r') as f:
                    self.position = json.load(f)
                    logger.info("持仓信息加载成功")
            else:
                self.position = {}
                logger.warning("持仓文件不存在，初始化空持仓")
        except Exception as e:
            logger.error(f"加载持仓失败: {str(e)}")
            self.position = {}
    
    def save_positions(self):
        """
        保存持仓信息
        """
        try:
            # 保存到 data/signals/ 目录
            positions_path = self.config.get('data_storage', {}).get('positions_path', 'data/signals/positions.json')
            
            # 确保目录存在
            os.makedirs(os.path.dirname(positions_path), exist_ok=True)
            
            with open(positions_path, 'w') as f:
                json.dump(self.position, f, indent=2)
            logger.debug("持仓信息已保存")
        except Exception as e:
            logger.error(f"保存持仓失败: {str(e)}")
    
    def get_available_cash(self):
        """
        获取可用资金
        :return: 可用资金
        """
        return self.available_cash
    
    def calculate_position_size(self, symbol, signal_type):
        """
        计算仓位大小 - 修复版
        :param symbol: 股票代码
        :param signal_type: 信号类型
        :return: 仓位比例
        """
        # 根据信号类型确定仓位比例
        if signal_type == 'first_buy' or signal_type == 'black_swan_buy':  # 修复：正确条件判断
            return [0.1, 0.15]
        elif signal_type == 'second_buy':
            return [0.4, 0.5]
        elif signal_type == 'third_buy':
            return [0.2, 0.25]
        else:
            return [0.3]  # 默认仓位
    
    def check_black_swan_conditions(self, data):
        """
        检查黑天鹅机会三重过滤条件 - 修复版
        :param data: 股票数据
        :return: 是否满足黑天鹅条件
        """
        try:
            if len(data) < 6:
                return False
            
            # 条件1: 使用实际存在的底分型检测方法
            bottom_divergence_confirmed = data['bottom_fractal'].iloc[-1] == True  # 修复：简化检查逻辑
            
            # 条件2: 波动率>5%
            volatility = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['close'].iloc[-1]
            volatility_ok = volatility > 0.05
            
            # 条件3: 量能异常放大>150%
            avg_volume = data['volume'].iloc[-6:-1].mean()
            volume_ok = data['volume'].iloc[-1] > avg_volume * 1.5 if avg_volume > 0 else False
            
            return bottom_divergence_confirmed and volatility_ok and volume_ok
        except Exception as e:
            logger.error(f"黑天鹅条件检查失败: {str(e)}")
            return False
    
    def generate_signal(self, data, market_condition, symbol):
        """
        生成交易信号
        :param data: 股票数据
        :param market_condition: 市场状况
        :param symbol: 股票代码
        :return: 交易信号字典
        """
        if data.empty:
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "action": "hold",
                "reason": "no_data",
                "price": 0,
                "target_price": 0,
                "stoploss": 0
            }
            
        # 获取最新价格
        current_price = data.iloc[-1]['close']
        
        # 初始化信号
        signal = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "market_condition": market_condition,
            "price": current_price,
            "target_price": current_price * 1.05,  # 目标价格+5%
            "stoploss": current_price * 0.97,     # 止损价格-3%
            "reason": market_condition,
            "valid_until": (datetime.now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 根据市场状况确定操作
        if 'buy' in market_condition or 'up' in market_condition:
            signal["action"] = "buy"
            
            # 黑天鹅机会特殊处理
            if 'black_swan' in market_condition:
                signal["position_size"] = [0.4]  # 黑天鹅仓位上限40%
                signal["reason"] = "black_swan_opportunity"
            else:
                # 根据信号类型确定仓位范围
                signal["position_size"] = self.calculate_position_size(symbol, market_condition)
        elif 'sell' in market_condition or 'down' in market_condition:
            signal["action"] = "sell"
            signal["position_size"] = [1.0]  # 默认全卖
        else:
            signal["action"] = "hold"
            signal["reason"] = "no_clear_signal"
        
        return signal
    
    def execute_buy(self, signal, symbol):
        """
        执行买入操作
        :param signal: 交易信号
        :param symbol: 股票代码
        """
        try:
            # 获取仓位范围
            position_range = signal.get('position_size', [0.3])
            
            # 计算仓位大小（取范围中间值）
            if len(position_range) == 2:
                position_size = (position_range[0] + position_range[1]) / 2
            else:
                position_size = position_range[0]
            
            # 检查仓位风险
            if not self.risk_engine.check_position_risk(symbol, position_size, self.available_cash, self.total_capital):  # 修复：方法名一致
                logger.warning(f"{symbol} 仓位风险检查未通过，取消买入")
                return
            
            # 计算买入金额
            buy_amount = self.available_cash * position_size
            buy_price = signal['price']
            
            if buy_price <= 0:
                logger.warning(f"买入价格无效: {buy_price}")
                return
                
            buy_shares = int(buy_amount / buy_price)
            
            if buy_shares <= 0:
                logger.warning(f"可用资金不足，无法买入 {symbol}")
                return
            
            # 更新持仓
            if symbol not in self.position:
                self.position[symbol] = {
                    "shares": 0,
                    "avg_price": 0,
                    "stoploss": signal['stoploss']
                }
            
            # 计算新的平均价格
            total_shares = self.position[symbol]['shares'] + buy_shares
            total_cost = self.position[symbol]['avg_price'] * self.position[symbol]['shares'] + buy_price * buy_shares
            
            if total_shares > 0:
                new_avg_price = total_cost / total_shares
            else:
                new_avg_price = buy_price
            
            # 更新持仓
            self.position[symbol]['shares'] = total_shares
            self.position[symbol]['avg_price'] = new_avg_price
            self.position[symbol]['stoploss'] = signal['stoploss']
            
            # 更新可用资金
            self.available_cash -= buy_shares * buy_price
            
            # 记录交易
            logger.info(f"买入 {symbol}: {buy_shares}股 @ {buy_price}, 总仓位: {total_shares}股")
            
            # 发送通知
            self.notifier.send_signal(symbol, signal)
            
            # 保存持仓
            self.save_positions()
            
        except Exception as e:
            logger.error(f"执行买入失败: {str(e)}")
    
    def execute_sell(self, signal, symbol):
        """
        执行卖出操作
        :param signal: 交易信号
        :param symbol: 股票代码
        """
        try:
            if symbol not in self.position or self.position[symbol]['shares'] <= 0:
                logger.warning(f"无 {symbol} 持仓可卖")
                return
            
            # 获取持仓信息
            position = self.position[symbol]
            sell_price = signal['price']
            sell_shares = position['shares']  # 默认全卖
            
            if sell_price <= 0:
                logger.warning(f"卖出价格无效: {sell_price}")
                return
            
            # 计算卖出金额
            sell_amount = sell_shares * sell_price
            
            # 计算盈亏
            cost = position['avg_price'] * sell_shares
            profit = sell_amount - cost
            profit_percent = (profit / cost) * 100 if cost > 0 else 0
            
            # 更新持仓
            self.position[symbol]['shares'] = 0
            self.available_cash += sell_amount
            
            # 记录交易
            logger.info(f"卖出 {symbol}: {sell_shares}股 @ {sell_price}, 盈亏: {profit:.2f}({profit_percent:.2f}%)")
            
            # 发送通知
            self.notifier.send_signal(symbol, signal)
            
            # 保存持仓
            self.save_positions()
            
        except Exception as e:
            logger.error(f"执行卖出失败: {str(e)}")
    
    def check_stoploss(self, symbol, current_price):
        """
        检查止损条件
        :param symbol: 股票代码
        :param current_price: 当前价格
        """
        if symbol in self.position and self.position[symbol]['shares'] > 0:
            position = self.position[symbol]
            stoploss = position.get('stoploss', 0)
            
            if current_price > 0 and stoploss > 0 and current_price <= stoploss:
                logger.warning(f"{symbol} 触发止损: 当前价 {current_price} <= 止损价 {stoploss}")
                signal = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "action": "sell",
                    "price": current_price,
                    "reason": "stoploss_triggered",
                    "position_size": [1.0]
                }
                self.execute_sell(signal, symbol)
    
    def get_market_status(self):
        """
        获取市场状态
        :return: 市场状态评估结果
        """
        logger.info("===== 市场状态评估 =====")
        
        status_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "unknown",
            "symbols_status": {},
            "trending_count": 0,
            "ranging_count": 0,
            "declining_count": 0,
            "breakout_count": 0,
            "recommendation": "暂无建议"
        }
        
        if not self.symbols:
            return status_report
            
        # 评估每只股票
        for symbol in self.symbols:
            try:
                # 获取日线数据
                start_date, end_date = get_valid_date_range_str(30)  # 修复：解包日期范围
                df = self.api.get_daily_data(symbol, start_date=start_date, end_date=end_date)
                
                if df.empty:
                    logger.warning(f"股票 {symbol} 获取数据为空")
                    status_report["symbols_status"][symbol] = {"condition": "no_data", "error": "数据为空"}
                    continue
                
                # 计算缠论指标
                result = self.calculator.calculate(df)
                
                # 获取市场状态
                market_condition = self.calculator.determine_market_condition(result)
                
                # 记录状态
                status_report["symbols_status"][symbol] = {
                    "condition": market_condition,
                    "price": df.iloc[-1]['close'],
                    "volume": df.iloc[-1]['volume'],
                    "signal_strength": self.calculate_signal_strength(result)
                }
                
                # 统计状态数量
                if 'trending_up' in market_condition or 'up' in market_condition:
                    status_report["trending_count"] += 1
                elif 'ranging' in market_condition:
                    status_report["ranging_count"] += 1
                elif 'trending_down' in market_condition or 'down' in market_condition:
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
        total_symbols = len(self.symbols)
        if total_symbols > 0:
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
    
    def calculate_signal_strength(self, result_df):
        """
        计算信号强度
        :param result_df: 包含缠论指标的DataFrame
        :return: 信号强度(0-100)
        """
        if result_df.empty:
            return 0
        
        latest = result_df.iloc[-1]
        strength = 50  # 基准强度
        
        try:
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
        except Exception as e:
            logger.warning(f"计算信号强度失败: {str(e)}")
        
        return max(0, min(100, strength))  # 限制在0-100范围内
    
    def check_symbol(self, symbol):
        """
        检查单个股票
        :param symbol: 股票代码
        """
        try:
            # 获取分钟数据
            df = self.api.get_minute_data(symbol, period=self.minute_period, days=self.minute_days)
            
            if df.empty:
                logger.warning(f"股票 {symbol} 获取数据为空")
                return
            
            # 获取最新价格
            current_price = df.iloc[-1]['close']
            
            # 检查止损
            self.check_stoploss(symbol, current_price)
            
            # 分析市场状况
            market_condition = self.calculator.determine_market_condition(df)
            
            # 检查黑天鹅机会
            if self.check_black_swan_conditions(df):
                market_condition = "black_swan_buy"
                logger.info(f"检测到黑天鹅机会: {symbol}")
            
            # 生成交易信号
            signal = self.generate_signal(df, market_condition, symbol)
            
            # 检查是否为新信号
            last_signal = self.last_signals.get(symbol, {})
            if (last_signal.get('action') != signal['action'] or 
                last_signal.get('reason') != signal['reason'] or
                last_signal.get('price', 0) != signal['price']):
                
                logger.info(f"检测到新信号: {symbol} {signal['action']} ({signal['reason']}) @ {signal['price']}")
                self.last_signals[symbol] = signal
                
                # 执行交易
                if signal['action'] == 'buy':
                    self.execute_buy(signal, symbol)
                elif signal['action'] == 'sell':
                    self.execute_sell(signal, symbol)
            
        except Exception as e:
            logger.error(f"检查股票 {symbol} 失败: {str(e)}")
    
    def start_weekly_scan(self):
        """
        启动周线扫描模式
        """
        logger.info("===== 周线扫描模式 =====")
        
        scan_results = []
        
        # 获取周线数据
        for symbol in self.symbols:
            try:
                # 获取周线数据（最近2年）
                start_date, end_date = get_valid_date_range_str(365 * 2)  # 修复：解包日期范围
                weekly_df = self.api.get_weekly_data(symbol, start_date=start_date, end_date=end_date)
                
                if weekly_df.empty:
                    logger.warning(f"无法获取 {symbol} 的周线数据")
                    continue
                
                # 设置计算器为周线模式
                self.calculator.set_timeframe_params('weekly', {
                    'fractal_sensitivity': self.config.get('chanlun', {}).get('weekly_fractal_sensitivity', 2),
                    'pen_min_length': self.config.get('chanlun', {}).get('weekly_pen_min_length', 3),
                    'central_min_length': self.config.get('chanlun', {}).get('weekly_central_min_length', 3)
                })
                
                # 计算缠论指标
                result_df = self.calculator.calculate(weekly_df)
                
                # 分析市场状况
                market_condition = self.calculator.determine_market_condition(result_df)
                
                # 生成周线级别信号
                signal = self.generate_signal(result_df, market_condition, symbol)
                
                # 记录信号
                logger.info(f"周线扫描信号 - {symbol}: {signal['action']} ({signal['reason']})")
                scan_results.append({
                    'symbol': symbol,
                    'signal': signal,
                    'data': result_df
                })
                
                # 发送周线扫描报告
                self.notifier.send_weekly_scan_report(symbol, signal, result_df)
                
            except Exception as e:
                logger.error(f"周线扫描 {symbol} 失败: {str(e)}")
                scan_results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        logger.info(f"周线扫描完成，共处理 {len(scan_results)} 只股票")
        return scan_results
    
    def start(self):
        """
        启动监控
        """
        logger.info("===== 启动实时监控 =====")
        logger.info(f"监控间隔: {self.interval}秒")
        logger.info(f"监控股票: {', '.join(self.symbols)}")
        logger.info(f"初始资金: {self.total_capital}, 可用资金: {self.available_cash}")
        
        try:
            while True:
                # 检查是否为交易时间
                if not is_trading_hour():
                    logger.info("非交易时间，暂停监控")
                    time.sleep(60)  # 非交易时间每分钟检查一次
                    continue
                
                # 检查每个股票
                for symbol in self.symbols:
                    self.check_symbol(symbol)
                
                # 等待下一次检查
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("监控已手动停止")
        except Exception as e:
            logger.critical(f"监控异常终止: {str(e)}")
            # 发送错误通知
            self.notifier.send_error(f"监控异常终止: {str(e)}")

# 命令行测试
if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description='缠论监控测试工具')
    parser.add_argument('-s', '--symbol', required=True, help='股票代码')
    parser.add_argument('-c', '--capital', type=float, default=600000, help='初始资金')
    parser.add_argument('-m', '--mode', choices=['realtime', 'weekly_scan'], default='realtime', help='运行模式')
    
    args = parser.parse_args()
    
    # 加载系统配置
    system_config = load_config()
    
    # 创建API
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
    
    # 创建监控器
    monitor = ChanlunMonitor(
        system_config=system_config,
        api=api,
        calculator=calculator,
        notifier=notifier
    )
    
    # 设置初始资金
    monitor.total_capital = args.capital
    monitor.available_cash = args.capital
    
    # 添加监控股票
    monitor.add_symbol(args.symbol)
    
    # 根据模式选择运行
    if args.mode == 'weekly_scan':
        monitor.start_weekly_scan()
    else:
        monitor.start()