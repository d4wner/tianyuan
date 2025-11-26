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
from src.data_fetcher import StockDataFetcher as StockDataAPI
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
        计算仓位大小 - 修复版，增加周线级别顶底背驰的仓位管理策略
        :param symbol: 股票代码
        :param signal_type: 信号类型
        :return: 仓位比例
        """
        try:
            # 默认仓位设置
            if signal_type == 'first_buy' or signal_type == 'black_swan_buy':
                base_position = [0.1, 0.15]
            elif signal_type == 'second_buy':
                base_position = [0.4, 0.5]
            elif signal_type == 'third_buy':
                base_position = [0.2, 0.25]
            else:
                base_position = [0.3]  # 默认仓位
            
            # 检查是否是买入信号
            if 'buy' in signal_type.lower():
                # 获取周线数据
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')  # 获取90天周线数据
                weekly_df = self.api.get_weekly_data(symbol, start_date=start_date, end_date=end_date)
                
                if not weekly_df.empty:
                    # 创建缠论计算器并计算周线指标
                    weekly_calculator = ChanlunCalculator()
                    weekly_result = weekly_calculator.calculate_chanlun(weekly_df)
                    
                    # 检查周线级别是否有背驰
                    weekly_has_divergence = False
                    if not weekly_result.empty:
                        latest_weekly = weekly_result.iloc[-1]
                        weekly_has_divergence = 'divergence' in latest_weekly and \
                                              latest_weekly['divergence'] in ['bull', 'bullish', 'bottom', 'bear', 'bearish', 'top']
                    
                    # 如果周线级别没有背驰，则降低仓位为原来的1/4
                    if not weekly_has_divergence:
                        # 检查日线级别是否有背驰（仅在周线没有背驰的情况下才检查）
                        daily_df = self.api.get_daily_data(symbol, start_date=start_date, end_date=end_date, force_refresh=True)
                        if not daily_df.empty:
                            daily_calculator = ChanlunCalculator()
                            daily_result = daily_calculator.calculate_chanlun(daily_df)
                            
                            if not daily_result.empty:
                                latest_daily = daily_result.iloc[-1]
                                daily_has_divergence = 'divergence' in latest_daily and \
                                                      latest_daily['divergence'] in ['bull', 'bullish', 'bottom', 'bear', 'bearish', 'top']
                                
                                # 如果只有日线级别有背驰，降低仓位为原来的1/4
                                if daily_has_divergence:
                                    logger.info(f"{symbol} 仅日线级别有背驰，仓位调整为原来的1/4")
                                    # 将仓位范围缩小为原来的1/4
                                    adjusted_position = []
                                    for p in base_position:
                                        adjusted_position.append(p / 4)
                                    return adjusted_position
            
            # 默认返回原始仓位（如果周线有背驰或计算失败）
            return base_position
            
        except Exception as e:
            logger.warning(f"计算周线背驰调整仓位失败: {str(e)}")
            # 发生异常时返回默认仓位
            return [0.2]  # 返回保守的默认仓位
    
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
        
        # 优先检查market_condition中是否包含明确的买入信号，特别是多级别联立信号
        if market_condition and ('multi_timeframe_buy' in market_condition or 'buy' in market_condition):
            # 获取最新价格
            current_price = data.iloc[-1]['close']
            
            # 初始化信号
            signal = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "market_condition": market_condition,
                "price": current_price,
                "target_price": current_price * 1.05,
                "stoploss": current_price * 0.97,
                "reason": market_condition,
                "action": "buy",
                "signal_type": "multi_timeframe" if 'multi_timeframe_buy' in market_condition else "normal",
                "strength": 85 if 'multi_timeframe_buy' in market_condition else 70,
                "position_size": [0.3, 0.4] if 'multi_timeframe_buy' in market_condition else self.calculate_position_size(symbol, market_condition),
                "valid_until": (datetime.now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"{symbol} 触发市场条件信号: {market_condition}")
            return signal
        
        # 然后尝试识别复合信号
        composite_signal = self.identify_composite_signals(data, symbol)
        if composite_signal['action'] != 'hold':
            return composite_signal
            
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
    
    def identify_composite_signals(self, data, symbol):
        """
        识别复合信号
        :param data: 股票数据
        :param symbol: 股票代码
        :return: 复合信号字典
        """
        try:
            # 获取当前价格
            current_price = data.iloc[-1]['close']
            
            # 初始化信号
            signal = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "action": "hold",
                "reason": "no_composite_signal",
                "price": current_price,
                "target_price": current_price * 1.05,
                "stoploss": current_price * 0.97,
                "signal_type": "normal",
                "strength": 50,
                "valid_until": (datetime.now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 1. 检查日线级别二买 + 分钟级别底背驰复合信号
            daily_df = self.api.get_daily_data(symbol, days=60, force_refresh=True)
            if not daily_df.empty:
                # 使用self.calculator而不是创建新实例，确保配置一致性
                daily_result = self.calculator.calculate_chanlun(daily_df)
                
                if not daily_result.empty:
                    latest_daily = daily_result.iloc[-1]
                    
                    # 检查日线二买条件
                    is_second_buy = latest_daily.get('buy_signal') == 'second_buy'
                    
                    if is_second_buy:
                        # 获取15分钟数据检查底背驰
                        minute_15m_df = self.api.get_minute_data(symbol, period='15m', days=5)
                        if not minute_15m_df.empty:
                            # 设置15分钟参数
                            self.calculator.set_timeframe_params('15m', {
                                'fractal_sensitivity': self.config.get('chanlun', {}).get('minute_15_fractal_sensitivity', 2),
                                'pen_min_length': self.config.get('chanlun', {}).get('minute_15_pen_min_length', 3),
                                'central_min_length': self.config.get('chanlun', {}).get('minute_15_central_min_length', 3)
                            })
                            minute_result = self.calculator.calculate_chanlun(minute_15m_df)
                            
                            if not minute_result.empty:
                                latest_minute = minute_result.iloc[-1]
                                
                                # 检查分钟级别底背驰
                                has_bottom_divergence = 'divergence' in latest_minute and \
                                                      latest_minute['divergence'] in ['bull', 'bullish', 'bottom']
                                
                                if has_bottom_divergence:
                                    signal.update({
                                        "action": "buy",
                                        "reason": "daily_second_buy_plus_minute_divergence",
                                        "signal_type": "daily_second_buy_minute_divergence",
                                        "strength": 85,  # 高信号强度
                                        "position_size": [0.3, 0.4]  # 较大仓位
                                    })
                                    logger.info(f"{symbol} 触发复合信号: 日线二买+分钟级别底背驰")
                                    self.save_signal(signal)  # 保存信号
                                    return signal
            
            # 2. 检查周线底分型 + 日线突破中枢复合信号
            weekly_df = self.api.get_weekly_data(symbol, days=120)
            if not weekly_df.empty:
                # 设置周线参数
                self.calculator.set_timeframe_params('weekly', {
                    'fractal_sensitivity': self.config.get('chanlun', {}).get('weekly_fractal_sensitivity', 2),
                    'pen_min_length': self.config.get('chanlun', {}).get('weekly_pen_min_length', 3),
                    'central_min_length': self.config.get('chanlun', {}).get('weekly_central_min_length', 3)
                })
                weekly_result = self.calculator.calculate_chanlun(weekly_df)
                
                if not weekly_result.empty:
                    latest_weekly = weekly_result.iloc[-1]
                    
                    # 检查周线底分型
                    has_weekly_bottom_fractal = latest_weekly.get('bottom_fractal', False)
                    
                    if has_weekly_bottom_fractal:
                        if not 'daily_result' in locals():
                            daily_df = self.api.get_daily_data(symbol, days=60, force_refresh=True)
                            daily_result = self.calculator.calculate_chanlun(daily_df)
                        
                        if not daily_result.empty:
                            latest_daily = daily_result.iloc[-1]
                            
                            # 检查日线突破中枢
                            current_price = latest_daily['close']
                            central_high = latest_daily.get('central_bank_high', current_price)
                            if current_price > central_high:
                                signal.update({
                                    "action": "buy",
                                    "reason": "weekly_bottom_fractal_plus_daily_breakthrough",
                                    "signal_type": "weekly_fractal_daily_breakthrough",
                                    "strength": 90,  # 非常高的信号强度
                                    "position_size": [0.4, 0.5]  # 大仓位
                                })
                                logger.info(f"{symbol} 触发复合信号: 周线底分型+日线突破中枢")
                                self.save_signal(signal)  # 保存信号
                                return signal
            
            # 3. 检查日线底背驰 + 分钟级别放量底分复合信号
            if 'daily_result' in locals() and not daily_result.empty:
                latest_daily = daily_result.iloc[-1]
                
                # 检查日线底背驰
                has_daily_divergence = 'divergence' in latest_daily and \
                                      latest_daily['divergence'] in ['bull', 'bullish', 'bottom']
                
                if has_daily_divergence:
                    # 获取5分钟数据检查放量底分
                    minute_5m_df = self.api.get_minute_data(symbol, period='5m', days=2)
                    if not minute_5m_df.empty:
                        # 设置5分钟参数
                        self.calculator.set_timeframe_params('5m', {
                            'fractal_sensitivity': self.config.get('chanlun', {}).get('minute_5_fractal_sensitivity', 2),
                            'pen_min_length': self.config.get('chanlun', {}).get('minute_5_pen_min_length', 3),
                            'central_min_length': self.config.get('chanlun', {}).get('minute_5_central_min_length', 3)
                        })
                        minute_result = self.calculator.calculate_chanlun(minute_5m_df)
                        
                        if not minute_result.empty:
                            latest_minute = minute_result.iloc[-1]
                            
                            # 检查放量底分型
                            has_bottom_fractal = latest_minute.get('bottom_fractal', False)
                            has_volume_expansion = 'volume' in latest_minute and 'avg_volume' in latest_minute and \
                                                 latest_minute['volume'] > latest_minute['avg_volume'] * 1.5
                            
                            if has_bottom_fractal and has_volume_expansion:
                                signal.update({
                                    "action": "buy",
                                    "reason": "daily_divergence_plus_minute_volume_fractal",
                                    "signal_type": "daily_divergence_minute_volume",
                                    "strength": 80,
                                    "position_size": [0.25, 0.35]
                                })
                                logger.info(f"{symbol} 触发复合信号: 日线底背驰+分钟级别放量底分")
                                self.save_signal(signal)  # 保存信号
                                return signal
        
        except Exception as e:
            logger.error(f"识别复合信号失败: {str(e)}")
        
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
                df = self.api.get_daily_data(symbol, start_date=start_date, end_date=end_date, force_refresh=True)
                
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
        计算信号强度，增强对背驰和放量条件的权重
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
            
            # 新增：根据背驰增加强度（高权重）
            if 'divergence' in latest:
                if latest['divergence'] in ['bull', 'bullish', 'bottom']:  # 底背驰
                    strength += 30  # 背驰具有较高权重
                elif latest['divergence'] in ['bear', 'bearish', 'top']:  # 顶背驰
                    strength -= 30
            
            # 新增：根据放量情况增加强度
            if 'volume' in latest and 'avg_volume' in latest:
                volume_ratio = latest['volume'] / latest['avg_volume']
                if volume_ratio > 1.5:  # 放量超过50%
                    # 结合当前趋势判断放量的方向性影响
                    if latest.get('pen_type') == 'up' or latest.get('segment_type') == 'up':
                        strength += 15  # 上涨趋势中放量
                    elif latest.get('pen_type') == 'down' or latest.get('segment_type') == 'down':
                        strength -= 15  # 下跌趋势中放量
                elif volume_ratio < 0.5:  # 缩量
                    # 缩量回调通常是买入机会
                    if latest.get('pen_type') == 'down' or latest.get('segment_type') == 'down':
                        strength += 8  # 下跌趋势中缩量
        except Exception as e:
            logger.warning(f"计算信号强度失败: {str(e)}")
        
        return max(0, min(100, strength))  # 限制在0-100范围内
    
    def check_symbol(self, symbol):
        """
        检查单个股票
        :param symbol: 股票代码
        """
        try:
            # 获取默认分钟数据
            df = self.api.get_minute_data(symbol, period=self.minute_period, days=self.minute_days)
            
            if df.empty:
                logger.warning(f"股票 {symbol} 获取数据为空")
                return
            
            # 获取最新价格
            current_price = df.iloc[-1]['close']
            
            # 检查止损
            self.check_stoploss(symbol, current_price)
            
            # 多级别联立分析逻辑 - 实现15分钟底背驰寻找5分钟放量底分
            market_condition = self.analyze_multi_timeframe_signal(symbol, df)
            
            # 如果没有找到复合信号，使用默认市场状况分析
            if not market_condition:
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
                
                # 保存所有信号，包括hold信号，便于分析
                self.save_signal(signal)
        except Exception as e:
            logger.error(f"检查股票 {symbol} 时出错: {str(e)}")
        finally:
            pass
            
    def analyze_multi_timeframe_signal(self, symbol, default_df):
        """
        多级别联立分析，寻找复合信号
        - 通过15分钟底背驰寻找5分钟放量底分的策略
        
        :param symbol: 股票代码
        :param default_df: 默认时间级别的数据
        :return: 复合信号类型或None
        """
        try:
            # 获取15分钟级别数据
            df_15m = self.api.get_minute_data(symbol, period='15m', days=2)  # 2天的15分钟数据
            if df_15m.empty:
                logger.debug(f"无法获取 {symbol} 的15分钟数据")
                return None
            
            # 获取5分钟级别数据
            df_5m = self.api.get_minute_data(symbol, period='5m', days=1)  # 1天的5分钟数据
            if df_5m.empty:
                logger.debug(f"无法获取 {symbol} 的5分钟数据")
                return None
            
            # 为不同时间周期设置参数
            self.calculator.set_timeframe_params('15m', {
                'fractal_sensitivity': self.config.get('chanlun', {}).get('minute_15_fractal_sensitivity', 2),
                'pen_min_length': self.config.get('chanlun', {}).get('minute_15_pen_min_length', 3),
                'central_min_length': self.config.get('chanlun', {}).get('minute_15_central_min_length', 3)
            })
            
            self.calculator.set_timeframe_params('5m', {
                'fractal_sensitivity': self.config.get('chanlun', {}).get('minute_5_fractal_sensitivity', 2),
                'pen_min_length': self.config.get('chanlun', {}).get('minute_5_pen_min_length', 3),
                'central_min_length': self.config.get('chanlun', {}).get('minute_5_central_min_length', 3)
            })
            
            # 计算15分钟级别缠论指标
            result_15m = self.calculator.calculate(df_15m)
            
            # 计算5分钟级别缠论指标
            result_5m = self.calculator.calculate(df_5m)
            
            # 检查15分钟底背驰
            has_15m_bottom_divergence = self.check_divergence(result_15m, 'bottom')
            
            # 检查5分钟放量底分型
            has_5m_bottom_fractal = self.check_volume_fractal(result_5m, 'bottom')
            
            # 如果同时满足15分钟底背驰和5分钟放量底分，返回复合买入信号
            if has_15m_bottom_divergence and has_5m_bottom_fractal:
                logger.info(f"检测到复合买入信号: {symbol} - 15分钟底背驰 + 5分钟放量底分型")
                return "multi_timeframe_buy"  # 复合买入信号
                
            # 也可以添加其他复合信号的判断逻辑
            # 例如日线二买 + 分钟级别底背驰
            
            return None
            
        except Exception as e:
            logger.error(f"多级别联立分析失败: {str(e)}")
            return None
            
    def check_divergence(self, result_df, divergence_type):
        """
        检查是否存在指定类型的背驰
        
        :param result_df: 计算结果DataFrame
        :param divergence_type: 'top' 或 'bottom'
        :return: 是否存在背驰
        """
        if result_df.empty:
            return False
            
        try:
            # 检查是否有背驰标识列
            if 'divergence' not in result_df.columns:
                # 如果没有直接的背驰列，可以基于其他指标计算
                # 这里使用简化的判断逻辑，实际项目中可能需要更复杂的算法
                return False
                
            # 检查最近的K线是否有背驰
            latest = result_df.iloc[-1]
            
            # 修复：支持多种背驰标记格式
            # 兼容calculator.py中的'bull'/'bear'格式和monitor.py中预期的'top'/'bottom'格式
            if divergence_type == 'bottom':
                # 底背驰可以是'bull'、'bullish'或任何表示底部的标记
                return latest['divergence'] in ['bull', 'bullish', 'bottom']
            elif divergence_type == 'top':
                # 顶背驰可以是'bear'、'bearish'或任何表示顶部的标记
                return latest['divergence'] in ['bear', 'bearish', 'top']
                
            return False
            
        except Exception as e:
            logger.warning(f"检查背驰失败: {str(e)}")
            return False
            
    def check_volume_fractal(self, result_df, fractal_type):
        """
        检查是否存在放量的指定类型分型
        
        :param result_df: 计算结果DataFrame
        :param fractal_type: 'top' 或 'bottom'
        :return: 是否存在放量分型
        """
        if result_df.empty:
            return False
            
        try:
            # 检查最近的分型
            if fractal_type == 'bottom' and 'bottom_fractal' in result_df.columns:
                # 找到最近的底分型
                latest_fractal = result_df[result_df['bottom_fractal'] == True].tail(1)
                if latest_fractal.empty:
                    return False
                    
                # 检查是否放量
                fractal_idx = latest_fractal.index[0]
                if fractal_idx > 5:  # 确保有足够的数据计算平均成交量
                    avg_volume = result_df.iloc[fractal_idx-5:fractal_idx]['volume'].mean()
                    current_volume = result_df.iloc[fractal_idx]['volume']
                    
                    # 放量阈值，可从配置中读取
                    volume_threshold = self.config.get('chanlun', {}).get('volume_increase_threshold', 1.5)
                    return current_volume > avg_volume * volume_threshold
                    
            return False
            
        except Exception as e:
            logger.warning(f"检查放量分型失败: {str(e)}")
            return False
    
    def save_signal(self, signal):
        """
        保存交易信号到文件，确保所有信号都被记录
        :param signal: 交易信号字典
        """
        try:
            # 创建信号目录
            signals_dir = os.path.join(self.config.get('output_dir', 'output'), 'signals')
            os.makedirs(signals_dir, exist_ok=True)
            
            # 生成文件名（按日期分组）
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"signals_{date_str}.json"
            filepath = os.path.join(signals_dir, filename)
            
            # 读取现有信号
            existing_signals = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_signals = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_signals = []
            
            # 添加新信号
            existing_signals.append(signal)
            
            # 保存回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing_signals, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存信号失败: {str(e)}")
            
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