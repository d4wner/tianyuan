#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主交易系统模块

整合所有交易相关模块，实现完整的交易信号检测和输出流程。
按照"数据源→周线多头→日线买点判定→中级别确认→小级别买点→建仓/加仓→止盈"的顺序执行。

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
from typing import Dict, Optional, Union, Tuple, List, Any

# 导入子模块
from .data_validator import DataValidator
from .weekly_trend_detector import WeeklyTrendDetector
from .daily_buy_signal_detector import BuySignalDetector
from .signal_formatter import SignalFormatter
from .invalid_signal_filter import InvalidSignalFilter
from .minute_position_allocator import MinutePositionAllocator
from .position_addition_rules import PositionAdditionRules
from .profit_taking_rules import ProfitTakingRules
# 新增：导入日线前置条件校验模块
from .daily_precondition_checker import (
    check_daily_precondition,
    short_cycle_trade_logic,
    save_daily_pre_pass_status,
    load_daily_pre_pass_status
)
# 新增：导入实盘容错模块
from .trading_executor import TradingExecutor
# 可选模块，根据实际情况导入
# from divergence_detector import DivergenceDetector
# from data_fetcher import StockDataFetcher

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MainTradingSystem:
    """主交易系统类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化主交易系统
        
        Args:
            config: 系统配置参数
        """
        logger.info("初始化主交易系统...")
        
        # 使用默认配置或用户自定义配置
        self.config = config or {}
        
        # 初始化各个子模块
        self.data_validator = DataValidator()
        self.weekly_trend_detector = WeeklyTrendDetector()
        self.daily_buy_signal_detector = BuySignalDetector()
        self.signal_formatter = SignalFormatter()
        self.invalid_signal_filter = InvalidSignalFilter()
        self.minute_position_allocator = MinutePositionAllocator()
        self.position_addition_rules = PositionAdditionRules()
        self.profit_taking_rules = ProfitTakingRules()
        # 新增：初始化实盘交易执行器
        self.trading_executor = TradingExecutor()
        
        # 可选模块，根据实际情况初始化
        # self.divergence_detector = DivergenceDetector()
        # self.data_fetcher = StockDataFetcher()
        
        # 系统状态标志
        self.system_status = {
            "initialized": True,
            "data_fetched": False,
            "data_validated": False,
            "signals_generated": False,
            "last_run_time": None
        }
        
        # 数据存储
        self.daily_data = None
        self.weekly_data = None
        self.minute_data = {
            "30min": None,
            "15min": None,
            "5min": None
        }
        
        # 检测结果存储
        self.validation_result = None
        self.weekly_trend_result = None
        self.daily_buy_result = None
        self.trading_signals = []
        self.executed_signals = []  # 新增：已执行的交易信号
        self.position_allocation = {}
        self.current_positions = {}
        self.total_pnl = 0.0  # 新增：总盈亏
        
        # 新增：日线前置条件状态
        self.daily_pre_pass = False
        self.daily_pre_check_details = {}
        self.daily_pre_check_date = ""
        
        logger.info("主交易系统初始化完成")
    
    def run(self, stock_code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, any]:
        """运行完整的交易信号检测流程
        
        按照以下步骤执行：
        1. 数据获取
        2. 数据源有效性校验
        3. 周线多头趋势判定
        4. 日线买点判定
        5. 分钟级别子分仓判定
        6. 向下加仓条件检查
        7. 止盈条件检查
        8. 无效信号过滤
        9. 生成交易信号并格式化
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            包含所有检测结果和交易信号的字典
        """
        logger.info(f"开始运行交易信号检测流程，股票代码: {stock_code}")
        
        # 初始化结果字典
        final_result = {
            "stock_code": stock_code,
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "system_status": "success",
            "validation_result": None,
            "weekly_trend_result": None,
            "daily_buy_result": None,
            "minute_allocation_result": None,
            "addition_check_result": None,
            "stop_profit_result": None,
            "trading_signals": [],
            "report": None,
            "errors": []
        }
        
        try:
            # 1. 数据获取
            logger.info("第1步：获取交易数据...")
            self._fetch_data(stock_code, start_date, end_date)
            self.system_status["data_fetched"] = True
            
            # 2. 数据源有效性校验
            logger.info("第2步：数据源有效性校验...")
            self._validate_data()
            self.system_status["data_validated"] = True
            final_result["validation_result"] = self.validation_result
            
            # 检查数据是否有效
            if not self.validation_result["all_valid"]:
                error_msg = f"数据源无效: {self.validation_result['error_message']}"
                logger.error(error_msg)
                final_result["system_status"] = "data_invalid"
                final_result["errors"].append(error_msg)
                
                # 生成无效信号并结束流程
                invalid_signal = self.signal_formatter.format_invalid_signal(
                    f"数据源不足: 日线{'满足' if self.validation_result['daily_valid'] else '不足'}, "
                    f"周线{'满足' if self.validation_result['weekly_valid'] else '不足'}"
                )
                final_result["trading_signals"].append(invalid_signal)
                return final_result
            
            # 3. 周线多头趋势判定
            logger.info("第3步：周线多头趋势判定...")
            self._detect_weekly_trend()
            final_result["weekly_trend_result"] = self.weekly_trend_result
            
            # 4. 日线买点判定
            logger.info("第4步：日线买点判定...")
            self._detect_daily_buy_signals()
            final_result["daily_buy_result"] = self.daily_buy_result
            
            # 新增：4.1 执行日线前置条件校验
            logger.info("第4.1步：执行日线前置条件校验...")
            self._check_daily_precondition(stock_code)
            final_result["daily_pre_pass"] = self.daily_pre_pass
            final_result["daily_pre_check_details"] = self.daily_pre_check_details
            
            # 5. 分钟级别子分仓判定（已集成日线前置条件检查）
            logger.info("第5步：分钟级别子分仓判定...")
            self._allocate_minute_positions()
            final_result["minute_allocation_result"] = self.position_allocation
            
            # 6. 向下加仓条件检查
            logger.info("第6步：向下加仓条件检查...")
            self._check_addition_conditions()
            final_result["addition_check_result"] = self.addition_check_result
            
            # 7. 止盈条件检查
            logger.info("第7步：止盈条件检查...")
            self._check_stop_profit_conditions()
            final_result["stop_profit_result"] = self.stop_profit_result
            
            # 8. 生成交易信号
            logger.info("第8步：生成交易信号...")
            self._generate_trading_signals()
            self.system_status["signals_generated"] = True
            final_result["trading_signals"] = self.trading_signals
            
            # 9. 生成综合报告
            logger.info("第9步：生成综合报告...")
            report = self._generate_report()
            final_result["report"] = report
            
            # 新增：第10步：执行交易信号（实盘容错执行）
            logger.info("第10步：执行交易信号（实盘容错执行）...")
            executed_signals = self._execute_trading_signals()
            final_result["executed_signals"] = executed_signals
            
            logger.info("交易信号检测流程完成")
            
        except Exception as e:
            error_msg = f"运行过程中发生错误: {str(e)}"
            logger.error(error_msg)
            final_result["system_status"] = "error"
            final_result["errors"].append(error_msg)
        
        # 更新最后运行时间
        self.system_status["last_run_time"] = pd.Timestamp.now()
        
        return final_result
    
    def _execute_trading_signals(self) -> List[Dict[str, Any]]:
        """执行交易信号（带实盘容错机制）
        
        Returns:
            已执行的交易信号列表
        """
        executed_signals = []
        
        for signal in self.trading_signals:
            # 解析交易信号，转换为订单格式
            order_info = self._parse_signal_to_order(signal)
            
            if order_info:
                # 使用实盘交易执行器执行订单
                success, result = self.trading_executor.execute_order(order_info)
                
                # 更新执行结果到信号
                executed_signal = signal.copy()
                executed_signal["execution_result"] = {
                    "success": success,
                    "details": result,
                    "executed_time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                executed_signals.append(executed_signal)
                self.executed_signals.append(executed_signal)
                
                logger.info(f"信号执行结果: {'成功' if success else '失败'}")
                logger.info(f"执行详情: {result}")
            else:
                logger.warning(f"无法解析信号为订单: {signal}")
        
        # 更新每日盈亏（模拟）
        # 实际应用中，这里应该从交易接口获取真实的盈亏数据
        self._simulate_pnl_update()
        
        return executed_signals
    
    def _parse_signal_to_order(self, signal: str) -> Optional[Dict[str, Any]]:
        """将交易信号解析为订单格式
        
        Args:
            signal: 交易信号字符串
            
        Returns:
            订单信息字典，如果无法解析则返回None
        """
        try:
            # 从信号中提取关键信息（实际应用中需要更复杂的解析逻辑）
            # 这里使用简单的字符串匹配作为示例
            
            # 提取股票代码
            import re
            stock_code_match = re.search(r'股票代码:([\d]+)', signal)
            if not stock_code_match:
                return None
            
            stock_code = stock_code_match.group(1)
            
            # 提取操作类型（买/卖）
            if '建仓' in signal or '加仓' in signal or '买入' in signal:
                order_type = 'buy'
            elif '止盈' in signal or '卖出' in signal:
                order_type = 'sell'
            else:
                return None
            
            # 提取价格和数量（实际应用中需要更复杂的解析或从行情数据获取）
            # 这里使用模拟数据
            price_match = re.search(r'价格:([\d.]+)', signal)
            price = float(price_match.group(1)) if price_match else 3.0
            
            volume_match = re.search(r'数量:(\d+)', signal)
            volume = int(volume_match.group(1)) if volume_match else 1000
            
            return {
                "symbol": stock_code,
                "type": order_type,
                "price": price,
                "volume": volume
            }
        except Exception as e:
            logger.error(f"解析信号为订单失败: {str(e)}")
            return None
    
    def _simulate_pnl_update(self) -> None:
        """模拟盈亏更新（实际应用中替换为真实API调用）
        
        模拟每日盈亏变化，用于测试熔断机制
        """
        import random
        
        # 模拟盈亏变化（-0.01到0.01之间的随机值）
        pnl_change = random.uniform(-0.01, 0.01)
        self.total_pnl += pnl_change
        
        # 更新交易执行器的每日盈亏
        self.trading_executor.update_daily_pnl(self.total_pnl)
        
        logger.info(f"盈亏更新（模拟）: 总盈亏={self.total_pnl:.4f}, 本次变化={pnl_change:.4f}")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """获取交易状态
        
        Returns:
            交易状态字典
        """
        return {
            "system_status": self.system_status,
            "trading_executor_status": self.trading_executor.get_trading_status(),
            "total_pnl": self.total_pnl,
            "current_positions": self.current_positions,
            "executed_signals": self.executed_signals[-10:]  # 返回最近10个已执行信号
        }
    
    def reset_trading_status(self) -> None:
        """重置交易状态
        
        包括重置熔断、盈亏等
        """
        self.trading_executor.reset_daily_status()
        self.total_pnl = 0.0
        self.executed_signals = []
        logger.info("交易状态已重置")

    def _fetch_data(self, stock_code: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """获取日线和周线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        """
        # 假设数据获取器有这些方法
        logger.info(f"正在获取{stock_code}的交易数据...")
        
        # 获取日线数据，为了确保至少60根，往前获取更多数据
        try:
            # 这里使用假数据进行测试，实际应调用data_fetcher获取真实数据
            self.daily_data = self._generate_mock_daily_data()
            logger.info(f"成功获取日线数据，共{len(self.daily_data)}条")
        except Exception as e:
            logger.error(f"获取日线数据失败: {str(e)}")
            raise Exception(f"获取日线数据失败: {str(e)}")
        
        # 获取周线数据，为了确保至少52根，往前获取更多数据
        try:
            # 这里使用假数据进行测试，实际应调用data_fetcher获取真实数据
            self.weekly_data = self._generate_mock_weekly_data()
            logger.info(f"成功获取周线数据，共{len(self.weekly_data)}条")
        except Exception as e:
            logger.error(f"获取周线数据失败: {str(e)}")
            raise Exception(f"获取周线数据失败: {str(e)}")
        
        # 获取分钟级别数据
        try:
            # 生成模拟的分钟级别数据
            self.minute_data["30min"] = self._generate_mock_minute_data("30min")
            self.minute_data["15min"] = self._generate_mock_minute_data("15min")
            self.minute_data["5min"] = self._generate_mock_minute_data("5min")
            logger.info("成功获取分钟级别数据")
        except Exception as e:
            logger.error(f"获取分钟数据失败: {str(e)}")
            raise Exception(f"获取分钟数据失败: {str(e)}")
    
    def _validate_data(self):
        """验证数据源有效性
        
        Returns:
            包含验证结果的字典
        """
        logger.info("验证数据源有效性...")
        
        # 验证日线数据
        daily_valid = self.data_validator.validate_daily_data(self.daily_data)
        
        # 验证周线数据
        weekly_valid = self.data_validator.validate_weekly_data(self.weekly_data)
        
        # 综合验证结果
        all_valid = daily_valid and weekly_valid
        
        # 构建验证结果
        self.validation_result = {
            "daily_valid": daily_valid,
            "weekly_valid": weekly_valid,
            "all_valid": all_valid,
            "error_message": "" if all_valid else (
                f"{'日线数据不足' if not daily_valid else ''}"
                f"{'，' if not daily_valid and not weekly_valid else ''}"
                f"{'周线数据不足' if not weekly_valid else ''}"
            )
        }
        
        logger.info(f"数据源验证结果: 日线{'有效' if daily_valid else '无效'}, 周线{'有效' if weekly_valid else '无效'}")
        return self.validation_result
    
    def _detect_weekly_trend(self):
        """检测周线多头趋势
        
        Returns:
            包含周线趋势检测结果的字典
        """
        logger.info("检测周线多头趋势...")
        
        # 调用周线趋势检测器
        try:
            self.weekly_trend_result = self.weekly_trend_detector.detect_weekly_bullish_trend(self.weekly_data)
            logger.info(f"周线多头趋势检测结果: {self.weekly_trend_result.get('status', '未知')}")
        except Exception as e:
            logger.error(f"周线趋势检测失败: {str(e)}")
            # 创建失败的结果
            self.weekly_trend_result = {
                "bullish_trend": False,
                "status": "检测失败",
                "error": str(e)
            }
        
        return self.weekly_trend_result
    
    def _detect_daily_buy_signals(self):
        """检测日线买点信号
        
        Returns:
            包含日线买点检测结果的字典
        """
        logger.info("检测日线买点信号...")
        
        # 调用日线买点检测器
        try:
            # 这里传递必要的参数，包括日线数据和可能需要的背驰检测结果
            self.daily_buy_result = self.daily_buy_signal_detector.detect_all_buy_signals(self.daily_data)
            
            # 确定最强信号和优先级
            self._determine_strongest_signal()
            
            logger.info(f"日线买点检测结果: 最强信号为{self.daily_buy_result.get('strongest_signal', '无')}")
        except Exception as e:
            logger.error(f"日线买点检测失败: {str(e)}")
            # 创建失败的结果
            self.daily_buy_result = {
                "strongest_signal": "无买点",
                "signal_type_priority": "无",
                "error": str(e)
            }
        
        return self.daily_buy_result
    
    def _determine_strongest_signal(self):
        """确定最强的日线买点信号
        
        根据优先级（二买>一买/三买>反抽）确定最强信号
        """
        # 优先级映射
        signal_priority = {
            "日线二买": (1, "核心"),
            "日线一买": (2, "辅助"),
            "日线三买": (2, "辅助"),
            "破中枢反抽": (3, "兜底")
        }
        
        # 检查每种买点信号
        signals = self.daily_buy_result.get("signals", {})
        strongest_signal = "无买点"
        highest_priority = float('inf')
        signal_type_priority = "无"
        
        # 按照优先级检查
        for signal_name, priority_info in signal_priority.items():
            # 根据signal_name映射到signals字典中的键
            signal_key_map = {
                "日线二买": "second_buy",
                "日线一买": "first_buy",
                "日线三买": "third_buy",
                "破中枢反抽": "reverse_pullback"
            }
            
            signal_key = signal_key_map.get(signal_name)
            if signal_key and signals.get(signal_key, {}).get("detected", False):
                priority_num = priority_info[0]
                if priority_num < highest_priority:
                    highest_priority = priority_num
                    strongest_signal = signal_name
                    signal_type_priority = priority_info[1]
        
        # 更新日线买点检测结果
        self.daily_buy_result["strongest_signal"] = strongest_signal
        self.daily_buy_result["signal_type_priority"] = signal_type_priority
        
        # 记录满足条件的信号数量
        satisfied_count = sum(1 for signal in signals.values() if signal.get("detected", False))
        self.daily_buy_result["satisfied_signals_count"] = satisfied_count
    
    def _check_daily_precondition(self, stock_code: str):
        """执行日线前置条件校验
        
        新增函数：根据当前日期和时间，确定是否需要执行前置条件校验
        并更新self.daily_pre_pass状态
        
        Args:
            stock_code: 股票代码
        """
        import datetime
        logger.info("检查日线前置条件...")
        
        # 获取当前日期和时间
        now = datetime.datetime.now()
        today_date = now.strftime("%Y%m%d")
        
        # 交易时间判断
        is_trading_hours = 9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30)
        is_after_market = 15 <= now.hour < 16 and now.minute <= 30
        
        # 如果是交易时间（9:30-15:00），从缓存读取前置条件状态
        if is_trading_hours:
            logger.info("当前为交易时间，从缓存读取日线前置条件状态")
            saved_status = load_daily_pre_pass_status(stock_code)
            
            if saved_status is not None:
                self.daily_pre_pass = saved_status
                logger.info(f"读取到日线前置条件状态: {'通过' if saved_status else '不通过'}")
            else:
                logger.warning("未找到缓存的日线前置条件状态，默认不通过")
                self.daily_pre_pass = False
        
        # 如果是收盘后（15:00-15:30），执行前置条件校验并保存
        elif is_after_market and self.daily_pre_check_date != today_date:
            logger.info("当前为收盘后时间，执行日线前置条件校验")
            
            # 准备前置条件校验所需参数
            # 1. 获取近1季度中枢列表
            central_banks_list = []
            if self.daily_buy_result and "signals" in self.daily_buy_result:
                # 从日线买点检测结果中提取中枢信息
                for signal_type in self.daily_buy_result["signals"].values():
                    if isinstance(signal_type, dict) and "central_banks" in signal_type:
                        central_banks_list.extend(signal_type["central_banks"])
                # 确保中枢列表中的每个中枢都有is_valid字段
                for central in central_banks_list:
                    if "is_valid" not in central:
                        # 如果没有is_valid字段，根据原框架逻辑判断
                        central["is_valid"] = central.get("valid", False)
            
            # 2. 获取日线信号基础状态
            strongest_signal = self.daily_buy_result.get("strongest_signal", "无买点") if self.daily_buy_result else "无买点"
            signal_basic_status = "潜在监控" if strongest_signal != "无买点" else "完全无效"
            
            # 3. 获取波动等级（假设从BuySignalDetector中获取）
            volatility_level = "medium"  # 默认值
            if hasattr(self.daily_buy_signal_detector, "volatility_level"):
                vol_level_obj = self.daily_buy_signal_detector.volatility_level
                if vol_level_obj:
                    volatility_level = vol_level_obj.value.lower() if hasattr(vol_level_obj, "value") else "medium"
            
            # 4. 计算数据完整度
            data_completeness = 95.0  # 默认值
            if self.validation_result:
                # 根据验证结果计算数据完整度
                valid_count = sum(1 for key in ['daily_valid', 'weekly_valid'] if self.validation_result.get(key, False))
                data_completeness = (valid_count / 2) * 100
            
            # 执行前置条件校验
            passed, reason, details = check_daily_precondition(
                central_banks_list,
                signal_basic_status,
                volatility_level,
                data_completeness
            )
            
            # 更新状态
            self.daily_pre_pass = passed
            self.daily_pre_check_details = details
            self.daily_pre_check_date = today_date
            
            # 保存到缓存
            save_daily_pre_pass_status(stock_code, passed, today_date)
            
            logger.info(f"日线前置条件校验结果: {'通过' if passed else '不通过'}，原因: {reason}")
        
        # 非交易时间和收盘后时间范围外，不执行任何操作
        else:
            logger.info("当前为非交易时间，不执行日线前置条件相关操作")
            self.daily_pre_pass = False
    
    def _allocate_minute_positions(self):
        """执行分钟级别子仓位分配
        
        根据日线买点类型和分钟级别的信号，进行子仓位分配
        新增：集成日线前置条件校验
        """
        logger.info("执行分钟级别子仓位分配...")
        
        # 初始化仓位分配结果
        self.position_allocation = {
            "30min": 0.0,
            "15min": 0.0,
            "5min": 0.0,
            "primary_allocation": None,
            "allocation_details": {},
            "total": 0.0,
            "daily_pre_pass": self.daily_pre_pass  # 新增：记录日线前置条件状态
        }
        
        # 新增：检查日线前置条件是否通过，不通过则直接返回
        if not self.daily_pre_pass:
            logger.info("日线前置条件不通过，不执行短周期层逻辑")
            self.position_allocation["status"] = "日线前置条件不通过"
            return
        
        # 检查必要的检测结果
        if not self.weekly_trend_result or not self.daily_buy_result:
            logger.error("缺少必要的检测结果，无法分配分钟级别仓位")
            return
        
        # 获取最强信号和日线买点类型
        strongest_signal = self.daily_buy_result.get("strongest_signal", "无买点")
        
        if strongest_signal == "无买点":
            logger.info("无有效日线买点，不进行仓位分配")
            return
        
        # 检查周线多头趋势
        is_weekly_bullish = self.weekly_trend_result.get("bullish_trend", False)
        
        if not is_weekly_bullish:
            logger.warning("周线非多头趋势，不进行仓位分配")
            return
        
        # 准备短周期层数据
        short_cycle_data = {
            "minute_data": self.minute_data,
            "daily_signal_type": strongest_signal,
            "daily_position_ratio": self.daily_buy_result.get("position_ratio", 0.0),
            "weekly_trend_result": self.weekly_trend_result
        }
        
        # 调用短周期层交易逻辑
        status_message, short_cycle_result = short_cycle_trade_logic(
            short_cycle_data,
            self.daily_pre_pass,
            self.minute_position_allocator
        )
        
        logger.info(f"短周期交易逻辑执行结果: {status_message}")
        
        # 如果short_cycle_result为None，说明前置条件不通过，不进行后续处理
        if short_cycle_result is None:
            logger.info("短周期层逻辑未触发")
            return
        
        # 继续执行原有的仓位分配逻辑
        # 根据日线买点类型确定分钟级别子仓位分配
        allocation_result = None
        
        if strongest_signal == "日线二买":
            # 二买优先匹配30分钟向上笔
            allocation_result = self.minute_position_allocator.allocate_position(
                self.minute_data,
                "日线二买",
                priority="high"
            )
        elif strongest_signal in ["日线一买", "日线三买"]:
            # 一买和三买优先匹配15分钟向上笔
            allocation_result = self.minute_position_allocator.allocate_position(
                self.minute_data,
                "日线一买" if strongest_signal == "日线一买" else "日线三买",
                priority="medium"
            )
        elif strongest_signal == "破中枢反抽":
            # 反抽使用低优先级分配
            allocation_result = self.minute_position_allocator.allocate_position(
                self.minute_data,
                "破中枢反抽",
                priority="low"
            )
        
        # 更新仓位分配结果
        if allocation_result:
            self.position_allocation.update(allocation_result)
            logger.info(f"分钟级别仓位分配完成: 30分钟={allocation_result.get('30min', 0)*100}%, 15分钟={allocation_result.get('15min', 0)*100}%")
        
    def _check_addition_conditions(self):
        """检查向下加仓条件
        
        根据加仓规则，检查是否满足加仓条件
        """
        logger.info("检查向下加仓条件...")
        
        # 初始化加仓检查结果
        self.addition_check_result = {
            "can_add": False,
            "addition_type": None,
            "addition_position": 0.0,
            "addition_reason": "",
            "addition_conditions": {}
        }
        
        # 检查必要的检测结果
        if not self.weekly_trend_result or not self.daily_buy_result or not self.position_allocation:
            logger.error("缺少必要的检测结果，无法检查加仓条件")
            return
        
        # 检查周线多头趋势是否未破坏
        is_weekly_bullish = self.weekly_trend_result.get("bullish_trend", False)
        
        if not is_weekly_bullish:
            logger.info("周线多头趋势已破坏，不满足加仓条件")
            self.addition_check_result["addition_reason"] = "周线多头趋势已破坏"
            return
        
        # 获取最强信号和日线买点类型
        strongest_signal = self.daily_buy_result.get("strongest_signal", "无买点")
        
        # 检查是否有现有的仓位（模拟）
        has_existing_position = sum(self.position_allocation.values()) > 0 or sum(self.current_positions.values()) > 0
        
        if not has_existing_position:
            logger.info("无现有仓位，不检查加仓条件")
            self.addition_check_result["addition_reason"] = "无现有仓位"
            return
        
        # 调用加仓规则模块检查加仓条件
        addition_check = self.position_addition_rules.check_addition_preconditions(
            self.weekly_data,
            self.daily_data,
            self.minute_data,
            strongest_signal,
            self.position_allocation
        )
        
        if addition_check.get("can_add", False):
            # 计算加仓仓位
            addition_position = self.position_addition_rules.calculate_addition_position(
                addition_check,
                strongest_signal,
                self.position_allocation
            )
            
            # 更新加仓结果
            self.addition_check_result.update({
                "can_add": True,
                "addition_type": addition_check.get("addition_type"),
                "addition_position": addition_position,
                "addition_reason": addition_check.get("reason", "满足加仓条件"),
                "addition_conditions": addition_check
            })
            
            logger.info(f"满足加仓条件，加仓类型: {addition_check.get('addition_type')}，加仓仓位: {addition_position*100}%")
        else:
            self.addition_check_result["addition_reason"] = addition_check.get("reason", "不满足加仓条件")
            logger.info(f"不满足加仓条件: {self.addition_check_result['addition_reason']}")
    
    def _check_stop_profit_conditions(self):
        """检查止盈条件
        
        根据止盈规则，检查是否满足止盈条件
        """
        logger.info("检查止盈条件...")
        
        # 初始化止盈检查结果
        self.stop_profit_result = {
            "should_take_profit": False,
            "stop_profit_amount": 0.0,
            "stop_profit_reason": "",
            "stop_profit_details": {}
        }
        
        # 检查必要的检测结果
        if not self.daily_buy_result or not self.position_allocation:
            logger.error("缺少必要的检测结果，无法检查止盈条件")
            return
        
        # 获取最强信号和日线买点类型
        strongest_signal = self.daily_buy_result.get("strongest_signal", "无买点")
        
        # 检查是否有现有的仓位（模拟）
        has_existing_position = sum(self.position_allocation.values()) > 0 or sum(self.current_positions.values()) > 0
        
        if not has_existing_position:
            logger.info("无现有仓位，不检查止盈条件")
            self.stop_profit_result["stop_profit_reason"] = "无现有仓位"
            return
        
        # 调用止盈规则模块检查止盈条件
        stop_profit_check = self.profit_taking_rules.check_stop_profit_conditions(
            self.daily_data,
            self.minute_data,
            strongest_signal,
            self.position_allocation
        )
        
        if stop_profit_check.get("should_take_profit", False):
            # 计算止盈仓位
            stop_profit_position = self.profit_taking_rules.calculate_stop_profit_position(
                stop_profit_check,
                self.position_allocation
            )
            
            # 生成止盈信号
            stop_profit_signal = self.profit_taking_rules.generate_stop_profit_signal(
                stop_profit_check,
                stop_profit_position
            )
            
            # 更新止盈结果
            self.stop_profit_result.update({
                "should_take_profit": True,
                "stop_profit_amount": stop_profit_position.get("stop_profit_amount", 0),
                "stop_profit_reason": f"日线顶背驰+{stop_profit_check.get('minute_confirmation', '')}顶分型",
                "stop_profit_details": stop_profit_signal
            })
            
            logger.info(f"满足止盈条件，止盈仓位: {stop_profit_position.get('stop_profit_amount', 0)*100}%")
        else:
            self.stop_profit_result["stop_profit_reason"] = "未满足止盈条件"
            logger.info("未满足止盈条件")
    
    def _generate_trading_signals(self):
        """生成交易信号
        
        根据所有检测结果生成完整的交易信号，并过滤无效信号
        """
        logger.info("生成交易信号...")
        
        # 确保必要的检测结果已存在
        if not all([self.weekly_trend_result, self.daily_buy_result, self.validation_result, self.position_allocation]):
            logger.error("缺少必要的检测结果，无法生成交易信号")
            return
        
        # 构建有效性字典
        data_validity = {
            "daily": self.validation_result.get("daily_valid", False),
            "weekly": self.validation_result.get("weekly_valid", False)
        }
        
        # 获取最强信号和信号类型优先级
        strongest_signal = self.daily_buy_result.get("strongest_signal", "无买点")
        signal_type_priority = self.daily_buy_result.get("signal_type_priority", "无")
        
        # 检查周线多头
        is_weekly_bullish = self.weekly_trend_result.get("bullish_trend", False)
        
        # 1. 生成建仓信号
        if strongest_signal != "无买点" and is_weekly_bullish:
            # 根据分钟级别分配结果生成建仓信号
            primary_allocation = self.position_allocation.get("primary_allocation")
            
            if primary_allocation:
                # 生成主建仓信号
                build_signal = self.signal_formatter.format_trading_signal(
                    self.weekly_trend_result,
                    self.daily_buy_result,
                    minute_signal_type=primary_allocation,
                    is_addition=False,
                    data_validity=data_validity,
                    current_total_position=sum(self.position_allocation.values()),
                    position_ratio=self.position_allocation.get(primary_allocation, 0),
                    allocation_details=self.position_allocation
                )
                
                # 过滤无效信号
                filtered_result = self.invalid_signal_filter.filter_invalid_signals(
                    build_signal,
                    is_build=True,
                    weekly_trend=is_weekly_bullish,
                    data_validity=data_validity
                )
                
                if filtered_result.get("valid", False):
                    self.trading_signals.append(filtered_result.get("signal"))
                else:
                    # 添加无效信号说明
                    invalid_reason = ", ".join(filtered_result.get("invalid_reasons", []))
                    invalid_signal = self.signal_formatter.format_invalid_signal(invalid_reason)
                    self.trading_signals.append(invalid_signal)
        
        # 2. 生成加仓信号（如果满足条件）
        if self.addition_check_result.get("can_add", False):
            addition_type = self.addition_check_result.get("addition_type")
            addition_position = self.addition_check_result.get("addition_position")
            
            # 生成加仓信号
            addition_signal = self.signal_formatter.format_trading_signal(
                self.weekly_trend_result,
                self.daily_buy_result,
                minute_signal_type="30分钟" if addition_type == "周线底背驰" else "15分钟",
                is_addition=True,
                data_validity=data_validity,
                current_total_position=sum(self.position_allocation.values()),
                position_ratio=addition_position,
                addition_condition=self.addition_check_result.get("addition_reason"),
                allocation_details=self.position_allocation
            )
            
            # 过滤无效信号
            filtered_result = self.invalid_signal_filter.filter_invalid_signals(
                addition_signal,
                is_build=False,
                is_addition=True,
                weekly_trend=is_weekly_bullish,
                data_validity=data_validity
            )
            
            if filtered_result.get("valid", False):
                self.trading_signals.append(filtered_result.get("signal"))
            else:
                # 添加无效信号说明
                invalid_reason = ", ".join(filtered_result.get("invalid_reasons", []))
                invalid_signal = self.signal_formatter.format_invalid_signal(invalid_reason)
                self.trading_signals.append(invalid_signal)
        
        # 3. 生成止盈信号（如果满足条件）
        if self.stop_profit_result.get("should_take_profit", False):
            # 构建止盈信号
            stop_profit_signal = self.signal_formatter.format_stop_profit_signal(
                self.stop_profit_result.get("stop_profit_amount", 0),
                self.stop_profit_result.get("stop_profit_reason"),
                strongest_signal,
                self.position_allocation
            )
            
            # 过滤无效止盈信号
            invalid_check = self.profit_taking_rules.check_invalid_stop_profit_conditions(
                self.stop_profit_result.get("stop_profit_reason"),
                self.stop_profit_result.get("stop_profit_details", {}).get("trigger_reason", "").split("+")[1].replace("顶分型", "") if "+" in self.stop_profit_result.get("stop_profit_reason", "") else None
            )
            
            if not invalid_check.get("is_invalid", False):
                self.trading_signals.append(stop_profit_signal)
            else:
                invalid_reason = ", ".join(invalid_check.get("invalid_reasons", []))
                invalid_signal = self.signal_formatter.format_invalid_signal(f"止盈无效: {invalid_reason}")
                self.trading_signals.append(invalid_signal)
        
        # 4. 如果没有任何信号，添加无信号说明
        if not self.trading_signals:
            if not is_weekly_bullish:
                no_signal = self.signal_formatter.format_invalid_signal("未满足周线多头")
            else:
                no_signal = self.signal_formatter.format_no_signal(strongest_signal)
            self.trading_signals.append(no_signal)
        
        logger.info(f"成功生成{len(self.trading_signals)}个交易信号")
    
    def _generate_report(self) -> str:
        """生成综合交易报告
        
        Returns:
            格式化的交易报告字符串
        """
        logger.info("生成综合交易报告...")
        
        # 构建数据有效性字典
        data_validity = {
            "daily": self.validation_result.get("daily_valid", False),
            "weekly": self.validation_result.get("weekly_valid", False)
        }
        
        # 更新signal_formatter的report_data属性（用于报告生成时访问额外信息）
        self.signal_formatter.report_data = {
            "position_allocation": self.position_allocation,
            "addition_check": self.addition_check_result,
            "stop_profit": self.stop_profit_result
        }
        
        # 生成综合报告
        report = self.signal_formatter.generate_trading_report(
            self.trading_signals,
            self.weekly_trend_result,
            self.daily_buy_result,
            data_validity
        )
        
        return report
    
    def _generate_mock_daily_data(self) -> pd.DataFrame:
        """生成模拟的日线数据（用于测试）
        
        Returns:
            模拟的日线数据DataFrame
        """
        # 生成60天的模拟数据
        dates = pd.date_range(end=pd.Timestamp.now(), periods=60)
        
        # 创建模拟数据
        data = {
            "date": dates,
            "open": [100 + i * 0.1 for i in range(60)],
            "high": [101 + i * 0.1 + 0.5 for i in range(60)],
            "low": [99 + i * 0.1 - 0.5 for i in range(60)],
            "close": [100 + i * 0.1 for i in range(60)],
            "volume": [1000000 + i * 1000 for i in range(60)]
        }
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 为了模拟更好的数据，添加一些波动
        import numpy as np
        np.random.seed(42)
        df["close"] = df["close"] + np.random.normal(0, 0.5, 60)
        df["volume"] = df["volume"] * np.random.normal(1, 0.2, 60)
        
        return df
    
    def _generate_mock_weekly_data(self) -> pd.DataFrame:
        """生成模拟的周线数据（用于测试）
        
        Returns:
            模拟的周线数据DataFrame
        """
        # 生成52周的模拟数据
        dates = pd.date_range(end=pd.Timestamp.now(), periods=52, freq='W')
        
        # 创建模拟数据（周线逐步抬升）
        data = {
            "date": dates,
            "open": [100 + i * 0.5 for i in range(52)],
            "high": [101 + i * 0.5 + 1 for i in range(52)],
            "low": [99 + i * 0.5 - 1 for i in range(52)],
            "close": [100 + i * 0.5 for i in range(52)],
            "volume": [5000000 + i * 5000 for i in range(52)]
        }
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 为了模拟更好的数据，添加一些波动
        import numpy as np
        np.random.seed(42)
        df["close"] = df["close"] + np.random.normal(0, 1, 52)
        df["volume"] = df["volume"] * np.random.normal(1, 0.3, 52)
        
        # 确保最近10根周线收盘价是逐步抬升的
        recent_days = min(10, len(df))
        for i in range(len(df) - recent_days + 1, len(df)):
            if i > 0:
                df.loc[i, "close"] = df.loc[i-1, "close"] + 0.3 + np.random.random() * 0.4
        
        return df
    
    def _generate_mock_minute_data(self, level: str = "30min") -> pd.DataFrame:
        """生成模拟的分钟级别数据（用于测试）
        
        Args:
            level: 分钟级别，如"30min"、"15min"、"5min"
            
        Returns:
            模拟的分钟级别数据DataFrame
        """
        # 根据级别确定频率和数量
        freq_map = {
            "30min": "30T",
            "15min": "15T",
            "5min": "5T"
        }
        count_map = {
            "30min": 200,  # 大约25个交易日的数据
            "15min": 300,  # 大约25个交易日的数据
            "5min": 500    # 大约25个交易日的数据
        }
        
        freq = freq_map.get(level, "30T")
        count = count_map.get(level, 200)
        
        # 生成数据
        dates = pd.date_range(end=pd.Timestamp.now(), periods=count, freq=freq)
        
        # 创建模拟数据，形成一定的趋势和形态
        import numpy as np
        np.random.seed(42)
        
        # 生成一个先跌后涨的数据，形成可能的买点
        base_prices = np.linspace(110, 100, count//3)  # 下跌段
        base_prices = np.concatenate([base_prices, np.linspace(100, 105, count-count//3)])  # 上涨段
        
        # 添加随机波动
        noise = np.random.normal(0, 0.2, count)
        prices = base_prices + noise
        
        data = {
            "date": dates,
            "open": prices,
            "high": prices + 0.3 + np.random.random(count) * 0.2,
            "low": prices - 0.3 - np.random.random(count) * 0.2,
            "close": prices + np.random.normal(0, 0.1, count),
            "volume": np.random.randint(100000, 500000, count)
        }
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 为不同级别的数据添加特定特征
        if level == "30min":
            # 30分钟数据：模拟向上笔完成后的回撤买点
            last_part = min(50, len(df))
            df.iloc[-last_part:, "volume"] = df.iloc[-last_part:, "volume"] * 1.5  # 放量
        elif level == "15min":
            # 15分钟数据：模拟形成底分型
            if len(df) > 20:
                # 创建一个底分型
                bottom_idx = len(df) - 10
                df.loc[bottom_idx, "low"] = df.loc[bottom_idx-2:bottom_idx+2, "low"].min() - 0.2
        
        return df
    
    def _generate_report(self) -> str:
        """生成综合交易报告
        
        Returns:
            格式化的交易报告字符串
        """
        logger.info("生成综合交易报告...")
        
        # 构建数据有效性字典
        data_validity = {
            "daily": self.validation_result.get("daily_valid", False),
            "weekly": self.validation_result.get("weekly_valid", False)
        }
        
        # 更新signal_formatter的report_data属性（用于报告生成时访问额外信息）
        self.signal_formatter.report_data = {
            "position_allocation": self.position_allocation,
            "addition_check": self.addition_check_result,
            "stop_profit": self.stop_profit_result
        }
        
        # 生成综合报告
        report = self.signal_formatter.generate_trading_report(
            self.trading_signals,
            self.weekly_trend_result,
            self.daily_buy_result,
            data_validity
        )
        
        return report


# 主函数用于测试
def main():
    """主函数，用于测试整个交易系统"""
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建主交易系统实例
    system = MainTradingSystem()
    
    # 运行交易系统，以512660军工ETF为例
    result = system.run("512660")
    
    # 打印结果
    print("\n==== 交易系统运行结果 ====")
    print(f"股票代码: {result['stock_code']}")
    print(f"运行时间: {result['timestamp']}")
    print(f"系统状态: {result['system_status']}")
    
    # 打印数据源验证结果
    print("\n数据源验证:")
    if result['validation_result']:
        print(f"  日线数据: {'有效' if result['validation_result']['daily_valid'] else '无效'}")
        print(f"  周线数据: {'有效' if result['validation_result']['weekly_valid'] else '无效'}")
    
    # 打印周线趋势结果
    print("\n周线趋势检测:")
    if result['weekly_trend_result']:
        print(f"  状态: {result['weekly_trend_result'].get('status', '未知')}")
        print(f"  多头趋势: {result['weekly_trend_result'].get('bullish_trend', False)}")
    
    # 打印日线买点结果
    print("\n日线买点检测:")
    if result['daily_buy_result']:
        print(f"  最强信号: {result['daily_buy_result'].get('strongest_signal', '无')}")
        print(f"  信号优先级: {result['daily_buy_result'].get('signal_type_priority', '无')}")
    
    # 打印交易信号
    print("\n交易信号:")
    for i, signal in enumerate(result['trading_signals'], 1):
        print(f"  {i}. {signal}")
    
    # 打印错误信息
    if result['errors']:
        print("\n错误信息:")
        for error in result['errors']:
            print(f"  - {error}")
    
    # 打印综合报告
    print("\n综合报告:")
    print(result['report'])


if __name__ == "__main__":
    main()