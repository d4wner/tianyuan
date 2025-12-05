#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘容错模块 - TradingExecutor

实现交易执行的容错机制，包括：
1. 行情延迟校验（≤500ms）
2. 订单梯度重试（3次）
3. 单日亏损5%熔断

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import time
import threading
from typing import Dict, Optional, Any, Tuple
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingExecutor:
    """实盘交易执行器，提供容错机制"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化交易执行器
        
        Args:
            config: 配置参数
        """
        logger.info("初始化实盘交易执行器...")
        
        # 默认配置
        default_config = {
            "max_market_delay": 500,  # 最大行情延迟（毫秒）
            "max_order_retries": 3,    # 最大订单重试次数
            "daily_loss_limit": 0.05,  # 单日亏损上限（5%）
            "retry_delay": [1000, 3000, 5000]  # 重试延迟（毫秒）
        }
        
        self.config = config or default_config
        self.max_market_delay = self.config["max_market_delay"]
        self.max_order_retries = self.config["max_order_retries"]
        self.daily_loss_limit = self.config["daily_loss_limit"]
        self.retry_delay = self.config["retry_delay"]
        
        # 交易状态
        self.trading_status = {
            "is_trading": True,
            "daily_pnl": 0.0,        # 单日盈亏
            "daily_start_pnl": 0.0,   # 每日开始时的盈亏
            "last_market_update": time.time(),  # 上次行情更新时间
            "order_failures": 0,      # 今日订单失败次数
            "market_delay": 0,        # 当前行情延迟（毫秒）
            "circuit_breaker_triggered": False,  # 是否已触发熔断
            "circuit_breaker_reason": ""
        }
        
        # 线程锁，用于保护共享状态
        self.lock = threading.Lock()
        
        logger.info(f"实盘交易执行器初始化完成：")
        logger.info(f"  - 最大行情延迟: {self.max_market_delay}ms")
        logger.info(f"  - 最大订单重试次数: {self.max_order_retries}次")
        logger.info(f"  - 单日亏损熔断: {self.daily_loss_limit*100}%")
        logger.info(f"  - 重试延迟序列: {self.retry_delay}ms")
    
    def check_market_delay(self) -> Tuple[bool, float]:
        """检查行情延迟
        
        Returns:
            (是否满足延迟要求, 当前延迟毫秒数)
        """
        with self.lock:
            current_time = time.time()
            delay_ms = (current_time - self.trading_status["last_market_update"]) * 1000
            self.trading_status["market_delay"] = delay_ms
        
        logger.info(f"当前行情延迟: {delay_ms:.2f}ms")
        
        if delay_ms > self.max_market_delay:
            logger.warning(f"行情延迟过高: {delay_ms:.2f}ms > {self.max_market_delay}ms")
            return False, delay_ms
        
        return True, delay_ms
    
    def update_market_data(self, data: Dict[str, Any]) -> None:
        """更新行情数据，同时更新行情时间戳
        
        Args:
            data: 行情数据
        """
        with self.lock:
            self.trading_status["last_market_update"] = time.time()
            logger.debug(f"行情数据已更新，当前时间戳: {self.trading_status['last_market_update']}")
    
    def update_daily_pnl(self, current_pnl: float) -> None:
        """更新每日盈亏
        
        Args:
            current_pnl: 当前总盈亏
        """
        with self.lock:
            # 如果是新的一天，重置每日开始盈亏
            if self._is_new_day():
                self.trading_status["daily_start_pnl"] = current_pnl
                logger.info(f"新的交易日开始，重置每日开始盈亏: {current_pnl}")
            
            # 计算单日盈亏
            self.trading_status["daily_pnl"] = current_pnl - self.trading_status["daily_start_pnl"]
            logger.info(f"单日盈亏已更新: {self.trading_status['daily_pnl']:.4f}")
            
            # 检查是否触发熔断
            self._check_circuit_breaker()
    
    def _check_circuit_breaker(self) -> None:
        """检查是否触发熔断"""
        with self.lock:
            if self.trading_status["daily_pnl"] <= -self.daily_loss_limit:
                if not self.trading_status["circuit_breaker_triggered"]:
                    self.trading_status["circuit_breaker_triggered"] = True
                    self.trading_status["is_trading"] = False
                    self.trading_status["circuit_breaker_reason"] = f"单日亏损达到{self.daily_loss_limit*100}%"
                    logger.warning(f"触发熔断机制: {self.trading_status['circuit_breaker_reason']}")
                    logger.warning(f"今日剩余时间将停止所有交易")
    
    def _is_new_day(self) -> bool:
        """判断是否是新的交易日
        
        Returns:
            是否是新的交易日
        """
        current_time = time.localtime()
        last_update_time = time.localtime(self.trading_status["last_market_update"])
        
        # 如果日期不同，认为是新的交易日
        if current_time.tm_year != last_update_time.tm_year or \
           current_time.tm_mon != last_update_time.tm_mon or \
           current_time.tm_mday != last_update_time.tm_mday:
            return True
        
        return False
    
    def reset_daily_status(self) -> None:
        """重置每日状态（用于手动重置熔断）"""
        with self.lock:
            self.trading_status["daily_pnl"] = 0.0
            self.trading_status["daily_start_pnl"] = 0.0
            self.trading_status["order_failures"] = 0
            self.trading_status["circuit_breaker_triggered"] = False
            self.trading_status["is_trading"] = True
            self.trading_status["circuit_breaker_reason"] = ""
            logger.info("每日状态已重置，熔断机制已解除")
    
    def execute_order(self, order_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """执行订单，带梯度重试机制
        
        Args:
            order_info: 订单信息
            example: {"symbol": "510300", "type": "buy", "price": 3.21, "volume": 1000}
            
        Returns:
            (是否执行成功, 执行结果)
        """
        # 检查交易状态
        if not self._check_trading_status():
            return False, {"error": "交易系统已暂停", "reason": self.trading_status["circuit_breaker_reason"]}
        
        # 检查行情延迟
        delay_ok, delay_ms = self.check_market_delay()
        if not delay_ok:
            return False, {"error": "行情延迟过高", "delay_ms": delay_ms}
        
        # 执行订单，带重试机制
        for retry in range(self.max_order_retries):
            try:
                logger.info(f"执行订单（第{retry+1}次尝试）: {order_info}")
                
                # 模拟订单执行（实际交易中替换为真实API调用）
                order_result = self._simulate_order_execution(order_info)
                
                # 检查订单是否成功
                if order_result.get("success", False):
                    logger.info(f"订单执行成功: {order_result}")
                    with self.lock:
                        self.trading_status["order_failures"] = 0  # 重置失败次数
                    return True, order_result
                else:
                    logger.warning(f"订单执行失败: {order_result.get('error', '未知错误')}")
                    
            except Exception as e:
                logger.error(f"订单执行异常: {str(e)}")
            
            # 如果不是最后一次尝试，等待重试延迟
            if retry < self.max_order_retries - 1:
                delay = self.retry_delay[min(retry, len(self.retry_delay)-1)]
                logger.info(f"等待{delay}ms后重试...")
                time.sleep(delay / 1000)
        
        # 所有重试都失败
        logger.error(f"所有{self.max_order_retries}次订单执行尝试都失败")
        with self.lock:
            self.trading_status["order_failures"] += 1
        
        return False, {"error": "订单执行失败", "retries": self.max_order_retries}
    
    def _check_trading_status(self) -> bool:
        """检查交易状态是否正常
        
        Returns:
            是否可以进行交易
        """
        with self.lock:
            if not self.trading_status["is_trading"]:
                logger.warning(f"交易已暂停: {self.trading_status['circuit_breaker_reason']}")
                return False
            
            if self.trading_status["circuit_breaker_triggered"]:
                logger.warning(f"熔断机制已触发: {self.trading_status['circuit_breaker_reason']}")
                return False
            
            return True
    
    def _simulate_order_execution(self, order_info: Dict[str, Any]) -> Dict[str, Any]:
        """模拟订单执行（实际交易中替换为真实API调用）
        
        Args:
            order_info: 订单信息
            
        Returns:
            订单执行结果
        """
        # 模拟订单执行逻辑
        # 在实际应用中，这里应该调用券商的交易API
        import random
        
        # 模拟80%的成功率
        if random.random() > 0.2:
            return {
                "success": True,
                "order_id": f"order_{int(time.time())}_{random.randint(1000, 9999)}",
                "symbol": order_info["symbol"],
                "type": order_info["type"],
                "price": order_info["price"],
                "volume": order_info["volume"],
                "executed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": "订单已成交"
            }
        else:
            return {
                "success": False,
                "error": "订单执行失败（模拟）",
                "symbol": order_info["symbol"],
                "type": order_info["type"],
                "price": order_info["price"],
                "volume": order_info["volume"],
                "message": "模拟订单失败"
            }
    
    def get_trading_status(self) -> Dict[str, Any]:
        """获取当前交易状态
        
        Returns:
            交易状态字典
        """
        with self.lock:
            # 返回状态的副本，避免外部修改
            return self.trading_status.copy()
    
    def reset_trading_status(self) -> None:
        """重置交易状态（包括熔断）
        
        注意：仅用于测试或手动干预
        """
        with self.lock:
            self.trading_status = {
                "is_trading": True,
                "daily_pnl": 0.0,
                "daily_start_pnl": 0.0,
                "last_market_update": time.time(),
                "order_failures": 0,
                "market_delay": 0,
                "circuit_breaker_triggered": False,
                "circuit_breaker_reason": ""
            }
            logger.info("交易状态已重置")


# 测试代码
if __name__ == "__main__":
    # 创建交易执行器
    executor = TradingExecutor()
    
    # 测试行情延迟检查
    print("=== 测试行情延迟检查 ===")
    time.sleep(0.2)
    ok, delay = executor.check_market_delay()
    print(f"行情延迟检查结果: {'通过' if ok else '失败'}, 延迟: {delay:.2f}ms")
    
    # 更新行情数据
    print("\n=== 测试更新行情数据 ===")
    executor.update_market_data({"symbol": "510300", "price": 3.21})
    ok, delay = executor.check_market_delay()
    print(f"行情延迟检查结果: {'通过' if ok else '失败'}, 延迟: {delay:.2f}ms")
    
    # 测试订单执行
    print("\n=== 测试订单执行 ===")
    order = {"symbol": "510300", "type": "buy", "price": 3.21, "volume": 1000}
    success, result = executor.execute_order(order)
    print(f"订单执行结果: {'成功' if success else '失败'}")
    print(f"结果详情: {result}")
    
    # 测试熔断机制
    print("\n=== 测试熔断机制 ===")
    print(f"当前交易状态: {'正常' if executor.get_trading_status()['is_trading'] else '暂停'}")
    executor.update_daily_pnl(-0.06)  # 超过5%的亏损
    print(f"触发熔断后交易状态: {'正常' if executor.get_trading_status()['is_trading'] else '暂停'}")
    print(f"熔断原因: {executor.get_trading_status()['circuit_breaker_reason']}")
    
    # 测试重置熔断
    print("\n=== 测试重置熔断 ===")
    executor.reset_daily_status()
    print(f"重置后交易状态: {'正常' if executor.get_trading_status()['is_trading'] else '暂停'}")