#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试缠论监控模式脚本
用于验证ChanlunMonitor类的正确初始化和基本功能
"""

import os
import sys
import logging
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from src.config import load_config
from src.data_fetcher import StockDataFetcher as StockDataAPI
from src.calculator import ChanlunCalculator
from src.notifier import DingdingNotifier
from src.monitor import ChanlunMonitor
from src.utils import is_trading_hour

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TestMonitor')

def test_monitor_initialization():
    """
    测试监控器的初始化功能
    """
    logger.info("===== 开始测试监控器初始化 =====")
    
    try:
        # 加载系统配置
        config = load_config('config/system.yaml')
        logger.info("配置文件加载成功")
        
        # 初始化组件
        api = StockDataAPI(config.get('data_fetcher', {}))
        calculator = ChanlunCalculator(config.get('chanlun', {}))
        notifier = DingdingNotifier(config.get('dingding', {}))
        
        # 正确初始化监控器（注意这里使用正确的参数顺序）
        monitor = ChanlunMonitor(
            system_config=config,
            api=api,
            calculator=calculator,
            notifier=notifier
        )
        
        # 设置初始资金（可选，如果需要自定义资金）
        # monitor.total_capital = 600000
        # monitor.available_cash = 600000
        
        logger.info(f"监控器初始化成功: 初始资金={monitor.total_capital}, 可用资金={monitor.available_cash}")
        logger.info(f"监控间隔: {monitor.interval}秒, 分钟周期: {monitor.minute_period}")
        
        return monitor
        
    except Exception as e:
        logger.error(f"监控器初始化失败: {str(e)}", exc_info=True)
        return None

def test_add_symbols(monitor):
    """
    测试添加股票功能
    """
    if not monitor:
        return False
    
    try:
        # 添加几个测试股票
        test_symbols = ['sh600000', 'sz000001', 'sh600036']
        for symbol in test_symbols:
            monitor.add_symbol(symbol)
            logger.info(f"已添加股票: {symbol}")
        
        logger.info(f"总共添加了 {len(monitor.symbols)} 只股票")
        return True
        
    except Exception as e:
        logger.error(f"添加股票失败: {str(e)}", exc_info=True)
        return False

def test_signal_detection(monitor, test_symbol='sh600000'):
    """
    测试信号检测功能（单次检查）
    """
    if not monitor or test_symbol not in monitor.symbols:
        return False
    
    try:
        logger.info(f"开始测试股票 {test_symbol} 的信号检测...")
        
        # 调用check_symbol方法进行单次检测
        monitor.check_symbol(test_symbol)
        
        # 检查是否生成了信号
        if test_symbol in monitor.last_signals:
            signal = monitor.last_signals[test_symbol]
            logger.info(f"检测到信号: {test_symbol} - 操作: {signal.get('action')}, 原因: {signal.get('reason')}")
            return True
        else:
            logger.warning(f"未检测到信号: {test_symbol}")
            return False
            
    except Exception as e:
        logger.error(f"信号检测失败: {str(e)}", exc_info=True)
        return False

def run_demo_mode(monitor, duration=60):
    """
    运行演示模式（短时间运行监控）
    :param duration: 运行时长（秒）
    """
    if not monitor or not monitor.symbols:
        logger.error("监控器未初始化或未添加股票")
        return
    
    logger.info(f"===== 开始演示模式 (运行{duration}秒) =====")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{current_time}] 执行股票检查...")
            
            # 检查是否在交易时间
            if not is_trading_hour():
                logger.info("当前非交易时间")
            
            # 检查所有股票
            for symbol in monitor.symbols:
                monitor.check_symbol(symbol)
            
            # 等待下一次检查（使用监控间隔的1/5，以便演示更快）
            wait_time = min(monitor.interval / 5, 10)
            logger.info(f"等待 {wait_time:.1f} 秒后再次检查")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        logger.info("演示模式被用户中断")
    except Exception as e:
        logger.error(f"演示模式出错: {str(e)}", exc_info=True)
    finally:
        logger.info("===== 演示模式结束 =====")
        
        # 保存最终持仓
        monitor.save_positions()
        logger.info(f"持仓信息已保存，当前可用资金: {monitor.available_cash}")
        
        # 打印当前持仓状态
        if monitor.position:
            logger.info("当前持仓:")
            for symbol, pos_info in monitor.position.items():
                logger.info(f"  {symbol}: {pos_info['shares']}股, 平均成本: {pos_info['avg_price']:.2f}")
        else:
            logger.info("当前无持仓")

def main():
    """
    主测试函数
    """
    # 1. 初始化监控器
    monitor = test_monitor_initialization()
    if not monitor:
        logger.error("监控器初始化失败，测试终止")
        return
    
    # 2. 添加测试股票
    if not test_add_symbols(monitor):
        logger.error("添加股票失败，测试终止")
        return
    
    # 3. 测试单次信号检测
    test_signal_detection(monitor)
    
    # 4. 运行短时间演示模式
    run_demo_mode(monitor, duration=30)  # 运行30秒演示
    
    logger.info("===== 所有测试完成 =====")
    logger.info("\n注意: 此脚本演示了ChanlunMonitor的正确使用方法")
    logger.info("在实际使用中，您可以直接使用monitor.py中的命令行模式:")
    logger.info("python src/monitor.py -s sh600000 -c 600000 -m realtime")
    logger.info("或者修复main.py中的run_monitor_mode函数后使用:")
    logger.info("python src/main.py --mode monitor --symbols sh600000 sh600036 -i 300")

if __name__ == "__main__":
    main()