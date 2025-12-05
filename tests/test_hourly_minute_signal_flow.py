#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试小时/分钟级别信号检测和钉钉通知流程
用于验证512660在交易时间内的信号检测和通知功能
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入必要的模块
from src.hourly_signal_detector import HourlySignalDetector
from src.minute_position_allocator import MinutePositionAllocator
from src.data_fetcher import StockDataFetcher
from src.utils import is_trading_hour
from src.notifier import DingdingNotifier
from src.config import load_config, load_system_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, 'logs', 'test_hourly_minute_signal.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class HourlyMinuteSignalTester:
    """小时/分钟级别信号测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 加载配置
        self.config = load_system_config() if hasattr(sys.modules['src.config'], 'load_system_config') else load_config()
        self.hourly_detector = HourlySignalDetector(self.config)
        self.minute_allocator = MinutePositionAllocator()
        self.data_fetcher = StockDataFetcher()
        self.notifier = DingdingNotifier(self.config)
        self.symbol = "512660"
        
        # 获取ETF中文名称
        etf_config = self.config.get('etfs', {})
        self.etf_name = etf_config.get(self.symbol, {}).get('name', self.symbol)
    
    def test_hourly_signal_detection(self):
        """测试小时级别信号检测"""
        logger.info(f"开始测试小时级别信号检测: {self.etf_name} ({self.symbol})")
        
        # 模拟预测日线底分型
        result = self.hourly_detector.predict_daily_bottom_fractal(self.symbol)
        
        logger.info(f"小时级别信号预测结果: {result}")
        
        return result
    
    def test_minute_signal_detection(self):
        """测试分钟级别信号检测"""
        logger.info(f"开始测试分钟级别信号检测: {self.etf_name} ({self.symbol})")
        
        # 获取最近的分钟数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        # 测试30分钟和15分钟级别
        minute_levels = ['30min', '15min']
        results = {}
        
        for level in minute_levels:
            logger.info(f"测试{level}级别信号检测")
            
            # 获取对应级别的分钟数据
            if level == '30min':
                df_minute = self.data_fetcher.get_minute_data(self.symbol, start_date.strftime('%Y-%m-%d'), 
                                                           end_date.strftime('%Y-%m-%d'), interval=30)
            elif level == '15min':
                df_minute = self.data_fetcher.get_minute_data(self.symbol, start_date.strftime('%Y-%m-%d'), 
                                                           end_date.strftime('%Y-%m-%d'), interval=15)
            
            if df_minute.empty:
                logger.warning(f"无法获取{level}数据")
                results[level] = None
                continue
            
            # 检测向上笔
            up_pen_result = self.minute_allocator.detect_minute_up_pen(df_minute, level)
            
            # 检测底分型（通过向上笔结果判断）
            has_bottom_fractal = up_pen_result.get('has_bottom_fractal', False)
            has_up_pen = up_pen_result.get('up_pen_completed', False)
            
            results[level] = {
                'has_bottom_fractal': has_bottom_fractal,
                'has_up_pen': has_up_pen,
                'validation_reason': up_pen_result.get('validation_reason', '无'),
                'signal': up_pen_result.get('signal', '无')
            }
            
            logger.info(f"{level}级别检测结果: {results[level]}")
        
        return results
    
    def test_dingding_notification(self, hourly_result=None, minute_results=None):
        """测试钉钉通知功能"""
        logger.info(f"开始测试钉钉通知功能: {self.etf_name} ({self.symbol})")
        
        try:
            # 构造测试通知内容
            alert_details = {
                "alert_type": "测试 - 小时/分钟级别信号",
                "symbol": self.symbol,
                "name": self.etf_name,
                "price": hourly_result.get('current_price', '未知') if hourly_result else '未知',
                "today_low": hourly_result.get('today_low', '未知') if hourly_result else '未知',
                "confidence": hourly_result.get('confidence', 0) if hourly_result else 0,
                "reason": f"小时级别: {hourly_result.get('reason', '无')}\n分钟级别: {str(minute_results)}" if hourly_result and minute_results else "测试通知",
                "time": datetime.now().strftime("%H:%M:%S"),
                "suggestion": "这是一条测试通知，用于验证通知功能是否正常"
            }
            
            # 发送通知
            success = self.notifier.send_hourly_alert(alert_details)
            
            if success:
                logger.info(f"钉钉通知发送成功！")
            else:
                logger.error(f"钉钉通知发送失败！")
            
            return success
            
        except Exception as e:
            logger.error(f"测试钉钉通知异常: {str(e)}")
            return False
    
    def simulate_real_time_detection(self, test_duration=60, interval=10):
        """模拟实时检测流程"""
        logger.info(f"开始模拟实时检测流程，持续时间: {test_duration}秒，检测间隔: {interval}秒")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=test_duration)
        
        while datetime.now() < end_time:
            logger.info(f"\n--- 执行实时检测 (当前时间: {datetime.now().strftime('%H:%M:%S')}) ---")
            
            # 检查是否在交易时间
            is_trading = is_trading_hour()
            logger.info(f"当前是否为交易时间: {is_trading}")
            
            # 执行小时级别检测
            hourly_result = self.test_hourly_signal_detection()
            
            # 执行分钟级别检测
            minute_results = self.test_minute_signal_detection()
            
            # 如果检测到信号，发送通知
            if hourly_result.get('prediction', False) or any(result.get('has_bottom_fractal', False) for result in minute_results.values() if result):
                logger.info(f"检测到信号，发送通知...")
                self.test_dingding_notification(hourly_result, minute_results)
            
            # 等待下一次检测
            time.sleep(interval)
            
        logger.info(f"模拟实时检测流程结束")

def main():
    """主函数"""
    logger.info("===== 小时/分钟级别信号检测和通知流程测试 ====")
    
    tester = HourlyMinuteSignalTester()
    
    try:
        # 测试小时级别信号检测
        hourly_result = tester.test_hourly_signal_detection()
        
        # 测试分钟级别信号检测
        minute_results = tester.test_minute_signal_detection()
        
        # 测试钉钉通知
        notification_success = tester.test_dingding_notification(hourly_result, minute_results)
        
        # 输出综合测试结果
        logger.info("\n===== 综合测试结果 ====")
        logger.info(f"小时级别信号检测: {'成功' if hourly_result else '失败'}")
        logger.info(f"分钟级别信号检测: {'成功' if minute_results else '失败'}")
        logger.info(f"钉钉通知功能: {'成功' if notification_success else '失败'}")
        
        # 询问用户是否要模拟实时检测
        logger.info("\n===== 实时检测模拟 ====")
        logger.info("系统已经具备实时检测小时/分钟级别信号的能力")
        logger.info("在实际应用中，可以通过以下方式运行:")
        logger.info("1. 使用monitor.py中的实时监控功能")
        logger.info("2. 配置定时任务(crontab)定期执行检测")
        logger.info("3. 直接运行hourly_signal_detector.py进行持续监控")
        
    except Exception as e:
        logger.error(f"测试过程中发生异常: {str(e)}")
        logger.exception("异常详情:")

if __name__ == "__main__":
    main()