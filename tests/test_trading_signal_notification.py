#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易信号获取和钉钉通知功能测试脚本（简化版）
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from src.config import load_config
from src.notifier import DingdingNotifier

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TestSignalNotification')

def test_dingding_notification():
    """
    直接测试钉钉通知功能
    """
    logger.info("开始测试钉钉通知功能")
    
    try:
        # 1. 加载配置
        logger.info("加载配置文件...")
        config = load_config()
        
        # 2. 初始化钉钉通知器
        logger.info("初始化钉钉通知器...")
        notifier = DingdingNotifier(config.get('system', {}))
        
        # 3. 测试钉钉连接
        logger.info("测试钉钉接口连接...")
        connection_test = notifier.test_connection("交易信号通知功能测试")
        if not connection_test:
            logger.error("钉钉接口连接测试失败，请检查配置和网络")
            return False
        logger.info("钉钉接口连接测试成功")
        
        # 4. 发送模拟交易信号
        logger.info("准备发送交易信号通知...")
        test_symbol = '510300'  # 沪深300ETF
        
        # 创建模拟信号详情，不依赖实际数据
        signal_details = {
            "signal_type": "buy",
            "price": 4.25,
            "target_price": 4.46,
            "stoploss": 4.12,
            "position_size": 0.3,
            "time": datetime.now().strftime("%H:%M:%S"),
            "strategy": "缠论底分型突破",
            "confidence": 0.85
        }
        
        # 发送信号通知
        notification_result = notifier.send_signal(test_symbol, signal_details)
        
        if notification_result:
            logger.info(f"交易信号通知发送成功: {test_symbol}")
        else:
            logger.error(f"交易信号通知发送失败: {test_symbol}")
            return False
        
        # 5. 测试错误通知
        logger.info("测试错误通知...")
        error_notification = notifier.send_error("测试错误通知: 交易信号功能验证")
        if error_notification:
            logger.info("错误通知发送成功")
        else:
            logger.error("错误通知发送失败")
        
        # 6. 测试风险警报通知
        logger.info("测试风险警报通知...")
        alert_details = {
            "alert_type": "止损触发",
            "price": 4.12,
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": "价格跌破关键支撑位",
            "suggestion": "关注并准备平仓"
        }
        alert_notification = notifier.send_alert(test_symbol, alert_details)
        if alert_notification:
            logger.info("风险警报通知发送成功")
        else:
            logger.error("风险警报通知发送失败")
        
        logger.info("钉钉通知功能测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中出现异常: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("===== 钉钉通知功能测试 =====")
    success = test_dingding_notification()
    
    if success:
        logger.info("所有测试通过!")
        sys.exit(0)
    else:
        logger.error("测试失败，请检查相关配置和代码")
        sys.exit(1)