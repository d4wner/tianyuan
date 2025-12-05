#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.notifier import DingdingNotifier
from src.config import load_system_config

def test_dingding_notification():
    """测试钉钉通知功能是否修复成功"""
    print("正在测试钉钉通知功能...")
    
    try:
        # 加载系统配置
        config = load_system_config()
        print(f"配置加载成功: {config.keys()}")
        
        # 创建钉钉通知器实例
        notifier = DingdingNotifier(config)
        print(f"DingdingNotifier创建成功")
        print(f"access_token: {'已配置' if notifier.access_token else '未配置'}")
        print(f"webhook_url: {notifier.webhook_url}")
        
        # 测试发送简单消息
        alert_details = {
            'symbol': '512660',
            'name': '军工ETF',
            'price': 1.189,
            'low': 1.180,
            'confidence': 0.85,
            'reason': '小时级底分型+MACD背离',
            'suggestion': '可考虑分批建仓',
            'time': '2024-05-20 14:30:00'
        }
        
        result = notifier.send_hourly_alert(alert_details)
        print(f"发送通知结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dingding_notification()
    if success:
        print("\n✅ 钉钉通知功能测试成功！")
    else:
        print("\n❌ 钉钉通知功能测试失败！")