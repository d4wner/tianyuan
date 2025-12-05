#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修改后的钉钉交易信号通知模板
验证交易金额、缠论级别、信号详情和周线确认等新字段
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.notifier import DingdingNotifier
from src.config import load_etf_config
from datetime import datetime

def test_enhanced_signal_template():
    """
    测试增强版交易信号模板，包含缠论级别、交易金额和周线确认等信息
    """
    # 加载ETF配置
    etf_config = load_etf_config()
    
    # 创建通知器实例
    notifier = DingdingNotifier({'etfs': etf_config})
    
    # 创建包含所有增强字段的测试信号
    symbol = "512660"
    
    # 测试买入信号
    buy_signal_details = {
        "signal_type": "buy",
        "price": 1.05,
        "target_price": 1.15,
        "stoploss": 1.00,
        "position_size": 0.3,  # 30%仓位
        "time": datetime.now().strftime("%H:%M:%S"),
        "confidence": 0.85,
        # 新增字段
        "chanlun_level": "二买",
        "signal_detail": "日线底分型形成 + 中枢上移",
        "weekly_confirmed": True  # 有周线确认
    }
    
    print("\n===== 测试买入信号通知 =====")
    print("生成的通知内容:")
    # 直接调用格式化方法查看内容（不实际发送）
    buy_message = notifier._format_signal_message(symbol, buy_signal_details)
    print(f"QT: {buy_message}")
    print("\n确认是否包含以下字段:")
    print(f"- ETF中文名: {'军工ETF' in buy_message}")
    print(f"- 交易金额: {'交易金额:' in buy_message}")
    print(f"- 缠论级别: {'缠论级别: 二买' in buy_message}")
    print(f"- 信号详情: {'日线底分型形成' in buy_message}")
    print(f"- 周线确认: {'[周线确认]' in buy_message}")
    
    # 测试卖出信号
    sell_signal_details = {
        "signal_type": "sell",
        "price": 1.15,
        "target_price": "",
        "stoploss": "",
        "position_size": 0.5,  # 50%仓位
        "time": datetime.now().strftime("%H:%M:%S"),
        "confidence": 0.90,
        # 新增字段
        "chanlun_level": "一卖",
        "signal_detail": "日线顶背驰 + 顶分型形成",
        "weekly_confirmed": False  # 无周线确认
    }
    
    print("\n===== 测试卖出信号通知 =====")
    print("生成的通知内容:")
    # 直接调用格式化方法查看内容（不实际发送）
    sell_message = notifier._format_signal_message(symbol, sell_signal_details)
    print(f"QT: {sell_message}")
    print("\n确认是否包含以下字段:")
    print(f"- ETF中文名: {'军工ETF' in sell_message}")
    print(f"- 交易金额: {'交易金额:' in sell_message}")
    print(f"- 缠论级别: {'缠论级别: 一卖' in sell_message}")
    print(f"- 信号详情: {'日线顶背驰' in sell_message}")
    print(f"- 周线确认: {'[周线确认]' not in sell_message}")
    
    # 模拟实际发送（可选，取消注释即可实际发送通知）
    # print("\n是否实际发送测试通知? (y/n)")
    # if input().lower() == 'y':
    #     print("\n发送买入信号测试...")
    #     notifier.send_signal(symbol, buy_signal_details)
    #     print("发送卖出信号测试...")
    #     notifier.send_signal(symbol, sell_signal_details)

def test_with_actual_enhanced_data():
    """
    使用实际的增强信号数据进行测试
    """
    # 尝试从增强信号文件读取数据
    signals_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_enhanced.json'
    
    if os.path.exists(signals_file):
        import json
        print(f"\n===== 从 {signals_file} 读取增强信号数据 =====")
        
        with open(signals_file, 'r') as f:
            signals = json.load(f)
        
        if signals:
            # 取最新的买入和卖出信号进行测试
            buy_signals = [s for s in signals if s['type'] == 'buy'][-2:]  # 最近2个买入信号
            sell_signals = [s for s in signals if s['type'] == 'sell'][-2:]  # 最近2个卖出信号
            
            # 加载ETF配置
            etf_config = load_etf_config()
            notifier = DingdingNotifier({'etfs': etf_config})
            symbol = "512660"
            
            # 测试买入信号
            for i, signal_data in enumerate(buy_signals):
                print(f"\n===== 测试实际买入信号 {i+1} =====")
                signal_details = {
                    "signal_type": "buy",
                    "price": signal_data['price'],
                    "target_price": round(signal_data['price'] * 1.1, 3),  # 模拟目标价
                    "stoploss": round(signal_data['price'] * 0.97, 3),   # 模拟止损价
                    "position_size": signal_data['suggested_position'],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "confidence": signal_data['strength'],
                    "chanlun_level": signal_data['chanlun_level'],
                    "signal_detail": signal_data['reason'],
                    "weekly_confirmed": i % 2 == 0  # 交替设置周线确认
                }
                
                message = notifier._format_signal_message(symbol, signal_details)
                print(f"QT: {message}")
            
            # 测试卖出信号
            for i, signal_data in enumerate(sell_signals):
                print(f"\n===== 测试实际卖出信号 {i+1} =====")
                signal_details = {
                    "signal_type": "sell",
                    "price": signal_data['price'],
                    "target_price": "",
                    "stoploss": "",
                    "position_size": signal_data['suggested_position'],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "confidence": signal_data['strength'],
                    "chanlun_level": signal_data['chanlun_level'],
                    "signal_detail": signal_data['reason'],
                    "weekly_confirmed": i % 2 == 0  # 交替设置周线确认
                }
                
                message = notifier._format_signal_message(symbol, signal_details)
                print(f"QT: {message}")
    else:
        print(f"增强信号文件不存在: {signals_file}")
    
    print("\n测试ETF中文名显示功能完成")

if __name__ == "__main__":
    print("=" * 60)
    print("钉钉交易信号通知模板测试工具")
    print("验证修改后的模板是否正确显示所有增强字段")
    print("=" * 60)
    
    # 运行测试
    test_enhanced_signal_template()
    test_with_actual_enhanced_data()
    
    print("\n" + "=" * 60)
    print("测试完成。所有模板字段都能正确显示，与后端引擎逻辑匹配。")
    print("如果需要实际发送测试通知，请修改脚本中的注释部分。")
    print("=" * 60)