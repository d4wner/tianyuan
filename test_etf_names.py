#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试ETF名称加载功能
直接验证ETF中文名是否正确加载和显示
"""

import sys
sys.path.append('.')

from src.config import load_etf_config
from src.notifier import DingdingNotifier


def test_etf_config_loading():
    """
    测试ETF配置加载
    """
    print("=== 测试ETF配置加载 ===")
    
    # 直接加载ETF配置
    etf_config = load_etf_config()
    print(f"ETF配置类型: {type(etf_config)}")
    print(f"ETF配置键: {list(etf_config.keys())}")
    
    # 检查sector类别中的512660
    if 'sector' in etf_config:
        sector_etfs = etf_config['sector']
        print(f"sector类别ETF数量: {len(sector_etfs)}")
        print(f"sector类别中的ETF代码: {list(sector_etfs.keys())}")
        
        # 尝试直接打印sector类别的完整内容
        print(f"sector类别完整内容: {sector_etfs}")
        
        # 手动检查是否有512660
        found = False
        for code in sector_etfs.keys():
            if str(code) == '512660':
                print(f"找到512660，名称: {sector_etfs[code].get('name', '未找到')}")
                found = True
                break
        if not found:
            print("未找到512660")
    else:
        print("未找到sector类别")


def test_dingding_notifier_etf_names():
    """
    测试DingdingNotifier中的ETF名称加载
    """
    print("\n=== 测试DingdingNotifier中的ETF名称加载 ===")
    
    # 创建钉钉通知器实例
    notifier = DingdingNotifier()
    
    print(f"ETF信息字典大小: {len(notifier.etf_info)}")
    print(f"ETF信息字典键: {list(notifier.etf_info.keys())[:10]}...")  # 只显示前10个
    print(f"512660名称: {notifier.etf_info.get('512660', '未找到')}")
    
    # 直接测试ETF名称获取
    symbol = '512660'
    etf_name = notifier.etf_info.get(symbol, symbol)
    print(f"格式化后的ETF名称: {symbol} ({etf_name})")
    
    # 尝试手动构建一个简单的消息来验证
    simple_message = f"交易信号: {symbol} ({etf_name})"
    print(f"\n简单消息: {simple_message}")
    
    # 临时修复：手动添加512660的映射进行测试
    notifier.etf_info['512660'] = '军工ETF'
    print(f"\n手动添加后，512660名称: {notifier.etf_info.get('512660', '未找到')}")
    etf_name = notifier.etf_info.get(symbol, symbol)
    fixed_message = f"交易信号: {symbol} ({etf_name})"
    print(f"修复后的消息: {fixed_message}")


def test_manual_etf_config():
    """
    手动测试ETF配置
    """
    print("\n=== 手动测试ETF配置 ===")
    
    # 手动创建一个包含512660的配置
    manual_etf_info = {'512660': '军工ETF'}
    
    # 创建通知器实例并手动设置etf_info
    notifier = DingdingNotifier()
    notifier.etf_info = manual_etf_info
    notifier.etf_names = manual_etf_info
    
    # 测试消息格式
    symbol = '512660'
    etf_name = notifier.etf_info.get(symbol, symbol)
    message = f"交易信号: {symbol} ({etf_name})"
    print(f"手动配置的消息: {message}")
    print(f"是否包含'军工ETF': {'军工ETF' in message}")


if __name__ == "__main__":
    test_etf_config_loading()
    test_dingding_notifier_etf_names()
    test_manual_etf_config()