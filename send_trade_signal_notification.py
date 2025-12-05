#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交易信号钉钉通知脚本 - 支持核心(日线)和参考(分钟)级别信号区分"""

import sys
import os
import json
import yaml
import requests
from datetime import datetime
from typing import Dict, List
from src.chanlun_daily_detector import ChanlunDailyDetector
from analyze_signal_statistics import SignalStatisticsAnalyzer

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SignalNotification')


class TradeSignalNotifier:
    """交易信号通知器"""
    
    def __init__(self, config_dir: str):
        """初始化通知器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = config_dir
        self.dingding_config = self._load_dingding_config()
        self.current_date = datetime.now()
    
    def _load_dingding_config(self) -> Dict:
        """加载钉钉配置
        
        Returns:
            钉钉配置字典
        """
        try:
            config_path = os.path.join(self.config_dir, 'system.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('system', {}).get('dingding', {})
        except Exception as e:
            logger.error(f"加载钉钉配置失败: {str(e)}")
            return {}
    
    def build_webhook_url(self) -> str:
        """构建Webhook URL
        
        Returns:
            Webhook URL
        """
        access_token = self.dingding_config.get('access_token', '')
        return f"https://oapi.dingtalk.com/robot/send?access_token={access_token}"
    
    def format_core_signal_info(self, signal: Dict) -> str:
        """格式化核心信号信息
        
        Args:
            signal: 核心信号字典
            
        Returns:
            格式化的信号信息字符串
        """
        formatted = []
        formatted.append(f"📅 信号触发日期: {signal.get('signal_date', 'N/A')}")
        formatted.append(f"📊 中枢范围: 上沿 {signal.get('central_upper_edge', 'N/A')} | 下沿 {signal.get('central_lower_edge', 'N/A')}")
        
        if signal.get('fractal_data'):
            fractal = signal['fractal_data']
            formatted.append(f"📈 底分型数据: K2高点 {fractal.get('k2_high', 'N/A')} | K5收盘价 {fractal.get('k5_close', 'N/A')}")
        
        formatted.append(f"📊 确认日量能放大比例: {signal.get('volume_ratio', 'N/A')}x")
        formatted.append(f"🔋 信号强度: {signal.get('signal_strength', 0) * 100:.1f}%")
        formatted.append(f"✅ 满足策略条件: {'是' if signal.get('meets_strategy', False) else '否'}")
        
        return "\n  • ".join(formatted)
    
    def build_notification_message(self, security_name: str, security_code: str, 
                                  core_statistics: Dict, minute_statistics: Dict, 
                                  current_price: float, current_meets_condition: bool) -> str:
        """构建通知消息
        
        Args:
            security_name: 证券名称
            security_code: 证券代码
            core_statistics: 核心统计数据
            minute_statistics: 分钟级别统计数据
            current_price: 当前价格
            current_meets_condition: 当前是否满足策略条件
            
        Returns:
            完整的通知消息
        """
        message_parts = []
        
        # 标题部分
        message_parts.append(f"【{security_name} ({security_code}) 交易信号汇总报告 - 缠论验证版】")
        message_parts.append("")
        
        # 基本信息
        message_parts.append(f"📅 分析时间: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        message_parts.append(f"📊 最新行情: {current_price:.3f}元")
        message_parts.append(f"🎯 当前是否满足策略条件: {'✅ 是' if current_meets_condition else '❌ 否'}")
        message_parts.append("")
        
        # 核心信号统计（日线级别）
        message_parts.append("🔥 核心信号统计（仅日线级别'创新低破中枢回抽'买点）:")
        message_parts.append(f"  • 统计周期: 过去3个月")
        message_parts.append(f"  • 核心信号数量: {core_statistics.get('signal_count', 0)}个")
        message_parts.append(f"  • 核心信号平均强度: {core_statistics.get('average_strength', 0) * 100:.1f}%")
        message_parts.append("")
        
        # 核心信号详情
        if core_statistics.get('signals'):
            message_parts.append("📋 核心信号详情:")
            for signal in core_statistics['signals']:
                message_parts.append(f"  • {self.format_core_signal_info(signal)}")
                message_parts.append("")
        
        # 参考信号统计（分钟级别）
        message_parts.append("📊 参考信号统计（分钟级别 - 仅短线参考）:")
        
        if minute_statistics.get('timeframe_counts'):
            for timeframe, count in minute_statistics['timeframe_counts'].items():
                message_parts.append(f"  • {timeframe}: {count}个 (非核心策略信号，短线参考)")
        else:
            message_parts.append("  • 当前暂无分钟级别参考信号")
        
        message_parts.append("")
        
        # 交易建议
        message_parts.append("🎯 交易建议:")
        if current_meets_condition:
            message_parts.append("  • ⚡ 建议建仓: 当前满足'日线级别创新低破中枢回抽'买点条件")
            message_parts.append(f"  • 入场价格: {current_price:.3f}元")
            message_parts.append(f"  • 建议仓位: 可考虑{core_statistics.get('average_strength', 0) * 100:.0f}%仓位")
        else:
            message_parts.append("  • 当前不满足核心策略买点条件，建议观望")
            message_parts.append("  • 可参考分钟级别信号进行短线操作")
        
        message_parts.append("")
        
        # 信号强度计算说明
        message_parts.append("🔍 信号强度计算方式:")
        message_parts.append("  • 背驰力度: 30%")
        message_parts.append("  • 量能: 40%")
        message_parts.append("  • 分型有效性: 30%")
        message_parts.append("")
        
        # 风险提示
        message_parts.append("⚠️ 风险提示:")
        message_parts.append("  • 严格执行止损策略，控制风险敞口")
        message_parts.append("  • 市场波动较大，建议分批建仓")
        message_parts.append("  • 分钟级别信号仅作短线参考，不纳入核心策略统计")
        message_parts.append("  • 仅供参考，风险自负")
        
        return "\n".join(message_parts)
    
    def send_dingding_message(self, message: str) -> bool:
        """发送钉钉消息
        
        Args:
            message: 消息内容
            
        Returns:
            是否发送成功
        """
        webhook_url = self.build_webhook_url()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        
        post_data = {
            "msgtype": "text",
            "text": {
                "content": f"QT: {message}"
            }
        }
        
        try:
            response = requests.post(
                webhook_url,
                data=json.dumps(post_data),
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info("钉钉消息发送成功")
                    return True
                else:
                    logger.error(f"钉钉消息发送失败: {result.get('errmsg')}")
            else:
                logger.error(f"HTTP错误: {response.status_code}")
        except Exception as e:
            logger.error(f"发送异常: {str(e)}")
        
        return False
    
    def send_notification(self, security_name: str, security_code: str, 
                         core_statistics: Dict, minute_statistics: Dict, 
                         current_price: float, current_meets_condition: bool) -> bool:
        """发送通知
        
        Args:
            security_name: 证券名称
            security_code: 证券代码
            core_statistics: 核心统计数据
            minute_statistics: 分钟级别统计数据
            current_price: 当前价格
            current_meets_condition: 当前是否满足策略条件
            
        Returns:
            是否发送成功
        """
        # 构建消息
        message = self.build_notification_message(
            security_name, security_code,
            core_statistics, minute_statistics,
            current_price, current_meets_condition
        )
        
        # 发送消息
        return self.send_dingding_message(message)


def generate_demo_data() -> Dict:
    """生成演示数据
    
    Returns:
        包含所有统计数据的字典
    """
    # 模拟核心统计数据
    core_statistics = {
        'signal_count': 3,
        'average_strength': 0.65,
        'signals': [
            {
                'signal_date': '2025-11-15',
                'central_upper_edge': 0.610,
                'central_lower_edge': 0.580,
                'fractal_data': {
                    'k2_high': 0.585,
                    'k5_close': 0.590
                },
                'volume_ratio': 1.45,
                'signal_strength': 0.72,
                'meets_strategy': True
            },
            {
                'signal_date': '2025-10-28',
                'central_upper_edge': 0.605,
                'central_lower_edge': 0.575,
                'fractal_data': {
                    'k2_high': 0.582,
                    'k5_close': 0.587
                },
                'volume_ratio': 1.32,
                'signal_strength': 0.68,
                'meets_strategy': True
            },
            {
                'signal_date': '2025-09-18',
                'central_upper_edge': 0.615,
                'central_lower_edge': 0.585,
                'fractal_data': {
                    'k2_high': 0.590,
                    'k5_close': 0.595
                },
                'volume_ratio': 1.38,
                'signal_strength': 0.55,
                'meets_strategy': True
            }
        ]
    }
    
    # 模拟分钟级别统计数据
    minute_statistics = {
        'timeframe_counts': {
            '15分钟': 2,
            '30分钟': 3,
            '60分钟': 1
        }
    }
    
    return {
        'core_statistics': core_statistics,
        'minute_statistics': minute_statistics
    }

def main():
    """主函数"""
    # 配置路径
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
    
    # 创建通知器实例
    notifier = TradeSignalNotifier(config_dir)
    
    # 生成演示数据（实际使用时应从统计分析器获取）
    demo_data = generate_demo_data()
    
    # 发送通知
    success = notifier.send_notification(
        security_name="军工ETF",
        security_code="512660",
        core_statistics=demo_data['core_statistics'],
        minute_statistics=demo_data['minute_statistics'],
        current_price=0.592,
        current_meets_condition=True
    )
    
    if success:
        print("交易信号通知已成功发送到钉钉")
    else:
        print("交易信号通知发送失败")


if __name__ == "__main__":
    main()