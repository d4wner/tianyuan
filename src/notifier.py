#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import requests
import logging
from datetime import datetime
from config import load_config

logger = logging.getLogger('DingdingNotifier')

class DingdingNotifier:
    """钉钉交易信号通知模块"""
    
    def __init__(self, config=None):
        """
        初始化通知器
        :param config: 可选配置参数，如果为None则自动加载配置
        """
        if config is None:
            config = load_config()
        dingding_config = config.get('dingding', {})
        
        self.access_token = dingding_config.get('access_token', '')
        self.webhook_url = f"https://oapi.dingtalk.com/robot/send?access_token={self.access_token}"
        self.position_units = dingding_config.get('position_units', 50)
        self.etf_names = config.get('etfs', {}).get('broad', {})
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }
        logger.info("钉钉通知器初始化完成")
    
    def send_signal(self, symbol, signal_details):
        """
        发送交易信号通知
        :param symbol: 股票代码
        :param signal_details: 信号详情字典
        """
        try:
            # 生成通知消息 - 使用text格式并包含关键词
            message = self._format_signal_message(symbol, signal_details)
            
            # 构建请求数据 - 使用text类型并包含关键词
            post_data = {
                "msgtype": "text",
                "text": {
                    "content": f"QT: {message}"
                }
            }
            
            # 发送请求
            response = requests.post(
                self.webhook_url,
                data=json.dumps(post_data),
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"交易信号发送成功: {symbol}")
                    return True
                else:
                    logger.error(f"交易信号发送失败: {result.get('errmsg')}")
                    return False
            else:
                logger.error(f"交易信号发送HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"发送交易信号异常: {str(e)}")
            return False
    
    def send_error(self, error_message):
        """
        发送错误通知
        :param error_message: 极速消息
        """
        try:
            # 生成错误通知消息
            message = self._format_error_message(error_message)
            
            # 构建请求数据 - 使用text类型并包含关键词
            post_data = {
                "msgtype": "text",
                "text": {
                    "content": f"QT: {message}"
                }
            }
            
            # 发送请求
            response = requests.post(
                self.webhook_url,
                data=json.dumps(post_data),
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info("错误通知发送成功")
                    return True
                else:
                    logger.error(f"错误通知发送失败: {result.get('errmsg')}")
                    return False
            else:
                logger.error(f"错误通知发送HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"发送错误通知异常: {str(e)}")
            return False
    
    def send_alert(self, symbol, alert_details):
        """
        发送风险警报通知
        :param symbol: 股票代码
        :param alert_details: 警报详情字典
        """
        try:
            # 生成风险警报消息
            message = self._format_alert_message(symbol, alert_details)
            
            # 构建请求数据 - 使用text类型并包含关键词
            post_data = {
                "msgtype": "text",
                "text": {
                    "content": f"QT: {message}"
                }
            }
            
            # 发送请求
            response = requests.post(
                self.webhook_url,
                data=json.dumps(post_data),
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"风险警报发送成功: {symbol}")
                    return True
                else:
                    logger.error(f"风险警报发送失败: {result.get('errmsg')}")
                    return False
            else:
                logger.error(f"风险警报发送HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"发送风险警报异常: {str(e)}")
            return False
    
    def _format_signal_message(self, symbol, signal_details):
        """
        格式化交易信号为钉钉消息
        :param symbol: 股票代码
        :param signal_details: 信号详情字典
        :return: 格式化后的消息文本
        """
        # 获取ETF名称
        etf_name = self.etf_names.get(symbol, symbol)
        
        # 计算操作数量
        quantity = int(self.position_units * signal_details.get('position_size', 0.5))
        
        # 构建消息 - 简化格式
        return (
            f"交易信号: {symbol} ({etf_name})\n"
            f"时间: {signal_details.get('time', datetime.now().strftime('%H:%M:%S'))}\n"
            f"操作: {signal_details.get('signal_type', '未知').upper()} {quantity}股\n"
            f"价格: {signal_details.get('price', '未知')}\n"
            f"目标: {signal_details.get('target_price', '未知')}\n"
            f"止损: {signal_details.get('stoploss', '未知')}\n"
            f"原因: {signal_details.get('strategy', '缠论分析')}\n"
            f"置信度: {signal_details.get('confidence', 0)*100:.1f}%\n"
            f"有效期: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    
    def _format_error_message(self, error_message):
        """
        格式化错误通知为钉钉消息
        :param error_message: 错误消息
        :return: 格式化后的消息文本
        """
        # 构建消息 - 简化格式
        return (
            f"系统错误通知\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"错误信息: {error_message}\n"
            f"建议操作: 立即检查系统日志"
        )
    
    def _format_alert_message(self, symbol, alert_details):
        """
        格式化风险警报为钉钉消息
        :param symbol: 股票代码
        :param alert_details: 警报详情字典
        :return: 格式化后的消息文本
        """
        # 获取ETF名称
        etf_name = self.etf_names.get(symbol, symbol)
        
        # 构建消息 - 简化格式
        return (
            f"风险警报通知\n"
            f"股票代码: {symbol} ({etf_name})\n"
            f"警报类型: {alert_details.get('alert_type', '未知')}\n"
            f"当前价格: {alert_details.get('price', '未知')}\n"
            f"触发时间: {alert_details.get('time', datetime.now().strftime('%H:%M:%S'))}\n"
            f"警报信息: {alert_details.get('message', '未知')}\n"
            f"操作建议: {alert_details.get('suggestion', '请检查')}"
        )
    
    def test_connection(self, message="钉钉接口测试消息"):
        """
        测试钉钉接口连通性
        :param message: 测试消息内容
        """
        try:
            # 构建测试消息 - 使用text类型并包含关键词
            test_data = {
                "msgtype": "text",
                "text": {
                    "content": f"QT: 接口测试通知 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
                }
            }
            
            # 发送请求
            response = requests.post(
                self.webhook_url,
                data=json.dumps(test_data),
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info("钉钉接口测试成功")
                    return True
                else:
                    logger.error(f"钉钉接口测试失败: {result.get('errmsg')}")
                    return False
            else:
                logger.error(f"钉钉接口测试HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"钉钉接口测试异常: {str(e)}")
            return False

# 添加命令行测试功能
if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description='钉钉通知测试工具')
    parser.add_argument('--test', action='store_true', help='测试钉钉接口连通性')
    parser.add_argument('--signal', action='store_true', help='测试交易信号通知')
    parser.add_argument('--error', action='store_true', help='测试错误通知')
    parser.add_argument('--alert', action='store_true', help='测试风险警报通知')
    parser.add_argument('--symbol', default='510300', help='股票代码')
    parser.add_argument('--message', default='测试消息', help='自定义测试消息')
    
    args = parser.parse_args()
    
    # 创建通知器
    notifier = DingdingNotifier()
    
    # 执行测试
    if args.test:
        success = notifier.test_connection(args.message)
        print(f"接口测试: {'成功' if success else '失败'}")
    
    elif args.signal:
        # 创建示例交易信号
        signal_details = {
            "signal_type": "buy",
            "price": 4.55,
            "target_price": 4.85,
            "stoploss": 4.40,
            "position_size": 0.3,
            "time": datetime.now().strftime("%H:%M:%S"),
            "strategy": "缠论底分型突破",
            "confidence": 0.85
        }
        success = notifier.send_signal(args.symbol, signal_details)
        print(f"交易信号测试: {'成功' if success else '失败'}")
    
    elif args.error:
        success = notifier.send_error(args.message)
        print(f"错误通知测试: {'成功' if success else '失败'}")
    
    elif args.alert:
        # 创建示例风险警报
        alert_details = {
            "alert_type": "止损触发",
            "price": 4.40,
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": "价格跌破关键支撑位",
            "suggestion": "立即平仓"
        }
        success = notifier.send_alert(args.symbol, alert_details)
        print(f"风险警报测试: {'成功' if success else '失败'}")
    
    else:
        # 默认测试所有类型
        success = notifier.test_connection("钉钉通知接口连通性测试")
        print(f"接口测试: {'成功' if success else '失败'}")
        
        if success:
            # 测试交易信号
            signal_details = {
                "signal_type": "buy",
                "price": 4.55,
                "target_price": 4.85,
                "stoploss": 4.40,
                "position_size": 0.3,
                "time": datetime.now().strftime("%H:%M:%S"),
                "strategy": "缠论底分型突破",
                "confidence": 0.85
            }
            success = notifier.send_signal(args.symbol, signal_details)
            print(f"交易信号测试: {'成功' if success else '失败'}")
            
            # 测试错误通知
            success = notifier.send_error("系统测试错误: 钉钉通知功能验证")
            print(f"错误通知测试: {'成功' if success else '失败'}")
            
            # 测试风险警报
            alert_details = {
                "alert_type": "止损触发",
                "price": 4.40,
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": "价格跌破关键支撑位",
                "suggestion": "立即平仓"
            }
            success = notifier.send_alert(args.symbol, alert_details)
            print(f"风险警报测试: {'成功' if success else '失败'}")