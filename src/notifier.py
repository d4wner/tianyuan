#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import requests
import logging
from datetime import datetime
from src.config import load_config

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
        
        # 初始化ETF信息字典
        self.etf_info = {}
        
        # 尝试从配置中加载ETF信息
        if 'etfs' in config:
            etfs_config = config.get('etfs', {})
            for category in etfs_config.values():
                if isinstance(category, dict):
                    for symbol, details in category.items():
                        if isinstance(details, dict) and 'name' in details:
                            # 将symbol转换为字符串，确保类型匹配
                            symbol_str = str(symbol)
                            self.etf_info[symbol_str] = details['name']
        
        # 直接加载etfs.yaml文件中的所有ETF配置，确保正确加载所有类别的ETF
        try:
            from .config import load_etf_config
            etfs_config = load_etf_config()
            
            # 遍历所有ETF类别（bond, broad, sector等）
            for category_name, category_data in etfs_config.items():
                # 确保category_data是字典类型
                if isinstance(category_data, dict):
                    # 遍历该类别下的所有ETF
                    for symbol, details in category_data.items():
                        # 确保details是字典类型且包含name字段
                        if isinstance(details, dict) and 'name' in details:
                            # 将symbol转换为字符串，确保类型匹配
                            symbol_str = str(symbol)
                            self.etf_info[symbol_str] = details['name']
            
            logger.info(f"成功加载{len(self.etf_info)}个ETF配置")
        except Exception as e:
            logger.error(f"加载ETF配置失败: {e}")
        
        # 为了兼容旧代码，设置etf_names为etf_info的别名
        self.etf_names = self.etf_info
        
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
    
    def send_hourly_alert(self, alert_details):
        """
        发送小时级别预警信号
        :param alert_details: 预警详情字典，包含symbol, name, price, confidence, reason等
        """
        try:
            symbol = alert_details.get('symbol')
            etf_name = alert_details.get('name', self.etf_info.get(symbol, symbol))
            price = alert_details.get('price', '未知')
            today_low = alert_details.get('today_low', '未知')
            confidence = alert_details.get('confidence', 0)
            reason = alert_details.get('reason', '')
            time = alert_details.get('time', datetime.now().strftime('%H:%M:%S'))
            suggestion = alert_details.get('suggestion', '')
            
            # 置信度转换为百分比
            confidence_pct = f"{confidence * 100:.1f}%"
            
            # 格式化小时级别预警内容
            message = (
                f"小时级别买入预警\n"
                f"[日内提醒] 不要等到收盘！\n"
                f"标的: {etf_name} ({symbol})\n"
                f"当前价: {price}\n"
                f"今日低: {today_low}\n"
                f"预测置信度: {confidence_pct}\n"
                f"依据: {reason}\n"
                f"建议: {suggestion}\n"
                f"时间: {time}\n"
                f"日期: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"注意: 此为小时级别预警，建议结合日线确认"
            )
            
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
                    logger.info(f"小时级别预警发送成功: {symbol}")
                    return True
                else:
                    logger.error(f"小时级别预警发送失败: {result.get('errmsg')}")
                    return False
            else:
                logger.error(f"小时级别预警发送HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"发送小时级别预警异常: {str(e)}")
            return False
    
    def _format_signal_message(self, symbol, signal_details):
        """
        格式化交易信号为钉钉消息
        :param symbol: 股票代码
        :param signal_details: 信号详情字典
        :return: 格式化后的消息文本
        """
        # 获取ETF名称 - 使用所有类别的ETF配置
        etf_name = self.etf_info.get(symbol, symbol)
        
        # 计算操作数量和金额
        position_size = signal_details.get('position_size', 0.5)
        quantity = int(self.position_units * position_size)
        price = signal_details.get('price', 0)
        # 计算交易金额
        amount = quantity * price
        
        # 获取缠论级别和信号详情
        chanlun_level = signal_details.get('chanlun_level', '未指定')
        signal_detail = signal_details.get('signal_detail', '缠论分析')
        # 如果有周线确认信息，添加到描述中
        weekly_confirmed = signal_details.get('weekly_confirmed', False)
        confirmation_text = "[周线确认] " if weekly_confirmed else ""
        
        # 构建消息 - 包含交易金额和详细缠论信号
        return (
            f"交易信号: {symbol} ({etf_name})\n"
            f"时间: {signal_details.get('time', datetime.now().strftime('%H:%M:%S'))}\n"
            f"操作: {signal_details.get('signal_type', '未知').upper()}\n"
            f"价格: {signal_details.get('price', '未知')}\n"
            f"交易金额: ¥{amount:.2f}\n"
            f"目标: {signal_details.get('target_price', '未知')}\n"
            f"止损: {signal_details.get('stoploss', '未知')}\n"
            f"缠论级别: {chanlun_level}\n"
            f"原因: {confirmation_text}{signal_detail}\n"
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
        # 获取ETF名称 - 使用所有类别的ETF配置
        etf_name = self.etf_info.get(symbol, symbol)
        
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
    parser.add_argument('--hourly', action='store_true', help='测试小时级别预警通知')
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
        
    elif args.hourly:
        # 创建示例小时级别预警
        hourly_details = {
            "alert_type": "日线底分型预警",
            "symbol": args.symbol,
            "name": "医药ETF",
            "price": 1.25,
            "today_low": 1.20,
            "confidence": 0.75,
            "reason": "小时级底分型确认；MACD绿柱减小；成交量放大",
            "time": datetime.now().strftime("%H:%M:%S"),
            "suggestion": "建议密切关注，准备买入"
        }
        success = notifier.send_hourly_alert(hourly_details)
        print(f"小时级别预警测试: {'成功' if success else '失败'}")
    
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
            
            # 测试小时级别预警
            hourly_details = {
                "alert_type": "日线底分型预警",
                "symbol": args.symbol,
                "name": "医药ETF",
                "price": 1.25,
                "today_low": 1.20,
                "confidence": 0.75,
                "reason": "小时级底分型确认；MACD绿柱减小；成交量放大",
                "time": datetime.now().strftime("%H:%M:%S"),
                "suggestion": "建议密切关注，准备买入"
            }
            success = notifier.send_hourly_alert(hourly_details)
            print(f"小时级别预警测试: {'成功' if success else '失败'}")