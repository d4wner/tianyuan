#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘输出格式模块

该模块负责生成统一的实盘交易信号输出格式，按照文档要求格式化所有交易信号。

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
from typing import Dict, Optional, Union

# 设置日志
logger = logging.getLogger(__name__)


class SignalFormatter:
    """交易信号格式化类"""
    
    def __init__(self):
        """初始化信号格式化器"""
        logger.info("实盘输出格式模块初始化")
        
        # 定义默认仓位比例（实际应从用户配置读取）
        self.default_daily_position_ratios = {
            "日线二买": 0.3,  # 假设用户原有二买仓位为30%
            "日线一买": 0.2,  # 假设用户原有一买仓位为20%
            "日线三买": 0.2,  # 假设用户原有三买仓位为20%
            "破中枢反抽": 0.1   # 假设用户原有反抽仓位为10%
        }
        
        # 分钟子仓位比例范围
        self.minute_subposition_ranges = {
            "30分钟": (0.6, 0.7),    # 30分钟向上笔：60%-70%
            "15分钟": (0.2, 0.4)     # 15分钟向上笔：20%-40%
        }
        
        # 加仓子仓位比例范围
        self.addition_subposition_ranges = {
            "周线底背驰+30分钟底分": (0.3, 0.4),  # 30%-40%
            "日线二次底背驰+15分钟底分": (0.1, 0.2)  # 10%-20%
        }
        
        # 额外报告数据，由外部设置
        self.report_data = {}
    
    def format_trading_signal(self, 
                            weekly_trend_result: Dict[str, any],
                            daily_buy_result: Dict[str, any],
                            minute_signal_type: Optional[str] = None,
                            is_addition: bool = False,
                            data_validity: Dict[str, bool] = None,
                            current_total_position: float = 0,
                            position_ratio_config: Optional[Dict[str, float]] = None,
                            position_ratio: float = 0,
                            addition_condition: str = "",
                            allocation_details: Dict = None) -> str:
        """格式化交易信号
        
        输出格式：
        「信号类型：XX（周线多头+日线二买/一买/三买/反抽）-XX（30/15分钟建仓/加仓）｜日线买点类型（核心/辅助/兜底）：XX｜分钟子仓位比例：XX%｜实际仓位：XX%（用户原有日线仓位×子比例）｜加仓条件：XX｜累计总仓位上限：XX%｜止盈触发条件：XX｜数据源：XX｜周线置信度：XX级｜动态校准因子：XX」
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            minute_signal_type: 分钟级别信号类型（"30分钟"或"15分钟"）
            is_addition: 是否为加仓信号
            data_validity: 数据源有效性（{"daily": True/False, "weekly": True/False}）
            current_total_position: 当前总仓位（百分比）
            position_ratio_config: 用户自定义仓位比例配置
            position_ratio: 当前仓位比例
            addition_condition: 加仓条件描述
            allocation_details: 仓位分配详情
            
        Returns:
            格式化的交易信号字符串
        """
        logger.info("开始格式化交易信号...")
        
        # 校验必要的输入数据
        if not weekly_trend_result or not daily_buy_result:
            logger.error("缺少必要的周线或日线信号数据")
            return "「无效原因：缺少必要的周线或日线信号数据」"
        
        # 使用用户自定义仓位比例或默认值
        if position_ratio_config:
            position_ratios = position_ratio_config
        else:
            position_ratios = self.default_daily_position_ratios
        
        # 检查数据源有效性
        if data_validity:
            data_status = "满足" if data_validity.get("daily", False) and data_validity.get("weekly", False) else "不足"
        else:
            data_status = "未知"
        
        # 检查是否为无效信号
        invalid_reason = self._check_invalid_signal(weekly_trend_result, daily_buy_result, data_validity)
        if invalid_reason:
            return f"「无效原因：{invalid_reason}」"
        
        # 构建信号类型部分
        weekly_trend = "周线多头" if weekly_trend_result.get("bullish_trend", False) else "周线非多头"
        daily_signal = daily_buy_result.get("strongest_signal", "无买点")
        
        # 确定分钟级别信号类型
        if not minute_signal_type:
            # 根据日线买点类型自动确定优先的分钟级别
            if daily_signal == "日线二买":
                minute_signal_type = "30分钟"
            else:
                minute_signal_type = "15分钟"
        
        # 构建分钟级别部分
        minute_part = f"{minute_signal_type}{'加仓' if is_addition else '建仓'}"
        
        # 信号类型组合
        signal_type = f"{weekly_trend}+{daily_signal}-{minute_part}"
        
        # 日线买点类型优先级
        signal_type_priority = daily_buy_result.get("signal_type_priority", "无")
        
        # 确定分钟子仓位比例
        if allocation_details and minute_signal_type in allocation_details:
            subposition_ratio = allocation_details[minute_signal_type]
        else:
            subposition_ratio = self._calculate_subposition_ratio(daily_signal, minute_signal_type, is_addition)
        subposition_percent = f"{subposition_ratio*100:.0f}%"
        
        # 计算实际仓位
        daily_position_ratio = position_ratios.get(daily_signal, 0.1)
        if position_ratio > 0:
            actual_position = position_ratio
        else:
            actual_position = daily_position_ratio * subposition_ratio
        actual_position_percent = f"{actual_position*100:.1f}%"
        actual_position_detail = f"{actual_position_percent}（用户原有日线仓位×{subposition_ratio*100:.0f}%）"
        
        # 检查加仓条件
        if not addition_condition:
            addition_condition = self._check_addition_condition(is_addition, weekly_trend_result, daily_buy_result)
        
        # 累计总仓位上限（根据文档不超过80%）
        total_position_limit = "80%"
        
        # 检查止盈触发条件
        take_profit_condition = "未满足"
        
        # 获取周线置信度等级（如果有），如果是数值则转换为等级
        confidence_level = weekly_trend_result.get("confidence_level", "未知")
        if isinstance(confidence_level, (int, float)):
            if confidence_level >= 80:
                confidence_level = "高"
            elif confidence_level >= 60:
                confidence_level = "中"
            else:
                confidence_level = "低"
        
        # 计算动态校准因子
        calibration_factor = self._calculate_calibration_factor(weekly_trend_result, daily_buy_result)
        
        # 构建完整的输出字符串
        formatted_signal = f"「信号类型：{signal_type}｜日线买点类型（核心/辅助/兜底）：{signal_type_priority}｜分钟子仓位比例：{subposition_percent}｜实际仓位：{actual_position_detail}｜加仓条件：{addition_condition}｜累计总仓位上限：{total_position_limit}｜止盈触发条件：{take_profit_condition}｜数据源：{data_status}｜周线置信度：{confidence_level}｜动态校准因子：{calibration_factor}」"
        
        logger.info(f"交易信号格式化完成: {formatted_signal}")
        return formatted_signal
    
    def _check_invalid_signal(self, 
                            weekly_trend_result: Dict[str, any], 
                            daily_buy_result: Dict[str, any],
                            data_validity: Optional[Dict[str, bool]] = None) -> Optional[str]:
        """检查信号是否无效
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            data_validity: 数据源有效性
            
        Returns:
            无效原因，如果信号有效则返回None
        """
        # 检查数据源是否不足
        if data_validity:
            if not data_validity.get("daily", False):
                return "日线数据源不足"
            if not data_validity.get("weekly", False):
                return "周线数据源不足"
        
        # 检查是否为周线多头
        if not weekly_trend_result.get("bullish_trend", False):
            return "未满足周线多头"
        
        # 检查是否有日线买点
        strongest_signal = daily_buy_result.get("strongest_signal", "无买点")
        if strongest_signal == "无买点":
            return "无有效的日线买点"
        
        # 检查是否为伪背驰（简化处理，实际需要更详细的背驰检测）
        if weekly_trend_result.get("is_fake_divergence", False):
            return "存在伪背驰"
        
        # 检查是否无量能底分型
        if weekly_trend_result.get("has_volume_fractal") is False:
            return "无量能底分型"
        
        # 检查是否处于极端行情
        if weekly_trend_result.get("extreme_market", False):
            return "处于极端行情"
        
        # 检查周线置信度等级
        confidence_level = weekly_trend_result.get("confidence_level", "未知")
        if confidence_level == "低":
            return "周线置信度等级低"
        # 如果是数值型置信度，检查是否低于阈值
        elif isinstance(confidence_level, (int, float)) and confidence_level < 40:
            return "周线置信度数值过低"
        
        # 检查日线买点有效性
        if daily_buy_result.get("signal_type_priority") == "兜底" and confidence_level in ["低", 0]:
            return "兜底买点且周线置信度低"
        
        return None
    
    def _calculate_subposition_ratio(self, 
                                    daily_signal: str, 
                                    minute_signal_type: str, 
                                    is_addition: bool) -> float:
        """计算分钟子仓位比例
        
        Args:
            daily_signal: 日线买点类型
            minute_signal_type: 分钟级别信号类型
            is_addition: 是否为加仓信号
            
        Returns:
            子仓位比例（0-1之间）
        """
        # 建仓的子仓位比例
        if not is_addition:
            if minute_signal_type == "30分钟":
                # 对于30分钟建仓，二买使用高比例，其他使用中等比例
                if daily_signal == "日线二买":
                    return 0.65  # 二买核心买点，优先使用高比例（65%）
                else:
                    return 0.60  # 其他买点使用较低比例（60%）
            elif minute_signal_type == "15分钟":
                # 对于15分钟建仓，二买也使用较高比例
                if daily_signal == "日线二买":
                    return 0.35  # 二买作为补充子仓位（35%）
                else:
                    return 0.30  # 一买/三买使用中等比例（30%）
        
        # 加仓的子仓位比例
        else:
            if daily_signal == "日线二买":
                return 0.35  # 二买加仓优先级最高（35%）
            elif daily_signal == "日线一买":
                return 0.15  # 一买加仓（15%）
            elif daily_signal == "日线三买":
                return 0.10  # 三买加仓（10%）
            else:
                return 0.05  # 反抽不加仓或最小仓位
        
        # 默认返回最小比例
        return 0.2
    
    def _check_addition_condition(self, 
                                is_addition: bool, 
                                weekly_trend_result: Dict[str, any], 
                                daily_buy_result: Dict[str, any]) -> str:
        """检查加仓条件
        
        Args:
            is_addition: 是否为加仓信号
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            
        Returns:
            加仓条件状态描述
        """
        if not is_addition:
            return "未满足"  # 不是加仓信号
        
        # 检查周线多头趋势是否保持
        is_weekly_bullish = weekly_trend_result.get("bullish_trend", False)
        
        # 检查是否有日线二次底背驰或周线底背驰（简化处理）
        has_divergence = "日线二次底背驰"  # 实际应该检查背驰条件
        
        # 检查日线买点类型
        daily_signal = daily_buy_result.get("strongest_signal", "无买点")
        
        if is_weekly_bullish and daily_signal == "日线二买":
            return f"{has_divergence} + 周线多头保持"
        elif is_weekly_bullish:
            return f"{has_divergence} + 周线多头保持"
        else:
            return "未满足周线多头"
    
    def format_invalid_signal(self, invalid_reason: str) -> str:
        """格式化无效信号
        
        Args:
            invalid_reason: 无效原因
            
        Returns:
            格式化的无效信号字符串
        """
        return f"「无效原因：{invalid_reason}」"
    
    def generate_trading_report(self, 
                              signals: list, 
                              weekly_trend_result: Dict[str, any], 
                              daily_buy_result: Dict[str, any], 
                              data_validity: Dict[str, bool]) -> str:
        """生成完整的交易报告
        
        Args:
            signals: 信号列表
            report_data: 包含所有报告数据的字典
            
        Returns:
            格式化的交易报告字符串
        """
        # 创建报告数据字典
        report_data = {
            "weekly_trend": weekly_trend_result,
            "daily_buy": daily_buy_result,
            "data_validity": data_validity
        }
        
        report_lines = ["===== 交易信号综合报告 ====="]
        
        # 基本信息
        report_lines.append(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据源状态: {'满足' if data_validity.get('daily', False) and data_validity.get('weekly', False) else '不足'}")
        report_lines.append(f"周线趋势: {'多头' if weekly_trend_result.get('bullish_trend', False) else '非多头'}")
        report_lines.append(f"周线置信度等级: {weekly_trend_result.get('confidence_level', '未知')}")
        report_lines.append(f"最强日线买点: {daily_buy_result.get('strongest_signal', '无买点')}")
        report_lines.append(f"日线买点优先级: {daily_buy_result.get('signal_type_priority', '无')}")
        
        # 仓位分配信息 - 从self.report_data获取或使用默认值
        position_allocation = getattr(self, 'report_data', {}).get("position_allocation", {})
        if position_allocation:
            report_lines.append("\n【仓位分配信息】")
            report_lines.append(f"  30分钟仓位: {position_allocation.get('30min', position_allocation.get('30分钟', 0))*100:.1f}%")
            report_lines.append(f"  15分钟仓位: {position_allocation.get('15min', position_allocation.get('15分钟', 0))*100:.1f}%")
            report_lines.append(f"  总仓位: {position_allocation.get('total', sum(position_allocation.values()))*100:.1f}%")
        
        # 加仓信息 - 从self.report_data获取或使用默认值
        addition_check = getattr(self, 'report_data', {}).get("addition_check", {})
        if addition_check:
            report_lines.append("\n【加仓信息】")
            report_lines.append(f"  可加仓: {'是' if addition_check.get('can_add', False) else '否'}")
            if addition_check.get('can_add', False):
                report_lines.append(f"  加仓类型: {addition_check.get('addition_type', '')}")
                report_lines.append(f"  加仓仓位: {addition_check.get('addition_position', 0)*100:.1f}%")
                report_lines.append(f"  加仓原因: {addition_check.get('addition_reason', '')}")
            else:
                report_lines.append(f"  不可加仓原因: {addition_check.get('addition_reason', '')}")
        
        # 止盈信息 - 从self.report_data获取或使用默认值
        stop_profit = getattr(self, 'report_data', {}).get("stop_profit", {})
        if stop_profit:
            report_lines.append("\n【止盈信息】")
            report_lines.append(f"  应止盈: {'是' if stop_profit.get('should_take_profit', False) else '否'}")
            if stop_profit.get('should_take_profit', False):
                report_lines.append(f"  止盈仓位: {stop_profit.get('stop_profit_amount', 0)*100:.1f}%")
                report_lines.append(f"  止盈原因: {stop_profit.get('stop_profit_reason', '')}")
        
        # 添加动态校准因子信息到报告
        report_lines.append("\n【动态校准因子信息】")
        calibration_factor = self._calculate_calibration_factor(weekly_trend_result, daily_buy_result)
        report_lines.append(f"  当前动态校准因子: {calibration_factor:.2f}")
        
        # 根据校准因子提供仓位调整建议
        if calibration_factor > 1.1:
            report_lines.append(f"  建议: 可适当增加仓位比例")
        elif calibration_factor < 0.9:
            report_lines.append(f"  建议: 应适当减少仓位比例，控制风险")
        else:
            report_lines.append(f"  建议: 保持正常仓位比例")
        
        report_lines.append("")
        
        # 信号列表
        report_lines.append("【交易信号】")
        if signals:
            for i, signal in enumerate(signals, 1):
                report_lines.append(f"{i}. {signal}")
        else:
            report_lines.append("无有效交易信号")
        report_lines.append("")
        
        # 交易建议
        report_lines.append("【交易建议】")
        
        # 基于周线和日线信号生成建议
        if weekly_trend_result.get('bullish_trend', False):
            report_lines.append("✓ 周线多头趋势确认，可考虑建仓")
            
            if daily_buy_result.get('strongest_signal') == "日线二买":
                report_lines.append("  - 重点关注日线二买信号，优先匹配30分钟向上笔")
                report_lines.append("  - 建议使用60%-70%的子仓位比例")
                report_lines.append("  - 二买加仓优先级最高")
            elif daily_buy_result.get('strongest_signal') in ["日线一买", "日线三买"]:
                report_lines.append("  - 当前为辅助买点，建议使用20%-40%的子仓位比例")
                report_lines.append("  - 可考虑15分钟向上笔作为建仓时机")
            elif daily_buy_result.get('strongest_signal') == "破中枢反抽":
                report_lines.append("  - 当前为兜底买点，建议最小仓位试探")
                report_lines.append("  - 严格监控风险，准备止损")
        else:
            report_lines.append("✗ 周线非多头趋势，建议谨慎操作")
            report_lines.append("  - 暂不建议建仓或加仓")
            report_lines.append("  - 等待周线多头趋势确认")
        
        report_lines.append("")
        report_lines.append("【风险提示】")
        report_lines.append("1. 严格遵循'周线多头+日线结构'的核心前提")
        report_lines.append("2. 二买为核心买点，确定性最高")
        report_lines.append("3. 无强制止损，但需依托更大级别底背驰加仓")
        report_lines.append("4. 止盈仅绑定日线顶背驰+大分钟顶分型")
        report_lines.append("5. 累计总仓位不超过80%")
        report_lines.append("6. 注意根据周线置信度等级和动态校准因子调整仓位")
        
        report_lines.append("===================================")
        
        return "\n".join(report_lines)
    
    def format_stop_profit_signal(self, stop_profit_amount: float, stop_profit_reason: str, 
                               strongest_signal: str, position_allocation: Dict) -> str:
        """格式化止盈信号
        
        Args:
            stop_profit_amount: 止盈仓位比例
            stop_profit_reason: 止盈原因
            strongest_signal: 最强信号类型
            position_allocation: 仓位分配信息
            
        Returns:
            格式化的止盈信号字符串
        """
        total_position = position_allocation.get("total", 0)
        stop_profit_percent = f"{stop_profit_amount*100:.1f}%"
        
        return f"「止盈信号：{strongest_signal}｜止盈仓位：{stop_profit_percent}（当前总仓位：{total_position*100:.1f}%）｜止盈原因：{stop_profit_reason}」"
    
    def format_no_signal(self, strongest_signal: str) -> str:
        """格式化无信号说明
        
        Args:
            strongest_signal: 最强信号类型
            
        Returns:
            格式化的无信号说明字符串
        """
        if strongest_signal == "无买点":
            return "「无信号：未检测到有效日线买点」"
        else:
            return "「无信号：未满足分钟级别建仓条件」"
    
    def _calculate_calibration_factor(self, weekly_trend_result: Dict[str, any], 
                                    daily_buy_result: Dict[str, any]) -> float:
        """计算动态校准因子
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            
        Returns:
            动态校准因子（0.8-1.2之间）
        """
        # 基础校准因子
        base_factor = 1.0
        
        # 根据周线置信度等级调整
        confidence_level = weekly_trend_result.get("confidence_level", "未知")
        if isinstance(confidence_level, (int, float)):
            # 直接基于置信度数值计算
            confidence_factor = 0.8 + (confidence_level / 100) * 0.4  # 置信度转换为0.8-1.2的因子
            base_factor *= confidence_factor
        else:
            # 基于等级计算
            if confidence_level == "高":
                base_factor += 0.1
            elif confidence_level == "低":
                base_factor -= 0.1
        
        # 根据日线买点类型调整
        strongest_signal = daily_buy_result.get("strongest_signal", "无买点")
        if strongest_signal == "日线二买":
            base_factor += 0.05
        elif strongest_signal == "破中枢反抽":
            base_factor -= 0.05
        
        # 确保校准因子在合理范围内
        base_factor = max(0.8, min(1.2, base_factor))
        
        return round(base_factor, 2)


def run_tests():
    """运行测试用例，验证信号格式化功能"""
    # 创建测试数据
    mock_daily_buy_result = {
        "strongest_signal": "日线二买",
        "signal_type_priority": "核心",
        "satisfied_signals_count": 1,
        "signals": {
            "second_buy": {"detected": True, "details": {}},
            "first_buy": {"detected": False, "details": {}},
            "third_buy": {"detected": False, "details": {}},
            "reverse_pullback": {"detected": False, "details": {}}
        }
    }
    
    mock_weekly_trend_result = {
        "bullish_trend": True,
        "confidence_level": 0.85
    }
    
    mock_minute_buy_result = {
        "signal_type": "30分钟向上笔"
    }
    
    mock_data_validity = {
        "daily": True,
        "weekly": True
    }
    
    # 创建信号格式化器实例
    formatter = SignalFormatter()
    
    # 测试1: 测试二买建仓信号格式化（包含完整参数）
    print("\n测试1: 测试二买建仓信号格式化（包含完整参数）")
    formatted_signal = formatter.format_trading_signal(
        weekly_trend_result=mock_weekly_trend_result,
        daily_buy_result=mock_daily_buy_result,
        minute_signal_type="30分钟",
        is_addition=False,
        data_validity=mock_data_validity,
        current_total_position=0.0,
        position_ratio=0.3
    )
    print(formatted_signal)
    
    # 测试2: 测试一买建仓信号格式化（包含完整参数）
    print("\n测试2: 测试一买建仓信号格式化（包含完整参数）")
    mock_daily_buy_result["strongest_signal"] = "日线一买"
    mock_daily_buy_result["signal_type_priority"] = "辅助"
    formatted_signal = formatter.format_trading_signal(
        weekly_trend_result=mock_weekly_trend_result,
        daily_buy_result=mock_daily_buy_result,
        minute_signal_type="15分钟",
        is_addition=False,
        data_validity=mock_data_validity,
        current_total_position=0.0,
        position_ratio=0.2
    )
    print(formatted_signal)
    
    # 测试3: 测试加仓信号格式化
    print("\n测试3: 测试加仓信号格式化")
    mock_weekly_trend_result["confidence_level"] = 0.9
    formatted_addition_signal = formatter.format_trading_signal(
        weekly_trend_result=mock_weekly_trend_result,
        daily_buy_result=mock_daily_buy_result,
        minute_signal_type="30分钟",
        is_addition=True,
        data_validity=mock_data_validity,
        current_total_position=0.3,
        position_ratio=0.15
    )
    print(formatted_addition_signal)
    
    # 测试4: 测试无效信号格式化
    print("\n测试4: 测试无效信号格式化")
    mock_data_validity["daily"] = False
    formatted_invalid_signal = formatter.format_invalid_signal(
        invalid_reason="日线数据无效"
    )
    print(formatted_invalid_signal)
    
    # 测试5: 测试止盈信号格式化
    print("\n测试5: 测试止盈信号格式化")
    stop_profit_signal = formatter.format_stop_profit_signal(
        stop_profit_amount=0.2,
        stop_profit_reason="满足止盈条件",
        strongest_signal="日线二买",
        position_allocation={"total": 0.5}
    )
    print(stop_profit_signal)
    
    # 测试6: 测试无信号说明格式化
    print("\n测试6: 测试无信号说明格式化")
    no_signal = formatter.format_no_signal(
        strongest_signal="无买点"
    )
    print(no_signal)
    
    # 测试7: 生成综合报告（完整信息）
    print("\n测试7: 生成综合报告（完整信息）")
    
    # 设置report_data
    formatter.report_data = {
        "position_allocation": {
            "30分钟": 0.21,
            "15分钟": 0.09,
            "total": 0.3
        },
        "addition_check": {
            "can_add": True,
            "addition_type": "周线底背驰+30分钟底分",
            "addition_position": 0.15,
            "addition_reason": "满足加仓条件"
        },
        "stop_profit": {
            "should_take_profit": False
        }
    }
    
    # 恢复有效数据
    mock_data_validity["daily"] = True
    mock_weekly_trend_result["bullish_trend"] = True
    
    trading_signals = [formatted_signal, formatted_addition_signal]
    report = formatter.generate_trading_report(
        signals=trading_signals,
        weekly_trend_result=mock_weekly_trend_result,
        daily_buy_result=mock_daily_buy_result,
        data_validity=mock_data_validity
    )
    print(report)
    
    # 测试8: 生成低周线置信度的报告
    print("\n测试8: 生成低周线置信度的报告")
    mock_weekly_trend_result["confidence_level"] = 0.6
    
    low_confidence_report = formatter.generate_trading_report(
        signals=trading_signals,
        weekly_trend_result=mock_weekly_trend_result,
        daily_buy_result=mock_daily_buy_result,
        data_validity=mock_data_validity
    )
    print(low_confidence_report)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    run_tests()