#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源有效性校验模块

该模块实现交易信号检测前的数据源有效性检查，确保：
1. 日线级别：至少覆盖近3个月日K线数据（≥60根）
2. 周线级别：至少覆盖近1年周K线数据（≥52根）
3. 输出数据源状态标记

作者: TradeTianYuan
日期: 2025-11-26
"""

import logging
import pandas as pd
from typing import Dict, Tuple, Optional

# 设置日志
logger = logging.getLogger(__name__)


class DataValidator:
    """数据源校验器类，提供数据有效性检查功能"""
    
    def __init__(self, min_daily_k_count: int = 60, min_weekly_k_count: int = 52):
        """初始化数据源校验器
        
        Args:
            min_daily_k_count: 日线K线最小数量要求（默认60根，约3个月）
            min_weekly_k_count: 周线K线最小数量要求（默认52根，约1年）
        """
        self.min_daily_k_count = min_daily_k_count
        self.min_weekly_k_count = min_weekly_k_count
        logger.info(f"数据源校验器初始化: 日线最小K线数量={min_daily_k_count}, 周线最小K线数量={min_weekly_k_count}")
    
    def validate_daily_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """验证日线数据有效性
        
        Args:
            df: 日线数据框
            
        Returns:
            包含验证结果的字典
        """
        logger.info("开始验证日线数据有效性...")
        
        # 检查数据框是否为空
        if df.empty:
            logger.error("日线数据为空")
            return {
                "valid": False,
                "reason": "日线数据为空",
                "data_count": 0,
                "min_required": self.min_daily_k_count,
                "status": "不足",
                "details": {
                    "can_detect_daily_divergence": False,
                    "can_detect_daily_buy_points": False
                }
            }
        
        # 检查数据列是否完整
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"日线数据缺少必要列: {missing_columns}")
            return {
                "valid": False,
                "reason": f"数据缺少必要列: {missing_columns}",
                "data_count": len(df),
                "min_required": self.min_daily_k_count,
                "status": "不足",
                "details": {
                    "can_detect_daily_divergence": False,
                    "can_detect_daily_buy_points": False
                }
            }
        
        # 检查数据量是否满足要求
        data_count = len(df)
        is_valid = data_count >= self.min_daily_k_count
        
        # 构建验证结果
        result = {
            "valid": is_valid,
            "reason": f"日线数据量{'满足' if is_valid else '不足'}要求（当前{data_count}根，最低要求{self.min_daily_k_count}根）",
            "data_count": data_count,
            "min_required": self.min_daily_k_count,
            "status": "满足" if is_valid else "不足",
            "details": {
                "can_detect_daily_divergence": is_valid,
                "can_detect_daily_buy_points": is_valid,
                "percentage_of_requirement": (data_count / self.min_daily_k_count) * 100 if self.min_daily_k_count > 0 else 0
            }
        }
        
        logger.info(f"日线数据验证结果: {'有效' if is_valid else '无效'} - {result['reason']}")
        return result
    
    def validate_weekly_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """验证周线数据有效性
        
        Args:
            df: 周线数据框
            
        Returns:
            包含验证结果的字典
        """
        logger.info("开始验证周线数据有效性...")
        
        # 检查数据框是否为空
        if df.empty:
            logger.error("周线数据为空")
            return {
                "valid": False,
                "reason": "周线数据为空",
                "data_count": 0,
                "min_required": self.min_weekly_k_count,
                "status": "不足",
                "details": {
                    "can_detect_weekly_trend": False,
                    "can_detect_weekly_divergence": False
                }
            }
        
        # 检查数据列是否完整
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"周线数据缺少必要列: {missing_columns}")
            return {
                "valid": False,
                "reason": f"数据缺少必要列: {missing_columns}",
                "data_count": len(df),
                "min_required": self.min_weekly_k_count,
                "status": "不足",
                "details": {
                    "can_detect_weekly_trend": False,
                    "can_detect_weekly_divergence": False
                }
            }
        
        # 检查数据量是否满足要求
        data_count = len(df)
        is_valid = data_count >= self.min_weekly_k_count
        
        # 构建验证结果
        result = {
            "valid": is_valid,
            "reason": f"周线数据量{'满足' if is_valid else '不足'}要求（当前{data_count}根，最低要求{self.min_weekly_k_count}根）",
            "data_count": data_count,
            "min_required": self.min_weekly_k_count,
            "status": "满足" if is_valid else "不足",
            "details": {
                "can_detect_weekly_trend": is_valid,
                "can_detect_weekly_divergence": is_valid,
                "percentage_of_requirement": (data_count / self.min_weekly_k_count) * 100 if self.min_weekly_k_count > 0 else 0
            }
        }
        
        logger.info(f"周线数据验证结果: {'有效' if is_valid else '无效'} - {result['reason']}")
        return result
    
    def validate_all_data(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> Dict[str, any]:
        """验证所有数据源的有效性
        
        Args:
            daily_df: 日线数据框
            weekly_df: 周线数据框
            
        Returns:
            综合验证结果
        """
        logger.info("开始综合验证所有数据源...")
        
        # 分别验证日线和周线数据
        daily_result = self.validate_daily_data(daily_df)
        weekly_result = self.validate_weekly_data(weekly_df)
        
        # 综合验证结果
        overall_valid = daily_result["valid"] and weekly_result["valid"]
        
        # 生成数据源状态标记
        daily_status = daily_result["status"]
        weekly_status = weekly_result["status"]
        
        # 综合状态说明
        status_description = f"日线数据: {daily_status}, 周线数据: {weekly_status}"
        
        # 构建综合结果
        result = {
            "overall_valid": overall_valid,
            "status_description": status_description,
            "daily_validation": daily_result,
            "weekly_validation": weekly_result,
            "can_proceed": overall_valid,
            "can_detect_daily_divergence": daily_result["details"]["can_detect_daily_divergence"],
            "can_detect_daily_buy_points": daily_result["details"]["can_detect_daily_buy_points"],
            "can_detect_weekly_trend": weekly_result["details"]["can_detect_weekly_trend"],
            "can_detect_weekly_divergence": weekly_result["details"]["can_detect_weekly_divergence"],
            "data_summary": {
                "daily_k_count": daily_result["data_count"],
                "daily_min_required": daily_result["min_required"],
                "weekly_k_count": weekly_result["data_count"],
                "weekly_min_required": weekly_result["min_required"]
            }
        }
        
        # 生成详细的验证报告
        report = self._generate_validation_report(result)
        logger.info(f"数据源综合验证结果: {'通过' if overall_valid else '未通过'}")
        logger.info(f"数据源状态: {status_description}")
        
        # 添加报告到结果
        result["validation_report"] = report
        
        return result
    
    def _generate_validation_report(self, result: Dict[str, any]) -> str:
        """生成详细的验证报告
        
        Args:
            result: 验证结果字典
            
        Returns:
            格式化的验证报告字符串
        """
        report_lines = ["===== 数据源验证报告 ====="]
        
        # 综合状态
        report_lines.append(f"综合状态: {'通过' if result['overall_valid'] else '未通过'}")
        report_lines.append(f"数据源状态: {result['status_description']}")
        report_lines.append("")
        
        # 日线数据详情
        daily = result["daily_validation"]
        report_lines.append("【日线数据验证】")
        report_lines.append(f"  数据量: {daily['data_count']}根 (要求≥{daily['min_required']}根)")
        report_lines.append(f"  状态: {daily['status']}")
        report_lines.append(f"  原因: {daily['reason']}")
        report_lines.append(f"  可检测日线背驰: {'是' if daily['details']['can_detect_daily_divergence'] else '否'}")
        report_lines.append(f"  可检测日线买点: {'是' if daily['details']['can_detect_daily_buy_points'] else '否'}")
        report_lines.append("")
        
        # 周线数据详情
        weekly = result["weekly_validation"]
        report_lines.append("【周线数据验证】")
        report_lines.append(f"  数据量: {weekly['data_count']}根 (要求≥{weekly['min_required']}根)")
        report_lines.append(f"  状态: {weekly['status']}")
        report_lines.append(f"  原因: {weekly['reason']}")
        report_lines.append(f"  可检测周线趋势: {'是' if weekly['details']['can_detect_weekly_trend'] else '否'}")
        report_lines.append(f"  可检测周线背驰: {'是' if weekly['details']['can_detect_weekly_divergence'] else '否'}")
        report_lines.append("")
        
        # 操作建议
        report_lines.append("【操作建议】")
        if result['overall_valid']:
            report_lines.append("  ✓ 数据验证通过，可以继续执行交易信号检测")
        else:
            report_lines.append("  ✗ 数据验证未通过，建议:")
            if not daily['valid']:
                report_lines.append(f"     - 日线数据不足，请获取至少{daily['min_required']}根日K线数据")
            if not weekly['valid']:
                report_lines.append(f"     - 周线数据不足，请获取至少{weekly['min_required']}根周K线数据")
        
        report_lines.append("=========================")
        
        return "\n".join(report_lines)
    
    def get_data_status_mark(self, daily_valid: bool, weekly_valid: bool) -> str:
        """获取数据源状态标记
        
        Args:
            daily_valid: 日线数据是否有效
            weekly_valid: 周线数据是否有效
            
        Returns:
            数据源状态标记字符串
        """
        if daily_valid and weekly_valid:
            return "数据源：满足"
        elif not daily_valid and not weekly_valid:
            return "数据源：日线和周线数据均不足"
        elif not daily_valid:
            return "数据源：日线数据不足"
        else:
            return "数据源：周线数据不足"


if __name__ == "__main__":
    # 测试用例
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建有效的日线数据
    dates = [datetime.now() - timedelta(days=i) for i in range(70)]
    dates.reverse()
    daily_data = {
        'date': dates,
        'open': np.random.random(70) * 10 + 100,
        'high': np.random.random(70) * 5 + 102,
        'low': np.random.random(70) * 5 + 98,
        'close': np.random.random(70) * 10 + 100,
        'volume': np.random.random(70) * 1000000
    }
    daily_df = pd.DataFrame(daily_data)
    
    # 创建有效的周线数据
    weekly_dates = [datetime.now() - timedelta(weeks=i) for i in range(60)]
    weekly_dates.reverse()
    weekly_data = {
        'date': weekly_dates,
        'open': np.random.random(60) * 10 + 100,
        'high': np.random.random(60) * 5 + 102,
        'low': np.random.random(60) * 5 + 98,
        'close': np.random.random(60) * 10 + 100,
        'volume': np.random.random(60) * 5000000
    }
    weekly_df = pd.DataFrame(weekly_data)
    
    # 创建验证器
    validator = DataValidator()
    
    # 执行验证
    result = validator.validate_all_data(daily_df, weekly_df)
    print(result['validation_report'])