#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精简版工具函数集 - 专注核心功能
包含日期处理、数据验证、基本计算等稳定功能
"""

import logging
import re
import warnings
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Tuple, List, Optional, Dict, Any

# 配置日志
logger = logging.getLogger('Utils')
logger.setLevel(logging.INFO)

# 创建控制台处理器
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def setup_logging(level: int = logging.INFO):
    """配置日志"""
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def validate_date_format(date_str: str, format_str: str = "%Y%m%d") -> bool:
    """
    验证日期格式
    :param date_str: 日期字符串
    :param format_str: 期望的格式
    :return: 是否有效
    """
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        raise ValueError(f"无效日期格式: {date_str}，期望格式: {format_str}")

def validate_date_range(start_date: str, end_date: str, format_str: str = "%Y%m%d") -> bool:
    """
    验证日期范围有效性
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param format_str: 日期格式
    :return: 是否有效
    """
    validate_date_format(start_date, format_str)
    validate_date_format(end_date, format_str)
    
    start_dt = datetime.strptime(start_date, format_str)
    end_dt = datetime.strptime(end_date, format_str)
    
    if start_dt > end_dt:
        raise ValueError(f"开始日期 {start_date} 不能晚于结束日期 {end_date}")
    
    # 检查日期范围是否合理（不超过5年）
    max_range = timedelta(days=365 * 5)
    if (end_dt - start_dt) > max_range:
        raise ValueError(f"日期范围不能超过5年")
    
    return True

def validate_symbol(symbol: str) -> str:
    """
    验证和标准化股票代码
    :param symbol: 股票代码
    :return: 标准化后的股票代码
    """
    if not isinstance(symbol, str):
        raise ValueError(f"股票代码必须是字符串: {type(symbol)}")
    
    # 支持多种格式: 000001, SH000001, SZ000001, 000001.SH, 000001.SZ
    pattern = r'^([A-Za-z]{2})?(\d{6})(\.[A-Za-z]{2})?$'
    match = re.match(pattern, symbol)
    if not match:
        raise ValueError(f"无效股票代码格式: {symbol}")
    
    # 提取数字部分
    digit_part = match.group(2)
    
    # 提取市场前缀
    prefix = match.group(1)
    suffix = match.group(3)
    
    # 确定市场前缀
    if prefix:
        market_prefix = prefix.upper()
    elif suffix:
        market_prefix = suffix[1:].upper()
    else:
        # 根据数字判断市场
        if digit_part.startswith(("6", "5", "9")):
            market_prefix = "SH"
        elif digit_part.startswith(("0", "3", "2")):
            market_prefix = "SZ"
        else:
            raise ValueError(f"无法识别的股票代码: {symbol}")
    
    # 返回标准化格式: SH000001 或 SZ000001
    return f"{market_prefix}{digit_part}"

def parse_date(date_str: str, format_str: str = "%Y%m%d") -> datetime:
    """
    解析日期字符串为日期对象
    :param date_str: 日期字符串
    :param format_str: 格式字符串
    :return: 日期对象
    """
    try:
        return datetime.strptime(date_str, format_str)
    except ValueError as e:
        raise ValueError(f"日期解析失败: {date_str} 格式: {format_str}") from e

def format_date(date_obj: datetime, format_str: str = "%Y%m%d") -> str:
    """
    格式化日期对象为字符串
    :param date_obj: 日期对象
    :param format_str: 格式字符串
    :return: 格式化后的日期字符串
    """
    return date_obj.strftime(format_str)

def convert_date_format(date_str: str, input_format: str, output_format: str) -> str:
    """
    转换日期字符串格式
    :param date_str: 原始日期字符串
    :param input_format: 输入格式
    :param output_format: 输出格式
    :return: 转换后的日期字符串
    """
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except ValueError as e:
        raise ValueError(f"日期格式转换失败: {date_str} 从 {input_format} 到 {output_format}") from e

def get_date_range(start_date: str, end_date: str, format_str: str = "%Y%m%d") -> List[str]:
    """
    获取日期范围内的所有日期
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param format_str: 日期格式
    :return: 日期字符串列表
    """
    start = parse_date(start_date, format_str)
    end = parse_date(end_date, format_str)
    
    dates = []
    current = start
    while current <= end:
        dates.append(format_date(current, format_str))
        current += timedelta(days=1)
    
    return dates

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    安全转换为浮点数
    :param value: 输入值
    :param default: 默认值
    :return: 转换后的浮点数
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    计算百分比变化
    :param old_value: 旧值
    :param new_value: 新值
    :return: 百分比变化
    """
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value * 100

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """
    计算最大回撤
    :param portfolio_values: 投资组合价值列表
    :return: 最大回撤百分比
    """
    if len(portfolio_values) < 2:
        return 0.0
        
    peak = portfolio_values[0]
    max_drawdown = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    return max_drawdown * 100  # 转换为百分比

def calculate_annual_return(portfolio_values: List[float], days: int) -> float:
    """
    计算年化收益率
    :param portfolio_values: 投资组合价值列表
    :param days: 交易天数
    :return: 年化收益率百分比
    """
    if len(portfolio_values) < 2:
        return 0.0
        
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    if days <= 0:
        return 0.0
        
    annual_return = (1 + total_return) ** (365 / days) - 1
    return annual_return * 100  # 转换为百分比

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率
    :param returns: 收益率列表（日收益率）
    :param risk_free_rate: 无风险利率（默认0）
    :return: 年化夏普比率
    """
    if len(returns) < 2:
        return 0.0
        
    # 计算超额收益率（减去无风险利率）
    excess_returns = [ret - risk_free_rate for ret in returns]
    
    # 计算均值和标准差
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess == 0:
        return 0.0
        
    # 年化处理（假设252个交易日）
    sharpe_ratio = mean_excess / std_excess * np.sqrt(252)
    return sharpe_ratio

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    计算Sortino比率（仅考虑下行风险的风险调整后收益指标）
    :param returns: 收益率列表（日收益率）
    :param risk_free_rate: 无风险利率（默认0）
    :return: 年化Sortino比率
    """
    if len(returns) < 2:
        return 0.0
    
    # 计算超额收益率（减去无风险利率）
    excess_returns = [ret - risk_free_rate for ret in returns]
    
    # 计算下行风险（仅考虑负的超额收益）
    negative_returns = [ret for ret in excess_returns if ret < 0]
    if not negative_returns:  # 无下行风险时，下行标准差为0（避免除零错误）
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    # 计算下行收益率的标准差（下行风险）
    downside_std = np.std(negative_returns)
    
    if downside_std == 0:
        return 0.0
    
    # 年化处理（假设252个交易日）
    sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252)
    return sortino_ratio

def validate_data_columns(data: Dict[str, Any], required_columns: List[str]) -> bool:
    """
    验证数据字典是否包含所需列
    :param data: 数据字典
    :param required_columns: 所需列名列表
    :return: 是否包含所有所需列
    """
    missing_columns = [col for col in required_columns if col not in data]
    if missing_columns:
        logger.warning(f"数据缺少必要列: {missing_columns}")
        return False
    return True

def normalize_dataframe_dates(df, date_column: str = 'date', date_format: str = "%Y%m%d"):
    """
    标准化DataFrame日期列
    :param df: DataFrame
    :param date_column: 日期列名
    :param date_format: 日期格式
    :return: 处理后的DataFrame
    """
    if date_column in df.columns:
        df[date_column] = df[date_column].astype(str)
        try:
            # 确保导入pandas（如果在当前环境中可用）
            import pandas as pd
            df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
        except Exception as e:
            logger.warning(f"日期列标准化失败: {e}")
    return df

def filter_data_by_date(data: List[Dict], start_date: str, end_date: str, 
                       date_field: str = 'date', date_format: str = "%Y%m%d") -> List[Dict]:
    """
    按日期范围过滤数据
    :param data: 数据列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param date_field: 日期字段名
    :param date_format: 日期格式
    :return: 过滤后的数据
    """
    if not data:
        return data
        
    try:
        start_dt = parse_date(start_date, date_format)
        end_dt = parse_date(end_date, date_format)
        
        filtered_data = []
        for item in data:
            if date_field in item:
                item_date = parse_date(str(item[date_field]), date_format)
                if start_dt <= item_date <= end_dt:
                    filtered_data.append(item)
        
        return filtered_data
    except Exception as e:
        logger.error(f"日期过滤失败: {e}")
        return data

def get_last_trading_day() -> datetime:
    """
    获取最后一个交易日
    :return: 最后一个交易日的日期对象
    """
    today = datetime.now()
    
    # 如果是周一，返回上周五
    if today.weekday() == 0:  # 周一
        last_trading_day = today - timedelta(days=3)
    # 如果是周日，返回上周五
    elif today.weekday() == 6:  # 周日
        last_trading_day = today - timedelta(days=2)
    # 其他情况返回前一天
    else:
        last_trading_day = today - timedelta(days=1)
    
    # 确保返回的日期是工作日（简单处理，不考虑节假日）
    return last_trading_day

def is_trading_hour() -> bool:
    """
    判断当前是否为交易时间
    :return: 是否为交易时间
    """
    now = datetime.now()
    
    # 检查是否为工作日（周一至周五）
    if now.weekday() >= 5:  # 周六或周日
        return False
    
    # 检查时间（A股交易时间：9:30-11:30, 13:00-15:00）
    current_time = now.time()
    morning_start = datetime.strptime("09:30:00", "%H:%M:%S").time()
    morning_end = datetime.strptime("11:30:00", "%H:%M:%S").time()
    afternoon_start = datetime.strptime("13:00:00", "%H:%M:%S").time()
    afternoon_end = datetime.strptime("15:00:00", "%H:%M:%S").time()
    
    return (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end)

def get_valid_date_range_str(days: int = 30) -> Tuple[str, str]:
    """
    获取有效的日期范围字符串
    :param days: 天数
    :return: (开始日期, 结束日期) 的字符串元组
    """
    end_date = get_last_trading_day()
    start_date = end_date - timedelta(days=days)
    return (format_date(start_date), format_date(end_date))