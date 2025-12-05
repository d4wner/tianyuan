#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用型缠论量化交易策略
适配全波动ETF的统一量化交易系统
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UniversalChanlunStrategy:
    """
    通用型缠论量化交易策略类
    支持全波动ETF的自动适配和参数动态调整
    """
    
    def __init__(self):
        """
        初始化策略参数
        """
        # 数据源要求
        self.MIN_DAILY_BARS = 60  # 日线数据要求：至少3个月（约60根）
        self.MIN_WEEKLY_BARS = 52  # 周线数据要求：至少1年（约52根）
        
        # 波动分类标准
        self.VOLATILITY_THRESHOLD_LOW = 10.0    # 低波动率阈值
        self.VOLATILITY_THRESHOLD_MEDIUM = 18.0  # 中波动率阈值
        
        # 动态参数映射表
        self.DYNAMIC_PARAMS = {
            'low': {
                'central_amplitude_min': 5.0,    # 中枢振幅要求
                'break_threshold_ratio': 0.995,  # 破中枢阈值比例
                'rebound_threshold_ratio': 1.005,  # 反抽阈值比例
                'consecutive_days': 2,           # 连续达标要求
                'rebound_time_window': 5,        # 反抽时间窗口
                'volume_threshold_ratio': 0.8,   # 量能验证阈值
                'min30_position_ratio_buy2': (65, 70),  # 30分钟子仓位比例（二买）
                'min15_position_ratio_buy13': (20, 30)  # 15分钟子仓位比例（一买/三买）
            },
            'medium': {
                'central_amplitude_min': 8.0,    # 中枢振幅要求
                'break_threshold_ratio': 0.99,   # 破中枢阈值比例
                'rebound_threshold_ratio': 1.01,  # 反抽阈值比例
                'consecutive_days': 1,           # 2日内≥1日达标
                'rebound_time_window': 6,        # 反抽时间窗口
                'volume_threshold_ratio': 0.85,  # 量能验证阈值
                'min30_position_ratio_buy2': (60, 65),  # 30分钟子仓位比例（二买）
                'min15_position_ratio_buy13': (25, 35)  # 15分钟子仓位比例（一买/三买）
            },
            'high': {
                'central_amplitude_min': 10.0,   # 中枢振幅要求
                'break_threshold_ratio': 0.985,  # 破中枢阈值比例
                'rebound_threshold_ratio': 1.015,  # 反抽阈值比例
                'consecutive_days': 1,           # 2日内≥1日达标
                'rebound_time_window': 7,        # 反抽时间窗口
                'volume_threshold_ratio': 0.9,   # 量能验证阈值
                'min30_position_ratio_buy2': (55, 60),  # 30分钟子仓位比例（二买）
                'min15_position_ratio_buy13': (30, 40)  # 15分钟子仓位比例（一买/三买）
            }
        }
        
        # 初始化结果变量
        self.current_etf_code = None
        self.volatility_level = None
        self.dynamic_params = None
        self.data_source_status = {'daily': False, 'weekly': False}
        self.central_range = None
        self.weekly_confidence = None
        
    def validate_data_source(self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame) -> Dict[str, bool]:
        """
        数据源有效性校验
        
        Args:
            daily_data: 日线数据
            weekly_data: 周线数据
            
        Returns:
            数据源状态字典
        """
        # 校验日线数据
        daily_status = len(daily_data) >= self.MIN_DAILY_BARS
        
        # 校验周线数据
        weekly_status = len(weekly_data) >= self.MIN_WEEKLY_BARS
        
        self.data_source_status = {
            'daily': daily_status,
            'weekly': weekly_status
        }
        
        return self.data_source_status
    
    def calculate_volatility(self, daily_data: pd.DataFrame) -> float:
        """
        计算ETF波动率
        
        Args:
            daily_data: 日线数据
            
        Returns:
            波动率百分比
        """
        # 获取最近60天数据
        recent_data = daily_data.tail(60)
        
        if len(recent_data) < 60:
            # 如果数据不足60天，使用所有可用数据
            recent_data = daily_data
        
        # 计算波动率
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        avg_price = recent_data['close'].mean()
        
        volatility = ((high_price - low_price) / avg_price) * 100
        
        return volatility
    
    def classify_volatility(self, volatility: float) -> str:
        """
        根据波动率分类ETF
        
        Args:
            volatility: 波动率百分比
            
        Returns:
            波动等级：'low'/'medium'/'high'
        """
        if volatility <= self.VOLATILITY_THRESHOLD_LOW:
            return 'low'
        elif volatility <= self.VOLATILITY_THRESHOLD_MEDIUM:
            return 'medium'
        else:
            return 'high'
    
    def get_dynamic_params(self, volatility_level: str, weekly_confidence: Optional[str] = None) -> Dict:
        """
        获取动态参数
        
        Args:
            volatility_level: 波动等级
            weekly_confidence: 周线置信度
            
        Returns:
            动态参数字典
        """
        base_params = self.DYNAMIC_PARAMS.get(volatility_level, self.DYNAMIC_PARAMS['medium'])
        
        # 根据周线置信度调整子仓位比例
        if weekly_confidence and isinstance(base_params.get('min30_position_ratio_buy2'), tuple):
            params = base_params.copy()
            
            if weekly_confidence == 'high':
                # 使用上限
                params['min30_position_ratio_buy2'] = base_params['min30_position_ratio_buy2'][1]
                params['min15_position_ratio_buy13'] = base_params['min15_position_ratio_buy13'][1]
            elif weekly_confidence == 'medium':
                # 使用中值
                params['min30_position_ratio_buy2'] = sum(base_params['min30_position_ratio_buy2']) / 2
                params['min15_position_ratio_buy13'] = sum(base_params['min15_position_ratio_buy13']) / 2
            else:
                # 使用下限
                params['min30_position_ratio_buy2'] = base_params['min30_position_ratio_buy2'][0]
                params['min15_position_ratio_buy13'] = base_params['min15_position_ratio_buy13'][0]
            
            return params
        
        return base_params
    
    def kline_inclusion_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        K线包含处理
        
        Args:
            data: 原始K线数据
            
        Returns:
            处理后的K线数据
        """
        processed_data = data.copy()
        i = 1
        
        while i < len(processed_data):
            prev = processed_data.iloc[i-1]
            curr = processed_data.iloc[i]
            
            # 检查是否有包含关系
            if (curr['high'] <= prev['high'] and curr['low'] >= prev['low']) or \
               (curr['high'] >= prev['high'] and curr['low'] <= prev['low']):
                # 处理包含关系
                if prev['high'] > prev['low']:  # 上升趋势
                    new_high = max(prev['high'], curr['high'])
                    new_low = max(prev['low'], curr['low'])
                else:  # 下降趋势
                    new_high = min(prev['high'], curr['high'])
                    new_low = min(prev['low'], curr['low'])
                
                # 替换前一根K线
                processed_data.iloc[i-1, processed_data.columns.get_loc('high')] = new_high
                processed_data.iloc[i-1, processed_data.columns.get_loc('low')] = new_low
                processed_data.iloc[i-1, processed_data.columns.get_loc('open')] = prev['open']
                processed_data.iloc[i-1, processed_data.columns.get_loc('close')] = curr['close']
                processed_data.iloc[i-1, processed_data.columns.get_loc('volume')] = prev['volume'] + curr['volume']
                
                # 删除当前K线
                processed_data = processed_data.drop(processed_data.index[i])
            else:
                i += 1
        
        return processed_data
    
    def dynamic_calibration(self, volatility_level: str, signal_history: List) -> str:
        """
        动态校准机制
        
        Args:
            volatility_level: 当前波动等级
            signal_history: 信号历史记录
            
        Returns:
            校准后的波动等级
        """
        # 获取最近3个月的信号
        three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
        recent_signals = [s for s in signal_history if s['date'] >= three_months_ago]
        
        # 规则1：连续3个月无有效信号，下调一个等级
        if len(recent_signals) == 0:
            if volatility_level == 'high':
                return 'medium'
            elif volatility_level == 'medium':
                return 'low'
        
        # 规则2：每月有效信号>2个，上调一个等级
        monthly_signals = len(recent_signals) / 3  # 估算每月信号数
        if monthly_signals > 2:
            if volatility_level == 'low':
                return 'medium'
            elif volatility_level == 'medium':
                return 'high'
        
        return volatility_level
    
    def format_output(self, etf_code: str, volatility_level: str, weekly_confidence: str,
                     signal_type: str, buy_point_type: str, min_position_ratio: float,
                     actual_position: float, add_condition: str, total_position_limit: float,
                     profit_condition: str, data_source: Dict, dynamic_params: Dict) -> str:
        """
        格式化实盘输出
        
        Args:
            etf_code: ETF代码
            volatility_level: 波动等级
            weekly_confidence: 周线置信度
            signal_type: 信号类型
            buy_point_type: 日线买点类型
            min_position_ratio: 分钟子仓位比例
            actual_position: 实际仓位
            add_condition: 加仓条件
            total_position_limit: 累计总仓位上限
            profit_condition: 止盈触发条件
            data_source: 数据源状态
            dynamic_params: 动态参数
            
        Returns:
            格式化输出字符串
        """
        # 波动等级中文映射
        vol_level_map = {
            'low': '低波动',
            'medium': '中波动',
            'high': '高波动'
        }
        
        # 数据源状态中文
        data_source_str = '满足' if all(data_source.values()) else '不足'
        
        # 构建动态参数字符串
        params_str = json.dumps({
            '中枢区间': f"{dynamic_params.get('central_low', 'N/A')}-{dynamic_params.get('central_high', 'N/A')}",
            '破位阈值': dynamic_params.get('break_threshold', 'N/A'),
            '反抽阈值': dynamic_params.get('rebound_threshold', 'N/A'),
            '量能阈值': f"{dynamic_params.get('volume_threshold_ratio', 'N/A') * 100}%"
        }, ensure_ascii=False)
        
        return f"「ETF代码：{etf_code}｜波动等级：{vol_level_map.get(volatility_level, volatility_level)}｜周线置信度：{weekly_confidence}｜信号类型：{signal_type}｜日线买点类型（核心/辅助/兜底）：{buy_point_type}｜分钟子仓位比例：{min_position_ratio}%｜实际仓位：{actual_position}%｜加仓条件：{add_condition}｜累计总仓位上限：{total_position_limit}%｜止盈触发条件：{profit_condition}｜数据源：{data_source_str}｜动态参数：{params_str}」"
    
    def format_invalid_output(self, etf_code: str, invalid_reason: str) -> str:
        """
        格式化无效信号输出
        
        Args:
            etf_code: ETF代码
            invalid_reason: 无效原因
            
        Returns:
            格式化输出字符串
        """
        return f"「ETF代码：{etf_code}｜无效原因：{invalid_reason}」"
    
    def load_data(self, etf_code: str, data_type: str = 'daily') -> Optional[pd.DataFrame]:
        """
        加载ETF数据
        
        Args:
            etf_code: ETF代码
            data_type: 数据类型 ('daily' 或 'weekly')
            
        Returns:
            数据DataFrame或None
        """
        # 设置数据路径
        if data_type == 'daily':
            file_path = f"data/daily/{etf_code}_daily.csv"
        else:
            file_path = f"data/weekly/{etf_code}_weekly.csv"
        
        # 尝试加载数据
        try:
            data = pd.read_csv(file_path)
            # 确保日期列格式正确
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            elif 'Date' in data.columns:
                data['date'] = pd.to_datetime(data['Date'])
                data = data.drop('Date', axis=1)
            
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    # 尝试不同的列名格式
                    for alt_col in [col.upper(), col.capitalize()]:
                        if alt_col in data.columns:
                            data[col] = data[alt_col]
                            break
            
            # 检查是否所有必要列都存在
            for col in required_columns + ['date']:
                if col not in data.columns:
                    print(f"错误：{file_path} 缺少必要列 {col}")
                    return None
            
            return data
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return None
    
    def main(self, etf_code: str):
        """
        主函数
        
        Args:
            etf_code: ETF代码
        """
        print(f"开始处理 {etf_code} 的通用缠论策略分析...")
        
        # 保存当前ETF代码
        self.current_etf_code = etf_code
        
        # 加载数据
        daily_data = self.load_data(etf_code, 'daily')
        weekly_data = self.load_data(etf_code, 'weekly')
        
        if daily_data is None or weekly_data is None:
            print(f"错误：无法加载 {etf_code} 的数据")
            print(self.format_invalid_output(etf_code, "数据源不足"))
            return
        
        # 数据源校验
        data_source_status = self.validate_data_source(daily_data, weekly_data)
        
        if not all(data_source_status.values()):
            invalid_reason = "数据源不足"
            print(self.format_invalid_output(etf_code, invalid_reason))
            return
        
        # 计算波动率并分类
        volatility = self.calculate_volatility(daily_data)
        self.volatility_level = self.classify_volatility(volatility)
        
        print(f"{etf_code} 波动率: {volatility:.2f}% | 波动等级: {self.volatility_level}")
        
        # 获取动态参数（默认中置信度）
        self.dynamic_params = self.get_dynamic_params(self.volatility_level, 'medium')
        
        print(f"动态参数: {self.dynamic_params}")
        print(f"数据源状态: {data_source_status}")
        
        # TODO: 后续模块将在这里被调用
        print(f"{etf_code} 初始分析完成，等待后续模块执行...")

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) > 1:
        etf_code = sys.argv[1]
    else:
        etf_code = "512690"  # 默认分析酒ETF
    
    strategy = UniversalChanlunStrategy()
    strategy.main(etf_code)