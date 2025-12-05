#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用缠论ETF分析器
通配所有宽基/行业ETF，自动识别数据异常、修正计算错误，精准判定中枢与缠论信号
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar


class UniversalChanlunETFAnalyzer:
    """通用缠论ETF分析器"""
    
    def __init__(self, symbol, year):
        """
        初始化分析器
        
        Args:
            symbol (str): ETF代码
            year (int): 分析年份
        """
        self.symbol = symbol
        self.year = year
        
        # 分析状态标志
        self.has_critical_error = False
        self.error_message = ""
        
        # 数据存储
        self.raw_data = None  # 原始K线数据
        self.standard_k_lines = None  # 标准K线数据
        self.year_low = None
        self.year_high = None
        
        # 数据完整性
        self.missing_trading_days = 0
        self.data_integrity_rating = ""
        self.integrity_impact = ""
        
        # 波动率相关
        self.volatility = None
        self.volatility_level = None  # 'low', 'medium', 'high'
        self.volatility_sample_size = None
        self.volatility_high = None
        self.volatility_low = None
        self.volatility_avg = None
        self.volatility_calculation_process = []
        
        # 波动等级参数映射
        self.volatility_params = {
            'low': {
                'segment_amplitude_threshold': 5.0,
                'central_break_threshold': 0.995,
                'central_rebound_threshold': 1.005
            },
            'medium': {
                'segment_amplitude_threshold': 8.0,
                'central_break_threshold': 0.99,
                'central_rebound_threshold': 1.01
            },
            'high': {
                'segment_amplitude_threshold': 10.0,
                'central_break_threshold': 0.985,
                'central_rebound_threshold': 1.015
            }
        }
        
        # 分析结果
        self.segments = []  # 盘整段列表
        self.valid_segments = []  # 有效盘整段列表
        self.valid_centrals = []  # 有效中枢列表
        self.buy_signals = []  # 买入信号列表
        
        # 异常处理记录
        self.abnormal_processing_logs = []
        
    def load_data(self):
        """
        加载基础数据，执行严格的数据完整性检查
        
        Returns:
            bool: 是否加载成功
        """
        try:
            # 尝试从多个路径加载数据
            data_paths = [
                f'./data/daily/{self.symbol}_daily.csv',
                f'./data/{self.symbol}_daily_data.csv',
                f'{self.symbol}_daily_data.csv',
                f'{self.symbol}_2025.csv'
            ]
            
            data_found = False
            for path in data_paths:
                try:
                    if os.path.exists(path):
                        print(f"尝试加载数据文件: {os.path.abspath(path)}")
                        self.raw_data = pd.read_csv(path)
                        data_found = True
                        print(f"成功加载数据文件: {os.path.abspath(path)}")
                        break
                except pd.errors.EmptyDataError:
                    print(f"警告: 文件{path}为空")
                except pd.errors.ParserError:
                    print(f"警告: 文件{path}格式错误，无法解析")
            
            if not data_found:
                self.has_critical_error = True
                searched_paths = ', '.join([os.path.abspath(p) for p in data_paths])
                self.error_message = f"未找到{self.symbol}的日线数据文件，已尝试路径: {searched_paths}"
                print(f"错误: {self.error_message}")
                return False
            
            # 数据字段检查
            required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
            for col in required_columns:
                if col not in self.raw_data.columns:
                    self.has_critical_error = True
                    self.error_message = f"数据文件缺少核心字段: {col}，当前文件包含的字段: {list(self.raw_data.columns)}"
                    print(f"错误: {self.error_message}")
                    return False
                
                # 检查缺失比例
                missing_count = self.raw_data[col].isnull().sum()
                total_count = len(self.raw_data)
                missing_ratio = missing_count / total_count if total_count > 0 else 0
                
                if missing_ratio > 0.2:
                    self.has_critical_error = True
                    self.error_message = f"核心字段{col}缺失比例超过20%({missing_count}/{total_count}={missing_ratio:.1%})，判定为核心数据缺失"
                    print(f"错误: {self.error_message}")
                    return False
                elif missing_count > 0:
                    print(f"警告: 字段{col}存在缺失值({missing_count}个)")
            
            # 处理日期格式
            try:
                self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
                print(f"日期格式转换完成")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"日期格式转换失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 筛选指定年份的数据
            try:
                self.raw_data = self.raw_data[
                    (self.raw_data['date'].dt.year == self.year)
                ].copy()
                print(f"筛选{self.year}年数据完成")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"年份筛选失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 基本数据清理：移除异常值
            for col in ['open', 'close', 'high', 'low', 'volume']:
                try:
                    # 移除明显异常值（如价格为0或负数）
                    before_count = len(self.raw_data)
                    self.raw_data = self.raw_data[self.raw_data[col] > 0].copy()
                    after_count = len(self.raw_data)
                    if before_count > after_count:
                        removed_count = before_count - after_count
                        print(f"警告: 字段{col}移除了{removed_count}条异常记录（值≤0）")
                except Exception as e:
                    self.has_critical_error = True
                    self.error_message = f"字段{col}异常值处理失败: {str(e)}"
                    print(f"错误: {self.error_message}")
                    return False
            
            # 检查筛选后的数据是否为空
            if len(self.raw_data) == 0:
                self.has_critical_error = True
                self.error_message = f"筛选后无有效数据，可能年份不匹配或数据质量问题"
                print(f"错误: {self.error_message}")
                return False
            
            # 计算缺失交易日
            try:
                self._calculate_missing_trading_days()
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"计算缺失交易日失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 检查数据完整性
            if not self._check_data_integrity():
                return False
            
            # 计算年度价格区间
            try:
                self.year_low = self.raw_data['low'].min()
                self.year_high = self.raw_data['high'].max()
                print(f"计算年度价格区间完成: 最低价={self.year_low}, 最高价={self.year_high}")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"计算价格区间失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            print(f"数据加载完成: 共{len(self.raw_data)}条记录")
            return True
            
        except Exception as e:
            import traceback
            self.has_critical_error = True
            self.error_message = f"标准K线生成失败: {str(e)}"
            print(f"错误: {self.error_message}")
            print(f"详细错误堆栈:\n{traceback.format_exc()}")
            return False
    
    def _calculate_missing_trading_days(self):
        """
        计算缺失的交易日数量
        考虑全年交易日数量和实际数据天数的差异
        """
        if self.raw_data is None or self.raw_data.empty:
            self.missing_trading_days = float('inf')
            return
        
        # 获取年度第一个和最后一个交易日
        first_date = self.raw_data['date'].min()
        last_date = self.raw_data['date'].max()
        
        # 1. 计算全年理论交易日数量（简化计算：全年工作日减去约15天假期）
        year_start = datetime(self.year, 1, 1)
        year_end = datetime(self.year, 12, 31)
        
        # 计算全年工作日数量
        total_weekdays = 0
        current_date = year_start
        while current_date <= year_end:
            if current_date.weekday() < 5:  # 周一到周五
                total_weekdays += 1
            current_date += timedelta(days=1)
        
        # 减去约15天假期
        expected_annual_trading_days = total_weekdays - 15
        
        # 2. 如果数据只覆盖部分年份，计算实际覆盖期间的理论交易日
        # 计算实际覆盖期间的工作日数量
        actual_covered_weekdays = 0
        current_date = first_date
        while current_date <= last_date:
            if current_date.weekday() < 5:  # 周一到周五
                actual_covered_weekdays += 1
            current_date += timedelta(days=1)
        
        # 减去约5天假期（如果覆盖期间较短）
        expected_trading_days = max(0, actual_covered_weekdays - 5)
        
        # 3. 计算实际缺失天数
        actual_days = len(self.raw_data)
        self.missing_trading_days = max(0, expected_trading_days - actual_days)
        
        print(f"数据缺失分析: 实际{actual_days}天, 期望{expected_trading_days}天, 缺失{self.missing_trading_days}天")
    
    def _check_data_integrity(self):
        """
        检查数据完整性，严格按照分级处理规则执行
        轻微缺失（缺失交易日≤10天）：正常分析
        中度缺失（11≤缺失≤20天）：正常分析但需标注
        严重缺失（缺失＞20天）：直接终止分析
        """
        try:
            # 检查missing_trading_days是否有效
            if not isinstance(self.missing_trading_days, (int, float)) or self.missing_trading_days < 0:
                self.has_critical_error = True
                self.error_message = f"无效的缺失交易日数值: {self.missing_trading_days}"
                print(f"数据完整性检查失败: {self.error_message}")
                return False
            
            # 严格按照规范进行分级处理
            print(f"开始数据完整性检查，缺失交易日数: {self.missing_trading_days}")
            
            if self.missing_trading_days <= 10:
                self.data_integrity_rating = "数据基本完整"
                self.integrity_impact = "可信度高"
                print(f"数据完整性检查: {self.data_integrity_rating}（{self.integrity_impact}）")
            elif self.missing_trading_days <= 20:
                self.data_integrity_rating = "数据部分缺失"
                self.integrity_impact = "关键结论需谨慎"
                # 确保abnormal_processing_logs已初始化
                if not hasattr(self, 'abnormal_processing_logs'):
                    self.abnormal_processing_logs = []
                self.abnormal_processing_logs.append(f"数据部分缺失（缺失{self.missing_trading_days}天），关键结论需谨慎")
                print(f"数据完整性检查: {self.data_integrity_rating}（{self.integrity_impact}）")
            else:
                self.has_critical_error = True
                self.error_message = f"数据严重缺失（缺失{self.missing_trading_days}天），无法进行有效缠论分析"
                print(f"数据完整性检查失败: {self.error_message}")
                return False
            
            print("数据完整性检查完成，继续分析")
            return True
        except Exception as e:
            # 捕获所有异常并记录详细错误信息
            import traceback
            self.has_critical_error = True
            self.error_message = f"数据完整性检查过程中发生异常: {str(e)}"
            print(f"数据完整性检查失败: {self.error_message}")
            print(f"详细错误堆栈: {traceback.format_exc()}")
            return False
    
    def generate_standard_k_lines(self):
        """
        生成标准K线（合并包含关系的K线）
        严格执行K线合并算法，确保标准K线数量满足分析要求
        
        Returns:
            bool: 是否生成成功
        """
        try:
            if self.raw_data is None or self.raw_data.empty:
                self.has_critical_error = True
                self.error_message = "原始数据为空，无法生成标准K线"
                print(f"错误: {self.error_message}")
                return False
            
            # 复制数据并排序
            try:
                df = self.raw_data.sort_values('date').copy()
                print(f"数据排序完成，共{len(df)}条记录")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"数据排序失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 合并包含K线（严格执行包含关系检查）
            try:
                standard_klines = []
                current_kline = None
                merge_count = 0
                
                for idx, row in df.iterrows():
                    if current_kline is None:
                        current_kline = row.to_dict()
                    else:
                        try:
                            # 检查包含关系：后K线高点≤前K线高点且低点≥前K线低点
                            if row['high'] <= current_kline['high'] and row['low'] >= current_kline['low']:
                                # 合并K线，保留较大的成交量和最新的收盘价
                                current_kline['high'] = max(current_kline['high'], row['high'])
                                current_kline['low'] = min(current_kline['low'], row['low'])
                                current_kline['volume'] = current_kline['volume'] + row['volume']
                                current_kline['close'] = row['close']
                                merge_count += 1
                            else:
                                # 不包含，将当前K线加入标准K线列表
                                standard_klines.append(current_kline)
                                current_kline = row.to_dict()
                        except Exception as e:
                            print(f"警告: 处理K线{idx+1}时发生错误: {str(e)}")
                            # 跳过错误的K线，继续处理
                            continue
                
                # 添加最后一根K线
                if current_kline is not None:
                    standard_klines.append(current_kline)
                
                print(f"K线合并完成: 合并了{merge_count}根K线")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"K线合并算法执行失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 转换为DataFrame并确保日期格式正确
            try:
                self.standard_k_lines = pd.DataFrame(standard_klines)
                print(f"标准K线DataFrame创建完成，共{len(self.standard_k_lines)}根")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"标准K线DataFrame创建失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            print(f"标准K线生成: 原始{len(df)}根 → 合并{merge_count}根 → 标准{len(self.standard_k_lines)}根")
            
            # 严格检查标准K线数量
            if len(self.standard_k_lines) < 50:
                self.has_critical_error = True
                self.error_message = f"标准K线数量不足（{len(self.standard_k_lines)}根），无法支撑缠论分析"
                print(f"标准K线数量检查失败: {self.error_message}")
                return False
            elif len(self.standard_k_lines) < 100:
                print(f"警告: 标准K线数量偏少（{len(self.standard_k_lines)}根），分析结果可能不够准确")
            
            # 执行标准K线的异常检查
            price_error_count = 0
            volume_error_count = 0
            
            for idx, kline in self.standard_k_lines.iterrows():
                try:
                    # 检查价格合理性
                    if kline['high'] <= kline['low'] or kline['close'] <= 0 or kline['open'] <= 0:
                        price_error_count += 1
                        print(f"警告: 标准K线{idx+1}存在价格异常: open={kline['open']}, close={kline['close']}, high={kline['high']}, low={kline['low']}")
                    # 检查成交量合理性
                    if kline['volume'] <= 0:
                        volume_error_count += 1
                        print(f"警告: 标准K线{idx+1}成交量异常: volume={kline['volume']}")
                except KeyError as e:
                    print(f"警告: 标准K线{idx+1}缺少必要字段: {str(e)}")
                except Exception as e:
                    print(f"警告: 检查K线{idx+1}时发生错误: {str(e)}")
            
            if price_error_count > 0:
                print(f"警告汇总: 发现{price_error_count}根K线存在价格异常")
            if volume_error_count > 0:
                print(f"警告汇总: 发现{volume_error_count}根K线存在成交量异常")
            
            return True
            
        except Exception as e:
            self.has_critical_error = True
            self.error_message = f"标准K线生成失败: {str(e)}"
            return False
    
    def calculate_volatility(self):
        """
        计算波动率，严格按照规范公式执行
        公式：近60日波动率 = (近60日最高价 - 近60日最低价) / 近60日均价 × 100%
        异常自动修正：首次异常→近30日；仍异常→终止
        
        Returns:
            bool: 是否计算成功
        """
        try:
            if self.standard_k_lines is None or self.standard_k_lines.empty:
                self.has_critical_error = True
                self.error_message = "标准K线数据为空，无法计算波动率"
                print(f"错误: {self.error_message}")
                return False
            
            # 按日期排序
            try:
                klines_sorted = self.standard_k_lines.sort_values('date').copy()
                total_klines = len(klines_sorted)
                print(f"标准K线数据排序完成，共{total_klines}根K线")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"K线数据排序失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 重置计算过程记录
            try:
                self.volatility_calculation_process = []
            except Exception as e:
                print(f"警告: 重置计算过程记录失败: {str(e)}")
            
            # 严格按照规范计算：首先尝试近60日数据
            print("开始波动率计算...")
            
            # 计算步骤1：使用近60日数据
            try:
                sample_size = min(60, total_klines)
                recent_klines = klines_sorted.tail(sample_size).copy()
                print(f"选择最近{sample_size}根K线作为初始计算样本")
                
                # 样本要求：样本K线≥30根，否则用"全部标准K线"计算
                if sample_size < 30:
                    self.volatility_calculation_process.append(f"样本K线数量不足30根，使用全部{total_klines}根标准K线")
                    print(f"样本量不足30根，改用全部{total_klines}根标准K线")
                    recent_klines = klines_sorted.copy()
                    sample_size = total_klines
                
                # 计算关键价格指标
                self.volatility_sample_size = sample_size
                self.volatility_high = recent_klines['high'].max()
                self.volatility_low = recent_klines['low'].min()
                self.volatility_avg = recent_klines['close'].mean()
                
                print(f"初始价格指标计算完成: 最高价={self.volatility_high:.3f}, 最低价={self.volatility_low:.3f}, 均价={self.volatility_avg:.3f}")
                
            except KeyError as e:
                self.has_critical_error = True
                self.error_message = f"价格数据字段缺失: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"价格指标计算失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 严格检查计算条件：近60日均价≠0，最高价＞最低价
            has_exception = False
            if self.volatility_avg == 0:
                has_exception = True
                error_msg = "首次计算异常（均价为0），尝试用近30日波动率重新计算"
                self.volatility_calculation_process.append(error_msg)
                print(f"警告: {error_msg}")
            elif self.volatility_high <= self.volatility_low:
                has_exception = True
                error_msg = "首次计算异常（最高价≤最低价），尝试用近30日波动率重新计算"
                self.volatility_calculation_process.append(error_msg)
                print(f"警告: {error_msg}")
            
            if has_exception:
                # 异常自动修正：首次计算异常→用"近30日波动率"重新计算
                self.abnormal_processing_logs.append("波动率首次计算异常，已触发自动修正机制")
                print("触发异常自动修正机制，尝试使用近30日数据")
                
                # 计算步骤2：使用近30日数据
                try:
                    recent_30_klines = klines_sorted.tail(min(30, total_klines))
                    self.volatility_sample_size = len(recent_30_klines)
                    self.volatility_high = recent_30_klines['high'].max()
                    self.volatility_low = recent_30_klines['low'].min()
                    self.volatility_avg = recent_30_klines['close'].mean()
                    print(f"使用近30日数据重新计算: 样本量={self.volatility_sample_size}, 最高价={self.volatility_high:.3f}, 最低价={self.volatility_low:.3f}, 均价={self.volatility_avg:.3f}")
                    
                    # 再次检查条件
                    if self.volatility_avg == 0 or self.volatility_high <= self.volatility_low:
                        self.has_critical_error = True
                        self.error_message = "波动率数据异常，无法计算有效波动率"
                        print(f"波动率计算失败: {self.error_message}")
                        return False
                except Exception as e:
                    self.has_critical_error = True
                    self.error_message = f"异常修正计算失败: {str(e)}"
                    print(f"错误: {self.error_message}")
                    return False
            
            # 计算波动率
            try:
                self.volatility = (self.volatility_high - self.volatility_low) / self.volatility_avg * 100
                print(f"初步波动率计算完成: {self.volatility:.2f}%")
            except ZeroDivisionError:
                self.has_critical_error = True
                self.error_message = "波动率计算时发生除零错误（均价为0）"
                print(f"错误: {self.error_message}")
                return False
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"波动率计算失败: {str(e)}"
                print(f"错误: {self.error_message}")
                return False
            
            # 严格执行异常阈值检查：波动率≤0% 或 ＞50% → 判定"波动率计算异常"
            if self.volatility <= 0 or self.volatility > 50:
                # 首次计算异常，尝试用近30日数据（如果之前没用过）
                if sample_size > 30:
                    error_msg = f"波动率首次计算异常（{self.volatility:.2f}%），已用近30日波动率修正"
                    self.volatility_calculation_process.append(error_msg)
                    print(f"警告: {error_msg}")
                    
                    try:
                        recent_30_klines = klines_sorted.tail(min(30, total_klines))
                        self.volatility_sample_size = len(recent_30_klines)
                        self.volatility_high = recent_30_klines['high'].max()
                        self.volatility_low = recent_30_klines['low'].min()
                        self.volatility_avg = recent_30_klines['close'].mean()
                        
                        # 确保满足计算条件
                        if self.volatility_avg > 0 and self.volatility_high > self.volatility_low:
                            self.volatility = (self.volatility_high - self.volatility_low) / self.volatility_avg * 100
                            print(f"使用近30日数据修正后波动率: {self.volatility:.2f}%")
                    except Exception as e:
                        print(f"警告: 波动率修正计算失败: {str(e)}")
                
                # 再次检查异常阈值：仍异常则终止分析
                if self.volatility <= 0 or self.volatility > 50:
                    self.has_critical_error = True
                    self.error_message = f"波动率数据异常（计算结果{self.volatility:.2f}%），无法判定波动等级"
                    print(f"波动率计算失败: {self.error_message}")
                    # 记录异常
                    if not hasattr(self, 'data_exception_records'):
                        self.data_exception_records = []
                    self.data_exception_records.append(self.error_message)
                    return False
            
            # 完整记录计算过程，确保可追溯性
            try:
                self.volatility_calculation_process.append(f"样本K线数量: {self.volatility_sample_size}")
                self.volatility_calculation_process.append(f"最高价: {self.volatility_high:.3f}")
                self.volatility_calculation_process.append(f"最低价: {self.volatility_low:.3f}")
                self.volatility_calculation_process.append(f"均价: {self.volatility_avg:.3f}")
                self.volatility_calculation_process.append(f"波动率计算结果: {self.volatility:.2f}%")
                print("波动率计算过程记录完成")
            except Exception as e:
                print(f"警告: 记录计算过程失败: {str(e)}")
            
            print(f"波动率计算完成: {self.volatility:.2f}%")
            return True
            
        except Exception as e:
            import traceback
            self.has_critical_error = True
            self.error_message = f"波动率计算失败: {str(e)}"
            print(f"错误: {self.error_message}")
            print(f"详细错误堆栈:\n{traceback.format_exc()}")
            return False
    
    def determine_volatility_level(self):
        """
        确定波动等级，严格按照规范的绑定规则执行
        - 低波动ETF：波动率≤10.0% → 盘整段达标阈值≥5%，破位阈值×0.995，反抽阈值×1.005
        - 中波动ETF：10.1%≤波动率≤18.0% → 盘整段达标阈值≥8%，破位阈值×0.99，反抽阈值×1.01
        - 高波动ETF：波动率＞18.0% → 盘整段达标阈值≥10%，破位阈值×0.985，反抽阈值×1.015
        
        Returns:
            bool: 是否确定成功
        """
        try:
            print("开始执行波动等级判定方法")
            
            # 检查volatility_calculation_process是否初始化
            if not hasattr(self, 'volatility_calculation_process'):
                self.volatility_calculation_process = []
                print("初始化波动率计算过程记录列表")
            
            # 检查volatility_params是否初始化
            if not hasattr(self, 'volatility_params'):
                self.volatility_params = {}
                print("初始化波动率参数字典")
            
            # 如果波动率未计算，先计算波动率
            try:
                if self.volatility is None:
                    print("波动率未计算，开始计算波动率")
                    if not self.calculate_volatility():
                        print("波动率计算失败，终止波动等级判定")
                        return False
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"调用calculate_volatility方法时发生异常: {str(e)}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            
            print(f"开始波动等级判定，波动率为{self.volatility:.2f}%")
            
            # 确保波动率有效
            try:
                if self.volatility is None:
                    raise ValueError("波动率为None")
                if not isinstance(self.volatility, (int, float)):
                    raise TypeError(f"波动率类型无效: {type(self.volatility).__name__}")
                if not (0 < self.volatility <= 50):
                    raise ValueError(f"波动率值超出有效范围: {self.volatility}%")
            except (ValueError, TypeError) as e:
                self.has_critical_error = True
                self.error_message = f"无效的波动率值: {str(e)}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            
            # 严格执行波动等级与波动率绑定规则（无模糊空间）
            try:
                if self.volatility <= 10.0:
                    self.volatility_level = 'low'
                    self.volatility_bound_reason = f"波动率{self.volatility:.2f}% ≤ 10.0%"
                    # 低波动ETF：波动率≤10.0% → 盘整段达标阈值≥5%，破位阈值×0.995，反抽阈值×1.005
                    self.volatility_params['low'] = {
                        'segment_amplitude_threshold': 5.0,
                        'central_break_threshold': 0.995,
                        'central_rebound_threshold': 1.005
                    }
                elif 10.1 <= self.volatility <= 18.0:
                    self.volatility_level = 'medium'
                    self.volatility_bound_reason = f"10.1% ≤ 波动率{self.volatility:.2f}% ≤ 18.0%"
                    # 中波动ETF：10.1%≤波动率≤18.0% → 盘整段达标阈值≥8%，破位阈值×0.99，反抽阈值×1.01
                    self.volatility_params['medium'] = {
                        'segment_amplitude_threshold': 8.0,
                        'central_break_threshold': 0.99,
                        'central_rebound_threshold': 1.01
                    }
                else:  # self.volatility > 18.0
                    self.volatility_level = 'high'
                    self.volatility_bound_reason = f"波动率{self.volatility:.2f}% > 18.0%"
                    # 高波动ETF：波动率＞18.0% → 盘整段达标阈值≥10%，破位阈值×0.985，反抽阈值×1.015
                    self.volatility_params['high'] = {
                        'segment_amplitude_threshold': 10.0,
                        'central_break_threshold': 0.985,
                        'central_rebound_threshold': 1.015
                    }
                print(f"波动等级判定完成，确定为{self.volatility_level}波动等级")
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"波动等级判定过程中发生异常: {str(e)}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            
            # 完整记录判定依据，确保可追溯性
            try:
                level_name_mapping = {
                    'low': '低波动ETF',
                    'medium': '中波动ETF',
                    'high': '高波动ETF'
                }
                self.volatility_level_name = level_name_mapping[self.volatility_level]
                print(f"波动等级名称映射完成: {self.volatility_level_name}")
            except KeyError:
                self.has_critical_error = True
                self.error_message = f"未知的波动等级: {self.volatility_level}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"波动等级名称映射过程中发生异常: {str(e)}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            
            # 验证参数映射的完整性
            try:
                if self.volatility_level not in self.volatility_params:
                    raise KeyError(f"波动等级 {self.volatility_level} 的参数不存在")
                
                current_params = self.volatility_params[self.volatility_level]
                required_keys = ['segment_amplitude_threshold', 'central_break_threshold', 'central_rebound_threshold']
                
                # 检查必需的键是否存在
                missing_keys = [key for key in required_keys if key not in current_params]
                if missing_keys:
                    raise ValueError(f"波动等级参数映射缺失键: {missing_keys}")
                
                # 检查参数值是否有效
                for key, value in current_params.items():
                    if not isinstance(value, (int, float)):
                        raise TypeError(f"参数 {key} 的值类型无效: {type(value).__name__}")
                        
                print("波动等级参数验证通过")
            except (KeyError, ValueError, TypeError) as e:
                self.has_critical_error = True
                self.error_message = f"波动等级参数映射验证失败: {str(e)}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            except Exception as e:
                self.has_critical_error = True
                self.error_message = f"波动等级参数验证过程中发生异常: {str(e)}"
                print(f"波动等级判定失败: {self.error_message}")
                return False
            
            # 添加到计算过程记录，确保透明度
            try:
                self.volatility_calculation_process.append(f"波动等级: {self.volatility_level_name} ({self.volatility_bound_reason})")
                self.volatility_calculation_process.append(f"对应参数 - 盘整段阈值: ≥{current_params['segment_amplitude_threshold']}%, 破位阈值: ×{current_params['central_break_threshold']}, 反抽阈值: ×{current_params['central_rebound_threshold']}")
                print("波动等级判定信息已添加到计算过程记录")
            except Exception as e:
                print(f"警告：添加波动等级判定信息到计算过程记录时发生异常: {str(e)}")
                # 这不是关键错误，可以继续执行
            
            # 记录到计算日志
            if hasattr(self, 'calculation_logs'):
                self.calculation_logs.append(f"波动等级判定: {self.volatility_level_name}, 依据{self.volatility_bound_reason}")
            
            print(f"波动等级判定成功完成，最终确定为: {self.volatility_level_name}")
            return True
        except Exception as e:
            # 捕获所有未预期的异常
            import traceback
            self.has_critical_error = True
            self.error_message = f"波动等级判定过程中发生未预期的异常: {str(e)}"
            print(f"波动等级判定失败: {self.error_message}")
            print(f"详细错误堆栈: {traceback.format_exc()}")
            return False
            return False
    
    def calculate_90pct_trading_range(self, prices):
        """
        计算90%成交区间，严格按照规范步骤执行
        步骤：① 提取标的年度所有标准K线的收盘价 → ② 剔除首尾各5%极值（若收盘价数量＜20根，剔除首尾各1根） → ③ 剩余收盘价的最小值=下沿，最大值=上沿
        异常判定：下沿=0 或 上沿=0 或 上沿≤下沿 → 判定"成交区间数据异常"
        
        Args:
            prices (list): 价格列表
            
        Returns:
            tuple: (下沿, 上沿, 是否有效)
        """
        try:
            # 确保有计算日志记录
            if not hasattr(self, 'calculation_logs'):
                self.calculation_logs = []
            
            # 步骤1：严格验证输入数据
            if not prices or not isinstance(prices, list) or len(prices) == 0:
                error_msg = "90%成交区间计算错误: 价格列表为空或格式错误"
                self.calculation_logs.append(error_msg)
                print(error_msg)
                return 0, 0, False
            
            # 过滤无效价格
            valid_prices = [p for p in prices if p is not None and p > 0]
            if len(valid_prices) < 5:
                error_msg = f"90%成交区间计算错误: 有效价格数量不足（仅{len(valid_prices)}个）"
                self.calculation_logs.append(error_msg)
                print(error_msg)
                return 0, 0, False
            
            # 步骤2：严格执行剔除首尾极值
            sorted_prices = sorted(valid_prices)
            n = len(sorted_prices)
            
            if n < 20:
                # 严格按规范：若收盘价数量＜20根，剔除首尾各1根
                if n < 3:  # 至少需要3根才能剔除首尾各1根
                    # 数据量太少，无法有效计算
                    error_msg = f"90%成交区间计算警告: 价格数量{n}过少，无法按规范剔除首尾极值"
                    self.calculation_logs.append(error_msg)
                    print(error_msg)
                    # 记录异常
                    if hasattr(self, 'data_exception_records'):
                        self.data_exception_records.append(error_msg)
                    return 0, 0, False
                else:
                    filtered_prices = sorted_prices[1:-1]
                    log_msg = f"90%成交区间计算: 价格数量{n}＜20，剔除首尾各1根，剩余{len(filtered_prices)}根"
                    self.calculation_logs.append(log_msg)
                    print(log_msg)
            else:
                # 严格按规范：剔除首尾各5%极值
                trim_count = max(1, int(n * 0.05))
                # 确保至少保留数据
                if trim_count * 2 >= n:
                    trim_count = max(1, int(n * 0.03))
                filtered_prices = sorted_prices[trim_count:-trim_count]
                log_msg = f"90%成交区间计算: 价格数量{n}≥20，剔除首尾各{trim_count}根（{trim_count/n*100:.1f}%），剩余{len(filtered_prices)}根"
                self.calculation_logs.append(log_msg)
                print(log_msg)
            
            # 步骤3：计算区间
            if len(filtered_prices) < 2:
                error_msg = "90%成交区间计算错误: 剔除后价格数量不足"
                self.calculation_logs.append(error_msg)
                print(error_msg)
                return 0, 0, False
            
            # 严格执行：剩余收盘价的最小值=下沿，最大值=上沿
            lower = min(filtered_prices)
            upper = max(filtered_prices)
            
            # 计算振幅用于日志
            amplitude = (upper - lower) / lower * 100 if lower > 0 else 0
            
            # 严格执行异常判定：下沿=0 或 上沿=0 或 上沿≤下沿 → 判定"成交区间数据异常"
            if lower == 0:
                error_msg = f"90%成交区间数据异常: 下沿={lower}（为0）"
                self.calculation_logs.append(error_msg)
                print(error_msg)
                if hasattr(self, 'data_exception_records'):
                    self.data_exception_records.append(error_msg)
                return 0, 0, False
            
            if upper == 0:
                error_msg = f"90%成交区间数据异常: 上沿={upper}（为0）"
                self.calculation_logs.append(error_msg)
                print(error_msg)
                if hasattr(self, 'data_exception_records'):
                    self.data_exception_records.append(error_msg)
                return 0, 0, False
            
            if upper <= lower:
                error_msg = f"90%成交区间数据异常: 上沿({upper})≤下沿({lower})"
                self.calculation_logs.append(error_msg)
                print(error_msg)
                if hasattr(self, 'data_exception_records'):
                    self.data_exception_records.append(error_msg)
                return 0, 0, False
            
            # 记录成功计算结果
            log_msg = f"90%成交区间计算成功: 下沿={lower:.3f}，上沿={upper:.3f}，振幅={amplitude:.2f}%"
            self.calculation_logs.append(log_msg)
            print(log_msg)
            
            return lower, upper, True
            
        except Exception as e:
            error_msg = f"计算90%成交区间时发生错误: {str(e)}"
            self.calculation_logs.append(error_msg)
            print(error_msg)
            if hasattr(self, 'data_exception_records'):
                self.data_exception_records.append(error_msg)
            return 0, 0, False
    
    def divide_segments(self):
        """
        划分走势段，严格按照规范要求
        盘整段需同时满足：
        - 连续≥8根标准K线
        - 90%成交区间振幅≥对应波动等级阈值
        - 无明显涨跌趋势（上涨段：连续≥5根收盘价抬升+高点创新高；下跌段：连续≥5根收盘价走低+低点创新低，两者均不满足）
        不满足任一条件→判定为"杂波震荡"，不纳入有效盘整段清单
        
        Returns:
            bool: 是否划分成功
        """
        try:
            if self.standard_k_lines is None or self.standard_k_lines.empty:
                self.has_critical_error = True
                self.error_message = "标准K线数据为空，无法划分走势段"
                print(f"错误: {self.error_message}")
                # 记录关键错误到异常记录
                if hasattr(self, 'data_exception_records'):
                    self.data_exception_records.append(self.error_message)
                return False
            
            if self.volatility_level is None:
                print("未确定波动等级，开始计算...")
                if not self.determine_volatility_level():
                    print("波动等级计算失败，无法继续划分走势段")
                    return False
            
            # 获取波动等级对应的阈值
            if self.volatility_level not in self.volatility_params:
                self.has_critical_error = True
                self.error_message = f"未知的波动等级: {self.volatility_level}"
                print(f"错误: {self.error_message}")
                if hasattr(self, 'data_exception_records'):
                    self.data_exception_records.append(self.error_message)
                return False
            
            amplitude_threshold = self.volatility_params[self.volatility_level]['segment_amplitude_threshold']
            
            # 按日期排序
            klines_sorted = self.standard_k_lines.sort_values('date').copy().reset_index(drop=True)
            total_klines = len(klines_sorted)
            
            # 记录计算日志
            log_msg = f"开始划分走势段，标准K线总数: {total_klines}，盘整段阈值: ≥{amplitude_threshold}%，波动等级: {self.volatility_level}"
            print(log_msg)
            if hasattr(self, 'calculation_logs'):
                self.calculation_logs.append(log_msg)
            
            segments = []
            segment_id = 1
            valid_count = 0
            invalid_count = 0
            
            # 扫描可能的盘整段（使用非重叠窗口以避免重复计算）
            i = 0
            min_segment_length = 8  # 最小K线数量
            max_segment_length = 30  # 最大K线数量
            
            while i <= total_klines - min_segment_length:
                # 尝试不同长度的盘整段（8-30根K线）
                best_segment = None
                best_segment_length = 0
                best_segment_score = 0
                
                # 计算当前窗口可以尝试的最大长度
                max_try_length = min(max_segment_length, total_klines - i)
                
                for segment_length in range(min_segment_length, max_try_length + 1):
                    segment_data = klines_sorted.iloc[i:i+segment_length].copy()
                    
                    # 提取数据
                    close_prices = segment_data['close'].tolist()
                    high_prices = segment_data['high'].tolist()
                    low_prices = segment_data['low'].tolist()
                    
                    # 数据有效性预检查
                    if len(close_prices) != segment_length or len(high_prices) != segment_length or len(low_prices) != segment_length:
                        continue
                    
                    # 计算90%成交区间
                    lower_90pct, upper_90pct, is_valid_range = self.calculate_90pct_trading_range(close_prices)
                    
                    # 计算振幅
                    amplitude_90pct = 0
                    if is_valid_range and lower_90pct > 0:
                        amplitude_90pct = (upper_90pct - lower_90pct) / lower_90pct * 100
                    
                    # 检查是否有明显涨跌趋势
                    has_rising_trend = self._has_rising_trend(close_prices, high_prices)
                    has_falling_trend = self._has_falling_trend(close_prices, low_prices)
                    
                    # 严格检查是否满足盘整段条件
                    is_valid = (segment_length >= 8 and
                               is_valid_range and
                               lower_90pct > 0 and
                               upper_90pct > 0 and
                               amplitude_90pct >= amplitude_threshold and
                               not has_rising_trend and
                               not has_falling_trend)
                    
                    # 计算分数（用于选择最佳段）
                    score = 0
                    if is_valid:
                        # 振幅越大、K线数量越适中，分数越高
                        score = min(amplitude_90pct / amplitude_threshold, 2.0)  # 振幅因子，上限2倍
                        # K线数量因子，偏好15-25根
                        kline_factor = 1.0 - abs(segment_length - 20) / 15  # 20根为最优，逐渐递减
                        kline_factor = max(0.5, kline_factor)  # 最低0.5倍
                        score *= kline_factor
                    
                    # 更新最佳段
                    if score > best_segment_score:
                        best_segment_score = score
                        best_segment = segment_data
                        best_segment_length = segment_length
                
                # 如果找到有效盘整段
                if best_segment is not None:
                    close_prices = best_segment['close'].tolist()
                    high_prices = best_segment['high'].tolist()
                    low_prices = best_segment['low'].tolist()
                    
                    # 重新计算90%成交区间（确保准确性）
                    lower_90pct, upper_90pct, is_valid_range = self.calculate_90pct_trading_range(close_prices)
                    amplitude_90pct = (upper_90pct - lower_90pct) / lower_90pct * 100 if is_valid_range and lower_90pct > 0 else 0
                    
                    # 重新检查趋势
                    has_rising_trend = self._has_rising_trend(close_prices, high_prices)
                    has_falling_trend = self._has_falling_trend(close_prices, low_prices)
                    
                    # 严格的有效性检查
                    is_valid = (best_segment_length >= 8 and
                               is_valid_range and
                               lower_90pct > 0 and
                               upper_90pct > 0 and
                               amplitude_90pct >= amplitude_threshold and
                               not has_rising_trend and
                               not has_falling_trend)
                    
                    # 创建盘整段对象
                    segment = {
                        'segment_id': segment_id,
                        'start_date': best_segment.iloc[0]['date'],
                        'end_date': best_segment.iloc[-1]['date'],
                        'k_count': best_segment_length,
                        'close_prices': close_prices.copy(),
                        'amplitude_90pct': amplitude_90pct,
                        'lower_90pct': lower_90pct,
                        'upper_90pct': upper_90pct,
                        'has_rising_trend': has_rising_trend,
                        'has_falling_trend': has_falling_trend,
                        'is_valid': is_valid,
                        'validity_checks': {
                            'k_count_check': f"{best_segment_length}≥8",
                            'amplitude_check': f"{amplitude_90pct:.2f}%≥{amplitude_threshold}%",
                            'trend_check': "无明显趋势" if not (has_rising_trend or has_falling_trend) else "有明显趋势",
                            'range_abnormal': "否" if is_valid_range else "是",
                            'price_validity': "有效" if (lower_90pct > 0 and upper_90pct > 0) else "无效"
                        },
                        'invalid_reason': []
                    }
                    
                    # 详细记录无效原因
                    if not is_valid:
                        if best_segment_length < 8:
                            segment['invalid_reason'].append(f'K线数量不足8根（实际{best_segment_length}根）')
                        if not is_valid_range:
                            segment['invalid_reason'].append('90%成交区间数据异常')
                        if lower_90pct <= 0:
                            segment['invalid_reason'].append(f'成交区间下沿无效（{lower_90pct}）')
                        if upper_90pct <= 0:
                            segment['invalid_reason'].append(f'成交区间上沿无效（{upper_90pct}）')
                        if amplitude_90pct < amplitude_threshold:
                            segment['invalid_reason'].append(f'振幅不满足阈值要求（{amplitude_90pct:.2f}% < {amplitude_threshold}%）')
                        if has_rising_trend:
                            segment['invalid_reason'].append('存在上涨趋势（连续≥5根收盘价抬升+高点创新高）')
                        if has_falling_trend:
                            segment['invalid_reason'].append('存在下跌趋势（连续≥5根收盘价走低+低点创新低）')
                    
                    segments.append(segment)
                    
                    # 记录详细日志
                    if is_valid:
                        valid_count += 1
                        log_msg = f"  - 识别有效盘整段{segment_id}: {best_segment.iloc[0]['date'].strftime('%Y-%m-%d')}至{best_segment.iloc[-1]['date'].strftime('%Y-%m-%d')}, {best_segment_length}根K线, 振幅{amplitude_90pct:.2f}%"
                        print(log_msg)
                        if hasattr(self, 'calculation_logs'):
                            self.calculation_logs.append(log_msg)
                    else:
                        invalid_count += 1
                        log_msg = f"  - 识别无效盘整段{segment_id}: {best_segment.iloc[0]['date'].strftime('%Y-%m-%d')}至{best_segment.iloc[-1]['date'].strftime('%Y-%m-%d')}, 原因: {'; '.join(segment['invalid_reason'])}"
                        print(log_msg)
                        if hasattr(self, 'calculation_logs'):
                            self.calculation_logs.append(log_msg)
                    
                    segment_id += 1
                    
                    # 跳过已识别的盘整段
                    i += best_segment_length
                else:
                    # 如果未找到有效盘整段，移动一个K线继续搜索
                    i += 1
            
            self.segments = segments
            self.valid_segments = [seg for seg in segments if seg['is_valid']]
            
            # 记录划分结果
            log_msg = f"走势段划分完成: 共识别{valid_count}个有效盘整段，{invalid_count}个无效区间"
            print(log_msg)
            if hasattr(self, 'calculation_logs'):
                self.calculation_logs.append(log_msg)
            
            # 检查所有盘整段是否都异常
            all_invalid_ranges = False
            if segments:
                all_invalid_ranges = all(not self.calculate_90pct_trading_range(seg['close_prices'])[2] for seg in segments)
            
            if segments and all_invalid_ranges:
                self.has_critical_error = True
                self.error_message = "所有盘整段成交区间数据异常，无法生成中枢"
                log_msg = f"错误: {self.error_message}"
                print(log_msg)
                if hasattr(self, 'calculation_logs'):
                    self.calculation_logs.append(log_msg)
                if hasattr(self, 'data_exception_records'):
                    self.data_exception_records.append(self.error_message)
                return False
            
            return True
            
        except Exception as e:
            self.has_critical_error = True
            self.error_message = f"走势段划分失败: {str(e)}"
            log_msg = f"错误: {self.error_message}"
            print(log_msg)
            if hasattr(self, 'calculation_logs'):
                self.calculation_logs.append(log_msg)
            if hasattr(self, 'data_exception_records'):
                self.data_exception_records.append(self.error_message)
            return False
    
    def _has_rising_trend(self, close_prices, high_prices):
        """
        检查是否有上涨趋势
        上涨段：连续≥5根收盘价抬升+高点创新高
        
        Args:
            close_prices (list): 收盘价列表
            high_prices (list): 最高价列表
            
        Returns:
            bool: 是否有上涨趋势
        """
        try:
            if len(close_prices) < 5 or len(high_prices) < 5:
                return False
            
            # 检查连续5根收盘价抬升
            for i in range(len(close_prices) - 4):
                rising_close = True
                for j in range(5):
                    if i + j + 1 >= len(close_prices) or close_prices[i + j + 1] <= close_prices[i + j]:
                        rising_close = False
                        break
                
                # 严格检查高点是否整体创新高
                if rising_close:
                    # 检查高点是否逐渐抬升
                    rising_high = True
                    for j in range(1, 5):
                        if i + j >= len(high_prices) or high_prices[i + j] <= high_prices[i + j - 1]:
                            rising_high = False
                            break
                    
                    # 检查最终高点是否高于起始高点（整体趋势向上）
                    if rising_high and high_prices[i + 4] > high_prices[i]:
                        print(f"  - 检测到上涨趋势: 从索引{i}开始的5根K线连续收盘价抬升且高点创新高")
                        return True
            
            return False
            
        except Exception as e:
            print(f"检查上涨趋势时发生错误: {str(e)}")
            return False
    
    def _has_falling_trend(self, close_prices, low_prices):
        """
        检查是否有下跌趋势
        下跌段：连续≥5根收盘价走低+低点创新低
        
        Args:
            close_prices (list): 收盘价列表
            low_prices (list): 最低价列表
            
        Returns:
            bool: 是否有下跌趋势
        """
        try:
            if len(close_prices) < 5 or len(low_prices) < 5:
                return False
            
            # 检查连续5根收盘价走低
            for i in range(len(close_prices) - 4):
                falling_close = True
                for j in range(5):
                    if i + j + 1 >= len(close_prices) or close_prices[i + j + 1] >= close_prices[i + j]:
                        falling_close = False
                        break
                
                # 严格检查低点是否整体创新低
                if falling_close:
                    # 检查低点是否逐渐走低
                    falling_low = True
                    for j in range(1, 5):
                        if i + j >= len(low_prices) or low_prices[i + j] >= low_prices[i + j - 1]:
                            falling_low = False
                            break
                    
                    # 检查最终低点是否低于起始低点（整体趋势向下）
                    if falling_low and low_prices[i + 4] < low_prices[i]:
                        print(f"  - 检测到下跌趋势: 从索引{i}开始的5根K线连续收盘价走低且低点创新低")
                        return True
            
            return False
            
        except Exception as e:
            print(f"检查下跌趋势时发生错误: {str(e)}")
            return False
    
    def generate_centrals(self):
        """
        生成中枢，严格按照规范要求
        1. 中枢生成：仅基于"有效盘整段"的90%成交区间，下沿=区间低点，上沿=区间高点，中轨=(下沿+上沿)/2
        2. 中枢有效性必须满足3个条件（缺一不可）：
           - 振幅≥对应波动等级阈值
           - 后续10根标准K线中≥8根在中枢区间内（覆盖度≥80%）
           - 支撑次数≥2次（价格回踩下沿后反弹≥1%）且压力次数≥2次（价格冲击上沿后回落≥1%）
        3. 未满足条件→标注"无效中枢"及具体原因
        
        Returns:
            bool: 是否生成成功
        """
        try:
            # 前置检查增强
            if self.has_critical_error:
                print(f"  跳过中枢生成: 已存在关键错误 - {self.error_message}")
                return False
                
            if not self.valid_segments:
                self.error_message = "无有效盘整段，无法生成中枢"
                print(f"警告: {self.error_message}")
                return True
            
            if self.standard_k_lines is None or self.standard_k_lines.empty:
                self.has_critical_error = True
                self.error_message = "标准K线数据为空，无法生成中枢"
                print(f"错误: {self.error_message}")
                return False
            
            # 验证波动等级参数
            if not hasattr(self, 'volatility_params') or not hasattr(self, 'volatility_level'):
                self.has_critical_error = True
                self.error_message = "波动等级参数未设置，无法生成中枢"
                print(f"错误: {self.error_message}")
                return False
            
            # 验证波动等级有效性
            if self.volatility_level not in ['low', 'medium', 'high']:
                self.has_critical_error = True
                self.error_message = f"无效的波动等级: {self.volatility_level}"
                print(f"错误: {self.error_message}")
                return False
            
            # 获取波动等级对应的阈值
            amplitude_threshold = self.volatility_params[self.volatility_level]['segment_amplitude_threshold']
            
            print(f"开始生成中枢，有效盘整段数量: {len(self.valid_segments)}, 振幅阈值: ≥{amplitude_threshold}%")
            
            # 按日期排序标准K线
            klines_sorted = self.standard_k_lines.sort_values('date').copy().reset_index(drop=True)
            
            centrals = []
            valid_centrals = []
            valid_central_count = 0
            invalid_central_count = 0
            
            for segment in self.valid_segments:
                print(f"\n处理盘整段{segment['segment_id']}生成中枢...")
                
                # 严格按照规范：中枢基于有效盘整段的90%成交区间
                # 下沿=区间低点，上沿=区间高点，中轨=(下沿+上沿)/2
                # 禁止任何系数调整
                try:
                    lower = segment['lower_90pct']
                    upper = segment['upper_90pct']
                    
                    # 严格检查成交区间有效性
                    if lower <= 0 or upper <= 0 or upper <= lower:
                        print(f"  警告: 盘整段{segment['segment_id']}成交区间数据异常 (下沿={lower}, 上沿={upper})，中枢无效")
                        # 记录无效中枢
                        central = {
                            'central_id': f"C{segment['segment_id']}",
                            'segment_id': segment['segment_id'],
                            'start_date': segment['start_date'],
                            'end_date': segment['end_date'],
                            'lower': lower,
                            'upper': upper,
                            'middle': (lower + upper) / 2 if (lower + upper) > 0 else 0,
                            'amplitude': 0,
                            'is_valid': False,
                            'invalid_reason': ['成交区间数据异常'],
                            'validity_checks': {
                                'amplitude_check': '未通过',
                                'coverage_check': '未通过',
                                'support_check': '未通过',
                                'resistance_check': '未通过'
                            }
                        }
                        centrals.append(central)
                        invalid_central_count += 1
                        print(f"  中枢{segment['segment_id']}生成结果: 无效中枢 - 成交区间数据异常")
                        continue
                    
                    mid = (lower + upper) / 2
                    amplitude = segment['amplitude_90pct']
                    
                    print(f"  中枢区间: [{lower:.4f}, {upper:.4f}], 中轨: {mid:.4f}, 振幅: {amplitude:.2f}%")
                    
                    # 找到中枢对应的K线索引
                    segment_end_idx = klines_sorted[klines_sorted['date'] == segment['end_date']].index
                    if len(segment_end_idx) == 0:
                        print(f"  警告: 未找到盘整段结束日期对应的K线，跳过该盘整段")
                        continue
                    
                    segment_end_idx = segment_end_idx[0]
                    
                    # 有效性条件1：振幅≥对应波动等级阈值 - 严格验证
                    amplitude_valid = amplitude >= amplitude_threshold
                    print(f"  条件1验证: 振幅{amplitude:.2f}% {'≥' if amplitude_valid else '<'} {amplitude_threshold}% → {'通过' if amplitude_valid else '不通过'}")
                    
                    # 有效性条件2：后续10根标准K线中≥8根在中枢区间内（覆盖度≥80%）- 严格验证
                    coverage_start_idx = min(segment_end_idx + 1, len(klines_sorted) - 1)
                    coverage_end_idx = min(coverage_start_idx + 9, len(klines_sorted) - 1)
                    
                    coverage_valid = False
                    covered_count = 0
                    total_coverage = 0
                    coverage = 0
                    covered_klines = []
                    
                    if coverage_start_idx > coverage_end_idx:
                        # 中枢后面没有足够的K线
                        print(f"  条件2验证: 中枢后K线不足10根，覆盖度无法验证 → 不通过")
                    else:
                        coverage_klines = klines_sorted.iloc[coverage_start_idx:coverage_end_idx + 1]
                        covered_count = 0
                        total_coverage = len(coverage_klines)
                        
                        for idx, kline in coverage_klines.iterrows():
                            # 严格检查：收盘价在中枢区间内才算覆盖
                            if lower <= kline['close'] <= upper:
                                covered_count += 1
                                covered_klines.append(idx)
                        
                        coverage = covered_count / total_coverage * 100 if total_coverage > 0 else 0
                        coverage_valid = covered_count >= 8
                        print(f"  条件2验证: 后续{total_coverage}根K线中{covered_count}根在中枢区间内，覆盖度{coverage:.1f}% → {'通过' if coverage_valid else '不通过'}")
                    
                    # 有效性条件3：支撑次数≥2次（价格回踩下沿后反弹≥1%）且压力次数≥2次（价格冲击上沿后回落≥1%）
                    support_count = 0
                    resistance_count = 0
                    
                    # 记录具体的支撑和压力事件
                    support_events = []
                    resistance_events = []
                    
                    # 遍历整个K线序列，检查对这个中枢的支撑和压力
                    # 注意：只检查中枢形成后的K线（中枢结束日期之后的K线）
                    for i in range(segment_end_idx + 1, len(klines_sorted)):
                        try:
                            # 检查支撑：价格回踩下沿后反弹≥1%
                            if i > 0 and klines_sorted.iloc[i-1]['low'] <= lower + (upper - lower) * 0.05:  # 允许5%的误差范围
                                # 计算反弹幅度
                                if klines_sorted.iloc[i-1]['close'] > 0:
                                    price_change = (klines_sorted.iloc[i]['close'] - klines_sorted.iloc[i-1]['close']) / klines_sorted.iloc[i-1]['close'] * 100
                                    if price_change >= 1.0:  # 反弹≥1%
                                        support_count += 1
                                        support_events.append({
                                            'date': klines_sorted.iloc[i]['date'],
                                            'low': klines_sorted.iloc[i-1]['low'],
                                            'change': price_change
                                        })
                            
                            # 检查压力：价格冲击上沿后回落≥1%
                            if i > 0 and klines_sorted.iloc[i-1]['high'] >= upper - (upper - lower) * 0.05:  # 允许5%的误差范围
                                # 计算回落幅度
                                if klines_sorted.iloc[i-1]['close'] > 0:
                                    price_change = (klines_sorted.iloc[i-1]['close'] - klines_sorted.iloc[i]['close']) / klines_sorted.iloc[i-1]['close'] * 100
                                    if price_change >= 1.0:  # 回落≥1%
                                        resistance_count += 1
                                        resistance_events.append({
                                            'date': klines_sorted.iloc[i]['date'],
                                            'high': klines_sorted.iloc[i-1]['high'],
                                            'change': price_change
                                        })
                        except Exception as inner_e:
                            print(f"  警告: 检查支撑/压力事件时出错: {str(inner_e)}")
                            continue
                    
                    support_valid = support_count >= 2
                    resistance_valid = resistance_count >= 2
                    pressure_support_valid = support_valid and resistance_valid
                    
                    print(f"  条件3验证: 支撑次数{support_count}{'≥' if support_valid else '<'}2, 压力次数{resistance_count}{'≥' if resistance_valid else '<'}2 → {'通过' if pressure_support_valid else '不通过'}")
                    
                    # 详细记录支撑事件
                    for event in support_events:
                        print(f"    - 支撑事件: {event['date'].strftime('%Y-%m-%d')}, 低点: {event['low']:.4f}, 反弹: {event['change']:.2f}%")
                    
                    # 详细记录压力事件
                    for event in resistance_events:
                        print(f"    - 压力事件: {event['date'].strftime('%Y-%m-%d')}, 高点: {event['high']:.4f}, 回落: {event['change']:.2f}%")
                    
                    # 中枢有效性判断（三个条件必须同时满足）
                    is_valid = amplitude_valid and coverage_valid and pressure_support_valid
                    
                    # 创建中枢对象，包含详细的验证信息
                    central = {
                        'central_id': f"C{segment['segment_id']}",
                        'segment_id': segment['segment_id'],
                        'start_date': segment['start_date'],
                        'end_date': segment['end_date'],
                        'lower': lower,
                        'upper': upper,
                        'middle': mid,
                        'amplitude': amplitude,
                        'amplitude_threshold': amplitude_threshold,
                        'coverage': coverage,
                        'covered_count': covered_count,
                        'total_coverage': total_coverage,
                        'covered_klines': covered_klines,
                        'support_count': support_count,
                        'resistance_count': resistance_count,
                        'support_valid': support_valid,
                        'resistance_valid': resistance_valid,
                        'support_events': support_events,
                        'resistance_events': resistance_events,
                        'is_valid': is_valid,
                        'validity_checks': {
                            'amplitude_check': f"{amplitude:.2f}%{'≥' if amplitude_valid else '<'}{amplitude_threshold}%",
                            'coverage_check': f"{covered_count}/10{'≥' if coverage_valid else '<'}8",
                            'support_check': f"{support_count}{'≥' if support_valid else '<'}2",
                            'resistance_check': f"{resistance_count}{'≥' if resistance_valid else '<'}2"
                        },
                        'invalid_reason': [],
                        'validation_details': {
                            'amplitude_check': {'result': amplitude_valid, 'value': amplitude, 'threshold': amplitude_threshold},
                            'coverage_check': {'result': coverage_valid, 'count': covered_count, 'rate': coverage},
                            'support_check': {'result': support_valid, 'count': support_count},
                            'resistance_check': {'result': resistance_valid, 'count': resistance_count}
                        }
                    }
                    
                    # 添加无效原因（详细且具体）
                    if not is_valid:
                        if not amplitude_valid:
                            central['invalid_reason'].append(f'振幅不满足阈值要求（{amplitude:.2f}% < {amplitude_threshold}%）')
                        if not coverage_valid:
                            central['invalid_reason'].append(f'覆盖度不足（仅{covered_count}/{total_coverage}根K线在区间内，<80%）')
                        if not support_valid:
                            central['invalid_reason'].append(f'支撑次数不足（{support_count}次 < 2次）')
                        if not resistance_valid:
                            central['invalid_reason'].append(f'压力次数不足（{resistance_count}次 < 2次）')
                    
                    centrals.append(central)
                    
                    if is_valid:
                        valid_central_count += 1
                        valid_centrals.append(central)
                        print(f"  中枢{segment['segment_id']}生成结果: 有效中枢")
                    else:
                        invalid_central_count += 1
                        print(f"  中枢{segment['segment_id']}生成结果: 无效中枢 - {'; '.join(central['invalid_reason'])}")
                except Exception as segment_e:
                    print(f"  处理盘整段{segment['segment_id']}时出错: {str(segment_e)}")
                    # 创建错误标记的中枢
                    central = {
                        'central_id': f"C{segment['segment_id']}",
                        'segment_id': segment['segment_id'],
                        'start_date': segment.get('start_date', 'N/A'),
                        'end_date': segment.get('end_date', 'N/A'),
                        'lower': 0,
                        'upper': 0,
                        'middle': 0,
                        'amplitude': 0,
                        'is_valid': False,
                        'invalid_reason': [f'处理错误: {str(segment_e)}'],
                        'validity_checks': {
                            'amplitude_check': '处理错误',
                            'coverage_check': '处理错误',
                            'support_check': '处理错误',
                            'resistance_check': '处理错误'
                        }
                    }
                    centrals.append(central)
                    invalid_central_count += 1
            
            self.centrals = centrals
            self.valid_centrals = valid_centrals
            
            print(f"\n中枢生成完成: 共生成{valid_central_count}个有效中枢，{invalid_central_count}个无效中枢")
            
            # 如果没有有效中枢，设置适当的错误信息
            if not self.valid_centrals:
                self.error_message = "无有效中枢，无对应缠论信号"
                print(f"注意: {self.error_message}")
            
            return True
            
        except Exception as e:
            self.has_critical_error = True
            self.error_message = f"中枢生成失败: {str(e)}"
            print(f"错误: {self.error_message}")
            return False
    
    def identify_buy_signals(self):
        """
        识别买入信号（1-3买+破中枢反抽一买）
        
        核心前提：仅基于"有效中枢"识别信号，无有效中枢→直接输出"无有效中枢，无对应缠论信号"
        
        Returns:
            bool: 是否识别成功
        """
        try:
            if not self.valid_centrals:
                self.error_message = "无有效中枢，无对应缠论信号"
                print(f"注意: {self.error_message}")
                return True
            
            if self.standard_k_lines is None or self.standard_k_lines.empty:
                self.has_critical_error = True
                self.error_message = "标准K线数据为空，无法识别买入信号"
                print(f"错误: {self.error_message}")
                return False
            
            # 按日期排序
            klines_sorted = self.standard_k_lines.sort_values('date').copy().reset_index(drop=True)
            
            # 计算MACD指标
            klines_sorted = self._calculate_macd(klines_sorted)
            
            buy_signals = []
            signal_id = 1
            
            print(f"\n开始识别买入信号，有效中枢数量: {len(self.valid_centrals)}")
            
            # 识别各种类型的买入信号
            for central in self.valid_centrals:
                print(f"\n处理中枢{central['central_id']}的买入信号识别...")
                
                # 识别一买信号
                buy1_signal = self._identify_buy1_signal(central, klines_sorted, signal_id)
                if buy1_signal:
                    buy_signals.append(buy1_signal)
                    signal_id += 1
                    print(f"  识别到一买信号: {buy1_signal['signal_id']} (日期: {buy1_signal['buy_date'].strftime('%Y-%m-%d')})")
                
                # 识别二买信号
                buy2_signal = self._identify_buy2_signal(central, klines_sorted, signal_id, buy_signals)
                if buy2_signal:
                    buy_signals.append(buy2_signal)
                    signal_id += 1
                    print(f"  识别到二买信号: {buy2_signal['signal_id']} (日期: {buy2_signal['buy_date'].strftime('%Y-%m-%d')})")
                
                # 识别三买信号
                buy3_signal = self._identify_buy3_signal(central, klines_sorted, signal_id)
                if buy3_signal:
                    buy_signals.append(buy3_signal)
                    signal_id += 1
                    print(f"  识别到三买信号: {buy3_signal['signal_id']} (日期: {buy3_signal['buy_date'].strftime('%Y-%m-%d')})")
                
                # 识别破中枢反抽一买信号
                break_rebound_signal = self._identify_break_central_rebound_buy_signal(central, klines_sorted, signal_id)
                if break_rebound_signal:
                    buy_signals.append(break_rebound_signal)
                    signal_id += 1
                    print(f"  识别到破中枢反抽一买信号: {break_rebound_signal['signal_id']} (日期: {break_rebound_signal['rebound_date'].strftime('%Y-%m-%d')})")
            
            self.buy_signals = buy_signals
            
            if not buy_signals:
                self.error_message = "有有效中枢但未触发任何买入信号"
                print(f"注意: {self.error_message}")
            else:
                print(f"\n买入信号识别完成: 共识别到{len(buy_signals)}个有效买入信号")
                # 按信号日期排序
                self.buy_signals = sorted(buy_signals, key=lambda x: x.get('buy_date') or x.get('rebound_date'))
            
            return True
            
        except Exception as e:
            self.has_critical_error = True
            self.error_message = f"买入信号识别失败: {str(e)}"
            print(f"错误: {self.error_message}")
            return False
    
    def _calculate_macd(self, klines_df, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        
        Args:
            klines_df: K线数据DataFrame
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            DataFrame: 添加了MACD指标的K线数据
        """
        # 计算EMA
        klines_df['ema_fast'] = klines_df['close'].ewm(span=fast_period, adjust=False).mean()
        klines_df['ema_slow'] = klines_df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF (快线 - 慢线)
        klines_df['dif'] = klines_df['ema_fast'] - klines_df['ema_slow']
        
        # 计算DEA (信号线)
        klines_df['dea'] = klines_df['dif'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        klines_df['macd_hist'] = (klines_df['dif'] - klines_df['dea']) * 2
        
        return klines_df
    
    def _has_standard_bottom_divergence(self, klines_df, start_idx, end_idx):
        """
        检查是否存在标准底背驰
        
        Args:
            klines_df: K线数据
            start_idx: 起始索引
            end_idx: 结束索引
            
        Returns:
            tuple: (是否存在底背驰, 详细信息)
        """
        try:
            # 价格新低 + 黄白线不新低 + 绿柱缩短≥30%
            price_lows = klines_df['low'].iloc[start_idx:end_idx + 1]
            macd_dif = klines_df['dif'].iloc[start_idx:end_idx + 1]
            macd_hist = klines_df['macd_hist'].iloc[start_idx:end_idx + 1]
            
            # 找到两个低点位置
            recent_low_idx = price_lows.idxmin()
            recent_low = price_lows.min()
            
            # 找到之前的低点
            previous_part = klines_df.iloc[:recent_low_idx]
            if len(previous_part) < 10:  # 需要足够的历史数据
                return False, {"reason": "历史数据不足"}
            
            previous_low_idx = previous_part['low'].iloc[-10:].idxmin()
            previous_low = previous_part['low'].iloc[-10:].min()
            
            # 1. 价格新低
            price_new_low = recent_low < previous_low * 0.98  # 至少2%的新低
            
            # 2. 黄白线不新低
            dif_recent = klines_df.loc[recent_low_idx, 'dif']
            dif_previous = klines_df.loc[previous_low_idx, 'dif']
            dif_not_new_low = dif_recent >= dif_previous * 0.99  # 不创新低或创新低不超过1%
            
            # 3. 绿柱缩短≥30%
            recent_hist = klines_df.loc[recent_low_idx, 'macd_hist']
            previous_hist = klines_df.loc[previous_low_idx, 'macd_hist']
            hist_shorter = False
            if previous_hist < 0 and recent_hist < 0:
                # 都是绿柱，检查缩短
                shortening_ratio = (previous_hist - recent_hist) / abs(previous_hist)
                hist_shorter = shortening_ratio >= 0.3  # 缩短30%
            elif recent_hist > previous_hist:
                # 可能从绿柱变红柱
                hist_shorter = True
            
            is_divergence = price_new_low and dif_not_new_low and hist_shorter
            
            details = {
                'price_new_low': price_new_low,
                'dif_not_new_low': dif_not_new_low,
                'hist_shorter': hist_shorter,
                'recent_low_price': recent_low,
                'previous_low_price': previous_low,
                'recent_dif': dif_recent,
                'previous_dif': dif_previous,
                'recent_hist': recent_hist,
                'previous_hist': previous_hist
            }
                
            return is_divergence, details
        except Exception as e:
            print(f"计算MACD底背驰时发生错误: {str(e)}")
            if hasattr(self, 'data_exception_records'):
                self.data_exception_records.append(f"MACD底背驰计算错误: {str(e)}")
            return False, {"error": str(e)}
    
    def _has_standard_bottom_pattern(self, klines_df, idx):
        """
        检查是否存在标准底分型
        
        Args:
            klines_df: K线数据
            idx: 检查的K线索引
            
        Returns:
            bool: 是否为标准底分型
        """
        # 中间K线低点最低，右侧收盘价＞左侧
        if idx < 1 or idx >= len(klines_df) - 1:
            return False
        
        # 检查低点关系
        middle_low = klines_df.iloc[idx]['low']
        left_low = klines_df.iloc[idx-1]['low']
        right_low = klines_df.iloc[idx+1]['low']
        
        # 中间低点最低
        if not (middle_low < left_low * 0.995 and middle_low < right_low * 0.995):
            return False
        
        # 右侧收盘价 > 左侧收盘价
        left_close = klines_df.iloc[idx-1]['close']
        right_close = klines_df.iloc[idx+1]['close']
        
        return right_close > left_close * 1.005  # 右侧收盘价至少高0.5%
    
    def _identify_buy1_signal(self, central, klines_sorted, signal_id):
        """
        识别一买信号
        
        信号识别规则：下跌段≥5根标准K线+价格创中枢后新低+MACD底背驰（价格新低+黄白线不新低+绿柱缩短≥30%）+ 标准底分型（中间K线低点最低，右侧收盘价＞左侧）
        
        Args:
            central: 中枢信息
            klines_sorted: 排序后的K线数据
            signal_id: 信号ID
            
        Returns:
            dict or None: 信号信息或None
        """
        # 找到中枢结束后的K线
        central_end_idx = klines_sorted[klines_sorted['date'] == central['end_date']].index
        if len(central_end_idx) == 0:
            return None
        
        central_end_idx = central_end_idx[0]
        
        # 遍历中枢后的K线，寻找下跌段
        for i in range(central_end_idx + 1, len(klines_sorted) - 4):  # 确保有足够的K线检查下跌段
            # 检查下跌段≥5根标准K线
            if i + 4 >= len(klines_sorted):
                break
            
            # 检查是否为下跌段（连续5根收盘价走低）
            is_falling = True
            for j in range(i, i + 4):
                if klines_sorted.iloc[j]['close'] <= klines_sorted.iloc[j + 1]['close']:
                    is_falling = False
                    break
            
            if not is_falling:
                continue
            
            # 检查价格是否创中枢后新低
            segment_low = klines_sorted.iloc[i:i+5]['low'].min()
            central_after_low = klines_sorted.iloc[central_end_idx + 1:i]['low'].min() if central_end_idx + 1 < i else float('inf')
            
            if segment_low >= central_after_low * 1.001:  # 没有创新低
                continue
            
            # 检查MACD底背驰
            divergence, divergence_details = self._has_standard_bottom_divergence(klines_sorted, central_end_idx, i + 4)
            if not divergence:
                continue
            
            # 检查标准底分型
            # 寻找下跌段中的底分型
            bottom_pattern_idx = None
            for j in range(i + 1, i + 4):  # 底分型通常在下跌段的末尾
                if self._has_standard_bottom_pattern(klines_sorted, j):
                    bottom_pattern_idx = j
                    break
            
            if bottom_pattern_idx is None:
                continue
            
            # 计算交易参数
            buy_date = klines_sorted.iloc[bottom_pattern_idx]['date']
            buy_price = klines_sorted.iloc[bottom_pattern_idx]['close']
            stop_loss = segment_low * 0.99  # 止损设置在最低点下方1%
            take_profit = central['middle']  # 目标位为中枢中轨
            risk_reward_ratio = (take_profit - buy_price) / (buy_price - stop_loss) if buy_price > stop_loss else 0
            
            # 计算技术共振详情
            tech_resonance = {
                'divergence': True,
                'divergence_details': divergence_details,
                'bottom_pattern': True,
                'falling_segment': True,
                'price_new_low': True
            }
            
            # 创建信号
            signal = {
                'signal_id': signal_id,
                'central_id': central['central_id'],
                'signal_type': '一买',
                'buy_date': buy_date,
                'buy_price': buy_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'tech_resonance': tech_resonance,
                'validation_status': '待验证',
                'segment_low': segment_low,
                'central_after_low': central_after_low
            }
            
            return signal
        
        return None
    
    def _identify_buy2_signal(self, central, klines_sorted, signal_id, existing_signals):
        """
        识别二买信号
        
        信号识别规则：一买确认+回调不跌破一买对应中枢下沿+回调段30分钟底背驰+量能≥近5日均量85%
        
        Args:
            central: 中枢信息
            klines_sorted: 排序后的K线数据
            signal_id: 信号ID
            existing_signals: 已识别的信号列表
            
        Returns:
            dict or None: 信号信息或None
        """
        # 查找该中枢对应的一买信号
        buy1_signals = [s for s in existing_signals if s['central_id'] == central['central_id'] and s['signal_type'] == '一买']
        
        if not buy1_signals:
            return None
        
        for buy1_signal in buy1_signals:
            buy1_date = buy1_signal['buy_date']
            buy1_idx = klines_sorted[klines_sorted['date'] == buy1_date].index
            
            if len(buy1_idx) == 0:
                continue
            
            buy1_idx = buy1_idx[0]
            
            # 查找一买后的回调段
            # 假设回调段在一买后5-20根K线内
            max_callback_len = min(buy1_idx + 21, len(klines_sorted))
            
            for i in range(buy1_idx + 5, max_callback_len):
                # 检查是否为回调段（价格从高点回落）
                # 1. 先找到一买后的高点
                high_point = None
                high_idx = None
                
                for j in range(buy1_idx, i):
                    if high_point is None or klines_sorted.iloc[j]['high'] > high_point:
                        high_point = klines_sorted.iloc[j]['high']
                        high_idx = j
                
                if high_idx is None or high_idx >= i - 2:
                    continue
                
                # 2. 检查是否回落（收盘价低于高点）
                if klines_sorted.iloc[i]['close'] >= high_point * 0.99:
                    continue
                
                # 3. 检查回调不跌破中枢下沿
                if klines_sorted.iloc[i]['low'] < central['lower'] * 0.995:
                    continue
                
                # 4. 检查回调段底背驰（简化版，使用日线MACD）
                divergence, _ = self._has_standard_bottom_divergence(klines_sorted, high_idx, i)
                if not divergence:
                    continue
                
                # 5. 检查量能≥近5日均量85%
                recent_volumes = klines_sorted.iloc[max(0, i - 5):i]['volume']
                if len(recent_volumes) > 0:
                    avg_volume = recent_volumes.mean()
                    if klines_sorted.iloc[i]['volume'] < avg_volume * 0.85:
                        continue
                else:
                    continue
                
                # 计算交易参数
                buy_date = klines_sorted.iloc[i]['date']
                buy_price = klines_sorted.iloc[i]['close']
                stop_loss = central['lower'] * 0.99  # 止损设置在中枢下沿下方1%
                take_profit = central['upper']  # 目标位为中枢上沿
                risk_reward_ratio = (take_profit - buy_price) / (buy_price - stop_loss) if buy_price > stop_loss else 0
                
                # 创建信号
                signal = {
                    'signal_id': signal_id,
                    'central_id': central['central_id'],
                    'signal_type': '二买',
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'related_buy1_id': buy1_signal['signal_id'],
                    'tech_resonance': {
                        'divergence': True,
                        'volume_condition': True,
                        'price_condition': True
                    },
                    'validation_status': '待验证'
                }
                
                return signal
        
        return None
    
    def _identify_buy3_signal(self, central, klines_sorted, signal_id):
        """
        识别三买信号
        
        信号识别规则：有效中枢+价格突破上沿（连续2根标准K线收盘价≥上沿×对应反抽阈值）+ 回抽不跌破上沿+回抽段底背驰
        
        Args:
            central: 中枢信息
            klines_sorted: 排序后的K线数据
            signal_id: 信号ID
            
        Returns:
            dict or None: 信号信息或None
        """
        # 获取参数
        rebound_threshold = self.volatility_params[self.volatility_level]['central_rebound_threshold']
        
        # 找到中枢结束后的K线
        central_end_idx = klines_sorted[klines_sorted['date'] == central['end_date']].index
        if len(central_end_idx) == 0:
            return None
        
        central_end_idx = central_end_idx[0]
        
        # 寻找价格突破上沿
        breakout_found = False
        breakout_start_idx = None
        
        for i in range(central_end_idx + 1, len(klines_sorted) - 1):
            # 检查连续2根收盘价≥上沿×反抽阈值
            if (klines_sorted.iloc[i]['close'] >= central['upper'] * rebound_threshold and
                klines_sorted.iloc[i + 1]['close'] >= central['upper'] * rebound_threshold):
                
                breakout_found = True
                breakout_start_idx = i
                break
        
        if not breakout_found:
            return None
        
        # 寻找回抽
        # 回抽时间窗口为突破后30根K线
        retracement_window_end = min(breakout_start_idx + 31, len(klines_sorted))
        
        # 找到突破后的高点
        high_point = None
        high_idx = None
        
        for i in range(breakout_start_idx, retracement_window_end):
            if high_point is None or klines_sorted.iloc[i]['high'] > high_point:
                high_point = klines_sorted.iloc[i]['high']
                high_idx = i
        
        if high_idx is None:
            return None
        
        # 寻找回抽段（价格从高点回落）
        for i in range(high_idx + 1, retracement_window_end):
            # 检查回抽不跌破上沿
            if klines_sorted.iloc[i]['low'] < central['upper'] * 0.995:
                continue
            
            # 检查是否回落足够（至少2%）
            if klines_sorted.iloc[i]['close'] >= high_point * 0.98:
                continue
            
            # 检查回抽段底背驰
            divergence, _ = self._has_standard_bottom_divergence(klines_sorted, high_idx, i)
            if not divergence:
                continue
            
            # 计算交易参数
            buy_date = klines_sorted.iloc[i]['date']
            buy_price = klines_sorted.iloc[i]['close']
            stop_loss = central['upper'] * 0.98  # 止损设置在中枢上沿下方2%
            take_profit = high_point + (high_point - central['upper'])  # 目标位为突破幅度的一倍
            risk_reward_ratio = (take_profit - buy_price) / (buy_price - stop_loss) if buy_price > stop_loss else 0
            
            # 创建信号
            signal = {
                'signal_id': signal_id,
                'central_id': central['central_id'],
                'signal_type': '三买',
                'buy_date': buy_date,
                'buy_price': buy_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'breakout_price': central['upper'] * rebound_threshold,
                'high_point': high_point,
                'tech_resonance': {
                    'divergence': True,
                    'retracement_condition': True,
                    'breakout_condition': True
                },
                'validation_status': '待验证'
            }
            
            return signal
        
        return None
    
    def _identify_break_central_rebound_buy_signal(self, central, klines_sorted, signal_id):
        """
        识别破中枢反抽一买信号
        
        有效破位：连续2根收盘价≤下沿×对应破位阈值+破位日新低
        有效反抽：破位后对应时间窗口内+连续2根收盘价≥下沿×对应反抽阈值+量能≥近5日均量要求
        MACD底背驰+标准底分型
        
        Args:
            central: 中枢信息
            klines_sorted: 排序后的K线数据
            signal_id: 信号ID
            
        Returns:
            dict or None: 信号信息或None
        """
        # 获取参数
        break_threshold = self.volatility_params[self.volatility_level]['central_break_threshold']
        rebound_threshold = self.volatility_params[self.volatility_level]['central_rebound_threshold']
        
        print(f"  破中枢反抽一买信号识别参数: 破位阈值={break_threshold}, 反抽阈值={rebound_threshold}")
        
        # 找到中枢结束后的K线
        central_end_idx = klines_sorted[klines_sorted['date'] == central['end_date']].index
        if len(central_end_idx) == 0:
            print(f"  警告: 未找到中枢结束日期对应的K线")
            return None
        
        central_end_idx = central_end_idx[0]
        break_found = False
        break_date = None
        break_price = None
        break_start_idx = None
        
        # 寻找破位
        print(f"  开始寻找破位（连续2根收盘价≤{central['lower'] * break_threshold:.4f}且破位日新低）...")
        
        for i in range(central_end_idx + 1, len(klines_sorted) - 1):
            # 检查连续2根收盘价≤下沿×破位阈值
            if (klines_sorted.iloc[i]['close'] <= central['lower'] * break_threshold and
                klines_sorted.iloc[i + 1]['close'] <= central['lower'] * break_threshold):
                
                # 检查破位日新低
                current_low = klines_sorted.iloc[i]['low']
                prev_10_low = klines_sorted.iloc[max(0, i - 10):i]['low'].min()
                
                if current_low <= prev_10_low * 1.001:  # 接近或创新低
                    break_found = True
                    break_date = klines_sorted.iloc[i]['date']
                    break_price = klines_sorted.iloc[i]['close']
                    break_start_idx = i
                    print(f"  找到破位: 日期={break_date.strftime('%Y-%m-%d')}, 价格={break_price:.4f}")
                    break
        
        if not break_found:
            print(f"  未找到有效破位")
            return None
        
        # 寻找反抽
        print(f"  开始寻找反抽（连续2根收盘价≥{central['lower'] * rebound_threshold:.4f}且量能符合要求）...")
        
        # 时间窗口为破位后30根K线
        rebound_window_end = min(break_start_idx + 31, len(klines_sorted))
        rebound_found = False
        rebound_date = None
        rebound_price = None
        rebound_idx = None
        
        for i in range(break_start_idx + 1, rebound_window_end - 1):
            # 检查连续2根收盘价≥下沿×反抽阈值
            if (klines_sorted.iloc[i]['close'] >= central['lower'] * rebound_threshold and
                klines_sorted.iloc[i + 1]['close'] >= central['lower'] * rebound_threshold):
                
                # 检查量能条件
                current_volume = klines_sorted.iloc[i]['volume']
                prev_5_volumes = klines_sorted.iloc[max(0, i - 5):i]['volume']
                if len(prev_5_volumes) > 0:
                    avg_volume = prev_5_volumes.mean()
                    if current_volume >= avg_volume * 0.85:
                        # 检查MACD底背驰
                        divergence, divergence_details = self._has_standard_bottom_divergence(klines_sorted, break_start_idx, i)
                        if divergence:
                            # 检查标准底分型
                            if self._has_standard_bottom_pattern(klines_sorted, i) or self._has_standard_bottom_pattern(klines_sorted, i - 1):
                                rebound_found = True
                                rebound_date = klines_sorted.iloc[i]['date']
                                rebound_price = klines_sorted.iloc[i]['close']
                                rebound_idx = i
                                print(f"  找到反抽: 日期={rebound_date.strftime('%Y-%m-%d')}, 价格={rebound_price:.4f}")
                                break
        
        if not rebound_found:
            print(f"  未找到有效反抽")
            return None
        
        # 计算技术共振详情
        tech_resonance = {
            'break_condition': True,
            'rebound_condition': True,
            'volume_condition': True,
            'divergence': True,
            'bottom_pattern': True
        }
        
        # 计算交易参数
        stop_loss = break_price * 0.98
        take_profit = central['upper']  # 目标位为中枢上沿
        risk_reward_ratio = (take_profit - rebound_price) / (rebound_price - stop_loss) if rebound_price > stop_loss else 0
        
        # 估算买入数量（假设资金10000元，按20%仓位）
        estimated_position = 10000 * 0.2  # 2000元
        buy_quantity = int(estimated_position / rebound_price)
        
        # 创建完整的信号
        signal = {
            'signal_id': signal_id,
            'central_id': central['central_id'],
            'signal_type': '破中枢反抽一买',
            'break_date': break_date,
            'break_price': break_price,
            'rebound_date': rebound_date,
            'rebound_price': rebound_price,
            'buy_date': rebound_date,  # 统一的买入日期字段
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'tech_resonance': tech_resonance,
            'validation_status': '待验证',
            'buy_quantity': buy_quantity,
            'break_threshold_used': break_threshold,
            'rebound_threshold_used': rebound_threshold,
            'central_range': {
                'lower': central['lower'],
                'upper': central['upper'],
                'middle': central['middle']
            }
        }
        
        print(f"  成功识别破中枢反抽一买信号")
        return signal
    
    def generate_report(self, output_file):
        """
        生成分析报告，包含所有强制要求的输出信息
        
        Args:
            output_file (str): 输出文件路径
            
        Returns:
            str: 报告内容
        """
        try:
            report_content = []
            report_content.append(f"# {self.symbol} {self.year}年通用缠论分析报告")
            report_content.append("\n## 前置说明")
            
            # 1. 标的代码、分析周期、真实价格区间 - 严格按顺序
            report_content.append("\n### 标的信息与价格区间")
            report_content.append(f"- 标的代码：{self.symbol}")
            report_content.append(f"- 分析周期：{self.year}年")
            if self.year_low is not None and self.year_high is not None:
                report_content.append(f"- 年度最高价：{self.year_high:.3f}元")
                report_content.append(f"- 年度最低价：{self.year_low:.3f}元")
                report_content.append(f"- 真实价格区间：{self.year_low:.3f}元 - {self.year_high:.3f}元")
            else:
                report_content.append("- 无法获取价格区间数据")
            
            # 2. 波动率计算过程和波动等级判定
            report_content.append("\n### 波动率计算与波动等级")
            for item in self.volatility_calculation_process:
                if item.startswith('样本K线数量'):
                    report_content.append(f"- {item}")
                elif item.startswith('最高价'):
                    report_content.append(f"- {item}")
                elif item.startswith('最低价'):
                    report_content.append(f"- {item}")
                elif item.startswith('均价'):
                    report_content.append(f"- {item}")
                elif item.startswith('波动率计算结果'):
                    report_content.append(f"- {item}")
                else:
                    # 异常处理记录
                    report_content.append(f"- {item}")
            
            if self.volatility is not None:
                report_content.append(f"- 波动率：{self.volatility:.2f}%")
                
                if self.volatility_level == 'low':
                    report_content.append(f"- 判定阈值：≤10.0%")
                elif self.volatility_level == 'medium':
                    report_content.append(f"- 判定阈值：10.1%-18.0%")
                else:
                    report_content.append(f"- 判定阈值：＞18.0%")
                
                level_text = {
                    'low': '低波动ETF',
                    'medium': '中波动ETF', 
                    'high': '高波动ETF'
                }
                report_content.append(f"- 波动等级：{level_text.get(self.volatility_level, '未知')}")
            
            # 3. 数据缺失天数、完整性评级、影响说明
            report_content.append("\n### 数据完整性检查")
            report_content.append(f"- 缺失交易日数量：{self.missing_trading_days}天")
            report_content.append(f"- 完整性评级：{self.data_integrity_rating}")
            report_content.append(f"- 影响说明：{self.integrity_impact}")
            
            # 4. 数据异常处理结果
            if hasattr(self, 'data_exception_records') and self.data_exception_records:
                report_content.append("\n### 数据异常处理记录")
                for record in self.data_exception_records:
                    report_content.append(f"- {record}")
            
            # 标准K线数量
            if self.standard_k_lines is not None:
                report_content.append(f"\n- 标准K线生成数量：{len(self.standard_k_lines)}根")
            
            # 如果有严重错误，只输出异常说明
            if self.has_critical_error:
                report_content.append(f"\n## 异常说明")
                report_content.append(f"\n- {self.error_message}")
            else:
                # 有效盘整段清单 - 增强显示90%成交区间
                report_content.append("\n## 一、有效盘整段清单")
                report_content.append("\n| 序号 | 盘整段ID | 起始日期 | 结束日期 | K线数量 | 振幅 | 90%成交区间 | 是否达标 | 不满足条件项 |")
                report_content.append("|------|----------|---------|---------|---------|------|------------|----------|------------|")
                
                if self.segments:
                    for idx, segment in enumerate(self.segments, 1):
                        status = "达标" if segment['is_valid'] else "不达标"
                        invalid_reasons = "; ".join(segment['invalid_reason']) if segment['invalid_reason'] else "无"
                        
                        # 确保90%成交区间具体数值显示，避免0.000-0.000
                        lower_90pct = segment.get('lower_90pct', 0)
                        upper_90pct = segment.get('upper_90pct', 0)
                        if lower_90pct == 0 and upper_90pct == 0:
                            range_text = "数据异常"
                        else:
                            range_text = f"{lower_90pct:.3f}-{upper_90pct:.3f}"
                        
                        report_content.append(f"| {idx} | {segment['segment_id']} | {segment['start_date'].strftime('%Y-%m-%d')} | {segment['end_date'].strftime('%Y-%m-%d')} | {segment['k_count']} | {segment['amplitude_90pct']:.2f}% | {range_text} | {status} | {invalid_reasons} |")
                else:
                    report_content.append("| - | - | - | - | - | - | - | - | 未识别到任何盘整段 |")
                
                # 有效中枢清单 - 增强有效性3条件显示
                report_content.append("\n## 二、有效中枢清单")
                report_content.append("\n| 中枢ID | 起始日期 | 结束日期 | 下沿 | 上沿 | 中轨 | 振幅 | 有效性3条件是否满足（覆盖度/支撑/压力） | 状态说明 |")
                report_content.append("|--------|---------|---------|------|------|------|------|------------------------------------------|----------|")
                
                if self.valid_centrals:
                    for central in self.valid_centrals:
                        # 详细显示有效性3条件
                        coverage_valid = central.get('coverage_valid', False)
                        support_valid = central.get('support_valid', False)
                        resistance_valid = central.get('resistance_valid', False)
                        
                        validity_details = f"覆盖度:{'满足' if coverage_valid else '不满足'} / 支撑次数:{'满足' if support_valid else '不满足'} / 压力次数:{'满足' if resistance_valid else '不满足'}"
                        status_text = "有效" if central['is_valid'] else "无效（" + "; ".join(central['invalid_reason']) + "）"
                        
                        report_content.append(f"| {central['central_id']} | {central['start_date'].strftime('%Y-%m-%d')} | {central['end_date'].strftime('%Y-%m-%d')} | {central['lower']:.3f} | {central['upper']:.3f} | {central['middle']:.3f} | {central['amplitude']:.2f}% | {validity_details} | {status_text} |")
                else:
                    report_content.append("| - | - | - | - | - | - | - | - | 未识别到有效中枢 |")
                
                # 有效信号清单 - 增强信息显示，包含所有交易参数
                report_content.append("\n## 三、有效信号清单")
                report_content.append("\n| 信号ID | 类型 | 中枢ID | 买入日期 | 买入价 | 目标位 | 止损位 | 买入数量 | 技术共振详情 | 验证状态 |")
                report_content.append("|--------|------|--------|---------|--------|--------|--------|----------|-------------|----------|")
                
                if self.buy_signals:
                    for signal in self.buy_signals:
                        signal_type = signal.get('signal_type', '破中枢反抽一买')
                        
                        # 处理不同类型信号的日期显示
                        if 'buy_date' in signal:
                            buy_date = signal['buy_date']
                        elif 'rebound_date' in signal:
                            buy_date = signal['rebound_date']
                        else:
                            buy_date = None
                        
                        # 买入价格
                        buy_price = signal.get('buy_price', signal.get('rebound_price', 0))
                        stop_loss = signal.get('stop_loss', 0)
                        take_profit = signal.get('take_profit', 0)
                        
                        # 不再显示固定仓位，直接使用计算的买入数量
                        
                        # 买入数量
                        buy_quantity = signal.get('buy_quantity', 0)
                        if buy_quantity == 0 and buy_price > 0:
                            # 估算买入数量（假设资金10000元，按30%仓位）
                            estimated_position = 10000 * 0.3  # 3000元
                            buy_quantity = int(estimated_position / buy_price)
                        
                        # 技术共振详情
                        resonance_details = []
                        tech_resonance = signal.get('tech_resonance', {})
                        
                        # 统一处理所有类型信号的技术共振信息
                        # 首先检查通用的技术共振条件
                        if tech_resonance.get('divergence', False) or signal.get('macd_divergence', False):
                            if signal_type == '一买' or signal_type == '破中枢反抽一买':
                                resonance_details.append('MACD底背驰')
                            elif signal_type == '二买':
                                resonance_details.append('回调段底背驰')
                            elif signal_type == '三买':
                                resonance_details.append('回抽段底背驰')
                        
                        if tech_resonance.get('bottom_pattern', False) or signal.get('bottom_pattern', False):
                            resonance_details.append('标准底分型')
                        
                        if tech_resonance.get('volume_condition', False):
                            resonance_details.append('量能满足')
                        
                        # 然后处理特定信号类型的条件
                        if signal_type == '二买' and tech_resonance.get('price_condition', False):
                            resonance_details.append('价格条件满足')
                        
                        if signal_type == '三买':
                            if tech_resonance.get('breakout_condition', False):
                                resonance_details.append('突破确认')
                            if tech_resonance.get('retracement_condition', False):
                                resonance_details.append('回抽有效')
                        
                        if signal_type == '破中枢反抽一买':
                            if tech_resonance.get('break_condition', False):
                                resonance_details.append('破位有效')
                            if tech_resonance.get('rebound_condition', False):
                                resonance_details.append('反抽确认')
                        
                        resonance_text = "; ".join(resonance_details) if resonance_details else "无"
                        date_str = buy_date.strftime('%Y-%m-%d') if buy_date else "-"
                        
                        report_content.append(f"| {signal['signal_id']} | {signal_type} | {signal['central_id']} | {date_str} | {buy_price:.3f} | {take_profit:.3f} | {stop_loss:.3f} | {buy_quantity} | {resonance_text} | {signal['validation_status']} |")
                else:
                    if self.valid_centrals:
                        report_content.append("| - | - | - | - | - | - | - | - | - | - | 有有效中枢但未触发任何买入信号条件 |")
                    else:
                        report_content.append("| - | - | - | - | - | - | - | - | - | - | 无满足条件的有效中枢 |")
            
            # 校验说明
            report_content.append("\n## 四、校验说明")
            report_content.append("\n- 所有分析基于标的自身2025年完整历史日K数据")
            if self.year_low is not None and self.year_high is not None:
                report_content.append(f"- 所有价格数据均在真实价格区间{self.year_low:.3f}-{self.year_high:.3f}元内")
            
            if self.volatility is not None:
                report_content.append(f"- 波动率计算结果：{self.volatility:.2f}%")
            
            if self.volatility_level:
                level_text = {
                    'low': '低波动',
                    'medium': '中波动',
                    'high': '高波动'
                }
                report_content.append(f"- 参数自适应匹配{level_text.get(self.volatility_level, '未知')}ETF特性")
            
            report_content.append("- 中枢区间严格按照盘整段90%成交区间计算，未使用系数调整")
            report_content.append("- 所有交易参数基于10000元资金模型自动计算")
            
            # 免责声明
            report_content.append("\n---")
            report_content.append("\n*本报告基于通用缠论分析规范生成，仅供参考，不构成投资建议。投资有风险，入市需谨慎。*")
            
            # 保存报告
            final_report = "\n".join(report_content)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            print(f"报告已保存至：{output_file}")
            return final_report
            
        except Exception as e:
            error_msg = f"报告生成失败: {str(e)}"
            print(error_msg)
            return error_msg


def main():
    """
    主函数
    """
    # 创建分析器实例
    analyzer = UniversalChanlunETFAnalyzer(symbol="512660", year=2025)
    
    # 执行分析流程
    print("===== 开始通用缠论ETF分析 =====")
    
    # 1. 加载数据
    if not analyzer.load_data():
        print(f"数据加载失败: {analyzer.error_message}")
        analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
        return
    
    # 检查是否有严重错误
    if analyzer.has_critical_error:
        print(f"分析终止：{analyzer.error_message}")
        analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
        return
    
    # 2. 生成标准K线
    if not analyzer.generate_standard_k_lines():
        print(f"标准K线生成失败: {analyzer.error_message}")
        analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
        return
    
    # 3. 计算波动率并确定波动等级
    if not analyzer.determine_volatility_level():
        print(f"波动等级确定失败: {analyzer.error_message}")
        analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
        return
    
    # 4. 划分走势段
    if not analyzer.divide_segments():
        print(f"走势段划分失败: {analyzer.error_message}")
        analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
        return
    
    # 5. 生成中枢
    if not analyzer.generate_centrals():
        print(f"中枢生成失败: {analyzer.error_message}")
        analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
        return
    
    # 6. 识别买入信号
    analyzer.identify_buy_signals()
    
    # 7. 生成报告
    analyzer.generate_report(f"{analyzer.symbol}_{analyzer.year}_universal_chanlun_report.md")
    
    print("===== 分析完成 =====")


if __name__ == "__main__":
    main()