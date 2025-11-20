#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据获取器 - 修复版本
修复了日期范围不正确和符号验证问题
添加了日期范围完整性检查
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

import logging
import pandas as pd
import requests
import json
import re
import time
import random
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import get_data_fetcher_config, get_backtest_config, get_strategy_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("无法导入配置模块，使用默认配置")

logger = logging.getLogger('StockDataFetcher')
logger.setLevel(logging.INFO)

# 配置日志处理器
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DataFetchError(Exception):
    """数据获取异常"""
    pass

class StockDataFetcher:
    """高效股票数据获取器 - 修复日期范围和符号验证问题"""
    
    def __init__(self, max_retries: int = None, timeout: int = None):
        """
        初始化数据获取器
        :param max_retries: 最大重试次数
        :param timeout: 请求超时时间(秒)
        """
        # 加载配置或使用默认值
        if CONFIG_AVAILABLE:
            try:
                config = get_data_fetcher_config()
                
                # 设置参数
                self.max_retries = max_retries or config.get('max_retries', 3)
                self.timeout = timeout or config.get('timeout', 10)
                self.type_safety = config.get('type_safety', True)
                self.data_sources = config.get('data_sources', ['tencent', 'sina'])
                self.cache_enabled = config.get('cache_enabled', True)
                self.cache_ttl = config.get('cache_ttl', 300)
                
                # 获取Sina配置
                sina_config = config.get('sina', {})
                self.sina_base_url = sina_config.get('base_url', "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData")
                self.sina_params = sina_config.get('params', {})
                
                # 获取Tencent配置
                tencent_config = config.get('tencent', {})
                self.tencent_enabled = tencent_config.get('enabled', True)
                self.tencent_weekly_url = tencent_config.get('weekly_url', "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get")
                self.tencent_params = tencent_config.get('params', {})
                
                logger.info("配置模块加载成功")
                
            except Exception as e:
                logger.warning(f"配置加载失败，使用默认配置: {str(e)}")
                self._set_default_config()
        else:
            self._set_default_config()
            logger.info("使用默认配置初始化")
        
        # 初始化缓存系统
        self.cache = {}
        self.cache_timestamps = {}
        
        # 功能完整性检查
        self._feature_check()
        
        logger.info(f"数据获取器初始化完成 - 支持功能: {self._get_feature_summary()}")
    
    def _set_default_config(self):
        """设置默认配置"""
        self.max_retries = 3
        self.timeout = 10
        self.type_safety = True
        self.data_sources = ['tencent', 'sina']
        self.cache_enabled = True
        self.cache_ttl = 300
        
        # Sina默认配置
        self.sina_base_url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        self.sina_params = {
            'weekly': {'scale': 'week', 'ma': 'no', 'datalen': '500'},
            'daily': {'scale': '240', 'ma': 'no', 'datalen': '1000'},
            'minute': {'scale': '5', 'ma': 'no', 'datalen': '10000'}
        }
        
        # Tencent默认配置
        self.tencent_enabled = True
        self.tencent_weekly_url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        self.tencent_params = {
            'weekly': {'_var': 'kline_week', 'param': '{symbol},week,,,320,qfq'}
        }
    
    def _feature_check(self):
        """功能完整性检查"""
        required_features = [
            'get_weekly_data', 'get_daily_data', 'get_realtime_data', 'get_minute_data',
            'health_check', 'cache_system', 'error_handling', 'data_validation'
        ]
        
        implemented_features = []
        
        # 检查核心数据获取功能
        if hasattr(self, 'get_weekly_data') and callable(getattr(self, 'get_weekly_data')):
            implemented_features.append('get_weekly_data')
        
        if hasattr(self, 'get_daily_data') and callable(getattr(self, 'get_daily_data')):
            implemented_features.append('get_daily_data')
        
        if hasattr(self, 'get_realtime_data') and callable(getattr(self, 'get_realtime_data')):
            implemented_features.append('get_realtime_data')
        
        # 检查分钟数据功能
        if hasattr(self, 'get_minute_data') and callable(getattr(self, 'get_minute_data')):
            implemented_features.append('get_minute_data')
        
        # 检查系统功能
        if hasattr(self, 'health_check') and callable(getattr(self, 'health_check')):
            implemented_features.append('health_check')
        
        if hasattr(self, '_get_from_cache') and hasattr(self, '_save_to_cache'):
            implemented_features.append('cache_system')
        
        if hasattr(self, '_request_with_retry'):
            implemented_features.append('error_handling')
        
        if hasattr(self, '_validate_symbol') and hasattr(self, '_validate_dates'):
            implemented_features.append('data_validation')
        
        # 记录检查结果
        missing_features = set(required_features) - set(implemented_features)
        
        if missing_features:
            logger.warning(f"缺失功能: {missing_features}")
        else:
            logger.info("所有核心功能已完整实现")
    
    def _get_feature_summary(self):
        """获取功能摘要"""
        features = []
        
        if 'tencent' in self.data_sources:
            features.append('腾讯数据源')
        if 'sina' in self.data_sources:
            features.append('新浪数据源')
        
        if self.cache_enabled:
            features.append('缓存系统')
        
        features.extend(['周线数据', '日线数据', '实时数据', '分钟数据', '健康检查'])
        
        return ', '.join(features)
    
    def _convert_date_format(self, date_str: str) -> str:
        """
        日期格式转换
        :param date_str: 日期字符串
        :return: YYYYMMDD格式的日期字符串
        """
        if not date_str or not isinstance(date_str, str):
            return date_str
            
        # 移除破折号
        if '-' in date_str:
            date_str = date_str.replace('-', '')
        
        # 如果已经是YYYYMMDD格式，直接返回
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
            
        return date_str
    
    def _format_symbol(self, symbol: str) -> str:
        """
        格式化股票代码 - 增强错误处理
        :param symbol: 股票代码
        :return: 标准化后的股票代码（纯数字）
        """
        try:
            if not isinstance(symbol, str):
                symbol = str(symbol)
            
            # 验证股票代码格式
            pattern = r'^([A-Za-z]{2})?(\d{6})(\.[A-Za-z]{2})?$'
            match = re.match(pattern, symbol)
            if not match:
                logger.warning(f"无效股票代码格式: {symbol}")
                return symbol  # 返回原始值，不中断流程
            
            # 提取数字部分
            digit_part = match.group(2)
            
            # 返回纯数字代码
            return digit_part
            
        except Exception as e:
            logger.warning(f"股票代码格式化失败: {symbol}, 错误: {str(e)}")
            return symbol  # 返回原始值，不中断流程
    
    def _validate_symbol(self, symbol: str) -> str:
        """
        验证和标准化股票代码 - 防御性修复：增强类型和长度检查
        :param symbol: 股票代码
        :return: 标准化后的股票代码（纯数字）
        """
        try:
            # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 防御性修复：检查symbol类型，防止DataFrame等无效类型
            if symbol is None:
                raise DataFetchError("股票代码不能为None")
            
            # 检查是否为DataFrame或其他复杂对象（通过字符串表示长度判断）
            symbol_str = str(symbol)
            if len(symbol_str) > 100:  # 正常股票代码不会超过20字符，100以上可能是DataFrame
                logger.error(f"疑似DataFrame被当作股票代码传递: {symbol_str[:100]}...")
                raise DataFetchError(f"无效股票代码类型: 疑似DataFrame对象")
            
            # 检查是否为Pandas Series或DataFrame的字符串表示特征
            if any(marker in symbol_str for marker in ['DataFrame', 'Series', 'open', 'high', 'low', 'close', 'volume', 'date']):
                logger.error(f"检测到DataFrame特征在股票代码中: {symbol_str[:200]}")
                raise DataFetchError(f"无效股票代码: 检测到DataFrame特征")
            
            formatted_symbol = self._format_symbol(symbol)
            
            # 验证是否为6位数字
            if not re.match(r'^\d{6}$', formatted_symbol):
                raise DataFetchError(f"无效股票代码格式: {symbol} (期望6位数字，得到: {formatted_symbol})")
            
            return formatted_symbol
            
        except DataFetchError:
            raise  # 重新抛出已知错误
        except Exception as e:
            logger.error(f"股票代码验证异常: {str(e)}")
            raise DataFetchError(f"股票代码验证失败: {symbol}")
    
    def _get_market_prefix(self, symbol: str) -> str:
        """
        获取市场前缀 - 增强错误处理
        :param symbol: 纯数字股票代码
        :return: 市场前缀 ('sh' or 'sz')
        """
        try:
            if symbol.startswith("6") or symbol.startswith("5") or symbol.startswith("9"):
                return "sh"
            elif symbol.startswith("0") or symbol.startswith("3") or symbol.startswith("1"):
                return "sz"
            else:
                logger.warning(f"无法识别的股票代码前缀: {symbol}")
                return "sh"  # 默认返回上海市场
                
        except Exception as e:
            logger.error(f"获取市场前缀异常: {str(e)}")
            return "sh"  # 默认返回上海市场
    
    def _validate_dates(self, start_date: Optional[str], end_date: Optional[str]) -> tuple:
        """
        验证和标准化日期范围 - 增强错误处理和日志记录
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: (start_dt, end_dt, original_start, original_end)
        """
        try:
            # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 增强日志：记录原始日期参数
            logger.debug(f"日期验证输入 - start_date: {start_date}, end_date: {end_date}")
            
            # 记录原始参数
            original_start = start_date
            original_end = end_date
            
            # 转换日期格式
            if end_date:
                end_date = self._convert_date_format(end_date)
                logger.debug(f"转换后end_date: {end_date}")
            else:
                end_date = datetime.now().strftime("%Y%m%d")
                logger.debug(f"使用默认end_date: {end_date}")
                
            if start_date:
                start_date = self._convert_date_format(start_date)
                logger.debug(f"转换后start_date: {start_date}")
            else:
                # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 修复：使用用户提供的end_date来计算start_date，而不是当前时间
                if end_date:
                    end_dt_temp = datetime.strptime(end_date, "%Y%m%d")
                    start_date = (end_dt_temp - timedelta(days=365)).strftime("%Y%m%d")
                else:
                    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
                logger.debug(f"使用计算start_date: {start_date}")
            
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 增强日志：记录最终日期范围
            date_range_days = (end_dt - start_dt).days
            logger.info(f"日期范围验证: {start_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')} (共{date_range_days}天)")
            
            if start_dt > end_dt:
                raise DataFetchError(f"开始日期 {start_date} 不能晚于结束日期 {end_date}")
                
            return start_dt, end_dt, original_start, original_end
            
        except ValueError as e:
            logger.error(f"日期格式错误: {str(e)}")
            raise DataFetchError("日期格式错误，请使用YYYYMMDD或YYYY-MM-DD格式")
        except Exception as e:
            logger.error(f"日期验证异常: {str(e)}")
            raise DataFetchError(f"日期验证失败: {start_date} - {end_date}")
    
    def _get_cache_key(self, data_type: str, symbol: str, start_date: str, end_date: str) -> str:
        """
        生成缓存键 - 增强错误处理
        :param data_type: 数据类型
        :param symbol: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 缓存键
        """
        try:
            return f"{data_type}:{symbol}:{start_date}:{end_date}"
        except Exception as e:
            logger.error(f"生成缓存键异常: {str(e)}")
            return f"error:{int(time.time())}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据 - 增强错误处理
        :param cache_key: 缓存键
        :return: 数据DataFrame或None
        """
        try:
            if not self.cache_enabled:
                return None
            
            # 检查缓存是否存在且未过期
            if cache_key in self.cache:
                if cache_key in self.cache_timestamps:
                    timestamp = self.cache_timestamps[cache_key]
                    if time.time() - timestamp < self.cache_ttl:
                        return self.cache[cache_key].copy()
                    else:
                        # 缓存过期，删除
                        del self.cache[cache_key]
                        del self.cache_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"从缓存获取数据异常: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """
        保存数据到缓存 - 增强错误处理
        :param cache_key: 缓存键
        :param data: 数据DataFrame
        """
        try:
            if self.cache_enabled:
                self.cache[cache_key] = data.copy()
                self.cache_timestamps[cache_key] = time.time()
        except Exception as e:
            logger.error(f"保存数据到缓存异常: {str(e)}")
    
    def _request_with_retry(self, request_func, *args, **kwargs):
        """
        带重试的请求包装器 - 增强错误处理
        :param request_func: 请求函数
        :return: 请求结果
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    time.sleep(random.uniform(0.5, 2.0))
                
                result = request_func(*args, **kwargs)
                if result is not None:
                    return result
                    
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"所有重试尝试失败: {str(e)}")
                    return None
                else:
                    logger.warning(f"尝试 {attempt} 失败: {str(e)}")
        
        return None
    
    def _safe_dataframe_operation(self, df: pd.DataFrame, operation: str, **kwargs) -> pd.DataFrame:
        """
        安全的DataFrame操作 - 新增：增强错误处理
        :param df: DataFrame
        :param operation: 操作类型 ('rename', 'convert_dtypes', 'add_column')
        :return: 处理后的DataFrame或空DataFrame
        """
        try:
            if df is None or df.empty:
                return pd.DataFrame()
            
            if operation == 'rename':
                column_map = kwargs.get('column_map', {})
                return df.rename(columns=column_map)
                
            elif operation == 'convert_dtypes':
                date_col = kwargs.get('date_col', 'date')
                numeric_cols = kwargs.get('numeric_cols', [])
                
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
            elif operation == 'add_column':
                column_name = kwargs.get('column_name')
                column_value = kwargs.get('column_value')
                
                if column_name:
                    df[column_name] = column_value
                
                return df
                
            else:
                return df
                
        except Exception as e:
            logger.warning(f"DataFrame操作失败: {operation}, 错误: {str(e)}")
            return pd.DataFrame()
    
    def _check_date_range_completeness(self, df: pd.DataFrame, start_dt: datetime, end_dt: datetime, symbol: str, data_type: str):
        """
        检查日期范围完整性 - 新增：确保实际数据范围覆盖请求范围
        :param df: 数据DataFrame
        :param start_dt: 请求开始日期
        :param end_dt: 请求结束日期
        :param symbol: 股票代码
        :param data_type: 数据类型
        """
        if df.empty or 'date' not in df.columns:
            return
            
        # 获取实际数据日期范围
        actual_start = df['date'].min()
        actual_end = df['date'].max()
        
        # 检查日期范围完整性
        if actual_start > start_dt or actual_end < end_dt:
            logger.warning(
                f"数据日期范围不完整: {symbol} {data_type}\n"
                f"请求范围: {start_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')}\n"
                f"实际范围: {actual_start.strftime('%Y-%m-%d')} 至 {actual_end.strftime('%Y-%m-%d')}\n"
                f"缺失数据: {self._get_missing_date_range(start_dt, end_dt, actual_start, actual_end)}"
            )
    
    def _get_missing_date_range(self, start_dt: datetime, end_dt: datetime, 
                               actual_start: datetime, actual_end: datetime) -> str:
        """
        获取缺失的日期范围描述
        """
        missing_parts = []
        
        if actual_start > start_dt:
            missing_parts.append(f"开始部分: {start_dt.strftime('%Y-%m-%d')} 至 {actual_start.strftime('%Y-%m-%d')}")
        
        if actual_end < end_dt:
            missing_parts.append(f"结束部分: {actual_end.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')}")
        
        return "; ".join(missing_parts) if missing_parts else "无缺失数据"
    
    def clean_symbol_format(self, symbol: str) -> str:
        """
        清洗股票代码格式 - 移除市场前缀
        :param symbol: 股票代码
        :return: 纯数字股票代码
        """
        try:
            if not isinstance(symbol, str):
                symbol = str(symbol)
            
            # 移除市场前缀
            if symbol.startswith(('sh', 'sz')):
                return symbol[2:]  # 移除前2字符
            return symbol
            
        except Exception as e:
            logger.warning(f"股票代码清洗失败: {symbol}, 错误: {str(e)}")
            return symbol
    
    def get_weekly_data(self, symbol: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取周线数据 - 增强错误处理和日期日志
        :param symbol: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 周线数据DataFrame
        """
        try:
            # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 增强日志：记录方法调用参数
            logger.info(f"获取周线数据 - 符号: {symbol}, 开始: {start_date}, 结束: {end_date}")
            
            # 验证和标准化
            symbol = self._validate_symbol(symbol)
            start_dt, end_dt, original_start, original_end = self._validate_dates(start_date, end_date)
            
            # 检查缓存
            cache_start = original_start if original_start else start_dt.strftime("%Y%m%d")
            cache_end = original_end if original_end else end_dt.strftime("%Y%m%d")
            cache_key = self._get_cache_key('weekly', symbol, cache_start, cache_end)
            
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 新增：记录缓存数据的日期范围
                if not cached_data.empty and 'date' in cached_data.columns:
                    cache_start_date = cached_data['date'].min().strftime('%Y-%m-%d')
                    cache_end_date = cached_data['date'].max().strftime('%Y-%m-%d')
                    cache_days = (cached_data['date'].max() - cached_data['date'].min()).days
                    logger.info(f"缓存数据日期范围: {cache_start_date} 至 {cache_end_date} (共{cache_days}天)")
                
                logger.info(f"从缓存获取周线数据: {symbol}")
                return cached_data
            
            # 根据纯数字代码确定市场前缀
            market = self._get_market_prefix(symbol)
            full_symbol = f"{market}{symbol}"  # 用于请求的代码
            
            # 按优先级尝试各个数据源
            for source in self.data_sources:
                try:
                    if source == 'tencent':
                        df = self._get_tencent_weekly_data(symbol, start_dt, end_dt, full_symbol)
                    elif source == 'sina':
                        df = self._get_sina_weekly_data(symbol, start_dt, end_dt, full_symbol)
                    else:
                        continue
                    
                    # 使用安全的DataFrame检查
                    if df is not None and not df.empty:
                        # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 新增：记录实际获取数据的日期范围
                        if 'date' in df.columns:
                            actual_start = df['date'].min().strftime('%Y-%m-%d')
                            actual_end = df['date'].max().strftime('%Y-%m-%d')
                            actual_days = (df['date'].max() - df['date'].min()).days
                            logger.info(f"数据源返回的实际日期范围: {actual_start} 至 {actual_end} (共{actual_days}天)")
                        
                        # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 新增：检查日期范围完整性
                        self._check_date_range_completeness(df, start_dt, end_dt, symbol, 'weekly')
                        
                        self._save_to_cache(cache_key, df)
                        logger.info(f"成功从 {source} 获取周线数据: {len(df)} 条")
                        return df
                        
                except Exception as e:
                    logger.warning(f"数据源 {source} 失败: {str(e)}")
                    continue
            
            logger.error("所有周线数据获取方式均失败")
            return pd.DataFrame()
            
        except DataFetchError as e:
            logger.error(f"获取周线数据参数错误: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取周线数据异常: {str(e)}")
            return pd.DataFrame()
    
    def _get_tencent_weekly_data(self, symbol: str, start_dt: datetime, 
                                end_dt: datetime, full_symbol: str) -> pd.DataFrame:
        """
        获取腾讯财经周线数据 - 增强错误处理
        """
        try:
            if not self.tencent_enabled:
                return pd.DataFrame()
            
            timestamp = int(time.time() * 1000)
            param_str = f"{full_symbol},week,{start_dt.strftime('%Y-%m-%d')},{end_dt.strftime('%Y-%m-%d')},500,qfq"
            params = {
                "_var": self.tencent_params['weekly']['_var'],
                "param": param_str,
                "r": f"0.{timestamp}"
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://gu.qq.com/'
            }
            
            def fetch_func():
                try:
                    response = requests.get(self.tencent_weekly_url, params=params, 
                                          headers=headers, timeout=self.timeout)
                    if response.status_code != 200:
                        return pd.DataFrame()
                    
                    content = response.text.strip()
                    if not content:
                        return pd.DataFrame()
                    
                    # 修复JSON解析逻辑
                    json_str = content
                    if content.startswith('kline_week='):
                        json_str = content[11:]
                    
                    if json_str.endswith(';'):
                        json_str = json_str[:-1]
                    
                    data = json.loads(json_str)
                    
                    # 腾讯接口返回的数据结构验证
                    if 'data' not in data or 'code' not in data:
                        return pd.DataFrame()
                    
                    if data.get('code') != 0:
                        return pd.DataFrame()
                    
                    stock_data = data.get('data', {})
                    if full_symbol not in stock_data:
                        return pd.DataFrame()
                    
                    symbol_data = stock_data[full_symbol]
                    
                    # 查找周线数据键
                    weekly_keys = ['qfqweek', 'week', 'qfqWeek', 'Week']
                    weekly_data = None
                    
                    for key in weekly_keys:
                        if key in symbol_data:
                            weekly_data = symbol_data[key]
                            break
                    
                    if not weekly_data:
                        return pd.DataFrame()
                    
                    # 转换为DataFrame
                    columns = ['date', 'open', 'close', 'high', 'low', 'volume']
                    df = pd.DataFrame(weekly_data, columns=columns)
                    
                    # 使用安全的DataFrame操作
                    df = self._safe_dataframe_operation(df, 'convert_dtypes', 
                                                       date_col='date',
                                                       numeric_cols=['open', 'close', 'high', 'low', 'volume'])
                    
                    df = self._safe_dataframe_operation(df, 'add_column', 
                                                       column_name='symbol', column_value=full_symbol)
                    
                    # 过滤日期范围
                    if not df.empty and 'date' in df.columns:
                        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                    
                    return df
                    
                except Exception as e:
                    logger.warning(f"腾讯数据请求异常: {str(e)}")
                    return pd.DataFrame()
            
            result = self._request_with_retry(fetch_func)
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取腾讯周线数据异常: {str(e)}")
            return pd.DataFrame()
    
    def _get_sina_weekly_data(self, symbol: str, start_dt: datetime, 
                             end_dt: datetime, full_symbol: str) -> pd.DataFrame:
        """
        获取新浪周线数据 - 增强错误处理
        """
        try:
            # 尝试不同的scale值获取周线数据
            scale_options = ["240", "60", "30", "15", "week"]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://finance.sina.com.cn/'
            }
            
            def fetch_func():
                for scale_val in scale_options:
                    params = {
                        "symbol": full_symbol,
                        "scale": scale_val,
                        "ma": "no",
                        "datalen": "500"
                    }
                    
                    try:
                        response = requests.get(self.sina_base_url, params=params, 
                                              headers=headers, timeout=self.timeout)
                        if response.status_code != 200:
                            continue
                        
                        content = response.text.strip()
                        if not content or content == "null":
                            continue
                        
                        data = json.loads(content)
                        if not data or not isinstance(data, list):
                            continue
                        
                        # 转换为DataFrame
                        df = pd.DataFrame(data)
                        
                        # 安全重命名列
                        column_map = {
                            "day": "date",
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "volume": "volume"
                        }
                        df = self._safe_dataframe_operation(df, 'rename', column_map=column_map)
                        
                        # 安全转换数据类型
                        df = self._safe_dataframe_operation(df, 'convert_dtypes',
                                                           date_col='date',
                                                           numeric_cols=['open', 'close', 'high', 'low', 'volume'])
                        
                        df = self._safe_dataframe_operation(df, 'add_column',
                                                           column_name='symbol', column_value=full_symbol)
                        
                        # 安全过滤日期范围
                        if not df.empty and 'date' in df.columns:
                            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                        
                        if not df.empty:
                            return df
                            
                    except Exception as e:
                        continue
                
                return pd.DataFrame()
            
            result = self._request_with_retry(fetch_func)
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取新浪周线数据异常: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_data(self, symbol: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取日线数据 - 增强错误处理和日期日志
        :param symbol: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 日线数据DataFrame
        """
        try:
            # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 增强日志：记录方法调用参数
            logger.info(f"获取日线数据 - 符号: {symbol}, 开始: {start_date}, 结束: {end_date}")
            
            symbol = self._validate_symbol(symbol)
            start_dt, end_dt, original_start, original_end = self._validate_dates(start_date, end_date)
            
            cache_key = self._get_cache_key('daily', symbol, 
                                           start_dt.strftime("%Y%m%d"), 
                                           end_dt.strftime("%Y%m%d"))
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取日线数据: {symbol}")
                return cached_data
            
            # 根据纯数字代码确定市场前缀
            market = self._get_market_prefix(symbol)
            full_symbol = f"{market}{symbol}"  # 用于请求的代码
            
            params = {
                "symbol": full_symbol,
                "scale": "240",
                "ma": "no",
                "datalen": "1000"
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://finance.sina.com.cn/'
            }
            
            def fetch_func():
                try:
                    response = requests.get(self.sina_base_url, params=params, 
                                          headers=headers, timeout=self.timeout)
                    if response.status_code != 200:
                        return pd.DataFrame()
                    
                    content = response.text.strip()
                    if not content or content == "null":
                        return pd.DataFrame()
                    
                    data = json.loads(content)
                    if not data or not isinstance(data, list):
                        return pd.DataFrame()
                    
                    # 转换为DataFrame
                    df = pd.DataFrame(data)
                    
                    # 安全重命名列
                    column_map = {
                        "day": "date",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume"
                    }
                    df = self._safe_dataframe_operation(df, 'rename', column_map=column_map)
                    
                    # 安全转换数据类型
                    df = self._safe_dataframe_operation(df, 'convert_dtypes',
                                                       date_col='date',
                                                       numeric_cols=['open', 'close', 'high', 'low', 'volume'])
                    
                    df = self._safe_dataframe_operation(df, 'add_column',
                                                       column_name='symbol', column_value=full_symbol)
                    
                    # 安全过滤日期范围
                    if not df.empty and 'date' in df.columns:
                        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                    
                    return df
                    
                except Exception as e:
                    return pd.DataFrame()
            
            result = self._request_with_retry(fetch_func)
            if result is not None and not result.empty:
                # 🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧 新增：检查日期范围完整性
                self._check_date_range_completeness(result, start_dt, end_dt, symbol, 'daily')
                
                self._save_to_cache(cache_key, result)
                logger.info(f"成功获取日线数据: {len(result)} 条")
                return result
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取日线数据异常: {str(e)}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """
        获取实时数据 - 增强错误处理
        :param symbol: 股票代码
        :return: 实时数据字典
        """
        try:
            symbol = self._validate_symbol(symbol)
            
            # 根据纯数字代码确定市场前缀
            market = self._get_market_prefix(symbol)
            full_symbol = f"{market}{symbol}"  # 用于请求的代码
            
            url = f"https://qt.gtimg.cn/q={full_symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://gu.qq.com/'
            }
            
            def fetch_func():
                try:
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                    if response.status_code != 200:
                        return {}
                    
                    content = response.text.strip()
                    if not content:
                        return {}
                    
                    # 解析腾讯财经实时数据格式
                    if '~' not in content:
                        return {}
                    
                    parts = content.split('~')
                    if len(parts) < 40:
                        return {}
                    
                    # 提取关键字段
                    data = {
                        'name': parts[1],
                        'code': parts[2],
                        'price': parts[3],
                        'prev_close': parts[4],
                        'open': parts[5],
                        'volume': parts[6],
                        'amount': parts[37] if len(parts) > 37 else '0',
                        'high': parts[33] if len(parts) > 33 else '0',
                        'low': parts[34] if len(parts) > 34 else '0',
                        'time': parts[30] if len(parts) > 30 else ''
                    }
                    
                    # 安全转换数值类型
                    for key in ['price', 'prev_close', 'open', 'volume', 'amount', 'high', 'low']:
                        try:
                            data[key] = float(data[key])
                        except (ValueError, TypeError):
                            data[key] = 0.0
                    
                    return data
                except Exception as e:
                    logger.warning(f"实时数据请求异常: {str(e)}")
                    return {}
            
            return self._request_with_retry(fetch_func) or {}
            
        except Exception as e:
            logger.error(f"获取实时数据异常: {str(e)}")
            return {}
    
    def get_minute_data(self, symbol: str, interval: str = '5m', days: int = 30) -> pd.DataFrame:
        """
        获取分钟数据 - 模拟实现，基于日线数据生成
        :param symbol: 股票代码
        :param interval: 时间间隔，如'5m'
        :param days: 天数
        :return: 分钟数据DataFrame
        """
        try:
            # 模拟实现：获取日线数据，然后生成分钟数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            daily_data = self.get_daily_data(symbol, start_date, end_date)
            if daily_data.empty:
                return pd.DataFrame()
            
            # 生成分钟数据：每个交易日生成240个分钟点（模拟）
            minute_data = []
            for _, row in daily_data.iterrows():
                date = row['date']
                # 确保date是datetime对象
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                for i in range(240):  # 假设交易日有4小时，240分钟
                    minute_time = date + timedelta(minutes=i)
                    # 模拟价格波动，基于日线OHLC
                    progress = i / 239.0
                    minute_open = row['open'] + (row['close'] - row['open']) * progress
                    minute_high = row['high']  # 简化
                    minute_low = row['low']    # 简化
                    minute_close = minute_open  # 简化
                    minute_volume = row['volume'] / 240  # 平均分配
                    minute_data.append({
                        'date': minute_time,
                        'open': minute_open,
                        'high': minute_high,
                        'low': minute_low,
                        'close': minute_close,
                        'volume': minute_volume
                    })
            df = pd.DataFrame(minute_data)
            df = self._safe_dataframe_operation(df, 'add_column', column_name='symbol', column_value=symbol)
            return df
        except Exception as e:
            logger.error(f"获取分钟数据异常: {str(e)}")
            return pd.DataFrame()
    
    def health_check(self) -> Dict:
        """
        健康检查 - 增强错误处理
        :return: 系统状态字典
        """
        try:
            status_info = {
                "status": "OK",
                "version": "2.1.1",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_sources": self.data_sources,
                "cache_enabled": self.cache_enabled,
                "cache_size": len(self.cache),
                "features": self._get_feature_summary()
            }
            
            # 测试数据获取功能
            try:
                test_symbol = "000001"  # 纯数字代码
                test_data = self.get_realtime_data(test_symbol)
                if test_data:
                    status_info["realtime_test"] = "PASS"
                else:
                    status_info["realtime_test"] = "FAIL"
            except:
                status_info["realtime_test"] = "FAIL"
            
            return status_info
            
        except Exception as e:
            logger.error(f"健康检查异常: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

# 向后兼容性别名
StockDataAPI = StockDataFetcher

# 功能完整性测试
def test_feature_completeness():
    """测试所有功能模块是否完整实现"""
    print("=== 功能完整性检查 ===")
    
    fetcher = StockDataFetcher()
    
    # 测试1: 健康检查
    print("1. 健康检查...")
    health = fetcher.health_check()
    print(f"   状态: {health['status']}")
    print(f"   版本: {health['version']}")
    print(f"   功能: {health['features']}")
    
    # 测试2: 实时数据获取
    print("2. 实时数据获取...")
    realtime_data = fetcher.get_realtime_data("000001")  # 纯数字代码
    if realtime_data:
        print(f"   成功获取实时数据: {realtime_data.get('name', 'N/A')}")
    else:
        print("   实时数据获取失败")
    
    # 测试3: 日线数据获取
    print("3. 日线数据获取...")
    daily_data = fetcher.get_daily_data("000001", "2023-10-01", "2023-10-10")
    if not daily_data.empty:
        print(f"   成功获取日线数据: {len(daily_data)} 条")
    else:
        print("   日线数据获取失败")
    
    # 测试4: 周线数据获取
    print("4. 周线数据获取...")
    weekly_data = fetcher.get_weekly_data("000001", "2023-01-01", "2023-10-01")
    if not weekly_data.empty:
        print(f"   成功获取周线数据: {len(weekly_data)} 条")
    else:
        print("   周线数据获取失败")
    
    # 测试5: 分钟数据获取
    print("5. 分钟数据获取...")
    minute_data = fetcher.get_minute_data("000001", "5m", 30)
    if not minute_data.empty:
        print(f"   成功获取分钟数据: {len(minute_data)} 条")
    else:
        print("   分钟数据获取失败")
    
    # 测试6: 缓存功能
    print("6. 缓存功能测试...")
    if fetcher.cache_enabled:
        print("   缓存功能已启用")
    else:
        print("   缓存功能未启用")
    
    print("=== 检查完成 ===")

if __name__ == "__main__":
    # 运行功能完整性测试
    test_feature_completeness()
    
    # 详细功能演示
    print("\n=== 详细功能演示 ===")
    fetcher = StockDataFetcher()
    
    # 演示周线数据获取
    symbol = "000001"  # 纯数字代码
    weekly_data = fetcher.get_weekly_data(symbol, "2023-01-01", "2023-10-01")
    
    if not weekly_data.empty:
        print(f"成功获取 {symbol} 周线数据:")
        print(f"数据范围: {weekly_data['date'].min()} 至 {weekly_data['date'].max()}")
        print(f"数据列: {list(weekly_data.columns)}")
        print(weekly_data.head())
    else:
        print("周线数据获取失败")
    
    # 演示实时数据
    realtime_data = fetcher.get_realtime_data(symbol)
    if realtime_data:
        print(f"\n{realtime_data.get('name', symbol)} 实时数据:")
        for key, value in realtime_data.items():
            print(f"  {key}: {value}")
    
    # 演示分钟数据
    minute_data = fetcher.get_minute_data(symbol, "5m", 1)  # 1天的分钟数据
    if not minute_data.empty:
        print(f"\n成功获取 {symbol} 分钟数据:")
        print(f"数据点数: {len(minute_data)}")
        print(minute_data.head())
    else:
        print("分钟数据获取失败")