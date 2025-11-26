#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""股票数据获取器 - 精简版（保留核心功能）"""

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
from typing import Optional, Dict, List, Union, Any, Callable, Tuple
from unittest.mock import Mock

# 项目路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 配置导入（兼容无配置文件场景）
try:
    from src.config import get_data_fetcher_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("未导入配置模块，使用默认配置")

# 日志配置（精简）
logger = logging.getLogger('StockDataFetcher')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class DataFetchError(Exception):
    """数据获取异常"""
    pass

class StockDataFetcher:
    """高效股票数据获取器（精简版）"""
    
    def __init__(self, max_retries: int = None, timeout: int = None,
                 http_client: Optional[Callable] = None):
        """初始化：依赖注入+配置加载"""
        self.http_client = http_client or requests.get
        
        # 加载配置或默认值
        if CONFIG_AVAILABLE:
            try:
                config = get_data_fetcher_config()
                self.max_retries = max_retries or config.get('max_retries', 3)
                self.timeout = timeout or config.get('timeout', 10)
                self.data_sources = config.get('data_sources', ['tencent', 'sina'])
                self.cache_enabled = config.get('cache_enabled', True)
                self.cache_ttl = config.get('cache_ttl', 300)
                
                # 新浪配置
                sina_cfg = config.get('sina', {})
                self.sina_base_url = sina_cfg.get('base_url', "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData")
                
                # 腾讯配置
                tencent_cfg = config.get('tencent', {})
                self.tencent_enabled = tencent_cfg.get('enabled', True)
                self.tencent_weekly_url = tencent_cfg.get('weekly_url', "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get")
            except Exception as e:
                logger.warning(f"配置加载失败，使用默认值: {str(e)}")
                self._set_default_config()
        else:
            self._set_default_config()
        
        # 缓存初始化
        self.cache = {}
        self.cache_timestamps = {}
        logger.info("数据获取器初始化完成")
    
    def _set_default_config(self):
        """默认配置"""
        self.max_retries = 3
        self.timeout = 10
        self.data_sources = ['tencent', 'sina']
        self.cache_enabled = True
        self.cache_ttl = 300
        
        self.sina_base_url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        self.tencent_enabled = True
        self.tencent_weekly_url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    
    def _convert_date_format(self, date_str: str) -> str:
        """日期格式统一为YYYYMMDD"""
        if not date_str or not isinstance(date_str, str):
            return date_str
        return date_str.replace('-', '') if '-' in date_str else date_str
    
    def _validate_symbol(self, symbol: str) -> str:
        """验证股票代码（支持6位数字A股代码和字母代码如TQQQ）"""
        if symbol is None:
            raise DataFetchError("股票代码不能为空")
        
        symbol_str = str(symbol)
        # 防止传入DataFrame等无效类型
        if len(symbol_str) > 100 or any(marker in symbol_str for marker in ['DataFrame', 'Series', 'open', 'close']):
            raise DataFetchError(f"无效股票代码类型: {symbol_str[:50]}...")
        
        # 对于纯数字代码，验证是否为6位数字（A股代码）
        if symbol_str.isdigit():
            if not re.match(r'^\d{6}$', symbol_str):
                raise DataFetchError(f"A股股票代码必须是6位数字: {symbol}")
            return symbol_str
        
        # 对于包含字母的代码（如TQQQ），直接返回原始代码
        # 去除可能的前缀如sh/sz，确保返回干净的代码
        return symbol_str.replace('sh', '').replace('sz', '')
    
    def _get_market_prefix(self, symbol: str) -> str:
        """获取市场前缀（sh/sz/美股特殊处理）"""
        # 如果是纯数字代码，使用原来的逻辑（A股）
        if symbol.isdigit():
            return "sh" if symbol.startswith(("6", "5", "9")) else "sz"
        # 对于非数字代码（如TQQQ），暂时返回空字符串，后续在fetch方法中特殊处理
        return ""
    
    def _parse_datetime(self, date_str: str) -> datetime:
        """解析多种日期时间格式"""
        for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析日期格式: {date_str}")
    
    def _validate_dates(self, start_date: Optional[str], end_date: Optional[str], 
                        need_time: bool = False) -> Tuple[datetime, datetime, Optional[str], Optional[str]]:
        """验证日期范围，支持带时间的格式"""
        original_start, original_end = start_date, end_date
        
        # 处理默认值
        now = datetime.now()
        end_date = end_date if end_date else (now.strftime("%Y-%m-%d %H:%M:%S") if need_time else now.strftime("%Y%m%d"))
        start_timedelta = timedelta(days=365) if not need_time else timedelta(days=7)
        start_date = start_date if start_date else ((now - start_timedelta).strftime("%Y-%m-%d %H:%M:%S") 
                                                  if need_time else (now - start_timedelta).strftime("%Y%m%d"))
        
        # 解析日期
        try:
            start_dt = self._parse_datetime(start_date)
            end_dt = self._parse_datetime(end_date)
            
            # 如果不需要时间信息，将开始时间设为00:00:00，结束时间设为23:59:59
            if not need_time:
                start_dt = start_dt.replace(hour=0, minute=0, second=0)
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            raise DataFetchError(f"日期格式错误，支持格式: YYYYMMDD, YYYY-MM-DD, {'YYYY-MM-DD HH:MM:SS' if need_time else ''}")
        
        if start_dt > end_dt:
            raise DataFetchError(f"开始日期{start_date}不能晚于结束日期{end_date}")
        return start_dt, end_dt, original_start, original_end
    
    def _get_cache_key(self, data_type: str, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        return f"{data_type}:{symbol}:{start_date}:{end_date}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        if not self.cache_enabled or cache_key not in self.cache:
            return None
        
        if time.time() - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl:
            return self.cache[cache_key].copy()
        else:
            # 缓存过期清理
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if self.cache_enabled:
            self.cache[cache_key] = data.copy()
            self.cache_timestamps[cache_key] = time.time()
    
    def _request_with_retry(self, url: str, params: dict = None, headers: dict = None) -> Optional[requests.Response]:
        """带重试的HTTP请求"""
        headers = headers or {}
        headers.setdefault('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    time.sleep(random.uniform(0.5, 2.0))
                
                response = self.http_client(url, params=params, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    return response
                logger.warning(f"请求失败，状态码: {response.status_code}")
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"所有重试失败: {str(e)}")
                else:
                    logger.warning(f"第{attempt}次请求失败: {str(e)}")
        return None
    
    def _safe_dataframe_process(self, df: pd.DataFrame, date_col: str = 'date', numeric_cols: List[str] = None) -> pd.DataFrame:
        """安全处理DataFrame（类型转换+列重命名）"""
        if df.empty:
            return df
        
        numeric_cols = numeric_cols or ['open', 'close', 'high', 'low', 'volume']
        
        # 先重命名day列为date（确保date列存在后再进行转换）
        if 'day' in df.columns and 'date' not in df.columns:
            df.rename(columns={'day': 'date'}, inplace=True)
        
        # 处理日期时间转换，支持多种格式
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        
        # 数值转换
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    
    def get_weekly_data(self, symbol: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
        """获取周线数据（核心接口）"""
        try:
            symbol = self._validate_symbol(symbol)
            start_dt, end_dt, orig_start, orig_end = self._validate_dates(start_date, end_date)
            market = self._get_market_prefix(symbol)
            full_symbol = f"{market}{symbol}"
            
            # 缓存检查
            cache_start = orig_start or start_dt.strftime('%Y-%m-%d')
            cache_end = orig_end or end_dt.strftime('%Y-%m-%d')
            cache_key = self._get_cache_key('weekly', symbol, cache_start, cache_end)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                actual_start = cached_data['date'].min().strftime('%Y-%m-%d') if not cached_data.empty else None
                actual_end = cached_data['date'].max().strftime('%Y-%m-%d') if not cached_data.empty else None
                return (cached_data, actual_start, actual_end)
            
            # 尝试数据源
            for source in self.data_sources:
                try:
                    if source == 'tencent' and self.tencent_enabled:
                        df = self._fetch_tencent_weekly(full_symbol, start_dt, end_dt)
                    elif source == 'sina':
                        df = self._fetch_sina_weekly(full_symbol, start_dt, end_dt)
                    else:
                        continue
                    
                    if not df.empty:
                        # 日期过滤
                        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                        if df.empty:
                            continue
                        
                        # 缓存并返回
                        self._save_to_cache(cache_key, df)
                        actual_start = df['date'].min().strftime('%Y-%m-%d')
                        actual_end = df['date'].max().strftime('%Y-%m-%d')
                        logger.info(f"从{source}获取周线数据: {symbol} ({actual_start}~{actual_end})")
                        return (df, actual_start, actual_end)
                except Exception as e:
                    logger.warning(f"{source}数据源失败: {str(e)}")
                    continue
            
            logger.error("所有数据源均失败")
            return (pd.DataFrame(), None, None)
        except DataFetchError as e:
            logger.error(f"参数错误: {str(e)}")
            return (pd.DataFrame(), None, None)
        except Exception as e:
            logger.error(f"获取周线数据异常: {str(e)}")
            return (pd.DataFrame(), None, None)
    
    def _fetch_tencent_weekly(self, full_symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """腾讯周线数据获取"""
        params = {
            "_var": "kline_week",
            "param": f"{full_symbol},week,{start_dt.strftime('%Y-%m-%d')},{end_dt.strftime('%Y-%m-%d')},500,qfq",
            "r": f"0.{int(time.time()*1000)}"
        }
        
        response = self._request_with_retry(self.tencent_weekly_url, params=params)
        if not response:
            logger.warning(f"腾讯请求失败，无响应: {full_symbol}")
            return pd.DataFrame()
        
        content = response.text.strip().lstrip('kline_week=').rstrip(';')
        try:
            data = json.loads(content)
            if data.get('code') != 0:
                logger.warning(f"腾讯返回错误代码: {data.get('code')}, 消息: {data.get('msg')}")
                return pd.DataFrame()
            
            if full_symbol not in data.get('data', {}):
                logger.warning(f"腾讯返回数据中不存在 {full_symbol}")
                return pd.DataFrame()
            
            weekly_data = data['data'][full_symbol].get('qfqweek') or data['data'][full_symbol].get('week')
            if not weekly_data or not isinstance(weekly_data, list) or len(weekly_data) == 0:
                logger.warning(f"腾讯返回空数据或格式错误: {full_symbol}")
                return pd.DataFrame()
            
            # 尝试创建DataFrame，直接捕获可能的列数不匹配错误
            try:
                # 优先尝试6列格式
                df = pd.DataFrame(weekly_data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                logger.debug(f"腾讯数据使用6列格式解析成功: {full_symbol}")
            except ValueError:
                try:
                    # 如果6列失败，尝试7列格式
                    df = pd.DataFrame(weekly_data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'extra'])
                    # 只保留需要的列
                    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
                    logger.debug(f"腾讯数据使用7列格式解析成功: {full_symbol}")
                except Exception as inner_e:
                    logger.error(f"腾讯数据创建DataFrame失败: {str(inner_e)}")
                    return pd.DataFrame()
            
            # 安全处理数据框
            return self._safe_dataframe_process(df)
        except json.JSONDecodeError as json_e:
            logger.error(f"腾讯数据JSON解析失败: {str(json_e)}, 内容: {content[:100]}...")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"腾讯数据处理异常: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_sina_weekly(self, full_symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """新浪周线数据获取（兼容scale=week/7）"""
        for scale in ["week", "7"]:
            params = {"symbol": full_symbol, "scale": scale, "ma": "no", "datalen": "500"}
            response = self._request_with_retry(self.sina_base_url, params=params)
            if not response:
                continue
            
            content = response.text.strip()
            if not content or content == "null":
                continue
            
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    continue
                df = pd.DataFrame(data)
                df = self._safe_dataframe_process(df)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"新浪周线数据解析失败: {str(e)}")
                continue
        return pd.DataFrame()
    
    def get_daily_data(self, symbol: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None, force_refresh: bool = False, 
                      days: Optional[int] = None) -> pd.DataFrame:
        """获取日线数据"""
        try:
            symbol = self._validate_symbol(symbol)
            # 处理days参数
            if days is not None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_dt = datetime.now() - timedelta(days=days)
                start_date = start_dt.strftime('%Y-%m-%d')
            start_dt, end_dt, _, _ = self._validate_dates(start_date, end_date)
            full_symbol = f"{self._get_market_prefix(symbol)}{symbol}"
            
            # 缓存检查 - 如果end_date是最近7天内或强制刷新，则不使用缓存
            cache_key = self._get_cache_key('daily', symbol, start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
            cached_data = self._get_from_cache(cache_key)
            
            # 检查是否需要强制刷新缓存（最近7天内的数据或用户要求强制刷新）
            should_refresh = force_refresh or (datetime.now() - end_dt).days < 7
            if cached_data is not None and not should_refresh:
                return cached_data
            
            # 新浪接口请求
            params = {"symbol": full_symbol, "scale": "240", "ma": "no", "datalen": "1000"}
            response = self._request_with_retry(self.sina_base_url, params=params)
            if not response:
                return pd.DataFrame()
            
            # 数据处理
            try:
                data = json.loads(response.text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"日线数据JSON解析失败: {str(e)}")
                return pd.DataFrame()
                
            if not isinstance(data, list):
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df = self._safe_dataframe_process(df)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            if not df.empty:
                self._save_to_cache(cache_key, df)
            return df
        except Exception as e:
            logger.error(f"获取日线数据异常: {str(e)}")
            return pd.DataFrame()
    
    def get_hourly_data(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取小时线数据（scale=60）"""
        try:
            symbol = self._validate_symbol(symbol)
            # 小时线需要时间信息，所以need_time=True
            start_dt, end_dt, _, _ = self._validate_dates(start_date, end_date, need_time=True)
            full_symbol = f"{self._get_market_prefix(symbol)}{symbol}"
            
            # 缓存检查
            cache_key = self._get_cache_key('hourly', symbol, 
                                           start_dt.strftime("%Y%m%d%H%M"), 
                                           end_dt.strftime("%Y%m%d%H%M"))
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # 新浪接口请求（60分钟=小时线）
            params = {"symbol": full_symbol, "scale": "60", "ma": "no", "datalen": "1000"}
            response = self._request_with_retry(self.sina_base_url, params=params)
            if not response:
                return pd.DataFrame()
            
            # 数据处理
            try:
                data = json.loads(response.text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"小时线数据JSON解析失败: {str(e)}")
                return pd.DataFrame()
                
            if not isinstance(data, list):
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df = self._safe_dataframe_process(df)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            if not df.empty:
                self._save_to_cache(cache_key, df)
                logger.info(f"获取小时线数据: {symbol} ({len(df)}条)")
            return df
        except Exception as e:
            logger.error(f"获取小时线数据异常: {str(e)}")
            return pd.DataFrame()
    
    def get_minute_data(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, interval: int = 5) -> pd.DataFrame:
        """获取分钟数据（5/15/30分钟）"""
        try:
            valid_intervals = [5, 15, 30]
            if interval not in valid_intervals:
                raise DataFetchError(f"支持的分钟间隔: {valid_intervals}")
            
            symbol = self._validate_symbol(symbol)
            # 分钟线需要时间信息，所以need_time=True
            start_dt, end_dt, _, _ = self._validate_dates(start_date, end_date, need_time=True)
            full_symbol = f"{self._get_market_prefix(symbol)}{symbol}"
            
            # 缓存检查
            cache_key = self._get_cache_key(f"minute_{interval}", symbol, 
                                           start_dt.strftime("%Y%m%d%H%M"), 
                                           end_dt.strftime("%Y%m%d%H%M"))
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # 新浪接口请求
            params = {"symbol": full_symbol, "scale": str(interval), "ma": "no", "datalen": "10000"}
            response = self._request_with_retry(self.sina_base_url, params=params)
            if not response:
                return pd.DataFrame()
            
            # 数据处理
            try:
                data = json.loads(response.text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"分钟线数据JSON解析失败: {str(e)}")
                return pd.DataFrame()
                
            if not isinstance(data, list):
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df = self._safe_dataframe_process(df)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            df['interval'] = interval
            
            if not df.empty:
                self._save_to_cache(cache_key, df)
            return df
        except DataFetchError as e:
            logger.error(f"参数错误: {str(e)}")
        except Exception as e:
            logger.error(f"获取分钟数据异常: {str(e)}")
        return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        try:
            symbol = self._validate_symbol(symbol)
            full_symbol = f"{self._get_market_prefix(symbol)}{symbol}"
            url = f"http://hq.sinajs.cn/list={full_symbol}"
            
            # 缓存检查
            cache_key = f"realtime:{full_symbol}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data
            
            # 请求实时数据
            response = self._request_with_retry(url)
            if not response:
                return {}
            
            # 解析新浪实时行情格式
            content = response.text.strip()
            match = re.match(r'var hq_str_[\w\d]+="([^"]+)"', content)
            if not match:
                return {}
            
            data_list = match.group(1).split(',')
            if len(data_list) < 32:
                return {}
            
            # 构造返回结果
            realtime_data = {
                'symbol': full_symbol, 'name': data_list[0],
                'open': float(data_list[1]) if data_list[1] else None,
                'pre_close': float(data_list[2]) if data_list[2] else None,
                'price': float(data_list[3]) if data_list[3] else None,
                'high': float(data_list[4]) if data_list[4] else None,
                'low': float(data_list[5]) if data_list[5] else None,
                'volume': float(data_list[8]) if data_list[8] else None,
                'amount': float(data_list[9]) if data_list[9] else None,
                'date': data_list[30], 'time': data_list[31],
                'timestamp': datetime.now().timestamp()
            }
            
            # 实时数据缓存30秒
            self.cache[cache_key] = realtime_data
            self.cache_timestamps[cache_key] = time.time()
            return realtime_data
        except Exception as e:
            logger.error(f"获取实时数据异常: {str(e)}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'sources': {},
            'errors': []
        }
        
        # 测试新浪
        try:
            response = self._request_with_retry(self.sina_base_url, params={'symbol': 'sh600000', 'scale': '5', 'ma': 'no', 'datalen': '10'})
            result['sources']['sina'] = 'healthy' if response else 'unhealthy'
        except Exception as e:
            result['sources']['sina'] = 'unhealthy'
            result['errors'].append(f"sina: {str(e)}")
        
        # 测试腾讯
        if self.tencent_enabled:
            try:
                params = {"_var": "kline_day", "param": "sh600000,day,,,10,qfq", "r": f"0.{int(time.time()*1000)}"}
                response = self._request_with_retry(self.tencent_weekly_url, params=params)
                result['sources']['tencent'] = 'healthy' if response else 'unhealthy'
            except Exception as e:
                result['sources']['tencent'] = 'unhealthy'
                result['errors'].append(f"tencent: {str(e)}")
        
        # 状态判断
        if any(s == 'unhealthy' for s in result['sources'].values()) or result['errors']:
            result['status'] = 'degraded' if result['sources'] else 'unhealthy'
        return result


# ------------------------------ 精简单元测试 ------------------------------
import unittest

class TestStockDataFetcher(unittest.TestCase):
    """核心功能单元测试（精简版）"""
    
    def setUp(self):
        """初始化测试环境"""
        self.mock_http = Mock()
        self.mock_response = Mock(status_code=200)
        self.mock_http.return_value = self.mock_response
        self.fetcher = StockDataFetcher(max_retries=1, timeout=5, http_client=self.mock_http)
        self.fetcher.cache_enabled = False  # 禁用缓存便于测试
    
    def test_symbol_validation(self):
        """测试股票代码验证"""
        # 有效代码
        valid_symbols = ['600000', 'sh600000', '000001', '300001']
        for s in valid_symbols:
            self.assertEqual(len(self.fetcher._validate_symbol(s)), 6)
        
        # 无效代码
        invalid_symbols = ['12345', 'sh6000', None, pd.DataFrame(), 'abc123']
        for s in invalid_symbols:
            with self.assertRaises(DataFetchError):
                self.fetcher._validate_symbol(s)
    
    def test_date_validation(self):
        """测试日期验证"""
        # 有效日期
        start, end, _, _ = self.fetcher._validate_dates('2023-01-01', '2024-01-01')
        self.assertLessEqual(start, end)
        
        # 带时间的日期验证
        start, end, _, _ = self.fetcher._validate_dates('2023-01-01 09:30', '2023-01-01 15:00', need_time=True)
        self.assertLessEqual(start, end)
        
        # 无效日期
        with self.assertRaises(DataFetchError):
            self.fetcher._validate_dates('20240101', '20230101')  # 开始>结束
        with self.assertRaises(DataFetchError):
            self.fetcher._validate_dates('20231301', '20240101')  # 无效月份
    
    def test_weekly_data_fetch(self):
        """测试周线数据获取"""
        mock_data = [{"day": "2023-10-09", "open": "10.0", "high": "10.5", "low": "9.8", "close": "10.2", "volume": "1000000"}]
        self.mock_response.text = json.dumps(mock_data)
        
        df, start, end = self.fetcher.get_weekly_data('600000', '20231001', '20231030')
        self.assertFalse(df.empty)
        self.assertEqual(start, '2023-10-09')
        self.assertEqual(end, '2023-10-09')
    
    def test_hourly_data_fetch(self):
        """测试小时线数据获取"""
        mock_data = [{"day": "2023-10-09 10:30:00", "open": "10.0", "close": "10.2", "high": "10.3", "low": "9.9", "volume": "500000"}]
        self.mock_response.text = json.dumps(mock_data)
        
        df = self.fetcher.get_hourly_data('600000', '2023-10-09 09:00', '2023-10-09 15:00')
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]['date'].strftime('%Y-%m-%d %H:%M'), '2023-10-09 10:30')
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        self.fetcher.cache_enabled = True
        test_df = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'open': [10.0]})
        cache_key = self.fetcher._get_cache_key('test', '600000', '20230101', '20230101')
        
        # 保存缓存
        self.fetcher._save_to_cache(cache_key, test_df)
        # 获取缓存
        cached_df = self.fetcher._get_from_cache(cache_key)
        self.assertFalse(cached_df.empty)
    
    def test_health_check(self):
        """测试健康检查"""
        result = self.fetcher.health_check()
        self.assertIn('status', result)
        self.assertIn('sources', result)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=1)