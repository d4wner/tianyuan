#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据获取和处理模块 - 支持日线和周线数据"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入依赖
from src.config import DATA_PATHS
from src.data_fetcher import StockDataFetcher as StockDataAPI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('DataProcessor')

class DataProcessor:
    """数据获取和处理类"""
    
    # 列名映射，标准化不同数据源的列名
    COLUMN_MAPPING = {
        'date': 'datetime',
        'datetime': 'datetime',
        'time': 'datetime',
        'close': 'close',
        'Close': 'close',
        'c': 'close',
        'open': 'open',
        'Open': 'open',
        'o': 'open',
        'high': 'high',
        'High': 'high',
        'h': 'high',
        'low': 'low',
        'Low': 'low',
        'l': 'low',
        'volume': 'volume',
        'Volume': 'volume',
        'v': 'volume',
        'amount': 'amount',
        'Amount': 'amount',
        'a': 'amount'
    }
    
    def __init__(self, config: Dict[str, any] = None):
        """初始化数据处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.api = StockDataAPI(self.config.get('data_fetcher', {}))
        self._data_cache = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        for path in DATA_PATHS.values():
            os.makedirs(path, exist_ok=True)
        logger.info("所有数据目录已创建/验证")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        # 先将所有列名转换为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 应用列名映射
        new_columns = {}
        for old_col, new_col in self.COLUMN_MAPPING.items():
            if old_col in df.columns:
                new_columns[old_col] = new_col
        
        df = df.rename(columns=new_columns)
        
        # 确保必要的列存在
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"数据缺少必要列: {missing_columns}")
        
        return df
    
    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换日期时间列
        
        Args:
            df: 数据DataFrame
            
        Returns:
            处理后的DataFrame
        """
        if 'datetime' in df.columns:
            try:
                # 尝试不同的日期格式
                date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']
                converted = False
                
                for fmt in date_formats:
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'], format=fmt)
                        converted = True
                        break
                    except ValueError:
                        continue
                        
                # 如果上述格式都失败，尝试自动推断
                if not converted:
                    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
                
                logger.info("成功转换日期时间列")
                
            except Exception as e:
                logger.error(f"转换日期时间列失败: {str(e)}")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        if df.empty:
            return df
        
        # 复制数据以避免修改原始数据
        clean_df = df.copy()
        
        # 删除重复行
        initial_rows = len(clean_df)
        clean_df = clean_df.drop_duplicates()
        if len(clean_df) < initial_rows:
            logger.info(f"删除了 {initial_rows - len(clean_df)} 行重复数据")
        
        # 删除空值
        clean_df = clean_df.dropna(subset=['open', 'high', 'low', 'close'])
        if len(clean_df) < initial_rows:
            logger.info(f"删除了 {initial_rows - len(clean_df)} 行空值数据")
        
        # 检查并修正异常值
        # 检查价格是否为负数
        for col in ['open', 'high', 'low', 'close']:
            if col in clean_df.columns:
                negative_values = clean_df[clean_df[col] < 0]
                if not negative_values.empty:
                    logger.warning(f"发现负值 {col}: {len(negative_values)} 行")
                    clean_df = clean_df[clean_df[col] >= 0]
        
        # 检查价格的合理性：high >= max(open, close) >= min(open, close) >= low
        valid_data = (
            (clean_df['high'] >= clean_df[['open', 'close']].max(axis=1)) & 
            (clean_df['low'] <= clean_df[['open', 'close']].min(axis=1))
        )
        
        if not valid_data.all():
            invalid_count = len(clean_df) - valid_data.sum()
            logger.warning(f"发现 {invalid_count} 行数据的价格关系不合理，已修正")
            
            # 修正价格关系
            clean_df.loc[~valid_data, 'high'] = clean_df.loc[~valid_data, ['open', 'high', 'low', 'close']].max(axis=1)
            clean_df.loc[~valid_data, 'low'] = clean_df.loc[~valid_data, ['open', 'high', 'low', 'close']].min(axis=1)
        
        return clean_df
    
    def _sort_by_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期排序
        
        Args:
            df: 数据DataFrame
            
        Returns:
            排序后的DataFrame
        """
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
            # 设置日期为索引
            df = df.set_index('datetime')
            logger.info("数据已按日期排序")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """添加技术指标
        
        Args:
            df: 数据DataFrame
            timeframe: 时间周期 ('daily' 或 'weekly')
            
        Returns:
            添加指标后的DataFrame
        """
        if df.empty:
            return df
        
        # 复制数据以避免修改原始数据
        tech_df = df.copy()
        
        # 添加常用均线
        ma_periods = [5, 10, 20, 60]
        for period in ma_periods:
            tech_df[f'ma{period}'] = tech_df['close'].rolling(window=period).mean()
        
        # 计算成交量变化率
        if 'volume' in tech_df.columns:
            tech_df['volume_change'] = tech_df['volume'].pct_change()
            # 计算均量线
            tech_df['ma_volume_20'] = tech_df['volume'].rolling(window=20).mean()
        
        # 计算涨跌幅
        tech_df['change_pct'] = tech_df['close'].pct_change() * 100
        
        logger.info(f"已添加技术指标，周期: {timeframe}")
        return tech_df
    
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """生成缓存键
        
        Args:
            symbol: 股票代码
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            缓存键
        """
        return f"{symbol}_{timeframe}_{start_date}_{end_date}"
    
    def get_daily_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                      force_refresh: bool = False, preprocess: bool = True) -> pd.DataFrame:
        """获取日线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            force_refresh: 是否强制刷新数据
            preprocess: 是否预处理数据
            
        Returns:
            日线数据DataFrame
        """
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            # 默认获取90天数据
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        
        # 生成缓存键
        cache_key = self._get_cache_key(symbol, "daily", start_date, end_date)
        
        # 检查缓存
        if not force_refresh and cache_key in self._data_cache:
            logger.info(f"从缓存获取日线数据: {symbol}")
            return self._data_cache[cache_key].copy()
        
        try:
            logger.info(f"获取日线数据: {symbol}, {start_date} 至 {end_date}")
            # 从API获取数据
            df = self.api.get_daily_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh
            )
            
            if df.empty:
                logger.error("获取的日线数据为空")
                return pd.DataFrame()
            
            # 预处理数据
            if preprocess:
                df = self.preprocess_data(df, "daily")
            
            # 存入缓存
            self._data_cache[cache_key] = df.copy()
            logger.info(f"成功获取并处理{len(df)}条日线数据")
            
            return df
            
        except Exception as e:
            logger.error(f"获取日线数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_weekly_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                       force_refresh: bool = False, preprocess: bool = True) -> pd.DataFrame:
        """获取周线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            force_refresh: 是否强制刷新数据
            preprocess: 是否预处理数据
            
        Returns:
            周线数据DataFrame
        """
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            # 默认获取180天数据
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        
        # 生成缓存键
        cache_key = self._get_cache_key(symbol, "weekly", start_date, end_date)
        
        # 检查缓存
        if not force_refresh and cache_key in self._data_cache:
            logger.info(f"从缓存获取周线数据: {symbol}")
            return self._data_cache[cache_key].copy()
        
        try:
            logger.info(f"获取周线数据: {symbol}, {start_date} 至 {end_date}")
            # 从API获取数据
            df, actual_start, actual_end = self.api.get_weekly_data(
                symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.error("获取的周线数据为空")
                return pd.DataFrame()
            
            # 预处理数据
            if preprocess:
                df = self.preprocess_data(df, "weekly")
            
            # 存入缓存
            self._data_cache[cache_key] = df.copy()
            logger.info(f"成功获取并处理{len(df)}条周线数据")
            
            return df
            
        except Exception as e:
            logger.error(f"获取周线数据失败: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
        """预处理数据
        
        Args:
            df: 原始DataFrame
            timeframe: 时间周期 ('daily' 或 'weekly')
            
        Returns:
            预处理后的DataFrame
        """
        if df.empty:
            return df
        
        logger.info(f"开始预处理数据，周期: {timeframe}")
        
        # 标准化列名
        df = self._standardize_columns(df)
        
        # 转换日期时间
        df = self._convert_datetime(df)
        
        # 清洗数据
        df = self._clean_data(df)
        
        # 按日期排序
        df = self._sort_by_datetime(df)
        
        # 添加技术指标
        df = self._add_technical_indicators(df, timeframe)
        
        logger.info(f"数据预处理完成，周期: {timeframe}, 处理后数据行数: {len(df)}")
        return df
    
    def save_data_to_csv(self, df: pd.DataFrame, filename: str, directory: str = None) -> bool:
        """保存数据到CSV文件
        
        Args:
            df: 要保存的DataFrame
            filename: 文件名
            directory: 目录，如果为None则使用默认数据目录
            
        Returns:
            是否保存成功
        """
        if df.empty:
            logger.warning("无法保存空数据")
            return False
        
        # 确定保存路径
        if directory is None:
            directory = DATA_PATHS['raw_data']
        
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)
        
        # 构建完整路径
        filepath = os.path.join(directory, filename)
        
        try:
            df.to_csv(filepath, index=True, encoding='utf-8-sig')
            logger.info(f"数据已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            return False
    
    def load_data_from_csv(self, filepath: str) -> pd.DataFrame:
        """从CSV文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的DataFrame
        """
        try:
            logger.info(f"从文件加载数据: {filepath}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"成功加载{len(df)}行数据")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_military_etf_data(self, timeframe: str = "daily", days: int = None) -> pd.DataFrame:
        """获取军工ETF数据的便捷方法
        
        Args:
            timeframe: 时间周期 ('daily' 或 'weekly')
            days: 数据天数
            
        Returns:
            军工ETF数据
        """
        # 军工ETF的代码
        symbol = "sh512660"
        
        # 设置日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        if days is None:
            days = 90 if timeframe == "daily" else 180
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        # 获取数据
        if timeframe == "daily":
            return self.get_daily_data(symbol, start_date, end_date, preprocess=True)
        else:  # weekly
            return self.get_weekly_data(symbol, start_date, end_date, preprocess=True)
    
    def clear_cache(self):
        """清除数据缓存"""
        self._data_cache.clear()
        logger.info("数据缓存已清除")
    
    def calculate_ma_slope(self, df: pd.DataFrame, ma_column: str, periods: int = 5) -> pd.Series:
        """计算均线斜率
        
        Args:
            df: 数据DataFrame
            ma_column: 均线列名
            periods: 计算周期
            
        Returns:
            斜率Series
        """
        if ma_column not in df.columns or len(df) < periods + 1:
            return pd.Series(index=df.index, dtype='float64')
        
        # 计算均线的变化率
        ma_values = df[ma_column]
        slopes = (ma_values - ma_values.shift(periods)) / ma_values.shift(periods) * 100
        
        return slopes
    
    def detect_price_patterns(self, df: pd.DataFrame, pattern_type: str = "all") -> pd.DataFrame:
        """检测价格模式
        
        Args:
            df: 数据DataFrame
            pattern_type: 模式类型 ('all', 'fractal', 'support_resistance')
            
        Returns:
            添加了模式标记的DataFrame
        """
        if df.empty:
            return df
        
        pattern_df = df.copy()
        
        # 检测分型
        if pattern_type in ["all", "fractal"]:
            # 顶分型: high > high[-1] and high > high[1] and low > low[-1] and low > low[1]
            pattern_df['top_fractal'] = False
            for i in range(1, len(df) - 1):
                if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                    df.iloc[i]['high'] > df.iloc[i+1]['high'] and
                    df.iloc[i]['low'] > df.iloc[i-1]['low'] and
                    df.iloc[i]['low'] > df.iloc[i+1]['low']):
                    pattern_df.iloc[i, pattern_df.columns.get_loc('top_fractal')] = True
            
            # 底分型: low < low[-1] and low < low[1] and high < high[-1] and high < high[1]
            pattern_df['bottom_fractal'] = False
            for i in range(1, len(df) - 1):
                if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                    df.iloc[i]['low'] < df.iloc[i+1]['low'] and
                    df.iloc[i]['high'] < df.iloc[i-1]['high'] and
                    df.iloc[i]['high'] < df.iloc[i+1]['high']):
                    pattern_df.iloc[i, pattern_df.columns.get_loc('bottom_fractal')] = True
        
        logger.info(f"价格模式检测完成: {pattern_type}")
        return pattern_df

def main():
    """测试数据处理器"""
    # 创建数据处理器实例
    processor = DataProcessor()
    
    # 测试获取日线数据
    print("\n测试获取军工ETF日线数据:")
    daily_data = processor.get_military_etf_data(timeframe="daily", days=30)
    print(f"日线数据形状: {daily_data.shape}")
    print("日线数据前几行:")
    print(daily_data.head())
    
    # 测试获取周线数据
    print("\n测试获取军工ETF周线数据:")
    weekly_data = processor.get_military_etf_data(timeframe="weekly", days=100)
    print(f"周线数据形状: {weekly_data.shape}")
    print("周线数据前几行:")
    print(weekly_data.head())
    
    # 保存数据到文件
    if not daily_data.empty:
        filename = f"military_etf_daily_{datetime.now().strftime('%Y%m%d')}.csv"
        processor.save_data_to_csv(daily_data, filename)
    
    if not weekly_data.empty:
        filename = f"military_etf_weekly_{datetime.now().strftime('%Y%m%d')}.csv"
        processor.save_data_to_csv(weekly_data, filename)

if __name__ == "__main__":
    main()