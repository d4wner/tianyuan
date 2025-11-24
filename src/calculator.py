#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论核心计算器（完整版V2.2）
支持：分型识别、笔划分、线段划分、中枢构建、背离检测、交易信号生成
核心修复：1. 初始资金优先使用用户传入值 2. 补充json模块导入 3. 修复numpy类型JSON序列化问题
         4. 修复backtester初始资金传递问题 5. 统一信号强度为0-1区间（适配回测引擎）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ===================== 日志配置 =====================
logger = logging.getLogger('ChanlunCalculator')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ===================== 常量定义 =====================
# 分型相关常量
FRACTAL_DEFAULT_SENSITIVITY = 3  # 默认分型灵敏度（左右各3根K线）
FRACTAL_MIN_PRICE_DIFF = 0.001  # 分型高低点最小价格差（避免极小波动误判）

# 笔相关常量
PEN_DEFAULT_MIN_LENGTH = 5  # 笔最小K线数量
PEN_DEFAULT_PRICE_RATIO = 0.01  # 笔最小价格波动比例（避免横盘误判）

# 线段相关常量
SEGMENT_DEFAULT_MIN_LENGTH = 3  # 线段最小笔数量
SEGMENT_DEFAULT_BREAK_RATIO = 0.5  # 线段破坏最小比例（超过50%视为有效破坏）

# 中枢相关常量
CENTRAL_BANK_DEFAULT_MIN_LENGTH = 5  # 中枢最小K线数量
CENTRAL_BANK_DEFAULT_EXPAND_RATIO = 0.02  # 中枢扩展比例
CENTRAL_BANK_DEFAULT_OVERLAP_RATIO = 0.3  # 中枢重叠比例

# 背离相关常量
DIVERGENCE_DEFAULT_THRESHOLD = 0.015  # 背离最小阈值
DIVERGENCE_DEFAULT_STRENGTH_LEVELS = 3  # 背离强度等级（1-3级）

# 信号相关常量
SIGNAL_DEFAULT_STRENGTH_THRESHOLD = 0.1  # 信号强度阈值（进一步降低阈值以便更容易生成信号）
SIGNAL_BUY = 'buy'
SIGNAL_SELL = 'sell'
SIGNAL_HOLD = 'hold'

# 市场状态常量
MARKET_BULL = 'bull'
MARKET_BEAR = 'bear'
MARKET_FLAT = 'flat'

# 初始资金默认值（仅作为fallback）
DEFAULT_INITIAL_CAPITAL = 100000.0

# ===================== 工具函数（新增：类型转换）=====================
def convert_numpy_to_python(obj):
    """
    将numpy数据类型转换为Python原生类型（支持JSON序列化）
    Args:
        obj: 输入对象（可能是numpy类型或Python类型）
    Returns:
        转换后的Python原生类型对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

# ===================== 缠论计算器主类 =====================
class ChanlunCalculator:
    """
    缠论核心计算器（完整版）
    核心功能：
    1. 分型识别：顶分型/底分型自动识别（支持灵敏度配置）
    2. 笔划分：基于分型的笔自动划分（包含笔验证逻辑）
    3. 线段划分：基于笔的线段自动划分（支持破坏验证）
    4. 中枢构建：标准中枢/扩展中枢/奔走中枢识别
    5. 背离检测：价格与MACD/RSI的顶背离/底背离
    6. 信号生成：结合分型/笔/线段/中枢/背离的交易信号
    7. 止损计算：基于最近分型/中枢的动态止损价
    8. 市场状态判断：牛市/熊市/横盘
    """
    
    def __init__(self, config: Dict[str, Any], initial_capital: Optional[float] = None):
        """
        初始化缠论计算器
        Args:
            config: 配置字典，包含：
                - chanlun: 缠论参数配置
                - risk_management: 风险管理配置
                - initial_capital: 用户传入的初始资金（次优先级）
                - data_validation_enabled: 数据验证开关
                - min_data_points: 最小数据量要求
            initial_capital: 从backtester传入的初始资金（最高优先级）
        """
        self.config = config or {}
        self.chanlun_config = self.config.get('chanlun', {})
        self.risk_config = self.config.get('risk_management', {})
        
        # 核心修复1：初始资金优先级 - backtester传入 > config传入 > 配置文件 > 默认值
        self.initial_capital = self._get_initial_capital(initial_capital)
        
        self.data_validation_enabled = self.config.get('data_validation_enabled', True)
        self.min_data_points = self.config.get('min_data_points', 50)
        
        # 初始化缠论参数（从配置读取，无则用默认值）
        self._init_chanlun_params()
        
        # 初始化技术指标参数
        self.macd_fast = self.chanlun_config.get('macd_fast', 12)
        self.macd_slow = self.chanlun_config.get('macd_slow', 26)
        self.macd_signal = self.chanlun_config.get('macd_signal', 9)
        self.rsi_period = self.chanlun_config.get('rsi_period', 14)
        self.atr_period = self.chanlun_config.get('atr_period', 14)
        
        logger.info(f"缠论计算器初始化完成 | 初始资金：{self.initial_capital:.2f}元（优先级：backtester传入 > config > 配置文件 > 默认值）")
        logger.info(f"缠论核心参数：灵敏度={self.fractal_sensitivity} | 笔最小长度={self.pen_min_length} | 中枢最小长度={self.central_bank_min_length}")

    def _get_initial_capital(self, external_capital: Optional[float] = None) -> float:
        """
        获取初始资金（核心修复1：严格按优先级处理）
        优先级顺序：
        1. backtester直接传入的initial_capital（最高优先级）
        2. config字典中直接传入的initial_capital（次优先级）
        3. 配置文件中risk_management.initial_capital（第三优先级）
        4. 默认值DEFAULT_INITIAL_CAPITAL（最低优先级）
        """
        # 1. 最优先使用backtester传入的初始资金
        if external_capital is not None and external_capital > 0:
            logger.info(f"✅ 使用backtester传入的初始资金：{external_capital:.2f}元")
            return external_capital
        
        # 2. 次优先使用config中直接传入的初始资金
        if 'initial_capital' in self.config and self.config['initial_capital'] > 0:
            user_capital = self.config['initial_capital']
            logger.info(f"✅ 使用config传入的初始资金：{user_capital:.2f}元")
            return user_capital
        
        # 3. 第三优先级使用配置文件中的初始资金
        config_capital = self.risk_config.get('initial_capital', 0.0)
        if config_capital > 0:
            logger.info(f"✅ 使用配置文件中的初始资金：{config_capital:.2f}元")
            return config_capital
        
        # 4. 最后使用默认值（仅当以上都未指定时）
        logger.warning(f"⚠️  未指定初始资金，使用默认值：{DEFAULT_INITIAL_CAPITAL:.2f}元")
        return DEFAULT_INITIAL_CAPITAL

    def _init_chanlun_params(self):
        """初始化缠论核心参数（原有完整逻辑，无改动）"""
        # 分型参数
        self.fractal_sensitivity = self.chanlun_config.get('fractal_sensitivity', FRACTAL_DEFAULT_SENSITIVITY)
        self.fractal_min_price_diff = self.chanlun_config.get('fractal_min_price_diff', FRACTAL_MIN_PRICE_DIFF)
        
        # 笔参数
        self.pen_min_length = self.chanlun_config.get('pen_min_length', PEN_DEFAULT_MIN_LENGTH)
        self.pen_min_price_ratio = self.chanlun_config.get('pen_min_price_ratio', PEN_DEFAULT_PRICE_RATIO)
        
        # 线段参数
        self.segment_min_length = self.chanlun_config.get('segment_min_length', SEGMENT_DEFAULT_MIN_LENGTH)
        self.segment_break_ratio = self.chanlun_config.get('segment_break_ratio', SEGMENT_DEFAULT_BREAK_RATIO)
        
        # 中枢参数
        self.central_bank_min_length = self.chanlun_config.get('central_bank_min_length', CENTRAL_BANK_DEFAULT_MIN_LENGTH)
        self.central_bank_expand_ratio = self.chanlun_config.get('central_bank_expand_ratio', CENTRAL_BANK_DEFAULT_EXPAND_RATIO)
        self.central_bank_overlap_ratio = self.chanlun_config.get('central_bank_overlap_ratio', CENTRAL_BANK_DEFAULT_OVERLAP_RATIO)
        
        # 背离参数
        self.divergence_threshold = self.chanlun_config.get('divergence_threshold', DIVERGENCE_DEFAULT_THRESHOLD)
        self.divergence_strength_levels = self.chanlun_config.get('divergence_strength_levels', DIVERGENCE_DEFAULT_STRENGTH_LEVELS)
        
        # 信号参数
        self.signal_strength_threshold = self.chanlun_config.get('signal_strength_threshold', SIGNAL_DEFAULT_STRENGTH_THRESHOLD)

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """验证输入数据合法性（原有完整逻辑，无改动）"""
        if self.data_validation_enabled is False:
            logger.info("数据验证已禁用，跳过验证")
            return True
        
        logger.info("开始数据合法性验证...")
        
        # 1. 数据量验证
        if len(df) < self.min_data_points:
            logger.error(f"数据量不足：需至少{self.min_data_points}条，当前{len(df)}条")
            return False
        
        # 2. 必要列验证
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"缺失必要列：{missing_cols}")
            return False
        
        # 3. 数据类型验证
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"列{col}数据类型错误：需数值类型，当前{df[col].dtype}")
                return False
        
        # 4. 数据有效性验证
        if (df[numeric_cols] < 0).any().any():
            logger.error(f"数值列存在负数：{numeric_cols}")
            return False
        
        # 5. 日期列验证
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                logger.error(f"日期列格式错误：{str(e)}")
                return False
        
        # 6. 排序验证
        if not df['date'].is_monotonic_increasing:
            df = df.sort_values('date').reset_index(drop=True)
            logger.warning("数据已按日期重新排序并重置索引")
        
        logger.info("数据验证通过")
        return True

    def calculate_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别顶分型和底分型（原有完整逻辑，无改动）"""
        df = df.copy()
        df['top_fractal'] = False
        df['bottom_fractal'] = False
        df['fractal_price'] = np.nan  # 分型价格（顶分型取high，底分型取low）
        df['fractal_index'] = -1  # 分型在原始数据中的索引
        
        # 分型识别核心逻辑
        for i in range(self.fractal_sensitivity, len(df) - self.fractal_sensitivity):
            # 顶分型识别：中间K线高点是前后n根K线的最高点，且满足最小价格差
            top_condition = True
            for j in range(1, self.fractal_sensitivity + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i-j] or df['high'].iloc[i] <= df['high'].iloc[i+j]:
                    top_condition = False
                    break
            if top_condition:
                price_diff = df['high'].iloc[i] - df['high'].iloc[i-1:i+1].min()
                if price_diff > self.fractal_min_price_diff:
                    df.at[df.index[i], 'top_fractal'] = True
                    df.at[df.index[i], 'fractal_price'] = df['high'].iloc[i]
                    df.at[df.index[i], 'fractal_index'] = i
        
        # 底分型识别：中间K线低点是前后n根K线的最低点，且满足最小价格差
        for i in range(self.fractal_sensitivity, len(df) - self.fractal_sensitivity):
            bottom_condition = True
            for j in range(1, self.fractal_sensitivity + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i-j] or df['low'].iloc[i] >= df['low'].iloc[i+j]:
                    bottom_condition = False
                    break
            if bottom_condition:
                price_diff = df['low'].iloc[i-1:i+1].max() - df['low'].iloc[i]
                if price_diff > self.fractal_min_price_diff:
                    df.at[df.index[i], 'bottom_fractal'] = True
                    df.at[df.index[i], 'fractal_price'] = df['low'].iloc[i]
                    df.at[df.index[i], 'fractal_index'] = i
        
        # 统计分型数量
        top_count = df['top_fractal'].sum()
        bottom_count = df['bottom_fractal'].sum()
        logger.info(f"分型识别完成：顶分型{top_count}个 | 底分型{bottom_count}个")
        return df

    def calculate_pens(self, df: pd.DataFrame) -> pd.DataFrame:
        """划分笔（原有完整逻辑，无改动）"""
        df = df.copy()
        df['pen_type'] = None  # up（上升笔）/ down（下降笔）/ None
        df['pen_start'] = False  # 笔起点标记
        df['pen_end'] = False    # 笔终点标记
        df['pen_id'] = -1        # 笔编号（从0开始）
        df['pen_length_kline'] = 0  # 笔包含的K线数量
        df['pen_price_change'] = 0.0  # 笔的价格变化（终点-起点）
        df['pen_price_ratio'] = 0.0   # 笔的价格变化比例（变化量/起点价格）
        
        # 筛选有效分型（排除连续相同类型的分型）
        fractals = df[(df['top_fractal'] | df['bottom_fractal'])].copy()
        if len(fractals) < 2:
            logger.warning("有效分型数量不足2个，无法划分笔")
            return df
        
        # 去重连续相同类型的分型（保留最后一个）
        filtered_fractals = []
        prev_type = None
        for idx, row in fractals.iterrows():
            curr_type = 'top' if row['top_fractal'] else 'bottom'
            if curr_type != prev_type:
                filtered_fractals.append((idx, curr_type, row['fractal_price']))
                prev_type = curr_type
            else:
                # 连续相同类型，替换为最新的分型
                filtered_fractals[-1] = (idx, curr_type, row['fractal_price'])
        
        if len(filtered_fractals) < 2:
            logger.warning("去重后有效分型不足2个，无法划分笔")
            return df
        
        # 划分笔（需满足：顶-底交替、最小K线数量、最小价格波动）
        pen_id = 0
        for i in range(1, len(filtered_fractals)):
            start_idx, start_type, start_price = filtered_fractals[i-1]
            end_idx, end_type, end_price = filtered_fractals[i]
            
            # 验证笔的类型（必须顶底交替）
            if (start_type == 'bottom' and end_type == 'top'):
                pen_type = 'up'
            elif (start_type == 'top' and end_type == 'bottom'):
                pen_type = 'down'
            else:
                logger.debug(f"笔类型错误：{start_type}->>{end_type}，跳过")
                continue
            
            # 计算笔的K线数量
            kline_count = end_idx - start_idx + 1
            if kline_count < self.pen_min_length:
                logger.debug(f"笔K线数量不足：{kline_count}（最小{self.pen_min_length}），跳过")
                continue
            
            # 计算价格变化比例
            price_change = end_price - start_price
            price_ratio = abs(price_change) / start_price
            if price_ratio < self.pen_min_price_ratio:
                logger.debug(f"笔价格波动不足：{price_ratio:.4f}（最小{self.pen_min_price_ratio}），跳过")
                continue
            
            # 标记笔信息
            df.loc[start_idx:end_idx, 'pen_id'] = pen_id
            df.loc[start_idx, 'pen_start'] = True
            df.loc[end_idx, 'pen_end'] = True
            df.loc[start_idx:end_idx, 'pen_type'] = pen_type
            df.loc[start_idx:end_idx, 'pen_length_kline'] = kline_count
            df.loc[start_idx:end_idx, 'pen_price_change'] = price_change
            df.loc[start_idx:end_idx, 'pen_price_ratio'] = price_ratio
            
            logger.debug(f"划分笔{pen_id}：类型{pen_type} | 区间[{start_idx}:{end_idx}] | K线数{kline_count} | 波动{price_ratio:.4f}")
            pen_id += 1
        
        logger.info(f"笔划分完成：有效笔数量{pen_id}支")
        return df

    def calculate_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """划分线段（原有完整逻辑，无改动）"""
        df = df.copy()
        df['segment_type'] = None  # up（上升线段）/ down（下降线段）/ None
        df['segment_start'] = False  # 线段起点标记
        df['segment_end'] = False    # 线段终点标记
        df['segment_id'] = -1        # 线段编号（从0开始）
        df['segment_length_pen'] = 0 # 线段包含的笔数量
        df['segment_price_change'] = 0.0  # 线段价格变化
        df['segment_price_ratio'] = 0.0   # 线段价格变化比例
        
        # 筛选有效笔的终点（线段由笔的交替构成）
        pen_ends = df[df['pen_end']].copy()
        if len(pen_ends) < self.segment_min_length:
            logger.warning(f"有效笔数量不足{self.segment_min_length}支，无法划分线段")
            return df
        
        # 去重连续相同类型的笔（保留最后一个）
        filtered_pens = []
        prev_pen_type = None
        for idx, row in pen_ends.iterrows():
            curr_pen_type = row['pen_type']
            if curr_pen_type != prev_pen_type and curr_pen_type is not None:
                filtered_pens.append((idx, curr_pen_type, row['fractal_price']))
                prev_pen_type = curr_pen_type
            elif curr_pen_type == prev_pen_type and curr_pen_type is not None:
                filtered_pens[-1] = (idx, curr_pen_type, row['fractal_price'])
        
        if len(filtered_pens) < self.segment_min_length:
            logger.warning(f"去重后有效笔不足{self.segment_min_length}支，无法划分线段")
            return df
        
        # 划分线段（需满足：笔类型交替、最小笔数量、有效破坏）
        segment_id = 0
        for i in range(self.segment_min_length - 1, len(filtered_pens)):
            # 取连续N支笔（N=segment_min_length）
            segment_pens = filtered_pens[i - self.segment_min_length + 1 : i + 1]
            start_idx, start_pen_type, start_price = segment_pens[0]
            end_idx, end_pen_type, end_price = segment_pens[-1]
            
            # 验证线段类型（由第一支笔类型决定）
            segment_type = start_pen_type
            if not all(p[1] != segment_type for p in segment_pens[1::2]) or not all(p[1] == segment_type for p in segment_pens[::2]):
                logger.debug(f"线段笔类型不交替，跳过")
                continue
            
            # 计算线段的高低点（用于破坏验证）
            segment_high = df.loc[start_idx:end_idx, 'high'].max()
            segment_low = df.loc[start_idx:end_idx, 'low'].min()
            
            # 验证线段是否被破坏（后续笔是否突破关键位置）
            if i + 1 < len(filtered_pens):
                next_idx, next_pen_type, next_price = filtered_pens[i + 1]
                break_valid = False
                
                if segment_type == 'up':
                    # 上升线段：被下降笔跌破线段低点的一定比例视为有效破坏
                    if next_pen_type == 'down' and next_price <= segment_low * (1 - self.segment_break_ratio):
                        break_valid = True
                else:
                    # 下降线段：被上升笔突破线段高点的一定比例视为有效破坏
                    if next_pen_type == 'up' and next_price >= segment_high * (1 + self.segment_break_ratio):
                        break_valid = True
                
                if not break_valid:
                    logger.debug(f"线段{segment_id}未被有效破坏，跳过")
                    continue
            
            # 计算线段统计信息
            pen_count = self.segment_min_length
            price_change = end_price - start_price
            price_ratio = abs(price_change) / start_price
            
            # 标记线段信息
            df.loc[start_idx:end_idx, 'segment_id'] = segment_id
            df.loc[start_idx, 'segment_start'] = True
            df.loc[end_idx, 'segment_end'] = True
            df.loc[start_idx:end_idx, 'segment_type'] = segment_type
            df.loc[start_idx:end_idx, 'segment_length_pen'] = pen_count
            df.loc[start_idx:end_idx, 'segment_price_change'] = price_change
            df.loc[start_idx:end_idx, 'segment_price_ratio'] = price_ratio
            
            logger.debug(f"划分线段{segment_id}：类型{segment_type} | 笔数{pen_count} | 波动{price_ratio:.4f}")
            segment_id += 1
        
        logger.info(f"线段划分完成：有效线段数量{segment_id}段")
        return df

    def calculate_central_banks(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建中枢（原有完整逻辑，无改动）"""
        df = df.copy()
        df['central_bank'] = False  # 是否在中枢内
        df['central_bank_id'] = -1  # 中枢编号
        df['central_bank_type'] = None  # standard（标准）/ expand（扩展）/ run（奔走）
        df['central_bank_high'] = np.nan  # 中枢上沿价格
        df['central_bank_low'] = np.nan   # 中枢下沿价格
        df['central_bank_mid'] = np.nan   # 中枢中轴价格（(上沿+下沿)/2）
        df['central_bank_length'] = 0     # 中枢包含的K线数量
        df['central_bank_overlap_ratio'] = 0.0  # 中枢重叠比例
        
        # 筛选有笔的K线（中枢基于笔的重叠构建）
        pen_klines = df[df['pen_id'] != -1].copy()
        if len(pen_klines) < self.central_bank_min_length:
            logger.warning(f"有效K线不足{self.central_bank_min_length}根，无法构建中枢")
            return df
        
        # 按笔分组，获取每笔的高低点
        pen_groups = pen_klines.groupby('pen_id').agg({
            'high': 'max',
            'low': 'min',
            'open': 'first',
            'close': 'last',
            'date': ['first', 'last']
        }).reset_index()
        pen_groups.columns = ['pen_id', 'pen_high', 'pen_low', 'pen_open', 'pen_close', 'pen_start_date', 'pen_end_date']
        
        if len(pen_groups) < 3:
            logger.warning("有效笔不足3支，无法构建中枢")
            return df
        
        # 构建中枢（至少3笔重叠）
        central_bank_id = 0
        for i in range(2, len(pen_groups)):
            # 取连续3笔作为候选中枢
            pen1 = pen_groups.iloc[i-2]
            pen2 = pen_groups.iloc[i-1]
            pen3 = pen_groups.iloc[i]
            
            # 计算3笔的重叠区间
            overlap_high = min(pen1['pen_high'], pen2['pen_high'], pen3['pen_high'])
            overlap_low = max(pen1['pen_low'], pen2['pen_low'], pen3['pen_low'])
            
            # 验证重叠有效性（必须有重叠，且重叠比例满足要求）
            if overlap_high <= overlap_low:
                logger.debug(f"3笔无重叠，无法构建中枢")
                continue
            
            # 计算重叠比例（重叠区间/3笔总区间）
            total_range = max(pen1['pen_high'], pen2['pen_high'], pen3['pen_high']) - min(pen1['pen_low'], pen2['pen_low'], pen3['pen_low'])
            overlap_range = overlap_high - overlap_low
            overlap_ratio = overlap_range / total_range if total_range != 0 else 0
            
            if overlap_ratio < self.central_bank_overlap_ratio:
                logger.debug(f"中枢重叠比例不足：{overlap_ratio:.4f}（最小{self.central_bank_overlap_ratio}），跳过")
                continue
            
            # 确定中枢的K线范围
            pen1_start = pen_klines[pen_klines['pen_id'] == pen1['pen_id']].index.min()
            pen3_end = pen_klines[pen_klines['pen_id'] == pen3['pen_id']].index.max()
            cb_kline_range = df.index[(df.index >= pen1_start) & (df.index <= pen3_end)]
            
            # 验证中枢K线数量
            if len(cb_kline_range) < self.central_bank_min_length:
                logger.debug(f"中枢K线数量不足：{len(cb_kline_range)}（最小{self.central_bank_min_length}），跳过")
                continue
            
            # 判断中枢类型
            cb_type = 'standard'
            # 扩展中枢：后续笔仍在中枢区间内（允许一定扩展比例）
            for j in range(i+1, len(pen_groups)):
                next_pen = pen_groups.iloc[j]
                if (next_pen['pen_low'] <= overlap_high * (1 + self.central_bank_expand_ratio) and
                    next_pen['pen_high'] >= overlap_low * (1 - self.central_bank_expand_ratio)):
                    # 扩展中枢范围
                    next_pen_end = pen_klines[pen_klines['pen_id'] == next_pen['pen_id']].index.max()
                    cb_kline_range = df.index[(df.index >= pen1_start) & (df.index <= next_pen_end)]
                    cb_type = 'expand'
                else:
                    break
            
            # 奔走中枢：重叠比例极低（趋势性强）
            if overlap_ratio < self.central_bank_overlap_ratio * 0.5:
                cb_type = 'run'
            
            # 计算中枢统计信息
            cb_high = overlap_high
            cb_low = overlap_low
            cb_mid = (cb_high + cb_low) / 2
            cb_length = len(cb_kline_range)
            
            # 标记中枢信息
            df.loc[cb_kline_range, 'central_bank'] = True
            df.loc[cb_kline_range, 'central_bank_id'] = central_bank_id
            df.loc[cb_kline_range, 'central_bank_type'] = cb_type
            df.loc[cb_kline_range, 'central_bank_high'] = cb_high
            df.loc[cb_kline_range, 'central_bank_low'] = cb_low
            df.loc[cb_kline_range, 'central_bank_mid'] = cb_mid
            df.loc[cb_kline_range, 'central_bank_length'] = cb_length
            df.loc[cb_kline_range, 'central_bank_overlap_ratio'] = overlap_ratio
            
            logger.debug(f"构建中枢{central_bank_id}：类型{cb_type} | 区间[{cb_kline_range.min()}:{cb_kline_range.max()}] | 长度{cb_length} | 重叠比例{overlap_ratio:.4f}")
            central_bank_id += 1
        
        logger.info(f"中枢构建完成：有效中枢数量{central_bank_id}个")
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标（MACD/RSI/ATR）（原有完整逻辑，无改动）"""
        df = df.copy()
        
        # 计算MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 计算RSI
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df

    def detect_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测背离（价格与MACD/RSI）（原有完整逻辑，无改动）"""
        df = df.copy()
        df = self._calculate_indicators(df)
        
        df['divergence'] = None  # bull（底背离）/ bear（顶背离）/ None
        df['divergence_indicator'] = None  # macd/rsi/both
        df['divergence_strength'] = 0.0  # 背离强度（0-1）
        
        # 筛选顶分型和底分型的位置
        top_fractals = df[df['top_fractal']].index.tolist()
        bottom_fractals = df[df['bottom_fractal']].index.tolist()
        
        # 检测顶背离（价格创新高，指标不创新高）
        if len(top_fractals) >= 2:
            for i in range(1, len(top_fractals)):
                prev_idx = top_fractals[i-1]
                curr_idx = top_fractals[i]
                
                # 价格创新高
                prev_price = df.loc[prev_idx, 'high']
                curr_price = df.loc[curr_idx, 'high']
                if curr_price <= prev_price * (1 + self.divergence_threshold):
                    continue
                
                # MACD顶背离
                prev_macd = df.loc[prev_idx, 'macd_hist']
                curr_macd = df.loc[curr_idx, 'macd_hist']
                macd_divergence = curr_macd < prev_macd * (1 - self.divergence_threshold)
                
                # RSI顶背离
                prev_rsi = df.loc[prev_idx, 'rsi']
                curr_rsi = df.loc[curr_idx, 'rsi']
                rsi_divergence = curr_rsi < prev_rsi * (1 - self.divergence_threshold)
                
                if macd_divergence or rsi_divergence:
                    # 确定背离指标类型
                    indicator_type = []
                    if macd_divergence:
                        indicator_type.append('macd')
                    if rsi_divergence:
                        indicator_type.append('rsi')
                    indicator_type = ','.join(indicator_type)
                    
                    # 计算背离强度
                    price_diff_ratio = (curr_price - prev_price) / prev_price
                    macd_diff_ratio = (prev_macd - curr_macd) / abs(prev_macd) if prev_macd != 0 else 0
                    rsi_diff_ratio = (prev_rsi - curr_rsi) / prev_rsi if prev_rsi != 0 else 0
                    
                    if indicator_type == 'both':
                        strength = (price_diff_ratio + macd_diff_ratio + rsi_diff_ratio) / (3 * self.divergence_threshold)
                    else:
                        strength = (price_diff_ratio + (macd_diff_ratio if indicator_type == 'macd' else rsi_diff_ratio)) / (2 * self.divergence_threshold)
                    
                    strength = min(1.0, max(0.0, strength))
                    
                    # 标记顶背离
                    df.loc[curr_idx, 'divergence'] = 'bear'
                    df.loc[curr_idx, 'divergence_indicator'] = indicator_type
                    df.loc[curr_idx, 'divergence_strength'] = strength
                    
                    logger.debug(f"检测到顶背离：位置{curr_idx} | 指标{indicator_type} | 强度{strength:.3f}")
        
        # 检测底背离（价格创新低，指标不创新低）
        if len(bottom_fractals) >= 2:
            for i in range(1, len(bottom_fractals)):
                prev_idx = bottom_fractals[i-1]
                curr_idx = bottom_fractals[i]
                
                # 价格创新低
                prev_price = df.loc[prev_idx, 'low']
                curr_price = df.loc[curr_idx, 'low']
                if curr_price >= prev_price * (1 - self.divergence_threshold):
                    continue
                
                # MACD底背离
                prev_macd = df.loc[prev_idx, 'macd_hist']
                curr_macd = df.loc[curr_idx, 'macd_hist']
                macd_divergence = curr_macd > prev_macd * (1 + self.divergence_threshold)
                
                # RSI底背离
                prev_rsi = df.loc[prev_idx, 'rsi']
                curr_rsi = df.loc[curr_idx, 'rsi']
                rsi_divergence = curr_rsi > prev_rsi * (1 + self.divergence_threshold)
                
                if macd_divergence or rsi_divergence:
                    # 确定背离指标类型
                    indicator_type = []
                    if macd_divergence:
                        indicator_type.append('macd')
                    if rsi_divergence:
                        indicator_type.append('rsi')
                    indicator_type = ','.join(indicator_type)
                    
                    # 计算背离强度
                    price_diff_ratio = (prev_price - curr_price) / prev_price
                    macd_diff_ratio = (curr_macd - prev_macd) / abs(prev_macd) if prev_macd != 0 else 0
                    rsi_diff_ratio = (curr_rsi - prev_rsi) / prev_rsi if prev_rsi != 0 else 0
                    
                    if indicator_type == 'both':
                        strength = (price_diff_ratio + macd_diff_ratio + rsi_diff_ratio) / (3 * self.divergence_threshold)
                    else:
                        strength = (price_diff_ratio + (macd_diff_ratio if indicator_type == 'macd' else rsi_diff_ratio)) / (2 * self.divergence_threshold)
                    
                    strength = min(1.0, max(0.0, strength))
                    
                    # 标记底背离
                    df.loc[curr_idx, 'divergence'] = 'bull'
                    df.loc[curr_idx, 'divergence_indicator'] = indicator_type
                    df.loc[curr_idx, 'divergence_strength'] = strength
                    
                    logger.debug(f"检测到底背离：位置{curr_idx} | 指标{indicator_type} | 强度{strength:.3f}")
        
        # 统计背离数量
        bear_count = (df['divergence'] == 'bear').sum()
        bull_count = (df['divergence'] == 'bull').sum()
        logger.info(f"背离检测完成：顶背离{bear_count}个 | 底背离{bull_count}个")
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号（核心修复2：统一信号强度为0-1区间）"""
        df = df.copy()
        df['signal'] = SIGNAL_HOLD  # 初始信号为持有
        df['signal_strength'] = 0.0  # 信号强度（0-1区间，适配回测引擎）
        df['signal_source'] = None   # 信号来源（fractal/pen/segment/central_bank/divergence）
        
        # 信号强度权重配置（原有逻辑）
        weights = {
            'fractal': 0.2,
            'pen': 0.25,
            'segment': 0.3,
            'central_bank': 0.35,
            'divergence': 0.4
        }
        
        logger.info(f"开始生成交易信号（阈值：{self.signal_strength_threshold:.2f}，0-1区间）")
        
        for i in range(max(self.fractal_sensitivity, self.central_bank_min_length), len(df)):
            row = df.iloc[i]
            total_strength = 0.0
            source = []
            
            # 1. 分型信号
            if row['top_fractal']:
                total_strength -= weights['fractal']
                source.append('fractal')
            elif row['bottom_fractal']:
                total_strength += weights['fractal']
                source.append('fractal')
            
            # 2. 笔信号
            if row['pen_end']:
                if row['pen_type'] == 'down':
                    total_strength += weights['pen']
                    source.append('pen')
                elif row['pen_type'] == 'up':
                    total_strength -= weights['pen']
                    source.append('pen')
            
            # 3. 线段信号
            if row['segment_end']:
                if row['segment_type'] == 'down':
                    total_strength += weights['segment']
                    source.append('segment')
                elif row['segment_type'] == 'up':
                    total_strength -= weights['segment']
                    source.append('segment')
            
            # 4. 中枢信号
            if row['central_bank']:
                # 中枢下沿附近 + 底分型 = 买入信号
                if (row['close'] >= row['central_bank_low'] and 
                    row['close'] <= row['central_bank_low'] * (1 + 0.01) and
                    row['bottom_fractal']):
                    total_strength += weights['central_bank']
                    source.append('central_bank')
                # 中枢上沿附近 + 顶分型 = 卖出信号
                elif (row['close'] <= row['central_bank_high'] and 
                      row['close'] >= row['central_bank_high'] * (1 - 0.01) and
                      row['top_fractal']):
                    total_strength -= weights['central_bank']
                    source.append('central_bank')
            
            # 5. 背离信号
            if row['divergence'] == 'bull':
                total_strength += weights['divergence'] * row['divergence_strength']
                source.append('divergence')
            elif row['divergence'] == 'bear':
                total_strength -= weights['divergence'] * row['divergence_strength']
                source.append('divergence')
            
            # 核心修复2：确保信号强度在0-1区间（买入为正，卖出为负，取绝对值后归一化）
            raw_strength = total_strength
            abs_strength = abs(raw_strength)
            
            # 信号强度归一化：理论最大信号强度是所有权重之和(0.2+0.25+0.3+0.35+0.4=1.5)
            # 将信号强度映射到0-1区间
            max_possible_strength = sum(weights.values())  # 1.5
            if max_possible_strength > 0:
                normalized_strength = abs_strength / max_possible_strength
                # 确保归一化后的值在0-1区间
                normalized_strength = min(1.0, max(0.0, normalized_strength))
            else:
                normalized_strength = 0.0
            
            # 信号判断（直接使用0-1区间阈值）
            threshold = self.signal_strength_threshold
            # 使用归一化后的信号强度进行判断
            normalized_raw_strength = raw_strength * (normalized_strength / abs_strength if abs_strength > 0 else 0)
            
            if normalized_raw_strength >= threshold:
                df.loc[df.index[i], 'signal'] = SIGNAL_BUY
                df.loc[df.index[i], 'signal_strength'] = normalized_strength  # 归一化后的0-1区间
                df.loc[df.index[i], 'signal_source'] = ','.join(source) if source else 'unknown'
            elif normalized_raw_strength <= -threshold:
                df.loc[df.index[i], 'signal'] = SIGNAL_SELL
                df.loc[df.index[i], 'signal_strength'] = normalized_strength  # 归一化后的0-1区间
                df.loc[df.index[i], 'signal_source'] = ','.join(source) if source else 'unknown'
            else:
                df.loc[df.index[i], 'signal'] = SIGNAL_HOLD
                df.loc[df.index[i], 'signal_strength'] = 0.0  # 持有信号强度为0
                df.loc[df.index[i], 'signal_source'] = None
        
        # 统计信号数量
        buy_count = (df['signal'] == SIGNAL_BUY).sum()
        sell_count = (df['signal'] == SIGNAL_SELL).sum()
        hold_count = (df['signal'] == SIGNAL_HOLD).sum()
        logger.info(f"信号生成完成：买入{buy_count}个 | 卖出{sell_count}个 | 持有{hold_count}个")
        return df

    def calculate_stop_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动态止损价（原有完整逻辑，无改动）"""
        df = df.copy()
        df['stop_loss_price'] = np.nan  # 止损价
        df['stop_loss_type'] = None     # 止损类型（fractal/central_bank）
        
        # 筛选有交易信号的K线
        signal_klines = df[df['signal'].isin([SIGNAL_BUY, SIGNAL_SELL])].index.tolist()
        
        for idx in signal_klines:
            signal = df.loc[idx, 'signal']
            recent_klines = df.iloc[max(0, idx-30):idx]  # 取最近30根K线
            
            if signal == SIGNAL_BUY:
                # 买入信号：止损价 = 最近底分型低点 * 0.995 或 中枢下沿 * 0.995
                recent_bottoms = recent_klines[recent_klines['bottom_fractal']]['low']
                if not recent_bottoms.empty:
                    stop_loss = recent_bottoms.min() * 0.995
                    stop_loss_type = 'fractal'
                elif df.loc[idx, 'central_bank']:
                    stop_loss = df.loc[idx, 'central_bank_low'] * 0.995
                    stop_loss_type = 'central_bank'
                else:
                    # 无分型和中枢，取最近10根K线低点 * 0.99
                    stop_loss = recent_klines['low'].min() * 0.99
                    stop_loss_type = 'recent_low'
                
                df.loc[idx, 'stop_loss_price'] = stop_loss
                df.loc[idx, 'stop_loss_type'] = stop_loss_type
            
            elif signal == SIGNAL_SELL:
                # 卖出信号：止损价 = 最近顶分型高点 * 1.005 或 中枢上沿 * 1.005
                recent_tops = recent_klines[recent_klines['top_fractal']]['high']
                if not recent_tops.empty:
                    stop_loss = recent_tops.max() * 1.005
                    stop_loss_type = 'fractal'
                elif df.loc[idx, 'central_bank']:
                    stop_loss = df.loc[idx, 'central_bank_high'] * 1.005
                    stop_loss_type = 'central_bank'
                else:
                    # 无分型和中枢，取最近10根K线高点 * 1.01
                    stop_loss = recent_klines['high'].max() * 1.01
                    stop_loss_type = 'recent_high'
                
                df.loc[idx, 'stop_loss_price'] = stop_loss
                df.loc[idx, 'stop_loss_type'] = stop_loss_type
        
        logger.info("动态止损价计算完成")
        return df

    def determine_market_condition(self, df: pd.DataFrame) -> str:
        """判断市场状态（原有完整逻辑，无改动）"""
        if len(df) < 60:
            logger.warning("数据量不足60根K线，无法准确判断市场状态，返回横盘")
            return MARKET_FLAT
        
        # 基于最近30根K线的价格波动和趋势判断
        recent_df = df.iloc[-30:]
        price_range = recent_df['high'].max() - recent_df['low'].min()
        price_mean = recent_df['close'].mean()
        volatility_ratio = price_range / price_mean  # 波动比例
        
        # 计算趋势斜率（线性回归）
        x = np.arange(len(recent_df))
        y = recent_df['close'].values
        slope = np.polyfit(x, y, 1)[0]
        trend_strength = abs(slope) / price_mean  # 趋势强度
        
        # 判断市场状态
        if volatility_ratio < 0.05:
            # 低波动 → 横盘
            return MARKET_FLAT
        elif slope > 0 and trend_strength > 0.001:
            # 上升趋势 → 牛市
            return MARKET_BULL
        elif slope < 0 and trend_strength > 0.001:
            # 下降趋势 → 熊市
            return MARKET_BEAR
        else:
            # 无明显趋势 → 横盘
            return MARKET_FLAT

    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """完整计算流程（原有完整逻辑，无改动）"""
        logger.info("="*50)
        logger.info("开始执行完整缠论计算流程")
        logger.info("="*50)
        
        # 1. 数据验证
        if not self._validate_data(df):
            raise ValueError("数据验证失败，终止缠论计算")
        
        # 2. 执行各步骤计算（按缠论逻辑顺序）
        df = self.calculate_fractals(df)
        df = self.calculate_pens(df)
        df = self.calculate_segments(df)
        df = self.calculate_central_banks(df)
        df = self.detect_divergence(df)
        df = self.generate_signals(df)
        df = self.calculate_stop_loss(df)
        
        # 3. 判断市场状态
        market_condition = self.determine_market_condition(df)
        logger.info(f"当前市场状态：{market_condition}")
        
        # 4. 转换numpy类型为Python原生类型（支持JSON序列化）
        result = convert_numpy_to_python({
            'data': df.to_dict('records'),
            'summary': {
                'initial_capital': self.initial_capital,
                'total_klines': len(df),
                'market_condition': market_condition,
                'fractal_count': {
                    'top': int(df['top_fractal'].sum()),
                    'bottom': int(df['bottom_fractal'].sum())
                },
                'pen_count': len(df['pen_id'].unique()) - (1 if -1 in df['pen_id'].unique() else 0),
                'segment_count': len(df['segment_id'].unique()) - (1 if -1 in df['segment_id'].unique() else 0),
                'central_bank_count': len(df['central_bank_id'].unique()) - (1 if -1 in df['central_bank_id'].unique() else 0),
                'divergence_count': {
                    'bull': int((df['divergence'] == 'bull').sum()),
                    'bear': int((df['divergence'] == 'bear').sum())
                },
                'signal_count': {
                    'buy': int((df['signal'] == SIGNAL_BUY).sum()),
                    'sell': int((df['signal'] == SIGNAL_SELL).sum()),
                    'hold': int((df['signal'] == SIGNAL_HOLD).sum())
                }
            }
        })
        
        logger.info("="*50)
        logger.info("完整缠论计算流程执行完毕")
        logger.info("="*50)
        return result