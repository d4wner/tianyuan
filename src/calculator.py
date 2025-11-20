#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论核心计算器（完整版V2.2）
支持：分型识别、笔划分、线段划分、中枢构建、背离检测、交易信号生成
核心修复：1. 初始资金优先使用用户传入值 2. 补充json模块导入 3. 修复numpy类型JSON序列化问题
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
SIGNAL_DEFAULT_STRENGTH_THRESHOLD = 60  # 信号强度阈值（0-100）
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
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化缠论计算器
        Args:
            config: 配置字典，包含：
                - chanlun: 缠论参数配置
                - risk_management: 风险管理配置
                - initial_capital: 用户传入的初始资金（优先使用）
                - data_validation_enabled: 数据验证开关
                - min_data_points: 最小数据量要求
        """
        self.config = config or {}
        self.chanlun_config = self.config.get('chanlun', {})
        self.risk_config = self.config.get('risk_management', {})
        
        # 核心修复：初始资金优先级 - 用户传入值 > 配置文件值 > 默认值
        self.initial_capital = self._get_initial_capital()
        
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
        
        logger.info(f"缠论计算器初始化完成 | 初始资金：{self.initial_capital:.2f}元（优先级：用户传入 > 配置文件 > 默认值）")
        logger.info(f"缠论核心参数：灵敏度={self.fractal_sensitivity} | 笔最小长度={self.pen_min_length} | 中枢最小长度={self.central_bank_min_length}")

    def _get_initial_capital(self) -> float:
        """
        获取初始资金（核心修复：优先级处理）
        优先级顺序：
        1. 用户从backtester传入的 initial_capital（config直接传入）
        2. 配置文件中的 risk_management.initial_capital
        3. 默认值 DEFAULT_INITIAL_CAPITAL
        """
        # 1. 优先使用用户从backtester传入的初始资金（最优先）
        if 'initial_capital' in self.config and self.config['initial_capital'] > 0:
            user_capital = self.config['initial_capital']
            logger.info(f"使用用户传入的初始资金：{user_capital:.2f}元")
            return user_capital
        
        # 2. 其次使用配置文件中的初始资金
        config_capital = self.risk_config.get('initial_capital', 0.0)
        if config_capital > 0:
            logger.info(f"使用配置文件中的初始资金：{config_capital:.2f}元")
            return config_capital
        
        # 3. 最后使用默认值
        logger.warning(f"未指定初始资金，使用默认值：{DEFAULT_INITIAL_CAPITAL:.2f}元")
        return DEFAULT_INITIAL_CAPITAL

    def _init_chanlun_params(self):
        """初始化缠论核心参数（原有完整逻辑）"""
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
        """验证输入数据合法性（原有完整逻辑）"""
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
            logger.warning("数据已按日期重新排序")
        
        logger.info("数据合法性验证通过")
        return True

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标（MACD/RSI/ATR）（原有完整逻辑）"""
        df = df.copy()
        
        # 1. MACD计算
        df['ema_fast'] = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 2. RSI计算
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. ATR计算
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info("技术指标计算完成（MACD/RSI/ATR）")
        return df

    def _identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别顶分型/底分型（原有完整逻辑）"""
        df = df.copy()
        n = self.fractal_sensitivity
        
        # 初始化分型列
        df['top_fractal'] = False
        df['bottom_fractal'] = False
        df['fractal_type'] = None  # top/bottom/None
        
        logger.info(f"开始分型识别（灵敏度：{n}）...")
        
        # 顶分型识别：中间K线高点是前后n根K线的最高点，且价格差满足阈值
        for i in range(n, len(df) - n):
            # 中间K线高点大于前后n根K线的高点
            high_max = df.iloc[i-n:i+n+1]['high'].max()
            if df.iloc[i]['high'] == high_max:
                # 验证价格差（避免极小波动）
                high_diff = (df.iloc[i]['high'] - df.iloc[i-n:i+n+1]['high'].min()) / df.iloc[i]['high']
                if high_diff >= self.fractal_min_price_diff:
                    df.at[i, 'top_fractal'] = True
                    df.at[i, 'fractal_type'] = 'top'
        
        # 底分型识别：中间K线低点是前后n根K线的最低点，且价格差满足阈值
        for i in range(n, len(df) - n):
            # 中间K线低点小于前后n根K线的低点
            low_min = df.iloc[i-n:i+n+1]['low'].min()
            if df.iloc[i]['low'] == low_min:
                # 验证价格差（避免极小波动）
                low_diff = (df.iloc[i-n:i+n+1]['low'].max() - df.iloc[i]['low']) / df.iloc[i]['low']
                if low_diff >= self.fractal_min_price_diff:
                    df.at[i, 'bottom_fractal'] = True
                    df.at[i, 'fractal_type'] = 'bottom'
        
        # 统计分型数量
        top_count = df['top_fractal'].sum()
        bottom_count = df['bottom_fractal'].sum()
        logger.info(f"分型识别完成：顶分型{top_count}个 | 底分型{bottom_count}个")
        
        return df

    def _divide_pens(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于分型划分笔（原有完整逻辑）"""
        df = df.copy()
        
        # 初始化笔相关列
        df['pen_type'] = None  # up/down/None
        df['pen_id'] = -1  # 笔ID（从0开始）
        df['pen_start'] = False  # 笔起点
        df['pen_end'] = False  # 笔终点
        df['pen_length'] = 0  # 笔包含的K线数量
        df['pen_price_change'] = 0.0  # 笔价格变化
        df['pen_price_ratio'] = 0.0  # 笔价格变化比例
        
        logger.info(f"开始笔划分（最小长度：{self.pen_min_length}根K线）...")
        
        # 筛选有效分型（按时间排序）
        fractals = df[df['fractal_type'].notna()].copy()
        if len(fractals) < 2:
            logger.warning("有效分型不足2个，无法划分笔")
            return df
        
        pen_id = 0
        current_pen_start = None
        current_pen_type = None
        
        # 遍历分型，按"底-顶-底-顶"顺序划分笔
        for i in range(1, len(fractals)):
            prev_fractal = fractals.iloc[i-1]
            curr_fractal = fractals.iloc[i]
            
            # 验证分型顺序（底->顶：上升笔；顶->底：下降笔）
            if prev_fractal['fractal_type'] == 'bottom' and curr_fractal['fractal_type'] == 'top':
                pen_type = 'up'
            elif prev_fractal['fractal_type'] == 'top' and curr_fractal['fractal_type'] == 'bottom':
                pen_type = 'down'
            else:
                # 分型顺序错误（底->底或顶->顶），跳过
                logger.debug(f"分型顺序错误：{prev_fractal['fractal_type']} -> {curr_fractal['fractal_type']}，跳过")
                continue
            
            # 计算笔的K线范围
            start_idx = prev_fractal.name
            end_idx = curr_fractal.name
            pen_kline_count = end_idx - start_idx + 1
            
            # 验证笔的最小长度
            if pen_kline_count < self.pen_min_length:
                logger.debug(f"笔长度不足：{pen_kline_count}根K线（最小{self.pen_min_length}），跳过")
                continue
            
            # 计算笔的价格变化
            start_price = prev_fractal['close']
            end_price = curr_fractal['close']
            price_change = end_price - start_price
            price_ratio = abs(price_change) / start_price
            
            # 验证笔的最小价格波动比例
            if price_ratio < self.pen_min_price_ratio:
                logger.debug(f"笔价格波动不足：{price_ratio:.4f}（最小{self.pen_min_price_ratio}），跳过")
                continue
            
            # 标记笔信息
            df.loc[start_idx:end_idx, 'pen_id'] = pen_id
            df.loc[start_idx, 'pen_start'] = True
            df.loc[end_idx, 'pen_end'] = True
            df.loc[start_idx:end_idx, 'pen_type'] = pen_type
            df.loc[start_idx:end_idx, 'pen_length'] = pen_kline_count
            df.loc[start_idx:end_idx, 'pen_price_change'] = price_change
            df.loc[start_idx:end_idx, 'pen_price_ratio'] = price_ratio
            
            logger.debug(f"划分笔{pen_id}：类型{pen_type} | 范围[{start_idx}:{end_idx}] | 长度{pen_kline_count} | 波动{price_ratio:.4f}")
            
            pen_id += 1
            current_pen_start = start_idx
            current_pen_type = pen_type
        
        # 统计笔数量
        valid_pen_count = pen_id
        logger.info(f"笔划分完成：有效笔{valid_pen_count}支")
        
        return df

    def _divide_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于笔划分线段（原有完整逻辑）"""
        df = df.copy()
        
        # 初始化线段相关列
        df['segment_type'] = None  # up/down/None
        df['segment_id'] = -1  # 线段ID（从0开始）
        df['segment_start'] = False  # 线段起点
        df['segment_end'] = False  # 线段终点
        df['segment_length'] = 0  # 线段包含的笔数量
        df['segment_price_change'] = 0.0  # 线段价格变化
        df['segment_price_ratio'] = 0.0  # 线段价格变化比例
        
        logger.info(f"开始线段划分（最小笔数：{self.segment_min_length}）...")
        
        # 筛选有效笔的起点（用于线段划分）
        pen_starts = df[df['pen_start']].copy()
        if len(pen_starts) < self.segment_min_length:
            logger.warning(f"有效笔不足{self.segment_min_length}支，无法划分线段")
            return df
        
        segment_id = 0
        current_segment_start = None
        current_segment_type = None
        
        # 遍历笔，按"上升笔-下降笔-上升笔"或"下降笔-上升笔-下降笔"顺序划分线段
        for i in range(1, len(pen_starts)):
            prev_pen_start = pen_starts.iloc[i-1]
            curr_pen_start = pen_starts.iloc[i]
            
            prev_pen_type = prev_pen_start['pen_type']
            curr_pen_type = curr_pen_start['pen_type']
            
            # 验证笔类型顺序（线段需要交替的笔类型）
            if prev_pen_type == curr_pen_type:
                logger.debug(f"笔类型重复：{prev_pen_type} -> {curr_pen_type}，跳过")
                continue
            
            # 确定线段类型（基于第一支笔的类型）
            if current_segment_type is None:
                current_segment_type = prev_pen_type
                current_segment_start = prev_pen_start.name
            
            # 检查线段是否被破坏（核心逻辑）
            segment_pens = df[df['segment_id'] == segment_id]
            if len(segment_pens) >= self.segment_min_length - 1:
                # 计算线段的高低点
                segment_high = df.loc[current_segment_start:curr_pen_start.name, 'high'].max()
                segment_low = df.loc[current_segment_start:curr_pen_start.name, 'low'].min()
                
                # 检查是否被破坏（下降线段被上升笔突破高点，上升线段被下降笔跌破低点）
                if current_segment_type == 'up':
                    # 上升线段：被下降笔跌破线段低点的一定比例视为破坏
                    if curr_pen_type == 'down' and curr_pen_start['low'] <= segment_low * (1 - self.segment_break_ratio):
                        # 线段被破坏，结束当前线段
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_id'] = segment_id
                        df.loc[current_segment_start, 'segment_start'] = True
                        df.loc[curr_pen_start.name, 'segment_end'] = True
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_type'] = current_segment_type
                        
                        # 计算线段统计信息
                        segment_pen_count = len(df[df['segment_id'] == segment_id]['pen_id'].unique())
                        start_price = df.loc[current_segment_start, 'close']
                        end_price = df.loc[curr_pen_start.name, 'close']
                        price_change = end_price - start_price
                        price_ratio = abs(price_change) / start_price
                        
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_length'] = segment_pen_count
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_price_change'] = price_change
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_price_ratio'] = price_ratio
                        
                        logger.debug(f"划分线段{segment_id}：类型{current_segment_type} | 笔数{segment_pen_count} | 波动{price_ratio:.4f}")
                        
                        # 重置状态，准备下一个线段
                        segment_id += 1
                        current_segment_start = curr_pen_start.name
                        current_segment_type = curr_pen_type
                else:
                    # 下降线段：被上升笔突破线段高点的一定比例视为破坏
                    if curr_pen_type == 'up' and curr_pen_start['high'] >= segment_high * (1 + self.segment_break_ratio):
                        # 线段被破坏，结束当前线段
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_id'] = segment_id
                        df.loc[current_segment_start, 'segment_start'] = True
                        df.loc[curr_pen_start.name, 'segment_end'] = True
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_type'] = current_segment_type
                        
                        # 计算线段统计信息
                        segment_pen_count = len(df[df['segment_id'] == segment_id]['pen_id'].unique())
                        start_price = df.loc[current_segment_start, 'close']
                        end_price = df.loc[curr_pen_start.name, 'close']
                        price_change = end_price - start_price
                        price_ratio = abs(price_change) / start_price
                        
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_length'] = segment_pen_count
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_price_change'] = price_change
                        df.loc[current_segment_start:curr_pen_start.name, 'segment_price_ratio'] = price_ratio
                        
                        logger.debug(f"划分线段{segment_id}：类型{current_segment_type} | 笔数{segment_pen_count} | 波动{price_ratio:.4f}")
                        
                        # 重置状态，准备下一个线段
                        segment_id += 1
                        current_segment_start = curr_pen_start.name
                        current_segment_type = curr_pen_type
        
        # 统计线段数量
        valid_segment_count = segment_id
        logger.info(f"线段划分完成：有效线段{valid_segment_count}段")
        
        return df

    def _build_central_banks(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建中枢（原有完整逻辑）"""
        df = df.copy()
        
        # 初始化中枢相关列
        df['central_bank'] = False  # 是否在中枢内
        df['central_bank_id'] = -1  # 中枢ID（从0开始）
        df['central_bank_type'] = None  # standard/expand/run（标准/扩展/奔走）
        df['central_bank_high'] = 0.0  # 中枢上沿
        df['central_bank_low'] = 0.0  # 中枢下沿
        df['central_bank_mid'] = 0.0  # 中枢中轴
        df['central_bank_length'] = 0  # 中枢包含的K线数量
        df['central_bank_overlap_ratio'] = 0.0  # 中枢重叠比例
        
        logger.info(f"开始中枢构建（最小长度：{self.central_bank_min_length}根K线）...")
        
        # 遍历K线，基于价格重叠构建中枢
        central_bank_id = 0
        i = 0
        while i < len(df) - self.central_bank_min_length + 1:
            # 取当前窗口的K线
            window_end = i + self.central_bank_min_length - 1
            window_df = df.iloc[i:window_end+1]
            
            # 计算窗口内的高低点
            window_high = window_df['high'].max()
            window_low = window_df['low'].min()
            window_mid = (window_high + window_low) / 2
            
            # 计算窗口内价格重叠比例（判断是否为中枢）
            price_range = window_high - window_low
            if price_range == 0:
                i += 1
                continue
            
            overlap_sum = 0.0
            for j in range(1, len(window_df)):
                prev_high = window_df.iloc[j-1]['high']
                prev_low = window_df.iloc[j-1]['low']
                curr_high = window_df.iloc[j]['high']
                curr_low = window_df.iloc[j]['low']
                
                # 计算两根K线的重叠部分
                overlap_high = min(prev_high, curr_high)
                overlap_low = max(prev_low, curr_low)
                if overlap_high > overlap_low:
                    overlap_sum += overlap_high - overlap_low
            
            overlap_ratio = overlap_sum / (price_range * (len(window_df) - 1))
            
            # 验证重叠比例（满足条件则视为中枢）
            if overlap_ratio >= self.central_bank_overlap_ratio:
                # 标记中枢基础信息
                df.loc[i:window_end, 'central_bank'] = True
                df.loc[i:window_end, 'central_bank_id'] = central_bank_id
                df.loc[i:window_end, 'central_bank_high'] = window_high
                df.loc[i:window_end, 'central_bank_low'] = window_low
                df.loc[i:window_end, 'central_bank_mid'] = window_mid
                df.loc[i:window_end, 'central_bank_length'] = self.central_bank_min_length
                df.loc[i:window_end, 'central_bank_overlap_ratio'] = overlap_ratio
                
                # 判断中枢类型
                # 标准中枢：长度刚好为最小长度，重叠比例适中
                if self.central_bank_min_length <= len(window_df) <= self.central_bank_min_length * 2:
                    df.loc[i:window_end, 'central_bank_type'] = 'standard'
                # 扩展中枢：长度超过最小长度2倍，重叠比例高
                elif len(window_df) > self.central_bank_min_length * 2:
                    df.loc[i:window_end, 'central_bank_type'] = 'expand'
                # 奔走中枢：重叠比例低，趋势性强
                elif overlap_ratio < self.central_bank_overlap_ratio * 0.5:
                    df.loc[i:window_end, 'central_bank_type'] = 'run'
                
                logger.debug(f"构建中枢{central_bank_id}：类型{df.loc[i, 'central_bank_type']} | 范围[{i}:{window_end}] | 重叠比例{overlap_ratio:.4f}")
                
                # 中枢向后扩展（检查后续K线是否仍在中枢范围内）
                extend_i = window_end + 1
                while extend_i < len(df):
                    extend_high = df.iloc[extend_i]['high']
                    extend_low = df.iloc[extend_i]['low']
                    
                    # 检查是否在中枢范围内（允许一定扩展比例）
                    if (extend_low <= window_high * (1 + self.central_bank_expand_ratio) and
                        extend_high >= window_low * (1 - self.central_bank_expand_ratio)):
                        # 扩展中枢范围
                        df.loc[extend_i, 'central_bank'] = True
                        df.loc[extend_i, 'central_bank_id'] = central_bank_id
                        df.loc[extend_i, 'central_bank_high'] = window_high
                        df.loc[extend_i, 'central_bank_low'] = window_low
                        df.loc[extend_i, 'central_bank_mid'] = window_mid
                        df.loc[extend_i, 'central_bank_length'] = self.central_bank_min_length + (extend_i - window_end)
                        df.loc[extend_i, 'central_bank_overlap_ratio'] = overlap_ratio
                        df.loc[extend_i, 'central_bank_type'] = df.loc[i, 'central_bank_type']
                        
                        extend_i += 1
                    else:
                        break
                
                # 跳过已处理的K线（中枢扩展后的终点）
                i = extend_i
                central_bank_id += 1
            else:
                i += 1
        
        # 统计中枢数量
        valid_central_bank_count = central_bank_id
        logger.info(f"中枢构建完成：有效中枢{valid_central_bank_count}个")
        
        return df

    def _detect_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测背离（价格与MACD/RSI）（原有完整逻辑）"""
        df = df.copy()
        
        # 初始化背离相关列
        df['divergence'] = None  # bull/bear/None（底背离/顶背离）
        df['divergence_type'] = None  # macd/rsi/both（MACD背离/RSI背离/双重背离）
        df['divergence_strength'] = 0  # 背离强度（0-100）
        df['divergence_threshold'] = self.divergence_threshold  # 背离阈值
        
        logger.info(f"开始背离检测（阈值：{self.divergence_threshold}）...")
        
        # 筛选有分型的K线（背离通常出现在分型位置）
        fractal_df = df[df['fractal_type'].notna()].copy()
        if len(fractal_df) < 2:
            logger.warning("有效分型不足2个，无法检测背离")
            return df
        
        # 遍历分型，检测顶背离/底背离
        for i in range(1, len(fractal_df)):
            prev_fractal = fractal_df.iloc[i-1]
            curr_fractal = fractal_df.iloc[i]
            prev_idx = prev_fractal.name
            curr_idx = curr_fractal.name
            
            # 1. 顶背离检测（价格创新高，指标不创新高）
            if (curr_fractal['fractal_type'] == 'top' and
                prev_fractal['fractal_type'] == 'top' and
                curr_fractal['high'] > prev_fractal['high'] * (1 + self.divergence_threshold)):
                
                # MACD顶背离
                macd_prev = prev_fractal['macd_hist']
                macd_curr = curr_fractal['macd_hist']
                macd_divergence = macd_curr < macd_prev * (1 - self.divergence_threshold)
                
                # RSI顶背离
                rsi_prev = prev_fractal['rsi']
                rsi_curr = curr_fractal['rsi']
                rsi_divergence = rsi_curr < rsi_prev * (1 - self.divergence_threshold)
                
                if macd_divergence or rsi_divergence:
                    # 确定背离类型
                    if macd_divergence and rsi_divergence:
                        divergence_type = 'both'
                    elif macd_divergence:
                        divergence_type = 'macd'
                    else:
                        divergence_type = 'rsi'
                    
                    # 计算背离强度（基于价格差和指标差）
                    price_diff_ratio = (curr_fractal['high'] - prev_fractal['high']) / prev_fractal['high']
                    if divergence_type == 'both':
                        indicator_diff_ratio = (abs(macd_curr - macd_prev) + abs(rsi_curr - rsi_prev)) / 2
                    elif divergence_type == 'macd':
                        indicator_diff_ratio = abs(macd_curr - macd_prev) / abs(macd_prev) if macd_prev != 0 else 0
                    else:
                        indicator_diff_ratio = abs(rsi_curr - rsi_prev) / rsi_prev if rsi_prev != 0 else 0
                    
                    divergence_strength = min(100, int((price_diff_ratio + indicator_diff_ratio) / (2 * self.divergence_threshold) * 100))
                    
                    # 标记背离信息
                    df.loc[curr_idx, 'divergence'] = 'bear'
                    df.loc[curr_idx, 'divergence_type'] = divergence_type
                    df.loc[curr_idx, 'divergence_strength'] = divergence_strength
                    
                    logger.debug(f"检测到顶背离：位置{curr_idx} | 类型{divergence_type} | 强度{divergence_strength}")
            
            # 2. 底背离检测（价格创新低，指标不创新低）
            elif (curr_fractal['fractal_type'] == 'bottom' and
                  prev_fractal['fractal_type'] == 'bottom' and
                  curr_fractal['low'] < prev_fractal['low'] * (1 - self.divergence_threshold)):
                
                # MACD底背离
                macd_prev = prev_fractal['macd_hist']
                macd_curr = curr_fractal['macd_hist']
                macd_divergence = macd_curr > macd_prev * (1 + self.divergence_threshold)
                
                # RSI底背离
                rsi_prev = prev_fractal['rsi']
                rsi_curr = curr_fractal['rsi']
                rsi_divergence = rsi_curr > rsi_prev * (1 + self.divergence_threshold)
                
                if macd_divergence or rsi_divergence:
                    # 确定背离类型
                    if macd_divergence and rsi_divergence:
                        divergence_type = 'both'
                    elif macd_divergence:
                        divergence_type = 'macd'
                    else:
                        divergence_type = 'rsi'
                    
                    # 计算背离强度（基于价格差和指标差）
                    price_diff_ratio = (prev_fractal['low'] - curr_fractal['low']) / prev_fractal['low']
                    if divergence_type == 'both':
                        indicator_diff_ratio = (abs(macd_curr - macd_prev) + abs(rsi_curr - rsi_prev)) / 2
                    elif divergence_type == 'macd':
                        indicator_diff_ratio = abs(macd_curr - macd_prev) / abs(macd_prev) if macd_prev != 0 else 0
                    else:
                        indicator_diff_ratio = abs(rsi_curr - rsi_prev) / rsi_prev if rsi_prev != 0 else 0
                    
                    divergence_strength = min(100, int((price_diff_ratio + indicator_diff_ratio) / (2 * self.divergence_threshold) * 100))
                    
                    # 标记背离信息
                    df.loc[curr_idx, 'divergence'] = 'bull'
                    df.loc[curr_idx, 'divergence_type'] = divergence_type
                    df.loc[curr_idx, 'divergence_strength'] = divergence_strength
                    
                    logger.debug(f"检测到底背离：位置{curr_idx} | 类型{divergence_type} | 强度{divergence_strength}")
        
        # 统计背离数量
        bear_divergence_count = (df['divergence'] == 'bear').sum()
        bull_divergence_count = (df['divergence'] == 'bull').sum()
        logger.info(f"背离检测完成：顶背离{bear_divergence_count}个 | 底背离{bull_divergence_count}个")
        
        return df

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号（原有完整逻辑）"""
        df = df.copy()
        
        # 初始化信号相关列
        df['signal'] = SIGNAL_HOLD  # 交易信号（buy/sell/hold）
        df['signal_strength'] = 0  # 信号强度（0-100）
        df['signal_source'] = None  # 信号来源（fractal/pen/segment/central_bank/divergence）
        df['stop_loss_price'] = 0.0  # 止损价格
        df['take_profit_price'] = 0.0  # 止盈价格（可选）
        
        logger.info(f"开始交易信号生成（强度阈值：{self.signal_strength_threshold}）...")
        
        # 遍历K线，基于多维度生成信号
        for i in range(max(self.fractal_sensitivity, self.central_bank_min_length), len(df)):
            current_row = df.iloc[i]
            current_price = current_row['close']
            
            # 初始化信号得分（多维度加权）
            signal_score = 0
            signal_sources = []
            
            # 1. 分型信号（基础信号）
            if current_row['fractal_type'] == 'bottom':
                # 底分型：潜在买入信号
                signal_score += 20
                signal_sources.append('fractal')
            elif current_row['fractal_type'] == 'top':
                # 顶分型：潜在卖出信号
                signal_score -= 20
                signal_sources.append('fractal')
            
            # 2. 笔信号（趋势信号）
            if current_row['pen_end']:
                if current_row['pen_type'] == 'down' and current_row['pen_price_ratio'] >= self.pen_min_price_ratio:
                    # 下降笔结束：买入信号加分
                    signal_score += 30
                    signal_sources.append('pen')
                elif current_row['pen_type'] == 'up' and current_row['pen_price_ratio'] >= self.pen_min_price_ratio:
                    # 上升笔结束：卖出信号加分
                    signal_score -= 30
                    signal_sources.append('pen')
            
            # 3. 线段信号（趋势确认）
            if current_row['segment_end']:
                if current_row['segment_type'] == 'down':
                    # 下降线段结束：买入信号加分
                    signal_score += 25
                    signal_sources.append('segment')
                elif current_row['segment_type'] == 'up':
                    # 上升线段结束：卖出信号加分
                    signal_score -= 25
                    signal_sources.append('segment')
            
            # 4. 中枢信号（支撑/压力）
            if current_row['central_bank']:
                # 价格在中枢下沿附近：买入信号
                if current_price <= current_row['central_bank_low'] * (1 + self.central_bank_expand_ratio * 0.5):
                    signal_score += 15
                    signal_sources.append('central_bank')
                # 价格在中枢上沿附近：卖出信号
                elif current_price >= current_row['central_bank_high'] * (1 - self.central_bank_expand_ratio * 0.5):
                    signal_score -= 15
                    signal_sources.append('central_bank')
            
            # 5. 背离信号（反转确认）
            if current_row['divergence'] == 'bull':
                # 底背离：买入信号大幅加分
                signal_score += current_row['divergence_strength'] * 0.3
                signal_sources.append('divergence')
            elif current_row['divergence'] == 'bear':
                # 顶背离：卖出信号大幅加分
                signal_score -= current_row['divergence_strength'] * 0.3
                signal_sources.append('divergence')
            
            # 6. 技术指标信号（辅助确认）
            if current_row['rsi'] < 30:
                # RSI超卖：买入信号加分
                signal_score += 10
                signal_sources.append('rsi')
            elif current_row['rsi'] > 70:
                # RSI超买：卖出信号加分
                signal_score -= 10
                signal_sources.append('rsi')
            
            if current_row['macd_hist'] > 0 and current_row['macd'] > current_row['macd_signal']:
                # MACD金叉：买入信号加分
                signal_score += 10
                signal_sources.append('macd')
            elif current_row['macd_hist'] < 0 and current_row['macd'] < current_row['macd_signal']:
                # MACD死叉：卖出信号加分
                signal_score -= 10
                signal_sources.append('macd')
            
            # 计算最终信号强度（0-100）
            signal_strength = min(100, max(0, abs(signal_score)))
            df.loc[i, 'signal_strength'] = signal_strength
            df.loc[i, 'signal_source'] = ','.join(signal_sources) if signal_sources else None
            
            # 生成最终信号（基于信号得分和强度阈值）
            if signal_score >= self.signal_strength_threshold:
                # 买入信号
                df.loc[i, 'signal'] = SIGNAL_BUY
                
                # 计算止损价格（基于最近底分型或中枢下沿）
                recent_bottoms = df.iloc[max(0, i-50):i+1][df['fractal_type'] == 'bottom']
                if not recent_bottoms.empty:
                    stop_loss = recent_bottoms['low'].min() * 0.995  # 低于最近底分型0.5%
                else:
                    stop_loss = current_row['central_bank_low'] * 0.99 if current_row['central_bank'] else current_price * 0.98
                df.loc[i, 'stop_loss_price'] = stop_loss
                
                # 计算止盈价格（基于最近顶分型或中枢上沿）
                recent_tops = df.iloc[max(0, i-50):i+1][df['fractal_type'] == 'top']
                if not recent_tops.empty:
                    take_profit = recent_tops['high'].max() * 1.005  # 高于最近顶分型0.5%
                else:
                    take_profit = current_row['central_bank_high'] * 1.01 if current_row['central_bank'] else current_price * 1.03
                df.loc[i, 'take_profit_price'] = take_profit
                
                logger.debug(f"生成买入信号：位置{i} | 强度{signal_strength} | 来源{df.loc[i, 'signal_source']} | 止损{stop_loss:.2f} | 止盈{take_profit:.2f}")
            
            elif signal_score <= -self.signal_strength_threshold:
                # 卖出信号
                df.loc[i, 'signal'] = SIGNAL_SELL
                
                # 计算止损价格（基于最近顶分型或中枢上沿）
                recent_tops = df.iloc[max(0, i-50):i+1][df['fractal_type'] == 'top']
                if not recent_tops.empty:
                    stop_loss = recent_tops['high'].max() * 1.005  # 高于最近顶分型0.5%
                else:
                    stop_loss = current_row['central_bank_high'] * 1.01 if current_row['central_bank'] else current_price * 1.02
                df.loc[i, 'stop_loss_price'] = stop_loss
                
                # 计算止盈价格（基于最近底分型或中枢下沿）
                recent_bottoms = df.iloc[max(0, i-50):i+1][df['fractal_type'] == 'bottom']
                if not recent_bottoms.empty:
                    take_profit = recent_bottoms['low'].min() * 0.995  # 低于最近底分型0.5%
                else:
                    take_profit = current_row['central_bank_low'] * 0.99 if current_row['central_bank'] else current_price * 0.97
                df.loc[i, 'take_profit_price'] = take_profit
                
                logger.debug(f"生成卖出信号：位置{i} | 强度{signal_strength} | 来源{df.loc[i, 'signal_source']} | 止损{stop_loss:.2f} | 止盈{take_profit:.2f}")
        
        # 统计信号数量
        buy_signal_count = (df['signal'] == SIGNAL_BUY).sum()
        sell_signal_count = (df['signal'] == SIGNAL_SELL).sum()
        logger.info(f"信号生成完成：买入信号{buy_signal_count}个 | 卖出信号{sell_signal_count}个")
        
        return df

    def _judge_market_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """判断市场状态（牛市/熊市/横盘）（原有完整逻辑）"""
        df = df.copy()
        df['market_condition'] = MARKET_FLAT  # 初始为横盘
        
        logger.info("开始市场状态判断...")
        
        # 基于最近N根K线的趋势判断
        trend_window = 60  # 趋势判断窗口（60根K线）
        if len(df) < trend_window:
            logger.warning(f"数据量不足{trend_window}根K线，无法准确判断市场状态")
            return df
        
        # 计算移动平均线（用于趋势判断）
        df['ma_short'] = df['close'].rolling(window=20).mean()
        df['ma_long'] = df['close'].rolling(window=60).mean()
        
        # 计算价格波动范围（用于判断横盘）
        df['price_range_20'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']
        
        # 遍历K线判断市场状态
        for i in range(trend_window, len(df)):
            ma_short = df.iloc[i]['ma_short']
            ma_long = df.iloc[i]['ma_long']
            price_range = df.iloc[i]['price_range_20']
            
            # 牛市：短期均线在长期均线上方，且价格上涨
            if ma_short > ma_long * 1.005 and df.iloc[i]['close'] > df.iloc[i-trend_window]['close'] * 1.05:
                df.loc[i, 'market_condition'] = MARKET_BULL
            # 熊市：短期均线在长期均线下方，且价格下跌
            elif ma_short < ma_long * 0.995 and df.iloc[i]['close'] < df.iloc[i-trend_window]['close'] * 0.95:
                df.loc[i, 'market_condition'] = MARKET_BEAR
            # 横盘：价格波动范围小
            elif price_range < 0.05:
                df.loc[i, 'market_condition'] = MARKET_FLAT
        
        # 统计市场状态分布
        bull_count = (df['market_condition'] == MARKET_BULL).sum()
        bear_count = (df['market_condition'] == MARKET_BEAR).sum()
        flat_count = (df['market_condition'] == MARKET_FLAT).sum()
        logger.info(f"市场状态判断完成：牛市{bull_count}根 | 熊市{bear_count}根 | 横盘{flat_count}根")
        
        return df

    def calculate(self, df: pd.DataFrame, timeframe: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        核心计算入口（完整保留原有逻辑）
        Args:
            df: 原始K线数据（含date/open/high/low/close/volume）
            timeframe: 时间级别（daily/weekly/60min等）
            **kwargs: 动态参数（覆盖初始化的缠论参数）
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 计算后的数据（含所有缠论指标） + 回测辅助结果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"开始缠论核心计算 | 时间级别：{timeframe} | 初始资金：{self.initial_capital:.2f}元")
        logger.info(f"{'='*80}")
        
        try:
            # 1. 接收动态参数（覆盖初始化参数）
            self._update_params_from_kwargs(**kwargs)
            
            # 2. 数据验证
            if not self._validate_data(df):
                raise ValueError("数据验证失败，无法执行缠论计算")
            
            # 3. 计算技术指标（MACD/RSI/ATR）
            df = self._calculate_technical_indicators(df)
            
            # 4. 分型识别
            df = self._identify_fractals(df)
            
            # 5. 笔划分
            df = self._divide_pens(df)
            
            # 6. 线段划分
            df = self._divide_segments(df)
            
            # 7. 中枢构建
            df = self._build_central_banks(df)
            
            # 8. 背离检测
            df = self._detect_divergence(df)
            
            # 9. 交易信号生成
            df = self._generate_signals(df)
            
            # 10. 市场状态判断
            df = self._judge_market_condition(df)
            
            # 11. 生成回测辅助结果
            backtest_aux_result = self._generate_backtest_aux_result(df, timeframe)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"缠论核心计算完成 | 时间级别：{timeframe} | 初始资金：{self.initial_capital:.2f}元")
            logger.info(f"{'='*80}")
            
            # 返回 tuple（计算后DataFrame + 回测辅助结果）
            return df, backtest_aux_result
        
        except Exception as e:
            logger.error(f"缠论核心计算失败：{str(e)}", exc_info=True)
            raise

    def _update_params_from_kwargs(self, **kwargs):
        """从动态参数更新缠论配置（原有完整逻辑）"""
        if not kwargs:
            return
        
        logger.info(f"接收动态参数：{kwargs}")
        
        # 更新分型参数
        if 'fractal_sensitivity' in kwargs:
            self.fractal_sensitivity = kwargs['fractal_sensitivity']
        if 'fractal_min_price_diff' in kwargs:
            self.fractal_min_price_diff = kwargs['fractal_min_price_diff']
        
        # 更新笔参数
        if 'pen_min_length' in kwargs:
            self.pen_min_length = kwargs['pen_min_length']
        if 'pen_min_price_ratio' in kwargs:
            self.pen_min_price_ratio = kwargs['pen_min_price_ratio']
        
        # 更新线段参数
        if 'segment_min_length' in kwargs:
            self.segment_min_length = kwargs['segment_min_length']
        if 'segment_break_ratio' in kwargs:
            self.segment_break_ratio = kwargs['segment_break_ratio']
        
        # 更新中枢参数
        if 'central_bank_min_length' in kwargs:
            self.central_bank_min_length = kwargs['central_bank_min_length']
        if 'central_bank_expand_ratio' in kwargs:
            self.central_bank_expand_ratio = kwargs['central_bank_expand_ratio']
        if 'central_bank_overlap_ratio' in kwargs:
            self.central_bank_overlap_ratio = kwargs['central_bank_overlap_ratio']
        
        # 更新背离参数
        if 'divergence_threshold' in kwargs:
            self.divergence_threshold = kwargs['divergence_threshold']
        
        logger.info(f"参数更新完成：灵敏度={self.fractal_sensitivity} | 笔最小长度={self.pen_min_length} | 中枢最小长度={self.central_bank_min_length}")

    def _generate_backtest_aux_result(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """生成回测辅助结果（修复numpy类型序列化问题）"""
        # 统计核心指标（先计算原始值）
        aux_result_raw = {
            'timeframe': timeframe,
            'initial_capital': self.initial_capital,  # 传递初始资金给回测引擎
            'data_count': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'fractal_stats': {
                'top_count': df['top_fractal'].sum(),
                'bottom_count': df['bottom_fractal'].sum()
            },
            'pen_stats': {
                'total_count': df['pen_id'].nunique() - 1,  # 排除-1
                'up_count': (df['pen_type'] == 'up').sum() // df[df['pen_type'] == 'up']['pen_length'].iloc[0] if (df['pen_type'] == 'up').any() else 0,
                'down_count': (df['pen_type'] == 'down').sum() // df[df['pen_type'] == 'down']['pen_length'].iloc[0] if (df['pen_type'] == 'down').any() else 0
            },
            'segment_stats': {
                'total_count': df['segment_id'].nunique() - 1,  # 排除-1
                'up_count': (df['segment_type'] == 'up').sum() // df[df['segment_type'] == 'up']['segment_length'].iloc[0] if (df['segment_type'] == 'up').any() else 0,
                'down_count': (df['segment_type'] == 'down').sum() // df[df['segment_type'] == 'down']['segment_length'].iloc[0] if (df['segment_type'] == 'down').any() else 0
            },
            'central_bank_stats': {
                'total_count': df['central_bank_id'].nunique() - 1,  # 排除-1
                'standard_count': (df['central_bank_type'] == 'standard').sum() // df[df['central_bank_type'] == 'standard']['central_bank_length'].iloc[0] if (df['central_bank_type'] == 'standard').any() else 0,
                'expand_count': (df['central_bank_type'] == 'expand').sum() // df[df['central_bank_type'] == 'expand']['central_bank_length'].iloc[0] if (df['central_bank_type'] == 'expand').any() else 0,
                'run_count': (df['central_bank_type'] == 'run').sum() // df[df['central_bank_type'] == 'run']['central_bank_length'].iloc[0] if (df['central_bank_type'] == 'run').any() else 0
            },
            'divergence_stats': {
                'bull_count': (df['divergence'] == 'bull').sum(),
                'bear_count': (df['divergence'] == 'bear').sum()
            },
            'signal_stats': {
                'buy_count': (df['signal'] == SIGNAL_BUY).sum(),
                'sell_count': (df['signal'] == SIGNAL_SELL).sum(),
                'avg_buy_strength': df[df['signal'] == SIGNAL_BUY]['signal_strength'].mean() if (df['signal'] == SIGNAL_BUY).any() else 0.0,
                'avg_sell_strength': df[df['signal'] == SIGNAL_SELL]['signal_strength'].mean() if (df['signal'] == SIGNAL_SELL).any() else 0.0
            },
            'market_condition_stats': {
                'bull_count': (df['market_condition'] == MARKET_BULL).sum(),
                'bear_count': (df['market_condition'] == MARKET_BEAR).sum(),
                'flat_count': (df['market_condition'] == MARKET_FLAT).sum()
            },
            'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 核心修复：将numpy类型转换为Python原生类型（支持JSON序列化）
        aux_result = convert_numpy_to_python(aux_result_raw)
        
        # 格式化浮点数（避免科学计数法）
        aux_result['signal_stats']['avg_buy_strength'] = round(aux_result['signal_stats']['avg_buy_strength'], 2)
        aux_result['signal_stats']['avg_sell_strength'] = round(aux_result['signal_stats']['avg_sell_strength'], 2)
        aux_result['initial_capital'] = round(aux_result['initial_capital'], 2)
        
        logger.info(f"回测辅助结果生成完成：{json.dumps(aux_result, ensure_ascii=False, indent=2)}")
        return aux_result

# ===================== 测试代码 =====================
def test_calculator():
    """测试缠论计算器（原有完整测试逻辑）"""
    logger.info("开始测试缠论计算器...")
    
    # 构造测试K线数据
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    close_prices = np.random.randn(len(dates)).cumsum() + 100
    high_prices = close_prices + np.random.rand(len(dates)) * 2
    low_prices = close_prices - np.random.rand(len(dates)) * 2
    open_prices = np.random.choice([close_prices[i-1] if i > 0 else close_prices[i] for i in range(len(dates))], len(dates))
    volumes = np.random.randint(1000, 10000, len(dates))
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # 配置参数（模拟用户传入初始资金）
    test_config = {
        'chanlun': {
            'fractal_sensitivity': 3,
            'pen_min_length': 5,
            'central_bank_min_length': 5
        },
        'risk_management': {
            'initial_capital': 150000.0  # 配置文件初始资金
        },
        'initial_capital': 200000.0,  # 用户传入初始资金（应优先使用）
        'data_validation_enabled': True,
        'min_data_points': 50
    }
    
    # 初始化计算器
    calculator = ChanlunCalculator(config=test_config)
    
    # 执行计算
    result_df, backtest_aux = calculator.calculate(
        df=test_df,
        timeframe='daily',
        fractal_sensitivity=4,  # 动态参数覆盖
        divergence_threshold=0.02
    )
    
    # 验证结果
    logger.info(f"测试结果验证：")
    logger.info(f"  - 数据行数：{len(result_df)}")
    logger.info(f"  - 顶分型数量：{result_df['top_fractal'].sum()}")
    logger.info(f"  - 底分型数量：{result_df['bottom_fractal'].sum()}")
    logger.info(f"  - 有效笔数量：{result_df['pen_id'].nunique() - 1}")
    logger.info(f"  - 有效线段数量：{result_df['segment_id'].nunique() - 1}")
    logger.info(f"  - 有效中枢数量：{result_df['central_bank_id'].nunique() - 1}")
    logger.info(f"  - 买入信号数量：{result_df['signal'].value_counts().get('buy', 0)}")
    logger.info(f"  - 卖出信号数量：{result_df['signal'].value_counts().get('sell', 0)}")
    logger.info(f"  - 初始资金（验证）：{backtest_aux['initial_capital']:.2f}元（应等于用户传入的200000.00元）")
    
    logger.info("缠论计算器测试完成！")

if __name__ == "__main__":
    """程序入口（测试用）"""
    test_calculator()