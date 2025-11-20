#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论核心计算器 - 完整功能版（修复索引命名冲突）
支持：多时间级别缠论指标计算、交易信号生成、风险控制、完整回测、绩效分析
修复：1. 索引命名冲突（index → orig_idx）2. 笔/线段/中枢计算的索引访问逻辑
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any, Union
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置日志系统
logger = logging.getLogger('ChanlunCalculator')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ChanlunCalculator:
    """
    缠论核心计算器（完整功能版）
    核心功能：
    1. 多时间级别（日线/周线/分钟线）缠论指标计算（分型/笔/线段/中枢/背离）
    2. 交易信号生成（买入/卖出/持有）
    3. 风险控制（动态止损/固定止损、仓位管理）
    4. 完整回测模拟（交易执行、成本计算、组合跟踪）
    5. 回测绩效分析（收益率、最大回撤、夏普比率等）
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化计算器，加载完整配置
        :param config: 全局配置字典（支持yaml/json加载）
        """
        self.config = config if config is not None else {}
        self._load_config()  # 加载完整配置
        self._validate_config()  # 验证配置合法性
        logger.info("ChanlunCalculator 初始化完成")

    def _load_config(self):
        """加载完整配置（覆盖所有时间级别和功能模块）"""
        # ===================== 基础全局配置 =====================
        self.debug_mode = self.config.get('debug_mode', False)
        self.data_cache_enabled = self.config.get('data_cache_enabled', True)
        self.cache_expire_hours = self.config.get('cache_expire_hours', 24)
        
        # ===================== 缠论核心参数（多时间级别） =====================
        chanlun_config = self.config.get('chanlun', {})
        
        # 周线参数（核心修复：支持动态覆盖）
        self.weekly = {
            'fractal_sensitivity': chanlun_config.get('weekly_fractal_sensitivity', 3),
            'pen_min_length': chanlun_config.get('weekly_pen_min_length', 5),
            'central_bank_min_length': chanlun_config.get('weekly_central_bank_min_length', 5),  # 统一参数名
            'segment_min_length': chanlun_config.get('weekly_segment_min_length', 3),
            'divergence_threshold': chanlun_config.get('weekly_divergence_threshold', 0.02)
        }
        
        # 日线参数
        self.daily = {
            'fractal_sensitivity': chanlun_config.get('daily_fractal_sensitivity', 3),
            'pen_min_length': chanlun_config.get('daily_pen_min_length', 5),
            'central_bank_min_length': chanlun_config.get('daily_central_bank_min_length', 5),  # 统一参数名
            'segment_min_length': chanlun_config.get('daily_segment_min_length', 3),
            'divergence_threshold': chanlun_config.get('daily_divergence_threshold', 0.015)
        }
        
        # 分钟线参数
        self.minute = {
            'fractal_sensitivity': chanlun_config.get('minute_fractal_sensitivity', 5),
            'pen_min_length': chanlun_config.get('minute_pen_min_length', 10),
            'central_bank_min_length': chanlun_config.get('minute_central_bank_min_length', 10),  # 统一参数名
            'segment_min_length': chanlun_config.get('minute_segment_min_length', 5),
            'divergence_threshold': chanlun_config.get('minute_divergence_threshold', 0.01)
        }
        
        # 默认参数（未指定时间级别时使用）
        self.default = {
            'fractal_sensitivity': chanlun_config.get('fractal_sensitivity', 3),
            'pen_min_length': chanlun_config.get('pen_min_length', 5),
            'central_bank_min_length': chanlun_config.get('central_bank_min_length', 5),  # 统一参数名
            'segment_min_length': chanlun_config.get('segment_min_length', 3),
            'divergence_threshold': chanlun_config.get('divergence_threshold', 0.015)
        }
        
        # ===================== 风险控制配置 =====================
        risk_config = self.config.get('risk_management', {})
        
        # 止损配置
        self.stop_loss = {
            'type': risk_config.get('stop_loss_type', 'dynamic'),  # dynamic/fixed
            'fixed_ratio': risk_config.get('fixed_stop_loss_ratio', 0.03),
            'atr_period': risk_config.get('atr_period', 14),
            'atr_multiplier': risk_config.get('atr_multiplier', 2.0),
            'trailing_stop_enabled': risk_config.get('trailing_stop_enabled', True),
            'trailing_stop_ratio': risk_config.get('trailing_stop_ratio', 0.01)
        }
        
        # 仓位管理
        self.position = {
            'max_single_position_ratio': risk_config.get('max_single_position_ratio', 0.8),
            'min_position_ratio': risk_config.get('min_position_ratio', 0.2),
            'signal_strength_weight': risk_config.get('signal_strength_weight', 0.5)
        }
        
        # 交易成本
        self.trade_cost = {
            'commission_rate': risk_config.get('commission_rate', 0.0005),  # 手续费率
            'slippage_ratio': risk_config.get('slippage_ratio', 0.001),  # 滑点率
            'min_commission': risk_config.get('min_commission', 5.0)  # 最低手续费
        }
        
        # ===================== 数据验证配置 =====================
        self.data_validation = {
            'enabled': self.config.get('data_validation_enabled', True),
            'min_data_points': self.config.get('min_data_points', 20),
            'max_date_range_days': self.config.get('max_date_range_days', 365 * 5),
            'allow_missing_data_ratio': self.config.get('allow_missing_data_ratio', 0.05)
        }
        
        # ===================== 回测配置 =====================
        self.backtest_config = {  # 改名避免与方法名冲突
            'initial_capital': self.config.get('initial_capital', 100000.0),
            'risk_free_rate': self.config.get('risk_free_rate', 0.03),  # 无风险利率（夏普比率计算）
            'max_trade_count': self.config.get('max_trade_count', 100),
            'position_rounding': self.config.get('position_rounding', True),  # 仓位是否取整
            'round_lot_size': self.config.get('round_lot_size', 100)  # 最小交易单位（股票100股）
        }
        
        # ===================== 市场状态判断配置 =====================
        self.market_condition = {
            'ranging_threshold': self.config.get('ranging_threshold', 0.015),
            'trend_threshold': self.config.get('trend_threshold', 0.03),
            'ema_short_period': self.config.get('ema_short_period', 5),
            'ema_long_period': self.config.get('ema_long_period', 20)
        }

    def _validate_config(self):
        """验证配置合法性（防御性编程）"""
        # 验证缠论参数
        for timeframe in ['weekly', 'daily', 'minute', 'default']:
            config = getattr(self, timeframe)
            assert config['fractal_sensitivity'] >= 1, f"{timeframe}分型敏感度必须≥1"
            assert config['pen_min_length'] >= 3, f"{timeframe}笔最小长度必须≥3"
            assert config['central_bank_min_length'] >= 3, f"{timeframe}中枢最小长度必须≥3"  # 统一参数名
        
        # 验证风险控制参数
        assert self.stop_loss['type'] in ['dynamic', 'fixed'], "止损类型必须是dynamic或fixed"
        assert 0 < self.stop_loss['fixed_ratio'] < 1, "固定止损比例必须在(0,1)之间"
        assert self.stop_loss['atr_period'] >= 5, "ATR周期必须≥5"
        assert self.stop_loss['atr_multiplier'] > 0, "ATR乘数必须>0"
        
        # 验证仓位参数
        assert 0 < self.position['max_single_position_ratio'] <= 1, "单票最大仓位必须在(0,1]之间"
        assert 0 < self.position['min_position_ratio'] <= self.position['max_single_position_ratio'], \
            "最小仓位必须≤最大仓位"
        
        # 验证交易成本
        assert 0 <= self.trade_cost['commission_rate'] < 0.01, "手续费率必须在[0,0.01)之间"
        assert 0 <= self.trade_cost['slippage_ratio'] < 0.01, "滑点率必须在[0,0.01)之间"
        
        # 验证回测参数
        assert self.backtest_config['initial_capital'] > 0, "初始资金必须>0"  # 引用修改后的配置名
        assert 0 <= self.backtest_config['risk_free_rate'] < 0.1, "无风险利率必须在[0,0.1)之间"  # 引用修改后的配置名
        
        # 日志输出关键配置
        logger.info(
            f"核心配置加载完成 - "
            f"周线分型敏感度={self.weekly['fractal_sensitivity']}, "
            f"止损类型={self.stop_loss['type']}, "
            f"初始资金={self.backtest_config['initial_capital']:.0f}, "  # 引用修改后的配置名
            f"最大单票仓位={self.position['max_single_position_ratio']:.1%}"
        )

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理（去重、缺失值填充、复权处理）"""
        df = df.copy()
        
        # 1. 去重（按日期）
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # 2. 缺失值处理
        missing_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in missing_cols:
            if col in df.columns:
                # 填充缺失值（前向填充+线性插值）
                df[col] = df[col].fillna(method='ffill').interpolate(method='linear')
        
        # 3. 日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 4. 复权处理（假设数据已复权，补充复权因子验证）
        if 'adj_factor' in df.columns:
            df['open'] = df['open'] * df['adj_factor']
            df['high'] = df['high'] * df['adj_factor']
            df['low'] = df['low'] * df['adj_factor']
            df['close'] = df['close'] * df['adj_factor']
        
        # 5. 验证缺失率
        total_rows = len(df)
        missing_ratio = df[missing_cols].isna().sum().sum() / (total_rows * len(missing_cols))
        if missing_ratio > self.data_validation['allow_missing_data_ratio']:
            logger.warning(f"数据缺失率过高: {missing_ratio:.1%}（阈值：{self.data_validation['allow_missing_data_ratio']:.1%}）")
        
        logger.debug(f"数据预处理完成: 原始{total_rows}条 → 处理后{len(df)}条")
        return df

    def _validate_input_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        """输入数据合法性验证"""
        if not self.data_validation['enabled']:
            return True, "数据验证已禁用"
        
        try:
            # 1. 非空验证
            if df is None or df.empty:
                return False, f"{timeframe}级别数据为空"
            
            # 2. 数据量验证
            if len(df) < self.data_validation['min_data_points']:
                return False, f"{timeframe}级别数据不足{self.data_validation['min_data_points']}条（当前{len(df)}条）"
            
            # 3. 必要列验证
            required_cols = ['date', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, f"{timeframe}级别数据缺失必要列：{missing_cols}"
            
            # 4. 日期格式验证
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_dates = df['date'].isna().sum()
                if invalid_dates > 0:
                    return False, f"{timeframe}级别数据包含{invalid_dates}个无效日期"
            
            # 5. 日期范围验证
            date_range = df['date'].max() - df['date'].min()
            if date_range.days > self.data_validation['max_date_range_days']:
                return False, f"{timeframe}级别数据范围过大（{date_range.days}天，阈值{self.data_validation['max_date_range_days']}天）"
            
            # 6. 价格有效性验证
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    return False, f"{timeframe}级别数据包含无效价格（{col}≤0）"
            
            logger.info(f"{timeframe}级别数据验证通过：{len(df)}条记录，日期范围{df['date'].min().strftime('%Y-%m-%d')}至{df['date'].max().strftime('%Y-%m-%d')}")
            return True, "数据验证通过"
        
        except Exception as e:
            error_msg = f"{timeframe}级别数据验证失败：{str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算平均真实波幅（ATR）- 用于动态止损
        :param df: K线数据（需包含high/low/close）
        :param period: ATR计算周期
        :return: ATR序列
        """
        df = df.copy()
        if len(df) < period:
            logger.warning(f"ATR计算数据不足：需{period}条，当前{len(df)}条，返回默认值0")
            return pd.Series([0.0] * len(df), index=df.index)
        
        # 计算真实波幅（TR）
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR（滚动平均）
        atr = df['true_range'].rolling(window=period, min_periods=1).mean()
        
        # 填充初始值（前period-1条用第一条ATR）
        atr.iloc[:period-1] = atr.iloc[period-1]
        
        logger.debug(f"ATR计算完成：周期{period}，最新值{atr.iloc[-1]:.4f}")
        return atr

    def calculate_dynamic_stop_loss(self, df: pd.DataFrame, position_type: str = 'long') -> pd.Series:
        """
        计算动态止损价（基于ATR）
        :param df: 带ATR的K线数据
        :param position_type: 仓位类型（long/short）
        :return: 动态止损价序列
        """
        df = df.copy()
        if 'atr' not in df.columns:
            df['atr'] = self.calculate_atr(df, self.stop_loss['atr_period'])
        
        stop_loss = pd.Series(index=df.index, dtype=float)
        multiplier = self.stop_loss['atr_multiplier']
        
        for i in range(len(df)):
            current_close = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr']
            
            if position_type == 'long':
                # 多头止损：收盘价 - ATR*乘数
                base_stop = current_close - current_atr * multiplier
                # 跟踪止损：若价格上涨，止损价上移
                if self.stop_loss['trailing_stop_enabled'] and i > 0:
                    prev_stop = stop_loss.iloc[i-1]
                    trailing_stop = current_close * (1 - self.stop_loss['trailing_stop_ratio'])
                    stop_loss.iloc[i] = max(base_stop, prev_stop, trailing_stop)
                else:
                    stop_loss.iloc[i] = base_stop
            else:
                # 空头止损：收盘价 + ATR*乘数
                base_stop = current_close + current_atr * multiplier
                if self.stop_loss['trailing_stop_enabled'] and i > 0:
                    prev_stop = stop_loss.iloc[i-1]
                    trailing_stop = current_close * (1 + self.stop_loss['trailing_stop_ratio'])
                    stop_loss.iloc[i] = min(base_stop, prev_stop, trailing_stop)
                else:
                    stop_loss.iloc[i] = base_stop
        
        # 止损价不能为负
        stop_loss = stop_loss.clip(lower=0.01)
        return stop_loss

    def calculate_fixed_stop_loss(self, df: pd.DataFrame, position_type: str = 'long') -> pd.Series:
        """计算固定比例止损价"""
        ratio = self.stop_loss['fixed_ratio']
        if position_type == 'long':
            return df['close'] * (1 - ratio)
        else:
            return df['close'] * (1 + ratio)

    def calculate_stop_loss_price(self, df: pd.DataFrame, position_type: str = 'long') -> pd.Series:
        """统一止损价计算入口"""
        if self.stop_loss['type'] == 'dynamic':
            return self.calculate_dynamic_stop_loss(df, position_type)
        else:
            return self.calculate_fixed_stop_loss(df, position_type)

    def calculate_fractal(self, df: pd.DataFrame, sensitivity: int = 3) -> pd.DataFrame:
        """
        计算顶分型/底分型（核心缠论指标）
        :param df: K线数据
        :param sensitivity: 分型敏感度（n=3表示前后3根K线对比）
        :return: 带分型标记的DataFrame
        """
        df = df.copy()
        n = sensitivity
        if len(df) < 2 * n + 1:
            logger.warning(f"分型计算数据不足：需{2*n+1}条，当前{len(df)}条，返回空标记")
            df['top_fractal'] = False
            df['bottom_fractal'] = False
            df['fractal_type'] = None
            return df
        
        # 初始化分型列
        df['top_fractal'] = False
        df['bottom_fractal'] = False
        df['fractal_type'] = None  # top/bottom/None
        
        # 顶分型：中间K线高点 > 前后n根K线高点，且不连续同高
        for i in range(n, len(df) - n):
            # 窗口内K线（前后n根+当前）
            window = df.iloc[i-n:i+n+1]
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # 顶分型条件
            if (current_high == window['high'].max()) and \
               (sum(window['high'] == current_high) == 1):  # 唯一高点
                df.iloc[i, df.columns.get_loc('top_fractal')] = True
                df.iloc[i, df.columns.get_loc('fractal_type')] = 'top'
            
            # 底分型条件
            if (current_low == window['low'].min()) and \
               (sum(window['low'] == current_low) == 1):  # 唯一低点
                df.iloc[i, df.columns.get_loc('bottom_fractal')] = True
                df.iloc[i, df.columns.get_loc('fractal_type')] = 'bottom'
        
        # 统计分型数量
        top_count = df['top_fractal'].sum()
        bottom_count = df['bottom_fractal'].sum()
        logger.debug(f"分型计算完成：顶分型{top_count}个，底分型{bottom_count}个，敏感度{n}")
        return df

    def calculate_pen(self, df: pd.DataFrame, min_length: int = 5, fractal_sensitivity: int = 3) -> pd.DataFrame:
        """
        计算笔（基于分型，缠论基本趋势单元）
        规则：顶底分型交替出现，中间至少min_length根K线
        :param df: 带分型标记的K线数据
        :param min_length: 笔最小长度（K线数）
        :param fractal_sensitivity: 分型敏感度（用于过滤无效分型）
        :return: 带笔标记的DataFrame
        """
        df = df.copy()
        # 初始化笔相关列
        df['pen_type'] = None  # up/down/None
        df['pen_start'] = False
        df['pen_end'] = False
        df['pen_id'] = -1  # 笔编号（用于关联线段）
        
        # 提取有效分型点（去重、排序）- 核心修复：将'index'改为'orig_idx'避免冲突
        fractal_points = []
        for idx, row in df.iterrows():
            if row['fractal_type'] in ['top', 'bottom']:
                fractal_points.append({
                    'orig_idx': idx,  # 存储原始索引（关键修复）
                    'type': row['fractal_type'],
                    'price': row['high'] if row['fractal_type'] == 'top' else row['low'],
                    'date': row['date']
                })
        
        if len(fractal_points) < 2:
            logger.warning("笔计算：有效分型点不足2个，无法构成笔")
            return df
        
        # 过滤连续同类型分型（保留极端值）
        filtered_fractals = [fractal_points[0]]
        for fp in fractal_points[1:]:
            last_fp = filtered_fractals[-1]
            if fp['type'] != last_fp['type']:
                filtered_fractals.append(fp)
            else:
                # 同类型分型：保留价格更极端的（顶分型取更高，底分型取更低）
                if (fp['type'] == 'top' and fp['price'] > last_fp['price']) or \
                   (fp['type'] == 'bottom' and fp['price'] < last_fp['price']):
                    filtered_fractals[-1] = fp
        
        # 生成笔（交替分型+满足最小长度）
        pen_id = 0
        for i in range(1, len(filtered_fractals)):
            prev_fp = filtered_fractals[i-1]
            curr_fp = filtered_fractals[i]
            
            # 计算笔长度（K线数）- 核心修复：使用'orig_idx'获取原始索引
            pen_length = curr_fp['orig_idx'] - prev_fp['orig_idx']
            if pen_length < min_length:
                logger.debug(f"笔过滤：长度{pen_length} < 最小长度{min_length}，跳过")
                continue
            
            # 确定笔方向：底分型→顶分型=up，顶分型→底分型=down
            if prev_fp['type'] == 'bottom' and curr_fp['type'] == 'top':
                pen_type = 'up'
            elif prev_fp['type'] == 'top' and curr_fp['type'] == 'bottom':
                pen_type = 'down'
            else:
                continue  # 异常情况，跳过
            
            # 标记笔的起止点 - 核心修复：使用'orig_idx'定位原始索引
            df.at[prev_fp['orig_idx'], 'pen_start'] = True
            df.at[curr_fp['orig_idx'], 'pen_end'] = True
            df.at[curr_fp['orig_idx'], 'pen_type'] = pen_type
            df.at[curr_fp['orig_idx'], 'pen_id'] = pen_id
            
            # 填充笔编号（起止点之间的K线）
            df.loc[prev_fp['orig_idx']:curr_fp['orig_idx'], 'pen_id'] = pen_id
            pen_id += 1
        
        logger.debug(f"笔计算完成：有效笔数量{pen_id}支，最小长度{min_length}")
        return df

    def calculate_segment(self, df: pd.DataFrame, min_length: int = 3) -> pd.DataFrame:
        """
        计算线段（基于笔，更高级别趋势单元）
        规则：3笔构成一段，需满足"笔破坏"条件
        :param df: 带笔标记的K线数据
        :param min_length: 线段最小长度（笔数）
        :return: 带线段标记的DataFrame
        """
        df = df.copy()
        # 初始化线段相关列
        df['segment_type'] = None  # up/down/None
        df['segment_start'] = False
        df['segment_end'] = False
        df['segment_id'] = -1
        
        # 提取笔的端点（pen_end标记的点）- 核心修复：保留原始索引（不reset_index或使用index_col）
        pen_ends = df[df['pen_end']].copy()
        if len(pen_ends) < 3:
            logger.warning("线段计算：有效笔不足3支，无法构成线段")
            return df
        
        # 生成线段（3笔为一段，满足方向交替）
        segment_id = 0
        # 核心修复：使用iloc遍历，通过name获取原始索引
        for i in range(2, len(pen_ends)):
            # 取3笔：i-2, i-1, i - 核心修复：通过name获取原始索引
            pen1 = pen_ends.iloc[i-2]
            pen2 = pen_ends.iloc[i-1]
            pen3 = pen_ends.iloc[i]
            
            # 核心修复：获取原始索引（pen_ends的index是原始df的索引）
            pen1_orig_idx = pen_ends.index[i-2]
            pen2_orig_idx = pen_ends.index[i-1]
            pen3_orig_idx = pen_ends.index[i]
            
            # 验证笔方向交替
            if pen1['pen_type'] == pen2['pen_type'] or pen2['pen_type'] == pen3['pen_type']:
                logger.debug(f"线段过滤：笔方向未交替（{pen1['pen_type']}→{pen2['pen_type']}→{pen3['pen_type']}）")
                continue
            
            # 验证线段长度（笔数）
            if (i - (i-2) + 1) < min_length:
                logger.debug(f"线段过滤：笔数{3} < 最小长度{min_length}")
                continue
            
            # 确定线段方向：基于首尾笔的价格
            start_price = pen1['high'] if pen1['fractal_type'] == 'top' else pen1['low']
            end_price = pen3['high'] if pen3['fractal_type'] == 'top' else pen3['low']
            segment_type = 'up' if end_price > start_price else 'down'
            
            # 标记线段起止点 - 核心修复：使用原始索引定位
            df.at[pen1_orig_idx, 'segment_start'] = True
            df.at[pen3_orig_idx, 'segment_end'] = True
            df.at[pen3_orig_idx, 'segment_type'] = segment_type
            df.at[pen3_orig_idx, 'segment_id'] = segment_id
            
            # 填充线段编号（起止点之间的K线）
            df.loc[pen1_orig_idx:pen3_orig_idx, 'segment_id'] = segment_id
            segment_id += 1
        
        logger.debug(f"线段计算完成：有效线段数量{segment_id}段，最小笔数{min_length}")
        return df

    def calculate_central_bank(self, df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
        """
        计算中枢（缠论核心，价格波动重叠区间）
        规则：3段线段重叠，重叠区间为中枢，长度≥min_length根K线
        :param df: 带线段标记的K线数据
        :param min_length: 中枢最小长度（K线数）
        :return: 带中枢标记的DataFrame
        """
        df = df.copy()
        # 初始化中枢相关列
        df['central_bank'] = False
        df['central_bank_high'] = np.nan  # 中枢上沿
        df['central_bank_low'] = np.nan   # 中枢下沿
        df['central_bank_id'] = -1
        df['central_bank_type'] = None    # 上升/下降/盘整中枢
        
        # 提取线段端点（segment_end标记的点）- 核心修复：保留原始索引
        segment_ends = df[df['segment_end']].copy()
        if len(segment_ends) < 4:  # 至少3段线段构成中枢
            logger.warning("中枢计算：有效线段不足4段，无法构成中枢")
            return df
        
        # 生成中枢（3段线段重叠）
        cb_id = 0
        # 核心修复：使用iloc遍历，通过index获取原始索引
        for i in range(3, len(segment_ends)):
            # 取3段线段：i-3→i-2, i-2→i-1, i-1→i - 核心修复：获取原始索引
            seg1_end_orig_idx = segment_ends.index[i-2]
            seg2_end_orig_idx = segment_ends.index[i-1]
            seg3_end_orig_idx = segment_ends.index[i]
            seg1_start_orig_idx = segment_ends.index[i-3]
            
            # 计算3段线段的价格范围
            seg1_high = df.loc[seg1_start_orig_idx:seg1_end_orig_idx, 'high'].max()
            seg1_low = df.loc[seg1_start_orig_idx:seg1_end_orig_idx, 'low'].min()
            seg2_high = df.loc[seg1_end_orig_idx:seg2_end_orig_idx, 'high'].max()
            seg2_low = df.loc[seg1_end_orig_idx:seg2_end_orig_idx, 'low'].min()
            seg3_high = df.loc[seg2_end_orig_idx:seg3_end_orig_idx, 'high'].max()
            seg3_low = df.loc[seg2_end_orig_idx:seg3_end_orig_idx, 'low'].min()
            
            # 计算重叠区间（中枢范围）
            cb_high = min(seg1_high, seg2_high, seg3_high)  # 上沿=三段最高点的最小值
            cb_low = max(seg1_low, seg2_low, seg3_low)      # 下沿=三段最低点的最大值
            
            # 验证有效中枢：上沿>下沿，且长度达标
            cb_length = seg3_end_orig_idx - seg1_start_orig_idx  # 中枢长度（K线数）
            if cb_high <= cb_low:
                logger.debug(f"中枢过滤：无有效重叠（上沿{cb_high:.2f} ≤ 下沿{cb_low:.2f}）")
                continue
            if cb_length < min_length:
                logger.debug(f"中枢过滤：长度{cb_length} < 最小长度{min_length}")
                continue
            
            # 确定中枢类型（基于线段方向）
            seg1_type = segment_ends.iloc[i-3]['segment_type']
            seg2_type = segment_ends.iloc[i-2]['segment_type']
            seg3_type = segment_ends.iloc[i-1]['segment_type']
            if seg1_type == 'up' and seg2_type == 'down' and seg3_type == 'up':
                cb_type = 'ascending'  # 上升中枢
            elif seg1_type == 'down' and seg2_type == 'up' and seg3_type == 'down':
                cb_type = 'descending'  # 下降中枢
            else:
                cb_type = 'sideways'  # 盘整中枢
            
            # 标记中枢区域 - 核心修复：使用原始索引定位
            df.loc[seg1_start_orig_idx:seg3_end_orig_idx, 'central_bank'] = True
            df.loc[seg1_start_orig_idx:seg3_end_orig_idx, 'central_bank_high'] = cb_high
            df.loc[seg1_start_orig_idx:seg3_end_orig_idx, 'central_bank_low'] = cb_low
            df.loc[seg1_start_orig_idx:seg3_end_orig_idx, 'central_bank_id'] = cb_id
            df.loc[seg1_start_orig_idx:seg3_end_orig_idx, 'central_bank_type'] = cb_type
            
            cb_id += 1
        
        logger.debug(f"中枢计算完成：有效中枢数量{cb_id}个，最小长度{min_length}")
        return df

    def calculate_divergence(self, df: pd.DataFrame, threshold: float = 0.015) -> pd.DataFrame:
        """
        计算背离（价格与指标的背离，缠论买卖点核心）
        支持：顶背离（价格新高+指标不新高）、底背离（价格新低+指标不新低）
        :param df: 带中枢/笔标记的K线数据
        :param threshold: 背离阈值（价格与指标的差值比例）
        :return: 带背离标记的DataFrame
        """
        df = df.copy()
        # 初始化背离列
        df['divergence'] = None  # top/bottom/None
        df['divergence_strength'] = 0.0  # 背离强度（0-1）
        
        # 计算MACD（默认参数：12,26,9）- 背离判断指标
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        # 提取关键价格点（顶分型/底分型）
        top_points = df[df['top_fractal']].copy()
        bottom_points = df[df['bottom_fractal']].copy()
        
        # 检测顶背离（价格新高，MACD不新高）
        if len(top_points) >= 2:
            for i in range(1, len(top_points)):
                prev_top = top_points.iloc[i-1]
                curr_top = top_points.iloc[i]
                
                # 价格新高
                if curr_top['high'] > prev_top['high']:
                    # MACD不新高
                    if curr_top['macd'] < prev_top['macd']:
                        # 计算背离强度（价格涨幅 - MACD涨幅）
                        price_gain = (curr_top['high'] - prev_top['high']) / prev_top['high']
                        macd_drop = (prev_top['macd'] - curr_top['macd']) / abs(prev_top['macd']) if prev_top['macd'] != 0 else 0
                        divergence_strength = min(1.0, (price_gain + macd_drop) / 2)
                        
                        if divergence_strength >= threshold:
                            # 核心修复：使用原始索引定位
                            df.at[top_points.index[i], 'divergence'] = 'top'
                            df.at[top_points.index[i], 'divergence_strength'] = divergence_strength
        
        # 检测底背离（价格新低，MACD不新低）
        if len(bottom_points) >= 2:
            for i in range(1, len(bottom_points)):
                prev_bottom = bottom_points.iloc[i-1]
                curr_bottom = bottom_points.iloc[i]
                
                # 价格新低
                if curr_bottom['low'] < prev_bottom['low']:
                    # MACD不新低
                    if curr_bottom['macd'] > prev_bottom['macd']:
                        # 计算背离强度
                        price_drop = (prev_bottom['low'] - curr_bottom['low']) / prev_bottom['low']
                        macd_rise = (curr_bottom['macd'] - prev_bottom['macd']) / abs(prev_bottom['macd']) if prev_bottom['macd'] != 0 else 0
                        divergence_strength = min(1.0, (price_drop + macd_rise) / 2)
                        
                        if divergence_strength >= threshold:
                            # 核心修复：使用原始索引定位
                            df.at[bottom_points.index[i], 'divergence'] = 'bottom'
                            df.at[bottom_points.index[i], 'divergence_strength'] = divergence_strength
        
        logger.debug(f"背离计算完成：顶背离{sum(df['divergence'] == 'top')}个，底背离{sum(df['divergence'] == 'bottom')}个")
        return df

    def determine_market_condition(self, df: pd.DataFrame) -> pd.Series:
        """
        判断市场状态（趋势向上/趋势向下/盘整）
        基于：EMA均线、中枢位置、价格波动幅度
        :return: 市场状态序列（trending_up/trending_down/ranging）
        """
        df = df.copy()
        short_period = self.market_condition['ema_short_period']
        long_period = self.market_condition['ema_long_period']
        
        # 计算EMA均线
        df['ema_short'] = df['close'].ewm(span=short_period, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long_period, adjust=False).mean()
        
        # 计算价格波动幅度（20日高低点比例）
        df['20d_high'] = df['high'].rolling(window=20).max()
        df['20d_low'] = df['low'].rolling(window=20).min()
        df['volatility_ratio'] = (df['20d_high'] - df['20d_low']) / df['close']
        
        # 初始化市场状态
        market_condition = pd.Series('ranging', index=df.index)
        
        # 趋势判断逻辑
        for i in range(long_period, len(df)):
            curr_close = df.iloc[i]['close']
            ema_short = df.iloc[i]['ema_short']
            ema_long = df.iloc[i]['ema_long']
            volatility = df.iloc[i]['volatility_ratio']
            in_central = df.iloc[i]['central_bank'] if 'central_bank' in df.columns else False
            
            # 趋势向上：EMA短期>长期 + 价格在中枢上沿之上 + 波动幅度达标
            if (ema_short > ema_long) and \
               (not in_central or curr_close > df.iloc[i]['central_bank_high']) and \
               (volatility >= self.market_condition['trend_threshold']):
                market_condition.iloc[i] = 'trending_up'
            
            # 趋势向下：EMA短期<长期 + 价格在中枢下沿之下 + 波动幅度达标
            elif (ema_short < ema_long) and \
                 (not in_central or curr_close < df.iloc[i]['central_bank_low']) and \
                 (volatility >= self.market_condition['trend_threshold']):
                market_condition.iloc[i] = 'trending_down'
            
            # 盘整：波动幅度<震荡阈值，或在中枢内
            else:
                market_condition.iloc[i] = 'ranging'
        
        logger.debug(f"市场状态判断完成：趋势向上{sum(market_condition == 'trending_up')}条，趋势向下{sum(market_condition == 'trending_down')}条，盘整{sum(market_condition == 'ranging')}条")
        return market_condition

    def calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        计算交易信号强度（0-100，越高信号越可靠）
        权重因子：分型强度、笔结束信号、线段方向、中枢位置、背离强度、市场状态
        """
        df = df.copy()
        signal_strength = pd.Series(50.0, index=df.index)  # 初始值50
        
        for i in range(len(df)):
            row = df.iloc[i]
            score = 50
            
            # 1. 分型强度（顶分型-10，底分型+10）
            if row['fractal_type'] == 'top':
                score -= 10
            elif row['fractal_type'] == 'bottom':
                score += 10
            
            # 2. 笔结束信号（笔结束+15）
            if row['pen_end']:
                if row['pen_type'] == 'up':
                    score += 15
                else:
                    score -= 15
            
            # 3. 线段方向（上升线段+20，下降线段-20）
            if row['segment_type'] == 'up':
                score += 20
            elif row['segment_type'] == 'down':
                score -= 20
            
            # 4. 中枢位置（突破上沿+25，跌破下沿-25，中枢内+5）
            if row['central_bank']:
                if row['close'] > row['central_bank_high']:
                    score += 25
                elif row['close'] < row['central_bank_low']:
                    score -= 25
                else:
                    score += 5
            
            # 5. 背离强度（底背离+30*强度，顶背离-30*强度）
            if row['divergence'] == 'bottom':
                score += 30 * row['divergence_strength']
            elif row['divergence'] == 'top':
                score -= 30 * row['divergence_strength']
            
            # 6. 市场状态（趋势向上+15，趋势向下-15）
            if row['market_condition'] == 'trending_up':
                score += 15
            elif row['market_condition'] == 'trending_down':
                score -= 15
            
            # 限制分数在0-100之间
            signal_strength.iloc[i] = max(0.0, min(100.0, score))
        
        return signal_strength

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（买入/卖出/持有）
        信号逻辑：
        - 买入信号：底背离 + 笔结束（up） + 信号强度≥60
        - 卖出信号：顶背离 + 笔结束（down） + 信号强度≤40
        - 持有信号：无买卖信号时
        """
        df = df.copy()
        df['signal'] = 'hold'  # 初始信号：持有
        
        # 买入信号条件
        buy_condition = (
            (df['divergence'] == 'bottom') &
            (df['pen_end'] == True) &
            (df['pen_type'] == 'up') &
            (df['signal_strength'] >= 60)
        )
        
        # 卖出信号条件
        sell_condition = (
            (df['divergence'] == 'top') &
            (df['pen_end'] == True) &
            (df['pen_type'] == 'down') &
            (df['signal_strength'] <= 40)
        )
        
        df.loc[buy_condition, 'signal'] = 'buy'
        df.loc[sell_condition, 'signal'] = 'sell'
        
        # 去重连续信号（避免重复买卖）
        df['signal'] = df['signal'].mask(df['signal'] == df['signal'].shift(1))
        
        # 统计信号数量
        buy_count = sum(df['signal'] == 'buy')
        sell_count = sum(df['signal'] == 'sell')
        logger.info(f"交易信号生成完成：买入信号{buy_count}个，卖出信号{sell_count}个")
        return df

    def calculate_position_size(self, df: pd.DataFrame, signal_strength: float, current_capital: float) -> int:
        """
        计算仓位大小（基于信号强度和资金管理规则）
        :param current_capital: 当前可用资金
        :return: 交易数量（取整到最小交易单位）
        """
        current_price = df.iloc[-1]['close']
        if current_price <= 0:
            logger.warning("仓位计算：当前价格≤0，返回0")
            return 0
        
        # 基础仓位比例（信号强度加权）
        base_ratio = self.position['min_position_ratio'] + \
                    (self.position['max_single_position_ratio'] - self.position['min_position_ratio']) * \
                    (signal_strength / 100) * self.position['signal_strength_weight']
        base_ratio = min(self.position['max_single_position_ratio'], max(self.position['min_position_ratio'], base_ratio))
        
        # 可交易金额
        trade_amount = current_capital * base_ratio
        
        # 扣除交易成本（预估）
        estimated_commission = max(self.trade_cost['min_commission'], trade_amount * self.trade_cost['commission_rate'])
        estimated_slippage = trade_amount * self.trade_cost['slippage_ratio']
        net_trade_amount = trade_amount - estimated_commission - estimated_slippage
        
        # 计算交易数量
        position_size = net_trade_amount / current_price
        if self.backtest_config['position_rounding']:  # 引用修改后的配置名
            # 取整到最小交易单位（如100股）
            position_size = int(position_size // self.backtest_config['round_lot_size'] * self.backtest_config['round_lot_size'])  # 引用修改后的配置名
        else:
            position_size = round(position_size, 2)
        
        # 最小交易数量限制
        position_size = max(self.backtest_config['round_lot_size'], position_size) if position_size > 0 else 0  # 引用修改后的配置名
        logger.debug(f"仓位计算完成：信号强度{signal_strength:.1f}，仓位比例{base_ratio:.1%}，交易数量{position_size}")
        return int(position_size)

    def calculate_trade_cost(self, trade_amount: float, position_size: int, price: float) -> Dict[str, float]:
        """计算实际交易成本（手续费+滑点）"""
        # 手续费（按金额计算，不低于最低手续费）
        commission = max(self.trade_cost['min_commission'], trade_amount * self.trade_cost['commission_rate'])
        
        # 滑点（按数量计算）
        slippage = position_size * price * self.trade_cost['slippage_ratio']
        
        total_cost = commission + slippage
        return {
            'commission': commission,
            'slippage': slippage,
            'total_cost': total_cost
        }

    def run_backtest(self, df: pd.DataFrame, timeframe: str = 'daily') -> Dict[str, Any]:
        """
        完整回测流程（改名避免与属性名冲突）
        :param df: 带交易信号的K线数据
        :param timeframe: 时间级别
        :return: 回测结果（交易记录、绩效指标、组合价值曲线）
        """
        df = df.copy()
        logger.info(f"开始{timeframe}级别回测：初始资金{self.backtest_config['initial_capital']:.0f}，数据区间{df['date'].min().strftime('%Y-%m-%d')}至{df['date'].max().strftime('%Y-%m-%d')}")
        
        # 初始化回测状态
        cash = self.backtest_config['initial_capital']  # 现金（引用修改后的配置名）
        position_size = 0  # 持仓数量
        avg_cost = 0.0     # 平均持仓成本
        trade_records = []  # 交易记录
        portfolio_value = []  # 组合价值（现金+持仓市值）
        trade_count = 0     # 交易次数
        
        # 遍历K线执行回测
        for i in range(len(df)):
            row = df.iloc[i]
            current_date = row['date']
            current_price = row['close']
            current_signal = row['signal']
            current_strength = row['signal_strength']
            
            # 计算当前组合价值
            position_value = position_size * current_price
            total_value = cash + position_value
            portfolio_value.append({
                'date': current_date,
                'cash': cash,
                'position_size': position_size,
                'position_value': position_value,
                'total_value': total_value
            })
            
            # 跳过无信号或交易次数超限的情况
            if current_signal == 'hold' or trade_count >= self.backtest_config['max_trade_count']:  # 引用修改后的配置名
                continue
            
            # 执行买入信号
            if current_signal == 'buy' and position_size == 0:
                # 计算仓位大小
                position_size = self.calculate_position_size(df.iloc[:i+1], current_strength, cash)
                if position_size == 0:
                    logger.debug(f"买入信号：仓位计算为0，跳过交易（日期：{current_date.strftime('%Y-%m-%d')}）")
                    continue
                
                # 计算交易金额和成本
                trade_amount = position_size * current_price
                costs = self.calculate_trade_cost(trade_amount, position_size, current_price)
                
                # 验证资金是否充足
                if cash < trade_amount + costs['total_cost']:
                    logger.warning(f"买入信号：资金不足（可用{cash:.2f}，需{trade_amount + costs['total_cost']:.2f}），跳过交易")
                    position_size = 0
                    continue
                
                # 更新账户状态
                cash -= (trade_amount + costs['total_cost'])
                avg_cost = (trade_amount + costs['total_cost']) / position_size
                trade_count += 1
                
                # 记录交易
                trade_records.append({
                    'trade_id': trade_count,
                    'date': current_date,
                    'signal': 'buy',
                    'price': current_price,
                    'position_size': position_size,
                    'trade_amount': trade_amount,
                    'commission': costs['commission'],
                    'slippage': costs['slippage'],
                    'total_cost': costs['total_cost'],
                    'cash_after_trade': cash,
                    'avg_cost': avg_cost
                })
                logger.info(f"买入交易执行：日期{current_date.strftime('%Y-%m-%d')}，价格{current_price:.2f}，数量{position_size}，总成本{costs['total_cost']:.2f}，剩余现金{cash:.2f}")
            
            # 执行卖出信号
            elif current_signal == 'sell' and position_size > 0:
                # 计算交易金额和成本
                trade_amount = position_size * current_price
                costs = self.calculate_trade_cost(trade_amount, position_size, current_price)
                
                # 计算盈亏
                gross_profit = trade_amount - (avg_cost * position_size)
                net_profit = gross_profit - costs['total_cost']
                profit_ratio = (net_profit / (avg_cost * position_size)) * 100 if avg_cost > 0 else 0
                
                # 更新账户状态
                cash += (trade_amount - costs['total_cost'])
                trade_count += 1
                
                # 记录交易
                trade_records.append({
                    'trade_id': trade_count,
                    'date': current_date,
                    'signal': 'sell',
                    'price': current_price,
                    'position_size': position_size,
                    'trade_amount': trade_amount,
                    'commission': costs['commission'],
                    'slippage': costs['slippage'],
                    'total_cost': costs['total_cost'],
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'profit_ratio': profit_ratio,
                    'cash_after_trade': cash
                })
                logger.info(f"卖出交易执行：日期{current_date.strftime('%Y-%m-%d')}，价格{current_price:.2f}，数量{position_size}，净利润{net_profit:.2f}（{profit_ratio:.1f}%），现金{cash:.2f}")
                
                # 清空仓位
                position_size = 0
                avg_cost = 0.0
        
        # 回测结束：计算最终组合价值
        final_position_value = position_size * df.iloc[-1]['close']
        final_total_value = cash + final_position_value
        
        # 计算绩效指标
        performance = self._calculate_performance_metrics(
            portfolio_value=portfolio_value,
            trade_records=trade_records,
            initial_capital=self.backtest_config['initial_capital'],  # 引用修改后的配置名
            risk_free_rate=self.backtest_config['risk_free_rate']  # 引用修改后的配置名
        )
        
        # 整理回测结果
        backtest_result = {
            'basic_info': {
                'timeframe': timeframe,
                'start_date': df['date'].min().strftime('%Y-%m-%d'),
                'end_date': df['date'].max().strftime('%Y-%m-%d'),
                'data_count': len(df),
                'initial_capital': self.backtest_config['initial_capital'],  # 引用修改后的配置名
                'final_capital': final_total_value,
                'trade_count': trade_count
            },
            'performance_metrics': performance,
            'trade_records': pd.DataFrame(trade_records),
            'portfolio_value': pd.DataFrame(portfolio_value)
        }
        
        logger.info(f"回测完成：最终资金{final_total_value:.2f}，总收益率{performance['total_return']:.1%}，年化收益率{performance['annual_return']:.1%}，最大回撤{performance['max_drawdown']:.1%}")
        return backtest_result

    def _calculate_performance_metrics(self, portfolio_value: List[Dict], trade_records: List[Dict], initial_capital: float, risk_free_rate: float) -> Dict[str, float]:
        """计算回测绩效指标"""
        if not portfolio_value:
            return {k: 0.0 for k in ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate', 'profit_factor']}
        
        # 转换为DataFrame便于计算
        pv_df = pd.DataFrame(portfolio_value)
        pv_df['date'] = pd.to_datetime(pv_df['date'])
        pv_df = pv_df.sort_values('date')
        
        # 1. 收益率指标
        total_return = (pv_df['total_value'].iloc[-1] - initial_capital) / initial_capital
        days = (pv_df['date'].iloc[-1] - pv_df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 2. 最大回撤
        pv_df['cumulative_max'] = pv_df['total_value'].cummax()
        pv_df['drawdown'] = (pv_df['total_value'] - pv_df['cumulative_max']) / pv_df['cumulative_max']
        max_drawdown = abs(pv_df['drawdown'].min())
        
        # 3. 夏普比率（日收益率）
        pv_df['daily_return'] = pv_df['total_value'].pct_change()
        daily_returns = pv_df['daily_return'].dropna()
        if len(daily_returns) < 2:
            sharpe_ratio = 0.0
        else:
            excess_returns = daily_returns - (risk_free_rate / 365)
            sharpe_ratio = np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
        
        # 4. 交易绩效指标（胜率、盈亏比）
        if not trade_records:
            win_rate = 0.0
            profit_factor = 0.0
        else:
            trades_df = pd.DataFrame(trade_records)
            sell_trades = trades_df[trades_df['signal'] == 'sell']
            if len(sell_trades) == 0:
                win_rate = 0.0
                profit_factor = 0.0
            else:
                # 胜率
                winning_trades = sell_trades[sell_trades['net_profit'] > 0]
                win_rate = len(winning_trades) / len(sell_trades)
                
                # 盈亏比
                total_profit = winning_trades['net_profit'].sum()
                total_loss = abs(sell_trades[sell_trades['net_profit'] <= 0]['net_profit'].sum())
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_count': len(trade_records) // 2  # 买入卖出为一对交易
        }

    def calculate(self, 
                 df: pd.DataFrame, 
                 timeframe: str = 'daily',
                 fractal_sensitivity: Optional[int] = None,
                 pen_min_length: Optional[int] = None,
                 central_bank_min_length: Optional[int] = None,  # 核心修复：参数名改为central_bank_min_length
                 segment_min_length: Optional[int] = None,
                 divergence_threshold: Optional[float] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        完整计算入口（核心修复：参数名与backtester.py一致）
        :param df: 原始K线数据
        :param timeframe: 时间级别（daily/weekly/minute）
        :param fractal_sensitivity: 分型敏感度（动态覆盖配置）
        :param pen_min_length: 笔最小长度（动态覆盖配置）
        :param central_bank_min_length: 中枢最小长度（动态覆盖配置，与backtester.py匹配）
        :param segment_min_length: 线段最小长度（动态覆盖配置）
        :param divergence_threshold: 背离阈值（动态覆盖配置）
        :return: (带所有指标的DataFrame, 回测结果)
        """
        # 1. 数据验证
        valid, msg = self._validate_input_data(df, timeframe)
        if not valid:
            logger.error(f"计算终止：{msg}")
            return df, None
        
        # 2. 数据预处理
        df = self._preprocess_data(df)
        
        # 3. 确定参数（优先级：方法参数 > 时间级别配置 > 默认配置）
        if timeframe == 'weekly':
            config = self.weekly
        elif timeframe == 'minute':
            config = self.minute
        else:  # daily
            config = self.daily
        
        # 动态参数覆盖（核心修复：参数名改为central_bank_min_length）
        fs = fractal_sensitivity or config['fractal_sensitivity']
        pm = pen_min_length or config['pen_min_length']
        cm = central_bank_min_length or config['central_bank_min_length']  # 统一参数名
        sm = segment_min_length or config['segment_min_length']
        dt = divergence_threshold or config['divergence_threshold']
        
        logger.info(
            f"开始{timeframe}级别完整计算 - "
            f"分型敏感度={fs}, 笔最小长度={pm}, 中枢最小长度={cm}, "
            f"线段最小长度={sm}, 背离阈值={dt:.3f}"
        )
        
        # 4. 分步计算缠论指标（核心流程）
        df = self.calculate_fractal(df, sensitivity=fs)          # 分型
        df = self.calculate_pen(df, min_length=pm, fractal_sensitivity=fs)  # 笔
        df = self.calculate_segment(df, min_length=sm)           # 线段
        df = self.calculate_central_bank(df, min_length=cm)      # 中枢（使用统一后的参数）
        df = self.calculate_divergence(df, threshold=dt)         # 背离
        df['market_condition'] = self.determine_market_condition(df)  # 市场状态
        df['signal_strength'] = self.calculate_signal_strength(df)  # 信号强度
        df = self.generate_signals(df)                           # 交易信号
        df['stop_loss_price'] = self.calculate_stop_loss_price(df, position_type='long')  # 止损价
        
        # 5. 执行回测（调用改名后的方法）
        backtest_result = self.run_backtest(df, timeframe=timeframe)
        
        logger.info(f"{timeframe}级别完整计算完成：共{len(df)}条数据，生成{len(backtest_result['trade_records'])}条交易记录")
        return df, backtest_result


if __name__ == "__main__":
    """测试入口"""
    # 构造测试K线数据
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    np.random.seed(42)
    close = np.random.randn(n).cumsum() + 100
    high = close + np.random.randn(n) * 2
    low = close - np.random.randn(n) * 2
    open_ = np.random.choice([high, low], size=n).T[0]
    volume = np.random.randint(1000, 10000, size=n)
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # 初始化配置
    test_config = {
        'chanlun': {
            'weekly_fractal_sensitivity': 3,
            'weekly_pen_min_length': 5,
            'weekly_central_bank_min_length': 5,  # 统一参数名
            'daily_fractal_sensitivity': 3,
            'daily_pen_min_length': 5,
            'daily_central_bank_min_length': 5  # 统一参数名
        },
        'risk_management': {
            'stop_loss_type': 'dynamic',
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'max_single_position_ratio': 0.8
        },
        'initial_capital': 100000.0
    }
    
    # 初始化计算器并执行计算
    calculator = ChanlunCalculator(config=test_config)
    result_df, backtest_res = calculator.calculate(
        df=test_df,
        timeframe='daily',
        fractal_sensitivity=3,
        pen_min_length=5,
        central_bank_min_length=5  # 传递统一后的参数名
    )
    
    # 输出结果摘要
    print(f"\n计算结果摘要：")
    print(f"数据条数：{len(result_df)}")
    print(f"顶分型数量：{result_df['top_fractal'].sum()}")
    print(f"底分型数量：{result_df['bottom_fractal'].sum()}")
    print(f"有效笔数量：{result_df['pen_id'].nunique() - 1}")
    print(f"有效中枢数量：{result_df['central_bank_id'].nunique() - 1}")
    print(f"买入信号数量：{sum(result_df['signal'] == 'buy')}")
    print(f"卖出信号数量：{sum(result_df['signal'] == 'sell')}")
    print(f"最终资金：{backtest_res['basic_info']['final_capital']:.2f}")
    print(f"总收益率：{backtest_res['performance_metrics']['total_return']:.1%}")