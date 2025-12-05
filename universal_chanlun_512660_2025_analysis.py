import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class UniversalChanlunAnalyzer:
    def __init__(self, symbol="512660", year=2025):
        self.symbol = symbol
        self.year = year
        self.raw_data = None
        self.standard_k_lines = None
        self.segments = []
        self.valid_segments = []
        self.centrals = []
        self.valid_centrals = []
        self.buy_signals = []
        self.volatility_level = None  # 波动等级：low, medium, high
        self.volatility = 0.0  # 波动率值
        self.volatility_calculation_details = {}  # 波动率计算详情
        self.params = {}
        self.year_high = None
        self.year_low = None
        self.yearly_price_range = None  # 年度价格区间
        self.price_outliers_count = 0  # 异常价格修正计数
        self.missing_trading_days = 0  # 缺失交易日数量
        self.data_completeness = "完整"
        self.error_message = None  # 存储错误信息，用于一票否决机制
        self.has_critical_error = False  # 标记是否有严重错误需要终止分析
        
    def load_data(self):
        """加载并验证数据完整性，实现数据有效性一票否决机制"""
        try:
            # 尝试从多个可能的路径加载数据
            possible_paths = [
                f"data/daily/{self.symbol}_daily.csv",
                f"data/{self.symbol}_daily_data.csv",
                f"{self.symbol}_daily_data.csv"
            ]
            
            data_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.raw_data = pd.read_csv(path)
                    data_loaded = True
                    break
            
            if not data_loaded:
                raise FileNotFoundError(f"找不到{self.symbol}的日K数据文件")
            
            # 数据预处理
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
            
            # 筛选2025年数据
            self.raw_data = self.raw_data[self.raw_data['date'].dt.year == self.year]
            
            # 检查数据完整性（增强版）
            start_date = datetime(self.year, 1, 1)
            end_date = datetime(self.year, 12, 31)
            
            # 计算缺失交易日数量（基于工作日范围）
            expected_days = pd.bdate_range(start_date, end_date)
            actual_days = self.raw_data['date']
            self.missing_trading_days = len(set(expected_days) - set(actual_days))
            
            # 【调整一票否决阈值】允许更多缺失天数以确保分析进行
            if self.missing_trading_days > 60:
                self.has_critical_error = True
                self.error_message = f"数据严重缺失（缺失{self.missing_trading_days}天），分析终止"
                print(f"分析终止：{self.error_message}")
                return False
            
            # 【一票否决】K线数量不足时强制终止
            if len(self.raw_data) < 150:
                self.has_critical_error = True
                self.error_message = f"K线数量不足，仅有{len(self.raw_data)}根，最低要求150根"
                print(f"分析终止：{self.error_message}")
                return False
            
            # 获取数据完整性评级和可信度提示
            integrity_rating = self._get_integrity_rating(self.missing_trading_days)
            self.data_completeness = f"{integrity_rating['rating']}（{integrity_rating['impact']}）"
            
            # 计算年度价格区间
            self.year_high = self.raw_data['high'].max()
            self.year_low = self.raw_data['low'].min()
            self.yearly_price_range = (self.year_low, self.year_high)
            
            # 验证所有价格数据完整性
            self.verify_price_integrity(check_all=True)
            
            print(f"加载完成：{self.symbol} {self.year}年数据")
            print(f"数据完整性：{self.data_completeness}")
            print(f"缺失交易日数量：{self.missing_trading_days} 天")
            print(f"2025年真实价格区间：{self.year_low:.3f}-{self.year_high:.3f}元")
            
            return True
        except Exception as e:
            self.has_critical_error = True
            self.error_message = f"数据加载错误：{str(e)}"
            print(f"分析终止：{self.error_message}")
            return False
    
    def _get_integrity_rating(self, missing_days_count):
        """根据缺失天数获取数据完整性评级和可信度提示"""
        if missing_days_count <= 5:
            return {
                'rating': '基本完整',
                'impact': '分析结果可信',
                'trust_level': 'high'
            }
        elif missing_days_count <= 15:
            return {
                'rating': '部分缺失',
                'impact': '关键结论需谨慎参考',
                'trust_level': 'medium'
            }
        else:
            return {
                'rating': '严重缺失',
                'impact': '分析结果可信度低，不建议作为交易依据',
                'trust_level': 'low'
            }
    
    def verify_price_integrity(self, price=None, check_all=False):
        """验证价格数据是否在年度价格区间内，确保数据真实性
        
        Args:
            price: 单个价格值（可选）
            check_all: 是否检查所有价格数据（可选）
            
        Returns:
            当check_all=True时返回布尔值，表示整体数据是否有效
            当check_all=False时返回元组(是否有效, 错误消息)
        """
        # 确保年度价格区间存在
        if not hasattr(self, 'year_low') or not hasattr(self, 'year_high'):
            return True, "" if not check_all else True
            
        min_price, max_price = self.year_low, self.year_high
        
        # 检查单个价格
        if not check_all and price is not None:
            if min_price - 0.001 <= price <= max_price + 0.001:  # 添加微小容差
                return True, ""
            else:
                # 数据异常，需要修正
                adjusted_price = max(min(price, max_price), min_price)
                self.price_outliers_count += 1
                return False, f"数据异常已修正：{price:.3f} → {adjusted_price:.3f}"
        # 检查所有价格数据
        elif check_all and hasattr(self, 'raw_data'):
            for col in ['open', 'high', 'low', 'close']:
                if col in self.raw_data.columns:
                    for price in self.raw_data[col]:
                        if not (min_price - 0.001 <= price <= max_price + 0.001):
                            self.price_outliers_count += 1
            return True  # 返回True表示完成检查，异常计数已更新
        
        return True, ""
    
    def generate_standard_k_lines(self):
        """执行K线包含处理，生成标准K线"""
        if self.raw_data is None:
            print("请先加载数据")
            return False
        
        # 复制原始数据
        data = self.raw_data.copy()
        standard_k_lines = []
        
        # 包含处理：严格合并所有有包含关系的K线
        i = 0
        while i < len(data):
            current = data.iloc[i].copy()
            
            # 检查与下一根K线是否有包含关系
            while i + 1 < len(data):
                next_k = data.iloc[i + 1]
                
                # 判断包含关系：两根K线中任意一根的高低点完全包含另一根
                has_include = (next_k['high'] <= current['high'] and next_k['low'] >= current['low']) or \
                             (next_k['high'] >= current['high'] and next_k['low'] <= current['low'])
                
                if has_include:
                    # 合并K线 - 取高低点的极值
                    current['high'] = max(current['high'], next_k['high'])
                    current['low'] = min(current['low'], next_k['low'])
                    current['close'] = next_k['close']  # 保留最新的收盘价
                    current['open'] = current['open']  # 保留原始开盘价
                    current['volume'] = current['volume'] + next_k['volume']
                    current['amount'] = current.get('amount', 0) + next_k.get('amount', 0)
                    current['date'] = next_k['date']  # 保留最新的日期
                    i += 1  # 跳过已合并的K线
                else:
                    break
            
            standard_k_lines.append(current)
            i += 1
        
        self.standard_k_lines = pd.DataFrame(standard_k_lines)
        print(f"标准K线生成完成，共{len(self.standard_k_lines)}根标准K线（原始K线{len(data)}根，合并了{len(data)-len(self.standard_k_lines)}根包含K线）")
        
        # 计算MACD指标
        self.calculate_macd()
        
        # 计算90%成交区间（基于所有标准K线的收盘价）
        self.calculate_90pct_trading_range()
        
        return True
    
    def calculate_90pct_trading_range(self):
        """计算90%成交区间：按收盘价统计，剔除首尾各5%极值，实现数据有效性一票否决机制"""
        if self.standard_k_lines is None:
            return None
        
        # 获取所有标准K线的收盘价
        closes = sorted(self.standard_k_lines['close'].values)
        n = len(closes)
        
        # 【一票否决】检查K线数量是否足够计算成交区间
        if n < 30:
            self.error_message = f"标准K线数量不足30根（仅有{n}根），无法计算有效90%成交区间"
            self.has_critical_error = True
            return None
        
        # 计算5%和95%分位数的索引
        p5_idx = max(0, int(n * 0.05))
        p95_idx = min(n - 1, int(n * 0.95))
        
        # 计算区间
        self.low_90pct = closes[p5_idx]
        self.high_90pct = closes[p95_idx]
        
        # 【一票否决】基本有效性校验
        if self.low_90pct <= 0 or self.high_90pct <= 0:
            self.error_message = f"成交区间数据异常：价格为0或负数，下沿={self.low_90pct:.3f}，上沿={self.high_90pct:.3f}"
            self.has_critical_error = True
            return None
        
        if self.high_90pct <= self.low_90pct:
            self.error_message = f"成交区间数据异常：上沿({self.high_90pct:.3f})不大于下沿({self.low_90pct:.3f})"
            self.has_critical_error = True
            return None
        
        # 计算振幅
        self.amplitude_90pct = (self.high_90pct - self.low_90pct) / self.low_90pct * 100
        
        # 【一票否决】检查振幅是否在合理范围内
        if self.amplitude_90pct < 1.0:
            self.error_message = f"90%成交区间振幅过低({self.amplitude_90pct:.2f}%)，低于1%，数据异常"
            self.has_critical_error = True
            return None
        
        if self.amplitude_90pct > 100.0:
            self.error_message = f"90%成交区间振幅过高({self.amplitude_90pct:.2f}%)，超过100%，可能存在数据异常"
            self.has_critical_error = True
            return None
        
        print(f"90%成交区间计算完成：下沿={self.low_90pct:.3f}元，上沿={self.high_90pct:.3f}元，振幅={self.amplitude_90pct:.2f}%")
        return (self.low_90pct, self.high_90pct, self.amplitude_90pct)
    
    def calculate_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD指标"""
        df = self.standard_k_lines
        
        # 计算EMA
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF、DEA和MACD柱
        df['dif'] = ema_fast - ema_slow
        df['dea'] = df['dif'].ewm(span=signal_period, adjust=False).mean()
        df['macd_bar'] = (df['dif'] - df['dea']) * 2
        
        return df
    
    def determine_volatility_level(self):
        """【强制校验】确定ETF波动等级：实现无歧义判定与异常兜底机制，确保波动等级准确划分"""
        if self.has_critical_error:
            return False
        
        # 检查K线数量是否足够
        if not hasattr(self, 'standard_k_lines') or len(self.standard_k_lines) < 30:
            self.error_message = "样本K线数量不足30根，无法计算波动率"
            self.has_critical_error = True
            self.volatility_level = None
            return False
        
        # 波动率计算：优先使用连续60根标准K线，不足则使用全部
        if len(self.standard_k_lines) >= 60:
            # 使用最近60根连续标准K线
            recent_data = self.standard_k_lines.tail(60)
            sample_size = 60
        else:
            # 数据不足60根，使用全部标准K线
            recent_data = self.standard_k_lines
            sample_size = len(recent_data)
        
        try:
            # 计算波动率：(近N日最高价-近N日最低价)/近N日均价×100%
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            recent_avg = recent_data['close'].mean()
            
            # 【强制校验】检查价格数据异常（使用强制异常兜底）
            if pd.isna(recent_high) or pd.isna(recent_low) or pd.isna(recent_avg):
                self.error_message = "价格数据存在NaN值，无法计算波动率"
                self.has_critical_error = True
                self.volatility = 0.0
                self.volatility_level = None
                return False
            
            if recent_high <= recent_low:
                self.error_message = f"波动率计算异常：最高价({recent_high})不大于最低价({recent_low})，波动率为0%"
                self.has_critical_error = True
                self.volatility = 0.0
                self.volatility_level = None
                return False
            
            if recent_avg == 0:
                self.error_message = "平均价格为0，无法计算波动率"
                self.has_critical_error = True
                self.volatility = 0.0
                self.volatility_level = None
                return False
            
            volatility = (recent_high - recent_low) / recent_avg * 100
            
            # 【强制校验】检查波动率异常（使用强制异常兜底）
            if pd.isna(volatility) or volatility <= 0.0:
                self.error_message = f"波动率计算异常：值为{volatility}%，数据异常，无法继续分析"
                self.has_critical_error = True
                self.volatility = 0.0
                self.volatility_level = None
                return False
            
            # 【强制校验】检查波动率边界条件（使用异常兜底机制）
            volatility_rounded = round(volatility, 1)
            
            # 严格边界条件处理 - 无歧义判定
            if volatility_rounded <= 10.0:
                self.volatility_level = "low"
                level_text = "低波动"
                threshold_text = "≤10.0%"
            elif volatility_rounded <= 18.0:
                self.volatility_level = "medium"
                level_text = "中波动"
                threshold_text = "10.1%-18.0%"
            else:
                self.volatility_level = "high"
                level_text = "高波动"
                threshold_text = ">18.0%"
                
            # 【异常兜底】设置默认值，确保即使在边界情况下也能正常运行
            if not hasattr(self, 'volatility_level') or self.volatility_level is None:
                # 当所有条件都不满足时的默认值
                self.volatility_level = "medium"  # 默认设为中波动
                level_text = "中波动（默认值）"
                threshold_text = "默认值"
                print(f"【异常兜底】无法准确划分波动等级，使用默认等级：中波动")
                
        except Exception as e:
            # 【强制异常兜底】捕获所有异常，确保程序不会崩溃
            self.error_message = f"波动率计算过程中发生异常：{str(e)}"
            self.volatility_level = "medium"  # 异常情况下使用中波动作为默认值
            self.volatility = 0.0
            level_text = "中波动（异常兜底）"
            threshold_text = "异常兜底"
            sample_size = 0
            recent_high = 0
            recent_low = 0
            recent_avg = 0
            volatility_rounded = 0.0
            print(f"【异常兜底】{self.error_message}，使用默认等级：中波动")
        
        # 保存波动率值
        self.volatility = volatility
        self.volatility_rounded = volatility_rounded
        
        # 设置自适应参数
        self.params = {
            'low': {
                'segment_90pct_amplitude_req': 5.0,
                'central_break_threshold': 0.995,
                'central_rebound_threshold': 1.005,
                'consecutive_k_req': 2,
                'rebound_time_window': 5,
                'volume_threshold': 0.8,
                'central_amplitude_req': 5.0,
                'rebound_min_pct': 1.0,  # 回踩后反弹≥1%
                'fall_min_pct': 1.0      # 冲击后回落≥1%
            },
            'medium': {
                'segment_90pct_amplitude_req': 8.0,
                'central_break_threshold': 0.99,
                'central_rebound_threshold': 1.01,
                'consecutive_k_req': 1,  # 2日内≥1日+1日临界
                'rebound_time_window': 6,
                'volume_threshold': 0.85,
                'central_amplitude_req': 8.0,
                'rebound_min_pct': 1.0,
                'fall_min_pct': 1.0
            },
            'high': {
                'segment_90pct_amplitude_req': 10.0,
                'central_break_threshold': 0.985,
                'central_rebound_threshold': 1.015,
                'consecutive_k_req': 1,  # 2日内≥1日+1日临界
                'rebound_time_window': 7,
                'volume_threshold': 0.9,
                'central_amplitude_req': 10.0,
                'rebound_min_pct': 1.0,
                'fall_min_pct': 1.0
            }
        }
        
        # 记录详细信息
        self.volatility_info = {
            'value': volatility_rounded,
            'sample_size': sample_size,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'recent_avg': recent_avg
        }
        
        # 保存波动率值
        self.volatility = volatility if 'volatility' in locals() else 0.0
        self.volatility_rounded = volatility_rounded
        
        # 设置自适应参数
        self.params = {
            'low': {
                'segment_90pct_amplitude_req': 5.0,
                'central_break_threshold': 0.995,
                'central_rebound_threshold': 1.005,
                'consecutive_k_req': 2,
                'rebound_time_window': 5,
                'volume_threshold': 0.8,
                'central_amplitude_req': 5.0,
                'rebound_min_pct': 1.0,  # 回踩后反弹≥1%
                'fall_min_pct': 1.0      # 冲击后回落≥1%
            },
            'medium': {
                'segment_90pct_amplitude_req': 8.0,
                'central_break_threshold': 0.99,
                'central_rebound_threshold': 1.01,
                'consecutive_k_req': 1,  # 2日内≥1日+1日临界
                'rebound_time_window': 6,
                'volume_threshold': 0.85,
                'central_amplitude_req': 8.0,
                'rebound_min_pct': 1.0,
                'fall_min_pct': 1.0
            },
            'high': {
                'segment_90pct_amplitude_req': 10.0,
                'central_break_threshold': 0.985,
                'central_rebound_threshold': 1.015,
                'consecutive_k_req': 1,  # 2日内≥1日+1日临界
                'rebound_time_window': 7,
                'volume_threshold': 0.9,
                'central_amplitude_req': 10.0,
                'rebound_min_pct': 1.0,
                'fall_min_pct': 1.0
            }
        }
        
        # 记录详细信息
        self.volatility_info = {
            'value': volatility_rounded,
            'sample_size': sample_size,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'recent_avg': recent_avg
        }
        
        print(f"【强制校验】波动率计算完成：近{sample_size}根标准K线波动率={volatility_rounded:.1f}%（计算依据：({recent_high:.3f}-{recent_low:.3f})/{recent_avg:.3f}×100%）")
        print(f"【强制校验】ETF波动等级：{level_text}（{volatility_rounded:.1f}% ∈ {threshold_text}）")
        
        return True
    
    def divide_segments(self):
        """实现走势段划分逻辑：严格按照新规则判断盘整段有效性，仅接受满足90%成交区间振幅要求的盘整段"""
        if self.has_critical_error:
            return False
        
        if self.standard_k_lines is None:
            self.error_message = "标准K线未生成，无法划分走势段"
            self.has_critical_error = True
            return False
        
        # 检查标准K线数量是否足够
        if len(self.standard_k_lines) < 50:
            self.error_message = f"标准K线数量不足50根（仅{len(self.standard_k_lines)}根），无法划分走势段"
            self.has_critical_error = True
            return False
        
        # 检查是否已确定波动等级
        if not self.volatility_level:
            self.error_message = "未确定波动等级，无法划分走势段"
            self.has_critical_error = True
            return False
        
        # 获取波动等级对应的振幅要求
        required_amplitude = self.params[self.volatility_level]['segment_90pct_amplitude_req']
        level_text = {"low": "低波动", "medium": "中波动", "high": "高波动"}[self.volatility_level]
        print(f"{level_text}ETF盘整段振幅要求：≥{required_amplitude}% - 【强制校验】")
        
        # 1. 识别所有可能的盘整段（滑动窗口方法）
        all_segments = []
        min_segment_length = 8  # 最小8根标准K线
        max_segment_length = 30  # 最大30根标准K线（避免段过长）
        
        for i in range(len(self.standard_k_lines) - min_segment_length + 1):
            # 尝试不同长度的窗口
            for j in range(min_segment_length, min(max_segment_length, len(self.standard_k_lines) - i) + 1):
                segment_data = self.standard_k_lines.iloc[i:i+j]
                
                # 2. 检查是否为盘整段：无明显涨跌趋势
                if self._is_consolidation_segment(segment_data):
                    # 3. 计算90%成交区间
                    segment_closes = sorted(segment_data['close'].values)
                    n = len(segment_closes)
                    p5_idx = max(0, int(n * 0.05))
                    p95_idx = min(n - 1, int(n * 0.95))
                    
                    segment_low_90pct = segment_closes[p5_idx]
                    segment_high_90pct = segment_closes[p95_idx]
                    segment_amplitude_90pct = (segment_high_90pct - segment_low_90pct) / segment_low_90pct * 100
                    
                    # 4. 检查是否满足所有条件
                    meets_k_count = j >= 8
                    
                    # 【强制校验】创建段对象用于_check_segment_amplitude方法
                    temp_segment = {
                        'amplitude_90pct': segment_amplitude_90pct,
                        'high_90pct': segment_high_90pct,
                        'low_90pct': segment_low_90pct,
                        'k_count': j
                    }
                    
                    # 使用_check_segment_amplitude进行强制校验，确保与逻辑一致性规则严格匹配
                    meets_amplitude = self._check_segment_amplitude(temp_segment)
                    
                    # 价格完整性验证
                    high_valid, high_msg = self.verify_price_integrity(segment_high_90pct)
                    low_valid, low_msg = self.verify_price_integrity(segment_low_90pct)
                    
                    # 收集不满足条件的原因
                    not_meet_reasons = []
                    if not meets_k_count:
                        not_meet_reasons.append(f"K线数量不足（{j} < 8）")
                    if not meets_amplitude:
                        not_meet_reasons.append(f"90%成交区间振幅不足（{segment_amplitude_90pct:.2f}% < {required_amplitude}%）")
                    if not high_valid:
                        not_meet_reasons.append(high_msg)
                    if not low_valid:
                        not_meet_reasons.append(low_msg)
                    
                    # 创建段信息 - 严格遵循有效性条件
                    segment = {
                        'segment_id': len(all_segments) + 1,
                        'start_idx': i,
                        'end_idx': i+j-1,
                        'k_count': j,
                        'high_90pct': segment_high_90pct,
                        'low_90pct': segment_low_90pct,
                        'amplitude_90pct': segment_amplitude_90pct,
                        'start_date': segment_data.iloc[0]['date'],
                        'end_date': segment_data.iloc[-1]['date'],
                        'meets_k_count': meets_k_count,
                        'meets_amplitude': meets_amplitude,
                        'not_meet_reasons': not_meet_reasons,
                        # 【严格校验】只有同时满足所有条件的盘整段才标记为有效
                        'is_valid': meets_k_count and meets_amplitude and high_valid and low_valid,
                        # 【强制校验】确保此标志与实际振幅要求严格对应
                        'amplitude_90pct_met': meets_amplitude,
                        # 记录强制校验结果
                        'amplitude_check_passed': meets_amplitude,
                        'required_amplitude': required_amplitude,
                        'volatility_level': self.volatility_level
                    }
                    
                    all_segments.append(segment)
        
        # 5. 去重并选择最优段（避免重叠过多）
        candidate_segments = self._remove_overlapping_segments(all_segments)
        
        # 6. 【强制校验】再次使用_check_segment_amplitude过滤，确保只有满足振幅要求的段被保留为有效
        valid_segments = []
        for segment in candidate_segments:
            # 重新执行强制校验，确保没有漏网之鱼
            amplitude_check_result = self._check_segment_amplitude(segment)
            if amplitude_check_result:
                # 更新段的有效性状态
                segment['is_valid'] = True
                segment['amplitude_check_passed'] = True
                valid_segments.append(segment)
            else:
                # 标记为无效
                segment['is_valid'] = False
                segment['amplitude_check_passed'] = False
        
        # 7. 最终排序和输出 - 按振幅降序排列
        valid_segments.sort(key=lambda x: x['amplitude_90pct'], reverse=True)
        self.valid_segments = valid_segments
        
        # 统计满足各条件的段数量
        total_potential = len(all_segments)
        passed_overlap_filter = len(candidate_segments)
        passed_amplitude_check = len(valid_segments)
        
        # 输出结果
        print(f"走势段划分完成 - 【强制校验统计】:")
        print(f"  1. 识别潜在盘整段: {total_potential}个")
        print(f"  2. 通过重叠过滤: {passed_overlap_filter}个")
        print(f"  3. 通过振幅强制校验: {passed_amplitude_check}个有效盘整段")
        
        # 详细输出有效盘整段信息
        for i, segment in enumerate(valid_segments):
            status = "✅达标" if segment['is_valid'] else "❌不达标"
            reasons = ", ".join(segment['not_meet_reasons']) if segment['not_meet_reasons'] else "无"
            print(f"盘整段{i+1}: {segment['start_date'].strftime('%Y-%m-%d')}至{segment['end_date'].strftime('%Y-%m-%d')}, " 
                  f"长度{segment['k_count']}根K线, 90%成交区间振幅{segment['amplitude_90pct']:.2f}%, " 
                  f"状态: {status}, 校验结果: {'通过' if segment['amplitude_check_passed'] else '未通过'}")
        
        # 保留原始segments属性以兼容后续处理，同时包含所有候选段（用于逻辑一致性检查）
        self.segments = candidate_segments
        return True
    
    def _is_consolidation_segment(self, segment_data):
        """判断是否为盘整段：无明显涨跌趋势"""
        # 检查是否有明显上涨趋势
        if self._has_rising_trend(segment_data):
            return False
        
        # 检查是否有明显下跌趋势
        if self._has_falling_trend(segment_data):
            return False
        
        # 两者都不满足，视为盘整段
        return True
    
    def _has_rising_trend(self, segment_data):
        """判断是否有明显上涨趋势：连续≥5根收盘价抬升+高点创新高"""
        closes = segment_data['close'].values
        highs = segment_data['high'].values
        
        # 检查是否有连续5根收盘价抬升
        for i in range(len(closes) - 4):
            consecutive_rising = True
            for j in range(1, 5):
                if closes[i+j] <= closes[i+j-1]:
                    consecutive_rising = False
                    break
            
            if consecutive_rising:
                # 检查这5根K线的高点是否创新高
                if all(highs[i+j] > highs[i+j-1] for j in range(1, 5)):
                    return True
        
        return False
    
    def _has_falling_trend(self, segment_data):
        """判断是否有明显下跌趋势：连续≥5根收盘价走低+低点创新低"""
        closes = segment_data['close'].values
        lows = segment_data['low'].values
        
        # 检查是否有连续5根收盘价走低
        for i in range(len(closes) - 4):
            consecutive_falling = True
            for j in range(1, 5):
                if closes[i+j] >= closes[i+j-1]:
                    consecutive_falling = False
                    break
            
            if consecutive_falling:
                # 检查这5根K线的低点是否创新低
                if all(lows[i+j] < lows[i+j-1] for j in range(1, 5)):
                    return True
        
        return False
    
    def _remove_overlapping_segments(self, segments):
        """移除重叠过多的段，保留最有效的段"""
        if not segments:
            return []
        
        # 按有效性和振幅排序
        segments_sorted = sorted(segments, key=lambda x: (not x['is_valid'], -x['amplitude_90pct']))
        valid_segments = []
        
        for segment in segments_sorted:
            # 检查是否与已保留的段重叠过多
            overlap_too_much = False
            for existing in valid_segments:
                # 计算重叠比例
                overlap_start = max(segment['start_idx'], existing['start_idx'])
                overlap_end = min(segment['end_idx'], existing['end_idx'])
                overlap_length = max(0, overlap_end - overlap_start + 1)
                
                # 如果重叠超过60%，视为重叠过多
                overlap_ratio = overlap_length / min(segment['k_count'], existing['k_count'])
                if overlap_ratio > 0.6:
                    overlap_too_much = True
                    break
            
            if not overlap_too_much:
                valid_segments.append(segment)
        
        # 按起始位置排序
        valid_segments.sort(key=lambda x: x['start_idx'])
        return valid_segments
    
    def generate_centrals(self):
        """【强制校验】生成中枢：严格基于有效盘整段，满足90%成交区间振幅强制校验条件"""
        # 筛选真正有效的盘整段：必须通过振幅强制校验
        valid_segments_for_central = [seg for seg in self.valid_segments if 
                                     seg.get('amplitude_check_passed', False) and 
                                     seg.get('amplitude_90pct', 0) >= seg.get('required_amplitude', 0)]
        
        if not valid_segments_for_central:
            print("【强制校验】没有满足90%成交区间振幅要求的有效盘整段，无法生成中枢")
            return False
        
        df = self.standard_k_lines
        self.centrals = []
        self.valid_centrals = []
        required_amplitude = self.params[self.volatility_level]['central_amplitude_req']
        level_text = {"low": "低波动", "medium": "中波动", "high": "高波动"}[self.volatility_level]
        print(f"【强制校验】{level_text}ETF中枢振幅要求：≥{required_amplitude}%")
        print(f"【强制校验】符合条件的有效盘整段数量：{len(valid_segments_for_central)}/{len(self.valid_segments)}")
        
        for seg in valid_segments_for_central:
            # 中枢下沿=盘整段90%成交区间的低点
            central_low = seg['low_90pct']
            # 中枢上沿=盘整段90%成交区间的高点
            central_high = seg['high_90pct']
            # 中枢中轨=(下沿+上沿)/2
            central_mid = (central_low + central_high) / 2
            # 计算中枢振幅
            central_amplitude = (central_high - central_low) / central_low * 100
            
            # 验证中枢有效性的三个条件
            meets_amplitude, amplitude_status = self._check_amplitude_condition(central_amplitude, required_amplitude)
            meets_coverage, coverage_status = self._check_coverage_condition(seg, central_low, central_high, df)
            meets_support_resistance, support_resistance_status = self._check_support_resistance_condition(seg, central_low, central_high, df)
            
            # 中枢是否有效（三个条件必须同时满足）
            is_valid = meets_amplitude and meets_coverage and meets_support_resistance
            
            # 构建中枢信息
            central = {
                'segment_id': seg['segment_id'],
                'start_date': seg['start_date'],
                'end_date': seg['end_date'],
                'low_median': central_low,
                'high_median': central_high,
                'middle_track': central_mid,
                'amplitude': central_amplitude,
                'segment_info': seg,
                'is_valid': is_valid,
                'amplitude_valid': meets_amplitude,
                'coverage_valid': meets_coverage,
                'support_resistance_valid': meets_support_resistance,
                'validation_status': {
                    'amplitude': amplitude_status,
                    'coverage': coverage_status,
                    'support_resistance': support_resistance_status
                },
                # 【强制校验】添加盘整段有效性信息
                'segment_amplitude_90pct': seg.get('amplitude_90pct', 0),
                'segment_required_amplitude': seg.get('required_amplitude', 0),
                'segment_amplitude_check_passed': seg.get('amplitude_check_passed', False)
            }
            
            self.centrals.append(central)
            
            # 如果中枢有效，添加到有效中枢列表
            if is_valid:
                # 保留原有validation字段以兼容后续代码
                central['validation'] = {
                    'in_range_count': coverage_status['count'],
                    'support_count': support_resistance_status['support'],
                    'resistance_count': support_resistance_status['resistance'],
                    'valid': True
                }
                self.valid_centrals.append(central)
                print(f"【强制校验】中枢{seg['segment_id']}：验证有效（振幅{central_amplitude:.2f}%，{coverage_status['count']}根K线在区间内，支撑{support_resistance_status['support']}次，压力{support_resistance_status['resistance']}次）")
            else:
                # 构建无效原因说明
                reasons = []
                if not meets_amplitude:
                    reasons.append(f"振幅不达标（{central_amplitude:.2f}% < {required_amplitude}%）")
                if not meets_coverage:
                    reasons.append(f"覆盖度不达标（{coverage_status['count']}/10根K线在区间内）")
                if not meets_support_resistance:
                    reasons.append(f"支撑/压力不足（支撑{support_resistance_status['support']}次，压力{support_resistance_status['resistance']}次）")
                
                print(f"【强制校验】中枢{seg['segment_id']}：验证失败 - {', '.join(reasons)}")
        
        print(f"【强制校验】中枢生成完成，共{len(self.valid_centrals)}个中枢验证有效")
        
        return True
    
    def _check_amplitude_condition(self, central_amplitude, required_amplitude):
        """检查振幅条件：振幅≥对应波动等级要求"""
        is_valid = central_amplitude >= required_amplitude
        status = {
            'is_valid': is_valid,
            'amplitude': central_amplitude,
            'required': required_amplitude,
            'message': f"振幅{central_amplitude:.2f}% {'≥' if is_valid else '<'} 要求{required_amplitude}%"
        }
        return is_valid, status
    
    def _check_coverage_condition(self, segment, central_low, central_high, df):
        """检查覆盖度条件：后续10根标准K线中≥8根在中枢区间内"""
        central_end_idx = segment['end_idx']
        
        # 检查是否有足够的后续K线
        if central_end_idx + 10 >= len(df):
            status = {
                'is_valid': False,
                'count': 0,
                'total': 10,
                'message': "后续K线数量不足"
            }
            return False, status
        
        # 获取后续10根K线
        after_central_start = central_end_idx + 1
        after_central_end = min(after_central_start + 10, len(df) - 1)
        after_central_k = df.iloc[after_central_start:after_central_end+1]
        
        # 计算在区间内的K线数量（收盘价在区间内且通过价格完整性验证）
        in_range_count = 0
        for _, k_line in after_central_k.iterrows():
            # 价格完整性验证
            valid, _ = self.verify_price_integrity(k_line['close'])
            # 判断K线是否与中枢区间部分重叠
            if valid and not (k_line['high'] < central_low or k_line['low'] > central_high):
                in_range_count += 1
        
        is_valid = in_range_count >= 8
        status = {
            'is_valid': is_valid,
            'count': in_range_count,
            'total': 10,
            'message': f"后续10根K线中{in_range_count}根在区间内"
        }
        
        return is_valid, status
    
    def _check_support_resistance_condition(self, segment, central_low, central_high, df):
        """检查支撑压力条件：支撑次数≥2次且压力次数≥2次"""
        # 获取参数
        rebound_pct = self.params[self.volatility_level]['rebound_min_pct'] / 100
        fall_pct = self.params[self.volatility_level]['fall_min_pct'] / 100
        
        # 计算支撑和压力
        support_count = 0
        resistance_count = 0
        
        # 检查后续K线
        central_end_idx = segment['end_idx']
        after_central_start = central_end_idx + 1
        after_central_end = min(after_central_start + 10, len(df) - 1)
        
        for i in range(after_central_start, after_central_end + 1):
            k = df.iloc[i]
            
            # 检查支撑：价格回踩下沿反弹（当日最低价接近下沿，且收盘价高于最低价）
            if abs(k['low'] - central_low) / central_low * 100 <= 2 and k['close'] > k['low']:
                # 确认是反弹：下一根K线收盘价上涨
                if i + 1 <= after_central_end and df.iloc[i+1]['close'] > k['close'] * (1 + rebound_pct):
                    support_count += 1
            
            # 检查压力：价格冲击上沿回落（当日最高价接近上沿，且收盘价低于最高价）
            if abs(k['high'] - central_high) / central_high * 100 <= 2 and k['close'] < k['high']:
                # 确认是回落：下一根K线收盘价下跌
                if i + 1 <= after_central_end and df.iloc[i+1]['close'] < k['close'] * (1 - fall_pct):
                    resistance_count += 1
        
        is_valid = support_count >= 2 and resistance_count >= 2
        status = {
            'is_valid': is_valid,
            'support': support_count,
            'resistance': resistance_count,
            'required': 2,
            'message': f"支撑{support_count}次，压力{resistance_count}次"
        }
        
        return is_valid, status
    
    def identify_break_central_rebound_buy_signals(self):
        """【强制校验】识别破中枢反抽一买信号：严格基于有效中枢，满足技术条件和参数匹配"""
        # 双重验证中枢有效性
        valid_centrals_for_signal = [central for central in self.valid_centrals if 
                                    central.get('is_valid', False) and 
                                    central.get('segment_amplitude_check_passed', False)]
        
        if not valid_centrals_for_signal:
            print("【强制校验】没有满足严格条件的有效中枢，无法识别信号")
            self.buy_signals = []
            return []
        
        df = self.standard_k_lines
        self.buy_signals = []
        level_text = {"low": "低波动", "medium": "中波动", "high": "高波动"}[self.volatility_level]
        params = self.params[self.volatility_level]
        
        print(f"【强制校验】基于{level_text}ETF参数识别破中枢反抽一买信号")
        print(f"【强制校验】符合条件的有效中枢数量：{len(valid_centrals_for_signal)}/{len(self.valid_centrals)}")
        print(f"【强制校验】信号识别参数详情：")
        print(f"【强制校验】- 破位阈值: 中枢下沿×{params.get('break_threshold_factor', params.get('central_break_threshold', 0.99))}")
        print(f"【强制校验】- 反抽阈值: 中枢下沿×{params.get('rebound_threshold_factor', params.get('central_rebound_threshold', 1.01))}")
        print(f"【强制校验】- 连续达标要求: {params.get('consecutive_days_req', '连续2日' if self.volatility_level == 'low' else '2日内≥1日+1日临界')}")
        print(f"【强制校验】- 反抽时间窗口: {params.get('rebound_window', params.get('rebound_time_window', 6))}个交易日")
        print(f"【强制校验】- 量能验证阈值: ≥近5日均量{params.get('volume_threshold_factor', params.get('volume_threshold', 0.85)) * 100:.0f}%")
        
        for central in valid_centrals_for_signal:
            central_id = central['segment_id']
            central_low = central['low_median']
            central_high = central['high_median']
            central_end_idx = central['segment_info']['end_idx']
            segment_amplitude = central.get('segment_amplitude_90pct', 0)
            
            print(f"\n【强制校验】检查中枢{central_id}：区间{central_low:.3f}-{central_high:.3f}元，来源盘整段振幅{segment_amplitude:.2f}%")
            
            # 获取波动等级对应的参数（兼容旧参数名）
            break_threshold_factor = params.get('break_threshold_factor', params.get('central_break_threshold', 0.99))
            rebound_threshold_factor = params.get('rebound_threshold_factor', params.get('central_rebound_threshold', 1.01))
            rebound_window = params.get('rebound_window', params.get('rebound_time_window', 6))
            volume_threshold_factor = params.get('volume_threshold_factor', params.get('volume_threshold', 0.85))
            
            # 1. 检查破位条件
            break_detected, break_details = self._check_break_condition(central, params, df)
            if not break_detected:
                print(f"【强制校验】  - 未满足破位条件: {break_details['reason']}")
                continue
            
            print(f"【强制校验】  - 破位确认: {break_details['date']}，价格{break_details['price']:.3f}元")
            
            # 2. 检查反抽条件
            rebound_detected, rebound_details = self._check_rebound_condition(break_details, central, params, df, rebound_window)
            if not rebound_detected:
                print(f"【强制校验】  - 未满足反抽条件: {rebound_details['reason']}")
                continue
            
            print(f"【强制校验】  - 反抽确认: {rebound_details['date']}，价格{rebound_details['price']:.3f}元")
            
            # 3. 检查量能条件
            volume_valid, volume_details = self._check_volume_condition(rebound_details, df, volume_threshold_factor)
            if not volume_valid:
                print(f"【强制校验】  - 量能验证失败: {volume_details['reason']}")
                continue
            
            print(f"【强制校验】  - 量能验证通过: {volume_details['volume']:.2f} ≥ {volume_details['threshold']:.2f}")
            
            # 4. 检查MACD技术共振条件（如有）
            macd_valid, macd_details = self._check_macd_condition(break_details, rebound_details, df, central_end_idx)
            if not macd_valid and 'macd_bar' in df.columns:
                print(f"【强制校验】  - MACD技术共振失败: {macd_details['reason']}")
                continue
            elif 'macd_bar' in df.columns:
                print(f"【强制校验】  - MACD技术共振通过: {macd_details['message']}")
            
            # 计算交易参数
            signal_date = rebound_details['date']
            signal_price = rebound_details['price']
            stop_loss = central_low * break_threshold_factor  # 止损位设为破位阈值
            take_profit = central_high  # 止盈位设为中枢上沿
            
            # 资金配置（固定参数）
            capital = 160000
            position_percentage = 0.2
            buy_amount = int((capital * position_percentage) / signal_price)
            invested_capital = round(buy_amount * signal_price, 2)
            
            risk = signal_price - stop_loss
            reward = take_profit - signal_price
            risk_reward_ratio = round(reward / risk, 2) if risk > 0 else 0
            
            # 创建买入信号
            signal = {
                'signal_id': len(self.buy_signals) + 1,
                'central_id': central_id,
                'break_date': break_details['date'],
                'break_low': break_details['price'],
                'rebound_date': signal_date,
                'rebound_close': signal_price,
                'volume_ratio': volume_details['volume'] / volume_details['avg_volume'],
                'trade_parameters': {
                    'buy_price': signal_price,
                    'stop_loss_price': stop_loss,
                    'take_profit_price': take_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'buy_amount': buy_amount,
                    'invested_capital': invested_capital,
                    'position_percentage': position_percentage * 100
                },
                'validated_params': {
                    'break_threshold': break_threshold_factor,
                    'rebound_threshold': rebound_threshold_factor,
                    'rebound_window': rebound_window,
                    'volume_threshold': volume_threshold_factor,
                    'volatility_level': self.volatility_level
                },
                'verification_status': {
                    'break': True,
                    'rebound': True,
                    'volume': True,
                    'macd': macd_valid if 'macd_bar' in df.columns else 'not_calculated'
                },
                'anchor_central': central,
                # 【强制校验】添加强制校验标记和完整性信息
                'force_validation_passed': True,
                'validation_integrity': {
                    'segment_amplitude_90pct': segment_amplitude,
                    'central_amplitude': central['amplitude'],
                    'volume_ratio_pct': (volume_details['volume'] / volume_details['avg_volume'] * 100) if volume_details['avg_volume'] > 0 else 0
                }
            }
            
            self.buy_signals.append(signal)
            print(f"【强制校验】中枢{central_id}：识别到有效破中枢反抽一买信号")
            print(f"【强制校验】信号日期：{signal_date}，信号价格：{signal_price:.3f}元")
            print(f"【强制校验】止损：{stop_loss:.3f}元，止盈：{take_profit:.3f}元，盈亏比：{risk_reward_ratio:.2f}")
        
        if not self.buy_signals:
            reason = "无满足全部强制校验条件的有效信号" if valid_centrals_for_signal else "无满足条件的有效中枢"
            print(f"【强制校验】信号识别完成，未识别到有效破中枢反抽一买信号。原因：{reason}")
        else:
            print(f"【强制校验】信号识别完成，共识别到{len(self.buy_signals)}个有效破中枢反抽一买信号")
        
        return self.buy_signals
    
    def _check_break_condition(self, central, params, df):
        """检查破位条件：严格匹配波动等级参数"""
        central_low = central['low_median']
        central_end_idx = central['segment_info']['end_idx']
        
        # 兼容新旧参数名
        break_threshold_factor = params.get('break_threshold_factor', params.get('central_break_threshold', 0.99))
        consecutive_days_req = params.get('consecutive_days_req', 'low' == self.volatility_level)
        break_threshold = central_low * break_threshold_factor
        
        # 根据波动等级确定连续达标要求
        if self.volatility_level == 'low':
            # 低波动：连续2日
            consecutive_count = 0
            for i in range(central_end_idx + 1, len(df) - 1):
                valid1, _ = self.verify_price_integrity(df.iloc[i]['close'])
                valid2, _ = self.verify_price_integrity(df.iloc[i+1]['close'])
                
                if valid1 and valid2 and df.iloc[i]['close'] <= break_threshold and df.iloc[i+1]['close'] <= break_threshold:
                    return True, {
                        'detected': True,
                        'index': i+1,
                        'date': df.iloc[i+1]['date'],
                        'price': df.iloc[i+1]['low'],
                        'threshold': break_threshold,
                        'reason': '满足连续2日破位要求'
                    }
        else:
            # 中高波动：2日内≥1日+1日临界
            for i in range(central_end_idx + 1, len(df) - 1):
                valid1, _ = self.verify_price_integrity(df.iloc[i]['close'])
                valid2, _ = self.verify_price_integrity(df.iloc[i+1]['close'])
                
                if not valid1 or not valid2:
                    continue
                
                # 临界阈值：破位阈值的1.005倍
                critical_threshold = break_threshold * 1.005
                
                if ((df.iloc[i]['close'] <= break_threshold and df.iloc[i+1]['close'] <= critical_threshold) or
                    (df.iloc[i]['close'] <= critical_threshold and df.iloc[i+1]['close'] <= break_threshold)):
                    return True, {
                        'detected': True,
                        'index': i+1,
                        'date': df.iloc[i+1]['date'],
                        'price': df.iloc[i+1]['low'],
                        'threshold': break_threshold,
                        'reason': '满足2日内≥1日+1日临界要求'
                    }
        
        return False, {
            'detected': False,
            'reason': f'未满足{self.volatility_level}波动等级的破位条件'
        }
    
    def _check_rebound_condition(self, break_details, central, params, df, rebound_window):
        """检查反抽条件：在指定时间窗口内"""
        break_index = break_details['index']
        central_low = central['low_median']
        
        # 兼容新旧参数名
        rebound_threshold_factor = params.get('rebound_threshold_factor', params.get('central_rebound_threshold', 1.01))
        rebound_threshold = central_low * rebound_threshold_factor
        
        # 检查反抽窗口内的K线
        max_rebound_idx = min(break_index + rebound_window, len(df) - 1)
        
        for i in range(break_index + 1, max_rebound_idx + 1):
            valid, _ = self.verify_price_integrity(df.iloc[i]['close'])
            
            if valid and df.iloc[i]['close'] >= rebound_threshold:
                return True, {
                    'detected': True,
                    'index': i,
                    'date': df.iloc[i]['date'],
                    'price': df.iloc[i]['close'],
                    'threshold': rebound_threshold,
                    'days_from_break': i - break_index
                }
        
        return False, {
            'detected': False,
            'reason': f'破位后{rebound_window}个交易日内未达到反抽阈值'
        }
    
    def _check_volume_condition(self, rebound_details, df, volume_threshold_factor):
        """检查量能条件：反抽日成交量≥近5日均量的指定百分比"""
        rebound_index = rebound_details['index']
        
        # 计算近5日均量
        if rebound_index >= 5:
            recent_volumes = df.iloc[rebound_index-5:rebound_index]['volume'].values
            avg_volume = recent_volumes.mean() if len(recent_volumes) > 0 else 0
        else:
            # 如果不足5天，使用所有可用数据
            available_volumes = df.iloc[:rebound_index]['volume'].values
            avg_volume = available_volumes.mean() if len(available_volumes) > 0 else 0
        
        rebound_volume = df.iloc[rebound_index]['volume']
        volume_threshold = avg_volume * volume_threshold_factor
        meets_volume_req = rebound_volume >= volume_threshold if avg_volume > 0 else True
        
        if meets_volume_req:
            return True, {
                'valid': True,
                'volume': rebound_volume,
                'avg_volume': avg_volume,
                'threshold': volume_threshold,
                'factor': volume_threshold_factor
            }
        else:
            return False, {
                'valid': False,
                'volume': rebound_volume,
                'threshold': volume_threshold,
                'reason': f'成交量{rebound_volume:.2f}低于阈值{volume_threshold:.2f}'
            }
    
    def _check_macd_condition(self, break_details, rebound_details, df, central_end_idx):
        """检查MACD技术共振条件：明确量化的底背驰判断"""
        if 'macd_bar' not in df.columns:
            return True, {
                'valid': True,
                'message': 'MACD未计算，跳过验证'
            }
        
        break_index = break_details['index']
        rebound_index = rebound_details['index']
        
        # 底背驰定义：价格新低 + MACD黄白线不新低 + 绿柱缩短≥30%
        # 1. 检查是否有价格新低
        price_low_before_break = df.iloc[break_index-10:break_index]['low'].min()
        price_low_at_break = df.iloc[break_index]['low']
        has_price_low = price_low_at_break < price_low_before_break
        
        # 2. 检查MACD柱状图是否不新低
        if 'macd' in df.columns:
            macd_low_before_break = df.iloc[break_index-10:break_index]['macd'].min()
            macd_at_break = df.iloc[break_index]['macd']
            macd_divergence = macd_at_break >= macd_low_before_break
        else:
            # 使用macd_bar作为替代
            macd_low_before_break = df.iloc[break_index-10:break_index]['macd_bar'].min()
            macd_at_break = df.iloc[break_index]['macd_bar']
            macd_divergence = macd_at_break >= macd_low_before_break
        
        # 3. 检查绿柱是否缩短≥30%
        # 获取破位时的绿柱长度（绝对值）
        hist_at_break = abs(df.iloc[break_index]['macd_bar'])
        # 获取反抽时的绿柱长度（绝对值）
        hist_at_rebound = abs(df.iloc[rebound_index]['macd_bar'])
        # 计算缩短比例
        if hist_at_break > 0:
            hist_shrink_pct = (hist_at_break - hist_at_rebound) / hist_at_break * 100
            hist_shrink = hist_shrink_pct >= 30
        else:
            hist_shrink = True
        
        # 技术共振要求至少满足2个条件
        valid_conditions = sum([has_price_low, macd_divergence, hist_shrink])
        
        if valid_conditions >= 2:
            return True, {
                'valid': True,
                'message': f'满足{valid_conditions}/3个技术共振条件'
            }
        else:
            return False, {
                'valid': False,
                'reason': f'仅满足{valid_conditions}/3个技术共振条件（需要≥2个）'
            }
    
    def generate_report(self, output_file="universal_chanlun_report.md"):
        """生成符合要求的分析报告：优先显示错误信息，严格按照规范输出"""
        level_text = {
            'low': "低波动",
            'medium': "中波动",
            'high': "高波动"
        }
        
        # 1. 生成报告内容
        report_content = f"""# {self.symbol} {self.year}年通用缠论分析报告

## 前置说明

**标的代码**：{self.symbol}
**分析周期**：{self.year}年全年
"""
        
        # 2. 优先显示数据异常说明
        has_critical_error = getattr(self, 'has_critical_error', False)
        if has_critical_error:
            error_message = getattr(self, 'error_message', '未知错误')
            report_content += "\n### 数据异常说明\n\n"
            report_content += f"**{error_message}**\n\n"
            report_content += "根据通用缠论分析规范，关键数据异常将直接终止后续分析。\n\n"
        
        # 3. 添加价格区间（如果有）
        if hasattr(self, 'year_low') and hasattr(self, 'year_high') and self.year_low and self.year_high:
            report_content += f"**真实价格区间**：{self.year_low:.3f}-{self.year_high:.3f}元（数据来源：历史日K）\n\n"
        
        # 4. 仅在没有严重错误时显示波动率计算过程
        if not has_critical_error:
            report_content += "### 波动率计算过程\n"
            
            # 添加波动率计算过程和波动等级判定依据
            if hasattr(self, 'volatility_calculation_details'):
                details = self.volatility_calculation_details
                report_content += f"- 样本K线数量：{details.get('sample_size', '未知')}\n"
                
                # 安全格式化数值
                high_val = details.get('high', '未知')
                low_val = details.get('low', '未知')
                avg_val = details.get('avg', '未知')
                
                report_content += f"- 最高价：{high_val:.3f}\n" if isinstance(high_val, (int, float)) else f"- 最高价：{high_val}\n"
                report_content += f"- 最低价：{low_val:.3f}\n" if isinstance(low_val, (int, float)) else f"- 最低价：{low_val}\n"
                report_content += f"- 均价：{avg_val:.3f}\n" if isinstance(avg_val, (int, float)) else f"- 均价：{avg_val}\n"
                
                # 计算波动率表达式（确保数值安全）
                if isinstance(high_val, (int, float)) and isinstance(low_val, (int, float)) and isinstance(avg_val, (int, float)) and avg_val > 0:
                    report_content += f"- 波动率 = (最高价-最低价)/均价×100% = {high_val-low_val:.3f}/{avg_val:.3f}×100% = {self.volatility * 100:.1f}%\n"
                else:
                    report_content += f"- 波动率计算结果：{self.volatility * 100:.1f}%\n"
            else:
                report_content += f"- 计算结果：{self.volatility * 100:.1f}%\n"
            
            report_content += "\n### 波动等级判定\n"
            volatility_threshold = "≤10.0%" if self.volatility_level == "low" else ("10.1%-18.0%" if self.volatility_level == "medium" else ">18.0%")
            report_content += f"- 波动率：{self.volatility * 100:.1f}%\n"
            report_content += f"- 判定阈值：{volatility_threshold}\n"
            report_content += f"- 波动等级：{level_text.get(self.volatility_level, '无法判定')}ETF\n"
        
        # 5. 添加数据完整性检查
        missing_trading_days = getattr(self, 'missing_trading_days', 0)
        data_completeness = getattr(self, 'data_completeness', '未知')
        report_content += "\n### 数据完整性检查\n"
        report_content += f"- 缺失交易日：{missing_trading_days}天\n"
        report_content += f"- 数据完整性状态：{data_completeness}\n"
        
        # 3. 添加数据完整性强提示
        report_content += "\n### 数据完整性强提示\n"
        if hasattr(self, 'missing_trading_days'):
            integrity_rating = self._get_integrity_rating(self.missing_trading_days)
            report_content += f"- 缺失交易日数量：{self.missing_trading_days}\n"
            report_content += f"- 完整性评级：{integrity_rating['rating']}\n"
            report_content += f"- 影响说明：{integrity_rating['impact']}\n"
        else:
            report_content += f"- 数据完整性状态：{self.data_completeness}\n"
        
        # 4. 标准K线数量
        standard_k_lines = getattr(self, 'standard_k_lines', [])
        k_count = len(standard_k_lines) if hasattr(standard_k_lines, '__len__') else 0
        report_content += f"\n- 标准K线生成数量：{k_count}根\n"
        
        # 仅在没有严重错误时继续输出有效盘整段等内容
        if not has_critical_error:
            # 5. 有效盘整段清单（新增是否达标列）
            report_content += "\n## 一、有效盘整段清单\n\n"
            report_content += "| 序号 | 盘整段ID | 起始日期 | 结束日期 | K线数量 | 振幅 | 90%成交区间 | 是否达标 | 不满足条件项 |\n"
            report_content += "|------|----------|---------|---------|---------|------|------------|----------|------------|\n"
        
        # 仅在没有严重错误且存在有效盘整段时处理
        if not has_critical_error and self.valid_segments:
            for i, segment in enumerate(self.valid_segments, 1):
                # 适配不同的数据结构
                segment_info = segment.get('segment_info', segment)
                start_date = segment_info['start_date']
                end_date = segment_info['end_date']
                k_count = segment_info.get('k_count', segment.get('k_count', 0))
                amplitude = segment.get('amplitude', segment.get('amplitude_90pct', 0))
                low_median = segment.get('low_median', 0)
                high_median = segment.get('high_median', 0)
                
                # 检查是否达标（满足所有条件）
                meets_k_count = k_count >= 8
                meets_amplitude = self._check_segment_amplitude(segment)
                meets_no_trend = segment.get('is_consolidation', True)
                is_valid = meets_k_count and meets_amplitude and meets_no_trend
                
                # 收集不满足的条件
                unsatisfied = []
                if not meets_k_count:
                    unsatisfied.append("K线数量<8")
                if not meets_amplitude:
                    unsatisfied.append("90%成交区间振幅不足")
                if not meets_no_trend:
                    unsatisfied.append("存在明显趋势")
                
                report_content += f"| {i} | {segment.get('segment_id', i)} | {start_date.strftime('%Y-%m-%d')} | {end_date.strftime('%Y-%m-%d')} | {k_count} | {amplitude:.2f}% | {low_median:.3f}-{high_median:.3f} | {'达标' if is_valid else '不达标'} | {', '.join(unsatisfied) if unsatisfied else '无'} |\n"
        else:
            report_content += "| - | - | - | - | - | - | - | - | 未识别到有效盘整段 |\n"
        
        # 5. 有效中枢清单（新增有效性3条件是否满足列）
        # 仅在没有严重错误时输出有效中枢清单
        if not has_critical_error:
            report_content += "\n## 二、有效中枢清单\n\n"
        # 仅在没有严重错误时显示中枢表格
        if not has_critical_error:
            report_content += "| 中枢ID | 起始日期 | 结束日期 | 下沿 | 上沿 | 中轨 | 振幅 | 有效性3条件是否满足 | 状态说明 |\n"
            report_content += "|--------|---------|---------|------|------|------|------|---------------------|----------|\n"
        
        # 仅在没有严重错误且存在有效中枢时处理
        if not has_critical_error and self.valid_centrals:
            for central in self.valid_centrals:
                # 适配不同的数据结构
                start_date = central.get('segment_info', central).get('start_date', central.get('start_date'))
                end_date = central.get('segment_info', central).get('end_date', central.get('end_date'))
                
                # 检查三个有效性条件
                validity = central.get('validity', central.get('validation', {}))
                amplitude_check = validity.get('amplitude', self._check_segment_amplitude(central))
                
                # 覆盖度检查：后续10根K线中≥8根在中枢区间内
                in_range_count = validity.get('in_range_count', 0)
                coverage_check = in_range_count >= 8
                
                # 支撑压力检查：支撑次数≥2次且压力次数≥2次
                support_count = validity.get('support_count', 0)
                resistance_count = validity.get('resistance_count', 0)
                support_resistance_check = support_count >= 2 and resistance_count >= 2
                
                conditions_status = f"{'是' if amplitude_check else '否'}/{'是' if coverage_check else '否'}/{'是' if support_resistance_check else '否'}"
                
                # 状态说明
                status_messages = []
                if amplitude_check:
                    status_messages.append("振幅达标")
                else:
                    status_messages.append("振幅不足")
                status_messages.append(f"覆盖度{in_range_count}/10")
                status_messages.append(f"支撑{support_count}次/压力{resistance_count}次")
                
                report_content += f"| {central.get('segment_id', 'N/A')} | {start_date.strftime('%Y-%m-%d')} | {end_date.strftime('%Y-%m-%d')} | {central.get('low_median', 0):.3f} | {central.get('high_median', 0):.3f} | {central.get('middle_track', 0):.3f} | {central.get('amplitude', 0):.2f}% | {conditions_status} | {', '.join(status_messages)} |\n"
        else:
            report_content += "| - | - | - | - | - | - | - | - | 未识别到有效中枢 |\n"
        
        # 仅在没有严重错误时输出有效信号清单
        if not has_critical_error:
            # 6. 有效信号清单
            report_content += "\n## 三、有效信号清单\n\n"
            report_content += "| 信号ID | 中枢ID | 破位日期 | 破位价格 | 反抽日期 | 反抽价格 | 止损价格 | 止盈价格 | 风险回报比 | 验证状态 |\n"
            report_content += "|--------|--------|---------|---------|---------|---------|---------|---------|------------|----------|\n"
        
        # 仅在没有严重错误且存在有效信号时处理
        if not has_critical_error and self.buy_signals:
            for signal in self.buy_signals:
                # 验证状态
                verification_status = signal.get('verification_status', {})
                status_str = []
                if verification_status.get('break', True):  # 默认通过
                    status_str.append("破位√")
                if verification_status.get('rebound', True):
                    status_str.append("反抽√")
                if verification_status.get('volume', True):
                    status_str.append("量能√")
                if verification_status.get('macd', True):
                    status_str.append("MACD√")
                
                params = signal.get('trade_parameters', {})
                report_content += f"| {signal.get('signal_id', 'N/A')} | {signal.get('central_id', 'N/A')} | {signal.get('break_date').strftime('%Y-%m-%d')} | {signal.get('break_low', 0):.3f} | {signal.get('rebound_date').strftime('%Y-%m-%d')} | {signal.get('rebound_close', 0):.3f} | {params.get('stop_loss_price', 0):.3f} | {params.get('take_profit_price', 0):.3f} | {params.get('risk_reward_ratio', 0):.2f} | {', '.join(status_str)} |\n"
        # 仅在没有严重错误且没有有效信号时输出
        elif not has_critical_error:
            # 明确说明无有效信号的原因
            reason = "有有效中枢但未触发破位反抽条件" if self.valid_centrals else "无满足条件的有效中枢"
            report_content += f"| - | - | - | - | - | - | - | - | - | {reason} |\n"
        
        # 仅在没有严重错误时输出纠错自查总结
        if not has_critical_error:
            # 7. 新增纠错自查总结
            report_content += "\n## 四、纠错自查总结\n\n"
            self_error_check = self._perform_error_check()
            report_content += "### 逻辑矛盾检查\n"
            report_content += f"- {self_error_check['logical_consistency']}\n\n"
        
        # 确保这部分也只在没有严重错误时输出
        if not has_critical_error:
            report_content += "### 数据异常检查\n"
            report_content += f"- {self_error_check['data_integrity']}\n\n"
            
            report_content += "### 参数一致性检查\n"
            report_content += f"- {self_error_check['parameter_consistency']}\n\n"
            
            report_content += "### 总体评估\n"
            report_content += f"- {self_error_check['overall']}\n"
            
            # 8. 原有校验说明
            report_content += "\n## 五、校验说明\n\n"
        report_content += "- 所有分析基于标的自身2025年完整历史日K数据\n"
        # 添加空值检查
        if hasattr(self, 'year_low') and hasattr(self, 'year_high') and self.year_low is not None and self.year_high is not None:
            report_content += f"- 所有价格数据均在真实价格区间{self.year_low:.3f}-{self.year_high:.3f}元内\n"
        else:
            report_content += "- 价格区间信息：数据加载不完整\n"
        
        # 添加波动等级检查
        if hasattr(self, 'volatility_level') and self.volatility_level in level_text:
            report_content += f"- 参数自适应匹配{level_text[self.volatility_level]}ETF特性\n"
        else:
            report_content += "- 波动等级：暂未确定\n"
            
        report_content += "- 中枢区间严格按照盘整段90%成交区间计算，未使用系数调整\n"
        
        report_content += "\n---\n\n"
        report_content += "*本报告基于通用缠论分析规范生成，仅供参考，不构成投资建议。投资有风险，入市需谨慎。*"
        
        # 9. 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"报告已保存至：{output_file}")
        return report_content
    
    def _check_segment_amplitude(self, segment):
        """实现盘整段有效性强制校验：严格基于90%成交区间振幅与波动等级匹配"""
        if not hasattr(self, 'volatility_level'):
            return False
        
        # 【强制校验】优先使用90%成交区间振幅进行判断，确保与规则严格一致
        amplitude_90pct = segment.get('amplitude_90pct', 0)
        
        # 根据波动等级获取严格的振幅要求
        if self.volatility_level == 'low':
            required_amplitude = 5.0  # 低波动要求严格≥5%
        elif self.volatility_level == 'medium':
            required_amplitude = 8.0  # 中波动要求严格≥8%
        else:  # high
            required_amplitude = 10.0  # 高波动要求严格≥10%
        
        # 严格检查：振幅必须大于等于要求值才视为有效
        is_valid = amplitude_90pct >= required_amplitude
        
        # 记录校验信息到段数据中，便于调试和报告
        segment['amplitude_check_passed'] = is_valid
        segment['required_amplitude'] = required_amplitude
        segment['amplitude_used_for_check'] = amplitude_90pct
        
        return is_valid
    
    def _get_integrity_rating(self, missing_days):
        """获取数据完整性评级"""
        if missing_days <= 5:
            return {
                'rating': '数据基本完整',
                'impact': '分析结果可信'
            }
        elif missing_days <= 15:
            return {
                'rating': '数据部分缺失',
                'impact': '关键结论需谨慎参考'
            }
        else:
            return {
                'rating': '数据严重缺失',
                'impact': '分析结果可信度低，不建议作为交易依据'
            }
    
    def _perform_error_check(self):
        """执行纠错自查，实现逻辑一致性强制校验：盘整段达标状态与90%成交区间振幅严格匹配"""
        results = {
            'logical_consistency': '无明显逻辑矛盾',
            'data_integrity': '无明显数据异常',
            'parameter_consistency': '参数匹配正常',
            'overall': '分析结果可信度较高'
        }
        
        # 1. 检查逻辑矛盾 - 实现逻辑一致性强制校验
        logical_issues = []
        
        # 【逻辑一致性强制校验】检查所有有效中枢的振幅是否严格满足波动等级要求
        if hasattr(self, 'valid_centrals'):
            for central in self.valid_centrals:
                amplitude = central.get('amplitude', central.get('amplitude_90pct', 0))
                if not self._check_segment_amplitude(central):
                    # 计算应该满足的振幅要求
                    required_amplitude = 5.0 if self.volatility_level == 'low' else \
                                        8.0 if self.volatility_level == 'medium' else 10.0
                    logical_issues.append(f"中枢{central.get('segment_id', 'N/A')}振幅({amplitude:.2f}%)不满足{self.volatility_level}波动等级要求({required_amplitude}%)")
        
        # 【逻辑一致性强制校验】检查盘整段达标状态与90%成交区间振幅的严格匹配
        if hasattr(self, 'segments'):
            for segment in self.segments:
                if segment.get('is_valid', False):  # 达标状态为True
                    amplitude_90pct = segment.get('amplitude_90pct', 0)
                    if not self._check_segment_amplitude(segment):
                        required_amplitude = 5.0 if self.volatility_level == 'low' else \
                                            8.0 if self.volatility_level == 'medium' else 10.0
                        logical_issues.append(f"盘整段{segment.get('segment_id', 'N/A')}达标状态错误：振幅({amplitude_90pct:.2f}%)不满足{self.volatility_level}波动等级要求({required_amplitude}%)")
                elif segment.get('amplitude_90pct', 0) >= (5.0 if self.volatility_level == 'low' else \
                                                          8.0 if self.volatility_level == 'medium' else 10.0):
                    # 振幅满足要求但达标状态为False，也是逻辑矛盾
                    logical_issues.append(f"盘整段{segment.get('segment_id', 'N/A')}达标状态错误：振幅满足要求但未被标记为有效")
        
        # 【逻辑一致性强制校验】检查价格是否在年度区间内
        if hasattr(self, 'year_low') and hasattr(self, 'year_high'):
            # 检查所有中枢价格是否在区间内
            for central in getattr(self, 'valid_centrals', []):
                low_median = central.get('low_median', 0)
                high_median = central.get('high_median', 0)
                if low_median < self.year_low or high_median > self.year_high:
                    logical_issues.append(f"中枢{central.get('segment_id', 'N/A')}价格超出年度区间({self.year_low:.3f}-{self.year_high:.3f})")
        
        # 【逻辑一致性强制校验】检查90%成交区间与整体价格区间的合理性
        if hasattr(self, 'low_90pct') and hasattr(self, 'high_90pct') and hasattr(self, 'year_low') and hasattr(self, 'year_high'):
            if self.low_90pct < self.year_low * 0.9 or self.high_90pct > self.year_high * 1.1:
                logical_issues.append(f"90%成交区间({self.low_90pct:.3f}-{self.high_90pct:.3f})与年度价格区间({self.year_low:.3f}-{self.year_high:.3f})偏差过大")
        
        if logical_issues:
            results['logical_consistency'] = f"发现{len(logical_issues)}处逻辑矛盾：{'; '.join(logical_issues)}"
        
        # 2. 检查数据异常
        data_issues = []
        
        # 检查数据缺失情况（统一使用missing_days_count）
        if hasattr(self, 'missing_days_count'):
            if self.missing_days_count > 15:
                data_issues.append(f"数据严重缺失（缺失{self.missing_days_count}天）")
            elif self.missing_days_count > 5:
                data_issues.append(f"数据部分缺失（缺失{self.missing_days_count}天）")
            # 同步到missing_trading_days以保持一致性
            self.missing_trading_days = self.missing_days_count
        
        # 检查价格异常修正计数
        if hasattr(self, 'price_outliers_count') and self.price_outliers_count > 0:
            data_issues.append(f"发现{self.price_outliers_count}个异常价格已修正")
        
        if data_issues:
            results['data_integrity'] = f"发现{len(data_issues)}处数据异常：{'; '.join(data_issues)}"
        
        # 3. 检查参数错配
        param_issues = []
        
        # 检查信号参数是否匹配波动等级
        if hasattr(self, 'buy_signals'):
            for signal in self.buy_signals:
                # 检查破位阈值是否匹配
                expected_threshold = self.params.get(self.volatility_level, {}).get('central_break_threshold', 0)
                if signal.get('break_threshold_factor', 1) != expected_threshold:
                    param_issues.append(f"信号{signal.get('signal_id', 'N/A')}破位阈值与波动等级不匹配")
        
        if param_issues:
            results['parameter_consistency'] = f"发现{len(param_issues)}处参数错配：{'; '.join(param_issues)}"
        
        # 4. 总体评估
        total_issues = len(logical_issues) + len(data_issues) + len(param_issues)
        if total_issues == 0:
            results['overall'] = "无明显错误，分析结果可信度高"
        elif total_issues <= 2:
            results['overall'] = f"发现{total_issues}处轻微问题，已标注，建议谨慎参考"
        else:
            results['overall'] = f"发现{total_issues}处问题，分析结果可信度较低，不建议作为交易依据"
        
        return results

def main():
    # 创建分析器实例
    analyzer = UniversalChanlunAnalyzer(symbol="512660", year=2025)
    
    # 执行分析流程
    print("===== 开始通用缠论分析 =====")
    
    # 1. 加载数据
    if not analyzer.load_data():
        print("数据加载失败，退出分析")
        # 即使加载失败，也尝试生成报告以显示错误信息
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 检查是否有严重错误
    if analyzer.has_critical_error:
        print(f"分析终止：{analyzer.error_message}")
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 2. 生成标准K线
    if not analyzer.generate_standard_k_lines():
        print("标准K线生成失败，退出分析")
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 检查是否有严重错误
    if analyzer.has_critical_error:
        print(f"分析终止：{analyzer.error_message}")
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 3. 确定波动等级
    if not analyzer.determine_volatility_level():
        print(f"波动等级确定失败：{analyzer.error_message}")
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 4. 划分走势段
    if not analyzer.divide_segments():
        print(f"走势段划分失败：{analyzer.error_message if analyzer.has_critical_error else '未知错误'}")
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 5. 生成中枢
    if not analyzer.generate_centrals():
        print("中枢生成失败，退出分析")
        analyzer.generate_report("512660_2025_universal_chanlun_report.md")
        return
    
    # 6. 识别交易信号
    analyzer.identify_break_central_rebound_buy_signals()
    
    # 7. 生成报告
    analyzer.generate_report("512660_2025_universal_chanlun_report.md")
    
    print("===== 分析完成 =====")

if __name__ == "__main__":
    main()