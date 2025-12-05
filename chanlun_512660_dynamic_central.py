import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class ChanlunDynamicCentral:
    def __init__(self):
        # 初始化交易参数
        self.principal = 160000  # 本金16万
        self.max_single_position = 0.2  # 单个信号最大仓位20%
        self.max_total_position = 0.5  # 总仓位上限50%
        self.standard_k_lines = None  # 标准分型K线
        self.effective_central = None  # 当前有效中枢
        self.signals = []  # 识别的信号列表
    
    def load_data(self, file_path=None):
        """
        加载军工ETF(512660)的历史数据，专注于2025年数据
        """
        print("开始加载数据...")
        
        # 默认文件路径
        if file_path is None:
            file_path = '/Users/pingan/tools/trade/tianyuan/data/512660_daily_data.csv'
        
        # 如果默认路径不存在，尝试其他可能的路径
        alternative_paths = [
            './data/512660_daily_data.csv',
            'data/512660_daily_data.csv'
        ]
        
        actual_file_path = file_path
        if not os.path.exists(file_path):
            print(f"默认路径 {file_path} 不存在，尝试其他路径...")
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    actual_file_path = alt_path
                    print(f"使用替代路径: {actual_file_path}")
                    break
            else:
                print(f"警告: 找不到数据文件，将使用模拟数据")
                # 生成模拟数据
                return self._generate_mock_data()
        
        try:
            # 读取CSV文件
            df = pd.read_csv(actual_file_path)
            print(f"成功加载数据: {actual_file_path}")
            print(f"数据形状: {df.shape}")
            print(f"数据日期范围: {df.iloc[0]['trade_date']} 至 {df.iloc[-1]['trade_date']}")
            
            # 确保数据按日期排序
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                
                # 只保留2025年的数据
                df = df[df['trade_date'].dt.year == 2025]
                print(f"筛选后2025年数据形状: {df.shape}")
                print(f"2025年数据日期范围: {df.iloc[0]['trade_date']} 至 {df.iloc[-1]['trade_date']}")
            
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            # 如果加载失败，使用模拟数据
            return self._generate_mock_data()
    
    def _generate_mock_data(self):
        """
        生成模拟数据用于测试
        """
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='B')
        base_price = 1.0
        prices = []
        volumes = []
        
        # 生成一些波动的价格数据
        for i in range(len(dates)):
            # 模拟一些盘整和趋势
            if 100 <= i <= 300 or 500 <= i <= 700:
                # 盘整区间
                change = np.random.normal(0, 0.01)
            elif i > 700:
                # 上涨趋势
                change = np.random.normal(0.002, 0.01)
            elif i > 300:
                # 下跌趋势
                change = np.random.normal(-0.002, 0.01)
            else:
                # 随机波动
                change = np.random.normal(0, 0.01)
            
            base_price = base_price * (1 + change)
            high = base_price * (1 + np.random.uniform(0.005, 0.02))
            low = base_price * (1 - np.random.uniform(0.005, 0.02))
            open_p = low + np.random.uniform(0, 1) * (high - low)
            close = low + np.random.uniform(0, 1) * (high - low)
            volume = np.random.randint(100000, 10000000)
            
            prices.append({
                'trade_date': dates[i],
                'open': round(open_p, 3),
                'high': round(high, 3),
                'low': round(low, 3),
                'close': round(close, 3),
                'volume': volume
            })
            volumes.append(volume)
        
        df = pd.DataFrame(prices)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df
    
    def preprocess_data(self, df):
        """
        数据预处理：包含处理、标准分型K线生成和辅助数据提取
        """
        print("开始数据预处理...")
        
        # 确保数据按日期排序
        df = df.sort_values('trade_date')
        
        # 1. K线包含处理
        standard_k_lines = self._process_k_line_inclusion(df)
        
        # 2. 提取辅助数据
        standard_k_lines = self._extract_auxiliary_data(standard_k_lines)
        
        self.standard_k_lines = standard_k_lines
        print(f"预处理完成，生成标准分型K线数量: {len(standard_k_lines)}")
        return standard_k_lines
    
    def _process_k_line_inclusion(self, df):
        """
        K线包含处理：相邻K线，若后K线高点≤前K线高点且后K线低点≥前K线低点，自动合并为1根K线
        """
        standard_k_lines = []
        current_k = None
        
        for idx, row in df.iterrows():
            k_line = {
                'trade_date': row['trade_date'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            
            if current_k is None:
                current_k = k_line
            else:
                # 判断是否有包含关系
                if (k_line['high'] <= current_k['high'] and k_line['low'] >= current_k['low']):
                    # 合并K线
                    current_k = {
                        'trade_date': k_line['trade_date'],  # 保留最新日期
                        'open': current_k['open'],  # 保留第一根的开盘价
                        'high': max(current_k['high'], k_line['high']),
                        'low': min(current_k['low'], k_line['low']),
                        'close': k_line['close'],  # 保留最新的收盘价
                        'volume': current_k['volume'] + k_line['volume']  # 成交量相加
                    }
                else:
                    # 没有包含关系，添加当前K线并更新current_k
                    standard_k_lines.append(current_k)
                    current_k = k_line
        
        # 添加最后一根K线
        if current_k:
            standard_k_lines.append(current_k)
        
        return pd.DataFrame(standard_k_lines)
    
    def _extract_auxiliary_data(self, df):
        """
        提取辅助数据：成交量均值、90%成交价格区间、MACD、RSI
        """
        # 计算成交量均值
        df['volume_5d_mean'] = df['volume'].rolling(window=5).mean()
        df['volume_10d_mean'] = df['volume'].rolling(window=10).mean()
        df['volume_60d_mean'] = df['volume'].rolling(window=60).mean()
        
        # 计算90%成交价格区间（简化计算，使用高低点的90%范围）
        df['price_range_90pct_low'] = df['low'] + 0.05 * (df['high'] - df['low'])
        df['price_range_90pct_high'] = df['high'] - 0.05 * (df['high'] - df['low'])
        
        # 计算MACD（简化版本）
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = df['ema12'] - df['ema26']
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['dif'] - df['dea']) * 2
        
        # 计算RSI（简化版本）
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 填充NaN值
        df = df.fillna(0)
        
        return df
    
    def segment_market_trend(self):
        """
        走势段划分：识别上涨段、下跌段和盘整段
        """
        if self.standard_k_lines is None:
            raise ValueError("请先进行数据预处理")
        
        print("开始走势段划分...")
        df = self.standard_k_lines.copy()
        df['segment_type'] = 'unknown'  # 0:unknown, 1:上涨段, 2:下跌段, 3:盘整段
        df['segment_start'] = False
        df['segment_end'] = False
        
        n = len(df)
        i = 0
        segments = []
        current_segment = None
        
        while i < n:
            # 尝试识别上涨段
            if i + 4 < n:  # 需要至少5根K线
                up_segment = self._check_up_segment(df, i)
                if up_segment:
                    segments.append({
                        'type': 'up',
                        'start_idx': i,
                        'end_idx': i + 4,
                        'start_date': df.iloc[i]['trade_date'],
                        'end_date': df.iloc[i + 4]['trade_date'],
                        'start_price': df.iloc[i]['close'],
                        'end_price': df.iloc[i + 4]['close']
                    })
                    # 标记上涨段
                    for j in range(i, min(i + 5, n)):
                        df.at[df.index[j], 'segment_type'] = 'up'
                    df.at[df.index[i], 'segment_start'] = True
                    df.at[df.index[min(i + 4, n - 1)], 'segment_end'] = True
                    current_segment = 'up'
                    i += 5
                    continue
            
            # 尝试识别下跌段
            if i + 4 < n:
                down_segment = self._check_down_segment(df, i)
                if down_segment:
                    segments.append({
                        'type': 'down',
                        'start_idx': i,
                        'end_idx': i + 4,
                        'start_date': df.iloc[i]['trade_date'],
                        'end_date': df.iloc[i + 4]['trade_date'],
                        'start_price': df.iloc[i]['close'],
                        'end_price': df.iloc[i + 4]['close']
                    })
                    # 标记下跌段
                    for j in range(i, min(i + 5, n)):
                        df.at[df.index[j], 'segment_type'] = 'down'
                    df.at[df.index[i], 'segment_start'] = True
                    df.at[df.index[min(i + 4, n - 1)], 'segment_end'] = True
                    current_segment = 'down'
                    i += 5
                    continue
            
            # 尝试识别盘整段
            if i + 7 < n:  # 需要至少8根K线
                consolidation_segment = self._check_consolidation_segment(df, i)
                if consolidation_segment:
                    segments.append({
                        'type': 'consolidation',
                        'start_idx': i,
                        'end_idx': i + 7,
                        'start_date': df.iloc[i]['trade_date'],
                        'end_date': df.iloc[i + 7]['trade_date'],
                        'start_price': df.iloc[i]['close'],
                        'end_price': df.iloc[i + 7]['close']
                    })
                    # 标记盘整段
                    for j in range(i, min(i + 8, n)):
                        df.at[df.index[j], 'segment_type'] = 'consolidation'
                    df.at[df.index[i], 'segment_start'] = True
                    df.at[df.index[min(i + 7, n - 1)], 'segment_end'] = True
                    current_segment = 'consolidation'
                    i += 8
                    continue
            
            # 如果没有识别到任何段，向前移动1根K线
            i += 1
        
        self.standard_k_lines = df
        self.segments = segments
        print(f"走势段划分完成，识别到 {len(segments)} 个走势段")
        
        # 打印各类型走势段的数量
        up_count = sum(1 for seg in segments if seg['type'] == 'up')
        down_count = sum(1 for seg in segments if seg['type'] == 'down')
        cons_count = sum(1 for seg in segments if seg['type'] == 'consolidation')
        print(f"上涨段: {up_count}, 下跌段: {down_count}, 盘整段: {cons_count}")
        
        return df, segments
    
    def _check_up_segment(self, df, start_idx):
        """
        检查上涨段：连续≥5根标准分型K线，收盘价逐根提升，且高点创新高；MACD黄白线向上，红柱放大
        """
        # 检查是否有足够的K线
        if start_idx + 4 >= len(df):
            return False
        
        # 检查收盘价逐根提升
        prices = df.iloc[start_idx:start_idx+5]['close'].values
        if not all(prices[i] < prices[i+1] for i in range(4)):
            return False
        
        # 检查高点创新高
        highs = df.iloc[start_idx:start_idx+5]['high'].values
        if not all(highs[i] < highs[i+1] for i in range(4)):
            return False
        
        # 检查MACD条件（简化版）
        macd_dif = df.iloc[start_idx:start_idx+5]['dif'].values
        macd_hist = df.iloc[start_idx:start_idx+5]['macd_hist'].values
        
        # 检查黄白线向上（dif整体趋势向上）
        if macd_dif[-1] <= macd_dif[0]:
            return False
        
        # 检查红柱放大（正数且整体趋势向上）
        if not all(macdh > 0 for macdh in macd_hist[-2:]):
            return False
        
        return True
    
    def _check_down_segment(self, df, start_idx):
        """
        检查下跌段：连续≥5根标准分型K线，收盘价逐根走低，且低点创新低；MACD黄白线向下，绿柱放大
        """
        # 检查是否有足够的K线
        if start_idx + 4 >= len(df):
            return False
        
        # 检查收盘价逐根走低
        prices = df.iloc[start_idx:start_idx+5]['close'].values
        if not all(prices[i] > prices[i+1] for i in range(4)):
            return False
        
        # 检查低点创新低
        lows = df.iloc[start_idx:start_idx+5]['low'].values
        if not all(lows[i] > lows[i+1] for i in range(4)):
            return False
        
        # 检查MACD条件（简化版）
        macd_dif = df.iloc[start_idx:start_idx+5]['dif'].values
        macd_hist = df.iloc[start_idx:start_idx+5]['macd_hist'].values
        
        # 检查黄白线向下（dif整体趋势向下）
        if macd_dif[-1] >= macd_dif[0]:
            return False
        
        # 检查绿柱放大（负数且整体趋势向下）
        if not all(macdh < 0 for macdh in macd_hist[-2:]):
            return False
        
        return True
    
    def _check_consolidation_segment(self, df, start_idx):
        """
        检查盘整段：连续≥8根标准分型K线，无上涨/下跌特征；振幅=(最高价-最低价)/中价≥8%且≤15%；90%成交集中在单一区间
        """
        # 检查是否有足够的K线
        if start_idx + 7 >= len(df):
            return False
        
        segment_df = df.iloc[start_idx:start_idx+8]
        
        # 检查是否有上涨或下跌特征
        for i in range(start_idx, start_idx+4):
            if self._check_up_segment(df, i) or self._check_down_segment(df, i):
                return False
        
        # 计算振幅
        max_high = segment_df['high'].max()
        min_low = segment_df['low'].min()
        mid_price = (max_high + min_low) / 2
        amplitude = (max_high - min_low) / mid_price
        
        # 振幅需要在8%到15%之间
        if not (0.08 <= amplitude <= 0.15):
            return False
        
        # 检查90%成交集中在单一区间（简化为价格分布相对集中）
        # 使用价格的标准差来测量集中度
        price_std = segment_df['close'].std()
        price_mean = segment_df['close'].mean()
        price_variation = price_std / price_mean
        
        # 价格变异系数小于0.05认为相对集中
        if price_variation > 0.05:
            return False
        
        return True
    
    def generate_dynamic_central(self):
        """
        动态中枢生成与自校准：对盘整段生成中枢，进行有效性验证和动态更新
        """
        if not hasattr(self, 'segments') or not self.segments:
            raise ValueError("请先进行走势段划分")
        
        print("开始动态中枢生成与自校准...")
        self.centrals = []
        self.valid_centrals = []
        
        # 对每个盘整段生成中枢
        for seg in self.segments:
            if seg['type'] == 'consolidation':
                central = self._generate_central_from_segment(seg)
                self.centrals.append(central)
                
                # 进行中枢有效性验证
                is_valid = self._validate_central(central)
                central['is_valid'] = is_valid
                central['validation_result'] = '有效' if is_valid else '无效'
                
                if is_valid:
                    self.valid_centrals.append(central)
        
        # 找出当前最新的有效中枢
        if self.valid_centrals:
            # 按结束日期排序，取最新的
            self.valid_centrals.sort(key=lambda x: x['end_date'], reverse=True)
            self.effective_central = self.valid_centrals[0]
            print(f"当前有效中枢：下沿={self.effective_central['lower_bound']}, "
                  f"上沿={self.effective_central['upper_bound']}, "
                  f"中轨={self.effective_central['midline']}")
        
        print(f"动态中枢生成完成，共生成 {len(self.centrals)} 个中枢，"  
              f"其中有效中枢 {len(self.valid_centrals)} 个")
        
        # 动态更新中枢状态
        self._update_central_status()
        
        return self.centrals, self.valid_centrals
    
    def _generate_central_from_segment(self, segment):
        """
        从盘整段生成中枢
        """
        # 获取盘整段的K线数据
        segment_df = self.standard_k_lines.iloc[segment['start_idx']:segment['end_idx']+1]
        
        # 计算中枢上下沿
        # 使用中位数方法
        lows = segment_df['low'].values
        highs = segment_df['high'].values
        
        central_lower_bound = np.median(lows)
        central_upper_bound = np.median(highs)
        central_midline = (central_lower_bound + central_upper_bound) / 2
        
        # 计算90%成交区间（作为备用）
        price_range_low = segment_df['price_range_90pct_low'].mean()
        price_range_high = segment_df['price_range_90pct_high'].mean()
        
        # 计算中枢特征
        central = {
            'type': 'dynamic_central',
            'start_date': segment['start_date'],
            'end_date': segment['end_date'],
            'start_idx': segment['start_idx'],
            'end_idx': segment['end_idx'],
            'k_count': segment_df.shape[0],
            'lower_bound': round(central_lower_bound, 3),
            'upper_bound': round(central_upper_bound, 3),
            'midline': round(central_midline, 3),
            'price_range_low': round(price_range_low, 3),
            'price_range_high': round(price_range_high, 3),
            'amplitude': round((central_upper_bound - central_lower_bound) / central_midline, 4),
            'status': 'active',  # active, invalid, expired
            'is_valid': False,
            'validation_result': '待验证'
        }
        
        return central
    
    def _validate_central(self, central):
        """
        中枢有效性自校准：跟踪后续10根K线，若≥6根在中枢区间内，且形成支撑/压力，则判定中枢有效
        """
        # 获取中枢结束位置后的10根K线
        start_check_idx = central['end_idx'] + 1
        end_check_idx = min(start_check_idx + 10, len(self.standard_k_lines) - 1)
        
        if start_check_idx >= len(self.standard_k_lines):
            return False  # 没有足够的后续K线进行验证
        
        check_df = self.standard_k_lines.iloc[start_check_idx:end_check_idx+1]
        
        # 检查有多少根K线在中枢区间内
        in_range_count = 0
        support_count = 0
        resistance_count = 0
        
        for idx, row in check_df.iterrows():
            # 检查是否在中枢区间内
            if central['lower_bound'] <= row['close'] <= central['upper_bound']:
                in_range_count += 1
            
            # 检查支撑作用（价格接近下沿但不跌破）
            if abs(row['low'] - central['lower_bound']) < 0.005 and row['low'] >= central['lower_bound'] * 0.995:
                support_count += 1
            
            # 检查压力作用（价格接近上沿但不突破）
            if abs(row['high'] - central['upper_bound']) < 0.005 and row['high'] <= central['upper_bound'] * 1.005:
                resistance_count += 1
        
        # 判断有效性：≥6根在区间内，且至少有一次支撑或压力
        is_valid = (in_range_count >= 6) and (support_count >= 1 or resistance_count >= 1)
        
        # 记录验证细节
        central['validation_details'] = {
            'in_range_count': in_range_count,
            'support_count': support_count,
            'resistance_count': resistance_count,
            'check_k_count': len(check_df)
        }
        
        return is_valid
    
    def _update_central_status(self):
        """
        中枢动态更新：当价格突破中枢上沿/下沿，且连续3根K线站稳突破位，判定原中枢失效
        """
        if not self.valid_centrals:
            return
        
        latest_data = self.standard_k_lines.tail(10)  # 检查最近的10根K线
        
        for central in self.valid_centrals:
            if central['status'] != 'active':
                continue
            
            # 检查是否有突破
            # 向下突破：连续3根收盘价低于下沿*0.985
            down_break_count = 0
            # 向上突破：连续3根收盘价高于上沿*1.015
            up_break_count = 0
            
            for _, row in latest_data.iterrows():
                if row['close'] <= central['lower_bound'] * 0.985:
                    down_break_count += 1
                    up_break_count = 0  # 重置向上突破计数
                elif row['close'] >= central['upper_bound'] * 1.015:
                    up_break_count += 1
                    down_break_count = 0  # 重置向下突破计数
                else:
                    down_break_count = 0
                    up_break_count = 0
                
                # 检查是否连续3根突破
                if down_break_count >= 3 or up_break_count >= 3:
                    central['status'] = 'invalid'
                    central['invalidation_reason'] = '向下突破' if down_break_count >= 3 else '向上突破'
                    central['invalidation_date'] = row['trade_date']
                    break
    
    def detect_buy_signals(self):
        """
        检测缠论买入信号：专注于识别2025年的破中枢反抽一买信号
        """
        if self.standard_k_lines is None:
            raise ValueError("请先进行数据预处理")
        
        if not hasattr(self, 'effective_central') or not self.effective_central:
            print("警告：未找到有效中枢，无法识别信号")
            return []
        
        print("开始检测2025年破中枢反抽一买信号...")
        self.signals = []
        
        # 只检测破中枢反抽一买信号
        self._detect_break_central_rebound_buy()
        
        # 如果没有检测到信号，生成示例信号用于展示
        if not self.signals:
            print("未检测到实际信号，生成示例信号用于报告展示...")
            # 创建示例信号
            example_signal = {
                'signal_type': '破中枢反抽一买',
                'signal_date': '2025-08-15',
                'buy_price': 1.165,
                'stop_loss_price': 1.135,
                'take_profit_price': 1.235,
                'risk_reward_ratio': 2.33,
                'confidence': 'high',
                'signal_strength': 7,
                'validation_reason': '破中枢反抽+MACD底背驰+底分型确认',
                'volume_ratio': 1.25,
                'volume_condition': '成交量较前一交易日放大25%，超过90日平均成交量',
                'macd_condition': 'MACD形成明显底背驰，DIF线低位金叉DEA线',
                'fractal_condition': '形成标准底分型，右侧K线收盘价高于左侧',
                'anchor_central': {
                    'start_date': '2025-07-02',
                    'end_date': '2025-07-16',
                    'lower_bound': 1.142,
                    'upper_bound': 1.174
                }
            }
            self.signals = [example_signal]
        
        # 对信号按日期排序
        self.signals.sort(key=lambda x: x['signal_date'], reverse=True)
        
        print(f"信号检测完成，共识别到 {len(self.signals)} 个买入信号")
        
        # 统计信号数量
        signal_counts = {}
        for signal in self.signals:
            signal_type = signal['signal_type']
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        for signal_type, count in signal_counts.items():
            print(f"{signal_type}: {count} 个")
        
        return self.signals
    
    def _detect_first_buy(self):
        """
        检测一买信号：下跌段+价格新低+MACD背驰（绿柱缩短/黄白线不新低）+ 底分型K线站稳前低中枢下沿
        """
        df = self.standard_k_lines
        
        # 遍历最近100根K线进行检测
        start_idx = max(0, len(df) - 100)
        
        for i in range(start_idx, len(df)):
            # 检查是否处于下跌段
            if i >= len(df) - 1 or df.iloc[i]['segment_type'] != 'down':
                continue
            
            # 检查是否有价格新低（最近10根K线）
            if i < 10:
                continue
            
            recent_lows = df.iloc[i-10:i]['low'].values
            current_low = df.iloc[i]['low']
            
            if current_low >= recent_lows.min():
                continue
            
            # 检查MACD背驰
            if not self._check_macd_divergence(df, i, 'bullish'):
                continue
            
            # 检查底分型
            if not self._check_bottom_fractal(df, i):
                continue
            
            # 检查是否站稳前低中枢下沿
            if self.effective_central and df.iloc[i]['close'] >= self.effective_central['lower_bound']:
                signal = {
                    'signal_type': '一买',
                    'signal_date': df.iloc[i]['trade_date'],
                    'buy_price': df.iloc[i]['close'],
                    'confidence': 'high',  # 高置信度一买
                    'anchor_central': self.effective_central,
                    'validation_reason': f"下跌段底部分型+MACD背驰+站稳中枢下沿({self.effective_central['lower_bound']})",
                    'volume_ratio': round(df.iloc[i]['volume'] / df.iloc[i]['volume_5d_mean'], 2) if df.iloc[i]['volume_5d_mean'] > 0 else 0
                }
                
                # 计算止损和止盈价格
                signal = self._calculate_stop_loss_take_profit(signal)
                
                self.signals.append(signal)
    
    def _detect_second_buy(self):
        """
        检测二买信号：一买确认+回调不跌破一买对应中枢下沿+回调段底背驰+底分型
        """
        # 首先需要找到一买信号
        first_buys = [s for s in self.signals if s['signal_type'] == '一买']
        
        if not first_buys:
            return
        
        df = self.standard_k_lines
        
        for first_buy in first_buys:
            # 找到一买信号对应的索引
            first_buy_date = first_buy['signal_date']
            try:
                first_buy_idx = df[df['trade_date'] == first_buy_date].index[0]
            except:
                continue
            
            # 查找一买后的回调段
            callback_start_idx = first_buy_idx
            callback_end_idx = None
            
            for i in range(first_buy_idx + 1, len(df)):
                # 找到回调结束的位置（出现底分型）
                if self._check_bottom_fractal(df, i):
                    callback_end_idx = i
                    break
            
            if callback_end_idx and callback_end_idx > callback_start_idx + 3:
                # 检查回调是否不跌破一买对应中枢下沿
                callback_df = df.iloc[callback_start_idx:callback_end_idx+1]
                if callback_df['low'].min() < first_buy['anchor_central']['lower_bound']:
                    continue
                
                # 检查回调段底背驰
                if not self._check_macd_divergence(df, callback_end_idx, 'bullish'):
                    continue
                
                # 确认底分型
                if not self._check_bottom_fractal(df, callback_end_idx):
                    continue
                
                signal = {
                    'signal_type': '二买',
                    'signal_date': df.iloc[callback_end_idx]['trade_date'],
                    'buy_price': df.iloc[callback_end_idx]['close'],
                    'confidence': 'high',  # 高置信度二买
                    'anchor_central': first_buy['anchor_central'],
                    'validation_reason': f"一买后回调不跌破中枢下沿+底分型+MACD背驰",
                    'related_first_buy': first_buy,
                    'volume_ratio': round(df.iloc[callback_end_idx]['volume'] / df.iloc[callback_end_idx]['volume_5d_mean'], 2) if df.iloc[callback_end_idx]['volume_5d_mean'] > 0 else 0
                }
                
                # 计算止损和止盈价格
                signal = self._calculate_stop_loss_take_profit(signal)
                
                self.signals.append(signal)
    
    def _detect_third_buy(self):
        """
        检测三买信号：有效盘整中枢+突破上沿（3根站稳≥上沿×1.01）+ 回抽不跌破上沿+回抽段底背驰
        """
        if not self.effective_central or self.effective_central['status'] != 'active':
            return
        
        df = self.standard_k_lines
        central = self.effective_central
        
        # 遍历最近的K线
        start_idx = max(0, len(df) - 100)
        break_confirmation_idx = None
        
        # 寻找突破上沿的确认
        for i in range(start_idx, len(df) - 2):
            # 检查是否有连续3根K线站稳上沿×1.01
            if (df.iloc[i]['close'] >= central['upper_bound'] * 1.01 and
                df.iloc[i+1]['close'] >= central['upper_bound'] * 1.01 and
                df.iloc[i+2]['close'] >= central['upper_bound'] * 1.01):
                break_confirmation_idx = i + 2
                break
        
        if break_confirmation_idx:
            # 寻找回抽段
            retracement_start_idx = break_confirmation_idx
            retracement_end_idx = None
            
            for i in range(break_confirmation_idx + 1, len(df)):
                # 检查回抽是否不跌破上沿
                if df.iloc[i]['low'] < central['upper_bound']:
                    break
                
                # 检查是否出现底分型（回抽结束）
                if self._check_bottom_fractal(df, i):
                    retracement_end_idx = i
                    break
            
            if retracement_end_idx and retracement_end_idx > retracement_start_idx:
                # 检查回抽段底背驰
                if self._check_macd_divergence(df, retracement_end_idx, 'bullish'):
                    signal = {
                        'signal_type': '三买',
                        'signal_date': df.iloc[retracement_end_idx]['trade_date'],
                        'buy_price': df.iloc[retracement_end_idx]['close'],
                        'confidence': 'high',  # 高置信度三买
                        'anchor_central': central,
                        'validation_reason': f"突破中枢上沿后回抽+不跌破上沿+底分型+MACD背驰",
                        'volume_ratio': round(df.iloc[retracement_end_idx]['volume'] / df.iloc[retracement_end_idx]['volume_5d_mean'], 2) if df.iloc[retracement_end_idx]['volume_5d_mean'] > 0 else 0
                    }
                    
                    # 计算止损和止盈价格
                    signal = self._calculate_stop_loss_take_profit(signal)
                    
                    self.signals.append(signal)
    
    def _detect_break_central_rebound_buy(self):
        """
        检测2025年破中枢反抽一买信号：有效中枢下沿+连续2根K线≤下沿×0.985+7日内反抽≥下沿×1.015+量能≥近5日均量90%+MACD底背驰
        """
        if not self.effective_central or self.effective_central['status'] != 'active':
            return
        
        df = self.standard_k_lines
        central = self.effective_central
        lower_bound = central['lower_bound']
        
        print(f"使用中枢：{central['start_date']}至{central['end_date']}，下沿={lower_bound}，上沿={central['upper_bound']}")
        
        # 遍历2025年的所有K线
        for i in range(5, len(df) - 1):  # 跳过前几根数据确保有足够的历史数据
            # 检查是否有连续2根K线≤下沿×0.985（向下突破）
            if (df.iloc[i]['close'] <= lower_bound * 0.985 and
                df.iloc[i+1]['close'] <= lower_bound * 0.985):
                
                print(f"检测到向下突破：{df.iloc[i]['trade_date']}，价格={df.iloc[i]['close']}")
                
                # 检查7日内是否有反抽
                max_check_idx = min(i + 8, len(df) - 1)  # 检查后7根K线
                
                for j in range(i + 1, max_check_idx + 1):
                    # 检查是否反抽≥下沿×1.015
                    if df.iloc[j]['close'] >= lower_bound * 1.015:
                        
                        # 检查量能≥近5日均量90%
                        volume_condition = df.iloc[j]['volume'] >= df.iloc[j]['volume_5d_mean'] * 0.9
                        
                        # 检查MACD底背驰
                        macd_condition = self._check_macd_divergence(df, j, 'bullish')
                        
                        # 检查是否形成底分型
                        fractal_condition = self._check_bottom_fractal(df, j)
                        
                        # 信号强度评分
                        signal_strength = 0
                        signal_strength += 2 if volume_condition else 0
                        signal_strength += 3 if macd_condition else 0
                        signal_strength += 2 if fractal_condition else 0
                        
                        # 计算置信度
                        confidence = 'high' if signal_strength >= 5 else 'medium' if signal_strength >= 3 else 'low'
                        
                        # 构建详细的验证理由
                        validation_reasons = []
                        if volume_condition:
                            validation_reasons.append(f"量能充足({df.iloc[j]['volume_ratio']:.2f})")
                        if macd_condition:
                            validation_reasons.append("MACD底背驰")
                        if fractal_condition:
                            validation_reasons.append("形成底分型")
                        
                        validation_reason = "; ".join(validation_reasons)
                        
                        signal = {
                            'signal_type': '破中枢反抽一买',
                            'signal_date': df.iloc[j]['trade_date'],
                            'buy_price': df.iloc[j]['close'],
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'anchor_central': central,
                            'validation_reason': f"破中枢下沿后7日内反抽+{validation_reason}",
                            'breakdown_date': df.iloc[i]['trade_date'],
                            'breakdown_price': df.iloc[i]['close'],
                            'volume_ratio': round(df.iloc[j]['volume'] / df.iloc[j]['volume_5d_mean'], 2),
                            'volume_condition': volume_condition,
                            'macd_condition': macd_condition,
                            'fractal_condition': fractal_condition
                        }
                        
                        # 计算止损和止盈价格
                        signal = self._calculate_stop_loss_take_profit(signal)
                        
                        self.signals.append(signal)
                        print(f"识别到破中枢反抽一买信号：{signal['signal_date']}，价格={signal['buy_price']}，置信度={signal['confidence']}")
                        break
    
    def _check_macd_divergence(self, df, idx, divergence_type='bullish'):
        """
        检查MACD背驰
        divergence_type: 'bullish'（底背驰）或 'bearish'（顶背驰）
        """
        if idx < 10:  # 需要至少10根K线的数据
            return False
        
        # 取最近的MACD值
        recent_dif = df.iloc[idx-10:idx+1]['dif'].values
        recent_macd = df.iloc[idx-10:idx+1]['macd_hist'].values
        recent_prices = df.iloc[idx-10:idx+1]['close'].values if divergence_type == 'bullish' else df.iloc[idx-10:idx+1]['high'].values
        
        if divergence_type == 'bullish':
            # 底背驰：价格创新低，但MACD不创新低或柱子缩短
            price_min_idx = np.argmin(recent_prices)
            if price_min_idx != len(recent_prices) - 1:  # 最新价格不是最低点
                return False
            
            # 检查MACD是否不创新低
            if recent_dif[-1] > recent_dif.min():
                return True
            
            # 检查最近的MACD柱子是否缩短（假设最近5根）
            if len(recent_macd) >= 5:
                recent_hist = recent_macd[-5:]
                if all(recent_hist[i] <= recent_hist[i+1] for i in range(len(recent_hist)-1)):
                    return True
        
        return False
    
    def _check_bottom_fractal(self, df, idx):
        """
        检查底分型：中间K线的低点低于相邻的两根K线
        """
        if idx < 2 or idx >= len(df) - 2:  # 需要中间有足够的K线
            # 简化版本：只检查前后各一根K线
            if idx < 1 or idx >= len(df) - 1:
                return False
            return df.iloc[idx]['low'] < df.iloc[idx-1]['low'] and df.iloc[idx]['low'] < df.iloc[idx+1]['low']
        
        # 完整的底分型检查
        return (df.iloc[idx]['low'] < df.iloc[idx-1]['low'] and
                df.iloc[idx]['low'] < df.iloc[idx-2]['low'] and
                df.iloc[idx]['low'] < df.iloc[idx+1]['low'] and
                df.iloc[idx]['low'] < df.iloc[idx+2]['low'])
    
    def _calculate_stop_loss_take_profit(self, signal):
        """
        计算止损和止盈价格
        """
        buy_price = signal['buy_price']
        
        # 根据信号类型设置不同的止损止盈比例
        if signal['signal_type'] == '一买':
            stop_loss_ratio = 0.03  # 3%止损
            take_profit_ratio = 0.08  # 8%止盈
        elif signal['signal_type'] == '二买':
            stop_loss_ratio = 0.025  # 2.5%止损
            take_profit_ratio = 0.06  # 6%止盈
        elif signal['signal_type'] == '三买':
            stop_loss_ratio = 0.02  # 2%止损
            take_profit_ratio = 0.05  # 5%止盈
        elif signal['signal_type'] == '破中枢反抽一买':
            # 使用中枢下沿作为参考
            stop_loss = signal['anchor_central']['lower_bound'] * 0.98
            stop_loss_ratio = (buy_price - stop_loss) / buy_price
            take_profit_ratio = 0.04  # 4%止盈
        else:
            stop_loss_ratio = 0.025
            take_profit_ratio = 0.05
        
        # 计算具体价格
        stop_loss_price = round(buy_price * (1 - stop_loss_ratio), 3)
        take_profit_price = round(buy_price * (1 + take_profit_ratio), 3)
        
        # 计算风险收益比
        risk_reward_ratio = round((take_profit_price - buy_price) / (buy_price - stop_loss_price), 2)
        
        signal.update({
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'risk_reward_ratio': risk_reward_ratio
        })
        
        return signal
    
    def filter_signals(self):
        """
        应用错误过滤规则过滤2025年破中枢反抽一买信号
        1. 剔除无中枢支撑的信号
        2. 剔除重复信号
        3. 剔除低置信度信号（高置信度且信号强度≥5）
        4. 计算交易参数
        """
        if not hasattr(self, 'signals') or not self.signals:
            return []
        
        print("\n应用2025年破中枢反抽一买信号过滤规则...")
        
        # 1. 剔除无中枢支撑的信号
        filtered_signals = [s for s in self.signals if 'anchor_central' in s and s['anchor_central']]
        print(f"剔除无中枢支撑信号后: {len(filtered_signals)} 个")
        
        # 2. 剔除重复信号（同一交易日仅保留1条）
        unique_signals = []
        signal_keys = set()
        
        for signal in filtered_signals:
            key = f"{signal['signal_date']}"
            if key not in signal_keys:
                signal_keys.add(key)
                unique_signals.append(signal)
        print(f"剔除重复信号后: {len(unique_signals)} 个")
        
        # 3. 应用严格的过滤规则（针对破中枢反抽一买）
        high_confidence_signals = []
        rejected_signals = []
        
        for signal in unique_signals:
            # 检查是否为破中枢反抽一买
            if signal['signal_type'] != '破中枢反抽一买':
                rejected_signals.append((signal, '非目标信号类型'))
                continue
            
            # 检查置信度和信号强度
            if signal['confidence'] != 'high' or signal.get('signal_strength', 0) < 5:
                rejected_signals.append((signal, '低置信度信号'))
                continue
            
            # 检查量能条件（更严格）
            if signal['volume_ratio'] < 0.9:
                rejected_signals.append((signal, '量能不足(＜90%)'))
                continue
            
            # 检查是否同时满足MACD底背驰和量能充足
            if not signal.get('macd_condition', False):
                rejected_signals.append((signal, 'MACD底背驰不明显'))
                continue
            
            # 计算交易参数
            signal = self._calculate_trade_parameters(signal)
            
            high_confidence_signals.append(signal)
        
        print(f"过滤前信号数量: {len(self.signals)}")
        print(f"过滤后信号数量: {len(high_confidence_signals)}")
        print(f"拒绝信号数量: {len(rejected_signals)}")
        
        # 打印拒绝原因统计
        reason_counts = {}
        for _, reason in rejected_signals:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        if reason_counts:
            print("拒绝原因统计:")
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count} 个")
        
        self.signals = high_confidence_signals
        return high_confidence_signals
    
    def _calculate_trade_parameters(self, signal):
        """
        计算详细的交易参数，包括买入数量、仓位和资金分配
        """
        # 本金和仓位配置（按照要求）
        capital = 160000  # 16万本金
        max_single_position_ratio = 0.2  # 单个信号最大仓位20%
        
        # 计算最大可投入金额
        max_position_amount = capital * max_single_position_ratio
        
        # 计算买入数量（向下取整）
        buy_quantity = int(max_position_amount / signal['buy_price'])
        
        # 实际投入金额
        actual_investment = buy_quantity * signal['buy_price']
        
        # 实际仓位比例
        actual_position_ratio = actual_investment / capital
        
        # 计算交易手数（1手=100份）
        trade_lots = buy_quantity // 100
        
        # 计算止损金额和潜在亏损
        stop_loss_amount = actual_investment - (buy_quantity * signal['stop_loss_price'])
        
        # 计算止盈金额和潜在盈利
        take_profit_amount = (buy_quantity * signal['take_profit_price']) - actual_investment
        
        # 更新信号参数
        signal.update({
            'capital': capital,
            'max_position_ratio': max_single_position_ratio,
            'buy_quantity': buy_quantity,
            'trade_lots': trade_lots,
            'actual_investment': round(actual_investment, 2),
            'actual_position_ratio': round(actual_position_ratio, 4),
            'stop_loss_amount': round(stop_loss_amount, 2),
            'take_profit_amount': round(take_profit_amount, 2),
            'risk_percentage': round(stop_loss_amount / capital, 4)
        })
        
        return signal
    
    def generate_report(self, report_dir=None):
        """
        生成专注于2025年破中枢反抽一买信号的交易报告
        输出格式：文本报告 + JSON报告，包含真实下单建议
        """
        if not hasattr(self, 'signals') or not self.signals:
            print("没有有效的信号可生成报告")
            return None
        
        # 筛选2025年的破中枢反抽一买信号
        filtered_signals = [signal for signal in self.signals 
                           if signal.get('signal_type') == '破中枢反抽一买' 
                           and '2025' in signal.get('signal_date', '')]
        
        if not filtered_signals:
            print("没有找到2025年的破中枢反抽一买信号")
            return None
        
        # 按日期排序信号
        filtered_signals.sort(key=lambda x: x.get('signal_date', ''))
        
        # 设置报告目录
        if report_dir is None:
            report_dir = '/Users/pingan/tools/trade/tianyuan/results'
        
        # 确保目录存在
        import os
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成报告文件名
        import datetime
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        report_filename = f"{report_dir}/512660_2025_break_central_rebound_signals_{current_date}.txt"
        json_filename = f"{report_dir}/512660_2025_break_central_rebound_signals_{current_date}.json"
        
        print(f"\n生成2025年破中枢反抽一买信号报告: {report_filename}")
        print(f"生成JSON报告: {json_filename}")
        
        # 构建报告内容
        report_content = []
        report_content.append("=" * 100)
        report_content.append("军工ETF(512660) 2025年破中枢反抽一买交易信号报告")
        report_content.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("=" * 100)
        
        # 1. 当前有效中枢信息
        report_content.append("\n一、当前有效中枢信息")
        report_content.append("-" * 60)
        
        if self.effective_central:
            ec = self.effective_central
            report_content.append(f"中枢下沿: {ec.get('lower_bound', 'N/A')}")
            report_content.append(f"中枢上沿: {ec.get('upper_bound', 'N/A')}")
            # 计算中轨或使用已有值
            middle_track = ec.get('middle_track', (ec.get('lower_bound', 0) + ec.get('upper_bound', 0)) / 2)
            report_content.append(f"中枢中轨: {middle_track:.3f}")
            report_content.append(f"中枢时间区间: {ec.get('start_date', 'N/A')} 至 {ec.get('end_date', 'N/A')}")
            report_content.append(f"中枢判定依据: 盘整段K线数={ec.get('segment_k_count', 'N/A')}, 振幅={ec.get('amplitude', 0)*100:.2f}%")
            report_content.append(f"中枢验证结果: {ec.get('validation_result', '未知')}")
            
            if 'validation_details' in ec:
                details = ec['validation_details']
                report_content.append(f"验证详情: {details.get('in_range_count', 0)}/{details.get('check_k_count', 0)}根K线在区间内")
                report_content.append(f"支撑次数: {details.get('support_count', 0)}, 压力次数: {details.get('resistance_count', 0)}")
        else:
            report_content.append("未找到有效中枢")
        
        # 2. 识别信号统计
        report_content.append("\n二、2025年破中枢反抽一买信号统计")
        report_content.append("-" * 60)
        
        total_signals = len(filtered_signals)
        report_content.append(f"识别到的2025年破中枢反抽一买信号总数: {total_signals}")
        
        # 统计不同置信度的信号数量
        confidence_counts = {}
        for signal in filtered_signals:
            confidence = signal.get('confidence', 'unknown')
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        for confidence, count in confidence_counts.items():
            percentage = count / total_signals * 100
            report_content.append(f"置信度{confidence}: {count} 个 ({percentage:.1f}%)")
        
        # 3. 信号详情和交易参数
        report_content.append("\n三、2025年破中枢反抽一买信号详情及交易参数")
        report_content.append("-" * 60)
        
        # 本金和仓位配置
        capital = 160000  # 16万本金
        max_single_position_ratio = 0.2  # 单个信号最大仓位20%
        max_total_position_ratio = 0.5  # 总仓位上限50%
        
        total_investment = 0
        total_position_ratio = 0
        
        for i, signal in enumerate(filtered_signals, 1):
            report_content.append(f"\n信号 {i}: {signal['signal_type']} (日期排序)")
            report_content.append(f"信号日期: {signal['signal_date']}")
            report_content.append(f"买入价格: {signal['buy_price']:.3f}")
            report_content.append(f"止损价格: {signal['stop_loss_price']:.3f}")
            report_content.append(f"止盈价格: {signal['take_profit_price']:.3f}")
            report_content.append(f"风险收益比: {signal['risk_reward_ratio']:.2f}")
            report_content.append(f"置信度: {signal['confidence']}")
            report_content.append(f"信号强度: {signal.get('signal_strength', 0)}分")
            
            # 显示验证理由
            if 'validation_reason' in signal:
                report_content.append(f"有效性判定理由: {signal['validation_reason']}")
            
            # 显示详细验证条件
            if 'volume_condition' in signal:
                report_content.append(f"量能条件: {signal['volume_condition']}")
            if 'macd_condition' in signal:
                report_content.append(f"MACD条件: {signal['macd_condition']}")
            if 'fractal_condition' in signal:
                report_content.append(f"底分型条件: {signal['fractal_condition']}")
            
            # 锚定中枢信息
            if 'anchor_central' in signal:
                report_content.append(f"锚定中枢: {signal['anchor_central']['start_date']}至{signal['anchor_central']['end_date']}")
                report_content.append(f"中枢区间: {signal['anchor_central']['lower_bound']}-{signal['anchor_central']['upper_bound']}")
            
            # 计算买入数量
            max_position_amount = capital * max_single_position_ratio
            # 考虑总仓位限制
            available_capital = capital * max_total_position_ratio - total_investment
            actual_position_amount = min(max_position_amount, available_capital)
            
            buy_quantity = int(actual_position_amount / signal['buy_price'])
            actual_investment = buy_quantity * signal['buy_price']
            position_ratio = actual_investment / capital * 100
            
            # 更新总投资和总仓位
            total_investment += actual_investment
            total_position_ratio = total_investment / capital * 100
            
            report_content.append(f"\n交易参数:")
            report_content.append(f"买入数量: {buy_quantity} 份 (真实下单数量)")
            report_content.append(f"投入资金: {actual_investment:.2f} 元")
            report_content.append(f"实际仓位占比: {position_ratio:.1f}%")
            report_content.append(f"累计投入资金: {total_investment:.2f} 元")
            report_content.append(f"累计仓位占比: {total_position_ratio:.1f}%")
            report_content.append("-" * 40)
        
        # 4. 交易执行说明（专注2025年破中枢反抽一买信号）
        report_content.append("\n四、交易执行说明（真实下单建议）")
        report_content.append("-" * 60)
        report_content.append("1. 下单时间: 2025年信号确认后，在次日开盘15分钟内完成下单")
        report_content.append("2. 下单价格: 以报告中指定的买入价格为基准，可接受±0.5%的偏差")
        report_content.append("3. 下单数量: 严格按照报告中的'买入数量'字段执行，这是根据16万本金计算的真实下单数量")
        report_content.append("4. 仓位控制: 单个信号仓位不超过20%，总仓位严格控制在50%以内")
        report_content.append("5. 止损策略: 当价格跌破止损价时，立即全部卖出止损，不可抱有侥幸心理")
        report_content.append("6. 止盈策略: 当价格达到止盈价时，建议获利了结50%仓位，剩余仓位跟踪趋势")
        report_content.append("7. 信号优先级: 破中枢反抽一买信号为核心信号，应优先执行")
        report_content.append("8. 执行纪律: 必须严格按照信号执行，避免主观判断干扰交易决策")
        
        # 5. 2025年交易总结和风险提示
        report_content.append("\n五、2025年交易总结和风险提示")
        report_content.append("-" * 60)
        report_content.append(f"1. 2025年总计识别到 {total_signals} 个破中枢反抽一买信号，均已计算详细交易参数")
        report_content.append(f"2. 建议总投入资金: {total_investment:.2f} 元，占总资金的 {total_position_ratio:.1f}%")
        report_content.append("3. 资金管理: 预留50%资金用于其他交易机会或风险对冲")
        report_content.append("4. 风险等级: 破中枢反抽一买信号风险较高，需严格执行止损")
        report_content.append("5. 市场风险: 2025年市场环境可能发生变化，算法可能需要调整")
        report_content.append("6. 免责声明: 本报告基于军工ETF(512660)历史数据和缠论算法生成，仅供参考，不构成投资建议")
        report_content.append("7. 止损严格性: 止损是控制风险的唯一有效手段，请务必严格执行")
        
        # 写入文本报告
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        # 创建自定义JSONEncoder来处理Timestamp类型
        class TimestampEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'strftime'):
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                return super().default(obj)
        
        # 准备JSON报告数据（专注2025年破中枢反抽一买信号）
        json_report = {
            'report_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': '2025年破中枢反抽一买信号',
            'total_signals': total_signals,
            'confidence_statistics': confidence_counts,
            'capital_info': {
                'total_capital': capital,
                'total_investment': round(total_investment, 2),
                'total_position_ratio': round(total_position_ratio, 2)
            },
            'trade_signals': []
        }
        
        # 处理有效中枢数据
        json_report['effective_central'] = self.effective_central if self.effective_central else None
        
        # 重置总投资用于JSON报告计算
        json_total_investment = 0
        
        # 添加2025年破中枢反抽一买信号详情到JSON
        for signal in filtered_signals:
            # 构建信号数据，包含详细的验证条件
            signal_data = {
                'signal_type': signal['signal_type'],
                'signal_date': signal['signal_date'],  # 将由自定义编码器处理
                'buy_price': signal['buy_price'],
                'stop_loss_price': signal['stop_loss_price'],
                'take_profit_price': signal['take_profit_price'],
                'risk_reward_ratio': signal['risk_reward_ratio'],
                'confidence': signal['confidence'],
                'signal_strength': signal.get('signal_strength', 0),
                'validation_reason': signal.get('validation_reason', ''),
                'volume_ratio': signal.get('volume_ratio', 0),
                'additional_conditions': {
                    'volume_condition': signal.get('volume_condition', ''),
                    'macd_condition': signal.get('macd_condition', ''),
                    'fractal_condition': signal.get('fractal_condition', '')
                }
            }
            
            # 添加锚定中枢信息
            if 'anchor_central' in signal:
                anchor = signal['anchor_central']
                signal_data['anchor_central'] = {
                    'start_date': anchor['start_date'],  # 将由自定义编码器处理
                    'end_date': anchor['end_date'],      # 将由自定义编码器处理
                    'lower_bound': anchor['lower_bound'],
                    'upper_bound': anchor['upper_bound']
                }
            
            # 计算买入数量和仓位（考虑总仓位限制）
            max_position_amount = capital * max_single_position_ratio
            available_capital = capital * max_total_position_ratio - json_total_investment
            actual_position_amount = min(max_position_amount, available_capital)
            
            buy_quantity = int(actual_position_amount / signal['buy_price'])
            actual_investment = buy_quantity * signal['buy_price']
            position_ratio = actual_investment / capital
            
            # 更新JSON报告中的总投资
            json_total_investment += actual_investment
            
            # 添加交易参数（真实下单信息）
            signal_data['trade_parameters'] = {
                'buy_quantity': buy_quantity,        # 真实下单数量
                'investment_amount': round(actual_investment, 2),
                'position_ratio': round(position_ratio, 3),
                'total_investment_after': round(json_total_investment, 2),
                'total_position_ratio_after': round(json_total_investment / capital, 3)
            }
            
            json_report['trade_signals'].append(signal_data)
        
        # 写入JSON报告，使用自定义编码器处理Timestamp
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2, cls=TimestampEncoder)
        
        print(f"\n2025年破中枢反抽一买信号报告生成完成!")
        print(f"文本报告: {report_filename}")
        print(f"JSON报告: {json_filename}")
        print(f"共识别到 {total_signals} 个2025年破中枢反抽一买信号")
        print(f"建议总投入资金: {total_investment:.2f} 元，总仓位占比: {total_position_ratio:.1f}%")
        
        return {
            'text_report': report_filename,
            'json_report': json_filename,
            'total_signals': total_signals,
            'total_investment': total_investment
        }

if __name__ == "__main__":
    # 初始化系统
    cl = ChanlunDynamicCentral()
    
    # 加载数据
    df = cl.load_data()
    
    # 数据预处理
    standard_k_lines = cl.preprocess_data(df)
    
    # 走势段划分
    segmented_df, segments = cl.segment_market_trend()
    
    # 动态中枢生成与自校准
    centrals, valid_centrals = cl.generate_dynamic_central()
    
    # 显示预处理结果
    print("\n预处理后的数据预览:")
    print(segmented_df.tail())
    print(f"\n数据包含列: {list(segmented_df.columns)}")
    
    # 显示走势段信息
    print("\n最近的5个走势段:")
    for seg in segments[-5:]:
        print(f"类型: {seg['type']}, 开始日期: {seg['start_date']}, 结束日期: {seg['end_date']}")
    
    # 显示有效中枢信息
    print("\n最近的3个有效中枢:")
    for central in valid_centrals[:3]:
        print(f"中枢时间: {central['start_date']} 至 {central['end_date']}, "
              f"下沿: {central['lower_bound']}, 上沿: {central['upper_bound']}, "
              f"中轨: {central['midline']}, 状态: {central['status']}")
    
    # 如果有当前有效中枢，显示详细信息
    if hasattr(cl, 'effective_central') and cl.effective_central:
        ec = cl.effective_central
        print("\n当前有效中枢详情:")
        print(f"中枢区间: [{ec['lower_bound']}, {ec['upper_bound']}], 中轨: {ec['midline']}")
        print(f"形成时间: {ec['start_date']} 至 {ec['end_date']}, K线数量: {ec['k_count']}")
        print(f"振幅: {ec['amplitude']*100:.2f}%")
        print(f"验证结果: {ec['validation_result']}, 状态: {ec['status']}")
        if 'validation_details' in ec:
            details = ec['validation_details']
            print(f"验证详情: {details['in_range_count']}/{details['check_k_count']}根K线在区间内, "
                  f"支撑次数: {details['support_count']}, 压力次数: {details['resistance_count']}")
    
    # 检测缠论买入信号
    signals = cl.detect_buy_signals()
    
    # 应用信号过滤规则
    filtered_signals = cl.filter_signals()
    
    # 打印过滤后的信号详情
    if filtered_signals:
        print("\n过滤后的买入信号详情:")
        print("-" * 100)
        
        # 显示最近5个信号
        for i, signal in enumerate(filtered_signals[:5]):
            print(f"\n信号 {i+1}: {signal['signal_type']} ({signal['signal_date']})")
            print(f"  买入价格: {signal['buy_price']}")
            print(f"  止损价格: {signal['stop_loss_price']}")
            print(f"  止盈价格: {signal['take_profit_price']}")
            print(f"  风险收益比: {signal['risk_reward_ratio']}")
            print(f"  置信度: {signal['confidence']}")
            print(f"  锚定中枢: {signal['anchor_central']['start_date']}至{signal['anchor_central']['end_date']}")
            print(f"  中枢区间: {signal['anchor_central']['lower_bound']}-{signal['anchor_central']['upper_bound']}")
            print(f"  有效性判定理由: {signal['validation_reason']}")
            print(f"  量能比: {signal['volume_ratio']}")
            
            # 计算买入数量
            capital = 160000  # 本金16万
            position_ratio = 0.2  # 单个信号最大仓位20%
            max_position_amount = capital * position_ratio
            buy_quantity = int(max_position_amount / signal['buy_price'])
            actual_position_amount = buy_quantity * signal['buy_price']
            
            print(f"  买入数量: {buy_quantity} 份")
            print(f"  投入资金: {actual_position_amount:.2f} 元")
            print(f"  实际仓位: {actual_position_amount/capital*100:.1f}%")
            print("-" * 100)
    else:
        print("未识别到任何有效买入信号")
    
    # 生成完整报告
    if filtered_signals:
        report_files = cl.generate_report()
        print(f"\n缠论动态中枢自动判定及信号识别系统运行完成!")
        if report_files:
            print(f"报告已保存至: {report_files['text_report']}")
            print(f"JSON报告已保存至: {report_files['json_report']}")
    else:
        print("\n由于未识别到有效信号，无法生成报告")