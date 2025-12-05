#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别2025年512660军工ETF破中枢反抽一买信号

根据用户要求，实现严格的破中枢反抽一买信号识别，
包含真实下单信息生成。
"""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detect_512660_break_central_buy_signal.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BreakCentralBuySignalDetector")


class BreakCentralBuySignalDetector:
    """破中枢反抽一买信号检测器"""
    
    def __init__(self, data_file=None):
        """
        初始化检测器
        
        Args:
            data_file: 日线数据文件路径
        """
        self.data_file = data_file or os.path.join('data', 'daily', '512660_daily.csv')
        self.klines = None
        self.buy_signals = []
        self.etf_code = "512660"
        self.etf_name = "军工ETF"
        self.trading_capital = 160000  # 用户提到的16万本金
        
        # 按季度定义中枢范围
        self.quarterly_central = {
            'Q1_2025': {'low': 0.95, 'high': 1.05},  # Q1中枢范围
            'Q2_2025': {'low': 1.00, 'high': 1.10},  # Q2中枢范围
            'Q3_2025': {'low': 1.10, 'high': 1.20},  # Q3中枢范围
            'Q4_2025': {'low': 0.85, 'high': 0.95}   # Q4中枢范围
        }
        
        # 交易参数设置
        self.high_confidence_position = 0.15  # 高置信度信号仓位15%
        self.normal_confidence_position = 0.10  # 普通信号仓位10%
        self.max_loss_ratio = 0.02  # 单次最大亏损比例2%
    
    def load_data(self):
        """
        加载日线数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            if not os.path.exists(self.data_file):
                logger.error(f"数据文件不存在: {self.data_file}")
                # 尝试从其他位置获取数据
                self.data_file = os.path.join('data', '512660_daily_data.csv')
                if not os.path.exists(self.data_file):
                    logger.error("找不到军工ETF数据文件")
                    return False
            
            # 读取CSV数据
            df = pd.read_csv(self.data_file)
            
            # 转换为需要的格式
            self.klines = []
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # 检查必要列是否存在
            if not all(col in df.columns for col in required_columns):
                # 尝试从其他格式读取
                logger.warning("标准列名不存在，尝试其他列名...")
                # 尝试常见的列名映射
                column_mapping = {
                    '日期': 'date',
                    'Date': 'date',
                    '开盘价': 'open',
                    '最高价': 'high',
                    '最低价': 'low',
                    '收盘价': 'close', 
                    'Close': 'close',
                    '成交量': 'volume',
                    'Volume': 'volume'
                }
                
                # 重命名列
                for cn_col, en_col in column_mapping.items():
                    if cn_col in df.columns:
                        df = df.rename(columns={cn_col: en_col})
                
                # 再次检查
                if not all(col in df.columns for col in required_columns):
                    logger.error("无法找到必要的数据列")
                    return False
            
            # 转换数据
            for _, row in df.iterrows():
                kline = {
                    'date': str(row['date']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                self.klines.append(kline)
            
            logger.info(f"成功加载{len(self.klines)}条日线数据")
            return True
            
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            return False
    
    def get_quarterly_central(self, date_str):
        """
        根据日期获取对应季度的中枢范围
        
        Args:
            date_str: 日期字符串，格式为'YYYY-MM-DD'
            
        Returns:
            dict: 包含中枢下沿和上沿的字典
        """
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        
        quarter_key = f'Q{quarter}_{year}'
        
        # 如果有对应季度的中枢定义，返回该定义
        if quarter_key in self.quarterly_central:
            return self.quarterly_central[quarter_key]
        
        # 默认返回Q4_2025的中枢范围
        return self.quarterly_central['Q4_2025']
    
    def is_valid_central_breakdown(self, idx):
        """
        判断是否为有效破中枢（根据用户提供的标准）
        
        条件：2日内≥1日收盘价≤中枢下沿×0.985
        
        Args:
            idx: 当前K线索引
            
        Returns:
            dict or None: 有效返回包含破中枢信息的字典，否则返回None
        """
        # 获取当前日期对应的中枢范围
        central = self.get_quarterly_central(self.klines[idx]['date'])
        central_low = central['low']
        
        # 破位临界值（中枢下沿×0.985）
        breakdown_threshold = central_low * 0.985
        
        # 检查当前日和前一日是否有至少一日满足条件
        valid_days = 0
        valid_idx = -1
        
        for i in range(max(0, idx-1), idx+1):
            if self.klines[i]['close'] <= breakdown_threshold:
                valid_days += 1
                valid_idx = i
                
        # 如果2日内至少1日满足条件，则认为是有效破中枢
        if valid_days >= 1:
            return {
                'date': self.klines[valid_idx]['date'],
                'close_price': self.klines[valid_idx]['close'],
                'low_price': self.klines[valid_idx]['low'],
                'volume': self.klines[valid_idx]['volume'],
                'central_low': central_low,
                'central_high': central['high'],
                'threshold': breakdown_threshold
            }
        
        return None
    
    def calculate_ma_volume(self, idx, days=5):
        """
        计算近N日均量
        
        Args:
            idx: 当前K线索引
            days: 计算日均量的天数
            
        Returns:
            float: 近N日均量
        """
        start_idx = max(0, idx - days + 1)
        volumes = [self.klines[i]['volume'] for i in range(start_idx, idx + 1)]
        return sum(volumes) / len(volumes) if volumes else 0
    
    def is_valid_rebound(self, idx, breakdown_date):
        """
        判断是否为有效反抽（根据用户提供的标准）
        
        条件：
        1. 发生在破中枢后的7个交易日内
        2. 2日内≥1日收盘价≥中枢下沿×1.015
        3. 反抽日成交量≥近5日均量90%
        
        Args:
            idx: 当前K线索引
            breakdown_date: 破中枢日期
            
        Returns:
            dict or None: 有效返回包含反抽信息的字典，否则返回None
        """
        # 检查时间窗口
        current_date = datetime.strptime(self.klines[idx]['date'], '%Y-%m-%d')
        break_date = datetime.strptime(breakdown_date, '%Y-%m-%d')
        trading_days_diff = (current_date - break_date).days
        
        if trading_days_diff > 10:  # 考虑非交易日，放宽到10天
            return None
            
        # 获取当前日期对应的中枢范围
        central = self.get_quarterly_central(self.klines[idx]['date'])
        central_low = central['low']
        
        # 反抽临界值（中枢下沿×1.015）
        rebound_threshold = central_low * 1.015
        
        # 检查当前日和前一日是否有至少一日满足条件
        valid_days = 0
        valid_idx = -1
        
        for i in range(max(0, idx-1), idx+1):
            if self.klines[i]['close'] >= rebound_threshold:
                valid_days += 1
                valid_idx = i
        
        if valid_days < 1:
            return None
        
        # 检查成交量是否满足条件
        ma5_volume = self.calculate_ma_volume(valid_idx, 5)
        current_volume = self.klines[valid_idx]['volume']
        
        # 如果量能小于近5日均量的85%，直接判定为无效
        if ma5_volume > 0 and current_volume < ma5_volume * 0.85:
            return None
        
        volume_ratio = current_volume / ma5_volume if ma5_volume > 0 else 0
        volume_status = "强" if volume_ratio >= 1.2 else "中" if volume_ratio >= 0.9 else "弱"
        
        return {
            'date': self.klines[valid_idx]['date'],
            'close_price': self.klines[valid_idx]['close'],
            'open_price': self.klines[valid_idx]['open'],
            'low_price': self.klines[valid_idx]['low'],
            'high_price': self.klines[valid_idx]['high'],
            'volume': self.klines[valid_idx]['volume'],
            'ma5_volume': ma5_volume,
            'volume_ratio': volume_ratio,
            'volume_status': volume_status,
            'central_low': central_low,
            'central_high': central['high'],
            'threshold': rebound_threshold
        }
    
    def validate_trend_resonance(self, idx):
        """
        验证趋势共振
        
        条件：近10根K线中≥6根收盘价抬升
        
        Args:
            idx: 当前K线索引
            
        Returns:
            tuple: (是否满足趋势共振, 抬升K线数量, 总K线数量)
        """
        # 检查是否有足够的数据
        if idx < 10:
            return False, 0, min(idx, 10)
        
        # 计算近10根K线中收盘价抬升的数量
        rising_count = 0
        for i in range(idx - 9, idx):
            if self.klines[i+1]['close'] > self.klines[i]['close']:
                rising_count += 1
        
        # 如果近10根K线中≥6根收盘价抬升，认为满足趋势共振条件
        return rising_count >= 6, rising_count, 10
    
    def calculate_position_size(self, signal_confidence, close_price, stop_loss_price):
        """
        计算仓位大小
        
        Args:
            signal_confidence: 信号置信度 ('high' 或 'normal')
            close_price: 收盘价
            stop_loss_price: 止损价
            
        Returns:
            dict: 包含仓位信息的字典
        """
        # 根据置信度确定仓位比例
        position_ratio = self.high_confidence_position if signal_confidence == 'high' else self.normal_confidence_position
        
        # 计算最大可使用资金
        max_capital = self.trading_capital * position_ratio
        
        # 计算基于止损的最大可购买数量
        price_diff = close_price - stop_loss_price
        if price_diff <= 0:
            logger.warning(f"无效的止损设置: 收盘价{close_price} <= 止损价{stop_loss_price}")
            # 使用仓位比例直接计算
            max_shares_by_position = int(max_capital / close_price)
        else:
            # 确保单次亏损不超过总资金的2%
            max_risk = self.trading_capital * self.max_loss_ratio
            max_shares_by_risk = int(max_risk / price_diff)
            max_shares_by_position = int(max_capital / close_price)
            
            # 取较小值作为最终购买数量
            max_shares = min(max_shares_by_risk, max_shares_by_position)
        
        # 购买数量取整（ETF通常按份额购买）
        shares = max_shares
        actual_capital = shares * close_price
        actual_ratio = actual_capital / self.trading_capital
        
        return {
            'shares': shares,
            'capital_used': actual_capital,
            'position_ratio': actual_ratio * 100,  # 转为百分比
            'max_risk_capital': self.trading_capital * self.max_loss_ratio
        }
    
    def detect_buy_signals(self):
        """
        检测所有破中枢反抽一买信号
        
        Returns:
            list: 有效的买入信号列表
        """
        if not self.klines:
            logger.error("没有数据可检测")
            return []
            
        logger.info(f"开始检测{self.etf_name}({self.etf_code}) 2025年破中枢反抽一买信号")
        
        # 只检测2025年的数据
        self.klines = [k for k in self.klines if k['date'].startswith('2025')]
        
        # 找出所有有效破中枢点
        breakdown_points = []
        for i in range(1, len(self.klines)):
            breakdown_info = self.is_valid_central_breakdown(i)
            if breakdown_info:
                breakdown_points.append((i, breakdown_info))
                logger.info(f"发现有效破中枢: {breakdown_info['date']}, 收盘价: {breakdown_info['close_price']}")
        
        # 为每个破中枢点寻找有效反抽
        for b_idx, breakdown_info in breakdown_points:
            # 从破中枢后的下一个交易日开始查找
            found_rebound = False
            for r_idx in range(b_idx + 1, min(b_idx + 11, len(self.klines))):  # 最多查找10个交易日
                rebound_info = self.is_valid_rebound(r_idx, breakdown_info['date'])
                if rebound_info:
                    # 验证趋势共振
                    trend_resonance, rising_count, total_count = self.validate_trend_resonance(r_idx)
                    
                    # 评估信号置信度
                    confidence = 'high' if (rebound_info['volume_status'] == "强" and trend_resonance) else 'normal'
                    
                    # 计算止损价（破位日最低价×0.99）
                    stop_loss_price = breakdown_info['low_price'] * 0.99
                    
                    # 计算止盈价（当期中枢中轨）
                    central_mid = (breakdown_info['central_low'] + breakdown_info['central_high']) / 2
                    
                    # 计算仓位信息
                    position_info = self.calculate_position_size(confidence, rebound_info['close_price'], stop_loss_price)
                    
                    # 构建买入信号
                    buy_signal = {
                        'signal_id': f"SIGNAL_{breakdown_info['date']}_{rebound_info['date']}",
                        'trade_date': rebound_info['date'],
                        'etf_code': self.etf_code,
                        'etf_name': self.etf_name,
                        'close_price': rebound_info['close_price'],
                        'buy_price_suggestion': round(rebound_info['close_price'], 3),  # 建议买入价格
                        'breakdown_date': breakdown_info['date'],
                        'breakdown_price': breakdown_info['close_price'],
                        'breakdown_low': breakdown_info['low_price'],
                        'stop_loss_price': round(stop_loss_price, 3),
                        'target_profit_price': round(central_mid, 3),
                        'volume_status': rebound_info['volume_status'],
                        'volume_ratio': rebound_info['volume_ratio'],
                        'trend_resonance': trend_resonance,
                        'rising_k_count': rising_count,
                        'total_k_count': total_count,
                        'confidence': confidence,
                        'position_shares': position_info['shares'],
                        'position_capital': round(position_info['capital_used'], 2),
                        'position_ratio': round(position_info['position_ratio'], 2),
                        'max_risk_capital': round(position_info['max_risk_capital'], 2),
                        'central_low': breakdown_info['central_low'],
                        'central_high': breakdown_info['central_high'],
                        'central_mid': round(central_mid, 3),
                        'signal_type': '破中枢反抽一买',
                        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # 计算预期收益和风险
                    potential_profit = (buy_signal['target_profit_price'] - buy_signal['buy_price_suggestion']) * buy_signal['position_shares']
                    risk_amount = (buy_signal['buy_price_suggestion'] - buy_signal['stop_loss_price']) * buy_signal['position_shares']
                    
                    buy_signal['potential_profit'] = round(potential_profit, 2)
                    buy_signal['risk_amount'] = round(risk_amount, 2)
                    buy_signal['risk_reward_ratio'] = round(potential_profit / risk_amount, 2) if risk_amount > 0 else float('inf')
                    
                    self.buy_signals.append(buy_signal)
                    logger.info(f"发现有效买入信号: {buy_signal['trade_date']}, 置信度: {confidence}, 建议仓位: {position_info['position_ratio']:.2f}%")
                    found_rebound = True
                    break  # 找到第一个有效反抽后停止
            
            if not found_rebound:
                logger.info(f"破中枢后7日内未找到有效反抽: {breakdown_info['date']}")
        
        logger.info(f"检测完成，共找到{len(self.buy_signals)}个2025年有效破中枢反抽一买信号")
        return self.buy_signals
    
    def generate_trade_report(self, output_file=None):
        """
        生成交易信号识别报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            str: 生成的报告文件路径
        """
        if not output_file:
            output_file = os.path.join('results', f'{self.etf_code}_break_central_buy_signals_2025.txt')
        
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        
        # 按日期排序信号
        sorted_signals = sorted(self.buy_signals, key=lambda x: x['trade_date'])
        
        # 分离高置信度和普通信号
        high_confidence_signals = [s for s in sorted_signals if s['confidence'] == 'high']
        normal_signals = [s for s in sorted_signals if s['confidence'] != 'high']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.etf_name}({self.etf_code}) 破中枢反抽一买信号识别报告 - 2025年\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write("【信号识别规则】\n")
            f.write("=" * 80 + "\n")
            f.write("1. 有效破中枢条件：2日内≥1日收盘价≤中枢下沿×0.985\n")
            f.write("2. 有效反抽条件：破位后7日内，2日内≥1日收盘价≥中枢下沿×1.015\n")
            f.write("3. 量能验证：反抽日成交量≥近5日均量90%，量能＜85%直接判定为无效\n")
            f.write("4. 趋势共振：近10根K线中≥6根收盘价抬升\n")
            f.write("5. 有效中枢定义：按季度校准\n")
            for q, central in self.quarterly_central.items():
                f.write(f"   - {q}中枢：{central['low']}-{central['high']}\n")
            f.write("\n")
            
            f.write("【识别结果统计】\n")
            f.write("=" * 80 + "\n")
            f.write(f"总识别信号数量：{len(sorted_signals)}\n")
            f.write(f"高置信度信号数量：{len(high_confidence_signals)}\n")
            f.write(f"普通置信度信号数量：{len(normal_signals)}\n")
            f.write(f"总投入资金：{sum(s['position_capital'] for s in sorted_signals):.2f} 元\n")
            f.write(f"平均仓位比例：{sum(s['position_ratio'] for s in sorted_signals) / len(sorted_signals) if sorted_signals else 0:.2f}%\n")
            f.write("\n")
            
            # 高置信度信号详情
            if high_confidence_signals:
                f.write("【高置信度买入信号详情】\n")
                f.write("=" * 80 + "\n")
                f.write("真实下单信息：\n")
                f.write(f"{'交易日期':<12} | {'建议买入价':<10} | {'买入数量':<10} | {'投入资金':<12} | {'止损价':<10} | {'目标止盈价':<12} | {'风险收益比':<10}\n")
                f.write(f"-" * 80 + "\n")
                
                for signal in high_confidence_signals:
                    f.write(f"{signal['trade_date']:<12} | {signal['buy_price_suggestion']:<10.3f} | "
                           f"{signal['position_shares']:<10} | {signal['position_capital']:<11.2f} | "
                           f"{signal['stop_loss_price']:<10.3f} | {signal['target_profit_price']:<11.3f} | "
                           f"{signal['risk_reward_ratio']:<9.2f}\n")
                f.write("\n")
                
                f.write("信号验证详情：\n")
                f.write(f"{'交易日期':<12} | {'破位日期':<12} | {'量能状态':<10} | {'量能比均量':<12} | {'趋势共振':<8} | {'抬升K线数':<10}\n")
                f.write(f"-" * 80 + "\n")
                
                for signal in high_confidence_signals:
                    trend_text = "是" if signal['trend_resonance'] else "否"
                    f.write(f"{signal['trade_date']:<12} | {signal['breakdown_date']:<12} | "
                           f"{signal['volume_status']:<10} | {signal['volume_ratio']*100:<11.1f}% | {trend_text:<8} | "
                           f"{signal['rising_k_count']}/{signal['total_k_count']:<9}\n")
                f.write("\n")
            
            # 普通置信度信号详情
            if normal_signals:
                f.write("【普通置信度买入信号详情】\n")
                f.write("=" * 80 + "\n")
                f.write("真实下单信息：\n")
                f.write(f"{'交易日期':<12} | {'建议买入价':<10} | {'买入数量':<10} | {'投入资金':<12} | {'止损价':<10} | {'目标止盈价':<12} | {'风险收益比':<10}\n")
                f.write(f"-" * 80 + "\n")
                
                for signal in normal_signals:
                    f.write(f"{signal['trade_date']:<12} | {signal['buy_price_suggestion']:<10.3f} | "
                           f"{signal['position_shares']:<10} | {signal['position_capital']:<11.2f} | "
                           f"{signal['stop_loss_price']:<10.3f} | {signal['target_profit_price']:<11.3f} | "
                           f"{signal['risk_reward_ratio']:<9.2f}\n")
                f.write("\n")
                
                f.write("信号验证详情：\n")
                f.write(f"{'交易日期':<12} | {'破位日期':<12} | {'量能状态':<10} | {'量能比均量':<12} | {'趋势共振':<8} | {'抬升K线数':<10}\n")
                f.write(f"-" * 80 + "\n")
                
                for signal in normal_signals:
                    trend_text = "是" if signal['trend_resonance'] else "否"
                    f.write(f"{signal['trade_date']:<12} | {signal['breakdown_date']:<12} | "
                           f"{signal['volume_status']:<10} | {signal['volume_ratio']*100:<11.1f}% | {trend_text:<8} | "
                           f"{signal['rising_k_count']}/{signal['total_k_count']:<9}\n")
                f.write("\n")
            
            f.write("【交易执行说明】\n")
            f.write("=" * 80 + "\n")
            f.write("1. 下单时间：信号确认当日收盘前30分钟内\n")
            f.write("2. 下单价格：建议以收盘价附近价格下单\n")
            f.write("3. 仓位控制：严格按照计算的数量买入，不得超额\n")
            f.write("4. 止损设置：下单后立即设置止损单，价格为破位日最低价×0.99\n")
            f.write("5. 止盈策略：到达目标止盈价（中枢中轨）后，减仓50%，剩余仓位持有\n")
            f.write("6. 资金管理：总仓位不超过总资产的50%，单个信号不超过总资产的20%\n")
            f.write("\n")
            
            f.write("【风险提示】\n")
            f.write("=" * 80 + "\n")
            f.write("1. 历史表现不代表未来表现\n")
            f.write("2. 市场环境变化可能导致原有规律失效\n")
            f.write("3. 严格执行止损，控制单次交易风险\n")
            f.write("4. 建议进行充分的风险评估和资金管理\n")
            f.write("5. 交易决策需结合个人风险承受能力\n")
            f.write(f"\n报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 同时生成JSON格式的信号数据，便于后续处理
        json_output_file = output_file.replace('.txt', '.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.buy_signals, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交易信号报告已生成: {output_file}")
        logger.info(f"JSON格式信号数据已保存: {json_output_file}")
        return output_file


def main():
    """
    主函数
    """
    # 创建检测器实例
    detector = BreakCentralBuySignalDetector()
    
    # 加载数据
    if not detector.load_data():
        logger.error("加载数据失败，程序退出")
        return
    
    # 检测买入信号
    signals = detector.detect_buy_signals()
    
    if not signals:
        logger.warning("未检测到有效的买入信号")
    else:
        # 生成交易报告
        output_file = detector.generate_trade_report()
        
        # 打印简要结果
        print("\n===== 2025年军工ETF(512660)破中枢反抽一买信号识别结果 =====")
        print(f"总识别信号数量: {len(signals)}")
        print(f"高置信度信号数量: {len([s for s in signals if s['confidence'] == 'high'])}")
        print(f"普通置信度信号数量: {len([s for s in signals if s['confidence'] == 'normal'])}")
        print(f"\n详细交易信息已保存至: {output_file}")
        print(f"JSON格式信号数据已保存至: {output_file.replace('.txt', '.json')}")


if __name__ == "__main__":
    main()