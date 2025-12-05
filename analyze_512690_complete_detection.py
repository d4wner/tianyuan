#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512690完整信号检测分析脚本

使用BuySignalDetector对512690进行完整分析，专注于检测反抽信号

作者: TradeTianYuan
日期: 2025-11-28
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入信号检测器
from daily_buy_signal_detector import BuySignalDetector

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("512690_complete_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('512690CompleteDetector')

class ETFSignalDetector:
    """
    ETF信号检测器 - 专注于反抽信号检测
    """
    
    def __init__(self, symbol: str = '512690'):
        """
        初始化检测器
        
        Args:
            symbol: 股票代码
        """
        self.symbol = symbol
        self.data_dir = './data/daily'
        self.daily_data = None
        self.detector = None
        self.signals = []
        self.recent_signals = []
        self.results = {
            'total_signals': 0,
            'reverse_pullback_signals': 0,
            'recent_signals': [],
            'has_recent_reverse_pullback': False
        }
    
    def load_data(self) -> bool:
        """
        加载日线数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            data_file = os.path.join(self.data_dir, f'{self.symbol}_daily.csv')
            if not os.path.exists(data_file):
                logger.error(f"数据文件不存在: {data_file}")
                return False
            
            self.daily_data = pd.read_csv(data_file)
            self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
            self.daily_data.sort_values('date', inplace=True)
            
            logger.info(f"成功加载{self.symbol}日线数据，共{len(self.daily_data)}条记录，时间范围: {self.daily_data['date'].min()} 到 {self.daily_data['date'].max()}")
            return True
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False
    
    def initialize_detector(self) -> bool:
        """
        初始化信号检测器
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            # 初始化信号检测器
            self.detector = BuySignalDetector()
            
            # 直接设置数据到检测器
            self.detector.df = self.daily_data.copy()
            
            # 检查检测器是否有calculate_indicators方法，如果没有则手动计算必要指标
            if hasattr(self.detector, 'calculate_indicators'):
                self.detector.calculate_indicators()
            else:
                # 手动计算MACD
                if hasattr(self.detector, '_calculate_macd'):
                    self.detector.df = self.detector._calculate_macd(self.detector.df)
                # 手动识别分型
                if hasattr(self.detector, '_identify_fractals'):
                    self.detector.df = self.detector._identify_fractals(self.detector.df)
            
            logger.info("信号检测器初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化检测器失败: {str(e)}")
            return False
    
    def detect_signals(self) -> List[Dict]:
        """
        检测所有买入信号，特别关注反抽信号
        
        Returns:
            List[Dict]: 信号列表
        """
        if self.detector is None or self.daily_data is None:
            logger.error("请先加载数据并初始化检测器")
            return []
        
        signals = []
        reverse_pullback_signals = 0
        
        # 对最近3个月的数据进行检测
        recent_3_months = datetime.now() - timedelta(days=90)
        recent_data = self.daily_data[self.daily_data['date'] >= recent_3_months].copy()
        
        logger.info(f"开始检测信号，分析最近3个月数据，共{len(recent_data)}个交易日")
        
        # 遍历最近的数据，跳过前120天以确保有足够的历史数据
        for i in range(120, len(self.detector.df)):
            try:
                # 获取日期并确保类型正确
                date_series = self.detector.df.iloc[i]['date']
                # 处理不同格式的日期
                if isinstance(date_series, pd.Timestamp):
                    date = date_series
                elif hasattr(date_series, 'strftime'):
                    date = date_series
                else:
                    # 尝试转换为datetime
                    date = pd.to_datetime(date_series)
                
                # 直接调用detect_daily_reverse_pullback方法检测反抽信号
                if hasattr(self.detector, 'detect_daily_reverse_pullback'):
                    try:
                        # 确保索引是整数
                        int_idx = int(i)
                        reverse_result = self.detector.detect_daily_reverse_pullback(int_idx)
                        
                        if isinstance(reverse_result, dict) and 'signal' in reverse_result and reverse_result['signal']:
                            reverse_pullback_signals += 1
                            # 获取收盘价并转换为浮点数
                            close_price = float(self.detector.df.iloc[i]['close'])
                            signal_record = {
                                'date': date.strftime('%Y-%m-%d'),
                                'price': close_price,
                                'signal_type': 'reverse_pullback',
                                'details': reverse_result
                            }
                            signals.append(signal_record)
                            logger.info(f"检测到反抽信号: {date.strftime('%Y-%m-%d')} - 价格: {close_price}")
                            
                            # 检查是否在最近30天内
                            if date >= (datetime.now() - timedelta(days=30)):
                                self.recent_signals.append(signal_record)
                    except Exception as rp_e:
                        logger.warning(f"检测第{i}天反抽信号时出错: {str(rp_e)}")
                
                # 同时也尝试使用detect_buy_signals进行全面检测
                if hasattr(self.detector, 'detect_buy_signals'):
                    try:
                        result = self.detector.detect_buy_signals(self.detector.df.iloc[:i+1])
                        # 检查是否有反抽信号
                        if result and isinstance(result, dict) and 'signals' in result and 'reverse_pullback' in result['signals']:
                            rp_signal = result['signals']['reverse_pullback']
                            # 处理不同的返回格式
                            if isinstance(rp_signal, bool) and rp_signal:
                                logger.info(f"通过detect_buy_signals检测到反抽信号: {date.strftime('%Y-%m-%d')}")
                            elif isinstance(rp_signal, dict) and 'result' in rp_signal and rp_signal['result']:
                                logger.info(f"通过detect_buy_signals检测到反抽信号: {date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        logger.warning(f"调用detect_buy_signals时出错，跳过: {str(e)}")
                    
            except Exception as e:
                logger.error(f"检测第{i}天信号时出错: {str(e)}")
                # 不记录详细堆栈以保持日志简洁
        
        # 更新结果统计
        self.results['total_signals'] = len(signals)
        self.results['reverse_pullback_signals'] = reverse_pullback_signals
        self.results['recent_signals'] = self.recent_signals
        self.results['has_recent_reverse_pullback'] = len(self.recent_signals) > 0
        
        logger.info(f"信号检测完成，共检测到{len(signals)}个买入信号")
        logger.info(f"其中反抽信号: {reverse_pullback_signals}个")
        logger.info(f"最近30天的信号: {len(self.recent_signals)}个")
        
        return signals
    
    def analyze_specific_dates(self, dates: List[str]) -> Dict:
        """
        分析特定日期的信号情况，专注于反抽信号
        
        Args:
            dates: 要分析的日期列表
            
        Returns:
            Dict: 特定日期的分析结果
        """
        specific_analysis = {}
        
        for target_date in dates:
            try:
                # 查找对应的索引
                # 将目标日期转换为datetime对象以匹配数据格式
                target_dt = pd.to_datetime(target_date)
                date_idx = self.daily_data[self.daily_data['date'] == target_dt].index
                
                # 检查是否找到日期
                if len(date_idx) == 0:
                    specific_analysis[target_date] = {'error': '日期不存在'}
                    continue
                
                # 确保idx是整数
                idx = int(date_idx[0])
                
                # 分析该日期的信号
                date_analysis = {
                    'price': float(self.detector.df.iloc[idx]['close']),
                    'volume': float(self.detector.df.iloc[idx]['volume']),
                    'reverse_pullback': False,
                    'reverse_pullback_details': None
                }
                
                # 检查反抽信号
                if hasattr(self.detector, 'detect_daily_reverse_pullback'):
                    # 使用try-except包装反抽信号检测
                    try:
                        reverse_result = self.detector.detect_daily_reverse_pullback(idx)
                        if isinstance(reverse_result, dict) and 'signal' in reverse_result and reverse_result['signal']:
                            date_analysis['reverse_pullback'] = True
                            date_analysis['reverse_pullback_details'] = reverse_result
                    except Exception as rp_e:
                        logger.warning(f"检测日期{target_date}的反抽信号时出错: {str(rp_e)}")
                
                specific_analysis[target_date] = date_analysis
                
            except Exception as e:
                error_msg = str(e)
                specific_analysis[target_date] = {'error': error_msg}
                logger.error(f"分析日期{target_date}时出错: {error_msg}")
        
        return specific_analysis
    
    def run_complete_analysis(self) -> Dict:
        """
        运行完整的分析流程
        
        Returns:
            Dict: 分析结果
        """
        logger.info(f"开始对{self.symbol}进行完整检测分析，专注于反抽信号")
        
        # 加载数据
        if not self.load_data():
            return {'error': '数据加载失败'}
        
        # 初始化检测器
        if not self.initialize_detector():
            return {'error': '检测器初始化失败'}
        
        # 检测所有信号
        self.signals = self.detect_signals()
        
        # 分析最近的关键日期
        recent_key_dates = ['2025-11-21', '2025-11-24', '2025-11-25', '2025-11-26', '2025-11-27']
        self.results['recent_key_dates_analysis'] = self.analyze_specific_dates(recent_key_dates)
        
        # 生成综合报告
        self.generate_report()
        
        logger.info("完整检测分析完成")
        return self.results
    
    def generate_report(self):
        """
        生成分析报告，专注于反抽信号
        """
        logger.info("=== 512690 完整检测分析报告 ===")
        logger.info(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总信号数量: {self.results['total_signals']}")
        logger.info(f"反抽信号数量: {self.results['reverse_pullback_signals']}")
        logger.info(f"最近30天是否存在反抽信号: {'是' if self.results['has_recent_reverse_pullback'] else '否'}")
        
        # 最近30天的反抽信号
        if self.results['recent_signals']:
            logger.info("\n=== 最近30天的反抽信号 ===")
            for signal in self.results['recent_signals']:
                logger.info(f"日期: {signal['date']}, 价格: {signal['price']:.6f}, 信号类型: {signal['signal_type']}")
        
        # 最近关键日期分析
        if self.results.get('recent_key_dates_analysis'):
            logger.info("\n=== 最近关键日期分析 ===")
            for date, analysis in self.results['recent_key_dates_analysis'].items():
                if 'error' not in analysis:
                    logger.info(f"\n日期: {date}")
                    logger.info(f"  收盘价: {analysis['price']:.6f}")
                    logger.info(f"  成交量: {analysis['volume']:,.0f}")
                    logger.info(f"  反抽信号: {'是' if analysis['reverse_pullback'] else '否'}")
                    
                    # 反抽信号详情
                    if analysis.get('reverse_pullback_details'):
                        details = analysis['reverse_pullback_details']
                        logger.info(f"  反抽信号详情: {details}")
        
        # 总结
        logger.info("\n=== 总结 ===")
        if self.results['has_recent_reverse_pullback']:
            logger.info(f"512690在最近30天内检测到{len(self.results['recent_signals'])}个反抽信号，建议关注！")
        else:
            logger.info("512690在最近30天内未检测到反抽信号，暂不建议操作。")

if __name__ == "__main__":
    # 创建检测器实例
    detector = ETFSignalDetector(symbol='512690')
    
    # 运行完整分析
    try:
        results = detector.run_complete_analysis()
        logger.info("\n分析完成，结果已保存到512690_complete_detection.log")
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)