#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修改后的破中枢反抽信号检测逻辑
特别测试2025年11月24-25日512660是否能正确识别跌破9月初中枢的反抽信号
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_reverse_pullback.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入修改后的检测器模块
try:
    from src.daily_buy_signal_detector import BuySignalDetector
    logger.info("成功导入BuySignalDetector模块")
except Exception as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)


class ReversePullbackValidator:
    """
    验证破中枢反抽信号检测逻辑的类
    """
    
    def __init__(self, data_file_path):
        """初始化验证器"""
        self.data_file_path = data_file_path
        self.df = None
        self.detector = None
        self.load_data()
        self.initialize_detector()
    
    def load_data(self):
        """加载历史数据"""
        try:
            logger.info(f"正在加载数据文件: {self.data_file_path}")
            
            # 读取CSV文件并获取实际列名
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                logger.info(f"数据文件列名: {first_line}")
            
            # 读取数据
            self.df = pd.read_csv(self.data_file_path)
            
            # 假设第一列是日期
            self.df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # 转换日期列
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # 设置日期为索引
            self.df.set_index('date', inplace=True)
            
            logger.info(f"数据加载成功，共有 {len(self.df)} 条记录")
            logger.info(f"数据日期范围: {self.df.index.min()} 至 {self.df.index.max()}")
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def initialize_detector(self):
        """初始化信号检测器"""
        try:
            # 创建检测器实例
            self.detector = BuySignalDetector()
            # 设置日志
            self.detector.logger = logger
            logger.info("信号检测器初始化成功")
        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
            raise
    
    def test_specific_date(self, test_date_str, lookback_days=120):
        """测试特定日期的反抽信号
        
        Args:
            test_date_str: 测试日期字符串，格式如 '2025-11-24'
            lookback_days: 回溯天数，默认为120天
        """
        try:
            test_date = pd.to_datetime(test_date_str)
            
            # 确保有足够的数据
            if test_date not in self.df.index:
                # 找到最接近的日期
                closest_date = self.df.index[self.df.index.get_indexer([test_date], method='nearest')[0]]
                logger.warning(f"精确日期 {test_date_str} 未找到，使用最接近的日期: {closest_date}")
                test_date = closest_date
            
            # 获取测试日期的位置
            date_idx = self.df.index.get_loc(test_date)
            
            # 获取足够的数据进行测试（需要包含历史中枢）
            start_idx = max(0, date_idx - lookback_days + 1)
            test_data = self.df.iloc[start_idx:date_idx + 1].copy()
            
            logger.info(f"\n=== 开始测试 {test_date.strftime('%Y-%m-%d')} 的反抽信号 ===")
            logger.info(f"测试数据范围: {test_data.index.min()} 至 {test_data.index.max()} (共{len(test_data)}条)")
            
            # 直接测试反抽信号检测
            is_reverse_pullback, details = self.detector.detect_daily_reverse_pullback(test_data, logger=logger)
            
            logger.info(f"\n=== 测试结果 ===")
            logger.info(f"反抽信号检测结果: {'满足' if is_reverse_pullback else '不满足'}")
            
            # 打印中枢信息
            if details and 'central_bank' in details:
                cb = details['central_bank']
                logger.info(f"\n中枢信息:")
                logger.info(f"  当前中枢上沿: {cb['top_main']:.6f}")
                logger.info(f"  当前中枢下沿: {cb['bottom_main']:.6f}")
                logger.info(f"  历史中枢上沿: {cb['historical_top']:.6f} (如有)")
                logger.info(f"  历史中枢下沿: {cb['historical_bottom']:.6f} (如有)")
            
            # 打印突破信息
            if details and 'breakdown' in details:
                bd = details['breakdown']
                logger.info(f"\n突破信息:")
                logger.info(f"  跌破当前中枢: {bd['has_below_central']}")
                logger.info(f"  跌破历史中枢: {bd['has_below_historical']}")
                logger.info(f"  跌破任一中枢: {bd['has_below_any_central']}")
            
            # 打印最近的价格数据
            recent_data = test_data.iloc[-5:]
            logger.info(f"\n最近5天价格数据:")
            for idx, row in recent_data.iterrows():
                logger.info(f"  {idx.strftime('%Y-%m-%d')}: 低={row['low']:.6f}, 收={row['close']:.6f}")
            
            # 分析9月初中枢和11月跌破情况
            self._analyze_september_november_relationship(test_data)
            
            return is_reverse_pullback, details
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            raise
    
    def _analyze_september_november_relationship(self, test_data):
        """分析9月初中枢和11月跌破关系"""
        try:
            # 提取9月初中枢（大约2025-09-01到2025-09-20）
            september_range = test_data[test_data.index.to_series().between('2025-09-01', '2025-09-20')]
            
            if not september_range.empty:
                sept_central_top = september_range['high'].max()
                sept_central_bottom = september_range['low'].min()
                logger.info(f"\n=== 9月初中枢分析 ===")
                logger.info(f"9月初中枢范围: 上沿={sept_central_top:.6f}, 下沿={sept_central_bottom:.6f}")
                
                # 提取11月价格数据
                november_range = test_data[test_data.index.to_series().between('2025-11-01', '2025-11-30')]
                
                if not november_range.empty:
                    # 检查11月是否跌破9月初中枢
                    below_sept_central = november_range[november_range['close'] < sept_central_bottom]
                    logger.info(f"\n11月跌破9月初中枢的天数: {len(below_sept_central)}")
                    
                    if not below_sept_central.empty:
                        logger.info(f"跌破9月初中枢的日期:")
                        for idx, row in below_sept_central.iterrows():
                            below_pct = ((row['close'] - sept_central_bottom) / sept_central_bottom) * 100
                            logger.info(f"  {idx.strftime('%Y-%m-%d')}: 收盘价={row['close']:.6f}, 跌破幅度={below_pct:.2f}%")
                    
                    # 检查11月最低点
                    nov_low = november_range['low'].min()
                    nov_low_date = november_range[november_range['low'] == nov_low].index[0]
                    below_pct = ((nov_low - sept_central_bottom) / sept_central_bottom) * 100
                    logger.info(f"\n11月最低点: {nov_low_date.strftime('%Y-%m-%d')}, 价格={nov_low:.6f}, 跌破9月初中枢幅度={below_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"9-11月关系分析失败: {e}")
    
    def batch_test(self):
        """批量测试多个关键日期"""
        test_dates = [
            '2025-11-21',  # 用户提到的跌破9月4日低点的日期
            '2025-11-24',  # 系统应该发出信号的日期
            '2025-11-25',  # 系统应该发出信号的日期
            '2025-11-28'   # 后续日期，验证信号持续性
        ]
        
        results = {}
        
        for date_str in test_dates:
            logger.info(f"\n\n=========================================")
            logger.info(f"测试日期: {date_str}")
            logger.info(f"=========================================")
            
            try:
                is_signal, details = self.test_specific_date(date_str)
                results[date_str] = {
                    'signal': is_signal,
                    'details': details
                }
            except Exception as e:
                logger.error(f"日期 {date_str} 测试失败: {e}")
                results[date_str] = {'error': str(e)}
        
        # 总结批量测试结果
        self._summarize_batch_results(results)
        
        return results
    
    def _summarize_batch_results(self, results):
        """总结批量测试结果"""
        logger.info(f"\n\n=========================================")
        logger.info(f"批量测试结果总结")
        logger.info(f"=========================================")
        
        signal_count = 0
        
        for date_str, result in results.items():
            if 'error' in result:
                status = f"失败: {result['error']}"
            else:
                status = "信号触发" if result['signal'] else "无信号"
                if result['signal']:
                    signal_count += 1
            
            logger.info(f"{date_str}: {status}")
        
        logger.info(f"\n触发信号的日期数量: {signal_count}/{len(results)}")
        
        # 如果11月24-25日触发了信号，记录成功
        if ('2025-11-24' in results and results['2025-11-24'].get('signal') and 
            '2025-11-25' in results and results['2025-11-25'].get('signal')):
            logger.info(f"\n✅ 修改成功！11月24-25日成功识别出破中枢反抽信号")
        else:
            logger.warning(f"\n❌ 修改未达成目标！11月24-25日未能识别出破中枢反抽信号")


def main():
    """主函数"""
    try:
        # 数据文件路径
        data_file = '/Users/pingan/tools/trade/tianyuan/data/512660_daily_data.csv'
        
        # 检查文件是否存在
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在: {data_file}")
            sys.exit(1)
        
        logger.info("开始测试修改后的破中枢反抽信号检测逻辑...")
        
        # 创建验证器
        validator = ReversePullbackValidator(data_file)
        
        # 执行批量测试
        validator.batch_test()
        
        logger.info("\n测试完成！详细结果请查看日志文件")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()