#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""缠论日线级别买点检测器测试脚本"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from src.chanlun_daily_detector import ChanlunDailyDetector
from analyze_signal_statistics import SignalStatisticsAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DetectorTest')


class ChanlunDetectorTester:
    """缠论检测器测试类"""
    
    def __init__(self):
        """初始化测试器"""
        self.detector = ChanlunDailyDetector()
    
    def generate_mock_daily_data(self, days: int = 60, pattern: str = 'normal') -> pd.DataFrame:
        """生成模拟的日线数据
        
        Args:
            days: 生成的天数
            pattern: 价格模式 ('normal', 'down_trend', 'uptrend', 'sideways', 'innovation_low_backtest')
            
        Returns:
            模拟的日线数据
        """
        dates = [datetime.now() - timedelta(days=i) for i in range(days-1, -1, -1)]
        
        # 基础价格
        base_price = 100.0
        close_prices = []
        
        # 根据不同模式生成价格序列
        if pattern == 'down_trend':
            # 下降趋势
            close_prices = [base_price * (1 - 0.002 * i) for i in range(days)]
        elif pattern == 'uptrend':
            # 上升趋势
            close_prices = [base_price * (1 + 0.002 * i) for i in range(days)]
        elif pattern == 'sideways':
            # 横盘震荡
            close_prices = [base_price + np.random.normal(0, 1) for _ in range(days)]
        elif pattern == 'innovation_low_backtest':
            # 创新低破中枢回抽模式
            # 1. 形成中枢
            # 2. 跌破中枢下沿并创新低
            # 3. 出现背驰
            # 4. 底分型确认回抽
            
            # 重新初始化close_prices，确保长度与days一致
            close_prices = [0.0] * days
            
            # 中枢形成阶段（前30天）
            for i in range(min(30, days)):
                if i < 10:
                    # 第一笔上升
                    close_prices[i] = base_price * (1 + 0.003 * i)
                elif i < 20:
                    # 第二笔下降
                    close_prices[i] = close_prices[9] * (1 - 0.002 * (i-9))
                else:
                    # 第三笔上升
                    close_prices[i] = close_prices[19] * (1 + 0.0015 * (i-19))
            
            if days > 30:
                # 中枢上沿和下沿
                central_high = max(close_prices[10:min(20, days)])
                central_low = min(close_prices[10:min(20, days)])
                
                # 跌破中枢并创新低阶段
                break_point = min(30, days)
                innov_low_end = min(45, days)
                for i in range(break_point, innov_low_end):
                    # 跌破中枢下沿并创新低
                    close_prices[i] = central_low * (1 - 0.003 * (i-break_point))
                
                # 底分型形成和确认阶段
                if days > 45:
                    # 确保在days范围内设置底分型
                    fractal_start = 45
                    fractal_end = min(50, days)
                    
                    if fractal_end - fractal_start >= 5:
                        # 故意制造一个标准的底分型
                        close_prices[fractal_start] = close_prices[fractal_start-1] - 0.5  # K1
                        close_prices[fractal_start+1] = close_prices[fractal_start] - 0.3   # K2（更低）
                        close_prices[fractal_start+2] = close_prices[fractal_start+1] - 0.2  # K3（更低）
                        close_prices[fractal_start+3] = close_prices[fractal_start+2] + 0.3  # K4
                        close_prices[fractal_start+4] = close_prices[fractal_start+3] + 0.5  # K5（确认日）
                    
                    # 填充剩余天数
                    for i in range(50, days):
                        close_prices[i] = close_prices[i-1] + np.random.normal(0, 0.2)
        else:
            # 普通模式 - 随机波动
            close_prices = [base_price]
            for _ in range(1, days):
                change = np.random.normal(0, 1)  # 随机涨跌幅
                close_prices.append(close_prices[-1] + change)
        
        # 生成Open, High, Low价格
        open_prices = []
        high_prices = []
        low_prices = []
        volumes = []
        
        for close in close_prices:
            # 开盘价在收盘价附近小幅波动
            open_p = close * (1 + np.random.normal(0, 0.002))
            
            # 最高价和最低价
            high_p = max(close, open_p) * (1 + np.random.uniform(0.001, 0.005))
            low_p = min(close, open_p) * (1 - np.random.uniform(0.001, 0.005))
            
            # 成交量
            volume = max(5000, np.random.normal(10000, 2000))
            
            open_prices.append(open_p)
            high_prices.append(high_p)
            low_prices.append(low_p)
            volumes.append(volume)
        
        # 为底分型确认日增加成交量
        if pattern == 'innovation_low_backtest' and len(close_prices) >= 55:
            volumes[54] = volumes[54] * 1.5  # K5确认日成交量放大
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        return df
    
    def load_real_data(self, file_path: str) -> pd.DataFrame:
        """加载真实的日线数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            日线数据DataFrame
        """
        try:
            # 支持CSV和JSON格式
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 处理可能是信号数据而不是原始K线数据的情况
                if isinstance(data, list) and len(data) > 0:
                    # 检查是否是交易信号数据（包含date、type、price等字段）
                    if 'date' in data[0] and 'price' in data[0]:
                        logger.info("检测到交易信号数据，创建模拟K线数据")
                        # 从信号数据创建简化的K线数据
                        kline_data = []
                        for signal in data:
                            timestamp = signal['date']
                            # 处理毫秒时间戳
                            if timestamp > 1e12:  # 如果是毫秒级时间戳
                                timestamp = timestamp / 1000
                            date = datetime.fromtimestamp(timestamp)
                            price = signal['price']
                            
                            # 创建简单的K线结构
                            kline_data.append({
                                'date': date,
                                'open': price * 0.995,
                                'high': price * 1.01,
                                'low': price * 0.99,
                                'close': price,
                                'volume': 10000 * np.random.uniform(0.8, 1.2)
                            })
                        df = pd.DataFrame(kline_data)
                    else:
                        # 假设是K线数据
                        df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 确保必要的K线列存在
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"数据缺少必要列: {missing_columns}，将跳过此数据")
                return pd.DataFrame()
            
            # 确保日期列转换
            df['date'] = pd.to_datetime(df['date'])
            
            # 排序
            df = df.sort_values('date')
            
            return df
        except Exception as e:
            logger.error(f"加载真实数据失败: {str(e)}")
            return pd.DataFrame()
    
    def test_detector_with_mock_data(self) -> Dict:
        """使用模拟数据测试检测器
        
        Returns:
            测试结果
        """
        test_results = []
        
        # 测试不同的价格模式
        patterns = ['normal', 'down_trend', 'uptrend', 'sideways', 'innovation_low_backtest']
        
        for pattern in patterns:
            logger.info(f"测试模式: {pattern}")
            
            # 生成模拟数据，增加天数以确保有足够的数据进行分析
            df = self.generate_mock_daily_data(days=120, pattern=pattern)
            
            try:
                # 运行检测器
                result = self.detector.analyze_daily_buy_condition(df)
                
                # 记录测试结果
                test_results.append({
                    'pattern': pattern,
                    'success': result.get('success', False),
                    'has_valid_buy_point': result.get('has_valid_buy_point', False),
                    'signal_strength': result.get('signal_strength', 0),
                    'error': result.get('error', None),
                    'conditions': result.get('conditions', {})
                })
                
                # 输出详细信息
                if result.get('success'):
                    logger.info(f"  检测到买点: {result.get('has_valid_buy_point')}")
                    logger.info(f"  信号强度: {result.get('signal_strength'):.4f}")
                    logger.info(f"  条件满足情况: {result.get('conditions', {})}")
                else:
                    logger.warning(f"  检测失败: {result.get('error')}")
            except Exception as e:
                logger.error(f"  运行检测器异常: {str(e)}")
                test_results.append({
                    'pattern': pattern,
                    'success': False,
                    'has_valid_buy_point': False,
                    'signal_strength': 0,
                    'error': str(e),
                    'conditions': {}
                })
        
        return {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': test_results
        }
    
    def test_detector_with_real_data(self, data_file: str) -> Dict:
        """使用真实数据测试检测器
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            测试结果
        """
        # 加载真实数据
        df = self.load_real_data(data_file)
        
        if df.empty:
            return {
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': False,
                'has_valid_buy_point': False,
                'signal_strength': 0,
                'conditions': {},
                'details': {},
                'data_statistics': {},
                'error': '无法加载有效数据'
            }
        
        try:
            # 运行检测器
            result = self.detector.analyze_daily_buy_condition(df)
            
            # 获取统计信息
            stats = {
                'data_count': len(df),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'avg': df['close'].mean()
                }
            }
            
            return {
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': result.get('success', False),
                'has_valid_buy_point': result.get('has_valid_buy_point', False),
                'signal_strength': result.get('signal_strength', 0),
                'conditions': result.get('conditions', {}),
                'details': result.get('details', {}),
                'data_statistics': stats,
                'error': result.get('error', None)
            }
        except Exception as e:
            logger.error(f"运行检测器时发生异常: {str(e)}")
            return {
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'success': False,
                'has_valid_buy_point': False,
                'signal_strength': 0,
                'conditions': {},
                'details': {},
                'data_statistics': {},
                'error': str(e)
            }
    
    def validate_signal_detection_accuracy(self, result: Dict) -> Dict:
        """验证信号检测的准确性
        
        Args:
            result: 检测结果
            
        Returns:
            准确性评估结果
        """
        if not result.get('success'):
            return {
                'accuracy_assessment': '无法评估',
                'reason': result.get('error', '检测失败')
            }
        
        # 获取条件满足情况
        conditions = result.get('conditions', {})
        
        # 计算条件满足率
        condition_count = len(conditions)
        met_count = sum(conditions.values())
        condition_met_rate = met_count / condition_count if condition_count > 0 else 0
        
        # 评估信号强度
        signal_strength = result.get('signal_strength', 0)
        strength_assessment = '强' if signal_strength >= 0.6 else '中' if signal_strength >= 0.4 else '弱'
        
        # 综合评估
        if result.get('has_valid_buy_point'):
            assessment = '高准确性'
        elif condition_met_rate >= 0.8:
            assessment = '较高准确性'
        elif condition_met_rate >= 0.5:
            assessment = '中等准确性'
        else:
            assessment = '低准确性'
        
        return {
            'accuracy_assessment': assessment,
            'condition_met_rate': f"{condition_met_rate:.2%}",
            'signal_strength_assessment': strength_assessment,
            'full_conditions_met': result.get('has_valid_buy_point', False),
            'condition_details': conditions
        }
    
    def run_comprehensive_test(self) -> Dict:
        """运行综合测试
        
        Returns:
            综合测试结果
        """
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mock_data_tests': [],
            'real_data_tests': []
        }
        
        # 1. 测试模拟数据
        logger.info("===== 开始模拟数据测试 =====")
        mock_results = self.test_detector_with_mock_data()
        test_results['mock_data_tests'] = mock_results['test_results']
        
        # 2. 测试真实数据（如果有）
        logger.info("\n===== 开始真实数据测试（可选）=====")
        # 尝试加载信号文件作为真实数据
        signals_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251124_120616.json'
        
        try:
            if os.path.exists(signals_file):
                real_result = self.test_detector_with_real_data(signals_file)
                if not real_result['error']:
                    test_results['real_data_tests'].append({
                        'data_source': signals_file,
                        'result': real_result,
                        'accuracy': self.validate_signal_detection_accuracy(real_result)
                    })
            else:
                logger.warning(f"真实数据文件不存在: {signals_file}")
        except Exception as e:
            logger.error(f"运行真实数据测试异常: {str(e)}")
        
        # 3. 统计测试结果
        total_mock_tests = len(test_results['mock_data_tests'])
        successful_mock_tests = sum(1 for test in test_results['mock_data_tests'] if test['success'])
        buy_points_detected = sum(1 for test in test_results['mock_data_tests'] if test['has_valid_buy_point'])
        
        test_results['statistics'] = {
            'total_mock_tests': total_mock_tests,
            'successful_mock_tests': successful_mock_tests,
            'success_rate': successful_mock_tests / total_mock_tests if total_mock_tests > 0 else 0,
            'buy_points_detected': buy_points_detected,
            'total_real_tests': len(test_results['real_data_tests'])
        }
        
        return test_results
    
    def print_test_report(self, test_results: Dict):
        """打印测试报告
        
        Args:
            test_results: 测试结果
        """
        print("\n" + "="*80)
        print(f"缠论日线级别买点检测器测试报告")
        print(f"测试时间: {test_results['test_date']}")
        print("="*80)
        
        # 打印模拟数据测试结果
        print("\n模拟数据测试结果:")
        print("-"*60)
        print(f"{'测试模式':<20} {'成功':<8} {'检测买点':<10} {'信号强度':<12}")
        print("-"*60)
        
        for test in test_results['mock_data_tests']:
            success_str = "✅" if test['success'] else "❌"
            buy_point_str = "✅" if test['has_valid_buy_point'] else "❌"
            print(f"{test['pattern']:<20} {success_str:<8} {buy_point_str:<10} {test['signal_strength']:.4f}")
        
        # 打印统计信息
        stats = test_results['statistics']
        print("\n测试统计:")
        print(f"  总模拟测试次数: {stats['total_mock_tests']}")
        print(f"  成功测试次数: {stats['successful_mock_tests']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        print(f"  检测到买点数量: {stats['buy_points_detected']}")
        print(f"  真实数据测试次数: {stats['total_real_tests']}")
        
        # 打印真实数据测试结果（如果有）
        if test_results['real_data_tests']:
            print("\n真实数据测试结果:")
            print("-"*60)
            
            for test in test_results['real_data_tests']:
                print(f"数据来源: {test['data_source']}")
                result = test['result']
                accuracy = test['accuracy']
                
                print(f"  检测状态: {'成功' if result['success'] else '失败'}")
                print(f"  检测到买点: {'是' if result['has_valid_buy_point'] else '否'}")
                print(f"  信号强度: {result['signal_strength']:.4f}")
                print(f"  准确性评估: {accuracy['accuracy_assessment']}")
                print(f"  条件满足率: {accuracy['condition_met_rate']}")
                print(f"  数据统计: {result['data_statistics']}")
                print("\n  条件满足详情:")
                for cond_name, is_met in result.get('conditions', {}).items():
                    print(f"    • {cond_name}: {'满足' if is_met else '不满足'}")
        
        print("\n" + "="*80)
        print("测试结论:")
        
        # 基于测试结果给出结论
        if stats['success_rate'] >= 0.8:
            print("✅ 检测器整体表现良好")
        elif stats['success_rate'] >= 0.5:
            print("⚠️ 检测器表现中等，建议进一步优化")
        else:
            print("❌ 检测器表现不佳，需要重大优化")
        
        # 特殊模式测试结果
        innovation_low_test = next((t for t in test_results['mock_data_tests'] if t['pattern'] == 'innovation_low_backtest'), None)
        if innovation_low_test:
            if innovation_low_test['has_valid_buy_point']:
                print("✅ 在'创新低破中枢回抽'模式中成功检测到买点")
            else:
                print("❌ 在'创新低破中枢回抽'模式中未能检测到买点，需要检查逻辑")
        
        # 输出成功检测买点的消息（如果有）
        buy_points_detected = stats['buy_points_detected']
        if buy_points_detected > 0:
            print(f"✅ 在{buy_points_detected}个模拟场景中成功检测到买点")
    
    def save_test_results(self, test_results: Dict, output_dir: str = './outputs/tests'):
        """保存测试结果到文件
        
        Args:
            test_results: 测试结果
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'test_results_{timestamp}.json')
        
        # 保存结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"测试结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {str(e)}")


def main():
    """主函数"""
    # 创建测试器实例
    tester = ChanlunDetectorTester()
    
    # 运行综合测试
    logger.info("开始运行缠论日线级别买点检测器综合测试...")
    test_results = tester.run_comprehensive_test()
    
    # 打印测试报告
    tester.print_test_report(test_results)
    
    # 保存测试结果
    tester.save_test_results(test_results)
    
    # 额外运行统计分析器测试
    logger.info("\n测试信号统计分析器...")
    try:
        # 生成一些模拟信号数据用于测试统计分析器
        config_dir = '/Users/pingan/tools/trade/tianyuan/config'
        analyzer = SignalStatisticsAnalyzer(config_dir)
        
        # 创建模拟信号
        mock_signals = [
            # 日线级别信号
            {
                'date': int((datetime.now() - timedelta(days=10)).timestamp() * 1000),
                'type': 'buy',
                'price': 0.591,
                'strength': 0.65,
                'reason': '日线级别特殊一买'
            },
            {
                'date': int((datetime.now() - timedelta(days=40)).timestamp() * 1000),
                'type': 'buy',
                'price': 0.582,
                'strength': 0.62,
                'reason': '日线底背驰确认'
            },
            # 分钟级别信号
            {
                'date': int((datetime.now() - timedelta(days=5)).timestamp() * 1000),
                'type': 'buy',
                'price': 0.588,
                'strength': 0.55,
                'reason': '30分钟一买'
            },
            {
                'date': int((datetime.now() - timedelta(days=8)).timestamp() * 1000),
                'type': 'buy',
                'price': 0.590,
                'strength': 0.58,
                'reason': '15分钟三买'
            },
            {
                'date': int((datetime.now() - timedelta(days=3)).timestamp() * 1000),
                'type': 'buy',
                'price': 0.587,
                'strength': 0.52,
                'reason': '60分钟一买'
            }
        ]
        
        # 测试周期解析
        for signal in mock_signals:
            timeframe_type, specific_timeframe = analyzer.parse_signal_timeframe(signal)
            logger.info(f"信号: {signal['reason']} -> 周期类型: {timeframe_type}, 具体周期: {specific_timeframe}")
        
        # 测试信号过滤
        core_signals = analyzer.filter_core_daily_signals(mock_signals)
        logger.info(f"筛选出的核心日线信号数量: {len(core_signals)}")
        
        # 测试统计计算
        core_stats = analyzer.calculate_core_statistics(core_signals)
        minute_stats = analyzer.calculate_minute_statistics(mock_signals)
        
        logger.info("统计分析器测试成功!")
        logger.info(f"核心信号统计: {core_stats}")
        logger.info(f"分钟级别统计: {minute_stats}")
        
    except Exception as e:
        logger.error(f"统计分析器测试失败: {str(e)}")


if __name__ == "__main__":
    main()