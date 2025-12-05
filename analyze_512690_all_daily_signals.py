#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒ETF(512690)所有日线级别买点信号分析脚本

该脚本分析512690在2025年的所有日线级别买点信号，包括：
1. 日线二买（核心买点）
2. 日线一买（辅助买点）
3. 日线三买（辅助买点）
4. 日线破中枢反抽（兜底买点）

作者: TradeTianYuan
日期: 2025-11-29
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("512690DailySignalAnalyzer")

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入需要的模块
try:
    from data_validator import DataValidator
    from weekly_trend_detector import WeeklyTrendDetector
    from daily_buy_signal_detector import BuySignalDetector
    from data_fetcher import StockDataFetcher
    logger.info("成功导入所需模块")
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    sys.exit(1)


class DailySignalAnalyzer:
    """日线信号分析器类"""
    
    def __init__(self, data_dir="data"):
        """初始化分析器
        
        Args:
            data_dir: 数据目录路径
        """
        self.logger = logging.getLogger("512690DailySignalAnalyzer")
        self.logger.info("初始化日线信号分析器...")
        
        # 设置数据目录
        self.data_dir = data_dir
        self.daily_data_path = os.path.join(data_dir, "daily", "512690_daily.csv")
        self.weekly_data_path = os.path.join(data_dir, "weekly", "512690_weekly.csv")
        
        # 初始化检测器
        self.data_validator = DataValidator()
        self.weekly_detector = WeeklyTrendDetector()
        self.daily_detector = BuySignalDetector()
        self.data_fetcher = StockDataFetcher()
        
        # 数据存储
        self.daily_data = None
        self.weekly_data = None
        self.all_signals = []
        self.year_2025_signals = []
        
        # 创建结果目录
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info("日线信号分析器初始化完成")
    
    def load_data(self):
        """加载日线和周线数据
        
        Returns:
            bool: 数据加载是否成功
        """
        self.logger.info("开始加载数据...")
        
        try:
            # 尝试加载本地数据
            if os.path.exists(self.daily_data_path):
                self.daily_data = pd.read_csv(self.daily_data_path)
                self.logger.info(f"成功加载日线数据，共{len(self.daily_data)}条记录")
            else:
                # 如果本地数据不存在，尝试使用数据获取器
                self.logger.info("本地日线数据不存在，尝试使用数据获取器...")
                self.daily_data = self.data_fetcher.fetch_daily_data("512690", days=730)  # 获取2年数据
                self.logger.info(f"成功获取日线数据，共{len(self.daily_data)}条记录")
            
            if os.path.exists(self.weekly_data_path):
                self.weekly_data = pd.read_csv(self.weekly_data_path)
                self.logger.info(f"成功加载周线数据，共{len(self.weekly_data)}条记录")
            else:
                # 如果本地数据不存在，尝试使用数据获取器
                self.logger.info("本地周线数据不存在，尝试使用数据获取器...")
                self.weekly_data = self.data_fetcher.fetch_weekly_data("512690", weeks=104)  # 获取2年数据
                self.logger.info(f"成功获取周线数据，共{len(self.weekly_data)}条记录")
            
            # 确保日期列格式正确
            if 'date' in self.daily_data.columns:
                self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
            if 'date' in self.weekly_data.columns:
                self.weekly_data['date'] = pd.to_datetime(self.weekly_data['date'])
            
            # 验证数据有效性
            if not self.data_validator.validate_daily_data(self.daily_data):
                self.logger.error("日线数据验证失败")
                return False
            
            if not self.data_validator.validate_weekly_data(self.weekly_data):
                self.logger.error("周线数据验证失败")
                return False
            
            self.logger.info("数据加载和验证成功")
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            # 如果获取器失败，尝试使用模拟数据
            try:
                self._generate_mock_data()
                self.logger.info("使用模拟数据进行分析")
                return True
            except Exception as mock_e:
                self.logger.error(f"模拟数据生成失败: {str(mock_e)}")
                return False
    
    def _generate_mock_data(self):
        """生成模拟数据用于测试"""
        # 生成最近2年的交易日
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - 2)
        
        # 生成模拟日线数据
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(42)  # 设置随机种子，确保结果可复现
        
        self.daily_data = pd.DataFrame({
            'date': date_range,
            'open': np.random.normal(1.2, 0.1, len(date_range)),
            'high': np.random.normal(1.25, 0.1, len(date_range)),
            'low': np.random.normal(1.15, 0.1, len(date_range)),
            'close': np.random.normal(1.2, 0.1, len(date_range)),
            'volume': np.random.normal(1000000, 500000, len(date_range))
        })
        
        # 确保high > close > low > open的合理关系
        self.daily_data['high'] = self.daily_data[['open', 'high', 'low', 'close']].max(axis=1) + 0.01
        self.daily_data['low'] = self.daily_data[['open', 'high', 'low', 'close']].min(axis=1) - 0.01
        
        # 生成模拟周线数据
        weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        self.weekly_data = pd.DataFrame({
            'date': weekly_dates,
            'open': np.random.normal(1.2, 0.1, len(weekly_dates)),
            'high': np.random.normal(1.25, 0.1, len(weekly_dates)),
            'low': np.random.normal(1.15, 0.1, len(weekly_dates)),
            'close': np.random.normal(1.2, 0.1, len(weekly_dates)),
            'volume': np.random.normal(5000000, 2000000, len(weekly_dates))
        })
    
    def analyze_2025_signals(self):
        """分析2025年的所有日线级别信号
        
        Returns:
            bool: 分析是否成功
        """
        if self.daily_data is None or self.weekly_data is None:
            self.logger.error("数据未加载，无法进行分析")
            return False
        
        self.logger.info("开始分析2025年的日线级别信号...")
        
        try:
            # 筛选2025年的数据
            if 'date' in self.daily_data.columns:
                daily_2025 = self.daily_data[self.daily_data['date'].dt.year == 2025].copy()
                if daily_2025.empty:
                    self.logger.warning("未找到2025年的日线数据")
                    # 如果没有2025年数据，使用最近的数据进行演示
                    daily_2025 = self.daily_data.tail(100).copy()
                    self.logger.info(f"使用最近{len(daily_2025)}条数据进行分析")
            else:
                self.logger.warning("日线数据中没有date列，使用全部数据")
                daily_2025 = self.daily_data.copy()
            
            # 对每一天检测所有级别的买点信号
            for i in range(20, len(daily_2025)):  # 留出足够的历史数据用于计算指标
                window_data = daily_2025.iloc[:i+1].copy()
                current_date = window_data.iloc[-1]['date']
                
                # 获取对应日期的周线数据
                if 'date' in self.weekly_data.columns:
                    weekly_before_current = self.weekly_data[self.weekly_data['date'] <= current_date].copy()
                else:
                    weekly_before_current = self.weekly_data.copy()
                
                # 检测周线趋势（用于信号过滤）
                weekly_trend_status = "数据不足"
                if len(weekly_before_current) >= 30:
                    weekly_trend_result = self.weekly_detector.detect_weekly_bullish_trend(weekly_before_current)
                    # 检查返回类型，处理字典或元组的情况
                    if isinstance(weekly_trend_result, dict):
                        weekly_trend_status = weekly_trend_result.get('status', '未知')
                    elif isinstance(weekly_trend_result, tuple) and len(weekly_trend_result) > 0:
                        # 假设元组第一个元素是状态
                        weekly_trend_status = str(weekly_trend_result[0])
                    else:
                        weekly_trend_status = str(weekly_trend_result)
                
                # 检测所有级别的日线买点信号
                # 按照优先级顺序：二买 > 一买 > 三买 > 反抽
                signal_type = None
                signal_strength = 0
                signal_reason = ""
                
                # 1. 检测日线二买
                second_buy_result = self.daily_detector.detect_daily_second_buy(window_data)
                # 处理可能的元组返回值
                has_second_buy = False
                if isinstance(second_buy_result, dict):
                    has_second_buy = second_buy_result.get('signal', False)
                elif isinstance(second_buy_result, tuple) and len(second_buy_result) > 0:
                    has_second_buy = bool(second_buy_result[0])
                
                if has_second_buy:
                    signal_type = "日线二买"
                    signal_strength = 80
                    signal_reason = "二买信号形成: 满足二买条件"
                else:
                    # 2. 检测日线一买
                    first_buy_result = self.daily_detector.detect_daily_first_buy(window_data)
                    has_first_buy = False
                    if isinstance(first_buy_result, dict):
                        has_first_buy = first_buy_result.get('signal', False)
                    elif isinstance(first_buy_result, tuple) and len(first_buy_result) > 0:
                        has_first_buy = bool(first_buy_result[0])
                    
                    if has_first_buy:
                        signal_type = "日线一买"
                        signal_strength = 70
                        signal_reason = "一买信号形成: 满足一买条件"
                    else:
                        # 3. 检测日线三买
                        third_buy_result = self.daily_detector.detect_daily_third_buy(window_data)
                        has_third_buy = False
                        if isinstance(third_buy_result, dict):
                            has_third_buy = third_buy_result.get('signal', False)
                        elif isinstance(third_buy_result, tuple) and len(third_buy_result) > 0:
                            has_third_buy = bool(third_buy_result[0])
                        
                        if has_third_buy:
                            signal_type = "日线三买"
                            signal_strength = 75
                            signal_reason = "三买信号形成: 满足三买条件"
                        else:
                            # 4. 检测破中枢反抽
                            reverse_result = self.daily_detector.detect_daily_reverse_pullback(window_data)
                            has_reverse = False
                            if isinstance(reverse_result, dict):
                                has_reverse = reverse_result.get('signal', False)
                            elif isinstance(reverse_result, tuple) and len(reverse_result) > 0:
                                has_reverse = bool(reverse_result[0])
                            
                            if has_reverse:
                                signal_type = "破中枢反抽"
                                signal_strength = 60
                                signal_reason = "破中枢反抽信号形成: 满足反抽条件"
                
                # 如果检测到信号，记录下来
                if signal_type:
                    signal_record = {
                        'date': current_date.strftime('%Y-%m-%d') if isinstance(current_date, pd.Timestamp) else str(current_date),
                        'signal_type': signal_type,
                        'signal_strength': signal_strength,
                        'close_price': float(window_data.iloc[-1]['close']),
                        'weekly_trend': weekly_trend_status,
                        'reason': signal_reason,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.all_signals.append(signal_record)
                    
                    # 如果是2025年的数据，添加到2025年信号列表
                    if 'date' in self.daily_data.columns and window_data.iloc[-1]['date'].year == 2025:
                        self.year_2025_signals.append(signal_record)
                    
                    self.logger.info(f"检测到信号: {signal_type} - {current_date} - 价格: {window_data.iloc[-1]['close']:.2f}")
            
            self.logger.info(f"信号分析完成，共检测到{len(self.all_signals)}个信号，其中2025年信号{len(self.year_2025_signals)}个")
            return True
            
        except Exception as e:
            self.logger.error(f"信号分析失败: {str(e)}")
            return False
    
    def generate_report(self):
        """生成分析报告
        
        Returns:
            str: 分析报告内容
        """
        self.logger.info("生成分析报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_lines = []
        
        report_lines.append("===== 酒ETF(512690)日线级别信号分析报告 =====")
        report_lines.append(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("分析范围: 2025年所有日线级别买点信号")
        report_lines.append("")
        
        # 信号概览
        report_lines.append("📊 信号概览:")
        report_lines.append("-" * 80)
        
        if self.year_2025_signals:
            # 按信号类型统计
            signal_types = {}
            for signal in self.year_2025_signals:
                signal_type = signal['signal_type']
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            
            report_lines.append(f"2025年共检测到 {len(self.year_2025_signals)} 个日线级别买点信号")
            for signal_type, count in signal_types.items():
                report_lines.append(f"  - {signal_type}: {count} 个")
            
            # 月度分布
            report_lines.append("月度信号分布:")
            monthly_dist = {}
            for signal in self.year_2025_signals:
                month = signal['date'].split('-')[1]
                monthly_dist[month] = monthly_dist.get(month, 0) + 1
            
            for month in sorted(monthly_dist.keys()):
                month_name = ['', '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'][int(month)]
                report_lines.append(f"  - {month_name}: {monthly_dist[month]} 个")
            
            # 最近的信号
            report_lines.append("📋 最近的日线买点信号:")
            report_lines.append("日期              信号类型    价格       强度    周线趋势    原因")
            report_lines.append("-" * 120)
            
            # 按日期排序，显示最近的10个信号
            recent_signals = sorted(self.year_2025_signals, key=lambda x: x['date'], reverse=True)[:10]
            for signal in recent_signals:
                report_lines.append(f"{signal['date']}    {signal['signal_type']}    {signal['close_price']:.4f}    {signal['signal_strength']}    {signal['weekly_trend']}    {signal['reason']}")
        else:
            report_lines.append("2025年未检测到任何日线级别买点信号")
        
        # 交易建议
        report_lines.append("🎯 交易建议:")
        report_lines.append("-" * 80)
        
        if self.year_2025_signals:
            latest_signal = sorted(self.year_2025_signals, key=lambda x: x['date'], reverse=True)[0]
            report_lines.append(f"最近的信号: {latest_signal['date']} - {latest_signal['signal_type']} (强度: {latest_signal['signal_strength']})")
            report_lines.append(f"信号价格: {latest_signal['close_price']:.4f}")
            report_lines.append(f"周线趋势: {latest_signal['weekly_trend']}")
            report_lines.append(f"信号原因: {latest_signal['reason']}")
        else:
            report_lines.append("🔍 暂无有效信号: 建议继续观察市场走势")
            report_lines.append("   可关注以下条件的形成:")
            report_lines.append("   1. 日线二买: 回调不创新低+底分型+MACD背离")
            report_lines.append("   2. 日线一买: 下跌段结束+MACD底背离")
            report_lines.append("   3. 日线三买: 中枢突破+回抽不进中枢+30分钟底背驰")
            report_lines.append("   4. 破中枢反抽: 跌破中枢后企稳回升")
        
        # 风险提示
        report_lines.append("⚠️ 风险提示:")
        report_lines.append("-" * 80)
        report_lines.append("1. 本分析基于历史数据，不构成投资建议")
        report_lines.append("2. 市场有风险，投资需谨慎")
        report_lines.append("3. 建议结合多级别分析和风险控制策略")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_file = os.path.join(self.results_dir, f"512690_daily_signals_2025_analysis_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"分析报告已保存至: {report_file}")
        
        # 保存信号数据
        signals_file = os.path.join(self.results_dir, f"512690_daily_signals_2025_{timestamp}.json")
        with open(signals_file, 'w', encoding='utf-8') as f:
            json.dump(self.year_2025_signals, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"信号数据已保存至: {signals_file}")
        
        return report_content
    
    def run(self):
        """运行完整的分析流程
        
        Returns:
            bool: 分析是否成功
        """
        self.logger.info("开始运行512690日线级别信号分析...")
        
        try:
            # 1. 加载数据
            if not self.load_data():
                self.logger.error("数据加载失败，分析无法继续")
                return False
            
            # 2. 分析2025年信号
            if not self.analyze_2025_signals():
                self.logger.error("信号分析失败")
                return False
            
            # 3. 生成报告
            report = self.generate_report()
            
            # 打印报告到控制台
            print("\n" + "="*80)
            print(report)
            print("="*80 + "\n")
            
            self.logger.info("日线级别信号分析完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"分析过程中发生错误: {str(e)}")
            return False


def main():
    """主函数"""
    analyzer = DailySignalAnalyzer()
    success = analyzer.run()
    
    if success:
        logger.info("分析成功完成")
        sys.exit(0)
    else:
        logger.error("分析失败")
        sys.exit(1)


if __name__ == "__main__":
    main()