#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""军工ETF(512660)交易信号验证测试脚本 - 模拟2025年10月后信号检测"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_military_etf_signal.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MilitaryETFTest')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入所有功能模块
from src.weekly_analyzer import WeeklyAnalyzer
from src.daily_analyzer import DailyAnalyzer
from src.data_processor import DataProcessor
from src.signal_exporter import SignalExporter
from src.signal_validator import SignalValidator
from src.military_etf_signal_detector import MilitaryETFDetector

class MilitaryETFTest:
    """军工ETF信号测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        logger.info("初始化军工ETF信号测试环境")
        
        # 创建各模块实例
        self.weekly_analyzer = WeeklyAnalyzer()
        self.daily_analyzer = DailyAnalyzer()
        self.data_processor = DataProcessor()
        self.signal_exporter = SignalExporter()
        self.signal_validator = SignalValidator()
        self.main_detector = MilitaryETFDetector()
        
        # 测试配置
        self.symbol = "sh512660"
        self.symbol_name = "军工ETF"
        self.current_year = datetime.now().year
        self.target_year = 2025
        self.target_month = 10
        self.test_days = 180  # 模拟半年的数据
        
        # 创建测试输出目录
        self.output_dir = os.path.join(project_root, "test_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"测试配置: 目标股票={self.symbol_name}({self.symbol}), 目标时间={self.target_year}-{self.target_month}, 模拟天数={self.test_days}")
    
    def get_historical_data(self) -> pd.DataFrame:
        """获取历史数据作为模拟基础
        
        Returns:
            历史日线数据
        """
        logger.info("获取历史数据作为模拟基础")
        
        try:
            # 获取更多历史数据用于模拟
            historical_data = self.data_processor.get_daily_data(
                symbol=self.symbol,
                days=365,  # 获取一年的历史数据
                force_fetch=False
            )
            
            if historical_data.empty:
                logger.error("无法获取历史数据，使用模拟数据进行测试")
                return self._generate_mock_data()
            
            logger.info(f"成功获取历史数据: {len(historical_data)} 条记录")
            return historical_data
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {str(e)}")
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟历史数据
        
        Returns:
            模拟的日线数据
        """
        logger.info("生成模拟历史数据")
        
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 设置初始价格
        start_price = 1.5
        
        # 生成随机价格序列，添加一些趋势和波动
        np.random.seed(42)  # 设置随机种子以保证结果可复现
        returns = np.random.normal(0.0001, 0.015, len(date_range))
        
        # 添加一些周期性模式和趋势
        trend = np.linspace(0, 0.2, len(date_range))  # 整体上涨趋势
        cycle = 0.1 * np.sin(np.linspace(0, 8 * np.pi, len(date_range)))  # 周期性波动
        
        # 组合生成价格
        cum_returns = (1 + returns + trend/365 + cycle/365).cumprod()
        prices = start_price * cum_returns
        
        # 生成OHLC数据
        open_prices = prices * (1 + np.random.normal(0, 0.005, len(date_range)))
        high_prices = np.maximum(open_prices, prices) * (1 + np.random.normal(0, 0.008, len(date_range)))
        low_prices = np.minimum(open_prices, prices) * (1 - np.random.normal(0, 0.008, len(date_range)))
        close_prices = prices
        
        # 生成成交量数据
        base_volume = 50000000  # 5千万
        volumes = base_volume * (1 + np.random.normal(0, 0.3, len(date_range)) + 0.5 * np.abs(returns))
        volumes = np.maximum(volumes, base_volume * 0.1)  # 确保成交量不为零
        
        # 创建DataFrame
        mock_data = pd.DataFrame({
            'date': date_range,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        # 转换日期格式
        mock_data['date'] = mock_data['date'].dt.strftime('%Y-%m-%d')
        
        logger.info(f"成功生成模拟数据: {len(mock_data)} 条记录")
        return mock_data
    
    def simulate_future_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """模拟2025年10月后的数据
        
        Args:
            historical_data: 历史数据
            
        Returns:
            模拟的未来数据
        """
        logger.info(f"开始模拟{self.target_year}年{self.target_month}月后的数据")
        
        # 计算历史数据的统计特征用于模拟
        historical_returns = historical_data['close'].pct_change().dropna()
        hist_mean = historical_returns.mean()
        hist_std = historical_returns.std()
        hist_volatility = hist_std
        
        # 确定模拟起始日期
        if self.current_year < self.target_year or \
           (self.current_year == self.target_year and datetime.now().month < self.target_month):
            # 如果目标时间在未来，从当前开始模拟
            start_date = datetime.now()
        else:
            # 如果目标时间已过，使用目标时间作为起点
            start_date = datetime(self.target_year, self.target_month, 1)
        
        # 生成未来日期序列
        end_date = start_date + timedelta(days=self.test_days)
        future_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 获取最近的价格作为起点
        if not historical_data.empty:
            last_price = historical_data['close'].iloc[-1]
        else:
            last_price = 1.5  # 默认起始价格
        
        # 生成模拟数据
        np.random.seed(43)  # 新的随机种子
        
        # 为了创建更真实的市场情境，模拟几个不同的市场阶段
        n_days = len(future_dates)
        
        # 创建不同市场阶段
        phase1_days = int(n_days * 0.2)  # 20% 时间为盘整
        phase2_days = int(n_days * 0.3)  # 30% 时间为下跌
        phase3_days = n_days - phase1_days - phase2_days  # 剩余时间为上涨（可能产生买入信号）
        
        # 生成不同阶段的收益率
        phase1_returns = np.random.normal(0, hist_std * 0.8, phase1_days)  # 盘整阶段
        phase2_returns = np.random.normal(-hist_std * 1.2, hist_std * 0.8, phase2_days)  # 下跌阶段
        
        # 上涨阶段：先生成一些下跌然后上涨，创造背驰条件
        phase3_part1 = int(phase3_days * 0.3)  # 先下跌一部分
        phase3_part2 = phase3_days - phase3_part1  # 然后上涨
        
        phase3_returns1 = np.random.normal(-hist_std * 0.8, hist_std * 0.8, phase3_part1)
        phase3_returns2 = np.random.normal(hist_std * 1.5, hist_std * 0.8, phase3_part2)
        
        # 合并所有阶段的收益率
        all_returns = np.concatenate([phase1_returns, phase2_returns, phase3_returns1, phase3_returns2])
        
        # 计算累计收益率
        cum_returns = (1 + all_returns).cumprod()
        future_prices = last_price * cum_returns
        
        # 生成OHLC数据
        open_prices = future_prices * (1 + np.random.normal(0, 0.005, n_days))
        high_prices = np.maximum(open_prices, future_prices) * (1 + np.random.normal(0, 0.008, n_days))
        low_prices = np.minimum(open_prices, future_prices) * (1 - np.random.normal(0, 0.008, n_days))
        close_prices = future_prices
        
        # 生成成交量数据（在关键转折点放大成交量）
        base_volume = 50000000  # 5千万
        volumes = base_volume * (1 + np.random.normal(0, 0.3, n_days))
        
        # 在阶段转换点增加成交量
        volumes[phase1_days-1:phase1_days+1] *= 1.5
        volumes[phase1_days+phase2_days-1:phase1_days+phase2_days+1] *= 2.0  # 下跌转上涨时成交量放大
        volumes[phase1_days+phase2_days+phase3_part1-1:phase1_days+phase2_days+phase3_part1+1] *= 1.8
        
        volumes = np.maximum(volumes, base_volume * 0.1)  # 确保成交量不为零
        
        # 创建模拟数据DataFrame
        future_data = pd.DataFrame({
            'date': future_dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        # 转换日期格式
        future_data['date'] = future_data['date'].dt.strftime('%Y-%m-%d')
        
        logger.info(f"成功模拟未来数据: {len(future_data)} 条记录，日期范围: {future_data['date'].iloc[0]} 至 {future_data['date'].iloc[-1]}")
        
        # 保存模拟数据
        future_data_path = os.path.join(self.output_dir, f"simulated_future_data_{self.target_year}_{self.target_month}.csv")
        future_data.to_csv(future_data_path, index=False, encoding='utf-8')
        logger.info(f"模拟数据已保存至: {future_data_path}")
        
        return future_data
    
    def run_signal_detection(self, future_data: pd.DataFrame) -> List[Dict[str, any]]:
        """在模拟数据上运行信号检测
        
        Args:
            future_data: 模拟的未来数据
            
        Returns:
            检测到的信号列表
        """
        logger.info("在模拟数据上运行信号检测")
        
        detected_signals = []
        
        # 模拟实时检测过程：使用滚动窗口
        window_size = 60  # 至少需要60天数据进行分析
        
        if len(future_data) < window_size:
            logger.warning(f"数据不足，需要至少{window_size}天，当前只有{len(future_data)}天")
            window_size = len(future_data)
        
        # 滚动检测信号
        for i in range(window_size, len(future_data) + 1):
            # 截取当前窗口的数据
            window_data = future_data.iloc[:i].copy()
            current_date = window_data['date'].iloc[-1]
            
            try:
                logger.info(f"检测日期: {current_date} 的信号")
                
                # 使用主检测器运行信号检测
                signals = self.main_detector.detect_signals(
                    daily_data=window_data,
                    symbol=self.symbol,
                    symbol_name=self.symbol_name,
                    test_mode=True
                )
                
                # 处理检测到的信号
                if signals:
                    for signal in signals:
                        signal['检测日期'] = current_date
                        detected_signals.append(signal)
                        logger.info(f"检测到信号: {signal.get('信号类型')}，置信度: {signal.get('置信度档位')}")
                        
            except Exception as e:
                logger.error(f"检测日期 {current_date} 的信号时出错: {str(e)}")
        
        # 去重处理：如果同一天检测到多个相同类型的信号，只保留最强的一个
        unique_signals = self._deduplicate_signals(detected_signals)
        
        logger.info(f"信号检测完成: 原始检测到 {len(detected_signals)} 个信号，去重后 {len(unique_signals)} 个信号")
        
        # 保存检测到的信号
        if unique_signals:
            signals_path = os.path.join(self.output_dir, f"detected_signals_{self.target_year}_{self.target_month}.csv")
            self.signal_exporter.export_to_csv(unique_signals, signals_path)
            logger.info(f"检测到的信号已保存至: {signals_path}")
        
        return unique_signals
    
    def _deduplicate_signals(self, signals: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """信号去重
        
        Args:
            signals: 原始信号列表
            
        Returns:
            去重后的信号列表
        """
        if not signals:
            return []
        
        # 按日期和信号类型分组，选择信号强度最高的
        signal_groups = {}
        
        for signal in signals:
            key = f"{signal.get('检测日期', '')}_{signal.get('信号类型', '')}"
            current_strength = signal.get('信号强度', 0)
            
            if key not in signal_groups or current_strength > signal_groups[key].get('信号强度', 0):
                signal_groups[key] = signal
        
        return list(signal_groups.values())
    
    def validate_detected_signals(self, signals: List[Dict[str, any]], 
                                future_data: pd.DataFrame) -> List[Dict[str, any]]:
        """验证检测到的信号
        
        Args:
            signals: 检测到的信号列表
            future_data: 模拟的未来数据
            
        Returns:
            验证后的信号列表
        """
        if not signals:
            logger.info("没有检测到信号，无需验证")
            return []
        
        logger.info(f"开始验证 {len(signals)} 个检测到的信号")
        
        validated_signals = []
        
        for signal in signals:
            try:
                # 找到信号检测日期在数据中的位置
                signal_date = signal.get('检测日期', '')
                if signal_date:
                    # 获取信号之后的数据用于验证
                    signal_idx = future_data[future_data['date'] == signal_date].index
                    
                    if not signal_idx.empty:
                        # 截取信号之后的数据（15天作为验证窗口）
                        validate_idx = signal_idx[0] + 1
                        if validate_idx < len(future_data):
                            validate_end_idx = min(validate_idx + 15, len(future_data))
                            validation_data = future_data.iloc[validate_idx:validate_end_idx].copy()
                            
                            # 使用验证器验证信号
                            validated_signal = self.signal_validator.validate_signal(
                                signal=signal,
                                latest_data=validation_data,
                                symbol=self.symbol
                            )
                            validated_signals.append(validated_signal)
                        else:
                            logger.warning(f"信号 {signal_date} 之后没有足够的数据用于验证")
                            validated_signals.append(signal)  # 保留原始信号
                    else:
                        logger.warning(f"信号检测日期 {signal_date} 在数据中找不到")
                        validated_signals.append(signal)  # 保留原始信号
                else:
                    logger.warning("信号缺少检测日期")
                    validated_signals.append(signal)  # 保留原始信号
                    
            except Exception as e:
                logger.error(f"验证信号时出错: {str(e)}")
                validated_signals.append(signal)  # 出错时保留原始信号
        
        # 生成验证报告
        report_path = os.path.join(self.output_dir, f"validation_report_{self.target_year}_{self.target_month}.json")
        self.signal_validator.generate_validation_report(validated_signals, report_path)
        
        # 统计验证结果
        valid_count = sum(1 for s in validated_signals if not s.get('expired', False))
        expired_count = len(validated_signals) - valid_count
        
        logger.info(f"信号验证完成: 有效信号 {valid_count} 个，失效信号 {expired_count} 个")
        
        return validated_signals
    
    def visualize_results(self, future_data: pd.DataFrame, signals: List[Dict[str, any]]):
        """可视化测试结果
        
        Args:
            future_data: 模拟的未来数据
            signals: 检测到的信号列表
        """
        logger.info("生成测试结果可视化图表")
        
        try:
            # 转换日期格式便于绘图
            plot_data = future_data.copy()
            plot_data['date'] = pd.to_datetime(plot_data['date'])
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制价格图
            ax1.plot(plot_data['date'], plot_data['close'], label='收盘价', linewidth=2, color='blue')
            
            # 标记买入信号
            buy_signals = [s for s in signals if s.get('信号类型', '').lower().find('买入') >= 0]
            for signal in buy_signals:
                signal_date = pd.to_datetime(signal.get('检测日期', ''))
                if not signal_date:  # 如果没有检测日期，使用生成时间
                    signal_date = pd.to_datetime(signal.get('生成时间', ''))
                
                # 查找对应的价格
                if signal_date in plot_data['date'].values:
                    idx = plot_data[plot_data['date'] == signal_date].index[0]
                    price = plot_data.loc[idx, 'close']
                    
                    # 根据信号强度设置颜色
                    strength = signal.get('信号强度', 0)
                    if strength >= 80:
                        color = 'green'
                    elif strength >= 60:
                        color = 'limegreen'
                    else:
                        color = 'lightgreen'
                    
                    # 标记买入信号
                    ax1.scatter(signal_date, price, marker='^', s=150, color=color, 
                               edgecolor='black', linewidth=1.5, zorder=5)
                    ax1.annotate(f"{signal.get('置信度档位', '')}\n{strength}",
                               xy=(signal_date, price), xytext=(0, 15),
                               textcoords='offset points', ha='center', va='bottom',
                               fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 绘制成交量
            ax2.bar(plot_data['date'], plot_data['volume'], color='gray', alpha=0.6)
            
            # 设置图表属性
            ax1.set_title(f'军工ETF(512660)模拟数据与交易信号 ({self.target_year}年{self.target_month}月后)', fontsize=15)
            ax1.set_ylabel('价格', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            ax2.set_xlabel('日期', fontsize=12)
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 设置x轴日期格式
            fig.autofmt_xdate()
            
            # 保存图表
            chart_path = os.path.join(self.output_dir, f"signal_visualization_{self.target_year}_{self.target_month}.png")
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"测试结果图表已保存至: {chart_path}")
            
            # 显示图表（可选）
            # plt.show()
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {str(e)}")
    
    def generate_test_report(self, signals: List[Dict[str, any]], 
                           validated_signals: List[Dict[str, any]]) -> Dict[str, any]:
        """生成测试报告
        
        Args:
            signals: 检测到的信号列表
            validated_signals: 验证后的信号列表
            
        Returns:
            测试报告数据
        """
        logger.info("生成测试报告")
        
        # 统计信号类型
        signal_types = {}
        confidence_levels = {}
        
        for signal in signals:
            sig_type = signal.get('信号类型', '未知')
            confidence = signal.get('置信度档位', '未知')
            
            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
        
        # 统计验证结果
        if validated_signals:
            valid_count = sum(1 for s in validated_signals if not s.get('expired', False))
            expired_count = len(validated_signals) - valid_count
            valid_ratio = valid_count / len(validated_signals) * 100
        else:
            valid_count = expired_count = valid_ratio = 0
        
        # 创建报告
        report = {
            "test_summary": {
                "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target_symbol": f"{self.symbol_name}({self.symbol})",
                "target_period": f"{self.target_year}年{self.target_month}月后",
                "simulation_days": self.test_days,
                "total_signals_detected": len(signals),
                "signal_types_distribution": signal_types,
                "confidence_level_distribution": confidence_levels
            },
            "validation_results": {
                "total_validated_signals": len(validated_signals),
                "valid_signals_count": valid_count,
                "expired_signals_count": expired_count,
                "valid_signals_ratio": round(valid_ratio, 2)
            },
            "detailed_signals": signals,
            "system_evaluation": {
                "detection_accuracy": round(valid_ratio, 2),
                "confidence_reliability": self._evaluate_confidence_reliability(validated_signals),
                "suggestions": self._generate_improvement_suggestions(signals, validated_signals)
            }
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, f"test_report_{self.target_year}_{self.target_month}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"测试报告已保存至: {report_path}")
        
        # 生成简单的文本报告
        txt_report_path = os.path.join(self.output_dir, f"test_report_summary_{self.target_year}_{self.target_month}.txt")
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("===== 军工ETF交易信号检测系统测试报告 =====\n\n")
            f.write(f"测试日期: {report['test_summary']['test_date']}\n")
            f.write(f"测试对象: {report['test_summary']['target_symbol']}\n")
            f.write(f"目标时间: {report['test_summary']['target_period']}\n")
            f.write(f"模拟天数: {report['test_summary']['simulation_days']}\n\n")
            
            f.write("信号检测统计:\n")
            f.write(f"  总检测信号数: {report['test_summary']['total_signals_detected']}\n")
            
            if report['test_summary']['signal_types_distribution']:
                f.write("  信号类型分布:\n")
                for sig_type, count in report['test_summary']['signal_types_distribution'].items():
                    f.write(f"    - {sig_type}: {count} 个\n")
            
            if report['test_summary']['confidence_level_distribution']:
                f.write("  置信度分布:\n")
                for level, count in report['test_summary']['confidence_level_distribution'].items():
                    f.write(f"    - {level}: {count} 个\n")
            
            f.write("\n信号验证结果:\n")
            f.write(f"  有效信号数: {report['validation_results']['valid_signals_count']}\n")
            f.write(f"  失效信号数: {report['validation_results']['expired_signals_count']}\n")
            f.write(f"  信号有效率: {report['validation_results']['valid_signals_ratio']}%\n\n")
            
            f.write("系统评估:\n")
            f.write(f"  检测准确率: {report['system_evaluation']['detection_accuracy']}%\n")
            
            if report['system_evaluation']['confidence_reliability']:
                f.write("  置信度可靠性:\n")
                for level, reliability in report['system_evaluation']['confidence_reliability'].items():
                    f.write(f"    - {level}: {reliability}%\n")
            
            f.write("\n改进建议:\n")
            for suggestion in report['system_evaluation']['suggestions']:
                f.write(f"  - {suggestion}\n")
        
        logger.info(f"测试报告摘要已保存至: {txt_report_path}")
        
        return report
    
    def _evaluate_confidence_reliability(self, validated_signals: List[Dict[str, any]]) -> Dict[str, float]:
        """评估不同置信度级别的可靠性
        
        Args:
            validated_signals: 验证后的信号列表
            
        Returns:
            不同置信度级别的可靠性字典
        """
        if not validated_signals:
            return {}
        
        # 按置信度分组统计
        confidence_stats = {}
        
        for signal in validated_signals:
            confidence = signal.get('置信度档位', '未知')
            is_valid = not signal.get('expired', False)
            
            if confidence not in confidence_stats:
                confidence_stats[confidence] = {'total': 0, 'valid': 0}
            
            confidence_stats[confidence]['total'] += 1
            if is_valid:
                confidence_stats[confidence]['valid'] += 1
        
        # 计算可靠性（有效率）
        reliability = {}
        for level, stats in confidence_stats.items():
            if stats['total'] > 0:
                reliability[level] = round(stats['valid'] / stats['total'] * 100, 2)
        
        return reliability
    
    def _generate_improvement_suggestions(self, signals: List[Dict[str, any]], 
                                        validated_signals: List[Dict[str, any]]) -> List[str]:
        """生成系统改进建议
        
        Args:
            signals: 检测到的信号列表
            validated_signals: 验证后的信号列表
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        if not signals:
            suggestions.append("系统未能检测到任何信号，建议检查信号检测算法的敏感度")
            return suggestions
        
        # 分析验证结果
        if validated_signals:
            valid_ratio = sum(1 for s in validated_signals if not s.get('expired', False)) / len(validated_signals)
            
            if valid_ratio < 0.5:
                suggestions.append("信号有效率低于50%，建议优化信号检测算法，增加更多过滤条件")
            elif valid_ratio < 0.7:
                suggestions.append("信号有效率中等，可考虑调整参数阈值提高准确性")
            else:
                suggestions.append("信号有效率较高，系统表现良好")
        
        # 分析失效原因
        expiry_reasons = {}
        for signal in validated_signals:
            if signal.get('expired', False):
                reason = signal.get('expiry_reason', '未知')
                expiry_reasons[reason] = expiry_reasons.get(reason, 0) + 1
        
        if expiry_reasons:
            # 找出最主要的失效原因
            most_common_reason = max(expiry_reasons.items(), key=lambda x: x[1])
            suggestions.append(f"主要失效原因: '{most_common_reason[0]}' (发生{most_common_reason[1]}次)，建议针对此原因优化验证逻辑")
        
        # 分析信号强度分布
        strengths = [s.get('信号强度', 0) for s in signals]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0
        
        if avg_strength < 60:
            suggestions.append(f"平均信号强度较低 ({avg_strength:.2f})，建议调整算法参数提高信号的确定性")
        
        # 提供通用建议
        suggestions.append("建议增加更多技术指标作为辅助验证，如成交量分析、波动率分析等")
        suggestions.append("考虑引入机器学习模型，根据历史信号效果动态调整参数")
        suggestions.append("定期回测并优化系统参数，适应市场变化")
        
        return suggestions
    
    def run_full_test(self):
        """运行完整测试流程"""
        logger.info("开始军工ETF交易信号检测系统完整测试")
        
        try:
            # 1. 获取历史数据
            historical_data = self.get_historical_data()
            
            # 2. 模拟未来数据
            future_data = self.simulate_future_data(historical_data)
            
            # 3. 运行信号检测
            detected_signals = self.run_signal_detection(future_data)
            
            # 4. 验证检测到的信号
            validated_signals = self.validate_detected_signals(detected_signals, future_data)
            
            # 5. 可视化结果
            self.visualize_results(future_data, detected_signals)
            
            # 6. 生成测试报告
            report = self.generate_test_report(detected_signals, validated_signals)
            
            logger.info("军工ETF交易信号检测系统测试完成！")
            logger.info(f"测试报告保存目录: {self.output_dir}")
            
            # 打印测试摘要
            print("\n===== 测试完成摘要 =====")
            print(f"目标: 军工ETF(512660) {self.target_year}年{self.target_month}月后信号检测")
            print(f"检测到信号数量: {len(detected_signals)}")
            print(f"信号有效率: {report['validation_results']['valid_signals_ratio']}%")
            print(f"详细报告已保存至: {self.output_dir}")
            print("=====================\n")
            
            return report
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {str(e)}")
            print(f"测试失败: {str(e)}")
            raise

def main():
    """主函数"""
    print("\n=== 军工ETF(512660)交易信号验证测试 ===")
    print(f"开始模拟{datetime.now().year + 1}年信号检测测试...\n")
    
    # 创建测试实例
    tester = MilitaryETFTest()
    
    # 运行完整测试
    tester.run_full_test()
    
    print("测试执行完毕！")

if __name__ == "__main__":
    main()