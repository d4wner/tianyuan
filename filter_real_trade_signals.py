#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选2025年512690需要真实下单的信号脚本

基于周线趋势和信号类型优先级筛选真实有效的交易信号
作者: TradeTianYuan
日期: 2025-11-29
"""

import os
import json
import pandas as pd
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTradeSignalFilter")

class RealTradeSignalFilter:
    """真实交易信号过滤器"""
    
    def __init__(self):
        """初始化过滤器"""
        self.results_dir = "results"
        self.output_file = os.path.join(self.results_dir, f"512690_real_trade_signals_2025_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.all_signals = []
        self.real_trade_signals = []
        
    def load_latest_signals(self):
        """加载最新的信号数据文件
        
        Returns:
            bool: 是否成功加载
        """
        try:
            # 查找最新的512690信号数据文件
            signal_files = [f for f in os.listdir(self.results_dir) 
                          if f.startswith("512690_daily_signals_2025_") and f.endswith(".json")]
            
            if not signal_files:
                logger.error("未找到512690信号数据文件")
                return False
            
            # 按时间戳排序，获取最新的文件
            signal_files.sort(reverse=True)
            latest_file = os.path.join(self.results_dir, signal_files[0])
            
            logger.info(f"加载最新信号文件: {latest_file}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                self.all_signals = json.load(f)
            
            logger.info(f"共加载 {len(self.all_signals)} 个信号")
            return True
            
        except Exception as e:
            logger.error(f"加载信号文件失败: {str(e)}")
            return False
    
    def filter_real_trade_signals(self):
        """筛选需要真实下单的信号
        
        筛选规则：
        1. 优先选择周线多头趋势下的信号
        2. 根据信号类型分配优先级
        3. 避免连续信号的重复下单
        """
        if not self.all_signals:
            logger.error("无信号数据，无法筛选")
            return False
        
        logger.info("开始筛选真实交易信号...")
        
        # 按日期排序
        sorted_signals = sorted(self.all_signals, key=lambda x: x['date'])
        
        # 记录上一个交易信号的日期，避免连续下单
        last_trade_date = None
        
        for signal in sorted_signals:
            # 1. 检查是否为周线多头趋势
            is_weekly_bullish = "多头" in signal['weekly_trend']
            
            # 2. 信号类型优先级检查（目前主要是破中枢反抽）
            signal_type = signal['signal_type']
            signal_strength = signal['signal_strength']
            
            # 3. 避免连续日期重复下单（至少间隔1个交易日）
            current_date = signal['date']
            skip_trade = False
            
            if last_trade_date:
                # 计算日期差
                last_dt = datetime.strptime(last_trade_date, '%Y-%m-%d')
                current_dt = datetime.strptime(current_date, '%Y-%m-%d')
                days_diff = (current_dt - last_dt).days
                
                # 如果是连续交易日，跳过
                if days_diff <= 1:
                    skip_trade = True
            
            # 4. 确定是否为真实交易信号
            # 核心逻辑：周线多头趋势下的信号，或者在非多头趋势下但强度较高的信号
            is_real_trade = False
            
            if is_weekly_bullish:
                is_real_trade = True
                logger.info(f"周线多头趋势下的交易信号: {current_date} - {signal_type} (强度: {signal_strength})")
            elif signal_strength >= 70:
                # 在非多头趋势下，只有强度>=70的信号才考虑
                is_real_trade = True
                logger.info(f"高强度信号（非多头趋势）: {current_date} - {signal_type} (强度: {signal_strength})")
            
            # 如果是真实交易信号且不连续，则添加到列表
            if is_real_trade and not skip_trade:
                # 添加交易建议
                signal['trade_recommendation'] = self._generate_trade_recommendation(signal)
                self.real_trade_signals.append(signal)
                last_trade_date = current_date
        
        logger.info(f"共筛选出 {len(self.real_trade_signals)} 个需要真实下单的信号")
        return True
    
    def _generate_trade_recommendation(self, signal):
        """根据信号生成交易建议
        
        Args:
            signal: 信号字典
            
        Returns:
            str: 交易建议
        """
        is_weekly_bullish = "多头" in signal['weekly_trend']
        signal_type = signal['signal_type']
        
        # 仓位建议（根据信号类型和周线趋势）
        if is_weekly_bullish:
            if signal_type == "日线二买":
                position = "60%-70%"
                timing = "日线二买确认，优先匹配30分钟向上笔"
            elif signal_type == "日线一买" or signal_type == "日线三买":
                position = "20%-40%"
                timing = "可考虑15分钟向上笔作为建仓时机"
            else:  # 破中枢反抽
                position = "10%-20%"
                timing = "严格监控风险，准备止损"
        else:
            # 非多头趋势下，即使有信号也只建议小仓位试探
            position = "5%-10%"
            timing = "极小仓位试探，严格设置止损"
        
        return f"建议仓位: {position}, {timing}"
    
    def generate_real_trade_report(self):
        """生成真实交易信号报告
        
        Returns:
            str: 报告内容
        """
        if not self.real_trade_signals:
            return "未找到需要真实下单的信号"
        
        report_lines = []
        report_lines.append("===== 512690（酒ETF）2025年真实交易信号报告 =====")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总共筛选出 {len(self.real_trade_signals)} 个需要真实下单的信号")
        report_lines.append("")
        
        # 信号详情表格
        report_lines.append("【真实交易信号详情】")
        report_lines.append("-" * 120)
        report_lines.append(f"{'日期':<12} {'信号类型':<10} {'价格':<10} {'强度':<6} {'周线趋势':<20} {'交易建议':<40}")
        report_lines.append("-" * 120)
        
        total_trades = 0
        bullish_trend_trades = 0
        
        for signal in sorted(self.real_trade_signals, key=lambda x: x['date']):
            report_lines.append(f"{signal['date']:<12} {signal['signal_type']:<10} {signal['close_price']:<10.4f} "
                              f"{signal['signal_strength']:<6} {signal['weekly_trend']:<20} {signal['trade_recommendation']:<40}")
            total_trades += 1
            if "多头" in signal['weekly_trend']:
                bullish_trend_trades += 1
        
        report_lines.append("-" * 120)
        report_lines.append("")
        
        # 交易统计
        report_lines.append("【交易统计】")
        report_lines.append(f"总交易次数: {total_trades}")
        report_lines.append(f"周线多头趋势下的交易次数: {bullish_trend_trades}")
        report_lines.append(f"非多头趋势下的交易次数: {total_trades - bullish_trend_trades}")
        report_lines.append("")
        
        # 交易策略建议
        report_lines.append("【交易策略建议】")
        report_lines.append("1. 严格遵循周线趋势：周线多头趋势是加仓的主要依据")
        report_lines.append("2. 仓位管理：根据信号类型和周线趋势合理控制仓位")
        report_lines.append("3. 执行时机：日线信号确认后，寻找分钟级别的买点执行")
        report_lines.append("4. 风险控制：所有交易必须设置止损，特别是非多头趋势下的交易")
        report_lines.append("5. 避免频繁交易：同一方向的信号至少间隔1个交易日")
        report_lines.append("")
        
        # 风险提示
        report_lines.append("【风险提示】")
        report_lines.append("1. 本报告基于量化模型分析，仅供参考")
        report_lines.append("2. 市场有风险，投资需谨慎")
        report_lines.append("3. 请结合自身风险承受能力进行交易决策")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"真实交易信号报告已保存至: {self.output_file}")
        
        return report_content
    
    def run(self):
        """运行完整的筛选流程
        
        Returns:
            bool: 是否成功
        """
        logger.info("开始筛选512690真实交易信号...")
        
        try:
            # 1. 加载最新信号
            if not self.load_latest_signals():
                return False
            
            # 2. 筛选真实交易信号
            if not self.filter_real_trade_signals():
                return False
            
            # 3. 生成报告
            report = self.generate_real_trade_report()
            
            # 打印报告
            print("\n" + "="*80)
            print(report)
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"筛选过程中发生错误: {str(e)}")
            return False

if __name__ == "__main__":
    filter = RealTradeSignalFilter()
    success = filter.run()
    
    if success:
        logger.info("真实交易信号筛选完成！")
    else:
        logger.error("真实交易信号筛选失败！")