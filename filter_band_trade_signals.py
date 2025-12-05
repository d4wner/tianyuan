#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波段交易策略专用的512690信号筛选脚本

针对波段交易特性（几周或几个月一次）筛选真实有效的交易信号
作者: TradeTianYuan
日期: 2025-11-29
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BandTradeSignalFilter")

class BandTradeSignalFilter:
    """波段交易信号过滤器"""
    
    def __init__(self):
        """初始化过滤器"""
        self.results_dir = "results"
        self.output_file = os.path.join(self.results_dir, f"512690_band_trade_signals_2025_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.all_signals = []
        self.band_trade_signals = []
        
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
    
    def filter_band_trade_signals(self):
        """筛选波段交易信号
        
        筛选规则：
        1. 保留周线多头趋势确立后的重要信号
        2. 设置合理的信号间隔时间（至少间隔2周）
        3. 优先选择周线多头趋势下的信号
        4. 适当放宽条件，确保筛选出合理数量的波段交易信号
        """
        if not self.all_signals:
            logger.error("无信号数据，无法筛选")
            return False
        
        logger.info("开始筛选波段交易信号...")
        
        # 按日期排序
        sorted_signals = sorted(self.all_signals, key=lambda x: x['date'])
        
        # 记录上一个波段交易信号的日期
        last_trade_date = None
        
        # 按月份统计信号，每个月最多选择1-2个重要信号
        monthly_signals = {}
        
        for signal in sorted_signals:
            current_date = signal['date']
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            monthly_key = current_dt.strftime('%Y-%m')
            weekly_trend = signal['weekly_trend']
            price = signal['close_price']
            
            # 判断是否需要跳过（间隔太短）
            skip_trade = False
            if last_trade_date:
                last_dt = datetime.strptime(last_trade_date, '%Y-%m-%d')
                weeks_diff = (current_dt - last_dt).days / 7
                
                # 波段交易至少间隔2周（放宽条件）
                if weeks_diff < 2:
                    skip_trade = True
            
            # 按月份统计，限制每个月的信号数量
            if monthly_key not in monthly_signals:
                monthly_signals[monthly_key] = []
            
            # 判断是否为波段交易信号
            is_band_trade = False
            
            # 1. 周线多头趋势（包括高置信度和疑似多头）
            is_weekly_bullish = "多头" in weekly_trend
            
            # 2. 简单的相对低位判断（前15个交易日的最低价附近）
            is_low_price = self._simplified_low_price_check(price, sorted_signals, current_date)
            
            # 波段交易信号判定条件（满足任一条件即可）
            if not skip_trade and len(monthly_signals[monthly_key]) < 2:  # 每月最多2个信号
                # 优先选择周线多头趋势下的信号
                if is_weekly_bullish:
                    is_band_trade = True
                    logger.info(f"波段交易信号（多头趋势）: {current_date} - {weekly_trend}")
                # 或者选择价格相对较低的信号
                elif is_low_price:
                    is_band_trade = True
                    logger.info(f"波段交易信号（低位）: {current_date} - 价格: {price}")
                # 如果当月还没有信号，且价格处于低位，也考虑
                elif len(monthly_signals[monthly_key]) == 0 and self._is_monthly_low(price, sorted_signals, monthly_key):
                    is_band_trade = True
                    logger.info(f"波段交易信号（月内低位）: {current_date} - 价格: {price}")
            
            # 如果是波段交易信号，则添加到列表
            if is_band_trade:
                # 添加交易建议
                signal['trade_recommendation'] = self._generate_band_trade_recommendation(signal)
                self.band_trade_signals.append(signal)
                last_trade_date = current_date
                # 添加到月度信号统计
                monthly_signals[monthly_key].append(signal)
        
        logger.info(f"共筛选出 {len(self.band_trade_signals)} 个波段交易信号")
        return True
    
    def _simplified_low_price_check(self, current_price, all_signals, current_date):
        """简化的相对低位判断
        
        Args:
            current_price: 当前价格
            all_signals: 所有信号列表
            current_date: 当前日期
            
        Returns:
            bool: 是否为相对低位
        """
        # 获取当前日期前15个交易日的价格数据
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        start_date = current_dt - timedelta(days=15)
        
        # 收集时间范围内的价格
        prices_in_range = []
        for signal in all_signals:
            signal_dt = datetime.strptime(signal['date'], '%Y-%m-%d')
            if start_date <= signal_dt <= current_dt:
                prices_in_range.append(signal['close_price'])
        
        if not prices_in_range:
            return False
        
        # 如果当前价格接近近期最低价（1%以内），则认为是相对低位
        min_price = min(prices_in_range)
        return current_price <= min_price * 1.01
        
    def _is_monthly_low(self, current_price, all_signals, monthly_key):
        """判断是否为月内相对低位
        
        Args:
            current_price: 当前价格
            all_signals: 所有信号列表
            monthly_key: 月份键（如'2025-03'）
            
        Returns:
            bool: 是否为月内低位
        """
        # 收集当月的价格
        monthly_prices = []
        for signal in all_signals:
            signal_month = signal['date'][:7]  # 获取YYYY-MM格式
            if signal_month == monthly_key:
                monthly_prices.append(signal['close_price'])
        
        if not monthly_prices:
            return False
        
        # 如果当前价格在月内价格的前30%，则认为是相对低位
        monthly_prices.sort()
        percentile_30 = monthly_prices[int(len(monthly_prices) * 0.3)]
        
        return current_price <= percentile_30
    
    def _generate_band_trade_recommendation(self, signal):
        """生成波段交易建议
        
        Args:
            signal: 信号字典
            
        Returns:
            str: 交易建议
        """
        is_weekly_bullish = "周线多头" in signal['weekly_trend']
        signal_type = signal['signal_type']
        
        # 波段交易仓位建议
        if is_weekly_bullish:
            if "高置信度" in signal['weekly_trend']:
                position = "40%-60%"
                timing = "周线多头确立，可分批建仓"
                duration = "预期持有周期：1-3个月"
            else:
                position = "20%-30%"
                timing = "疑似多头，小仓位试探"
                duration = "预期持有周期：2-4周"
        else:
            position = "10%-20%"
            timing = "重要支撑位试探性建仓"
            duration = "预期持有周期：1-2周"
        
        return f"波段交易-建议仓位: {position}, {timing}, {duration}"
    
    def generate_band_trade_report(self):
        """生成波段交易信号报告
        
        Returns:
            str: 报告内容
        """
        if not self.band_trade_signals:
            return "未找到波段交易信号"
        
        report_lines = []
        report_lines.append("===== 512690（酒ETF）2025年波段交易信号报告 =====")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总共筛选出 {len(self.band_trade_signals)} 个波段交易信号")
        report_lines.append("")
        
        # 信号详情表格
        report_lines.append("【波段交易信号详情】")
        report_lines.append("-" * 120)
        report_lines.append(f"{'日期':<12} {'信号类型':<10} {'价格':<10} {'周线趋势':<20} {'交易建议':<50}")
        report_lines.append("-" * 120)
        
        for signal in sorted(self.band_trade_signals, key=lambda x: x['date']):
            report_lines.append(f"{signal['date']:<12} {signal['signal_type']:<10} {signal['close_price']:<10.4f} "
                              f"{signal['weekly_trend']:<20} {signal['trade_recommendation']:<50}")
        
        report_lines.append("-" * 120)
        report_lines.append("")
        
        # 波段交易策略建议
        report_lines.append("【波段交易策略建议】")
        report_lines.append("1. 波段交易核心：捕捉周线级别的趋势变化，避免频繁交易")
        report_lines.append("2. 仓位管理：根据周线趋势强度调整仓位，多头趋势可加大仓位")
        report_lines.append("3. 持有周期：通常为2-8周，避免日内或日线级别的频繁进出")
        report_lines.append("4. 止盈止损：设置较大止损（如8%-12%），对应波段波动特点")
        report_lines.append("5. 信号过滤：严格控制交易频率，确保每次交易间隔充分")
        report_lines.append("")
        
        # 风险提示
        report_lines.append("【风险提示】")
        report_lines.append("1. 本报告基于量化模型分析，仅供参考")
        report_lines.append("2. 波段交易仍存在市场风险，请合理控制仓位")
        report_lines.append("3. 请结合自身风险承受能力和资金管理计划进行交易决策")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"波段交易信号报告已保存至: {self.output_file}")
        
        return report_content
    
    def run(self):
        """运行完整的筛选流程
        
        Returns:
            bool: 是否成功
        """
        logger.info("开始筛选512690波段交易信号...")
        
        try:
            # 1. 加载最新信号
            if not self.load_latest_signals():
                return False
            
            # 2. 筛选波段交易信号
            if not self.filter_band_trade_signals():
                return False
            
            # 3. 生成报告
            report = self.generate_band_trade_report()
            
            # 打印报告
            print("\n" + "="*80)
            print(report)
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"筛选过程中发生错误: {str(e)}")
            return False

if __name__ == "__main__":
    filter = BandTradeSignalFilter()
    success = filter.run()
    
    if success:
        logger.info("波段交易信号筛选完成！")
    else:
        logger.error("波段交易信号筛选失败！")