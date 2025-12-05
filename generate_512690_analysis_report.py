#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512690完整分析报告生成器
整合MACD背驰分析、买卖信号分析和缠论验证结果，生成综合分析报告
"""

import os
import json
import datetime
import pandas as pd
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('512690ReportGenerator')

class ReportGenerator:
    """
    分析报告生成器类
    负责整合各种分析结果并生成综合报告
    """
    
    def __init__(self, symbol="512690", results_dir="./results", data_dir="./data"):
        """
        初始化报告生成器
        
        Args:
            symbol (str): 股票代码
            results_dir (str): 分析结果存储目录
            data_dir (str): 原始数据存储目录
        """
        self.symbol = symbol
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.report_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = f"{results_dir}/{symbol}_comprehensive_report_{self.report_time}.txt"
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 存储各分析结果
        self.macd_results = None
        self.signal_results = None
        self.chanlun_results = None
        self.latest_price = None
        
    def load_macd_results(self):
        """加载MACD背驰分析结果"""
        try:
            # 查找最新的MACD分析结果文件
            macd_files = [f for f in os.listdir(self.results_dir) 
                         if f.startswith(f"{self.symbol}_macd_divergence_results") and f.endswith(".json")]
            if macd_files:
                latest_file = sorted(macd_files)[-1]  # 获取最新的文件
                file_path = os.path.join(self.results_dir, latest_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.macd_results = json.load(f)
                logger.info(f"已加载MACD背驰分析结果: {latest_file}")
            else:
                logger.warning("未找到MACD背驰分析结果文件")
        except Exception as e:
            logger.error(f"加载MACD背驰分析结果失败: {str(e)}")
    
    def load_signal_results(self):
        """加载买卖信号分析结果"""
        try:
            # 查找最新的信号分析结果文件
            signal_files = [f for f in os.listdir(self.results_dir) 
                          if f.startswith(f"{self.symbol}_signals_") and f.endswith(".json")]
            if signal_files:
                latest_file = sorted(signal_files)[-1]  # 获取最新的文件
                file_path = os.path.join(self.results_dir, latest_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.signal_results = json.load(f)
                logger.info(f"已加载买卖信号分析结果: {latest_file}")
            else:
                logger.warning("未找到买卖信号分析结果文件")
        except Exception as e:
            logger.error(f"加载买卖信号分析结果失败: {str(e)}")
    
    def load_chanlun_results(self):
        """加载缠论验证结果"""
        try:
            # 查找最新的缠论验证结果文件
            chanlun_files = [f for f in os.listdir(self.results_dir) 
                            if f.startswith(f"{self.symbol}_chanlun_verification") and f.endswith(".txt")]
            if chanlun_files:
                latest_file = sorted(chanlun_files)[-1]  # 获取最新的文件
                self.chanlun_results = latest_file
                logger.info(f"已找到缠论验证报告: {latest_file}")
        except Exception as e:
            logger.error(f"加载缠论验证结果失败: {str(e)}")
    
    def load_latest_price(self):
        """加载最新价格数据"""
        try:
            daily_data_path = os.path.join(self.data_dir, "daily", f"{self.symbol}_daily.csv")
            if os.path.exists(daily_data_path):
                df = pd.read_csv(daily_data_path)
                if not df.empty:
                    # 假设日期列是第一列，收盘价列是'close'或类似名称
                    # 根据实际CSV格式调整
                    date_columns = [col for col in df.columns if 'date' in col.lower()]
                    if not date_columns:
                        date_columns = [df.columns[0]]  # 假设第一列是日期
                    
                    close_columns = [col for col in df.columns if 'close' in col.lower()]
                    if not close_columns:
                        close_columns = [df.columns[4]]  # 假设第5列是收盘价
                    
                    df = df.sort_values(by=date_columns[0], ascending=False)
                    self.latest_price = df.iloc[0][close_columns[0]]
                    logger.info(f"已加载最新价格: {self.latest_price}")
        except Exception as e:
            logger.error(f"加载最新价格失败: {str(e)}")
    
    def generate_report(self):
        """生成综合分析报告"""
        try:
            # 加载所有分析结果
            self.load_macd_results()
            self.load_signal_results()
            self.load_chanlun_results()
            self.load_latest_price()
            
            # 生成报告
            with open(self.report_file, 'w', encoding='utf-8') as f:
                # 报告标题和基本信息
                f.write("=" * 80 + "\n")
                f.write(f"{self.symbol} 综合分析报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.latest_price:
                    f.write(f"最新价格: {self.latest_price}\n\n")
                
                # MACD背驰分析摘要
                f.write("\n" + "-" * 80 + "\n")
                f.write("1. MACD背驰分析摘要\n")
                f.write("-" * 80 + "\n")
                if self.macd_results:
                    # 更灵活地处理MACD结果结构
                    bottom_divergences = self.macd_results.get("bottom_divergences", [])
                    top_divergences = self.macd_results.get("top_divergences", [])
                    current_trend = self.macd_results.get("current_trend", {})
                    
                    f.write(f"底背驰信号数量: {len(bottom_divergences)}\n")
                    f.write(f"顶背驰信号数量: {len(top_divergences)}\n")
                    
                    # 显示最新的背驰信号（添加错误处理）
                    if bottom_divergences:
                        try:
                            # 尝试使用不同的日期键名
                            for date_key in ["end_date", "date", "time", "datetime"]:
                                if all(date_key in item for item in bottom_divergences):
                                    latest_bottom = sorted(bottom_divergences, key=lambda x: x[date_key], reverse=True)[0]
                                    break
                            else:
                                # 如果没有找到日期键，使用第一个元素
                                latest_bottom = bottom_divergences[0]
                            
                            f.write(f"\n最新底背驰信号:\n")
                            f.write(f"  强度: {latest_bottom.get('strength', 0):.2f}/100\n")
                            
                            # 尝试获取日期信息
                            start_date = latest_bottom.get('start_date', latest_bottom.get('date', ''))
                            end_date = latest_bottom.get('end_date', latest_bottom.get('date', ''))
                            if start_date and end_date and start_date != end_date:
                                f.write(f"  期间: {start_date} 至 {end_date}\n")
                            else:
                                f.write(f"  日期: {start_date or end_date}\n")
                            
                            f.write(f"  描述: {latest_bottom.get('description', '')}\n")
                        except Exception:
                            f.write("  无法解析底背驰信号详细信息\n")
                    
                    if top_divergences:
                        try:
                            # 尝试使用不同的日期键名
                            for date_key in ["end_date", "date", "time", "datetime"]:
                                if all(date_key in item for item in top_divergences):
                                    latest_top = sorted(top_divergences, key=lambda x: x[date_key], reverse=True)[0]
                                    break
                            else:
                                # 如果没有找到日期键，使用第一个元素
                                latest_top = top_divergences[0]
                            
                            f.write(f"\n最新顶背驰信号:\n")
                            f.write(f"  强度: {latest_top.get('strength', 0):.2f}/100\n")
                            
                            # 尝试获取日期信息
                            start_date = latest_top.get('start_date', latest_top.get('date', ''))
                            end_date = latest_top.get('end_date', latest_top.get('date', ''))
                            if start_date and end_date and start_date != end_date:
                                f.write(f"  期间: {start_date} 至 {end_date}\n")
                            else:
                                f.write(f"  日期: {start_date or end_date}\n")
                            
                            f.write(f"  描述: {latest_top.get('description', '')}\n")
                        except Exception:
                            f.write("  无法解析顶背驰信号详细信息\n")
                    
                    # 当前MACD趋势
                    f.write(f"\n当前MACD趋势:\n")
                    f.write(f"  方向: {'向上' if current_trend.get('direction', 'down') == 'up' else '向下'}\n")
                    f.write(f"  MACD柱值: {current_trend.get('histogram_value', 0):.6f}\n")
                    f.write(f"  趋势强度: {current_trend.get('strength', '弱')}\n")
                    f.write(f"  操作建议: {current_trend.get('suggestion', '观望')}\n")
                else:
                    f.write("  未找到MACD背驰分析结果\n")
                
                # 买卖信号分析摘要
                f.write("\n" + "-" * 80 + "\n")
                f.write("2. 买卖信号分析摘要\n")
                f.write("-" * 80 + "\n")
                if self.signal_results:
                    try:
                        # 灵活处理信号数据结构 - 可能是列表或字典
                        if isinstance(self.signal_results, dict):
                            signals = self.signal_results.get("signals", [])
                        elif isinstance(self.signal_results, list):
                            signals = self.signal_results
                        else:
                            signals = []
                        
                        # 过滤有效的信号（具有必要字段）
                        valid_signals = []
                        for s in signals:
                            if isinstance(s, dict) and 'type' in s:
                                valid_signals.append(s)
                        
                        if not valid_signals:
                            f.write("  未找到有效的信号数据\n")
                        else:
                            buy_signals = [s for s in valid_signals if s.get("type", "").lower() == "buy"]
                            sell_signals = [s for s in valid_signals if s.get("type", "").lower() == "sell"]
                            
                            f.write(f"信号总数: {len(valid_signals)}\n")
                            f.write(f"买入信号: {len(buy_signals)}\n")
                            f.write(f"卖出信号: {len(sell_signals)}\n")
                            
                            # 显示最近的几个信号（如果有日期字段）
                            try:
                                # 找出含有日期信息的信号并排序
                                dated_signals = [s for s in valid_signals if any(k in s for k in ['date', 'time', 'datetime'])]
                                for date_key in ['date', 'time', 'datetime']:
                                    if any(date_key in s for s in dated_signals):
                                        recent_signals = sorted(dated_signals, key=lambda x: x.get(date_key, ''), reverse=True)[:5]
                                        break
                                else:
                                    # 如果没有日期信息，只显示前5个
                                    recent_signals = valid_signals[:5]
                                
                                f.write(f"\n最近5个信号:\n")
                                for signal in recent_signals:
                                    # 获取日期信息
                                    date = signal.get('date', signal.get('time', signal.get('datetime', '')))
                                    signal_type = signal.get('type', '未知')
                                    description = signal.get('description', '')
                                    f.write(f"  {date} - {signal_type} - {description}\n")
                                
                                # 显示最近信号类型
                                if recent_signals:
                                    latest_signal = recent_signals[0]
                                    latest_date = latest_signal.get('date', latest_signal.get('time', latest_signal.get('datetime', '')))
                                    latest_type = latest_signal.get('type', '未知')
                                    latest_desc = latest_signal.get('description', '')
                                    f.write(f"\n最新信号: {latest_date} - {latest_type}\n")
                                    f.write(f"信号描述: {latest_desc}\n")
                            except Exception:
                                f.write("  无法显示最近信号详情\n")
                    except Exception as e:
                        logger.error(f"处理买卖信号数据时出错: {str(e)}")
                        f.write("  处理信号数据时出错\n")
                else:
                    f.write("  未找到买卖信号分析结果\n")
                
                # 缠论分析摘要
                f.write("\n" + "-" * 80 + "\n")
                f.write("3. 缠论分析摘要\n")
                f.write("-" * 80 + "\n")
                if self.chanlun_results:
                    f.write(f"缠论验证报告: {self.chanlun_results}\n")
                    # 尝试从缠论验证报告中提取关键信息
                    chanlun_report_path = os.path.join(self.results_dir, self.chanlun_results)
                    try:
                        with open(chanlun_report_path, 'r', encoding='utf-8') as cf:
                            chanlun_content = cf.read()
                            
                            # 提取分型数量
                            if "分型数量" in chanlun_content:
                                for line in chanlun_content.split('\n'):
                                    if "分型数量" in line:
                                        f.write(f"  {line.strip()}\n")
                                        break
                            
                            # 提取笔划分信息
                            if "笔总数" in chanlun_content:
                                for line in chanlun_content.split('\n'):
                                    if "笔总数" in line:
                                        f.write(f"  {line.strip()}\n")
                                        break
                            
                            # 提取最近的笔划分
                            if "最近的笔划分" in chanlun_content:
                                f.write("  最近的笔划分信息:\n")
                                capture = False
                                count = 0
                                for line in chanlun_content.split('\n'):
                                    if "最近的笔划分" in line:
                                        capture = True
                                    elif capture and count < 5:  # 只提取前几行有用信息
                                        if line.strip():  # 忽略空行
                                            f.write(f"    {line.strip()}\n")
                                            count += 1
                                    elif capture and count >= 5:
                                        break
                        
                    except Exception as e:
                        logger.error(f"读取缠论验证报告失败: {str(e)}")
                else:
                    f.write("  未找到缠论验证结果\n")
                
                # 综合结论和建议
                f.write("\n" + "-" * 80 + "\n")
                f.write("4. 综合结论与投资建议\n")
                f.write("-" * 80 + "\n")
                
                # 基于各分析结果生成综合建议
                recommendations = []
                
                # MACD建议
                if self.macd_results and "current_trend" in self.macd_results:
                    macd_suggestion = self.macd_results["current_trend"].get("suggestion", "")
                    if macd_suggestion:
                        recommendations.append(f"MACD分析建议: {macd_suggestion}")
                
                # 信号建议
                if self.signal_results and "signals" in self.signal_results and self.signal_results["signals"]:
                    latest_signal = sorted(self.signal_results["signals"], key=lambda x: x["date"], reverse=True)[0]
                    recommendations.append(f"最近信号({latest_signal['date']}): {latest_signal['type']} - {latest_signal['description']}")
                
                # 缠论建议
                # 从缠论报告中提取建议
                if self.chanlun_results:
                    chanlun_report_path = os.path.join(self.results_dir, self.chanlun_results)
                    try:
                        with open(chanlun_report_path, 'r', encoding='utf-8') as cf:
                            chanlun_content = cf.read()
                            if "交易建议" in chanlun_content:
                                for line in chanlun_content.split('\n'):
                                    if "交易建议" in line or "注意风险" in line:
                                        recommendations.append(f"缠论建议: {line.strip().replace('📉', '').replace('🎯 交易建议:', '').strip()}")
                    except Exception:
                        pass
                
                # 写入建议
                if recommendations:
                    for rec in recommendations:
                        f.write(f"{rec}\n")
                else:
                    f.write("根据现有分析数据不足，建议结合更多指标进行判断\n")
                
                # 综合风险提示
                f.write("\n" + "-" * 80 + "\n")
                f.write("5. 风险提示\n")
                f.write("-" * 80 + "\n")
                f.write("1. 本报告基于历史数据和技术分析，不构成投资建议\n")
                f.write("2. 市场存在不确定性，实际走势可能与分析结果不符\n")
                f.write("3. 投资有风险，入市需谨慎\n")
                f.write("4. 请结合个人风险偏好和资金状况做出投资决策\n")
                
            logger.info(f"综合分析报告已生成: {self.report_file}")
            return self.report_file
            
        except Exception as e:
            logger.error(f"生成综合分析报告失败: {str(e)}")
            raise

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成ETF综合分析报告")
    parser.add_argument("--symbol", type=str, default="512690", help="ETF代码，默认512690")
    parser.add_argument("--results-dir", type=str, default="./results", help="分析结果存储目录")
    parser.add_argument("--data-dir", type=str, default="./data", help="原始数据存储目录")
    
    args = parser.parse_args()
    
    # 创建报告生成器
    generator = ReportGenerator(
        symbol=args.symbol,
        results_dir=args.results_dir,
        data_dir=args.data_dir
    )
    
    # 生成报告
    report_file = generator.generate_report()
    print(f"综合分析报告已成功生成: {report_file}")

if __name__ == "__main__":
    main()