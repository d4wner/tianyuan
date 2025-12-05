import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    """综合交易信号报告生成器"""
    
    def __init__(self, results_dir='results'):
        """
        初始化报告生成器
        
        Args:
            results_dir: 分析结果文件的目录
        """
        self.results_dir = results_dir
        self.all_analysis_results = {}
        self.report_content = []
    
    def find_latest_result_files(self):
        """
        查找最新的分析结果文件
        
        Returns:
            包含各分析结果文件路径的字典
        """
        if not os.path.exists(self.results_dir):
            logger.error(f"结果目录不存在: {self.results_dir}")
            return None
        
        # 文件模式和对应的分析类型
        file_patterns = {
            'weekly_trend': '512660_weekly_trend_analysis_',
            'daily_buy_signal': '512660_daily_buy_signal_analysis_',
            'macd_divergence': '512660_macd_divergence_analysis_'
        }
        
        latest_files = {}
        
        for analysis_type, pattern in file_patterns.items():
            matching_files = []
            for filename in os.listdir(self.results_dir):
                if filename.startswith(pattern) and filename.endswith('.json'):
                    filepath = os.path.join(self.results_dir, filename)
                    # 提取时间戳
                    try:
                        # 从文件名提取时间戳：格式为YYYYMMDD_HHMMSS
                        timestamp_str = filename[len(pattern):filename.rfind('.json')]
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        matching_files.append((filepath, timestamp))
                    except Exception as e:
                        logger.warning(f"无法解析文件{filename}的时间戳: {e}")
            
            if matching_files:
                # 按时间戳排序，获取最新的文件
                matching_files.sort(key=lambda x: x[1], reverse=True)
                latest_files[analysis_type] = matching_files[0][0]
                logger.info(f"找到最新的{analysis_type}分析文件: {matching_files[0][0]}")
            else:
                logger.warning(f"未找到{analysis_type}分析结果文件")
        
        return latest_files
    
    def load_analysis_results(self):
        """
        加载所有分析结果
        
        Returns:
            是否成功加载所有结果
        """
        latest_files = self.find_latest_result_files()
        
        if not latest_files:
            return False
        
        # 即使某些分析结果缺失，也尝试加载可用的结果
        for analysis_type, filepath in latest_files.items():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.all_analysis_results[analysis_type] = json.load(f)
                logger.info(f"成功加载{analysis_type}分析结果")
            except Exception as e:
                logger.error(f"加载{analysis_type}分析结果失败: {e}")
        
        # 如果没有任何分析结果，返回失败
        if not self.all_analysis_results:
            return False
        
        return True
    
    def generate_heading(self):
        """
        生成报告标题和基础信息
        """
        now = datetime.now()
        self.report_content.append("=" * 60)
        self.report_content.append("          512660 综合交易信号分析报告          ")
        self.report_content.append("=" * 60)
        self.report_content.append(f"生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.report_content.append(f"分析标的: 512660")
        
        # 从分析结果中提取最新交易日信息
        latest_date = None
        for results in self.all_analysis_results.values():
            if 'latest_date' in results:
                try:
                    result_date = datetime.strptime(results['latest_date'].split()[0], '%Y-%m-%d')
                    if latest_date is None or result_date > latest_date:
                        latest_date = result_date
                except:
                    pass
        
        if latest_date:
            self.report_content.append(f"分析截至: {latest_date.strftime('%Y-%m-%d')}")
        
        self.report_content.append("" * 2)
    
    def generate_summary(self):
        """
        生成分析结果摘要
        """
        self.report_content.append("【分析结果摘要】")
        self.report_content.append("-" * 30)
        
        # 周线趋势摘要
        if 'weekly_trend' in self.all_analysis_results:
            weekly = self.all_analysis_results['weekly_trend']
            trend_status = weekly.get('trend_status', '未知')
            trend_confirmed = '✓ 已确认' if trend_status == '多头趋势确认' else \
                            '⚠ 疑似多头' if trend_status == '疑似多头' else \
                            '✗ 未确认'
            self.report_content.append(f"周线多头趋势: {trend_confirmed}")
        else:
            self.report_content.append(f"周线多头趋势: ? 数据缺失")
        
        # 日线买点摘要
        if 'daily_buy_signal' in self.all_analysis_results:
            daily = self.all_analysis_results['daily_buy_signal']
            best_signal = daily.get('best_buy_signal', '未知')
            signal_status = '✓ 存在' if best_signal != '无买点' else '✗ 不存在'
            self.report_content.append(f"日线买点信号: {signal_status}")
            if best_signal != '无买点':
                self.report_content.append(f"  最强信号: {best_signal}")
        else:
            self.report_content.append(f"日线买点信号: ? 数据缺失")
        
        # MACD背驰摘要
        if 'macd_divergence' in self.all_analysis_results:
            macd = self.all_analysis_results['macd_divergence']
            has_divergence = macd.get('has_divergence', False)
            divergence_status = '✓ 存在' if has_divergence else '✗ 不存在'
            self.report_content.append(f"MACD背驰信号: {divergence_status}")
            if has_divergence and 'priority_divergence' in macd and macd['priority_divergence']:
                div_type = macd['priority_divergence'].get('type', '')
                div_strength = macd['priority_divergence'].get('strength', '')
                self.report_content.append(f"  主要背驰: {div_type}({div_strength})")
        else:
            self.report_content.append(f"MACD背驰信号: ? 数据缺失")
        
        self.report_content.append("" * 2)
    
    def generate_weekly_trend_details(self):
        """
        生成周线趋势详细分析
        """
        if 'weekly_trend' not in self.all_analysis_results:
            return
        
        weekly = self.all_analysis_results['weekly_trend']
        
        self.report_content.append("【周线多头趋势分析】")
        self.report_content.append("-" * 30)
        
        # 基本信息
        self.report_content.append(f"趋势状态: {weekly.get('trend_status', '未知')}")
        self.report_content.append(f"趋势确认度: {weekly.get('trend_confidence', '未知')}")
        
        # 条件满足情况
        conditions = weekly.get('conditions_met', {})
        self.report_content.append("条件满足情况:")
        
        # 收盘价逐步抬升
        price_rising = conditions.get('price_gradually_rising', False)
        self.report_content.append(f"  收盘价逐步抬升: {'✓ 满足' if price_rising else '✗ 不满足'}")
        
        # MACD黄白线在零轴上方
        macd_above_zero = conditions.get('macd_above_zero', False)
        self.report_content.append(f"  MACD黄白线在零轴上方: {'✓ 满足' if macd_above_zero else '✗ 不满足'}")
        
        # MACD红柱未连续缩小
        hist_not_decreasing = conditions.get('hist_not_decreasing', False)
        self.report_content.append(f"  MACD红柱未连续缩小: {'✓ 满足' if hist_not_decreasing else '✗ 不满足'}")
        
        # 满足条件数量
        total_conditions = 3
        met_count = sum([price_rising, macd_above_zero, hist_not_decreasing])
        self.report_content.append(f"  满足条件数量: {met_count}/{total_conditions}")
        
        # 额外分析（如果有）
        if 'additional_analysis' in weekly:
            additional = weekly['additional_analysis']
            self.report_content.append("\n额外分析:")
            if 'recent_returns' in additional:
                returns = additional['recent_returns']
                for period, value in returns.items():
                    self.report_content.append(f"  {period}涨跌幅: {value}%")
            
            if 'volatility' in additional:
                self.report_content.append(f"  26周波动率: {additional['volatility']}%")
            
            if 'ma_alignment' in additional:
                ma_status = '多头排列' if additional['ma_alignment'] else '非多头排列'
                self.report_content.append(f"  均线系统: {ma_status}")
        
        self.report_content.append("" * 2)
    
    def generate_daily_signal_details(self):
        """
        生成日线买点详细分析
        """
        if 'daily_buy_signal' not in self.all_analysis_results:
            return
        
        daily = self.all_analysis_results['daily_buy_signal']
        
        self.report_content.append("【日线买点信号分析】")
        self.report_content.append("-" * 30)
        
        # 基本信息
        self.report_content.append(f"最强买点信号: {daily.get('best_buy_signal', '未知')}")
        self.report_content.append(f"信号优先级: {daily.get('signal_priority', '未知')}")
        
        # 各买点信号状态
        signal_status = daily.get('signal_status', {})
        self.report_content.append("各买点信号状态:")
        
        signal_names = {
            '日线二买（核心）': 'daily_second_buy',
            '日线一买（辅助）': 'daily_first_buy',
            '日线三买（辅助）': 'daily_third_buy',
            '破中枢反抽（兜底）': 'break_central_rebound'
        }
        
        for display_name, key in signal_names.items():
            status = signal_status.get(key, False)
            self.report_content.append(f"  {display_name}: {'✓ 满足' if status else '✗ 不满足'}")
        
        # 技术指标分析（如果有）
        if 'technical_indicators' in daily:
            tech = daily['technical_indicators']
            self.report_content.append("\n技术指标分析:")
            
            if 'ma_status' in tech:
                self.report_content.append(f"  均线状态: {tech['ma_status']}")
            
            if 'price_position' in tech:
                self.report_content.append("  当前价格位置:")
                for ma, pos in tech['price_position'].items():
                    self.report_content.append(f"    - 相对{ma}: {pos}")
            
            if 'macd_status' in tech:
                self.report_content.append(f"  MACD状态: {tech['macd_status']}")
                if 'macd_position' in tech:
                    self.report_content.append("  MACD位置:")
                    for component, pos in tech['macd_position'].items():
                        self.report_content.append(f"    - {component}: {pos}")
            
            if 'rsi' in tech:
                self.report_content.append(f"  RSI: {tech['rsi']} ({tech.get('rsi_status', '正常')})")
            
            if 'volume_price_coordination' in tech:
                self.report_content.append(f"  量价配合度: {tech['volume_price_coordination']}%")
            
            if 'volume_trend' in tech:
                self.report_content.append(f"  成交量趋势: {tech['volume_trend']}")
        
        self.report_content.append("" * 2)
    
    def generate_macd_divergence_details(self):
        """
        生成MACD背驰详细分析
        """
        if 'macd_divergence' not in self.all_analysis_results:
            return
        
        macd = self.all_analysis_results['macd_divergence']
        
        self.report_content.append("【MACD背驰分析】")
        self.report_content.append("-" * 30)
        
        # MACD指标状态
        if 'macd_trend' in macd:
            macd_trend = macd['macd_trend']
            self.report_content.append(f"MACD趋势: {macd_trend.get('trend_status', '未知')}")
            self.report_content.append(f"MACD线: {macd_trend.get('latest_macd_line', '未知')} {'(零轴上方)' if macd_trend.get('macd_above_zero', False) else '(零轴下方)'}")
            self.report_content.append(f"信号线: {macd_trend.get('latest_signal_line', '未知')}")
            self.report_content.append(f"MACD柱状图: {macd_trend.get('latest_macd_hist', '未知')}")
            self.report_content.append(f"MACD线趋势: {'上升' if macd_trend.get('macd_line_increasing', False) else '下降'}")
            self.report_content.append(f"柱状图趋势: {'扩大' if macd_trend.get('hist_increasing', False) else '缩小'}")
        
        # 背驰检测结果
        self.report_content.append("\n背驰检测结果:")
        has_divergence = macd.get('has_divergence', False)
        self.report_content.append(f"  背驰信号: {'✓ 存在' if has_divergence else '✗ 不存在'}")
        
        if has_divergence and 'priority_divergence' in macd and macd['priority_divergence']:
            div = macd['priority_divergence']
            self.report_content.append(f"  主要背驰类型: {div.get('type', '')}")
            self.report_content.append(f"  背驰强度: {div.get('strength', '')} ({div.get('strength_score', '')})")
            
            if div.get('type') == '底部背驰':
                self.report_content.append(f"  第一个底部: {div.get('first_bottom_date', '')}, 价格: {div.get('first_bottom_price', '')}")
                self.report_content.append(f"  第二个底部: {div.get('second_bottom_date', '')}, 价格: {div.get('second_bottom_price', '')}")
            else:
                self.report_content.append(f"  第一个顶部: {div.get('first_top_date', '')}, 价格: {div.get('first_top_price', '')}")
                self.report_content.append(f"  第二个顶部: {div.get('second_top_date', '')}, 价格: {div.get('second_top_price', '')}")
            
            self.report_content.append(f"  价格变化: {div.get('price_diff_pct', '')}%")
            self.report_content.append(f"  MACD变化: {div.get('macd_diff_pct', '')}%")
        
        # 详细背驰分析
        self.report_content.append("\n详细背驰分析:")
        bottom = macd.get('bottom_divergence', {})
        top = macd.get('top_divergence', {})
        
        self.report_content.append(f"  底部背驰: {'✓ 存在' if bottom.get('has_divergence', False) else '✗ 不存在'}")
        if bottom.get('has_divergence', False):
            self.report_content.append(f"    强度: {bottom.get('strength', '')}")
            self.report_content.append(f"    详情: {bottom.get('message', '')}")
        
        self.report_content.append(f"  顶部背驰: {'✓ 存在' if top.get('has_divergence', False) else '✗ 不存在'}")
        if top.get('has_divergence', False):
            self.report_content.append(f"    强度: {top.get('strength', '')}")
            self.report_content.append(f"    详情: {top.get('message', '')}")
        
        self.report_content.append("" * 2)
    
    def generate_comprehensive_advice(self):
        """
        生成综合交易建议
        """
        self.report_content.append("【综合交易建议】")
        self.report_content.append("-" * 30)
        
        # 获取各分析结果
        weekly = self.all_analysis_results.get('weekly_trend', {})
        daily = self.all_analysis_results.get('daily_buy_signal', {})
        macd = self.all_analysis_results.get('macd_divergence', {})
        
        # 提取关键指标
        weekly_trend = weekly.get('trend_status', '未知')
        daily_signal = daily.get('best_buy_signal', '无买点')
        has_macd_divergence = macd.get('has_divergence', False)
        divergence_type = None
        if has_macd_divergence and 'priority_divergence' in macd and macd['priority_divergence']:
            divergence_type = macd['priority_divergence'].get('type', '')
        
        # 综合判定逻辑
        # 1. 周线多头趋势确认 + 日线买点存在 = 强烈买入信号
        # 2. 周线多头趋势确认 + 无日线买点 = 等待日线买点
        # 3. 周线疑似多头 + 日线买点存在 = 谨慎买入
        # 4. 周线未确认 + 日线买点存在 = 观望为主，小仓位试探
        # 5. 底部背驰存在 = 可能的反转信号，结合其他指标
        # 6. 顶部背驰存在 = 谨慎，考虑减仓
        # 7. 无明显信号 = 观望为主
        
        # 计算信号强度得分 (0-100)
        signal_score = 0
        
        # 周线趋势评分 (0-40分)
        if weekly_trend == '多头趋势确认':
            signal_score += 40
        elif weekly_trend == '疑似多头':
            signal_score += 20
        
        # 日线买点评分 (0-40分)
        if daily_signal == '日线二买（核心）':
            signal_score += 40
        elif daily_signal == '日线一买（辅助）' or daily_signal == '日线三买（辅助）':
            signal_score += 25
        elif daily_signal == '破中枢反抽（兜底）':
            signal_score += 15
        
        # MACD背驰评分 (-20 到 +20分)
        if has_macd_divergence:
            if divergence_type == '底部背驰':
                # 底部背驰加分
                div_strength = macd['priority_divergence'].get('strength_score', 50)
                signal_score += min(20, div_strength / 5)
            elif divergence_type == '顶部背驰':
                # 顶部背驰减分
                signal_score -= 20
        
        # 生成建议
        if signal_score >= 60:
            advice_level = "强烈建议买入"
            advice_detail = [
                "✓ 综合信号强烈看好",
                "  周线趋势明确，日线买点出现，技术面支撑充分",
                "  建议积极建仓或加仓",
                "  可考虑分批买入策略，设置合理止损"
            ]
        elif signal_score >= 40:
            advice_level = "建议买入"
            advice_detail = [
                "✓ 综合信号看好",
                "  技术面整体偏强，存在买入依据",
                "  建议适量买入，控制仓位在合理水平",
                "  设置止损保护资金安全"
            ]
        elif signal_score >= 20:
            advice_level = "谨慎买入"
            advice_detail = [
                "⚠ 信号强度一般",
                "  技术面存在一定机会，但不够明确",
                "  建议小仓位试探性买入",
                "  密切关注后续走势确认"
            ]
        elif signal_score >= 0:
            advice_level = "观望为主"
            advice_detail = [
                "? 信号不明确",
                "  技术面缺乏明显的买卖依据",
                "  建议保持观望，等待更明确的信号",
                "  可关注但暂不介入"
            ]
        else:
            advice_level = "建议谨慎，考虑减仓"
            advice_detail = [
                "✗ 存在风险信号",
                "  技术面出现不利迹象",
                "  建议已有持仓考虑减仓或止盈",
                "  空仓者继续观望"
            ]
        
        self.report_content.append(f"信号强度评分: {signal_score}/100")
        self.report_content.append(f"综合建议: {advice_level}")
        self.report_content.append("详细建议:")
        for detail in advice_detail:
            self.report_content.append(detail)
        
        # 特别提示
        self.report_content.append("\n特别提示:")
        special_notes = []
        
        if weekly_trend == '多头趋势确认':
            special_notes.append("✓ 周线多头趋势已确认，中长期看好")
        elif weekly_trend == '疑似多头':
            special_notes.append("⚠ 周线疑似多头，需进一步确认")
        else:
            special_notes.append("✗ 周线多头趋势未确认，需谨慎")
        
        if daily_signal != '无买点':
            special_notes.append(f"✓ 日线{daily_signal}信号出现，短期可能有机会")
        else:
            special_notes.append("✗ 暂无明确日线买点，短期观望为宜")
        
        if has_macd_divergence:
            if divergence_type == '底部背驰':
                special_notes.append("✓ 底部背驰信号出现，可能即将反转")
            else:
                special_notes.append("⚠ 顶部背驰信号出现，注意风险")
        
        # 添加均线系统提示
        if 'weekly_trend' in self.all_analysis_results and 'additional_analysis' in weekly:
            ma_alignment = weekly['additional_analysis'].get('ma_alignment', False)
            if ma_alignment:
                special_notes.append("✓ 均线多头排列，趋势较强")
            else:
                special_notes.append("⚠ 均线非多头排列，趋势有待确认")
        
        for note in special_notes:
            self.report_content.append(note)
        
        self.report_content.append("" * 2)
    
    def generate_risk_warnings(self):
        """
        生成风险提示
        """
        self.report_content.append("【风险提示】")
        self.report_content.append("-" * 30)
        
        warnings = [
            "1. 技术分析存在局限性，不构成投资建议",
            "2. 市场有风险，投资需谨慎",
            "3. 请结合自身风险承受能力制定投资策略",
            "4. 建议设置合理止损，控制单笔交易风险",
            "5. 市场环境变化快，请实时关注最新行情",
            "6. 本报告基于历史数据，不保证未来表现"
        ]
        
        for warning in warnings:
            self.report_content.append(warning)
        
        self.report_content.append("" * 2)
    
    def generate_complete_report(self):
        """
        生成完整报告
        
        Returns:
            完整报告文本
        """
        # 生成各个部分
        self.generate_heading()
        self.generate_summary()
        self.generate_weekly_trend_details()
        self.generate_daily_signal_details()
        self.generate_macd_divergence_details()
        self.generate_comprehensive_advice()
        self.generate_risk_warnings()
        
        # 添加结尾
        self.report_content.append("=" * 60)
        self.report_content.append("报告生成完毕，仅供参考")
        self.report_content.append("=" * 60)
        
        return '\n'.join(self.report_content)
    
    def save_report(self, report_text):
        """
        保存报告到文件
        
        Args:
            report_text: 报告文本
        """
        # 确保results目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.results_dir, f'512660_complete_trading_report_{timestamp}.txt')
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"综合交易信号报告已保存至: {report_file}")
        
        # 同时保存分析结果汇总
        summary_file = os.path.join(self.results_dir, f'512660_analysis_summary_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_results': self.all_analysis_results
            }, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"分析结果汇总已保存至: {summary_file}")
        
        return report_file

def main():
    """
    主函数，生成512660完整交易信号报告
    """
    try:
        logger.info("开始生成512660综合交易信号报告...")
        
        # 创建报告生成器
        generator = ComprehensiveReportGenerator()
        
        # 加载分析结果
        if not generator.load_analysis_results():
            logger.error("无法加载分析结果，报告生成失败")
            return
        
        # 生成报告
        report_text = generator.generate_complete_report()
        
        # 输出报告
        print(report_text)
        
        # 保存报告
        report_file = generator.save_report(report_text)
        
        logger.info(f"综合交易信号报告生成成功！")
        logger.info(f"报告文件位置: {report_file}")
        
    except Exception as e:
        logger.error(f"报告生成过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()