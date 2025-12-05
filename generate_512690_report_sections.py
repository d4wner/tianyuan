#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512690酒ETF报告生成工具

生成信号分类汇总和交易参数设置的标准格式
"""

import json
import os
from typing import Dict, List

class ReportSectionGenerator:
    """
    报告章节生成器
    """
    
    def __init__(self, analysis_file: str = '512690_2025_analysis_result.json'):
        """
        初始化生成器
        
        Args:
            analysis_file: 分析结果文件路径
        """
        self.analysis_file = analysis_file
        self.analysis_data = None
        self.load_analysis_data()
    
    def load_analysis_data(self) -> bool:
        """
        加载分析数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                self.analysis_data = json.load(f)
            return True
        except Exception as e:
            print(f"加载分析数据失败: {str(e)}")
            return False
    
    def generate_signal_summary(self) -> Dict:
        """
        生成信号分类汇总
        
        Returns:
            Dict: 信号分类汇总数据
        """
        if not self.analysis_data or 'signal_analysis' not in self.analysis_data:
            return {}
        
        signals = self.analysis_data['signal_analysis']
        
        # 统计信号
        total_signals = len(signals)
        if total_signals == 0:
            return {
                'total_signals': 0,
                'signal_types': [],
                'signal_summary': "无有效交易信号"
            }
        
        # 分析信号类型分布
        signal_types = {}
        for signal in signals:
            signal_type = signal['final_signal_type']
            if signal_type not in signal_types:
                signal_types[signal_type] = 0
            signal_types[signal_type] += 1
        
        # 生成信号类型列表
        type_list = []
        for signal_type, count in signal_types.items():
            type_list.append({
                'type': signal_type,
                'count': count,
                'percentage': round(count / total_signals * 100, 1)
            })
        
        # 找到最优先的信号
        primary_signal = None
        for signal in signals:
            if "待量能" in signal['final_signal_type']:
                primary_signal = signal
                break
        
        if not primary_signal and signals:
            primary_signal = signals[0]
        
        # 生成汇总文本
        summary_text = []
        summary_text.append(f"2025年512690酒ETF共产生{total_signals}个潜在交易信号")
        summary_text.append(f"信号类型分布：")
        for st in type_list:
            summary_text.append(f"- {st['type']}：{st['count']}个（{st['percentage']}%）")
        
        if primary_signal:
            summary_text.append("\n重点关注信号：")
            summary_text.append(f"- 信号类型：{primary_signal['signal_type']}")
            summary_text.append(f"- 对应中枢：{primary_signal['central_name']}（{primary_signal['central_range']}）")
            summary_text.append(f"- 信号状态：{primary_signal['final_signal_type']}")
            summary_text.append(f"- 破位验证：{primary_signal['break_verify']}")
            summary_text.append(f"- 反抽验证：{primary_signal['rebound_verify']}")
        
        return {
            'total_signals': total_signals,
            'signal_types': type_list,
            'signal_summary': '\n'.join(summary_text),
            'primary_signal': primary_signal
        }
    
    def generate_risk_control(self) -> Dict:
        """
        生成风险控制说明
        
        Returns:
            Dict: 风险控制数据
        """
        # 基于分析结果生成风险控制说明
        volatility = self.analysis_data['basic_info'].get('annual_volatility', 0)
        volatility_level = self.analysis_data['basic_info'].get('volatility_level', '')
        
        risk_control_items = [
            {
                'risk_type': "波动率风险",
                'description': f"该ETF波动率为{volatility}%（{volatility_level}），应采取相应仓位控制",
                'control_measure': "中波动ETF，单次最高仓位不超过15%"
            },
            {
                'risk_type': "市场系统性风险",
                'description': "ETF受大盘系统性影响较大，需关注整体市场走势",
                'control_measure': "设置组合止损线，若大盘下跌超过5%，考虑减仓"
            },
            {
                'risk_type': "信号置信度风险",
                'description': "当前信号为潜在监控信号，量能和MACD未完全验证",
                'control_measure': "分批建仓，首次不超过5%仓位，验证后再加仓"
            },
            {
                'risk_type': "止损风险",
                'description': "需严格执行止损纪律，避免单笔亏损过大",
                'control_measure': "设置硬性止损单，价格触发自动卖出"
            }
        ]
        
        risk_summary = """
        512690酒ETF 2025年风险控制要点：
        1. 严格执行仓位控制：单次不超过15%，分批建仓降低风险
        2. 严格止损：设置硬性止损单，止损价格不低于中枢下沿×0.98
        3. 信号验证：重点关注量能和MACD指标是否验证信号有效性
        4. 分批操作：可将计划仓位分2-3次执行，降低单次决策风险
        5. 定期再评估：每两周重新评估信号有效性，及时调整策略
        """.strip()
        
        return {
            'risk_control_items': risk_control_items,
            'risk_summary': risk_summary
        }
    
    def generate_conclusion(self) -> Dict:
        """
        生成结论
        
        Returns:
            Dict: 结论数据
        """
        # 获取关键信息
        basic_info = self.analysis_data['basic_info']
        central_analysis = self.analysis_data['central_analysis']
        signal_summary = self.generate_signal_summary()
        trading_params = self.analysis_data['trading_params']
        
        # 统计有效中枢
        valid_centrals = [c for c in central_analysis if '有效中枢' in c['central_judgment']]
        
        # 找到最新有效中枢
        latest_valid_central = None
        for c in central_analysis[::-1]:
            if '有效中枢' in c['central_judgment']:
                latest_valid_central = c
                break
        
        # 生成结论文本
        conclusion_text = []
        conclusion_text.append(f"512690酒ETF 2025年分析结论")
        conclusion_text.append(f"\n一、基本情况：")
        conclusion_text.append(f"- 年度波动率：{basic_info['annual_volatility']}%（{basic_info['volatility_level']}）")
        conclusion_text.append(f"- 2025年价格区间：{basic_info['price_range']['min']} - {basic_info['price_range']['max']}")
        conclusion_text.append(f"- 有效中枢数量：{len(valid_centrals)}个")
        
        if latest_valid_central:
            conclusion_text.append(f"\n二、最新有效中枢：")
            conclusion_text.append(f"- 时期：{latest_valid_central['quarter']}")
            conclusion_text.append(f"- 价格区间：{latest_valid_central['trading_range_90']}")
            conclusion_text.append(f"- 振幅：{latest_valid_central['amplitude_90']}%")
        
        conclusion_text.append(f"\n三、信号结论：")
        if signal_summary['total_signals'] > 0:
            conclusion_text.append(f"- 当前有{signal_summary['total_signals']}个潜在交易信号")
            if signal_summary['primary_signal']:
                conclusion_text.append(f"- 重点关注信号：{signal_summary['primary_signal']['final_signal_type']}")
                conclusion_text.append(f"- 对应中枢：{signal_summary['primary_signal']['central_name']}")
        else:
            conclusion_text.append("- 当前无有效交易信号，建议继续观察")
        
        if trading_params:
            conclusion_text.append(f"\n四、交易建议：")
            risk_reward_param = next((p for p in trading_params if p['param_type'] == '风险收益比'), None)
            if risk_reward_param:
                conclusion_text.append(f"- {risk_reward_param['explain']}")
            
            position_param = next((p for p in trading_params if p['param_type'] == '建议仓位'), None)
            if position_param:
                conclusion_text.append(f"- {position_param['explain']}")
        
        conclusion_text.append(f"\n五、执行策略：")
        conclusion_text.append(f"- 建议采用分批建仓策略，降低建仓风险")
        conclusion_text.append(f"- 严格执行止损纪律，避免单笔亏损过大")
        conclusion_text.append(f"- 定期再评估信号有效性，及时调整策略")
        
        return {
            'conclusion_text': '\n'.join(conclusion_text),
            'valid_central_count': len(valid_centrals),
            'signal_count': signal_summary['total_signals']
        }
    
    def generate_full_report_data(self) -> Dict:
        """
        生成完整报告数据
        
        Returns:
            Dict: 完整报告数据
        """
        signal_summary = self.generate_signal_summary()
        risk_control = self.generate_risk_control()
        conclusion = self.generate_conclusion()
        
        report_data = {
            'basic_info': self.analysis_data['basic_info'],
            'central_analysis': self.analysis_data['central_analysis'],
            'signal_analysis': self.analysis_data['signal_analysis'],
            'trading_params': self.analysis_data['trading_params'],
            'signal_summary': signal_summary,
            'risk_control': risk_control,
            'conclusion': conclusion
        }
        
        # 保存完整报告数据
        with open('512690_2025_full_report_data.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print("完整报告数据已保存到 512690_2025_full_report_data.json")
        return report_data
    
    def generate_markdown_report(self) -> str:
        """
        生成Markdown格式报告
        
        Returns:
            str: Markdown报告内容
        """
        report_data = self.generate_full_report_data()
        
        # 开始构建Markdown内容
        markdown_content = []
        markdown_content.append("# 512690（酒ETF）2025年可下单信号评估报告")
        markdown_content.append("\n## 一、标的基本信息")
        
        # 基本信息
        basic_info = report_data['basic_info']
        markdown_content.append(f"- **代码**: {basic_info['code']}")
        markdown_content.append(f"- **名称**: {basic_info['name']}")
        markdown_content.append(f"- **分析年度**: {basic_info['analysis_year']}")
        markdown_content.append(f"- **数据范围**: {basic_info['data_range']}")
        markdown_content.append(f"- **数据完整性**: 实际交易日数{basic_info['data_completeness']['actual_days']}天，预计{basic_info['data_completeness']['expected_days']}天")
        markdown_content.append(f"- **价格区间**: {basic_info['price_range']['min']} - {basic_info['price_range']['max']}")
        
        # 波动率分析
        markdown_content.append("\n## 二、波动率分析")
        markdown_content.append(f"- **年度波动率**: {basic_info['annual_volatility']}%")
        markdown_content.append(f"- **波动等级**: {basic_info['volatility_level']}")
        
        # 有效中枢分析
        markdown_content.append("\n## 三、有效中枢分析")
        markdown_content.append("\n### 3.1 中枢详情")
        
        for central in report_data['central_analysis']:
            markdown_content.append(f"\n#### {central['quarter']}")
            markdown_content.append(f"- **90%成交区间**: {central['trading_range_90']}")
            markdown_content.append(f"- **振幅**: {central['amplitude_90']}%")
            markdown_content.append(f"- **覆盖度**: {central['coverage_rate']}%")
            markdown_content.append(f"- **覆盖样本**: {central['coverage_sample']}")
            markdown_content.append(f"- **支撑次数**: {central['support_times']}次")
            markdown_content.append(f"- **压力次数**: {central['pressure_times']}次")
            markdown_content.append(f"- **中枢判定**: {central['central_judgment']}")
        
        # 交易信号分析
        markdown_content.append("\n## 四、交易信号分析")
        
        for signal in report_data['signal_analysis']:
            markdown_content.append(f"\n### 4.1 {signal['central_name']}信号")
            markdown_content.append(f"- **信号类型**: {signal['signal_type']}")
            markdown_content.append(f"- **对应中枢**: {signal['central_range']}")
            markdown_content.append(f"- **破位验证**: {signal['break_verify']}")
            markdown_content.append(f"- **反抽验证**: {signal['rebound_verify']}")
            markdown_content.append(f"- **量能验证**: {signal['volume_verify']}")
            markdown_content.append(f"- **MACD验证**: {signal['macd_verify']}")
            markdown_content.append(f"- **信号类型**: {signal['final_signal_type']}")
        
        # 信号分类汇总
        markdown_content.append("\n## 五、信号分类汇总")
        markdown_content.append(f"\n{report_data['signal_summary']['signal_summary']}")
        
        # 交易参数设置
        markdown_content.append("\n## 六、交易参数设置")
        
        markdown_content.append("\n| 参数类型 | 取值规则 | 说明 |")
        markdown_content.append("|---------|---------|------|")
        
        for param in report_data['trading_params']:
            # 修复风险收益比说明中的多余右括号
            explain = param['explain'].replace("≥1.5）", "≥1.5")
            markdown_content.append(f"| {param['param_type']} | {param['value_rule']} | {explain} |")
        
        # 风险提示
        markdown_content.append("\n## 七、风险提示")
        markdown_content.append(f"\n{report_data['risk_control']['risk_summary']}")
        
        # 结论
        markdown_content.append("\n## 八、结论")
        markdown_content.append(f"\n{report_data['conclusion']['conclusion_text']}")
        
        # JSON格式结构化数据
        markdown_content.append("\n## 九、JSON格式结构化数据")
        
        # 准备JSON数据，移除多余的右括号
        json_data = {
            'basic_info': basic_info,
            'central_analysis': report_data['central_analysis'],
            'signal_analysis': report_data['signal_analysis'],
            'trading_params': [{
                'param_type': p['param_type'],
                'value_rule': p['value_rule'],
                'explain': p['explain'].replace("≥1.5）", "≥1.5")
            } for p in report_data['trading_params']]
        }
        
        markdown_content.append("```json")
        markdown_content.append(json.dumps(json_data, ensure_ascii=False, indent=2))
        markdown_content.append("```")
        
        return '\n'.join(markdown_content)
    
    def save_markdown_report(self, output_file: str = '512690_2025_central_report.md') -> bool:
        """
        保存Markdown报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            markdown_content = self.generate_markdown_report()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"报告已保存到 {output_file}")
            return True
        except Exception as e:
            print(f"保存报告失败: {str(e)}")
            return False

if __name__ == "__main__":
    generator = ReportSectionGenerator()
    generator.save_markdown_report()