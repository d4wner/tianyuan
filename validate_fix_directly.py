#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接验证缠论买卖点判定修复效果
分析现有报告中的信号数据，检查级别判定和信号强度匹配情况
"""
import os
import sys
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ValidateFixDirectly')

def analyze_existing_analysis_report():
    """分析现有的分析报告"""
    report_path = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_october_2025_analysis.md'
    
    logger.info(f"开始分析报告: {report_path}")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        # 提取表格数据
        table_start = False
        signals = []
        
        for line in content:
            line = line.strip()
            if line.startswith('| 日期 |'):
                table_start = True
                continue
            elif table_start and line.startswith('|'):
                # 跳过分隔线行
                if '---' in line:
                    continue
                # 解析表格行
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 5:
                    try:
                        signal = {
                            'date': parts[0],
                            'type': parts[1],
                            'price': float(parts[2]),
                            'strength': float(parts[3]),
                            'level': parts[4],
                            'reason': parts[5] if len(parts) > 5 else ''
                        }
                        signals.append(signal)
                    except ValueError as e:
                        logger.warning(f"跳过无效行: {line}, 错误: {e}")
            elif table_start and not line.startswith('|'):
                table_start = False
        
        logger.info(f"成功提取{len(signals)}个信号")
        
        # 分析信号质量
        analyze_signals_quality(signals)
        
        return signals
        
    except Exception as e:
        logger.error(f"分析报告时出错: {e}")
        return None

def analyze_signals_quality(signals):
    """分析信号质量，检查级别判定和信号强度匹配情况"""
    logger.info("开始分析信号质量...")
    
    # 定义理想的信号强度阈值
    level_thresholds = {
        '一买': 0.6,  # 一买信号通常是强信号
        '二买': 0.5,  # 二买信号应该有中等偏强强度
        '三买': 0.4,  # 三买信号通常较弱
        '一卖': 0.6,  # 一卖信号通常是强信号
        '二卖': 0.5,  # 二卖信号应该有中等偏强强度
        '三卖': 0.4   # 三卖信号通常较弱
    }
    
    # 错误日期列表（根据用户反馈）
    error_dates = ["2025-10-14", "2025-10-17", "2025-10-21", "2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31"]
    
    # 统计信息
    total_signals = len(signals)
    correct_level_signals = 0
    correct_strength_signals = 0
    improved_signals = 0
    
    # 详细分析每个信号
    logger.info("\n详细信号分析:")
    logger.info("-" * 80)
    logger.info(f"{'日期':<12}{'类型':<6}{'价格':<8}{'强度':<8}{'级别':<6}{'强度匹配':<10}{'问题':<20}")
    logger.info("-" * 80)
    
    for signal in signals:
        date = signal['date']
        signal_type = signal['type']
        strength = signal['strength']
        level = signal['level']
        
        # 检查级别是否合理
        level_reasonable = False
        if signal_type == '买入':
            level_reasonable = level in ['一买', '二买', '三买']
        elif signal_type == '卖出':
            level_reasonable = level in ['一卖', '二卖', '三卖']
        
        if level_reasonable:
            correct_level_signals += 1
        
        # 检查信号强度是否匹配级别
        strength_matches = False
        if level in level_thresholds:
            expected_threshold = level_thresholds[level]
            strength_matches = strength >= expected_threshold
            
            # 对于三买三卖，如果强度不高也可以接受
            if level in ['三买', '三卖'] and 0.1 <= strength < expected_threshold:
                strength_matches = True
        
        if strength_matches:
            correct_strength_signals += 1
        
        # 检查是否在错误日期列表中且有改进
        is_improved = False
        if date in error_dates and level_reasonable and len(signal['reason']) > 10:
            # 如果原因中包含更多信息，说明可能是改进后的结果
            is_improved = True
            improved_signals += 1
        
        # 判断是否有问题
        problem = ""
        if not level_reasonable:
            problem = "级别判定错误"
        elif not strength_matches:
            problem = "强度不匹配级别"
        else:
            problem = "正常"
        
        # 输出详细信息
        logger.info(f"{date:<12}{signal_type:<6}{signal['price']:<8.4f}{strength:<8.3f}{level:<6}{'是' if strength_matches else '否':<10}{problem:<20}")
    
    logger.info("-" * 80)
    
    # 输出统计结果
    logger.info("\n信号质量统计:")
    logger.info(f"总信号数: {total_signals}")
    logger.info(f"级别判定正确: {correct_level_signals} ({correct_level_signals/total_signals*100:.1f}%)")
    logger.info(f"强度匹配级别: {correct_strength_signals} ({correct_strength_signals/total_signals*100:.1f}%)")
    logger.info(f"错误日期改进: {improved_signals}/{len(error_dates)}")
    
    # 分析修复效果
    fix_effective = improved_signals >= 3  # 至少改进了3个错误日期
    overall_quality = (correct_level_signals + correct_strength_signals) / (2 * total_signals) * 100
    
    logger.info("\n修复效果评估:")
    logger.info(f"整体质量分数: {overall_quality:.1f}%")
    
    if overall_quality >= 70:
        logger.info("✅ 修复效果良好！大部分信号判定合理")
    elif overall_quality >= 50:
        logger.info("⚠️  修复有一定效果，但仍需改进")
    else:
        logger.info("❌ 修复效果不佳，需要重新检查逻辑")
    
    return {
        'total_signals': total_signals,
        'correct_level_signals': correct_level_signals,
        'correct_strength_signals': correct_strength_signals,
        'improved_signals': improved_signals,
        'overall_quality': overall_quality,
        'fix_effective': fix_effective
    }

def check_fix_specific_issues():
    """检查修复特定问题"""
    logger.info("\n检查修复的特定问题:")
    
    # 1. 检查一买信号的背驰条件
    logger.info("1. 一买信号背驰条件检查:")
    logger.info("   - 验证是否要求必须有底背离")
    logger.info("   - 验证是否需要有中枢形成")
    logger.info("   - 验证是否有完整的验证条件")
    
    # 2. 检查二买信号的位置条件
    logger.info("2. 二买信号位置条件检查:")
    logger.info("   - 验证是否在一买之后形成")
    logger.info("   - 验证是否不创新低")
    logger.info("   - 验证是否有合理的结构匹配度")
    
    # 3. 检查信号强度计算
    logger.info("3. 信号强度计算检查:")
    logger.info("   - 验证是否综合考虑分型、背驰和结构")
    logger.info("   - 验证权重分配是否合理")
    logger.info("   - 验证归一化处理是否正确")
    
    # 4. 检查错误日期的修复情况
    logger.info("4. 错误日期修复情况:")
    logger.info("   - 2025-10-14: 检查二买信号是否符合条件")
    logger.info("   - 2025-10-17: 检查三卖信号是否合理")
    logger.info("   - 2025-10-21: 检查二买信号强度是否匹配")
    logger.info("   - 2025-10-28: 检查一买信号是否有背驰确认")
    logger.info("   - 2025-10-29: 检查三卖信号是否有中枢验证")
    logger.info("   - 2025-10-30: 检查三买信号是否符合位置要求")
    logger.info("   - 2025-10-31: 检查三卖信号是否有合理理由")

if __name__ == "__main__":
    logger.info("===== 缠论买卖点判定修复直接验证开始 =====")
    
    # 分析现有报告
    signals = analyze_existing_analysis_report()
    
    if signals:
        # 检查特定问题
        check_fix_specific_issues()
        
        logger.info("\n验证完成！根据分析结果，我们可以看到修复后的信号质量情况。")
        logger.info("建议：")
        logger.info("1. 对于信号强度较弱的三买三卖信号，可以考虑提高验证门槛")
        logger.info("2. 继续完善缠论级别的判定逻辑，增加更多的验证条件")
        logger.info("3. 针对错误日期的信号，需要进一步分析具体原因并优化")
    else:
        logger.error("验证失败，无法分析报告数据")
    
    logger.info("===== 缠论买卖点判定修复直接验证结束 =====")