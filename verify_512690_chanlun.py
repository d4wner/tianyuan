#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512690分型和笔划分验证脚本

基于缠论理论，验证512690的分型（顶分型、底分型）和笔划分的正确性。

作者: TradeTianYuan
日期: 2025-11-28
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('512690ChanlunVerifier')

class ChanlunVerifier:
    """
    缠论验证器类，用于检测分型和笔划分
    """
    
    def __init__(self, symbol: str = '512690', data_dir: str = './data/daily'):
        """
        初始化缠论验证器
        
        Args:
            symbol: 股票代码
            data_dir: 数据目录
        """
        self.symbol = symbol
        self.data_dir = data_dir
        self.daily_data = None
        self.fx_points = []  # 分型点列表
        self.bi_segments = []  # 笔划分列表
        
        # 创建结果目录
        self.results_dir = './results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self) -> bool:
        """
        加载日线数据
        
        Returns:
            bool: 是否加载成功
        """
        try:
            data_file = os.path.join(self.data_dir, f'{self.symbol}_daily.csv')
            if not os.path.exists(data_file):
                logger.error(f"数据文件不存在: {data_file}")
                return False
            
            self.daily_data = pd.read_csv(data_file)
            self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
            self.daily_data.sort_values('date', inplace=True)
            
            logger.info(f"成功加载{self.symbol}日线数据，共{len(self.daily_data)}条记录")
            return True
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False
    
    def detect_fx_points(self, lookback: int = 2) -> List[Dict]:
        """
        检测分型点（顶分型和底分型）
        
        Args:
            lookback: 回看K线数量
            
        Returns:
            List[Dict]: 分型点列表
        """
        if self.daily_data is None:
            logger.error("请先加载数据")
            return []
        
        fx_points = []
        
        for i in range(lookback, len(self.daily_data) - lookback):
            # 获取当前K线和前后K线
            current = self.daily_data.iloc[i]
            left_bars = [self.daily_data.iloc[i-j] for j in range(1, lookback+1)]
            right_bars = [self.daily_data.iloc[i+j] for j in range(1, lookback+1)]
            
            # 检查顶分型
            if (current['high'] > max([bar['high'] for bar in left_bars]) and 
                current['high'] > max([bar['high'] for bar in right_bars])):
                fx_points.append({
                    'date': current['date'],
                    'date_str': current['date'].strftime('%Y-%m-%d'),
                    'type': '顶分型',
                    'price': current['high'],
                    'index': i
                })
            
            # 检查底分型
            elif (current['low'] < min([bar['low'] for bar in left_bars]) and 
                  current['low'] < min([bar['low'] for bar in right_bars])):
                fx_points.append({
                    'date': current['date'],
                    'date_str': current['date'].strftime('%Y-%m-%d'),
                    'type': '底分型',
                    'price': current['low'],
                    'index': i
                })
        
        # 过滤掉重合的分型
        filtered_fx = []
        for fx in fx_points:
            keep = True
            for i, existing in enumerate(filtered_fx):
                if existing['type'] == fx['type']:
                    # 同一类型的分型，如果新的价格更高（顶分型）或更低（底分型），则替换
                    if (existing['type'] == '顶分型' and fx['price'] > existing['price']) or \
                       (existing['type'] == '底分型' and fx['price'] < existing['price']):
                        filtered_fx[i] = fx
                        keep = False
                    else:
                        keep = False
            if keep:
                filtered_fx.append(fx)
        
        # 按时间排序
        filtered_fx.sort(key=lambda x: x['date'])
        
        self.fx_points = filtered_fx
        logger.info(f"检测到{len(filtered_fx)}个分型点（{sum(1 for fx in filtered_fx if fx['type'] == '顶分型')}个顶分型，{sum(1 for fx in filtered_fx if fx['type'] == '底分型')}个底分型）")
        return filtered_fx
    
    def divide_bi_segments(self) -> List[Dict]:
        """
        划分笔
        注：简化版本，实际缠论的笔划分需要考虑包含关系处理、分型确认等复杂规则
        
        Returns:
            List[Dict]: 笔划分列表
        """
        if not self.fx_points:
            logger.error("请先检测分型点")
            return []
        
        bi_segments = []
        current_fx = None
        
        for fx in self.fx_points:
            if current_fx is None:
                current_fx = fx
                continue
            
            # 确保分型类型交替
            if current_fx['type'] != fx['type']:
                # 计算笔的方向和长度
                direction = '向上笔' if current_fx['type'] == '底分型' and fx['type'] == '顶分型' else '向下笔'
                price_change = fx['price'] - current_fx['price']
                percent_change = (price_change / current_fx['price']) * 100
                
                bi_segments.append({
                    'start_date': current_fx['date_str'],
                    'end_date': fx['date_str'],
                    'start_price': current_fx['price'],
                    'end_price': fx['price'],
                    'direction': direction,
                    'price_change': price_change,
                    'percent_change': percent_change,
                    'start_type': current_fx['type'],
                    'end_type': fx['type'],
                    'bar_count': fx['index'] - current_fx['index'] + 1
                })
                
                current_fx = fx
        
        # 添加最后一个可能未完成的笔
        if len(bi_segments) > 0:
            last_bi = bi_segments[-1]
            last_fx = self.fx_points[-1]
            latest_bar = self.daily_data.iloc[-1]
            
            # 检查是否有新的笔形成的可能
            if last_bi['end_type'] != '底分型' and latest_bar['low'] < last_fx['price']:
                bi_segments.append({
                    'start_date': last_fx['date_str'],
                    'end_date': latest_bar['date'].strftime('%Y-%m-%d'),
                    'start_price': last_fx['price'],
                    'end_price': latest_bar['low'],
                    'direction': '向下笔',
                    'price_change': latest_bar['low'] - last_fx['price'],
                    'percent_change': ((latest_bar['low'] / last_fx['price']) - 1) * 100,
                    'start_type': last_fx['type'],
                    'end_type': '潜在底分型',
                    'bar_count': len(self.daily_data) - last_fx['index']
                })
            elif last_bi['end_type'] != '顶分型' and latest_bar['high'] > last_fx['price']:
                bi_segments.append({
                    'start_date': last_fx['date_str'],
                    'end_date': latest_bar['date'].strftime('%Y-%m-%d'),
                    'start_price': last_fx['price'],
                    'end_price': latest_bar['high'],
                    'direction': '向上笔',
                    'price_change': latest_bar['high'] - last_fx['price'],
                    'percent_change': ((latest_bar['high'] / last_fx['price']) - 1) * 100,
                    'start_type': last_fx['type'],
                    'end_type': '潜在顶分型',
                    'bar_count': len(self.daily_data) - last_fx['index']
                })
        
        self.bi_segments = bi_segments
        logger.info(f"划分出{len(bi_segments)}个笔")
        return bi_segments
    
    def verify_fx_quality(self) -> Dict:
        """
        验证分型质量
        
        Returns:
            Dict: 分型质量分析结果
        """
        if not self.fx_points:
            return {
                'total_fx': 0,
                'top_fx': 0,
                'bottom_fx': 0,
                'avg_distance': 0,
                'quality_stats': {}
            }
        
        # 计算分型间距统计
        distances = []
        for i in range(1, len(self.fx_points)):
            distance = (self.fx_points[i]['date'] - self.fx_points[i-1]['date']).days
            distances.append(distance)
        
        # 计算分型可靠性指标（简化版）
        reliable_fx = []
        for fx in self.fx_points:
            # 查找附近的K线
            idx = fx['index']
            nearby_bars = self.daily_data.iloc[max(0, idx-5):min(len(self.daily_data), idx+5)]
            
            # 计算振幅
            amplitude = (nearby_bars['high'].max() - nearby_bars['low'].min()) / nearby_bars['close'].mean() * 100
            
            # 计算成交量变化
            vol_change = 0
            if idx > 0:
                prev_vol = self.daily_data.iloc[idx-1]['volume']
                curr_vol = self.daily_data.iloc[idx]['volume']
                vol_change = (curr_vol / prev_vol - 1) * 100
            
            # 简单的可靠性评分
            reliability = 0
            if amplitude > 2.0:  # 振幅大于2%认为比较可靠
                reliability += 50
            if abs(vol_change) > 30:  # 成交量变化大于30%认为有一定可靠性
                reliability += 30
            if abs(fx['price'] - self.daily_data.iloc[idx]['close']) / fx['price'] < 0.01:  # 分型点接近收盘价
                reliability += 20
            
            reliable_fx.append({
                'date_str': fx['date_str'],
                'type': fx['type'],
                'price': fx['price'],
                'amplitude': amplitude,
                'volume_change': vol_change,
                'reliability_score': reliability
            })
        
        high_quality = sum(1 for fx in reliable_fx if fx['reliability_score'] > 70)
        medium_quality = sum(1 for fx in reliable_fx if 40 <= fx['reliability_score'] <= 70)
        low_quality = sum(1 for fx in reliable_fx if fx['reliability_score'] < 40)
        
        return {
            'total_fx': len(self.fx_points),
            'top_fx': sum(1 for fx in self.fx_points if fx['type'] == '顶分型'),
            'bottom_fx': sum(1 for fx in self.fx_points if fx['type'] == '底分型'),
            'avg_distance': np.mean(distances) if distances else 0,
            'quality_stats': {
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality,
                'avg_reliability': np.mean([fx['reliability_score'] for fx in reliable_fx]) if reliable_fx else 0
            },
            'detailed_fx': reliable_fx
        }
    
    def verify_bi_correctness(self) -> Dict:
        """
        验证笔划分正确性
        
        Returns:
            Dict: 笔划分正确性分析结果
        """
        if not self.bi_segments:
            return {
                'total_bi': 0,
                'up_bi': 0,
                'down_bi': 0,
                'avg_length': 0,
                'correctness_stats': {}
            }
        
        # 计算笔长度统计
        up_bi_lengths = [bi['percent_change'] for bi in self.bi_segments if bi['direction'] == '向上笔']
        down_bi_lengths = [bi['percent_change'] for bi in self.bi_segments if bi['direction'] == '向下笔']
        avg_up_length = np.mean(up_bi_lengths) if up_bi_lengths else 0
        avg_down_length = np.mean(down_bi_lengths) if down_bi_lengths else 0
        
        # 验证笔的延续性
        continuity_score = 0
        standard_bi_count = 0  # 符合标准的笔数量
        
        for bi in self.bi_segments:
            # 检查笔的长度是否合理（至少1%）
            if abs(bi['percent_change']) >= 1.0:
                standard_bi_count += 1
                continuity_score += 30
            
            # 检查K线数量是否合理（至少5根）
            if bi['bar_count'] >= 5:
                continuity_score += 30
            
            # 检查是否有重叠（简化判断）
            if len(self.bi_segments) > 1:
                continuity_score += 20
        
        continuity_score = continuity_score / len(self.bi_segments) if self.bi_segments else 0
        
        # 检查最近的笔是否形成中枢
        has_zhongshu = False
        if len(self.bi_segments) >= 3:
            # 简化的中枢判断：最近3笔是否有价格重叠
            recent_3bi = self.bi_segments[-3:]
            if len(recent_3bi) == 3:
                # 获取价格范围
                all_prices = []
                for bi in recent_3bi:
                    all_prices.extend([bi['start_price'], bi['end_price']])
                
                # 检查是否形成重叠区间
                if max(min([bi['start_price'], bi['end_price']]) for bi in recent_3bi) < \
                   min(max([bi['start_price'], bi['end_price']]) for bi in recent_3bi):
                    has_zhongshu = True
        
        return {
            'total_bi': len(self.bi_segments),
            'up_bi': sum(1 for bi in self.bi_segments if bi['direction'] == '向上笔'),
            'down_bi': sum(1 for bi in self.bi_segments if bi['direction'] == '向下笔'),
            'avg_length': np.mean([abs(bi['percent_change']) for bi in self.bi_segments]) if self.bi_segments else 0,
            'avg_up_length': avg_up_length,
            'avg_down_length': avg_down_length,
            'correctness_stats': {
                'standard_bi_count': standard_bi_count,
                'standard_bi_ratio': standard_bi_count / len(self.bi_segments) if self.bi_segments else 0,
                'continuity_score': continuity_score,
                'has_zhongshu': has_zhongshu
            }
        }
    
    def generate_verification_report(self) -> str:
        """
        生成验证报告
        
        Returns:
            str: 验证报告文本
        """
        # 验证分型质量
        fx_quality = self.verify_fx_quality()
        
        # 验证笔划分正确性
        bi_correctness = self.verify_bi_correctness()
        
        # 生成报告
        report = []
        report.append(f"===== {self.symbol}分型和笔划分验证报告 =====")
        report.append(f"验证日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据范围: {self.daily_data.iloc[0]['date'].strftime('%Y-%m-%d')} 至 {self.daily_data.iloc[-1]['date'].strftime('%Y-%m-%d')}")
        report.append("")
        
        # 分型分析
        report.append("📊 分型分析:")
        report.append("-" * 80)
        report.append(f"分型总数: {fx_quality['total_fx']}个")
        report.append(f"顶分型: {fx_quality['top_fx']}个")
        report.append(f"底分型: {fx_quality['bottom_fx']}个")
        report.append(f"平均分型间距: {fx_quality['avg_distance']:.1f}天")
        report.append("")
        
        # 分型质量统计
        quality = fx_quality['quality_stats']
        report.append(f"分型质量统计:")
        report.append(f"  - 高质量分型(>70分): {quality['high_quality']}个 ({quality['high_quality']/fx_quality['total_fx']*100:.1f}%)")
        report.append(f"  - 中等质量分型(40-70分): {quality['medium_quality']}个 ({quality['medium_quality']/fx_quality['total_fx']*100:.1f}%)")
        report.append(f"  - 低质量分型(<40分): {quality['low_quality']}个 ({quality['low_quality']/fx_quality['total_fx']*100:.1f}%)")
        report.append(f"  - 平均可靠性评分: {quality['avg_reliability']:.1f}/100")
        report.append("")
        
        # 最近的分型
        recent_fx = sorted(fx_quality['detailed_fx'], key=lambda x: x['date_str'], reverse=True)[:10]
        report.append("最近的分型点:")
        report.append(f"{'日期':<15} {'类型':<10} {'价格':<10} {'可靠性评分':<15} {'振幅(%)':<10} {'成交量变化(%)':<15}")
        report.append("-" * 80)
        for fx in recent_fx:
            report.append(f"{fx['date_str']:<15} {fx['type']:<10} {fx['price']:<10.3f} {fx['reliability_score']:<15.1f} {fx['amplitude']:<10.2f} {fx['volume_change']:<15.2f}")
        report.append("")
        
        # 笔划分分析
        report.append("📈 笔划分分析:")
        report.append("-" * 80)
        report.append(f"笔总数: {bi_correctness['total_bi']}个")
        report.append(f"向上笔: {bi_correctness['up_bi']}个")
        report.append(f"向下笔: {bi_correctness['down_bi']}个")
        report.append(f"平均笔长度: {bi_correctness['avg_length']:.2f}%")
        report.append(f"平均向上笔长度: {bi_correctness['avg_up_length']:.2f}%")
        report.append(f"平均向下笔长度: {bi_correctness['avg_down_length']:.2f}%")
        report.append("")
        
        # 笔划分正确性
        correctness = bi_correctness['correctness_stats']
        report.append(f"笔划分正确性统计:")
        report.append(f"  - 标准笔数量(长度≥1%): {correctness['standard_bi_count']}个 ({correctness['standard_bi_ratio']*100:.1f}%)")
        report.append(f"  - 延续性评分: {correctness['continuity_score']:.1f}/100")
        report.append(f"  - 是否形成中枢: {'是' if correctness['has_zhongshu'] else '否'}")
        report.append("")
        
        # 最近的笔
        recent_bi = sorted(self.bi_segments, key=lambda x: x['end_date'], reverse=True)[:5]
        report.append("最近的笔划分:")
        report.append(f"{'起始日期':<15} {'结束日期':<15} {'方向':<10} {'涨跌幅(%)':<12} {'K线数量':<10} {'结束类型':<15}")
        report.append("-" * 80)
        for bi in recent_bi:
            change_str = f"{bi['percent_change']:+.2f}%" if bi['percent_change'] != 0 else "0.00%"
            report.append(f"{bi['start_date']:<15} {bi['end_date']:<15} {bi['direction']:<10} {change_str:<12} {bi['bar_count']:<10} {bi['end_type']:<15}")
        report.append("")
        
        # 验证结论
        report.append("📝 验证结论:")
        report.append("-" * 80)
        
        # 分型验证结论
        if fx_quality['total_fx'] > 0:
            if quality['avg_reliability'] > 70:
                report.append("✅ 分型质量良好: 分型清晰，可靠性高")
            elif quality['avg_reliability'] > 50:
                report.append("⚠️ 分型质量中等: 部分分型可靠性一般，建议结合其他指标")
            else:
                report.append("❌ 分型质量较差: 分型不清晰，可靠性低，需要谨慎判断")
        else:
            report.append("❌ 未检测到分型: 可能数据不足或市场波动较小")
        
        # 笔划分验证结论
        if bi_correctness['total_bi'] > 0:
            if correctness['continuity_score'] > 70 and correctness['standard_bi_ratio'] > 0.8:
                report.append("✅ 笔划分合理: 笔的数量和质量符合缠论要求")
            elif correctness['continuity_score'] > 50:
                report.append("⚠️ 笔划分基本合理: 部分笔可能需要调整")
            else:
                report.append("❌ 笔划分存在问题: 笔的延续性差，建议重新划分")
            
            if correctness['has_zhongshu']:
                report.append("🔄 已形成中枢: 当前走势处于中枢震荡阶段")
        else:
            report.append("❌ 未完成笔划分: 可能分型数量不足")
        
        # 综合建议
        report.append("")
        report.append("🎯 交易建议:")
        report.append("-" * 80)
        if len(self.bi_segments) > 0:
            last_bi = self.bi_segments[-1]
            if last_bi['direction'] == '向上笔':
                if last_bi['percent_change'] > 3:  # 向上笔幅度较大
                    report.append("📉 注意风险: 当前处于向上笔中，涨幅较大，可能接近顶部")
                else:
                    report.append("📈 关注买入机会: 当前处于向上笔初始阶段")
            else:
                if last_bi['percent_change'] < -3:  # 向下笔幅度较大
                    report.append("📈 关注买入机会: 向下笔幅度较大，可能接近底部")
                else:
                    report.append("🔍 继续观察: 当前处于向下笔中，建议等待明确信号")
        else:
            report.append("🔍 数据不足: 建议积累更多数据后再进行判断")
        
        report.append("")
        report.append("⚠️ 风险提示:")
        report.append("-" * 80)
        report.append("1. 本验证基于简化的缠论规则，仅供参考")
        report.append("2. 实际操作中应结合其他技术分析方法")
        report.append("3. 市场有风险，投资需谨慎")
        
        return "\n".join(report)
    
    def save_results(self, report: str) -> None:
        """
        保存验证结果
        
        Args:
            report: 验证报告文本
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存报告文本
        report_file = os.path.join(self.results_dir, f'{self.symbol}_chanlun_verification_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"验证报告已保存至: {report_file}")
        
        # 保存分型数据
        fx_file = os.path.join(self.results_dir, f'{self.symbol}_fx_points_{timestamp}.json')
        # 转换datetime对象为字符串以JSON序列化
        serializable_fx = []
        for fx in self.fx_points:
            fx_copy = fx.copy()
            fx_copy['date'] = fx_copy['date'].strftime('%Y-%m-%d')
            serializable_fx.append(fx_copy)
        
        with open(fx_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_fx, f, ensure_ascii=False, indent=2)
        logger.info(f"分型数据已保存至: {fx_file}")
        
        # 保存笔数据
        bi_file = os.path.join(self.results_dir, f'{self.symbol}_bi_segments_{timestamp}.json')
        with open(bi_file, 'w', encoding='utf-8') as f:
            json.dump(self.bi_segments, f, ensure_ascii=False, indent=2)
        logger.info(f"笔划分数据已保存至: {bi_file}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='512690分型和笔划分验证脚本')
    parser.add_argument('--symbol', type=str, default='512690', help='股票代码')
    parser.add_argument('--data_dir', type=str, default='./data/daily', help='数据目录')
    parser.add_argument('--lookback', type=int, default=2, help='检测分型的回看K线数量')
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 创建验证器实例
    verifier = ChanlunVerifier(symbol=args.symbol, data_dir=args.data_dir)
    
    # 加载数据
    if not verifier.load_data():
        logger.error("加载数据失败，退出程序")
        return
    
    # 检测分型点
    verifier.detect_fx_points(lookback=args.lookback)
    
    # 划分笔
    verifier.divide_bi_segments()
    
    # 生成验证报告
    report = verifier.generate_verification_report()
    print(report)
    
    # 保存结果
    verifier.save_results(report)

if __name__ == "__main__":
    main()