import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class MacdDivergenceAnalyzer:
    """MACD背驰分析器"""
    
    def __init__(self, data, lookback_period=90):  # 增加lookback_period到90天，覆盖9-11月
        """
        初始化MACD背驰分析器
        
        Args:
            data: 包含OHLCV数据的DataFrame
            lookback_period: 回溯分析的周期数
        """
        self.data = data
        self.lookback_period = lookback_period
        self.analysis_results = {}
    
    def calculate_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
        
        Returns:
            包含MACD指标的DataFrame
        """
        df = self.data.copy()
        
        # 计算EMA
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        
        # 计算信号线
        df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        
        self.data = df
        return df
    
    def find_price_extrema(self, period=8):  # 减小周期，更容易识别底部
        """
        查找价格的局部高低点
        
        Args:
            period: 局部高低点判断的周期
        
        Returns:
            包含局部高低点标记的DataFrame
        """
        df = self.data.copy()
        
        # 标记局部高点
        df['is_higher'] = df['close'] > df['close'].shift(period) 
        df['is_lower'] = df['close'] < df['close'].shift(period)
        
        # 查找底部 - 放宽条件，更容易识别底部
        # 方式1：传统的底部识别
        df['is_bottom'] = (df['close'] < df['close'].shift(1)) & \
                         (df['close'] < df['close'].shift(-1)) & \
                         (df['low'] == df['low'].rolling(period, center=True).min())
        
        # 方式2：额外标记MACD绿柱缩小的区域为潜在底部
        if 'macd_hist' in df.columns:
            # 找出绿柱（负值）缩小的区域
            df['green_hist_shrinking'] = (df['macd_hist'] < 0) & \
                                        (df['macd_hist'] > df['macd_hist'].shift(1)) & \
                                        (df['macd_hist'] > df['macd_hist'].shift(2))
            # 结合两种方式，增加底部识别的可能性
            df['is_bottom'] = df['is_bottom'] | (df['green_hist_shrinking'] & (df['low'] <= df['low'].rolling(5, center=True).min()))
        
        # 查找顶部
        df['is_top'] = (df['close'] > df['close'].shift(1)) & \
                      (df['close'] > df['close'].shift(-1)) & \
                      (df['high'] == df['high'].rolling(period, center=True).max())
        
        self.data = df
        return df
    
    def detect_bottom_divergence(self):
        """
        检测底部背驰，增强绿柱缩小趋势的检测
        
        Returns:
            底部背驰检测结果
        """
        df = self.data.tail(self.lookback_period).copy()
        
        # 方式1：基于底部点的背驰检测
        bottom_indices = df[df['is_bottom']].index.tolist()
        
        # 方式2：检测绿柱缩小趋势的背驰
        has_green_hist_divergence = False
        green_divergence_details = {}
        
        # 检查最近的绿柱是否有明显缩小趋势
        if 'macd_hist' in df.columns:
            # 筛选出绿柱区域（负值）
            green_hist_periods = df[df['macd_hist'] < 0]
            if len(green_hist_periods) >= 10:  # 至少需要10个绿柱周期
                # 找到最近的两个绿柱区间的最低点
                green_hist_values = green_hist_periods['macd_hist'].values
                price_values = green_hist_periods['low'].values
                
                # 分段检查绿柱变化
                if len(green_hist_values) >= 15:
                    # 前半段和后半段的比较
                    mid_point = len(green_hist_values) // 2
                    first_half_green_avg = abs(green_hist_values[:mid_point].mean())
                    second_half_green_avg = abs(green_hist_values[mid_point:].mean())
                    
                    first_half_price_min = price_values[:mid_point].min()
                    second_half_price_min = price_values[mid_point:].min()
                    
                    # 价格新低但绿柱缩小至少20%
                    if second_half_price_min < first_half_price_min and second_half_green_avg < first_half_green_avg * 0.8:
                        has_green_hist_divergence = True
                        green_divergence_details = {
                            'has_divergence': True,
                            'type': '底部背驰',
                            'strength': '中',
                            'strength_score': 60,
                            'first_bottom_date': str(green_hist_periods.index[:mid_point][0]),
                            'first_bottom_price': first_half_price_min,
                            'second_bottom_date': str(green_hist_periods.index[mid_point:][0]),
                            'second_bottom_price': second_half_price_min,
                            'price_diff_pct': round((first_half_price_min - second_half_price_min) / first_half_price_min * 100, 2),
                            'macd_diff_pct': round((first_half_green_avg - second_half_green_avg) / first_half_green_avg * 100, 2),
                            'message': '检测到中强度底部背驰: 价格创新低但MACD绿柱明显缩小'
                        }
        
        # 传统的基于底部点的背驰检测
        if len(bottom_indices) >= 2:
            # 检查最近两个底部是否形成背驰
            recent_bottoms = sorted(bottom_indices)[-2:]
            first_bottom, second_bottom = recent_bottoms
            
            # 放宽价格新低条件：允许3%误差
            price_lower = df.loc[second_bottom, 'low'] < df.loc[first_bottom, 'low'] * 1.03
            # 价格真正创新低
            price_really_lower = df.loc[second_bottom, 'low'] < df.loc[first_bottom, 'low']
            
            # MACD柱高度未创新低（柱状图值更大）
            macd_higher = df.loc[second_bottom, 'macd_hist'] > df.loc[first_bottom, 'macd_hist']
            
            # 或者价格创新低，但MACD线值未创新低
            macd_line_higher = df.loc[second_bottom, 'macd_line'] > df.loc[first_bottom, 'macd_line']
            
            # 增强检测：检查绿柱是否有缩小趋势
            has_green_trend = False
            if 'macd_hist' in df.columns:
                # 获取两个底部点之间的绿柱数据
                between_bottoms = df.loc[first_bottom:second_bottom]
                green_hist = between_bottoms[between_bottoms['macd_hist'] < 0]['macd_hist'].values
                if len(green_hist) >= 3:
                    # 计算绿柱绝对值的趋势
                    first_half_green = abs(green_hist[:len(green_hist)//2]).mean()
                    second_half_green = abs(green_hist[len(green_hist)//2:]).mean()
                    if second_half_green < first_half_green * 0.8:  # 绿柱缩小20%以上
                        has_green_trend = True
        
        if price_lower and (macd_higher or macd_line_higher or has_green_trend):
            # 计算背驰强度 - 调整评分逻辑，更重视绿柱缩小趋势
            price_diff_pct = (df.loc[first_bottom, 'low'] - df.loc[second_bottom, 'low']) / df.loc[first_bottom, 'low'] * 100 if df.loc[first_bottom, 'low'] != 0 else 0
            macd_diff_pct = (df.loc[second_bottom, 'macd_hist'] - df.loc[first_bottom, 'macd_hist']) / abs(df.loc[first_bottom, 'macd_hist']) * 100 if df.loc[first_bottom, 'macd_hist'] != 0 else 0
            
            # 背驰强度评分 (0-100) - 增加绿柱趋势的权重
            base_score = 50 + (price_diff_pct if price_really_lower else price_diff_pct * 0.5) + abs(macd_diff_pct) / 2
            if has_green_trend:
                base_score += 20  # 绿柱有缩小趋势，加20分
            
            divergence_strength = min(100, max(0, base_score))
            
            strength_level = '强' if divergence_strength > 70 else '中' if divergence_strength > 40 else '弱'
            
            result = {
                'has_divergence': True,
                'type': '底部背驰',
                'strength': strength_level,
                'strength_score': round(divergence_strength, 2),
                'first_bottom_date': df.loc[first_bottom, 'date'] if 'date' in df.columns else str(first_bottom),
                'first_bottom_price': df.loc[first_bottom, 'low'],
                'second_bottom_date': df.loc[second_bottom, 'date'] if 'date' in df.columns else str(second_bottom),
                'second_bottom_price': df.loc[second_bottom, 'low'],
                'price_diff_pct': round(price_diff_pct, 2),
                'macd_diff_pct': round(macd_diff_pct, 2),
                'message': f'检测到{strength_level}底部背驰: 价格创新低但MACD{"绿柱明显缩小" if has_green_trend else "未创新低"}'
            }
            
            # 如果同时有基于绿柱趋势的背驰，选择强度更高的
            if has_green_hist_divergence and green_divergence_details['strength_score'] > result['strength_score']:
                return green_divergence_details
            return result
        
        # 如果传统方法未检测到，但绿柱趋势方法检测到了，返回绿柱趋势的结果
        if has_green_hist_divergence:
            return green_divergence_details
        
        return {
            'has_divergence': False,
            'type': '底部背驰',
            'message': '未检测到底部背驰'
        }
    
    def detect_top_divergence(self):
        """
        检测顶部背驰
        
        Returns:
            顶部背驰检测结果
        """
        df = self.data.tail(self.lookback_period).copy()
        
        # 查找顶部
        top_indices = df[df['is_top']].index.tolist()
        
        if len(top_indices) < 2:
            return {
                'has_divergence': False,
                'type': '顶部背驰',
                'message': '未发现足够的顶部点进行背驰比较'
            }
        
        # 检查最近两个顶部是否形成背驰
        recent_tops = sorted(top_indices)[-2:]
        first_top, second_top = recent_tops
        
        # 价格创新高，但MACD柱高度未创新高（柱状图值更小）
        price_higher = df.loc[second_top, 'high'] > df.loc[first_top, 'high']
        macd_lower = df.loc[second_top, 'macd_hist'] < df.loc[first_top, 'macd_hist']
        
        # 或者价格创新高，但MACD线值未创新高
        macd_line_lower = df.loc[second_top, 'macd_line'] < df.loc[first_top, 'macd_line']
        
        if price_higher and (macd_lower or macd_line_lower):
            # 计算背驰强度
            price_diff_pct = (df.loc[second_top, 'high'] - df.loc[first_top, 'high']) / df.loc[first_top, 'high'] * 100
            macd_diff_pct = (df.loc[first_top, 'macd_hist'] - df.loc[second_top, 'macd_hist']) / abs(df.loc[first_top, 'macd_hist']) * 100 if df.loc[first_top, 'macd_hist'] != 0 else 0
            
            # 背驰强度评分 (0-100)
            divergence_strength = min(100, max(0, 50 + price_diff_pct + abs(macd_diff_pct) / 2))
            
            strength_level = '强' if divergence_strength > 70 else '中' if divergence_strength > 40 else '弱'
            
            return {
                'has_divergence': True,
                'type': '顶部背驰',
                'strength': strength_level,
                'strength_score': round(divergence_strength, 2),
                'first_top_date': df.loc[first_top, 'date'] if 'date' in df.columns else str(first_top),
                'first_top_price': df.loc[first_top, 'high'],
                'second_top_date': df.loc[second_top, 'date'] if 'date' in df.columns else str(second_top),
                'second_top_price': df.loc[second_top, 'high'],
                'price_diff_pct': round(price_diff_pct, 2),
                'macd_diff_pct': round(macd_diff_pct, 2),
                'message': f'检测到{strength_level}顶部背驰: 价格创新高但MACD未创新高'
            }
        
        return {
            'has_divergence': False,
            'type': '顶部背驰',
            'message': '未检测到顶部背驰'
        }
    
    def analyze_macd_trend(self):
        """
        分析MACD趋势状态
        
        Returns:
            MACD趋势分析结果
        """
        df = self.data.tail(30).copy()  # 增加分析周期到30个交易日，更全面地捕捉趋势变化
        
        # MACD线与信号线的关系
        macd_above_signal = df['macd_line'].iloc[-1] > df['macd_line'].iloc[-1]
        macd_cross_signal = (df['macd_line'].iloc[-1] > df['signal_line'].iloc[-1]) and \
                          (df['macd_line'].iloc[-2] <= df['signal_line'].iloc[-2])
        signal_cross_macd = (df['macd_line'].iloc[-1] < df['signal_line'].iloc[-1]) and \
                          (df['macd_line'].iloc[-2] >= df['signal_line'].iloc[-2])
        
        # MACD线位置
        macd_above_zero = df['macd_line'].iloc[-1] > 0
        macd_below_zero = df['macd_line'].iloc[-1] < 0
        
        # MACD柱状图趋势
        hist_increasing = df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-5]
        hist_decreasing = df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-5]
        
        # MACD线趋势
        macd_line_increasing = df['macd_line'].iloc[-1] > df['macd_line'].iloc[-5]
        macd_line_decreasing = df['macd_line'].iloc[-1] < df['macd_line'].iloc[-5]
        
        # 综合判断
        if macd_cross_signal:
            trend_status = '金叉'
        elif signal_cross_macd:
            trend_status = '死叉'
        elif macd_above_zero and hist_increasing:
            trend_status = '多头增强'
        elif macd_above_zero and hist_decreasing:
            trend_status = '多头减弱'
        elif macd_below_zero and hist_decreasing:
            trend_status = '空头增强'
        elif macd_below_zero and hist_increasing:
            trend_status = '空头减弱'
        else:
            trend_status = '震荡'
        
        return {
            'trend_status': trend_status,
            'macd_above_signal': macd_above_signal,
            'macd_above_zero': macd_above_zero,
            'hist_increasing': hist_increasing,
            'macd_line_increasing': macd_line_increasing,
            'latest_macd_line': round(df['macd_line'].iloc[-1], 6),
            'latest_signal_line': round(df['signal_line'].iloc[-1], 6),
            'latest_macd_hist': round(df['macd_hist'].iloc[-1], 6)
        }
    
    def run_full_analysis(self):
        """
        运行完整的MACD背驰分析
        
        Returns:
            综合分析结果
        """
        logger.info("开始MACD指标计算...")
        self.calculate_macd()
        
        logger.info("查找价格高低点...")
        self.find_price_extrema()
        
        logger.info("检测底部背驰...")
        bottom_divergence = self.detect_bottom_divergence()
        
        logger.info("检测顶部背驰...")
        top_divergence = self.detect_top_divergence()
        
        logger.info("分析MACD趋势状态...")
        macd_trend = self.analyze_macd_trend()
        
        # 综合判断
        has_divergence = bottom_divergence['has_divergence'] or top_divergence['has_divergence']
        
        # 优先级：强底部背驰 > 中底部背驰 > 弱底部背驰 > 弱顶部背驰 > 中顶部背驰 > 强顶部背驰
        priority_divergence = None
        if bottom_divergence['has_divergence'] and top_divergence['has_divergence']:
            # 如果同时存在底部和顶部背驰，根据强度决定优先级
            if bottom_divergence['strength_score'] > top_divergence['strength_score']:
                priority_divergence = bottom_divergence
            else:
                priority_divergence = top_divergence
        elif bottom_divergence['has_divergence']:
            priority_divergence = bottom_divergence
        elif top_divergence['has_divergence']:
            priority_divergence = top_divergence
        
        self.analysis_results = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': '512660',
            'has_divergence': has_divergence,
            'priority_divergence': priority_divergence,
            'bottom_divergence': bottom_divergence,
            'top_divergence': top_divergence,
            'macd_trend': macd_trend,
            'latest_price': float(self.data['close'].iloc[-1]),
            'latest_date': str(self.data.index[-1])
        }
        
        return self.analysis_results
    
    def generate_divergence_report(self):
        """
        生成MACD背驰分析报告
        
        Returns:
            格式化的分析报告文本
        """
        if not self.analysis_results:
            return "未执行分析，无报告可生成"
        
        results = self.analysis_results
        
        report = []
        report.append("===== 512660 MACD背驰分析报告 =====")
        report.append(f"分析时间: {results['analysis_time']}")
        report.append(f"最新交易日: {results['latest_date']}")
        report.append(f"最新收盘价: {results['latest_price']:.4f}")
        report.append("")
        
        # MACD指标状态
        report.append("【MACD指标状态】")
        macd_trend = results['macd_trend']
        report.append(f"MACD趋势: {macd_trend['trend_status']}")
        report.append(f"MACD线: {macd_trend['latest_macd_line']:.6f} {'(零轴上方)' if macd_trend['macd_above_zero'] else '(零轴下方)'}")
        report.append(f"信号线: {macd_trend['latest_signal_line']:.6f}")
        report.append(f"MACD柱状图: {macd_trend['latest_macd_hist']:.6f}")
        report.append(f"MACD线趋势: {'上升' if macd_trend['macd_line_increasing'] else '下降'}")
        report.append(f"柱状图趋势: {'扩大' if macd_trend['hist_increasing'] else '缩小'}")
        report.append("")
        
        # 背驰检测结果
        report.append("【背驰检测结果】")
        if results['has_divergence']:
            report.append(f"✓ 检测到背驰信号")
            if results['priority_divergence']:
                div = results['priority_divergence']
                report.append(f"主要背驰类型: {div['type']}")
                report.append(f"背驰强度: {div['strength']} ({div['strength_score']})")
                
                if div['type'] == '底部背驰':
                    report.append(f"第一个底部: {div['first_bottom_date']}, 价格: {div['first_bottom_price']:.4f}")
                    report.append(f"第二个底部: {div['second_bottom_date']}, 价格: {div['second_bottom_price']:.4f}")
                else:
                    report.append(f"第一个顶部: {div['first_top_date']}, 价格: {div['first_top_price']:.4f}")
                    report.append(f"第二个顶部: {div['second_top_date']}, 价格: {div['second_top_price']:.4f}")
                
                report.append(f"价格变化: {div['price_diff_pct']}%")
                report.append(f"MACD变化: {div['macd_diff_pct']}%")
                report.append(f"{div['message']}")
        else:
            report.append("✗ 未检测到背驰信号")
            report.append("底部背驰: " + results['bottom_divergence']['message'])
            report.append("顶部背驰: " + results['top_divergence']['message'])
        report.append("")
        
        # 详细背驰分析
        report.append("【详细背驰分析】")
        report.append("底部背驰: " + ("✓ 存在" if results['bottom_divergence']['has_divergence'] else "✗ 不存在"))
        if results['bottom_divergence']['has_divergence']:
            report.append(f"  强度: {results['bottom_divergence']['strength']}")
            report.append(f"  详情: {results['bottom_divergence']['message']}")
        
        report.append("顶部背驰: " + ("✓ 存在" if results['top_divergence']['has_divergence'] else "✗ 不存在"))
        if results['top_divergence']['has_divergence']:
            report.append(f"  强度: {results['top_divergence']['strength']}")
            report.append(f"  详情: {results['top_divergence']['message']}")
        report.append("")
        
        # 交易建议
        report.append("【交易建议】")
        if results['priority_divergence'] and results['priority_divergence']['type'] == '底部背驰':
            div = results['priority_divergence']
            if div['strength'] == '强':
                report.append("✓ 建议考虑买入")
                report.append("  强底部背驰通常是较好的买入时机")
                report.append(f"  可考虑在{div['second_bottom_price']:.4f}附近设置买入点")
            elif div['strength'] == '中':
                report.append("⚠ 谨慎考虑买入")
                report.append("  中等强度底部背驰，建议结合其他技术指标确认")
                report.append("  可设置小仓位试探性买入")
            else:  # 弱
                report.append("? 观望为主")
                report.append("  弱底部背驰，信号可靠性较低")
                report.append("  建议等待更明确的信号")
        elif results['priority_divergence'] and results['priority_divergence']['type'] == '顶部背驰':
            div = results['priority_divergence']
            report.append("✗ 建议谨慎，考虑减仓")
            report.append(f"  {div['strength']}顶部背驰，可能预示上涨动能减弱")
            report.append("  已有持仓可考虑减仓或设置止盈")
        else:
            # 无明显背驰
            if macd_trend['trend_status'] in ['金叉', '多头增强']:
                report.append("? 观望或轻仓跟进")
                report.append(f"  MACD呈现{macd_trend['trend_status']}，但无明显背驰确认")
                report.append("  可考虑小仓位试探性跟进")
            elif macd_trend['trend_status'] in ['死叉', '空头增强']:
                report.append("✗ 建议观望")
                report.append(f"  MACD呈现{macd_trend['trend_status']}，且无背驰信号")
                report.append("  建议等待明确的底部信号")
            else:
                report.append("? 观望为主")
                report.append("  MACD处于震荡状态，且无明显背驰")
                report.append("  建议等待趋势明确后再做决策")
        report.append("")
        
        # 风险提示
        report.append("【风险提示】")
        report.append("1. 技术分析存在局限性，不构成投资建议")
        report.append("2. 背驰信号可能被市场其他因素影响而失效")
        report.append("3. 建议结合基本面和其他技术指标综合判断")
        report.append("4. 请根据自身风险承受能力制定投资策略")
        report.append("")
        report.append("========================================")
        
        return '\n'.join(report)

def load_data():
    """
    加载512660历史行情数据
    
    Returns:
        处理后的DataFrame
    """
    try:
        # 尝试从数据文件加载
        data_file = os.path.join(os.path.dirname(__file__), 'data', '512660_daily_data.csv')
        if os.path.exists(data_file):
            logger.info(f"从文件加载数据: {data_file}")
            df = pd.read_csv(data_file)
            # 确保日期列格式正确
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        
        # 如果没有数据文件，生成模拟数据（实际使用时应该从数据源获取）
        logger.info("数据文件不存在，生成模拟数据...")
        date_range = pd.date_range(end=datetime.now(), periods=200)
        np.random.seed(42)  # 设置随机种子以确保结果可重现
        
        # 生成模拟价格数据
        base_price = 1.15
        prices = [base_price]
        for _ in range(199):
            # 添加一些随机波动
            change = (np.random.random() - 0.5) * 0.02  # -1%到1%的随机变化
            # 添加一些趋势
            if len(prices) > 100:  # 后半段添加下降趋势
                change -= 0.001
            prices.append(prices[-1] * (1 + change))
        
        # 生成OHLC数据
        open_prices = [p * (1 + (np.random.random() - 0.5) * 0.01) for p in prices]
        high_prices = [max(o, p) * (1 + np.random.random() * 0.01) for o, p in zip(open_prices, prices)]
        low_prices = [min(o, p) * (1 - np.random.random() * 0.01) for o, p in zip(open_prices, prices)]
        volume = [int(np.random.random() * 100000000) for _ in range(200)]
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': volume
        }, index=date_range)
        
        # 确保数据目录存在
        os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
        
        # 保存数据供下次使用
        df.to_csv(data_file)
        logger.info(f"模拟数据已保存至: {data_file}")
        
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def main():
    """
    主函数，分析512660的MACD背驰情况
    """
    try:
        logger.info("开始512660 MACD背驰分析...")
        
        # 加载数据
        data = load_data()
        
        # 验证数据
        if data is None or len(data) < 60:
            logger.error("数据不足，无法进行MACD背驰分析")
            return
        
        logger.info(f"数据加载完成，共{len(data)}条记录")
        
        # 创建分析器实例
        analyzer = MacdDivergenceAnalyzer(data, lookback_period=60)
        
        # 执行分析
        results = analyzer.run_full_analysis()
        
        # 生成报告
        report = analyzer.generate_divergence_report()
        print(report)
        
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = f'results/512660_macd_divergence_analysis_{timestamp}.json'
        txt_file = f'results/512660_macd_divergence_report_{timestamp}.txt'
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析结果已保存至: {json_file}")
        logger.info(f"报告已保存至: {txt_file}")
        logger.info("512660 MACD背驰分析完成！")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()