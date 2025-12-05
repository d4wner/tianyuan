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
    
    def __init__(self, data, lookback_period=90):
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
    
    def find_price_extrema(self, period=8):
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
        检测底部背驰
        
        Returns:
            包含底部背驰信息的字典
        """
        df = self.data.copy()
        divergence_points = []
        
        # 检查是否存在必要的列
        required_columns = ['macd_hist', 'is_bottom', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 从后往前遍历数据，查找可能的底部背驰
        for i in range(len(df) - 1, 0, -1):
            # 如果当前点是底部
            if df.iloc[i]['is_bottom']:
                current_low = df.iloc[i]['low']
                current_macd = df.iloc[i]['macd_hist']
                
                # 查找之前的底部
                for j in range(i - 1, max(-1, i - 30), -1):  # 向前搜索最多30天
                    if df.iloc[j]['is_bottom']:
                        prev_low = df.iloc[j]['low']
                        prev_macd = df.iloc[j]['macd_hist']
                        
                        # 检查是否满足底部背驰条件
                        # 价格创新低，但MACD未创新低（或者绿柱缩小）
                        price_condition = current_low < prev_low * 1.03  # 允许3%的误差
                        macd_condition = current_macd > prev_macd  # MACD未创新低
                        
                        if price_condition and macd_condition:
                            # 计算背驰强度评分
                            price_diff_pct = (prev_low - current_low) / prev_low * 100
                            macd_diff_pct = (current_macd - prev_macd) / abs(prev_macd) * 100 if prev_macd != 0 else 0
                            
                            # 调整背驰强度评分逻辑，增加绿柱趋势权重
                            has_green_trend = (current_macd > 0) or (current_macd > df.iloc[i-1]['macd_hist'] and current_macd > df.iloc[i-2]['macd_hist'])
                            strength_score = min(100, max(0, (price_diff_pct + abs(macd_diff_pct)) * 2 + (20 if has_green_trend else 0)))
                            
                            # 计算背离形成的持续时间（天数）
                            days_between = (df.iloc[i]['date'] - df.iloc[j]['date']).days if hasattr(df.iloc[i]['date'], 'days') else 0
                            
                            # 确定背驰类型
                            divergence_type = "底部背驰"
                            if price_diff_pct > 5:  # 价格下跌超过5%
                                divergence_type = "强烈底部背驰"
                            elif price_diff_pct > 2:  # 价格下跌超过2%
                                divergence_type = "中等强度底部背驰"
                            
                            # 存储背驰信息
                            divergence_info = {
                                'date1': df.iloc[j]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[j]['date'], 'strftime') else str(df.iloc[j]['date']),
                                'price1': float(prev_low),
                                'macd1': float(prev_macd),
                                'date2': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                                'price2': float(current_low),
                                'macd2': float(current_macd),
                                'price_change_pct': float(price_diff_pct),
                                'macd_change_pct': float(macd_diff_pct),
                                'strength_score': float(strength_score),
                                'days_between': int(days_between),
                                'divergence_type': divergence_type,
                                'green_trend': bool(has_green_trend),
                                'description': f"价格创新低{'（绿柱明显缩小）' if has_green_trend else ''}"
                            }
                            
                            divergence_points.append(divergence_info)
                            break  # 找到一个匹配的之前底部后退出内层循环
        
        return divergence_points
    
    def detect_top_divergence(self):
        """
        检测顶部背驰
        
        Returns:
            包含顶部背驰信息的字典
        """
        df = self.data.copy()
        divergence_points = []
        
        # 检查是否存在必要的列
        required_columns = ['macd_hist', 'is_top', 'high']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 从后往前遍历数据，查找可能的顶部背驰
        for i in range(len(df) - 1, 0, -1):
            # 如果当前点是顶部
            if df.iloc[i]['is_top']:
                current_high = df.iloc[i]['high']
                current_macd = df.iloc[i]['macd_hist']
                
                # 查找之前的顶部
                for j in range(i - 1, max(-1, i - 30), -1):  # 向前搜索最多30天
                    if df.iloc[j]['is_top']:
                        prev_high = df.iloc[j]['high']
                        prev_macd = df.iloc[j]['macd_hist']
                        
                        # 检查是否满足顶部背驰条件
                        # 价格创新高，但MACD未创新高
                        price_condition = current_high > prev_high
                        macd_condition = current_macd < prev_macd  # MACD未创新高
                        
                        if price_condition and macd_condition:
                            # 计算背驰强度评分
                            price_diff_pct = (current_high - prev_high) / prev_high * 100
                            macd_diff_pct = (prev_macd - current_macd) / abs(prev_macd) * 100 if prev_macd != 0 else 0
                            strength_score = min(100, max(0, (price_diff_pct + abs(macd_diff_pct)) * 2))
                            
                            # 计算背离形成的持续时间（天数）
                            days_between = (df.iloc[i]['date'] - df.iloc[j]['date']).days if hasattr(df.iloc[i]['date'], 'days') else 0
                            
                            # 确定背驰类型
                            divergence_type = "顶部背驰"
                            if price_diff_pct > 5:  # 价格上涨超过5%
                                divergence_type = "强烈顶部背驰"
                            elif price_diff_pct > 2:  # 价格上涨超过2%
                                divergence_type = "中等强度顶部背驰"
                            
                            # 存储背驰信息
                            divergence_info = {
                                'date1': df.iloc[j]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[j]['date'], 'strftime') else str(df.iloc[j]['date']),
                                'price1': float(prev_high),
                                'macd1': float(prev_macd),
                                'date2': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                                'price2': float(current_high),
                                'macd2': float(current_macd),
                                'price_change_pct': float(price_diff_pct),
                                'macd_change_pct': float(macd_diff_pct),
                                'strength_score': float(strength_score),
                                'days_between': int(days_between),
                                'divergence_type': divergence_type,
                                'description': "价格创新高但MACD未创新高"
                            }
                            
                            divergence_points.append(divergence_info)
                            break  # 找到一个匹配的之前顶部后退出内层循环
        
        return divergence_points
    
    def analyze_macd_trend(self):
        """
        分析MACD趋势
        
        Returns:
            趋势分析结果
        """
        df = self.data.copy()
        
        if 'macd_hist' not in df.columns:
            raise ValueError("缺少MACD柱状图数据")
        
        # 获取最近的MACD数据
        recent_macd = df['macd_hist'].iloc[-10:].values  # 最近10个交易日的MACD柱状图
        current_macd = df['macd_hist'].iloc[-1]
        
        # 计算趋势
        # 1. MACD柱状图趋势（正值增长/负值缩小为向上，负值增长/正值缩小为向下）
        if len(recent_macd) > 1:
            macd_trend = '向上' if recent_macd[-1] > recent_macd[0] else '向下'
        else:
            macd_trend = '持平'
        
        # 2. MACD柱状图当前状态（正值/负值）
        macd_status = '红柱' if current_macd > 0 else '绿柱'
        
        # 3. 趋势变化判断
        if recent_macd[-1] > 0 and recent_macd[0] > 0:
            trend_change = '红柱继续增长' if recent_macd[-1] > recent_macd[0] else '红柱开始缩短'
        elif recent_macd[-1] < 0 and recent_macd[0] < 0:
            trend_change = '绿柱明显缩小' if recent_macd[-1] > recent_macd[0] else '绿柱继续放大'
        elif recent_macd[-1] > 0 and recent_macd[0] < 0:
            trend_change = '由绿柱转为红柱（金叉）'
        elif recent_macd[-1] < 0 and recent_macd[0] > 0:
            trend_change = '由红柱转为绿柱（死叉）'
        else:
            trend_change = '持平'
        
        # 4. 空头/多头趋势判断
        is_bullish = current_macd > 0 or (recent_macd[-1] > recent_macd[-2] and recent_macd[-1] > recent_macd[-3])
        market_trend = '多头趋势' if is_bullish else '空头趋势'
        trend_strength = '强势' if abs(current_macd) > abs(df['macd_hist'].mean()) else '弱势'
        
        return {
            'current_macd': float(current_macd),
            'macd_status': macd_status,
            'macd_trend': macd_trend,
            'trend_change': trend_change,
            'market_trend': market_trend,
            'trend_strength': trend_strength
        }
    
    def run_analysis(self):
        """
        运行完整的背驰分析
        
        Returns:
            分析结果字典
        """
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'date']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # 按照日期排序
        self.data = self.data.sort_values('date')
        
        # 如果回溯周期设置，则只使用最近的指定天数数据
        if self.lookback_period > 0:
            self.data = self.data.tail(self.lookback_period)
        
        # 计算MACD指标
        self.calculate_macd()
        
        # 查找价格极值点
        self.find_price_extrema()
        
        # 检测底部背驰
        bottom_divergences = self.detect_bottom_divergence()
        
        # 检测顶部背驰
        top_divergences = self.detect_top_divergence()
        
        # 分析MACD趋势
        macd_trend = self.analyze_macd_trend()
        
        # 汇总分析结果
        self.analysis_results = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'data_date_range': {
                'start': self.data['date'].min().strftime('%Y-%m-%d'),
                'end': self.data['date'].max().strftime('%Y-%m-%d')
            },
            'data_count': len(self.data),
            'bottom_divergences': bottom_divergences,
            'top_divergences': top_divergences,
            'macd_trend': macd_trend,
            'divergence_summary': {
                'has_bottom_divergence': len(bottom_divergences) > 0,
                'has_top_divergence': len(top_divergences) > 0,
                'strongest_bottom_divergence': max(bottom_divergences, key=lambda x: x['strength_score']) if bottom_divergences else None,
                'strongest_top_divergence': max(top_divergences, key=lambda x: x['strength_score']) if top_divergences else None
            }
        }
        
        return self.analysis_results
    
    def generate_divergence_report(self):
        """
        生成背驰分析报告
        
        Returns:
            报告文本
        """
        if not self.analysis_results:
            raise ValueError("请先运行分析")
        
        report = []
        report.append(f"===== MACD背驰分析报告 =====")
        report.append(f"分析日期: {self.analysis_results['analysis_date']}")
        report.append(f"数据范围: {self.analysis_results['data_date_range']['start']} 至 {self.analysis_results['data_date_range']['end']}")
        report.append(f"数据条数: {self.analysis_results['data_count']} 条")
        report.append("")
        
        # MACD趋势分析
        macd_trend = self.analysis_results['macd_trend']
        report.append(f"----- MACD趋势分析 -----")
        report.append(f"当前MACD柱状图: {macd_trend['macd_status']} ({macd_trend['current_macd']:.6f})")
        report.append(f"MACD趋势: {macd_trend['macd_trend']}")
        report.append(f"趋势变化: {macd_trend['trend_change']}")
        report.append(f"市场趋势: {macd_trend['market_trend']} ({macd_trend['trend_strength']})")
        report.append("")
        
        # 底部背驰分析
        bottom_divergences = self.analysis_results['bottom_divergences']
        report.append(f"----- 底部背驰分析 -----")
        if not bottom_divergences:
            report.append("未检测到底部背驰信号")
        else:
            report.append(f"检测到底部背驰信号数量: {len(bottom_divergences)} 个")
            # 按强度排序
            sorted_bottom_divergences = sorted(bottom_divergences, key=lambda x: x['strength_score'], reverse=True)
            
            for i, divergence in enumerate(sorted_bottom_divergences, 1):
                report.append(f"\n底部背驰 #{i} ({divergence['divergence_type']})")
                report.append(f"  强度评分: {divergence['strength_score']:.2f}/100")
                report.append(f"  第一低点日期: {divergence['date1']}, 价格: {divergence['price1']:.4f}, MACD: {divergence['macd1']:.6f}")
                report.append(f"  第二低点日期: {divergence['date2']}, 价格: {divergence['price2']:.4f}, MACD: {divergence['macd2']:.6f}")
                report.append(f"  价格变化: {divergence['price_change_pct']:.2f}%")
                report.append(f"  MACD变化: {divergence['macd_change_pct']:.2f}%")
                report.append(f"  持续时间: {divergence['days_between']} 天")
                report.append(f"  特征: {divergence['description']}")
        report.append("")
        
        # 顶部背驰分析
        top_divergences = self.analysis_results['top_divergences']
        report.append(f"----- 顶部背驰分析 -----")
        if not top_divergences:
            report.append("未检测到顶部背驰信号")
        else:
            report.append(f"检测到顶部背驰信号数量: {len(top_divergences)} 个")
            # 按强度排序
            sorted_top_divergences = sorted(top_divergences, key=lambda x: x['strength_score'], reverse=True)
            
            for i, divergence in enumerate(sorted_top_divergences, 1):
                report.append(f"\n顶部背驰 #{i} ({divergence['divergence_type']})")
                report.append(f"  强度评分: {divergence['strength_score']:.2f}/100")
                report.append(f"  第一高点日期: {divergence['date1']}, 价格: {divergence['price1']:.4f}, MACD: {divergence['macd1']:.6f}")
                report.append(f"  第二高点日期: {divergence['date2']}, 价格: {divergence['price2']:.4f}, MACD: {divergence['macd2']:.6f}")
                report.append(f"  价格变化: {divergence['price_change_pct']:.2f}%")
                report.append(f"  MACD变化: {divergence['macd_change_pct']:.2f}%")
                report.append(f"  持续时间: {divergence['days_between']} 天")
        report.append("")
        
        # 交易建议
        report.append(f"----- 交易建议 -----")
        has_bottom = self.analysis_results['divergence_summary']['has_bottom_divergence']
        has_top = self.analysis_results['divergence_summary']['has_top_divergence']
        market_trend = macd_trend['market_trend']
        trend_strength = macd_trend['trend_strength']
        
        if has_bottom and not has_top:
            strongest_bottom = self.analysis_results['divergence_summary']['strongest_bottom_divergence']
            strength_score = strongest_bottom['strength_score'] if strongest_bottom else 0
            
            if strength_score > 70:
                report.append("强烈建议买入：检测到高强度底部背驰信号，MACD趋势正在改善")
            elif strength_score > 50:
                report.append("建议买入：检测到底部背驰信号，可考虑小仓位试探性买入")
            else:
                report.append("谨慎买入：检测到弱底部背驰信号，可小仓位试探")
        elif has_top and not has_bottom:
            strongest_top = self.analysis_results['divergence_summary']['strongest_top_divergence']
            strength_score = strongest_top['strength_score'] if strongest_top else 0
            
            if strength_score > 70:
                report.append("强烈建议卖出：检测到高强度顶部背驰信号，MACD趋势正在恶化")
            elif strength_score > 50:
                report.append("建议卖出：检测到顶部背驰信号，考虑减仓或清仓")
            else:
                report.append("谨慎卖出：检测到弱顶部背驰信号，可考虑适当减仓")
        elif has_bottom and has_top:
            # 比较两者强度
            bottom_strength = self.analysis_results['divergence_summary']['strongest_bottom_divergence']['strength_score'] if self.analysis_results['divergence_summary']['strongest_bottom_divergence'] else 0
            top_strength = self.analysis_results['divergence_summary']['strongest_top_divergence']['strength_score'] if self.analysis_results['divergence_summary']['strongest_top_divergence'] else 0
            
            if bottom_strength > top_strength + 20:
                report.append("建议买入：底部背驰信号强度明显强于顶部背驰信号")
            elif top_strength > bottom_strength + 20:
                report.append("建议卖出：顶部背驰信号强度明显强于底部背驰信号")
            else:
                report.append("信号复杂：同时检测到底部和顶部背驰信号，强度相近，建议观望")
        else:
            if market_trend == '多头趋势' and trend_strength == '强势':
                report.append("趋势良好：当前处于多头强势趋势，未检测到背驰信号，可继续持有")
            elif market_trend == '空头趋势' and trend_strength == '强势':
                report.append("趋势弱势：当前处于空头强势趋势，未检测到背驰信号，建议观望")
            else:
                report.append("横盘震荡：未检测到明显趋势和背驰信号，建议观望或小仓位操作")
        
        report.append("")
        report.append(f"----- 风险提示 -----")
        report.append("1. 背驰信号不是百分百准确，需要结合其他技术指标和基本面分析")
        report.append("2. 市场有风险，投资需谨慎，建议严格控制仓位和止损")
        report.append("3. 本报告仅供参考，不构成任何投资建议")
        
        return '\n'.join(report)

def load_data(symbol="512690", is_simulation=False):
    """
    加载数据
    
    Args:
        symbol: ETF代码
        is_simulation: 是否使用模拟数据
    
    Returns:
        DataFrame: 加载的数据
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if is_simulation:
        # 生成模拟数据用于测试
        logger.info(f"生成{symbol}的模拟数据...")
        dates = pd.date_range(end=datetime.now(), periods=90)
        np.random.seed(42)  # 设置随机种子以保证结果可重复
        
        # 创建一个模拟的下降后上升趋势，以测试底部背驰检测
        base_price = 1.2
        prices = []
        for i in range(len(dates)):
            # 先下降后上升的模式
            if i < 45:
                # 下降趋势，添加一些随机波动
                price = base_price * (1 - i * 0.005) + np.random.normal(0, 0.005)
            else:
                # 上升趋势，添加一些随机波动
                price = base_price * (1 - 0.225 + (i - 45) * 0.004) + np.random.normal(0, 0.005)
            prices.append(max(price, 0.01))  # 确保价格不为负
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'high': [p * (1 + np.random.normal(0.001, 0.002)) for p in prices],
            'low': [p * (1 - np.random.normal(0.001, 0.002)) for p in prices],
            'close': prices,
            'volume': [int(np.random.normal(100000, 20000)) for _ in prices]
        })
        
        # 保存模拟数据到文件
        sim_dir = os.path.join(current_dir, 'data', 'simulation')
        os.makedirs(sim_dir, exist_ok=True)
        sim_file = os.path.join(sim_dir, f"{symbol}_simulation.csv")
        df.to_csv(sim_file, index=False, encoding='utf-8')
        logger.info(f"模拟数据已保存至：{sim_file}")
        return df
    else:
        # 加载真实数据
        data_dir = os.path.join(current_dir, 'data', 'daily')
        data_file = os.path.join(data_dir, f"{symbol}_daily.csv")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在：{data_file}")
        
        logger.info(f"加载{symbol}的日线数据：{data_file}")
        df = pd.read_csv(data_file)
        
        # 确保日期列是日期类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'trade_date' in df.columns:
            df['date'] = pd.to_datetime(df['trade_date'])
            df = df.drop('trade_date', axis=1)
        else:
            raise ValueError("数据文件中没有日期列")
        
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据文件缺少必要的列：{col}")
        
        logger.info(f"数据加载成功，共{len(df)}条记录")
        return df

def main(symbol="512690", lookback_period=90, is_simulation=False, output_dir=None):
    """
    主函数
    
    Args:
        symbol: ETF代码
        lookback_period: 回溯分析的周期数
        is_simulation: 是否使用模拟数据
        output_dir: 输出目录
    """
    try:
        # 加载数据
        data = load_data(symbol=symbol, is_simulation=is_simulation)
        
        # 初始化分析器
        analyzer = MacdDivergenceAnalyzer(data, lookback_period=lookback_period)
        
        # 运行分析
        results = analyzer.run_analysis()
        
        # 生成报告
        report = analyzer.generate_divergence_report()
        print(report)
        
        # 设置输出目录
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果到JSON文件
        json_file = os.path.join(output_dir, f"{symbol}_macd_divergence_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"分析结果已保存至：{json_file}")
        
        # 保存报告到文本文件
        report_file = os.path.join(output_dir, f"{symbol}_macd_divergence_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"分析报告已保存至：{report_file}")
        
        return results, report
    
    except Exception as e:
        logger.error(f"分析过程中出现错误：{str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MACD背驰分析工具')
    parser.add_argument('--symbol', type=str, default='512690', help='ETF代码，默认为512690')
    parser.add_argument('--lookback', type=int, default=90, help='回溯分析的天数，默认为90天')
    parser.add_argument('--simulation', action='store_true', help='是否使用模拟数据')
    args = parser.parse_args()
    
    main(symbol=args.symbol, lookback_period=args.lookback, is_simulation=args.simulation)