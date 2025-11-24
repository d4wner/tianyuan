"""缠论图表绘制器"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime  # 正确导入，支持 datetime.now()
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def setup_matplotlib_font():
    """设置matplotlib字体，避免指定可能不存在的中文字体"""
    try:
        plt.rcParams["font.family"] = ['sans-serif']
        plt.rcParams["font.sans-serif"] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial', 'Helvetica', 'Verdana']
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号方块问题
        logger.info("字体配置完成（使用通用字体列表）")
    except Exception as e:
        logger.warning(f"字体配置警告: {str(e)}，中文可能显示异常")

class ChanlunPlotter:
    """缠论图表绘制器"""
    
    def __init__(self, config=None):
        """
        初始化绘制器
        :param config: 绘图配置
        """
        # 初始化时配置字体
        setup_matplotlib_font()
        
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'outputs/plots')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("缠论绘图器初始化完成")
    
    def plot(self, df, symbol):
        """
        绘制缠论图表
        :param df: 包含缠论指标的DataFrame
        :param symbol: 股票代码
        """
        if df.empty:
            logger.warning("数据为空，无法绘图")
            return
            
        try:
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制价格曲线
            plt.plot(df['date'], df['close'], label='收盘价', color='blue')
            
            # 标记分型
            if 'top_fractal' in df.columns:
                top_fractals = df[df['top_fractal']]
                plt.scatter(top_fractals['date'], top_fractals['high'], 
                           marker='v', color='red', label='顶分型')
            
            if 'bottom_fractal' in df.columns:
                bottom_fractals = df[df['bottom_fractal']]
                plt.scatter(bottom_fractals['date'], bottom_fractals['low'], 
                           marker='^', color='green', label='底分型')
            
            # 标记笔
            if 'pen_start' in df.columns:
                pen_starts = df[df['pen_start']]
                plt.scatter(pen_starts['date'], pen_starts['close'], 
                           marker='o', color='purple', label='笔起点')
            
            if 'pen_end' in df.columns:
                pen_ends = df[df['pen_end']]
                plt.scatter(pen_ends['date'], pen_ends['close'], 
                           marker='s', color='orange', label='笔终点')
            
            # 标记中枢
            if 'central_bank' in df.columns:
                central_banks = df[df['central_bank']]
                for idx, row in central_banks.iterrows():
                    # 确保日期是datetime类型
                    if not pd.api.types.is_datetime64_any_dtype(row['date']):
                        start_date = pd.to_datetime(row['date'])
                    else:
                        start_date = row['date']
                    end_date = start_date + pd.Timedelta(days=1)
                    plt.axvspan(start_date, end_date, 
                               alpha=0.2, color='gray', label='中枢' if idx == 0 else "")
            
            # 设置图表属性
            plt.title(f"缠论分析 - {symbol}")
            plt.xlabel("日期")
            plt.ylabel("价格")
            plt.legend()
            plt.grid(True)
            
            # 自动调整日期标签显示
            plt.gcf().autofmt_xdate()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')  # 确保标签完整显示
            plt.close()
            
            logger.info(f"图表已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}")
    
    def plot_minute(self, df, symbol, period):
        """
        绘制分钟数据图表
        :param df: 分钟数据DataFrame
        :param symbol: 股票代码
        :param period: 分钟周期
        """
        if df.empty:
            logger.warning("分钟数据为空，无法绘图")
            return
            
        try:
            # 创建图表
            plt.figure(figsize=(15, 8))
            
            # 绘制价格曲线
            plt.plot(df['date'], df['close'], label='收盘价', color='blue')
            
            # 设置图表属性
            plt.title(f"{symbol} {period}分钟图")
            plt.xlabel("时间")
            plt.ylabel("价格")
            plt.legend()
            plt.grid(True)
            
            # 自动调整时间标签显示
            plt.gcf().autofmt_xdate()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{period}_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
            logger.info(f"分钟图表已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"分钟数据绘图失败: {str(e)}")
    
    def plot_price_with_signals(self, df, symbol, trade_records):
        """
        绘制带有交易信号的价格图表
        :param df: 价格数据DataFrame
        :param symbol: 股票代码
        :param trade_records: 交易记录对象列表
        """
        if df.empty:
            logger.warning("数据为空，无法绘图")
            return
            
        try:
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制价格曲线
            plt.plot(df['date'], df['close'], label='收盘价', color='blue')
            
            # 标记买入信号
            buy_signals = [record for record in trade_records if record.type == 'buy']
            for i, signal in enumerate(buy_signals):
                date = pd.to_datetime(signal.date)
                price = signal.price
                plt.scatter(date, price, marker='^', color='green', s=100, label='买入信号' if i == 0 else "")
            
            # 标记卖出信号
            sell_signals = [record for record in trade_records if record.type == 'sell']
            for i, signal in enumerate(sell_signals):
                date = pd.to_datetime(signal.date)
                price = signal.price
                plt.scatter(date, price, marker='v', color='red', s=100, label='卖出信号' if i == 0 else "")
            
            # 设置图表属性
            plt.title(f"{symbol} 价格与交易信号")
            plt.xlabel("日期")
            plt.ylabel("价格")
            plt.legend()
            plt.grid(True)
            
            # 自动调整日期标签显示
            plt.gcf().autofmt_xdate()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_price_with_signals_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
            logger.info(f"带交易信号的价格图表已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"绘制带交易信号的价格图表失败: {str(e)}")
    
    def plot_performance_metrics(self, performance, symbol):
        """
        绘制性能指标图表
        :param performance: 包含性能指标的字典
        :param symbol: 股票代码
        """
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"{symbol} 回测性能指标", fontsize=16)
            
            # 1. 总收益图
            if 'cumulative_returns' in performance and 'dates' in performance:
                axes[0, 0].plot(performance['dates'], performance['cumulative_returns'])
                axes[0, 0].set_title('累计收益率')
                axes[0, 0].grid(True)
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 胜率饼图
            if 'win_rate' in performance:
                win_rate = performance['win_rate']
                lose_rate = 1 - win_rate
                axes[0, 1].pie([win_rate, lose_rate], labels=['盈利', '亏损'], 
                              autopct='%1.1f%%', colors=['green', 'red'])
                axes[0, 1].set_title('胜率分布')
            
            # 3. 交易次数柱状图
            if 'long_count' in performance and 'short_count' in performance:
                axes[1, 0].bar(['多头', '空头'], [performance['long_count'], performance['short_count']],
                              color=['blue', 'orange'])
                axes[1, 0].set_title('交易次数分布')
                axes[1, 0].grid(True, axis='y')
            
            # 4. 最大回撤图
            if 'max_drawdown' in performance and 'drawdown_dates' in performance and 'drawdown_values' in performance:
                axes[1, 1].plot(performance['drawdown_dates'], performance['drawdown_values'])
                axes[1, 1].axhline(y=performance['max_drawdown'], color='r', linestyle='--', 
                                  label=f'最大回撤: {performance["max_drawdown"]:.2%}')
                axes[1, 1].set_title('回撤曲线')
                axes[1, 1].grid(True)
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_performance_metrics_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
            logger.info(f"性能指标图表已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"绘制性能指标图表失败: {str(e)}")