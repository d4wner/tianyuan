#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512660日线买点分析脚本

该脚本用于分析512660的日线买点信号，包括：
1. 日线二买（核心买点）
2. 日线一买（辅助买点）
3. 日线三买（辅助买点）
4. 破中枢反抽（兜底买点）

作者: TradeTianYuan
日期: 2025-11-27
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from daily_buy_signal_detector import BuySignalDetector
from data_validator import DataValidator

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyBuySignalAnalyzer:
    """日线买点分析器类"""
    
    def __init__(self):
        """初始化分析器"""
        self.detector = BuySignalDetector()
        self.validator = DataValidator()
        self.symbol = "512660"
        self.daily_data_path = os.path.join("data", "daily", f"{self.symbol}_daily.csv")
        self.results_dir = os.path.join("results")
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_daily_data(self) -> pd.DataFrame:
        """加载日线数据"""
        logger.info(f"加载{self.symbol}日线数据...")
        
        try:
            df = pd.read_csv(self.daily_data_path)
            
            # 确保日期列格式正确
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'trade_date' in df.columns:
                df['date'] = pd.to_datetime(df['trade_date'])
                df = df.rename(columns={'trade_date': 'date'})
            
            # 确保数据按日期排序
            df = df.sort_values('date')
            
            logger.info(f"成功加载{len(df)}条日线数据")
            return df
        except Exception as e:
            logger.error(f"加载日线数据失败: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证数据有效性"""
        logger.info(f"验证{self.symbol}日线数据有效性...")
        
        # 使用DataValidator验证数据
        daily_validation_result = self.validator.validate_daily_data(df)
        
        logger.info(f"数据验证结果: {'有效' if daily_validation_result['valid'] else '无效'}")
        logger.info(f"{daily_validation_result['reason']}")
        
        return daily_validation_result
    
    def calculate_additional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算额外指标用于辅助分析"""
        df_copy = df.copy()
        
        # 计算均线
        df_copy['ma5'] = df_copy['close'].rolling(window=5).mean()
        df_copy['ma10'] = df_copy['close'].rolling(window=10).mean()
        df_copy['ma20'] = df_copy['close'].rolling(window=20).mean()
        df_copy['ma60'] = df_copy['close'].rolling(window=60).mean()
        
        # 计算RSI
        delta = df_copy['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算MACD (复制检测器中的实现)
        ema_fast = df_copy['close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = df_copy['close'].ewm(span=26, adjust=False, min_periods=26).mean()
        df_copy['macd_diff'] = ema_fast - ema_slow
        df_copy['macd_dea'] = df_copy['macd_diff'].ewm(span=9, adjust=False, min_periods=9).mean()
        df_copy['macd_hist'] = df_copy['macd_diff'] - df_copy['macd_dea']
        
        return df_copy
    
    def analyze_ma_system(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析均线系统"""
        # 获取最近的均线值
        recent = df.iloc[-1]
        
        # 判断均线多头排列
        ma_bullish = (recent['ma5'] > recent['ma10'] and 
                     recent['ma10'] > recent['ma20'] and 
                     recent['ma20'] > recent['ma60'])
        
        # 判断均线空头排列
        ma_bearish = (recent['ma5'] < recent['ma10'] and 
                     recent['ma10'] < recent['ma20'] and 
                     recent['ma20'] < recent['ma60'])
        
        # 判断当前价格相对于均线的位置
        price_above_ma5 = recent['close'] > recent['ma5']
        price_above_ma10 = recent['close'] > recent['ma10']
        price_above_ma20 = recent['close'] > recent['ma20']
        price_above_ma60 = recent['close'] > recent['ma60']
        
        return {
            'ma_bullish': ma_bullish,
            'ma_bearish': ma_bearish,
            'ma_position': {
                'price_above_ma5': price_above_ma5,
                'price_above_ma10': price_above_ma10,
                'price_above_ma20': price_above_ma20,
                'price_above_ma60': price_above_ma60
            },
            'current_ma_values': {
                'ma5': round(float(recent['ma5']), 4),
                'ma10': round(float(recent['ma10']), 4),
                'ma20': round(float(recent['ma20']), 4),
                'ma60': round(float(recent['ma60']), 4)
            }
        }
    
    def analyze_volume_price_relationship(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析量价关系"""
        recent_data = df.tail(20)
        
        # 计算量价配合度
        price_changes = recent_data['close'].pct_change()
        volume_changes = recent_data['volume'].pct_change()
        
        # 上涨时有量，下跌时缩量为良好配合
        good_relationship_days = 0
        for i in range(1, len(recent_data)):
            if (price_changes.iloc[i] > 0 and volume_changes.iloc[i] > 0) or \
               (price_changes.iloc[i] < 0 and volume_changes.iloc[i] < 0):
                good_relationship_days += 1
        
        # 最近5天平均成交量 vs 前15天平均成交量
        recent_5d_avg_vol = recent_data['volume'].tail(5).mean()
        previous_15d_avg_vol = recent_data['volume'].head(15).mean()
        volume_ratio = recent_5d_avg_vol / previous_15d_avg_vol if previous_15d_avg_vol > 0 else 0
        
        return {
            'good_relationship_ratio': good_relationship_days / len(recent_data) * 100,
            'volume_trend': '放量' if volume_ratio > 1.2 else '缩量' if volume_ratio < 0.8 else '正常',
            'volume_ratio': round(float(volume_ratio), 2),
            'recent_5d_avg_volume': int(recent_5d_avg_vol),
            'previous_15d_avg_volume': int(previous_15d_avg_vol)
        }
    
    def generate_comprehensive_analysis(self, df: pd.DataFrame, 
                                      buy_signal_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合分析报告"""
        logger.info("生成综合分析报告...")
        
        # 计算额外指标
        df_with_indicators = self.calculate_additional_indicators(df)
        
        # 分析均线系统
        ma_analysis = self.analyze_ma_system(df_with_indicators)
        
        # 分析量价关系
        volume_price_analysis = self.analyze_volume_price_relationship(df)
        
        # 获取最新数据
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # 计算近期涨跌幅
        daily_change_pct = (latest['close'] - latest['open']) / latest['open'] * 100
        weekly_change_pct = (latest['close'] - df.iloc[-6]['close']) / df.iloc[-6]['close'] * 100 if len(df) > 6 else 0
        monthly_change_pct = (latest['close'] - df.iloc[-22]['close']) / df.iloc[-22]['close'] * 100 if len(df) > 22 else 0
        
        # 分析MACD状态
        recent_macd = df_with_indicators.iloc[-1]
        macd_status = {
            'diff_above_zero': recent_macd['macd_diff'] > 0,
            'dea_above_zero': recent_macd['macd_dea'] > 0,
            'histogram_above_zero': recent_macd['macd_hist'] > 0,
            'diff_dea_relationship': '金叉' if recent_macd['macd_diff'] > recent_macd['macd_dea'] else '死叉',
            'current_values': {
                'diff': round(float(recent_macd['macd_diff']), 4),
                'dea': round(float(recent_macd['macd_dea']), 4),
                'hist': round(float(recent_macd['macd_hist']), 4)
            }
        }
        
        # 分析RSI状态
        rsi_status = {
            'value': round(float(recent_macd['rsi']), 2) if not pd.isna(recent_macd['rsi']) else None,
            'overbought': recent_macd['rsi'] > 70 if not pd.isna(recent_macd['rsi']) else False,
            'oversold': recent_macd['rsi'] < 30 if not pd.isna(recent_macd['rsi']) else False
        }
        
        comprehensive = {
            'recent_price_data': {
                'latest_date': latest['date'].strftime('%Y-%m-%d'),
                'latest_close': round(float(latest['close']), 4),
                'latest_open': round(float(latest['open']), 4),
                'latest_high': round(float(latest['high']), 4),
                'latest_low': round(float(latest['low']), 4),
                'latest_volume': int(latest['volume'])
            },
            'price_changes': {
                'daily_change_pct': round(float(daily_change_pct), 2),
                'weekly_change_pct': round(float(weekly_change_pct), 2),
                'monthly_change_pct': round(float(monthly_change_pct), 2)
            },
            'ma_analysis': ma_analysis,
            'volume_price_analysis': volume_price_analysis,
            'macd_status': macd_status,
            'rsi_status': rsi_status,
            'market_context': self.analyze_market_context(df_with_indicators)
        }
        
        return comprehensive
    
    def analyze_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析市场整体环境"""
        # 最近30天的市场表现
        recent_30d = df.tail(30)
        
        # 计算波动率
        volatility = recent_30d['close'].pct_change().std() * np.sqrt(252) * 100
        
        # 判断趋势强度
        trend_strength = abs((recent_30d['close'].iloc[-1] - recent_30d['close'].iloc[0]) / 
                           recent_30d['close'].iloc[0] * 100)
        
        # 计算支撑和阻力位（简化版）
        recent_high = recent_30d['high'].max()
        recent_low = recent_30d['low'].min()
        
        # 判断当前位置
        current_price = df.iloc[-1]['close']
        position_in_range = (current_price - recent_low) / (recent_high - recent_low) * 100
        
        return {
            'volatility': round(float(volatility), 2),
            'trend_strength': round(float(trend_strength), 2),
            'price_range': {
                'recent_high': round(float(recent_high), 4),
                'recent_low': round(float(recent_low), 4),
                'current_position_pct': round(float(position_in_range), 2)
            }
        }
    
    def generate_summary_report(self, buy_signal_result: Dict[str, Any], 
                              comprehensive_analysis: Dict[str, Any]) -> str:
        """生成汇总报告"""
        report_lines = [f"===== {self.symbol}日线买点综合分析报告 ====="]
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 基础信息
        report_lines.append("【基础信息】")
        report_lines.append(f"最新交易日: {comprehensive_analysis['recent_price_data']['latest_date']}")
        report_lines.append(f"最新收盘价: {comprehensive_analysis['recent_price_data']['latest_close']}")
        report_lines.append(f"日涨跌幅: {comprehensive_analysis['price_changes']['daily_change_pct']}%")
        report_lines.append(f"周涨跌幅: {comprehensive_analysis['price_changes']['weekly_change_pct']}%")
        report_lines.append(f"月涨跌幅: {comprehensive_analysis['price_changes']['monthly_change_pct']}%")
        report_lines.append("")
        
        # 买点信号
        report_lines.append("【买点信号分析】")
        report_lines.append(f"最强买点信号: {buy_signal_result['strongest_signal']}")
        report_lines.append(f"信号优先级: {buy_signal_result['signal_type_priority']}")
        report_lines.append(f"满足买点数量: {buy_signal_result['satisfied_signals_count']}/4")
        report_lines.append("")
        
        # 详细买点信号
        report_lines.append("【详细买点信号】")
        signals = buy_signal_result['signals']
        
        # 二买
        report_lines.append(f"1. 日线二买（核心）: {'✓ 满足' if signals['second_buy']['detected'] else '✗ 不满足'}")
        if signals['second_buy']['detected']:
            details = signals['second_buy']['details']
            report_lines.append(f"   - 一买价格: {details['first_buy']['price']}")
            report_lines.append(f"   - 回调最低点: {details['callback']['low']}")
        
        # 一买
        report_lines.append(f"2. 日线一买（辅助）: {'✓ 满足' if signals['first_buy']['detected'] else '✗ 不满足'}")
        
        # 三买
        report_lines.append(f"3. 日线三买（辅助）: {'✓ 满足' if signals['third_buy']['detected'] else '✗ 不满足'}")
        
        # 反抽
        report_lines.append(f"4. 破中枢反抽（兜底）: {'✓ 满足' if signals['reverse_pullback']['detected'] else '✗ 不满足'}")
        report_lines.append("")
        
        # 技术指标分析
        report_lines.append("【技术指标分析】")
        
        # 均线分析
        ma_analysis = comprehensive_analysis['ma_analysis']
        report_lines.append(f"均线状态: {'多头排列' if ma_analysis['ma_bullish'] else '空头排列' if ma_analysis['ma_bearish'] else '震荡'}")
        report_lines.append(f"当前价格位置: ")
        report_lines.append(f"  - 相对MA5: {'上方' if ma_analysis['ma_position']['price_above_ma5'] else '下方'}")
        report_lines.append(f"  - 相对MA10: {'上方' if ma_analysis['ma_position']['price_above_ma10'] else '下方'}")
        report_lines.append(f"  - 相对MA20: {'上方' if ma_analysis['ma_position']['price_above_ma20'] else '下方'}")
        report_lines.append(f"  - 相对MA60: {'上方' if ma_analysis['ma_position']['price_above_ma60'] else '下方'}")
        
        # MACD分析
        macd = comprehensive_analysis['macd_status']
        report_lines.append(f"MACD状态: {macd['diff_dea_relationship']}")
        report_lines.append(f"MACD位置: ")
        report_lines.append(f"  - DIFF: {'零轴上方' if macd['diff_above_zero'] else '零轴下方'}")
        report_lines.append(f"  - DEA: {'零轴上方' if macd['dea_above_zero'] else '零轴下方'}")
        report_lines.append(f"  - 柱状图: {'红柱' if macd['histogram_above_zero'] else '绿柱'}")
        
        # RSI分析
        rsi = comprehensive_analysis['rsi_status']
        if rsi['value'] is not None:
            report_lines.append(f"RSI: {rsi['value']} ({'超买' if rsi['overbought'] else '超卖' if rsi['oversold'] else '正常'})")
        
        # 量价分析
        vol_price = comprehensive_analysis['volume_price_analysis']
        report_lines.append(f"量价配合度: {vol_price['good_relationship_ratio']:.1f}%")
        report_lines.append(f"成交量趋势: {vol_price['volume_trend']} (近期5日/前15日: {vol_price['volume_ratio']:.2f}倍)")
        report_lines.append("")
        
        # 交易建议
        report_lines.append("【交易建议】")
        if buy_signal_result['strongest_signal'] == "日线二买":
            report_lines.append("✓ 日线二买（核心买点）确认，建议重点关注")
            report_lines.append("  - 优先匹配30分钟向上笔建仓（子仓位比例60%-70%）")
            report_lines.append("  - 加仓优先级最高")
        elif buy_signal_result['strongest_signal'] == "日线一买":
            report_lines.append("△ 日线一买（辅助买点）确认")
            report_lines.append("  - 可匹配15分钟向上笔建仓（子仓位比例20%-40%）")
            report_lines.append("  - 需谨慎，建议控制仓位")
        elif buy_signal_result['strongest_signal'] == "日线三买":
            report_lines.append("△ 日线三买（辅助买点）确认")
            report_lines.append("  - 可匹配15分钟向上笔建仓（子仓位比例20%-40%）")
            report_lines.append("  - 需确认突破有效性")
        elif buy_signal_result['strongest_signal'] == "破中枢反抽":
            report_lines.append("! 破中枢反抽（兜底买点）确认")
            report_lines.append("  - 仅作为兜底策略，建议最小仓位试探")
            report_lines.append("  - 严格设置止损")
        else:
            report_lines.append("✗ 当前无明确日线买点信号")
            report_lines.append("  - 建议继续等待信号明确")
            report_lines.append("  - 优先关注周线多头趋势下的日线二买")
        
        # 综合风险提示
        report_lines.append("")
        report_lines.append("【风险提示】")
        report_lines.append("1. 交易有风险，入市需谨慎")
        report_lines.append("2. 技术分析仅供参考，不构成投资建议")
        report_lines.append("3. 请结合自身风险承受能力制定投资策略")
        report_lines.append("4. 市场环境变化快，请实时关注最新行情")
        
        report_lines.append("" * 2)
        report_lines.append("=====================================================")
        
        return "\n".join(report_lines)
    
    def save_results(self, buy_signal_result: Dict[str, Any], 
                    comprehensive_analysis: Dict[str, Any], 
                    summary_report: str):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果（JSON）
        detailed_result = {
            'symbol': self.symbol,
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'buy_signal_result': buy_signal_result,
            'comprehensive_analysis': comprehensive_analysis
        }
        
        json_file_path = os.path.join(self.results_dir, 
                                     f"{self.symbol}_daily_buy_signal_analysis_{timestamp}.json")
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_result, f, ensure_ascii=False, indent=2, 
                     default=str)  # 使用default=str处理datetime对象
        
        # 保存文本报告
        report_file_path = os.path.join(self.results_dir, 
                                      f"{self.symbol}_daily_buy_signal_report_{timestamp}.txt")
        
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        logger.info(f"分析结果已保存至: {json_file_path}")
        logger.info(f"报告已保存至: {report_file_path}")
    
    def run(self):
        """运行分析"""
        logger.info(f"开始分析{self.symbol}日线买点信号...")
        
        try:
            # 1. 加载数据
            df = self.load_daily_data()
            
            # 2. 验证数据
            validation_result = self.validate_data(df)
            
            if not validation_result['valid']:
                logger.warning(f"数据验证失败: {validation_result['reason']}")
                logger.warning("尝试继续分析，但结果可能不准确")
            
            # 3. 检测买点信号
            buy_signal_result = self.detector.detect_buy_signals(df)
            
            # 4. 生成综合分析
            comprehensive_analysis = self.generate_comprehensive_analysis(df, buy_signal_result)
            
            # 5. 生成汇总报告
            summary_report = self.generate_summary_report(buy_signal_result, comprehensive_analysis)
            
            # 6. 保存结果
            self.save_results(buy_signal_result, comprehensive_analysis, summary_report)
            
            # 7. 输出报告
            print(summary_report)
            
            logger.info(f"{self.symbol}日线买点分析完成！")
            return buy_signal_result
            
        except Exception as e:
            logger.error(f"分析过程中发生错误: {str(e)}")
            raise


if __name__ == "__main__":
    import sys
    
    try:
        analyzer = DailyBuySignalAnalyzer()
        analyzer.run()
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)