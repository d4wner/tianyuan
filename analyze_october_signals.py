#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析2025年10月军工ETF交易信号合理性
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SignalAnalyzer')

def load_price_data(file_path):
    """加载价格数据"""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"成功加载价格数据，共{len(df)}条记录")
        return df
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        raise

def extract_signals_from_report():
    """从报告中提取交易信号"""
    # 根据之前看到的报告内容，手动提取信号
    signals = [
        {'date': '2025-10-14', 'type': '买入', 'price': 1.192, 'strength': 0.933, 'level': '二买', 'reason': '底分型形成 + 买入信号'},
        {'date': '2025-10-17', 'type': '卖出', 'price': 1.196, 'strength': 0.133, 'level': '三卖', 'reason': '顶分型形成 + 卖出信号'},
        {'date': '2025-10-21', 'type': '买入', 'price': 1.200, 'strength': 0.533, 'level': '二买', 'reason': '底分型形成 + 买入信号'},
        {'date': '2025-10-28', 'type': '买入', 'price': 1.249, 'strength': 0.300, 'level': '一买', 'reason': '底分型形成 + 买入信号'},
        {'date': '2025-10-29', 'type': '卖出', 'price': 1.233, 'strength': 0.133, 'level': '三卖', 'reason': '顶分型形成 + 卖出信号'},
        {'date': '2025-10-30', 'type': '买入', 'price': 1.230, 'strength': 0.133, 'level': '三买', 'reason': '底分型形成 + 买入信号'},
        {'date': '2025-10-31', 'type': '卖出', 'price': 1.218, 'strength': 0.133, 'level': '三卖', 'reason': '顶分型形成 + 卖出信号'}
    ]
    
    # 转换为DataFrame并添加日期索引
    df_signals = pd.DataFrame(signals)
    df_signals['date'] = pd.to_datetime(df_signals['date'])
    logger.info(f"成功提取{len(df_signals)}个交易信号")
    return df_signals

def analyze_signal_accuracy(df_prices, df_signals):
    """分析信号准确性"""
    results = []
    
    for _, signal in df_signals.iterrows():
        # 查找信号日期的实际价格
        signal_date = signal['date']
        price_row = df_prices[df_prices['date'] == signal_date]
        
        if not price_row.empty:
            actual_price = price_row.iloc[0]['close']
            price_diff = abs(actual_price - signal['price']) / actual_price * 100
            
            # 分析信号前后的价格走势
            signal_idx = df_prices[df_prices['date'] == signal_date].index[0]
            
            # 检查信号前后各3天的价格
            lookback = min(3, signal_idx)
            lookforward = min(3, len(df_prices) - signal_idx - 1)
            
            if signal['type'] == '买入':
                # 买入信号后是否上涨
                future_prices = df_prices.iloc[signal_idx+1:signal_idx+1+lookforward]['close'].values
                max_future_price = max(future_prices) if len(future_prices) > 0 else actual_price
                future_return = (max_future_price - actual_price) / actual_price * 100
                success = future_return > 0
            else:  # 卖出信号
                # 卖出信号后是否下跌
                future_prices = df_prices.iloc[signal_idx+1:signal_idx+1+lookforward]['close'].values
                min_future_price = min(future_prices) if len(future_prices) > 0 else actual_price
                future_return = (min_future_price - actual_price) / actual_price * 100
                success = future_return < 0
            
            result = {
                'date': signal_date.strftime('%Y-%m-%d'),
                'type': signal['type'],
                'signal_price': signal['price'],
                'actual_price': actual_price,
                'price_accuracy': 100 - price_diff,
                'future_return': future_return,
                'signal_success': success,
                'level': signal['level'],
                'strength': signal['strength']
            }
            results.append(result)
    
    return pd.DataFrame(results)

def analyze_fractal_formation(df_prices, df_signals):
    """分析顶底分型形成是否合理"""
    fractal_results = []
    
    for _, signal in df_signals.iterrows():
        signal_date = signal['date']
        signal_idx = df_prices[df_prices['date'] == signal_date].index[0]
        
        # 检查是否有足够的数据形成顶底分型
        if signal_idx >= 2 and signal_idx <= len(df_prices) - 3:
            # 底分型: 中间低点，两边都比它高
            # 顶分型: 中间高点，两边都比它低
            if signal['type'] == '买入':  # 底分型
                # 检查底分型形成条件
                prices_around = df_prices.iloc[signal_idx-2:signal_idx+3]['low'].values
                if len(prices_around) == 5:
                    is_bottom_fractal = (prices_around[2] < prices_around[0] and 
                                         prices_around[2] < prices_around[1] and 
                                         prices_around[2] < prices_around[3] and 
                                         prices_around[2] < prices_around[4])
                    
                    # 检查MACD背离（简化版）
                    recent_lows = df_prices.iloc[max(0, signal_idx-10):signal_idx+1]['low'].values
                    recent_lows_before = df_prices.iloc[max(0, signal_idx-20):max(0, signal_idx-10)]['low'].values
                    
                    if len(recent_lows) > 5 and len(recent_lows_before) > 5:
                        has_divergence = recent_lows.min() < recent_lows_before.min()
                    else:
                        has_divergence = False
                    
                    fractal_results.append({
                        'date': signal_date.strftime('%Y-%m-%d'),
                        'type': '底分型',
                        'formed_correctly': is_bottom_fractal,
                        'has_divergence': has_divergence
                    })
            else:  # 顶分型
                # 检查顶分型形成条件
                prices_around = df_prices.iloc[signal_idx-2:signal_idx+3]['high'].values
                if len(prices_around) == 5:
                    is_top_fractal = (prices_around[2] > prices_around[0] and 
                                      prices_around[2] > prices_around[1] and 
                                      prices_around[2] > prices_around[3] and 
                                      prices_around[2] > prices_around[4])
                    
                    # 检查MACD背离（简化版）
                    recent_highs = df_prices.iloc[max(0, signal_idx-10):signal_idx+1]['high'].values
                    recent_highs_before = df_prices.iloc[max(0, signal_idx-20):max(0, signal_idx-10)]['high'].values
                    
                    if len(recent_highs) > 5 and len(recent_highs_before) > 5:
                        has_divergence = recent_highs.max() > recent_highs_before.max()
                    else:
                        has_divergence = False
                    
                    fractal_results.append({
                        'date': signal_date.strftime('%Y-%m-%d'),
                        'type': '顶分型',
                        'formed_correctly': is_top_fractal,
                        'has_divergence': has_divergence
                    })
    
    return pd.DataFrame(fractal_results)

def analyze_level_classification(df_signals, df_prices):
    """分析缠论级别分类是否合理"""
    level_results = []
    
    for _, signal in df_signals.iterrows():
        signal_date = signal['date']
        signal_idx = df_prices[df_prices['date'] == signal_date].index[0]
        signal_price = signal['price']
        signal_level = signal['level']
        
        # 获取更长期的数据来分析趋势和中枢
        lookback_period = 60  # 约3个月
        start_idx = max(0, signal_idx - lookback_period)
        historical_data = df_prices.iloc[start_idx:signal_idx+1]
        
        # 分析趋势方向
        if len(historical_data) > 20:
            # 简单移动平均线判断趋势
            historical_data['MA20'] = historical_data['close'].rolling(window=20).mean()
            historical_data['MA60'] = historical_data['close'].rolling(window=60).mean()
            
            latest_ma20 = historical_data.iloc[-1]['MA20']
            latest_ma60 = historical_data.iloc[-1]['MA60']
            
            # 判断趋势
            if latest_ma20 > latest_ma60:
                trend = '上升'
            elif latest_ma20 < latest_ma60:
                trend = '下降'
            else:
                trend = '震荡'
            
            # 分析中枢位置（简化）
            recent_prices = historical_data['close'].values[-30:]
            if len(recent_prices) > 10:
                # 简化的中枢判断：价格密集区域
                price_range = recent_prices.max() - recent_prices.min()
                mid_price = (recent_prices.max() + recent_prices.min()) / 2
                
                # 判断信号价格相对于中枢的位置
                if price_range > 0.05:  # 有一定波动
                    if signal['type'] == '买入':
                        if trend == '下降' and signal_level == '一买':
                            level_reasonable = True
                        elif trend == '上升' and signal_level == '二买':
                            level_reasonable = True
                        elif trend == '上升' and signal_level == '三买' and signal_price > mid_price * 1.01:
                            level_reasonable = True
                        else:
                            level_reasonable = False
                    else:  # 卖出
                        if trend == '上升' and signal_level == '一卖':
                            level_reasonable = True
                        elif trend == '下降' and signal_level == '二卖':
                            level_reasonable = True
                        elif trend == '下降' and signal_level == '三卖' and signal_price < mid_price * 0.99:
                            level_reasonable = True
                        else:
                            level_reasonable = False
                else:
                    level_reasonable = None  # 无法判断
            else:
                level_reasonable = None
        else:
            trend = '数据不足'
            level_reasonable = None
        
        level_results.append({
            'date': signal_date.strftime('%Y-%m-%d'),
            'signal_level': signal_level,
            'trend': trend,
            'level_reasonable': level_reasonable
        })
    
    return pd.DataFrame(level_results)

def generate_summary(signal_analysis, fractal_analysis, level_analysis):
    """生成分析总结"""
    summary = {
        'signal_accuracy': {
            'avg_accuracy': signal_analysis['price_accuracy'].mean(),
            'success_rate': signal_analysis['signal_success'].mean() * 100
        },
        'fractal_formation': {
            'correct_formation_rate': fractal_analysis['formed_correctly'].mean() * 100 if not fractal_analysis.empty else 0
        },
        'level_classification': {
            'reasonable_rate': level_analysis[level_analysis['level_reasonable'].notna()]['level_reasonable'].mean() * 100 if not level_analysis[level_analysis['level_reasonable'].notna()].empty else 0
        }
    }
    
    return summary

def main():
    try:
        logger.info("===== 开始分析2025年10月军工ETF交易信号 =====")
        
        # 1. 加载价格数据
        price_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_daily_20251125_084859.csv'
        df_prices = load_price_data(price_file)
        
        # 2. 提取报告中的信号
        df_signals = extract_signals_from_report()
        
        # 3. 分析信号准确性
        logger.info("分析信号价格准确性和后续表现...")
        signal_analysis = analyze_signal_accuracy(df_prices, df_signals)
        logger.info(f"信号分析完成，价格平均准确率: {signal_analysis['price_accuracy'].mean():.2f}%")
        logger.info(f"信号成功率: {signal_analysis['signal_success'].mean() * 100:.2f}%")
        
        # 4. 分析顶底分型形成
        logger.info("分析顶底分型形成合理性...")
        fractal_analysis = analyze_fractal_formation(df_prices, df_signals)
        if not fractal_analysis.empty:
            correct_rate = fractal_analysis['formed_correctly'].mean() * 100
            logger.info(f"顶底分型正确形成率: {correct_rate:.2f}%")
        
        # 5. 分析缠论级别判定
        logger.info("分析缠论级别分类合理性...")
        level_analysis = analyze_level_classification(df_signals, df_prices)
        if not level_analysis[level_analysis['level_reasonable'].notna()].empty:
            reasonable_rate = level_analysis[level_analysis['level_reasonable'].notna()]['level_reasonable'].mean() * 100
            logger.info(f"级别判定合理率: {reasonable_rate:.2f}%")
        
        # 6. 生成详细报告
        logger.info("\n详细信号分析结果:")
        for _, row in signal_analysis.iterrows():
            logger.info(f"{row['date']} {row['type']} 信号 (级别: {row['level']}, 强度: {row['strength']:.3f}): ")
            logger.info(f"  - 信号价格: {row['signal_price']}, 实际收盘价: {row['actual_price']}, 价格准确率: {row['price_accuracy']:.2f}%")
            logger.info(f"  - 后续表现: {row['future_return']:+.2f}%, 信号{'成功' if row['signal_success'] else '失败'}")
        
        logger.info("\n顶底分型分析:")
        for _, row in fractal_analysis.iterrows():
            logger.info(f"{row['date']} {row['type']}: {'正确形成' if row['formed_correctly'] else '形成条件不满足'}, {'有背离' if row['has_divergence'] else '无明显背离'}")
        
        logger.info("\n缠论级别分析:")
        for _, row in level_analysis.iterrows():
            level_status = "合理" if row['level_reasonable'] else ("不合理" if row['level_reasonable'] is not None else "无法判断")
            logger.info(f"{row['date']} {row['signal_level']}: 趋势: {row['trend']}, 级别判定: {level_status}")
        
        # 7. 生成总体评价
        summary = generate_summary(signal_analysis, fractal_analysis, level_analysis)
        logger.info("\n===== 分析总结 =====")
        logger.info(f"信号价格平均准确率: {summary['signal_accuracy']['avg_accuracy']:.2f}%")
        logger.info(f"信号后续表现成功率: {summary['signal_accuracy']['success_rate']:.2f}%")
        logger.info(f"顶底分型正确形成率: {summary['fractal_formation']['correct_formation_rate']:.2f}%")
        logger.info(f"缠论级别判定合理率: {summary['level_classification']['reasonable_rate']:.2f}%")
        
        # 整体评价
        overall_score = (summary['signal_accuracy']['avg_accuracy'] + 
                        summary['signal_accuracy']['success_rate'] + 
                        summary['fractal_formation']['correct_formation_rate'] + 
                        summary['level_classification']['reasonable_rate']) / 4
        
        if overall_score >= 80:
            overall_evaluation = "优秀"
        elif overall_score >= 60:
            overall_evaluation = "良好"
        elif overall_score >= 40:
            overall_evaluation = "一般"
        else:
            overall_evaluation = "需要改进"
        
        logger.info(f"\n综合评分: {overall_score:.2f} - 评价: {overall_evaluation}")
        logger.info("===== 分析完成 =====")
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()