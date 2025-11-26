#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细验证顶底分型形成和缠论级别判定
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FractalValidator')

def load_price_data(file_path):
    """加载价格数据"""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        # 设置日期索引以便更好地切片
        df.set_index('date', inplace=True)
        logger.info(f"成功加载价格数据，共{len(df)}条记录")
        return df
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        raise

def get_october_data(df_prices):
    """提取2025年10月的数据"""
    october_data = df_prices[(df_prices.index.year == 2025) & (df_prices.index.month == 10)]
    logger.info(f"提取10月数据，共{len(october_data)}条记录")
    return october_data

def validate_fractal_formation(df_prices, signal_date, signal_type, look_back=5):
    """
    验证顶底分型是否正确形成
    
    参数:
    - df_prices: 价格数据
    - signal_date: 信号日期
    - signal_type: 信号类型 ('买入' 或 '卖出')
    - look_back: 向前查看的天数
    
    返回:
    - 验证结果字典
    """
    try:
        # 转换日期格式
        if isinstance(signal_date, str):
            signal_date = datetime.strptime(signal_date, '%Y-%m-%d')
        
        # 确保日期在数据中
        if signal_date not in df_prices.index:
            return {'status': 'error', 'message': f'日期 {signal_date.strftime("%Y-%m-%d")} 不在数据中'}
        
        # 获取信号日期的索引位置
        signal_idx = df_prices.index.get_loc(signal_date)
        
        # 确保有足够的数据来验证分型
        if signal_idx < look_back or signal_idx >= len(df_prices) - look_back:
            return {'status': 'error', 'message': '数据不足，无法验证分型'}
        
        # 提取信号日期前后的数据
        window_data = df_prices.iloc[signal_idx - look_back:signal_idx + look_back + 1]
        
        if signal_type == '买入':
            # 验证底分型：中间低点，两边都比它高
            # 标准底分型：第3天是低点，第1、2天和第4、5天的低点都比它高
            if len(window_data) >= 5:
                # 取中间位置
                mid_idx = len(window_data) // 2
                mid_low = window_data.iloc[mid_idx]['low']
                
                # 检查前后的低点
                left_lows = window_data.iloc[:mid_idx]['low'].values
                right_lows = window_data.iloc[mid_idx+1:]['low'].values
                
                is_bottom_fractal = (all(left_lows > mid_low) and all(right_lows > mid_low))
                
                # 检查简化版MACD背离（价格新低，MACD不新低）
                # 这里使用前后价格比较作为简化版
                recent_prices = df_prices.iloc[max(0, signal_idx-20):signal_idx+1]['low'].values
                
                if len(recent_prices) > 10:
                    # 检查是否有底背离条件
                    has_divergence = recent_prices[-1] < np.median(recent_prices[:-5])
                else:
                    has_divergence = False
                
                # 检查是否有明显的下跌趋势
                recent_changes = np.diff(window_data['close'].values)
                downtrend = np.mean(recent_changes[:mid_idx]) < 0
                
                return {
                    'status': 'success',
                    'is_fractal': is_bottom_fractal,
                    'has_divergence': has_divergence,
                    'in_downtrend': downtrend,
                    'mid_low': mid_low,
                    'left_lows': list(left_lows),
                    'right_lows': list(right_lows),
                    'message': f"底分型验证: {'正确' if is_bottom_fractal else '不正确'}, 趋势: {'下跌' if downtrend else '非下跌'}"
                }
            else:
                return {'status': 'error', 'message': '窗口数据不足'}
        else:  # 卖出信号，验证顶分型
            if len(window_data) >= 5:
                # 取中间位置
                mid_idx = len(window_data) // 2
                mid_high = window_data.iloc[mid_idx]['high']
                
                # 检查前后的高点
                left_highs = window_data.iloc[:mid_idx]['high'].values
                right_highs = window_data.iloc[mid_idx+1:]['high'].values
                
                is_top_fractal = (all(left_highs < mid_high) and all(right_highs < mid_high))
                
                # 检查简化版MACD背离
                recent_prices = df_prices.iloc[max(0, signal_idx-20):signal_idx+1]['high'].values
                
                if len(recent_prices) > 10:
                    # 检查是否有顶背离条件
                    has_divergence = recent_prices[-1] > np.median(recent_prices[:-5])
                else:
                    has_divergence = False
                
                # 检查是否有明显的上升趋势
                recent_changes = np.diff(window_data['close'].values)
                uptrend = np.mean(recent_changes[:mid_idx]) > 0
                
                return {
                    'status': 'success',
                    'is_fractal': is_top_fractal,
                    'has_divergence': has_divergence,
                    'in_uptrend': uptrend,
                    'mid_high': mid_high,
                    'left_highs': list(left_highs),
                    'right_highs': list(right_highs),
                    'message': f"顶分型验证: {'正确' if is_top_fractal else '不正确'}, 趋势: {'上升' if uptrend else '非上升'}"
                }
            else:
                return {'status': 'error', 'message': '窗口数据不足'}
    except Exception as e:
        return {'status': 'error', 'message': f'验证过程中出错: {str(e)}'}

def validate_chanlun_level(df_prices, signal_date, signal_type, signal_level, look_back=60):
    """
    验证缠论级别判定是否合理
    
    参数:
    - df_prices: 价格数据
    - signal_date: 信号日期
    - signal_type: 信号类型 ('买入' 或 '卖出')
    - signal_level: 缠论级别 ('一买', '二买', '三买', '一卖', '二卖', '三卖')
    - look_back: 向前查看的天数
    
    返回:
    - 验证结果字典
    """
    try:
        # 转换日期格式
        if isinstance(signal_date, str):
            signal_date = datetime.strptime(signal_date, '%Y-%m-%d')
        
        # 确保日期在数据中
        if signal_date not in df_prices.index:
            return {'status': 'error', 'message': f'日期 {signal_date.strftime("%Y-%m-%d")} 不在数据中'}
        
        # 获取信号日期的索引位置
        signal_idx = df_prices.index.get_loc(signal_date)
        
        # 获取历史数据用于分析
        start_idx = max(0, signal_idx - look_back)
        historical_data = df_prices.iloc[start_idx:signal_idx+1].copy()
        
        # 计算移动平均线以判断趋势
        historical_data['MA20'] = historical_data['close'].rolling(window=20).mean()
        historical_data['MA60'] = historical_data['close'].rolling(window=60).mean()
        historical_data['MA120'] = historical_data['close'].rolling(window=120).mean()
        
        # 计算趋势指标
        if len(historical_data) >= 20:
            # 判断当前趋势
            latest_close = historical_data.iloc[-1]['close']
            latest_ma20 = historical_data.iloc[-1]['MA20']
            latest_ma60 = historical_data.iloc[-1]['MA60'] if len(historical_data) >= 60 else None
            
            if latest_ma60 is not None:
                if latest_ma20 > latest_ma60:
                    trend = '上升'
                elif latest_ma20 < latest_ma60:
                    trend = '下降'
                else:
                    trend = '震荡'
            else:
                # 使用20日均线判断短期趋势
                if latest_close > latest_ma20:
                    trend = '上升'
                elif latest_close < latest_ma20:
                    trend = '下降'
                else:
                    trend = '震荡'
            
            # 计算波动率（用于判断中枢）
            recent_30 = historical_data.iloc[-30:]
            price_range = recent_30['high'].max() - recent_30['low'].min()
            avg_price = recent_30['close'].mean()
            volatility = price_range / avg_price * 100
            
            # 计算中枢区间（简化为价格密集区域）
            if volatility > 5:  # 有足够的波动形成中枢
                central_high = recent_30['high'].quantile(0.75)
                central_low = recent_30['low'].quantile(0.25)
                is_in_central = central_low <= latest_close <= central_high
            else:
                central_high = central_low = latest_close
                is_in_central = True
            
            # 判断级别合理性
            level_reasonable = False
            reason = ""
            
            if signal_type == '买入':
                if signal_level == '一买':
                    # 一买条件：下降趋势中，出现底分型，可能有背离
                    if trend == '下降':
                        # 检查是否在近期低点附近
                        recent_lows = historical_data['low'].tail(20).values
                        is_near_low = latest_close <= recent_lows.mean() + (recent_lows.max() - recent_lows.min()) * 0.2
                        if is_near_low:
                            level_reasonable = True
                            reason = "下降趋势中，价格接近近期低点"
                
                elif signal_level == '二买':
                    # 二买条件：形成第一个中枢后，回调不创新低
                    if trend in ['上升', '震荡']:
                        # 检查是否有一个上涨后的回调
                        recent_20 = historical_data['close'].tail(20)
                        max_in_20 = recent_20.max()
                        pullback_ratio = (max_in_20 - latest_close) / max_in_20 * 100
                        if 3 <= pullback_ratio <= 15:  # 合理回调幅度
                            level_reasonable = True
                            reason = f"上涨趋势中的合理回调，回调幅度{pullback_ratio:.1f}%"
                
                elif signal_level == '三买':
                    # 三买条件：离开中枢后，回调不进中枢
                    if trend == '上升' and not is_in_central:
                        # 检查是否在中枢上方
                        if latest_close > central_high:
                            level_reasonable = True
                            reason = "上升趋势中，价格在中枢上方"
            
            else:  # 卖出信号
                if signal_level == '一卖':
                    # 一卖条件：上升趋势中，出现顶分型，可能有背离
                    if trend == '上升':
                        # 检查是否在近期高点附近
                        recent_highs = historical_data['high'].tail(20).values
                        is_near_high = latest_close >= recent_highs.mean() - (recent_highs.max() - recent_highs.min()) * 0.2
                        if is_near_high:
                            level_reasonable = True
                            reason = "上升趋势中，价格接近近期高点"
                
                elif signal_level == '二卖':
                    # 二卖条件：形成第一个中枢后，反弹不创新高
                    if trend in ['下降', '震荡']:
                        # 检查是否有一个下跌后的反弹
                        recent_20 = historical_data['close'].tail(20)
                        min_in_20 = recent_20.min()
                        rally_ratio = (latest_close - min_in_20) / min_in_20 * 100
                        if 3 <= rally_ratio <= 15:  # 合理反弹幅度
                            level_reasonable = True
                            reason = f"下降趋势中的合理反弹，反弹幅度{rally_ratio:.1f}%"
                
                elif signal_level == '三卖':
                    # 三卖条件：离开中枢后，反弹不进中枢
                    if trend == '下降' and not is_in_central:
                        # 检查是否在中枢下方
                        if latest_close < central_low:
                            level_reasonable = True
                            reason = "下降趋势中，价格在中枢下方"
            
            return {
                'status': 'success',
                'trend': trend,
                'volatility': volatility,
                'is_in_central': is_in_central,
                'level_reasonable': level_reasonable,
                'reason': reason,
                'message': f"级别验证: {signal_level}在{trend}趋势中{'' if level_reasonable else '不'}合理 - {reason}"
            }
        else:
            return {'status': 'error', 'message': '历史数据不足，无法分析趋势'}
    
    except Exception as e:
        return {'status': 'error', 'message': f'验证过程中出错: {str(e)}'}

def analyze_all_signals():
    """分析所有10月份的交易信号"""
    # 加载价格数据
    price_file = '/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_daily_20251125_084859.csv'
    df_prices = load_price_data(price_file)
    
    # 定义10月份的信号
    signals = [
        {'date': '2025-10-14', 'type': '买入', 'price': 1.192, 'strength': 0.933, 'level': '二买'},
        {'date': '2025-10-17', 'type': '卖出', 'price': 1.196, 'strength': 0.133, 'level': '三卖'},
        {'date': '2025-10-21', 'type': '买入', 'price': 1.200, 'strength': 0.533, 'level': '二买'},
        {'date': '2025-10-28', 'type': '买入', 'price': 1.249, 'strength': 0.300, 'level': '一买'},
        {'date': '2025-10-29', 'type': '卖出', 'price': 1.233, 'strength': 0.133, 'level': '三卖'},
        {'date': '2025-10-30', 'type': '买入', 'price': 1.230, 'strength': 0.133, 'level': '三买'},
        {'date': '2025-10-31', 'type': '卖出', 'price': 1.218, 'strength': 0.133, 'level': '三卖'}
    ]
    
    # 分析每个信号
    results = []
    
    logger.info("\n" + "="*60)
    logger.info("        2025年10月军工ETF交易信号详细验证")
    logger.info("="*60)
    
    for signal in signals:
        logger.info(f"\n[信号分析] {signal['date']} {signal['type']} 信号 (级别: {signal['level']}, 强度: {signal['strength']:.3f})")
        
        # 1. 验证顶底分型
        fractal_result = validate_fractal_formation(df_prices, signal['date'], signal['type'])
        logger.info(f"分型验证: {fractal_result['message']}")
        
        # 2. 验证缠论级别
        level_result = validate_chanlun_level(df_prices, signal['date'], signal['type'], signal['level'])
        logger.info(f"级别验证: {level_result['message']}")
        
        # 3. 检查信号强度是否合理
        strength_reasonable = True
        strength_issue = ""
        
        if signal['strength'] > 0.8:
            # 高强度信号应该有更强的确认条件
            if (not fractal_result.get('is_fractal', False)) or (not level_result.get('level_reasonable', False)):
                strength_reasonable = False
                strength_issue = "高强度信号但分型或级别验证不通过"
        elif signal['strength'] < 0.2:
            # 低强度信号是可以接受的
            pass
        
        # 4. 分析信号前后的价格表现
        signal_date_dt = datetime.strptime(signal['date'], '%Y-%m-%d')
        if signal_date_dt in df_prices.index:
            signal_idx = df_prices.index.get_loc(signal_date_dt)
            signal_price = df_prices.iloc[signal_idx]['close']
            
            # 检查后续3天的表现
            if signal_idx < len(df_prices) - 3:
                future_prices = df_prices.iloc[signal_idx+1:signal_idx+4]['close'].values
                
                if signal['type'] == '买入':
                    future_return = (future_prices.max() - signal_price) / signal_price * 100
                    future_success = future_return > 0
                else:
                    future_return = (future_prices.min() - signal_price) / signal_price * 100
                    future_success = future_return < 0
                
                logger.info(f"后续3天表现: {future_return:+.2f}%, {'成功' if future_success else '失败'}")
            else:
                future_return = 0
                future_success = None
                logger.info("后续数据不足，无法评估表现")
        
        # 保存结果
        results.append({
            'date': signal['date'],
            'type': signal['type'],
            'level': signal['level'],
            'strength': signal['strength'],
            'fractal_valid': fractal_result.get('is_fractal', False),
            'level_valid': level_result.get('level_reasonable', False),
            'strength_valid': strength_reasonable,
            'future_success': future_success,
            'future_return': future_return,
            'issues': [
                issue for issue in [
                    "分型形成不正确" if not fractal_result.get('is_fractal', False) else None,
                    "级别判定不合理" if not level_result.get('level_reasonable', False) else None,
                    strength_issue if not strength_reasonable else None
                ] if issue is not None
            ]
        })
    
    return results

def generate_verification_summary(results):
    """生成验证总结报告"""
    logger.info("\n" + "="*60)
    logger.info("           验证总结报告")
    logger.info("="*60)
    
    # 统计各项指标
    total_signals = len(results)
    
    # 分型验证统计
    fractal_valid = sum(1 for r in results if r['fractal_valid'])
    fractal_valid_rate = fractal_valid / total_signals * 100
    
    # 级别验证统计
    level_valid = sum(1 for r in results if r['level_valid'])
    level_valid_rate = level_valid / total_signals * 100
    
    # 强度验证统计
    strength_valid = sum(1 for r in results if r['strength_valid'])
    strength_valid_rate = strength_valid / total_signals * 100
    
    # 信号成功率统计
    success_count = sum(1 for r in results if r['future_success'] is True)
    success_rate = success_count / total_signals * 100 if total_signals > 0 else 0
    
    # 综合评分
    overall_score = (fractal_valid_rate + level_valid_rate + strength_valid_rate + success_rate) / 4
    
    logger.info(f"\n验证统计:")
    logger.info(f"1. 顶底分型正确形成率: {fractal_valid}/{total_signals} ({fractal_valid_rate:.1f}%)")
    logger.info(f"2. 缠论级别判定合理率: {level_valid}/{total_signals} ({level_valid_rate:.1f}%)")
    logger.info(f"3. 信号强度合理性: {strength_valid}/{total_signals} ({strength_valid_rate:.1f}%)")
    logger.info(f"4. 信号后续表现成功率: {success_count}/{total_signals} ({success_rate:.1f}%)")
    logger.info(f"\n综合评分: {overall_score:.1f}/100")
    
    # 分析主要问题
    all_issues = []
    for result in results:
        if result['issues']:
            all_issues.extend(result['issues'])
    
    # 统计问题频率
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    logger.info(f"\n主要问题分析:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"- {issue}: {count}次 ({count/total_signals*100:.1f}%)")
    
    # 提供改进建议
    logger.info(f"\n改进建议:")
    
    if fractal_valid_rate < 50:
        logger.info("1. 重新校准顶底分型的判定算法，确保符合标准的分型定义")
        logger.info("   - 底分型需要中间K线的低点是5根K线中的最低点")
        logger.info("   - 顶分型需要中间K线的高点是5根K线中的最高点")
    
    if level_valid_rate < 50:
        logger.info("2. 优化缠论级别的判定逻辑:")
        logger.info("   - 一买：确保在下降趋势末端，出现明显的底背离")
        logger.info("   - 二买：确保在第一个中枢形成后，回调不创新低")
        logger.info("   - 三买：确保在离开中枢后，回调不回到中枢内")
        logger.info("   - 卖出信号同理，需要符合相应的级别定义")
    
    if strength_valid_rate < 50:
        logger.info("3. 调整信号强度的计算方法:")
        logger.info("   - 高强度信号(>0.8)必须同时满足分型形成正确和级别判定合理")
        logger.info("   - 低强度信号(<0.3)应明确标记为试验性信号")
    
    # 整体评价
    if overall_score >= 80:
        evaluation = "优秀"
    elif overall_score >= 60:
        evaluation = "良好"
    elif overall_score >= 40:
        evaluation = "一般"
    else:
        evaluation = "需要改进"
    
    logger.info(f"\n整体评价: {evaluation}")
    
    return {
        'overall_score': overall_score,
        'evaluation': evaluation,
        'fractal_valid_rate': fractal_valid_rate,
        'level_valid_rate': level_valid_rate,
        'strength_valid_rate': strength_valid_rate,
        'success_rate': success_rate
    }

def main():
    try:
        logger.info("开始验证顶底分型形成和缠论级别判定...")
        
        # 分析所有信号
        results = analyze_all_signals()
        
        # 生成总结报告
        summary = generate_verification_summary(results)
        
        logger.info("\n验证完成！")
        
    except Exception as e:
        logger.error(f"验证过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()