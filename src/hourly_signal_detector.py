#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小时级别信号检测器
用于在日线底分型形成前进行预估和提醒，避免收盘后错过买入时机
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_fetcher import StockDataFetcher
from src.calculator import ChanlunCalculator
from src.notifier import DingdingNotifier
from src.utils import is_trading_hour
from src.config import load_config

logger = logging.getLogger(__name__)

class HourlySignalDetector:
    """小时级别信号检测器，用于提前预警可能的日线底分型"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化小时级别信号检测器"""
        self.config = config or (load_system_config() if hasattr(sys.modules['src.config'], 'load_system_config') else load_config())
        self.data_fetcher = StockDataFetcher()
        self.calculator = ChanlunCalculator(self.config)
        self.notifier = DingdingNotifier(self.config)
        
        # 配置参数
        self.hourly_fractal_sensitivity = self.config.get('hourly_signal', {}).get('fractal_sensitivity', 3)
        self.prediction_threshold = self.config.get('hourly_signal', {}).get('prediction_threshold', 0.6)
        
    def get_hourly_data(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """获取小时级别数据（基于30分钟数据聚合）"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 获取30分钟数据并聚合为小时级别
            df_30min = self.data_fetcher.get_minute_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                                        end_date.strftime('%Y-%m-%d'), interval=30)
            
            if df_30min.empty:
                logger.warning(f"无法获取 {symbol} 的30分钟数据")
                return pd.DataFrame()
            
            # 聚合为小时级别
            df_30min['hour'] = df_30min['date'].dt.floor('H')
            df_hourly = df_30min.groupby('hour').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            df_hourly.rename(columns={'hour': 'date'}, inplace=True)
            df_hourly['symbol'] = symbol
            
            logger.info(f"成功获取并聚合 {symbol} 的小时级别数据，共 {len(df_hourly)} 条")
            return df_hourly
            
        except Exception as e:
            logger.error(f"获取小时级别数据异常: {str(e)}")
            return pd.DataFrame()
    
    def detect_hourly_bottom_fractal(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测小时级别的底分型"""
        if df.empty:
            return df
        
        df = df.copy()
        df['hourly_bottom_fractal'] = False
        sensitivity = self.hourly_fractal_sensitivity
        
        for i in range(sensitivity, len(df) - sensitivity):
            # 当前K线的低点
            current_low = df.iloc[i]['low']
            
            # 检查左边sensitivity根K线的低点都高于当前低点
            left_lower = all(df.iloc[i-sensitivity:i]['low'] > current_low)
            
            # 检查右边sensitivity根K线的低点都高于当前低点
            right_lower = all(df.iloc[i+1:i+sensitivity+1]['low'] > current_low)
            
            if left_lower and right_lower:
                df.loc[df.index[i], 'hourly_bottom_fractal'] = True
        
        return df
    
    def predict_daily_bottom_fractal(self, symbol: str) -> Dict[str, Any]:
        """预测是否可能形成日线底分型
        
        返回:
            Dict: 包含预测结果的字典
        """
        try:
            # 获取当日的小时级别数据
            today = datetime.now().strftime('%Y-%m-%d')
            df_30min = self.data_fetcher.get_minute_data(symbol, today, today, interval=30)
            
            if df_30min.empty:
                return {"prediction": False, "confidence": 0.0, "reason": "无法获取今日小时数据"}
            
            # 获取最近的日线数据
            df_daily = self.data_fetcher.get_daily_data(symbol, days=10)
            if len(df_daily) < 7:  # 需要至少7天数据来检测底分型
                return {"prediction": False, "confidence": 0.0, "reason": "日线数据不足"}
            
            # 计算今日累计数据（模拟今日日线）
            today_open = df_30min['open'].iloc[0]
            today_high = df_30min['high'].max()
            today_low = df_30min['low'].min()
            today_close = df_30min['close'].iloc[-1]
            today_volume = df_30min['volume'].sum()
            
            # 检查当前是否满足底分型的潜在条件
            confidence = 0.0
            reasons = []
            
            # 条件1: 今日低点是否可能成为底分型的最低点
            recent_days = df_daily.tail(5)
            if today_low < recent_days['low'].iloc[-2]:  # 低于前天低点
                if len(recent_days) >= 4:
                    # 检查前天和大前天的低点关系
                    if recent_days['low'].iloc[-2] < recent_days['low'].iloc[-3]:
                        confidence += 0.3
                        reasons.append("今日低点可能成为底分型最低点")
            
            # 条件2: 小时级别是否出现底分型
            df_hourly = self.get_hourly_data(symbol, days=2)
            df_hourly = self.detect_hourly_bottom_fractal(df_hourly)
            recent_hourly_fractals = df_hourly[df_hourly['hourly_bottom_fractal']].tail(2)
            
            if len(recent_hourly_fractals) > 0:
                confidence += 0.25
                reasons.append(f"近24小时出现{len(recent_hourly_fractals)}个小时级底分型")
            
            # 条件3: 当前是否收阳或正在收阳
            if today_close > today_open:
                confidence += 0.2
                reasons.append("今日K线收阳")
            elif today_close > today_low * 1.01:  # 虽然没完全收阳，但已经从低点回升较多
                confidence += 0.1
                reasons.append("今日K线从低点明显回升")
            
            # 条件4: 成交量变化
            if len(df_daily) >= 2:
                avg_volume = df_daily['volume'].tail(5).mean()
                if today_volume > avg_volume * 1.2:
                    confidence += 0.15
                    reasons.append("成交量明显放大")
                elif today_volume > avg_volume * 0.8:
                    confidence += 0.05
                    reasons.append("成交量保持稳定")
            
            # 条件5: 避免连续绿柱风险
            if hasattr(self.calculator, '_calculate_indicators'):
                df_recent = df_daily.tail(3).copy()
                # 计算MACD指标（简化版）
                exp1 = df_recent['close'].ewm(span=12, adjust=False).mean()
                exp2 = df_recent['close'].ewm(span=26, adjust=False).mean()
                df_recent['macd'] = exp1 - exp2
                df_recent['signal'] = df_recent['macd'].ewm(span=9, adjust=False).mean()
                df_recent['histogram'] = df_recent['macd'] - df_recent['signal']
                
                # 检查MACD柱状图是否开始减小
                if len(df_recent) >= 2 and df_recent['histogram'].iloc[-1] > df_recent['histogram'].iloc[-2] and df_recent['histogram'].iloc[-1] < 0:
                    confidence += 0.1
                    reasons.append("MACD绿柱开始减小")
            
            # 综合判断
            prediction = confidence >= self.prediction_threshold
            
            return {
                "prediction": prediction,
                "confidence": min(1.0, confidence),  # 确保置信度在0-1之间
                "reason": "；".join(reasons) if reasons else "暂无明确理由",
                "current_price": today_close,
                "today_low": today_low,
                "hourly_fractals_count": len(recent_hourly_fractals)
            }
            
        except Exception as e:
            logger.error(f"预测日线底分型异常: {str(e)}")
            return {"prediction": False, "confidence": 0.0, "reason": f"预测过程异常: {str(e)}"}
    
    def check_and_notify(self, symbol: str) -> bool:
        """检查并发送小时级别预警信号
        
        Args:
            symbol: 股票代码
            
        Returns:
            bool: 是否发送了通知
        """
        try:
            # 只有在交易时间才进行检查
            if not is_trading_hour():
                logger.info(f"当前非交易时间，跳过小时级别信号检查: {symbol}")
                return False
            
            # 预测是否可能形成日线底分型
            result = self.predict_daily_bottom_fractal(symbol)
            
            if result['prediction']:
                # 获取ETF中文名称
                etf_config = self.config.get('etfs', {})
                etf_name = etf_config.get(symbol, {}).get('name', symbol)
                
                # 构造预警信号详情
                alert_details = {
                    "alert_type": "日线底分型预警",
                    "symbol": symbol,
                    "name": etf_name,
                    "price": result['current_price'],
                    "today_low": result['today_low'],
                    "confidence": result['confidence'],
                    "reason": result['reason'],
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "suggestion": "建议密切关注，准备买入"
                }
                
                # 发送预警通知
                success = self.notifier.send_hourly_alert(alert_details)
                logger.info(f"小时级别预警通知{'发送成功' if success else '发送失败'}: {symbol}, 置信度: {result['confidence']:.2f}")
                return success
            else:
                logger.debug(f"未达到预警条件: {symbol}, 置信度: {result['confidence']:.2f}")
                return False
                
        except Exception as e:
            logger.error(f"小时级别信号检查异常: {str(e)}")
            return False
    
    def batch_check(self, symbols: List[str]) -> Dict[str, bool]:
        """批量检查多个股票的小时级别信号
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            Dict: {股票代码: 是否发送了通知}
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.check_and_notify(symbol)
        return results

# 主函数，用于测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 测试单个ETF的小时级别信号检测
    detector = HourlySignalDetector()
    # 这里可以测试512660，就是用户提到的ETF
    result = detector.check_and_notify("512660")
    print(f"小时级别信号检查结果: {result}")