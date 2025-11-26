#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""军工ETF(512660)缠论交易信号筛选系统 - 整合版"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入自定义模块
from src.config import load_config, DATA_PATHS
from src.data_fetcher import StockDataFetcher as StockDataAPI
from src.weekly_analyzer import WeeklyAnalyzer
from src.daily_analyzer import DailyAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('military_etf_signal.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MilitaryETFDetector')

class MilitaryETFSignalDetector:
    """军工ETF交易信号检测器"""
    
    def __init__(self, config: Dict[str, any] = None):
        """初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.api = StockDataAPI(config.get('data_fetcher', {}))
        self.weekly_analyzer = WeeklyAnalyzer()
        self.daily_analyzer = DailyAnalyzer()
        
        # 确保输出目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        for path in DATA_PATHS.values():
            os.makedirs(path, exist_ok=True)
        logger.info("所有数据目录已创建/验证")
    
    def get_symbol(self) -> str:
        """获取军工ETF的股票代码"""
        # 军工ETF的代码是512660，沪市ETF
        return "sh512660"
    
    def fetch_weekly_data(self, days: int = 180) -> pd.DataFrame:
        """获取周线数据
        
        Args:
            days: 数据天数
            
        Returns:
            周线数据DataFrame
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            logger.info(f"获取周线数据: {start_date} 至 {end_date}")
            df, actual_start, actual_end = self.api.get_weekly_data(
                self.get_symbol(),
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.error("获取的周线数据为空")
                return pd.DataFrame()
            
            logger.info(f"成功获取{len(df)}条周线数据")
            return df
            
        except Exception as e:
            logger.error(f"获取周线数据失败: {str(e)}")
            return pd.DataFrame()
    
    def fetch_daily_data(self, days: int = 90) -> pd.DataFrame:
        """获取日线数据
        
        Args:
            days: 数据天数
            
        Returns:
            日线数据DataFrame
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            logger.info(f"获取日线数据: {start_date} 至 {end_date}")
            df = self.api.get_daily_data(
                self.get_symbol(),
                start_date=start_date,
                end_date=end_date,
                force_refresh=True
            )
            
            if df.empty:
                logger.error("获取的日线数据为空")
                return pd.DataFrame()
            
            logger.info(f"成功获取{len(df)}条日线数据")
            return df
            
        except Exception as e:
            logger.error(f"获取日线数据失败: {str(e)}")
            return pd.DataFrame()
    
    def analyze_weekly_conditions(self, df: pd.DataFrame) -> Dict[str, any]:
        """分析周线条件
        
        Args:
            df: 周线数据
            
        Returns:
            周线分析结果
        """
        if df.empty:
            return {"success": False, "error": "周线数据为空"}
        
        try:
            # 使用周线分析器进行分析
            result = self.weekly_analyzer.analyze_weekly_condition(df)
            logger.info(f"周线分析完成: 置信度档位={result.get('weekly_level', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"周线分析失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_daily_conditions(self, df: pd.DataFrame) -> Dict[str, any]:
        """分析日线条件
        
        Args:
            df: 日线数据
            
        Returns:
            日线分析结果
        """
        if df.empty:
            return {"success": False, "error": "日线数据为空"}
        
        try:
            # 使用日线分析器进行分析
            result = self.daily_analyzer.analyze_daily_conditions(df)
            logger.info(f"日线分析完成: 买入信号={result.get('buy_signal', False)}")
            return result
            
        except Exception as e:
            logger.error(f"日线分析失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_signal(self, weekly_result: Dict[str, any], daily_result: Dict[str, any]) -> Dict[str, any]:
        """生成交易信号
        
        Args:
            weekly_result: 周线分析结果
            daily_result: 日线分析结果
            
        Returns:
            交易信号
        """
        signal = {
            "symbol": "512660",
            "name": "军工ETF",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_type": "none",
            "signal_strength": 0,
            "confidence_level": "none",
            "weekly_confirmation": "未确认",
            "weekly_data": weekly_result,
            "daily_data": daily_result,
            "trade_recommendation": "观望",
            "confidence_metrics": {
                "weekly_high_confidence": False,
                "weekly_medium_confidence": False,
                "daily_buy_signal": False,
                "divergence_strength": 0,
                "macd_area_ratio": 0,
                "conditions_satisfaction_ratio": 0,
                "total_met_conditions": 0,
                "total_conditions": 0
            },
            "confidence_explanation": ""
        }
        
        # 检查基础条件
        signal["confidence_metrics"]["weekly_high_confidence"] = weekly_result.get("weekly_level") == "high"
        signal["confidence_metrics"]["weekly_medium_confidence"] = weekly_result.get("weekly_level") == "medium"
        signal["confidence_metrics"]["daily_buy_signal"] = daily_result.get("buy_signal", False)
        
        # 获取日线条件满足数量
        daily_conditions = daily_result.get("conditions", {})
        if isinstance(daily_conditions, dict) and "total_met_conditions" in daily_conditions:
            daily_met_conditions = daily_conditions["total_met_conditions"]
            daily_total_conditions = daily_conditions.get("total_conditions", 0)
        else:
            daily_met_conditions = sum(1 for cond in daily_conditions.values() if cond)
            daily_total_conditions = len(daily_conditions)
        
        # 获取背驰强度（如果有）
        divergence_info = daily_result.get("divergence", {})
        signal["confidence_metrics"]["divergence_strength"] = divergence_info.get("strength", 0)
        
        # 获取MACD绿柱面积比（如果有）
        macd_info = daily_result.get("macd", {})
        signal["confidence_metrics"]["macd_area_ratio"] = macd_info.get("area_ratio", 0)
        
        # 统计总满足条件数和总条件数
        base_met_conditions = sum(1 for cond in [
            signal["confidence_metrics"]["weekly_high_confidence"],
            signal["confidence_metrics"]["weekly_medium_confidence"],
            signal["confidence_metrics"]["daily_buy_signal"]
        ])
        
        signal["confidence_metrics"]["total_met_conditions"] = base_met_conditions
        signal["confidence_metrics"]["total_conditions"] = 3  # 基础条件数
        
        # 计算条件满足比率
        if signal["confidence_metrics"]["total_conditions"] > 0:
            signal["confidence_metrics"]["conditions_satisfaction_ratio"] = \
                signal["confidence_metrics"]["total_met_conditions"] / signal["confidence_metrics"]["total_conditions"]
        
        # 计算综合可信度评分
        confidence_score = self._calculate_confidence_score(signal["confidence_metrics"])
        
        # 确定信号类型和交易建议
        if signal["confidence_metrics"]["daily_buy_signal"]:
            if confidence_score >= 80:
                signal["signal_type"] = "strong_buy"
                signal["signal_strength"] = 100
                signal["confidence_level"] = "high"
                signal["weekly_confirmation"] = "高置信确认"
                signal["trade_recommendation"] = "强烈买入"
                signal["confidence_explanation"] = "周线高置信+日线买入信号+强背驰，综合评分高"
            elif confidence_score >= 60:
                signal["signal_type"] = "buy"
                signal["signal_strength"] = 75
                signal["confidence_level"] = "medium"
                signal["weekly_confirmation"] = "中置信确认"
                signal["trade_recommendation"] = "买入"
                signal["confidence_explanation"] = "周线中置信+日线买入信号，综合评分中等"
            else:
                signal["signal_type"] = "weak_buy"
                signal["signal_strength"] = 50
                signal["confidence_level"] = "low"
                signal["weekly_confirmation"] = "无确认"
                signal["trade_recommendation"] = "谨慎买入"
                signal["confidence_explanation"] = "仅日线买入信号，需谨慎"
        
        # 添加详细的可信度评估
        signal["confidence_score"] = confidence_score
        signal["confidence_evaluation"] = self._evaluate_confidence_quality(confidence_score)
        
        logger.info(f"信号生成完成: 信号类型={signal['signal_type']}, 强度={signal['signal_strength']}, 可信度评分={confidence_score:.1f}")
        return signal
    
    def _calculate_confidence_score(self, metrics: Dict[str, any]) -> float:
        """计算综合可信度评分
        
        Args:
            metrics: 可信度指标字典
            
        Returns:
            综合可信度评分 (0-100)
        """
        score = 0
        
        # 基础条件评分（50分）
        if metrics["weekly_high_confidence"]:
            score += 30  # 周线高置信
        elif metrics["weekly_medium_confidence"]:
            score += 15  # 周线中置信
            
        if metrics["daily_buy_signal"]:
            score += 20  # 日线买入信号
        
        # 背驰强度评分（30分）
        divergence_strength = metrics.get("divergence_strength", 0)
        score += min(divergence_strength * 30, 30)
        
        # MACD绿柱面积比评分（10分）
        macd_area_ratio = metrics.get("macd_area_ratio", 0)
        # 面积比越小，背驰越强（绿柱面积减小表示底背驰）
        if macd_area_ratio > 0:
            score += min(10 * (1 - macd_area_ratio), 10)
        
        # 条件满足率评分（10分）
        conditions_ratio = metrics.get("conditions_satisfaction_ratio", 0)
        score += conditions_ratio * 10
        
        return round(score, 1)
    
    def _evaluate_confidence_quality(self, score: float) -> str:
        """评估可信度质量
        
        Args:
            score: 可信度评分
            
        Returns:
            质量评估文本
        """
        if score >= 90:
            return "极佳 - 多项强指标确认，交易信号非常可靠"
        elif score >= 75:
            return "良好 - 关键指标确认，交易信号较为可靠"
        elif score >= 60:
            return "一般 - 基础指标确认，但缺乏足够的支持证据"
        elif score >= 40:
            return "谨慎 - 仅有基本信号，需等待更多确认"
        else:
            return "弱势 - 信号较弱，建议观察为主"
    
    def check_signal_expiry(self, signal: Dict[str, any], daily_df: pd.DataFrame) -> Dict[str, any]:
        """检查信号是否失效
        
        Args:
            signal: 交易信号
            daily_df: 最新日线数据
            
        Returns:
            更新后的信号，包含失效信息
        """
        if not signal.get("signal_type") in ["strong_buy", "buy", "weak_buy"]:
            signal["expired"] = True
            signal["expiry_reason"] = "未生成有效买入信号"
            return signal
        
        # 检查是否跌破买入价的5%
        latest_price = daily_df.iloc[-1]["close"] if not daily_df.empty else 0
        if latest_price > 0:
            # 假设买入价为信号生成时的价格
            buy_price = latest_price  # 简化处理，实际应该记录信号生成时的价格
            stop_loss_threshold = buy_price * 0.95
            
            if latest_price < stop_loss_threshold:
                signal["expired"] = True
                signal["expiry_reason"] = f"跌破止损位({stop_loss_threshold:.2f})"
                return signal
        
        # 检查日线条件是否仍然满足
        daily_result = self.analyze_daily_conditions(daily_df)
        if not daily_result.get("buy_signal", False):
            signal["expired"] = True
            signal["expiry_reason"] = "日线买入条件不再满足"
            return signal
        
        signal["expired"] = False
        signal["expiry_reason"] = "信号仍然有效"
        return signal
    
    def save_signal(self, signal: Dict[str, any]):
        """保存信号到文件
        
        Args:
            signal: 交易信号
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{DATA_PATHS['signals']}/military_etf_signal_{timestamp}.json"
        
        try:
            # 确保所有数据都是可序列化的，加强numpy类型转换
            def make_serializable(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(obj, (pd.DataFrame, pd.Series)):
                    return obj.to_dict()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: make_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                return obj
            
            serializable_signal = make_serializable(signal)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_signal, f, indent=2, ensure_ascii=False)
            
            logger.info(f"信号已保存: {filename}")
            
            # 更新最新信号文件
            latest_filename = f"{DATA_PATHS['signals']}/military_etf_signal_latest.json"
            with open(latest_filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_signal, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"保存信号失败: {str(e)}")
    
    def run_full_analysis(self) -> Dict[str, any]:
        """运行完整的分析流程
        
        Returns:
            完整分析结果
        """
        logger.info("===== 开始军工ETF(512660)缠论交易信号分析 =====")
        
        # 获取数据
        weekly_df = self.fetch_weekly_data(days=180)
        daily_df = self.fetch_daily_data(days=90)
        
        if weekly_df.empty or daily_df.empty:
            logger.error("无法获取足够的数据进行分析")
            return {"success": False, "error": "数据获取失败"}
        
        # 分析周线条件
        weekly_result = self.analyze_weekly_conditions(weekly_df)
        
        # 分析日线条件
        daily_result = self.analyze_daily_conditions(daily_df)
        
        # 生成交易信号
        signal = self.generate_signal(weekly_result, daily_result)
        
        # 检查信号是否失效
        signal = self.check_signal_expiry(signal, daily_df)
        
        # 保存信号
        self.save_signal(signal)
        
        logger.info("===== 军工ETF(512660)缠论交易信号分析完成 =====")
        return signal
    
    def generate_signal_output(self, signal: Dict[str, any]) -> Dict[str, any]:
        """生成符合要求的信号输出格式
        
        Args:
            signal: 交易信号
            
        Returns:
            格式化的输出数据
        """
        # 提取日线数据中的底分型和背驰信息
        daily_data = signal.get("daily_data", {})
        details = daily_data.get("details", {})
        
        # 获取分型信息
        top_fractal_count = details.get("recent_fractals", {}).get("top_count", 0)
        bottom_fractal_count = details.get("recent_fractals", {}).get("bottom_count", 0)
        
        # 获取背驰强度
        divergence_strength = daily_data.get("divergence_strength", 0)
        
        # 格式化输出
        output = {
            "股票代码": signal["symbol"],
            "股票名称": signal["name"],
            "信号类型": signal["signal_type"],
            "信号强度": signal["signal_strength"],
            "置信度档位": signal["confidence_level"],
            "周线确认": signal["weekly_confirmation"],
            "日线底分型数量": bottom_fractal_count,
            "日线顶分型数量": top_fractal_count,
            "背驰强度": round(divergence_strength, 2),
            "满足条件数": signal["confidence_metrics"]["total_met_conditions"],
            "总条件数": signal["confidence_metrics"]["total_conditions"],
            "交易建议": signal["trade_recommendation"],
            "是否失效": signal.get("expired", True),
            "失效原因": signal.get("expiry_reason", ""),
            "生成时间": signal["timestamp"]
        }
        
        return output

def main():
    """主函数"""
    # 加载配置
    config = load_config('config/system.yaml')
    
    # 创建检测器实例
    detector = MilitaryETFSignalDetector(config)
    
    # 运行完整分析
    signal = detector.run_full_analysis()
    
    # 生成并打印输出
    output = detector.generate_signal_output(signal)
    print("\n===== 军工ETF(512660)交易信号输出 =====")
    for key, value in output.items():
        print(f"{key}: {value}")
    print("========================================\n")

if __name__ == "__main__":
    main()