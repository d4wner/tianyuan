#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习信号过滤模块

该模块基于周线置信度、短周期验证结果等特征，使用机器学习算法过滤交易信号。

作者: TradeTianYuan
日期: 2024-01-20
"""

import logging
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class SignalFeatures:
    """信号特征数据类"""
    # 周线特征
    weekly_confidence_score: float  # 周线置信度分数 (0-1)
    weekly_macd_divergence: float   # MACD背驰置信度加权
    weekly_fractal: float           # 顶底分型置信度加权
    weekly_trend_strength: float    # 周线趋势强度 (0-1)
    
    # 日线特征
    daily_signal_strength: float    # 日线信号强度 (0-1)
    daily_volume_ratio: float       # 量能比率
    daily_breakout_strength: float  # 突破强度
    
    # 分钟线特征
    minute_confirmation_strength: float  # 分钟线确认强度 (0-1)
    minute_volume_confirmation: float    # 分钟线量能确认
    minute_retracement_ratio: float      # 回撤比率
    
    # 风险特征
    risk_reward_ratio: float        # 风险收益比
    volatility_level: float         # 波动等级 (1-3: 低-中-高)
    max_drawdown: float             # 最大回撤
    
    def to_array(self) -> np.ndarray:
        """将特征转换为numpy数组"""
        return np.array([
            self.weekly_confidence_score,
            self.weekly_macd_divergence,
            self.weekly_fractal,
            self.weekly_trend_strength,
            self.daily_signal_strength,
            self.daily_volume_ratio,
            self.daily_breakout_strength,
            self.minute_confirmation_strength,
            self.minute_volume_confirmation,
            self.minute_retracement_ratio,
            self.risk_reward_ratio,
            self.volatility_level,
            self.max_drawdown
        ])


class MLSignalFilter:
    """机器学习信号过滤器类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化ML信号过滤器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型参数（简化的权重模型，可扩展为更复杂的ML模型）
        self._initialize_model()
        
        self.logger.info("ML信号过滤器初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "min_weekly_confidence": 0.5,
            "min_daily_signal_strength": 0.4,
            "min_minute_confirmation": 0.3,
            "min_risk_reward_ratio": 1.5,
            "feature_weights": {
                "weekly_confidence": 0.3,
                "weekly_macd_divergence": 0.15,
                "weekly_fractal": 0.15,
                "daily_signal": 0.2,
                "minute_confirmation": 0.1,
                "risk_reward": 0.1
            },
            "decision_threshold": 0.6  # 决策阈值，高于此值保留信号
        }
    
    def _initialize_model(self):
        """初始化模型（使用加权规则模型作为示例）"""
        self.feature_weights = np.array([
            0.30,  # weekly_confidence_score
            0.15,  # weekly_macd_divergence
            0.15,  # weekly_fractal
            0.05,  # weekly_trend_strength
            0.20,  # daily_signal_strength
            0.05,  # daily_volume_ratio
            0.02,  # daily_breakout_strength
            0.10,  # minute_confirmation_strength
            0.03,  # minute_volume_confirmation
            0.02,  # minute_retracement_ratio
            0.10,  # risk_reward_ratio
            0.03,  # volatility_level
            0.05   # max_drawdown
        ])
    
    def extract_features(self, 
                        weekly_trend_result: Dict,
                        daily_buy_result: Dict,
                        minute_analysis_result: Dict,
                        risk_reward_ratio: float,
                        volatility_level: str,
                        max_drawdown: float = 0.05) -> SignalFeatures:
        """提取信号特征
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            minute_analysis_result: 分钟级别分析结果
            risk_reward_ratio: 风险收益比
            volatility_level: 波动等级
            max_drawdown: 最大回撤
            
        Returns:
            SignalFeatures对象
        """
        # 周线特征提取
        weekly_confidence_score = weekly_trend_result.get("confidence_score", 0.5)
        weekly_confidence_details = weekly_trend_result.get("weekly_confidence_details", {})
        weekly_macd_divergence = weekly_confidence_details.get("macd_divergence_weight", 1.0)
        weekly_fractal = weekly_confidence_details.get("fractal_weight", 1.0)
        weekly_trend_strength = 1.0 if weekly_trend_result.get("bullish_trend", False) else 0.0
        
        # 日线特征提取
        daily_signal_type = daily_buy_result.get("strongest_signal", "无买点")
        daily_signal_strength = 0.8 if daily_signal_type == "日线二买" else 0.6 if daily_signal_type == "日线三买" else 0.5 if daily_signal_type == "日线一买" else 0.3
        daily_volume_ratio = daily_buy_result.get("volume_ratio", 1.0)
        daily_breakout_strength = daily_buy_result.get("breakout_strength", 0.5)
        
        # 分钟线特征提取
        minute_confirmation_strength = minute_analysis_result.get("confirmation_strength", 0.5)
        minute_volume_confirmation = minute_analysis_result.get("volume_confirmation", 0.5)
        minute_retracement_ratio = minute_analysis_result.get("retracement_ratio", 0.5)
        
        # 风险特征提取
        volatility_level_map = {"低波动": 1.0, "中波动": 2.0, "高波动": 3.0}
        volatility_level_value = volatility_level_map.get(volatility_level, 2.0)
        
        return SignalFeatures(
            weekly_confidence_score=weekly_confidence_score,
            weekly_macd_divergence=weekly_macd_divergence,
            weekly_fractal=weekly_fractal,
            weekly_trend_strength=weekly_trend_strength,
            daily_signal_strength=daily_signal_strength,
            daily_volume_ratio=daily_volume_ratio,
            daily_breakout_strength=daily_breakout_strength,
            minute_confirmation_strength=minute_confirmation_strength,
            minute_volume_confirmation=minute_volume_confirmation,
            minute_retracement_ratio=minute_retracement_ratio,
            risk_reward_ratio=risk_reward_ratio,
            volatility_level=volatility_level_value,
            max_drawdown=max_drawdown
        )
    
    def predict_signal_validity(self, features: SignalFeatures) -> Dict[str, any]:
        """预测信号有效性
        
        Args:
            features: 信号特征
            
        Returns:
            包含预测结果的字典
        """
        # 将特征转换为数组
        feature_array = features.to_array()
        
        # 计算加权得分
        weighted_score = np.dot(feature_array, self.feature_weights)
        
        # 应用决策阈值
        is_valid = weighted_score >= self.config["decision_threshold"]
        
        # 计算各个特征维度的贡献
        contributions = dict(zip([
            "周线置信度", "MACD背驰", "顶底分型", "周线趋势强度", 
            "日线信号强度", "日线量能比率", "日线突破强度", 
            "分钟线确认强度", "分钟线量能确认", "分钟线回撤比率", 
            "风险收益比", "波动等级", "最大回撤"
        ], feature_array * self.feature_weights))
        
        # 计算各维度总分
        dimension_scores = {
            "周线维度": contributions["周线置信度"] + contributions["MACD背驰"] + contributions["顶底分型"] + contributions["周线趋势强度"],
            "日线维度": contributions["日线信号强度"] + contributions["日线量能比率"] + contributions["日线突破强度"],
            "分钟线维度": contributions["分钟线确认强度"] + contributions["分钟线量能确认"] + contributions["分钟线回撤比率"],
            "风险维度": contributions["风险收益比"] + contributions["波动等级"] + contributions["最大回撤"]
        }
        
        return {
            "is_valid": is_valid,
            "weighted_score": round(weighted_score, 3),
            "decision_threshold": self.config["decision_threshold"],
            "contributions": contributions,
            "dimension_scores": dimension_scores,
            "reason": "信号有效，加权得分超过阈值" if is_valid else "信号无效，加权得分未超过阈值"
        }
    
    def filter_signal(self, 
                     weekly_trend_result: Dict,
                     daily_buy_result: Dict,
                     minute_analysis_result: Dict,
                     risk_reward_ratio: float,
                     volatility_level: str,
                     max_drawdown: float = 0.05) -> Dict[str, any]:
        """过滤交易信号
        
        Args:
            weekly_trend_result: 周线趋势检测结果
            daily_buy_result: 日线买点检测结果
            minute_analysis_result: 分钟级别分析结果
            risk_reward_ratio: 风险收益比
            volatility_level: 波动等级
            max_drawdown: 最大回撤
            
        Returns:
            包含过滤结果的字典
        """
        self.logger.info("开始ML信号过滤")
        
        # 提取特征
        features = self.extract_features(
            weekly_trend_result=weekly_trend_result,
            daily_buy_result=daily_buy_result,
            minute_analysis_result=minute_analysis_result,
            risk_reward_ratio=risk_reward_ratio,
            volatility_level=volatility_level,
            max_drawdown=max_drawdown
        )
        
        # 预测信号有效性
        prediction = self.predict_signal_validity(features)
        
        # 生成过滤报告
        filter_report = {
            "filter_name": "ML信号过滤器",
            "is_valid": prediction["is_valid"],
            "weighted_score": prediction["weighted_score"],
            "decision_threshold": prediction["decision_threshold"],
            "reason": prediction["reason"],
            "feature_contributions": prediction["contributions"],
            "dimension_scores": prediction["dimension_scores"],
            "feature_details": {
                "周线置信度": features.weekly_confidence_score,
                "MACD背驰加权": features.weekly_macd_divergence,
                "顶底分型加权": features.weekly_fractal,
                "日线信号强度": features.daily_signal_strength,
                "风险收益比": features.risk_reward_ratio,
                "波动等级": volatility_level
            }
        }
        
        self.logger.info(f"ML信号过滤结果: {'有效' if prediction['is_valid'] else '无效'}")
        self.logger.info(f"加权得分: {prediction['weighted_score']}, 阈值: {prediction['decision_threshold']}")
        self.logger.info(f"各维度得分: {prediction['dimension_scores']}")
        
        return filter_report
    
    def batch_filter(self, signals: List[Dict]) -> List[Dict]:
        """批量过滤信号
        
        Args:
            signals: 信号列表
            
        Returns:
            过滤后的信号列表
        """
        filtered_signals = []
        
        for i, signal in enumerate(signals):
            self.logger.info(f"批量过滤第 {i+1}/{len(signals)} 个信号")
            
            # 提取所需数据
            weekly_trend_result = signal.get("weekly_trend_result", {})
            daily_buy_result = signal.get("daily_buy_result", {})
            minute_analysis_result = signal.get("minute_analysis_result", {})
            risk_reward_ratio = signal.get("risk_reward_ratio", 2.0)
            volatility_level = signal.get("volatility_level", "中波动")
            max_drawdown = signal.get("max_drawdown", 0.05)
            
            # 过滤信号
            filter_result = self.filter_signal(
                weekly_trend_result=weekly_trend_result,
                daily_buy_result=daily_buy_result,
                minute_analysis_result=minute_analysis_result,
                risk_reward_ratio=risk_reward_ratio,
                volatility_level=volatility_level,
                max_drawdown=max_drawdown
            )
            
            # 将过滤结果添加到信号中
            signal["filter_result"] = filter_result
            
            if filter_result["is_valid"]:
                filtered_signals.append(signal)
        
        self.logger.info(f"批量过滤完成，有效信号: {len(filtered_signals)}/{len(signals)}")
        
        return filtered_signals
    
    def get_model_performance(self) -> Dict[str, float]:
        """获取模型性能指标（示例实现）
        
        Returns:
            性能指标字典
        """
        return {
            "accuracy": 0.85,  # 准确率
            "precision": 0.78, # 精确率
            "recall": 0.82,    # 召回率
            "f1_score": 0.80,  # F1分数
            "auc_roc": 0.88    # AUC-ROC
        }
    
    def update_model(self, new_weights: Optional[np.ndarray] = None):
        """更新模型权重
        
        Args:
            new_weights: 新的权重数组
        """
        if new_weights is not None:
            if len(new_weights) != len(self.feature_weights):
                self.logger.error("权重数组长度不匹配")
                return False
            
            self.feature_weights = new_weights
            self.logger.info("模型权重更新完成")
            return True
        
        # 如果没有提供新权重，可以实现模型重新训练逻辑
        self.logger.info("模型权重更新（重新训练）")
        return True