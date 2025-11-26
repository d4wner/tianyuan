#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""信号输出格式化模块 - 支持多种输出格式和标准化字段"""

import os
import sys
import json
import csv
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入配置
from src.config import DATA_PATHS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('signal_exporter.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('SignalExporter')

class SignalExporter:
    """信号输出器类 - 处理信号的格式化和导出"""
    
    # 定义标准输出字段和默认值
    STANDARD_FIELDS = {
        "股票代码": None,
        "股票名称": None,
        "信号类型": "none",
        "信号强度": 0,
        "置信度档位": "none",
        "周线确认": "未确认",
        "日线底分型数量": 0,
        "日线顶分型数量": 0,
        "背驰强度": 0.0,
        "满足条件数": 0,
        "总条件数": 3,
        "交易建议": "观望",
        "是否失效": True,
        "失效原因": "",
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 信号类型映射
    SIGNAL_TYPE_MAPPING = {
        "strong_buy": "强烈买入",
        "buy": "买入",
        "weak_buy": "谨慎买入",
        "none": "无信号",
        "sell": "卖出",
        "strong_sell": "强烈卖出"
    }
    
    # 置信度档位映射
    CONFIDENCE_LEVEL_MAPPING = {
        "high": "高置信",
        "medium": "中置信",
        "low": "低置信",
        "none": "无"
    }
    
    def __init__(self, config: Dict[str, any] = None):
        """初始化信号输出器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        for path in DATA_PATHS.values():
            os.makedirs(path, exist_ok=True)
        logger.info("所有数据目录已创建/验证")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化的格式
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            可序列化的对象
        """
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):  # 处理numpy布尔类型
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj
    
    def format_signal(self, raw_signal: Dict[str, any]) -> Dict[str, any]:
        """格式化原始信号为标准格式
        
        Args:
            raw_signal: 原始信号字典
            
        Returns:
            标准化的信号字典
        """
        logger.info("开始格式化信号输出")
        
        # 创建标准字段的副本
        formatted_signal = self.STANDARD_FIELDS.copy()
        
        # 从原始信号中提取基本信息
        formatted_signal["股票代码"] = raw_signal.get("symbol", "512660")
        formatted_signal["股票名称"] = raw_signal.get("name", "军工ETF")
        formatted_signal["生成时间"] = raw_signal.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 获取信号类型和强度
        signal_type = raw_signal.get("signal_type", "none")
        formatted_signal["信号类型"] = self.SIGNAL_TYPE_MAPPING.get(signal_type, signal_type)
        formatted_signal["信号强度"] = raw_signal.get("signal_strength", 0)
        
        # 获取置信度信息
        confidence_level = raw_signal.get("confidence_level", "none")
        formatted_signal["置信度档位"] = self.CONFIDENCE_LEVEL_MAPPING.get(confidence_level, confidence_level)
        formatted_signal["周线确认"] = raw_signal.get("weekly_confirmation", "未确认")
        
        # 获取日线分析数据
        daily_data = raw_signal.get("daily_data", {})
        details = daily_data.get("details", {})
        
        # 提取分型信息
        recent_fractals = details.get("recent_fractals", {})
        formatted_signal["日线底分型数量"] = recent_fractals.get("bottom_count", 0)
        formatted_signal["日线顶分型数量"] = recent_fractals.get("top_count", 0)
        
        # 提取背驰强度
        formatted_signal["背驰强度"] = round(daily_data.get("divergence_strength", 0.0), 2)
        
        # 获取条件信息
        conditions = raw_signal.get("conditions", {})
        formatted_signal["满足条件数"] = conditions.get("total_met_conditions", 0)
        formatted_signal["总条件数"] = conditions.get("total_conditions", 3)
        
        # 获取交易建议
        formatted_signal["交易建议"] = raw_signal.get("trade_recommendation", "观望")
        
        # 获取信号失效信息
        formatted_signal["是否失效"] = raw_signal.get("expired", True)
        formatted_signal["失效原因"] = raw_signal.get("expiry_reason", "")
        
        logger.info("信号格式化完成")
        return formatted_signal
    
    def export_to_json(self, signal: Dict[str, any], filename: str = None, 
                      directory: str = None, pretty: bool = True) -> bool:
        """导出信号为JSON文件
        
        Args:
            signal: 信号字典
            filename: 文件名，如果为None则自动生成
            directory: 目录，如果为None则使用默认信号目录
            pretty: 是否美化输出
            
        Returns:
            是否导出成功
        """
        # 确定目录
        if directory is None:
            directory = DATA_PATHS['signals']
        
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = signal.get("股票代码", "unknown")
            filename = f"signal_{symbol}_{timestamp}.json"
        
        # 构建完整路径
        filepath = os.path.join(directory, filename)
        
        try:
            # 确保数据可序列化
            serializable_signal = self._make_serializable(signal)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(serializable_signal, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(serializable_signal, f, ensure_ascii=False)
            
            logger.info(f"信号已导出为JSON: {filepath}")
            
            # 同时更新最新信号文件
            latest_filepath = os.path.join(directory, f"signal_latest.json")
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(serializable_signal, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(serializable_signal, f, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"导出JSON失败: {str(e)}")
            return False
    
    def export_to_csv(self, signals: Union[Dict[str, any], List[Dict[str, any]]], 
                     filename: str = None, directory: str = None, 
                     mode: str = "w") -> bool:
        """导出信号为CSV文件
        
        Args:
            signals: 单个信号字典或信号列表
            filename: 文件名，如果为None则自动生成
            directory: 目录，如果为None则使用默认信号目录
            mode: 文件写入模式 ('w' 或 'a')
            
        Returns:
            是否导出成功
        """
        # 确定目录
        if directory is None:
            directory = DATA_PATHS['signals']
        
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            today = datetime.now().strftime("%Y%m%d")
            filename = f"signals_{today}.csv"
        
        # 构建完整路径
        filepath = os.path.join(directory, filename)
        
        try:
            # 确保signals是列表格式
            if isinstance(signals, dict):
                signals_list = [signals]
            else:
                signals_list = signals
            
            # 获取字段名
            if signals_list:
                fieldnames = list(self.STANDARD_FIELDS.keys())
                
                # 检查文件是否存在以决定是否写入表头
                file_exists = os.path.exists(filepath) and mode == "a"
                
                with open(filepath, mode, newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    # 写入表头（如果是新建文件或追加模式且文件不存在）
                    if mode == "w" or not file_exists:
                        writer.writeheader()
                    
                    # 写入数据
                    for signal in signals_list:
                        # 确保所有字段都存在
                        row = {field: signal.get(field, self.STANDARD_FIELDS[field]) for field in fieldnames}
                        writer.writerow(row)
                
                logger.info(f"信号已导出为CSV: {filepath}, 写入了{len(signals_list)}条记录")
            else:
                logger.warning("没有信号数据可导出")
            
            return True
            
        except Exception as e:
            logger.error(f"导出CSV失败: {str(e)}")
            return False
    
    def export_to_excel(self, signals: Union[Dict[str, any], List[Dict[str, any]]], 
                       filename: str = None, directory: str = None) -> bool:
        """导出信号为Excel文件
        
        Args:
            signals: 单个信号字典或信号列表
            filename: 文件名，如果为None则自动生成
            directory: 目录，如果为None则使用默认信号目录
            
        Returns:
            是否导出成功
        """
        # 确定目录
        if directory is None:
            directory = DATA_PATHS['signals']
        
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            today = datetime.now().strftime("%Y%m%d")
            filename = f"signals_{today}.xlsx"
        
        # 构建完整路径
        filepath = os.path.join(directory, filename)
        
        try:
            # 确保signals是列表格式
            if isinstance(signals, dict):
                signals_list = [signals]
            else:
                signals_list = signals
            
            if signals_list:
                # 创建DataFrame
                df = pd.DataFrame(signals_list)
                
                # 确保所有标准字段都存在
                for field, default in self.STANDARD_FIELDS.items():
                    if field not in df.columns:
                        df[field] = default
                
                # 按照标准字段顺序排列
                df = df[list(self.STANDARD_FIELDS.keys())]
                
                # 写入Excel文件
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Signals')
                
                logger.info(f"信号已导出为Excel: {filepath}, 写入了{len(signals_list)}条记录")
            else:
                logger.warning("没有信号数据可导出")
            
            return True
            
        except Exception as e:
            logger.error(f"导出Excel失败: {str(e)}")
            return False
    
    def print_signal_console(self, signal: Dict[str, any], format_type: str = "table") -> None:
        """在控制台打印信号
        
        Args:
            signal: 信号字典
            format_type: 输出格式 ('table' 或 'list')
        """
        if format_type == "table":
            # 表格格式输出
            print("\n" + "="*70)
            print("军工ETF(512660)缠论交易信号".center(68))
            print("="*70)
            
            # 打印基本信息
            print(f"{'股票代码:':<15} {signal.get('股票代码', '')}")
            print(f"{'股票名称:':<15} {signal.get('股票名称', '')}")
            print(f"{'生成时间:':<15} {signal.get('生成时间', '')}")
            print("-"*70)
            
            # 打印信号信息
            print(f"{'信号类型:':<15} {signal.get('信号类型', '')}")
            print(f"{'信号强度:':<15} {signal.get('信号强度', 0)}/100")
            print(f"{'置信度档位:':<15} {signal.get('置信度档位', '')}")
            print(f"{'周线确认:':<15} {signal.get('周线确认', '')}")
            print(f"{'交易建议:':<15} {signal.get('交易建议', '')}")
            print("-"*70)
            
            # 打印技术指标信息
            print(f"{'日线底分型数量:':<15} {signal.get('日线底分型数量', 0)}")
            print(f"{'日线顶分型数量:':<15} {signal.get('日线顶分型数量', 0)}")
            print(f"{'背驰强度:':<15} {signal.get('背驰强度', 0.0)}")
            print(f"{'满足条件数:':<15} {signal.get('满足条件数', 0)}/{signal.get('总条件数', 3)}")
            print("-"*70)
            
            # 打印信号状态
            print(f"{'是否失效:':<15} {'是' if signal.get('是否失效', False) else '否'}")
            if signal.get('是否失效', False):
                print(f"{'失效原因:':<15} {signal.get('失效原因', '')}")
            print("="*70 + "\n")
            
        else:  # list format
            # 列表格式输出
            print("\n军工ETF(512660)缠论交易信号:")
            for key, value in signal.items():
                print(f"{key}: {value}")
            print()
    
    def generate_summary_report(self, signals: List[Dict[str, any]], 
                               filename: str = None, directory: str = None) -> Dict[str, any]:
        """生成信号汇总报告
        
        Args:
            signals: 信号列表
            filename: 文件名，如果为None则不保存
            directory: 目录，如果为None则使用默认信号目录
            
        Returns:
            汇总报告字典
        """
        if not signals:
            logger.warning("没有信号数据用于生成汇总报告")
            return {"error": "没有信号数据"}
        
        # 创建DataFrame进行统计
        df = pd.DataFrame(signals)
        
        # 生成汇总统计
        summary = {
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_signals": len(signals),
            "signal_distribution": {
                "强烈买入": len(df[df["信号类型"] == "强烈买入"]),
                "买入": len(df[df["信号类型"] == "买入"]),
                "谨慎买入": len(df[df["信号类型"] == "谨慎买入"]),
                "无信号": len(df[df["信号类型"] == "无信号"]),
                "卖出": len(df[df["信号类型"] == "卖出"]),
                "强烈卖出": len(df[df["信号类型"] == "强烈卖出"])
            },
            "confidence_distribution": {
                "高置信": len(df[df["置信度档位"] == "高置信"]),
                "中置信": len(df[df["置信度档位"] == "中置信"]),
                "低置信": len(df[df["置信度档位"] == "低置信"]),
                "无": len(df[df["置信度档位"] == "无"])
            },
            "average_strength": round(df["信号强度"].mean(), 2) if not df.empty else 0,
            "failure_rate": round((df["是否失效"].sum() / len(df)) * 100, 2) if not df.empty else 0,
            "recent_signals": signals[-5:]  # 最近5个信号
        }
        
        logger.info(f"汇总报告生成完成: 总信号数={summary['total_signals']}")
        
        # 保存报告
        if filename:
            if directory is None:
                directory = DATA_PATHS['signals']
            
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"汇总报告已保存: {filepath}")
            except Exception as e:
                logger.error(f"保存汇总报告失败: {str(e)}")
        
        return summary
    
    def batch_export(self, signals: List[Dict[str, any]], formats: List[str] = None, 
                    directory: str = None) -> Dict[str, bool]:
        """批量导出信号为多种格式
        
        Args:
            signals: 信号列表
            formats: 导出格式列表，默认 ['json', 'csv', 'excel']
            directory: 目录，如果为None则使用默认信号目录
            
        Returns:
            各格式导出结果字典
        """
        if formats is None:
            formats = ['json', 'csv', 'excel']
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出为JSON格式
        if 'json' in formats:
            filename = f"signals_batch_{timestamp}.json"
            results['json'] = self.export_to_json(signals, filename, directory)
        
        # 导出为CSV格式
        if 'csv' in formats:
            filename = f"signals_batch_{timestamp}.csv"
            results['csv'] = self.export_to_csv(signals, filename, directory)
        
        # 导出为Excel格式
        if 'excel' in formats:
            filename = f"signals_batch_{timestamp}.xlsx"
            results['excel'] = self.export_to_excel(signals, filename, directory)
        
        return results

def main():
    """测试信号输出器"""
    # 创建信号输出器实例
    exporter = SignalExporter()
    
    # 创建测试信号
    test_signal = {
        "symbol": "512660",
        "name": "军工ETF",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_type": "strong_buy",
        "signal_strength": 95,
        "confidence_level": "high",
        "weekly_confirmation": "高置信确认",
        "weekly_data": {"confidence_level": "high"},
        "daily_data": {
            "buy_signal": True,
            "divergence_strength": 0.85,
            "details": {
                "recent_fractals": {
                    "bottom_count": 3,
                    "top_count": 1
                }
            }
        },
        "conditions": {
            "weekly_high_confidence": True,
            "weekly_medium_confidence": False,
            "daily_buy_signal": True,
            "total_met_conditions": 2,
            "total_conditions": 3
        },
        "trade_recommendation": "强烈买入",
        "expired": False,
        "expiry_reason": "信号仍然有效"
    }
    
    # 格式化信号
    formatted_signal = exporter.format_signal(test_signal)
    
    # 在控制台打印信号
    exporter.print_signal_console(formatted_signal)
    
    # 导出为JSON
    exporter.export_to_json(formatted_signal, "test_signal.json")
    
    # 导出为CSV
    exporter.export_to_csv(formatted_signal, "test_signals.csv")
    
    # 导出为Excel
    exporter.export_to_excel(formatted_signal, "test_signals.xlsx")

if __name__ == "__main__":
    main()