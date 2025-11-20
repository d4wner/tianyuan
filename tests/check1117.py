#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论周线参数调试脚本 - 专门用于验证参数传递问题
打印关键数据流，帮助诊断问题根源
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_weekly_params.log', encoding='utf-8')
    ]
)

logger = logging.getLogger('WeeklyParamsDebug')

class MockDataFetcher:
    """模拟数据获取器，用于测试"""
    
    def get_weekly_data(self, symbol, start_date, end_date):
        """生成模拟周线数据"""
        logger.info(f"🔧 生成模拟周线数据: {symbol}, {start_date} 至 {end_date}")
        
        # 创建测试数据
        dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        data = {
            'date': dates,
            'open': np.random.rand(len(dates)) * 100 + 100,
            'high': np.random.rand(len(dates)) * 10 + 110,
            'low': np.random.rand(len(dates)) * 10 + 90,
            'close': np.random.rand(len(dates)) * 10 + 100,
            'volume': np.random.rand(len(dates)) * 1000000
        }
        
        df = pd.DataFrame(data)
        logger.info(f"📊 模拟数据生成完成: {len(df)}条记录")
        logger.info(f"📅 日期范围: {df['date'].min()} 至 {df['date'].max()}")
        return df

class DebugChanlunCalculator:
    """调试版缠论计算器 - 专门打印参数传递"""
    
    def __init__(self, config=None):
        logger.info("🚀 初始化调试版缠论计算器")
        self.config = config or {}
        
        # 打印所有配置参数
        logger.info("=== 配置参数详情 ===")
        for key, value in self.config.items():
            logger.info(f"🔧 {key}: {value}")
        
        # 提取关键参数
        self.weekly_fractal_sensitivity = self.config.get('weekly_fractal_sensitivity', 'NOT_SET')
        self.weekly_pen_min_length = self.config.get('weekly_pen_min_length', 'NOT_SET')
        self.weekly_central_min_length = self.config.get('weekly_central_min_length', 'NOT_SET')
        
        logger.info("=== 周线参数提取结果 ===")
        logger.info(f"📈 weekly_fractal_sensitivity: {self.weekly_fractal_sensitivity}")
        logger.info(f"📈 weekly_pen_min_length: {self.weekly_pen_min_length}")
        logger.info(f"📈 weekly_central_min_length: {self.weekly_central_min_length}")
        
        # 默认参数
        self.default_fractal_sensitivity = 3
        self.default_pen_min_length = 5
        self.default_central_min_length = 5
        
        logger.info("=== 默认参数 ===")
        logger.info(f"📊 default_fractal_sensitivity: {self.default_fractal_sensitivity}")
        logger.info(f"📊 default_pen_min_length: {self.default_pen_min_length}")
        logger.info(f"📊 default_central_min_length: {self.default_central_min_length}")

    def calculate(self, df, timeframe='daily'):
        """计算缠论元素 - 详细打印参数使用情况"""
        logger.info(f"\n🎯 开始计算缠论元素")
        logger.info(f"⏰ 时间级别: {timeframe}")
        logger.info(f"📊 数据点数: {len(df)}")
        
        # 打印当前使用的参数
        if timeframe == 'weekly':
            logger.info("=== 周线参数应用 ===")
            fractal_param = self.weekly_fractal_sensitivity if self.weekly_fractal_sensitivity != 'NOT_SET' else self.default_fractal_sensitivity
            pen_param = self.weekly_pen_min_length if self.weekly_pen_min_length != 'NOT_SET' else self.default_pen_min_length
            central_param = self.weekly_central_min_length if self.weekly_central_min_length != 'NOT_SET' else self.default_central_min_length
            
            logger.info(f"✅ 实际使用分型敏感度: {fractal_param}")
            logger.info(f"✅ 实际使用笔最小长度: {pen_param}")
            logger.info(f"✅ 实际使用中枢最小长度: {central_param}")
            
        else:
            logger.info(f"ℹ️  {timeframe}级别使用默认参数")
            logger.info(f"📊 分型敏感度: {self.default_fractal_sensitivity}")
            logger.info(f"📊 笔最小长度: {self.default_pen_min_length}")
            logger.info(f"📊 中枢最小长度: {self.default_central_min_length}")
        
        # 模拟计算过程
        result = self._mock_calculation(df, timeframe)
        return result
    
    def _mock_calculation(self, df, timeframe):
        """模拟计算过程"""
        logger.info(f"\n🔍 模拟计算过程开始")
        
        # 添加一些测试列
        df['top_fractal'] = False
        df['bottom_fractal'] = False
        df['pen_type'] = None
        df['central_bank'] = False
        
        # 模拟一些计算逻辑
        if len(df) > 5:
            # 随机标记一些分型点用于测试
            indices = np.random.choice(len(df), min(5, len(df)//3), replace=False)
            for idx in indices:
                if np.random.rand() > 0.5:
                    df.loc[idx, 'top_fractal'] = True
                    logger.debug(f"📌 标记顶分型 at index {idx}")
                else:
                    df.loc[idx, 'bottom_fractal'] = True
                    logger.debug(f"📌 标记底分型 at index {idx}")
        
        logger.info(f"✅ 模拟计算完成")
        logger.info(f"📊 顶分型数量: {df['top_fractal'].sum()}")
        logger.info(f"📊 底分型数量: {df['bottom_fractal'].sum()}")
        
        return df

class ParameterValidator:
    """参数验证器 - 专门检查配置传递"""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_config_structure(self, config):
        """验证配置结构"""
        logger.info("\n🔍 开始验证配置结构")
        
        results = []
        
        # 检查根级别配置
        if not config:
            results.append("❌ 配置为空")
            return results
        
        # 检查缠论配置
        chanlun_config = config.get('chanlun', {})
        if not chanlun_config:
            results.append("❌ 缺少chanlun配置")
        else:
            results.append("✅ 找到chanlun配置")
            
            # 检查周线参数
            weekly_params = [
                'weekly_fractal_sensitivity',
                'weekly_pen_min_length', 
                'weekly_central_min_length'
            ]
            
            for param in weekly_params:
                if param in chanlun_config:
                    results.append(f"✅ 找到{param}: {chanlun_config[param]}")
                else:
                    results.append(f"❌ 缺少{param}")
        
        self.validation_results.extend(results)
        return results
    
    def test_parameter_flow(self, test_cases):
        """测试参数流转"""
        logger.info("\n🔬 开始参数流转测试")
        
        for i, (config, timeframe) in enumerate(test_cases, 1):
            logger.info(f"\n📋 测试用例 {i}: timeframe={timeframe}")
            logger.info(f"⚙️ 配置: {config}")
            
            calculator = DebugChanlunCalculator(config)
            result = calculator.calculate(pd.DataFrame(), timeframe)
            
            logger.info(f"✅ 测试用例 {i} 完成")

def create_test_configs():
    """创建测试配置"""
    
    # 正确配置
    correct_config = {
        'chanlun': {
            'weekly_fractal_sensitivity': 2,
            'weekly_pen_min_length': 3,
            'weekly_central_min_length': 3,
            'fractal_sensitivity': 3,
            'pen_min_length': 5
        }
    }
    
    # 错误配置 - 参数在错误的位置
    wrong_location_config = {
        'fractal_sensitivity': 3,
        'pen_min_length': 5,
        'weekly_fractal_sensitivity': 2,  # 应该在chanlun子配置中
        'weekly_pen_min_length': 3
    }
    
    # 缺失配置
    missing_config = {
        'chanlun': {
            'fractal_sensitivity': 3,
            'pen_min_length': 5
            # 缺少周线参数
        }
    }
    
    return [
        (correct_config, 'weekly'),
        (wrong_location_config, 'weekly'), 
        (missing_config, 'weekly'),
        (correct_config, 'daily')  # 测试日线级别
    ]

def test_dataframe_integrity():
    """测试DataFrame完整性"""
    logger.info("\n📊 开始DataFrame完整性测试")
    
    # 测试1: 基本DataFrame
    df1 = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'open': range(10),
        'high': range(10, 20),
        'low': range(20, 30),
        'close': range(30, 40)
    })
    
    logger.info(f"✅ 基本DataFrame创建成功")
    logger.info(f"📋 列名: {list(df1.columns)}")
    logger.info(f"📏 形状: {df1.shape}")
    
    # 测试2: 包含缠论列的DataFrame
    df2 = df1.copy()
    df2['top_fractal'] = False
    df2['pen_type'] = 'up'
    
    logger.info(f"✅ 缠论DataFrame创建成功")
    logger.info(f"📋 列名: {list(df2.columns)}")
    
    return df1, df2

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("🧪 缠论周线参数调试脚本启动")
    logger.info("=" * 60)
    
    # 1. 测试配置验证
    validator = ParameterValidator()
    test_configs = create_test_configs()
    
    logger.info("\n" + "="*50)
    logger.info("1. 配置结构验证")
    logger.info("="*50)
    
    for config, _ in test_configs:
        validator.validate_config_structure(config)
    
    # 2. 测试参数流转
    logger.info("\n" + "="*50)
    logger.info("2. 参数流转测试")
    logger.info("="*50)
    
    validator.test_parameter_flow(test_configs)
    
    # 3. 测试DataFrame完整性
    logger.info("\n" + "="*50)
    logger.info("3. DataFrame完整性测试")
    logger.info("="*50)
    
    test_dataframe_integrity()
    
    # 4. 完整流程测试
    logger.info("\n" + "="*50)
    logger.info("4. 完整流程测试")
    logger.info("="*50)
    
    # 使用正确的配置测试完整流程
    correct_config = {
        'chanlun': {
            'weekly_fractal_sensitivity': 2,
            'weekly_pen_min_length': 3,
            'weekly_central_min_length': 3
        }
    }
    
    calculator = DebugChanlunCalculator(correct_config)
    data_fetcher = MockDataFetcher()
    
    # 获取数据
    df = data_fetcher.get_weekly_data('000001', '2024-01-01', '2024-06-01')
    
    # 进行计算
    result = calculator.calculate(df, 'weekly')
    
    logger.info("\n" + "="*50)
    logger.info("📋 最终结果摘要")
    logger.info("="*50)
    logger.info(f"✅ 计算完成")
    logger.info(f"📊 结果DataFrame形状: {result.shape}")
    logger.info(f"📋 结果列名: {list(result.columns)}")
    logger.info(f"📈 缠论列是否存在: {'top_fractal' in result.columns}")
    
    # 打印验证结果摘要
    logger.info("\n" + "="*60)
    logger.info("📋 验证结果摘要")
    logger.info("="*60)
    
    if hasattr(validator, 'validation_results'):
        for result in validator.validation_results:
            logger.info(result)
    
    logger.info("\n" + "="*60)
    logger.info("✅ 调试脚本执行完成")
    logger.info("📁 详细日志已保存到: debug_weekly_params.log")
    logger.info("="*60)

if __name__ == "__main__":
    main()