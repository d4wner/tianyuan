#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import pytest
import pandas as pd
import numpy as np
import logging

# 添加src目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calculator import ChanlunCalculator

# 配置日志
logger = logging.getLogger('TestCalculator')
logger.setLevel(logging.INFO)

class TestChanlunCalculator:
    """ChanlunCalculator测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        data = {
            'date': dates,
            'open': [10.0, 10.5, 10.8, 11.2, 10.9, 11.5, 11.8, 12.2, 11.9, 12.5],
            'high': [10.5, 11.0, 11.2, 11.5, 11.3, 12.0, 12.5, 12.8, 12.5, 13.0],
            'low': [9.8, 10.2, 10.5, 10.8, 10.5, 11.2, 11.5, 11.8, 11.5, 12.0],
            'close': [10.2, 10.8, 11.0, 11.3, 11.0, 11.8, 12.2, 12.5, 12.0, 12.8],
            'volume': [1000000, 1200000, 1500000, 1300000, 1100000, 1400000, 1600000, 1700000, 1500000, 1800000]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        return ChanlunCalculator(config={
            'fractal_sensitivity': 3,
            'pen_min_length': 2,
            'central_bank_min_bars': 3
        })
    
    def test_calculate_fractal(self, calculator, sample_data):
        """测试分型计算"""
        logger.info("测试分型计算")
        df = calculator.calculate_fractal(sample_data.copy())
        assert 'top_fractal' in df.columns
        assert 'bottom_fractal' in df.columns
        assert df['top_fractal'].dtype == bool
        assert df['bottom_fractal'].dtype == bool
        logger.info("分型计算测试通过")
    
    def test_calculate_pen(self, calculator, sample_data):
        """测试笔划分"""
        logger.info("测试笔划分")
        df = calculator.calculate_fractal(sample_data.copy())
        df = calculator.calculate_pen(df)
        assert 'pen_type' in df.columns
        assert 'pen_start' in df.columns
        assert 'pen_end' in df.columns
        logger.info("笔划分测试通过")
    
    def test_calculate_segment(self, calculator, sample_data):
        """测试线段划分"""
        logger.info("测试线段划分")
        df = calculator.calculate_fractal(sample_data.copy())
        df = calculator.calculate_pen(df)
        df = calculator.calculate_segment(df)
        assert 'segment_type' in df.columns
        assert 'segment_start' in df.columns
        assert 'segment_end' in df.columns
        logger.info("线段划分测试通过")
    
    def test_calculate_central_bank(self, calculator, sample_data):
        """测试中枢识别"""
        logger.info("测试中枢识别")
        df = calculator.calculate_fractal(sample_data.copy())
        df = calculator.calculate_pen(df)
        df = calculator.calculate_segment(df)
        df = calculator.calculate_central_bank(df)
        assert 'central_bank' in df.columns
        assert 'central_bank_high' in df.columns
        assert 'central_bank_low' in df.columns
        logger.info("中枢识别测试通过")
    
    def test_determine_market_condition(self, calculator, sample_data):
        """测试市场状况判断"""
        logger.info("测试市场状况判断")
        df = calculator.calculate(sample_data.copy())
        condition = calculator.determine_market_condition(df)
        assert condition in ['trending_up', 'trending_down', 'ranging', 'unknown']
        logger.info("市场状况判断测试通过")
    
    def test_calculate_stoploss(self, calculator, sample_data):
        """测试止损计算"""
        logger.info("测试止损计算")
        df = calculator.calculate(sample_data.copy())
        stoploss = calculator.calculate_stoploss(df)
        assert isinstance(stoploss, float)
        assert stoploss > 0
        logger.info("止损计算测试通过")
    
    def test_calculate_target_price(self, calculator, sample_data):
        """测试目标价计算"""
        logger.info("测试目标价计算")
        df = calculator.calculate(sample_data.copy())
        target_price = calculator.calculate_target_price(df, 'buy')
        assert isinstance(target_price, float)
        assert target_price > 0
        logger.info("目标价计算测试通过")
    
    def test_calculate_all_indicators(self, calculator, sample_data):
        """测试完整计算流程"""
        logger.info("测试完整计算流程")
        df = calculator.calculate(sample_data.copy())
        
        # 检查所有必要的列是否存在
        required_columns = [
            'top_fractal', 'bottom_fractal', 'pen_type', 'pen_start', 'pen_end',
            'segment_type', 'segment_start', 'segment_end', 'central_bank',
            'central_bank_high', 'central_bank_low'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"缺少必要列: {col}"
        
        logger.info("完整计算流程测试通过")

# 运行测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建测试实例
    test_instance = TestChanlunCalculator()
    
    # 准备测试数据
    sample_data = test_instance.sample_data()
    calculator = test_instance.calculator()
    
    # 运行所有测试
    test_instance.test_calculate_fractal(calculator, sample_data)
    test_instance.test_calculate_pen(calculator, sample_data)
    test_instance.test_calculate_segment(calculator, sample_data)
    test_instance.test_calculate_central_bank(calculator, sample_data)
    test_instance.test_determine_market_condition(calculator, sample_data)
    test_instance.test_calculate_stoploss(calculator, sample_data)
    test_instance.test_calculate_target_price(calculator, sample_data)
    test_instance.test_calculate_all_indicators(calculator, sample_data)
    
    logger.info("所有测试通过！")