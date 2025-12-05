#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的UniversalChanlunETFAnalyzer类的关键方法
"""

from universal_chanlun_etf_analyzer import UniversalChanlunETFAnalyzer

def test_analyzer_methods():
    print("开始测试UniversalChanlunETFAnalyzer类")
    
    # 初始化分析器
    analyzer = UniversalChanlunETFAnalyzer(symbol='512660', year=2025)
    print("初始化成功，开始测试关键方法")
    
    # 测试load_data方法
    print("\n测试load_data方法:")
    success = analyzer.load_data()
    print(f"数据加载{'成功' if success else '失败'}")
    
    if hasattr(analyzer, 'raw_data') and analyzer.raw_data is not None:
        print(f"原始数据行数: {len(analyzer.raw_data)}")
        print(f"数据字段: {list(analyzer.raw_data.columns)}")
    else:
        print("未加载到数据")
    
    # 测试generate_standard_k_lines方法
    print("\n测试generate_standard_k_lines方法:")
    if not analyzer.has_critical_error:
        success = analyzer.generate_standard_k_lines()
        print(f"标准K线生成{'成功' if success else '失败'}")
        
        if hasattr(analyzer, 'standard_k_lines') and analyzer.standard_k_lines is not None:
            print(f"标准K线数据行数: {len(analyzer.standard_k_lines)}")
        else:
            print("未生成标准K线数据")
    else:
        print("因前序错误，跳过标准K线生成测试")
    
    # 测试calculate_volatility方法
    print("\n测试calculate_volatility方法:")
    if not analyzer.has_critical_error:
        success = analyzer.calculate_volatility()
        print(f"波动率计算{'成功' if success else '失败'}")
        
        if hasattr(analyzer, 'volatility') and analyzer.volatility is not None:
            print(f"计算得到的波动率: {analyzer.volatility:.2f}%")
        else:
            print("未计算出波动率")
    else:
        print("因前序错误，跳过波动率计算测试")
    
    # 测试determine_volatility_level方法
    print("\n测试determine_volatility_level方法:")
    if not analyzer.has_critical_error:
        success = analyzer.determine_volatility_level()
        print(f"波动等级判定{'成功' if success else '失败'}")
        
        if hasattr(analyzer, 'volatility_level_name'):
            print(f"确定的波动等级: {analyzer.volatility_level_name}")
        else:
            print("未确定波动等级")
    else:
        print("因前序错误，跳过波动等级判定测试")
    
    print("\n所有关键方法测试完成")

if __name__ == "__main__":
    test_analyzer_methods()