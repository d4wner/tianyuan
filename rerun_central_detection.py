#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新运行中枢识别算法，检测512660 ETF 2025年的有效中枢
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的类
from universal_chanlun_etf_analyzer import UniversalChanlunETFAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RerunCentralDetection")


def rerun_central_detection():
    """
    重新运行中枢识别算法，检测2025年的有效中枢
    """
    print("="*80)
    print(f"开始重新运行中枢识别算法 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # 初始化分析器
        analyzer = UniversalChanlunETFAnalyzer(
            symbol='512660',
            year=2025
        )
        
        print("\n1. 加载数据...")
        analyzer.load_data()
        print(f"   数据加载成功")
        
        print("\n2. 生成标准K线...")
        analyzer.generate_standard_k_lines()
        print(f"   成功生成标准K线: {len(analyzer.standard_k_lines)}根")
        
        print("\n3. 计算波动率...")
        analyzer.calculate_volatility()
        print(f"   波动率: {analyzer.volatility:.2f}%")
        
        # 手动设置波动等级
        if hasattr(analyzer, 'volatility'):
            if analyzer.volatility < 10:
                analyzer.volatility_level = "低波动"
            elif analyzer.volatility < 15:
                analyzer.volatility_level = "中波动"
            else:
                analyzer.volatility_level = "高波动"
        print(f"   波动等级: {analyzer.volatility_level}")
        
        print("\n4. 识别有效盘整段...")
        # 初始化valid_segments为空列表
        analyzer.valid_segments = []
        
        # 尝试调用内部方法来识别盘整段
        try:
            if hasattr(analyzer, '_identify_consolidation_segments'):
                analyzer._identify_consolidation_segments()
                if hasattr(analyzer, 'valid_segments'):
                    print(f"   有效盘整段数量: {len(analyzer.valid_segments)}")
                else:
                    print("   无法获取有效盘整段信息")
            else:
                print("   未找到识别盘整段的方法")
        except Exception as e:
            print(f"   识别盘整段时出错: {str(e)}")
        
        # 只有在有有效盘整段时才尝试生成中枢
        valid_centrals = []
        if hasattr(analyzer, 'valid_segments') and analyzer.valid_segments:
            print("\n5. 生成中枢...")
            try:
                # 调用中枢生成方法
                if hasattr(analyzer, 'generate_centrals'):
                    analyzer.generate_centrals()
                    
                    # 检查是否有centrals和valid_centrals属性
                    centrals_count = len(getattr(analyzer, 'centrals', []))
                    valid_centrals_count = len(getattr(analyzer, 'valid_centrals', []))
                    valid_centrals = getattr(analyzer, 'valid_centrals', [])
                    
                    print(f"   生成中枢总数: {centrals_count}")
                    print(f"   有效中枢数量: {valid_centrals_count}")
                else:
                    print("   未找到生成中枢的方法")
            except Exception as e:
                print(f"   生成中枢时出错: {str(e)}")
        else:
            print("\n5. 生成中枢...")
            print("   跳过中枢生成: 无有效盘整段")
        
        # 显示有效中枢详情
        print("\n6. 有效中枢详情:")
        if valid_centrals:
            for i, central in enumerate(valid_centrals, 1):
                print(f"\n   中枢{i}:")
                print(f"   - 中枢ID: {central.get('central_id')}")
                print(f"   - 盘整段ID: {central.get('segment_id')}")
                print(f"   - 时间范围: {central.get('start_date')} 至 {central.get('end_date')}")
                print(f"   - 价格区间: {central.get('lower'):.3f} - {central.get('upper'):.3f}")
                print(f"   - 振幅: {central.get('amplitude'):.2f}%")
        else:
            print("   未识别到有效中枢")
            print("   注意：基于盘整段的中枢识别算法未找到有效中枢，但按季度预定义的中枢范围可能存在有效中枢")
        
        # 显示按季度预定义的中枢范围（从detect_512660_break_central_buy_signal.py中提取）
        print("\n7. 按季度预定义中枢范围:")
        quarterly_central = {
            'Q1_2025': {'low': 0.95, 'high': 1.05},  # Q1中枢范围
            'Q2_2025': {'low': 1.00, 'high': 1.10},  # Q2中枢范围
            'Q3_2025': {'low': 1.10, 'high': 1.20},  # Q3中枢范围
            'Q4_2025': {'low': 0.85, 'high': 0.95}   # Q4中枢范围
        }
        
        for quarter, range_info in quarterly_central.items():
            amplitude = ((range_info['high'] - range_info['low']) / range_info['low']) * 100
            print(f"   - {quarter}: {range_info['low']:.2f} - {range_info['high']:.2f} (振幅: {amplitude:.2f}%)")
        
        # 额外分析：直接基于价格数据识别潜在中枢
        print("\n8. 基于价格数据的额外中枢分析:")
        analyze_price_based_centrals()
        
        return analyzer
        
    except Exception as e:
        logger.error(f"重新运行中枢识别时出错: {str(e)}")
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_price_based_centrals():
    """
    直接从CSV文件加载数据并分析潜在中枢
    """
    try:
        # 直接从文件加载数据
        data_file = '/Users/pingan/tools/trade/tianyuan/data/daily/512660_daily.csv'
        df = pd.read_csv(data_file)
        
        # 筛选2025年数据
        df['date'] = pd.to_datetime(df['date'])
        df_2025 = df[df['date'].dt.year == 2025].copy()
        
        # 按季度分组分析
        df_2025['quarter'] = df_2025['date'].dt.to_period('Q')
        
        for quarter, quarter_data in df_2025.groupby('quarter'):
            # 计算该季度的最高价和最低价
            q_high = quarter_data['high'].max()
            q_low = quarter_data['low'].min()
            q_amplitude = ((q_high - q_low) / q_low) * 100
            
            # 计算90%成交区间
            prices = pd.concat([quarter_data['open'], quarter_data['high'], quarter_data['low'], quarter_data['close']])
            q_90_low = prices.quantile(0.05)
            q_90_high = prices.quantile(0.95)
            q_90_amplitude = ((q_90_high - q_90_low) / q_90_low) * 100
            
            print(f"   - 2025年{quarter.quarter}季度:")
            print(f"     价格范围: {q_low:.3f} - {q_high:.3f} (振幅: {q_amplitude:.2f}%)")
            print(f"     90%成交区间: {q_90_low:.3f} - {q_90_high:.3f} (振幅: {q_90_amplitude:.2f}%)")
    except Exception as e:
        print(f"   额外分析出错: {str(e)}")


if __name__ == "__main__":
    analyzer = rerun_central_detection()
    
    if analyzer:
        print("\n" + "="*80)
        print(f"中枢识别完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    else:
        print("\n中枢识别失败")