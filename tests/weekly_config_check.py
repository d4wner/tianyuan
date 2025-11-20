#!/usr/bin/env python3
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_weekly_strategy():
    """周线策略专项验证"""
    try:
        # 初始化路径
        from import_helper import setup_paths
        setup_paths()
        
        # 1. 测试数据获取
        from src.data_fetcher import DataFetcher  # 使用实际类名
        fetcher = DataFetcher()
        weekly_data = fetcher.get_weekly_data('510300.SH', periods=10)
        print(f"✅ 周线数据获取: {len(weekly_data)}条记录")
        
        # 2. 测试缠论计算
        from src.calculator import ChanlunCalculator
        calc = ChanlunCalculator()
        signals = calc.analyze_weekly(weekly_data)
        print(f"✅ 缠论分析: 生成{len(signals)}个信号")
        
        # 3. 测试监控模块
        from src.monitor import Monitor
        monitor = Monitor()
        trades = monitor.generate_signals(signals, timeframe='weekly')
        print(f"✅ 信号监控: 生成{len(trades)}个交易信号")
        
        # 4. 测试回测
        from src.backtester import Backtester
        bt = Backtester()
        result = bt.run_backtest(trades)
        print(f"✅ 回测完成: 收益率{result.get('return_pct', 0):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if validate_weekly_strategy():
        print("\n🎉 周线策略验证通过！")
    else:
        print("\n💥 周线策略需要修复")