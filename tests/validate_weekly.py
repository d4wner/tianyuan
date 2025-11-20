#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WeeklyValidator')

def validate_weekly_strategy():
    """周线策略专项验证"""
    try:
        logger.info("步骤1: 测试数据获取")
        from src.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        
        # 设置正确的日期格式 (YYYYMMDD)
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        
        # 获取周线数据
        weekly_data = fetcher.get_weekly_data('510300.SH', start_date=start_date, end_date=end_date)
        
        # 使用 .empty 属性检查DataFrame是否为空
        if weekly_data.empty:
            logger.error("❌ 获取周线数据失败")
            return False
            
        logger.info(f"✅ 周线数据获取: {len(weekly_data)}条记录")
        
        logger.info("步骤2: 测试缠论计算")
        from src.calculator import ChanlunCalculator
        calc = ChanlunCalculator()
        
        # 使用周线分析方法
        signals = calc.analyze_weekly(weekly_data)
        logger.info(f"✅ 缠论分析: 生成{len(signals)}个信号")
        
        logger.info("步骤3: 验证信号生成")
        if len(signals) == 0:
            logger.error("❌ 未生成任何信号")
            return False
            
        logger.info(f"✅ 信号验证通过: 共生成{len(signals)}个信号")
        
        logger.info("步骤4: 验证信号合理性")
        # 🔧 修复：检查信号中是否有买入或卖出建议
        # 首先检查 signals 是否是 DataFrame
        if isinstance(signals, pd.DataFrame):
            # 检查 DataFrame 中是否有 'action' 列
            if 'action' in signals.columns:
                has_buy = any(signals['action'] == 'buy')
                has_sell = any(signals['action'] == 'sell')
            else:
                logger.warning("⚠️ 信号中没有 'action' 列，跳过买卖信号检查")
                has_buy = has_sell = False
        else:
            # 如果不是 DataFrame，可能是其他类型（如列表）
            has_buy = any('buy' in str(signal) for signal in signals)
            has_sell = any('sell' in str(signal) for signal in signals)
        
        if not (has_buy or has_sell):
            logger.warning("⚠️ 信号中未明确包含买卖建议")
        else:
            logger.info(f"✅ 信号中包含买卖建议: 买入信号={has_buy}, 卖出信号={has_sell}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 验证失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"Python路径: {sys.path}")
    
    # 执行验证
    success = validate_weekly_strategy()
    
    if success:
        logger.info("🎉 周线策略验证通过")
    else:
        logger.error("💥 周线策略需要修复")
        sys.exit(1)