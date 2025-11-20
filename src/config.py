#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块 - 修复版
修复了策略配置加载问题，无需新增策略文件
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger('ConfigManager')

# 数据存储路径配置
DATA_PATHS = {
    'daily': 'data/daily',
    'weekly': 'data/weekly',
    'minute': 'data/minute',
    'signals': 'data/signals',
    'reports': 'outputs/reports',
    'exports': 'outputs/exports',
    'plots': 'outputs/plots'
}

def load_config(file_path='config/system.yaml'):
    """
    加载YAML配置文件
    :param file_path: 配置文件路径
    :return: 配置字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.warning(f"配置文件不存在: {file_path}")
            return {}
        
        # 读取配置文件
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {file_path}")
            return config or {}
    
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return {}

def save_config(config, file_path='config/system.yaml'):
    """
    保存配置到YAML文件
    :param config: 配置字典
    :param file_path: 配置文件路径
    """
    try:
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 写入配置文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            logger.info(f"配置文件保存成功: {file_path}")
    
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}")

def load_system_config():
    """
    加载系统配置
    :return: 系统配置字典
    """
    config = load_config('config/system.yaml')
    # 新的system.yaml结构是顶层有system键
    return config.get('system', {})

def get_data_fetcher_config():
    """
    获取数据获取器配置
    :return: 数据获取器配置字典
    """
    system_config = load_system_config()
    return system_config.get('data_fetcher', {})

def get_backtest_config():
    """
    获取回测配置
    :return: 回测配置字典
    """
    system_config = load_system_config()
    return system_config.get('backtest', {})

def get_strategy_config():
    """
    获取策略配置
    :return: 策略配置字典
    """
    system_config = load_system_config()
    return system_config.get('strategy', {})

def get_chanlun_config():
    """
    获取缠论计算配置
    :return: 缠论配置字典
    """
    system_config = load_system_config()
    return system_config.get('chanlun', {})

def get_risk_management_config():
    """
    获取风险管理配置
    :return: 风险管理配置字典
    """
    system_config = load_system_config()
    return system_config.get('risk_management', {})

def get_dingding_config():
    """
    获取钉钉通知配置
    :return: 钉钉配置字典
    """
    system_config = load_system_config()
    return system_config.get('dingding', {})

def get_data_paths():
    """
    获取数据存储路径配置
    :return: 数据路径字典
    """
    system_config = load_system_config()
    data_paths_config = system_config.get('data_paths', {})
    
    # 使用配置中的路径或默认路径
    return {
        'daily': data_paths_config.get('daily', DATA_PATHS['daily']),
        'weekly': data_paths_config.get('weekly', DATA_PATHS['weekly']),
        'minute': data_paths_config.get('minute', DATA_PATHS['minute']),
        'signals': data_paths_config.get('signals', DATA_PATHS['signals']),
        'reports': data_paths_config.get('reports', DATA_PATHS['reports']),
        'exports': data_paths_config.get('exports', DATA_PATHS['exports']),
        'plots': data_paths_config.get('plots', DATA_PATHS['plots'])
    }

def ensure_data_directories():
    """
    确保所有数据目录存在
    """
    data_paths = get_data_paths()
    
    for path_type, path in data_paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"确保目录存在: {path}")
        except Exception as e:
            logger.error(f"创建目录失败 {path}: {str(e)}")

def load_etf_config():
    """
    加载ETF配置文件
    :return: ETF配置字典
    """
    return load_config('config/etfs.yaml')

def load_risk_rules():
    """
    加载风控规则配置文件
    :return: 风控规则字典
    """
    return load_config('config/risk_rules.yaml')

def load_strategy(timeframe):
    """
    动态加载对应时间级别的策略配置
    :param timeframe: 时间级别，如 'daily', 'weekly', 'minute'
    :return: 策略配置字典
    """
    try:
        # 加载整个配置文件
        config = load_config('config/system.yaml')
        
        # 首先尝试从strategies节点加载（如果存在）
        strategies = config.get('strategies', {})
        if strategies and timeframe in strategies:
            strategy_config = strategies.get(timeframe)
            logger.info(f"策略配置加载成功: 时间级别 {timeframe}")
            return strategy_config
        
        # 如果strategies节点不存在，尝试从strategy_mapping加载外部文件
        strategy_mapping = config.get('strategy_mapping', {})
        if not strategy_mapping:
            logger.warning("策略映射配置未找到，请检查system.yaml中的strategy_mapping设置")
            return {}
        
        # 根据时间级别获取对应的配置文件路径
        config_file = strategy_mapping.get(timeframe)
        if not config_file:
            logger.error(f"未找到时间级别 '{timeframe}' 的策略映射配置")
            return {}
        
        # 检查文件是否存在
        if not os.path.exists(config_file):
            logger.warning(f"策略文件不存在: {config_file}")
            return {}
        
        # 加载对应的策略配置文件
        strategy_config = load_config(config_file)
        logger.info(f"策略配置加载成功: {config_file} for timeframe {timeframe}")
        return strategy_config
        
    except Exception as e:
        logger.error(f"加载策略配置失败: {str(e)}")
        return {}

# 配置日志
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 测试配置加载
    system_config = load_system_config()
    data_fetcher_config = get_data_fetcher_config()
    backtest_config = get_backtest_config()
    
    print("系统配置加载测试:")
    print(f"数据获取器配置: {bool(data_fetcher_config)}")
    print(f"回测配置: {bool(backtest_config)}")
    
    # 测试数据路径配置
    data_paths = get_data_paths()
    print("数据存储路径配置:")
    for path_type, path in data_paths.items():
        print(f"  {path_type}: {path}")
    
    # 确保目录存在
    ensure_data_directories()
    
    # 测试ETF和风控配置
    etf_config = load_etf_config()
    risk_rules = load_risk_rules()
    
    print(f"ETF配置: {bool(etf_config)}")
    print(f"风控规则: {bool(risk_rules)}")
    
    # 测试新增加的load_strategy函数
    # 注意：需要先确保system.yaml中已添加strategy_mapping配置
    test_timeframe = 'weekly'
    strategy_config = load_strategy(test_timeframe)
    print(f"策略配置 for {test_timeframe}: {bool(strategy_config)}")
    
    logger.info("配置模块加载完成")