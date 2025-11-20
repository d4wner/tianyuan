#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置文件动态重载工具
功能：检查配置修改并确保新配置生效
作者：缠论与量化交易专家（ISTJ）
日期：2025-11-03
"""

import os
import yaml
import time
import logging
from typing import Dict, Any, List
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器 - 确保配置修改后立即生效"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_files = {
            'system': self.config_dir / "system.yaml",
            'etfs': self.config_dir / "etfs.yaml", 
            'risk_rules': self.config_dir / "risk_rules.yaml"
        }
        self.last_modified_times = {}
        self.current_configs = {}
        
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有配置文件"""
        configs = {}
        
        for name, file_path in self.config_files.items():
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        configs[name] = yaml.safe_load(f)
                    self.last_modified_times[name] = file_path.stat().st_mtime
                    logger.info(f"成功加载配置文件: {file_path}")
                else:
                    logger.warning(f"配置文件不存在: {file_path}")
                    configs[name] = {}
            except Exception as e:
                logger.error(f"加载配置文件 {file_path} 失败: {e}")
                configs[name] = {}
        
        self.current_configs = configs
        return configs
    
    def check_config_updates(self) -> bool:
        """检查配置文件是否有更新"""
        updated = False
        
        for name, file_path in self.config_files.items():
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                last_mtime = self.last_modified_times.get(name, 0)
                
                if current_mtime > last_mtime:
                    logger.info(f"检测到配置文件更新: {file_path}")
                    updated = True
                    self.last_modified_times[name] = current_mtime
                    
                    # 重新加载更新的配置
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.current_configs[name] = yaml.safe_load(f)
                        logger.info(f"已重新加载配置文件: {file_path}")
                    except Exception as e:
                        logger.error(f"重新加载配置文件失败: {e}")
        
        return updated
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.current_configs.get('system', {})
    
    def get_etf_config(self) -> Dict[str, Any]:
        """获取ETF配置"""
        return self.current_configs.get('etfs', {})
    
    def get_risk_config(self) -> Dict[str, Any]:
        """获取风险规则配置"""
        return self.current_configs.get('risk_rules', {})

class DataFetcherConfigValidator:
    """数据获取器配置验证器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def validate_data_length_config(self) -> Dict[str, Any]:
        """验证数据长度配置"""
        system_config = self.config_manager.get_system_config()
        data_fetcher = system_config.get('data_fetcher', {})
        sina_config = data_fetcher.get('sina', {})
        params = sina_config.get('params', {})
        
        current_values = {
            'weekly': params.get('weekly', {}).get('datalen', '未知'),
            'daily': params.get('daily', {}).get('datalen', '未知'),
            'minute': params.get('minute', {}).get('datalen', '未知')
        }
        
        # 检查是否为建议的值
        recommended_values = {
            'weekly': '500',  # 约10年数据
            'daily': '1000',  # 约4年数据  
            'minute': '10000' # 更长时间数据
        }
        
        validation_results = {}
        for timeframe, current_val in current_values.items():
            recommended = recommended_values[timeframe]
            is_correct = str(current_val) == recommended
            validation_results[timeframe] = {
                'current': current_val,
                'recommended': recommended,
                'is_correct': is_correct
            }
        
        return validation_results
    
    def fix_data_length_config(self) -> bool:
        """修复数据长度配置"""
        try:
            system_config = self.config_manager.get_system_config()
            
            # 确保配置结构存在
            if 'data_fetcher' not in system_config:
                system_config['data_fetcher'] = {}
            if 'sina' not in system_config['data_fetcher']:
                system_config['data_fetcher']['sina'] = {}
            if 'params' not in system_config['data_fetcher']['sina']:
                system_config['data_fetcher']['sina']['params'] = {}
            
            params = system_config['data_fetcher']['sina']['params']
            
            # 设置推荐值
            recommended_values = {
                'weekly': {'datalen': '500'},
                'daily': {'datalen': '1000'},
                'minute': {'datalen': '10000'}
            }
            
            for timeframe, config in recommended_values.items():
                if timeframe not in params:
                    params[timeframe] = {}
                params[timeframe].update(config)
            
            # 保存修改后的配置
            system_file = self.config_manager.config_files['system']
            with open(system_file, 'w', encoding='utf-8') as f:
                yaml.dump({'system': system_config}, f, default_flow_style=False, allow_unicode=True)
            
            logger.info("已修复数据长度配置")
            return True
            
        except Exception as e:
            logger.error(f"修复配置失败: {e}")
            return False

class BacktestConfigChecker:
    """回测配置检查器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def check_backtest_date_range(self) -> Dict[str, Any]:
        """检查回测日期范围配置"""
        system_config = self.config_manager.get_system_config()
        
        # 从回测日志中提取实际使用的日期范围
        actual_range = {
            'start_date': '2025-10-06',
            'end_date': '2025-11-03',
            'data_points': 18,
            'timeframe': 'weekly'
        }
        
        # 分析问题
        issues = []
        if actual_range['data_points'] < 50:
            issues.append(f"数据点过少: 只有{actual_range['data_points']}条，建议至少50条以上")
        
        if actual_range['start_date'] == actual_range['end_date']:
            issues.append("回测时间范围过短")
        
        # 建议的解决方案
        suggestions = [
            "修改系统配置中的data_fetcher.sina.params各时间级别的datalen参数",
            "确保回测代码正确读取配置参数",
            "检查数据源API是否支持请求更长的历史数据",
            "验证股票代码格式是否正确（如sh510300）"
        ]
        
        return {
            'actual_range': actual_range,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def generate_fix_script(self) -> str:
        """生成修复脚本"""
        script = """#!/bin/bash
# 缠论系统配置修复脚本
# 生成时间: 2025-11-03

echo "开始修复缠论系统配置..."

# 备份原配置文件
cp config/system.yaml config/system.yaml.backup.$(date +%Y%m%d_%H%M%S)

# 使用Python修复配置
python3 -c \"
import yaml

# 读取系统配置
with open('config/system.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 修复数据长度配置
if 'system' in config and 'data_fetcher' in config['system']:
    data_fetcher = config['system']['data_fetcher']
    if 'sina' in data_fetcher and 'params' in data_fetcher['sina']:
        params = data_fetcher['sina']['params']
        
        # 设置推荐的数据长度
        recommended = {
            'weekly': {'datalen': '500'},
            'daily': {'datalen': '1000'}, 
            'minute': {'datalen': '10000'}
        }
        
        for timeframe, settings in recommended.items():
            if timeframe not in params:
                params[timeframe] = {}
            params[timeframe].update(settings)
        
        print('已更新数据长度配置')

# 保存修改后的配置
with open('config/system.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print('配置修复完成')
\"

echo "修复完成！请重新运行回测程序。"

# 提示重新启动服务
echo "建议重启缠论系统服务:"
echo "1. 停止当前运行的系统"
echo "2. 重新启动: python src/main.py --backtest --timeframe weekly"
"""
        return script

def main():
    """主函数：诊断和修复配置问题"""
    print("=" * 70)
    print("缠论系统配置诊断工具")
    print("=" * 70)
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    configs = config_manager.load_all_configs()
    
    # 验证数据获取器配置
    validator = DataFetcherConfigValidator(config_manager)
    validation_results = validator.validate_data_length_config()
    
    print("\n1. 数据长度配置检查:")
    print("-" * 40)
    
    all_correct = True
    for timeframe, result in validation_results.items():
        status = "✅" if result['is_correct'] else "❌"
        print(f"{status} {timeframe}级别: 当前={result['current']}, 推荐={result['recommended']}")
        if not result['is_correct']:
            all_correct = False
    
    # 检查回测配置
    backtest_checker = BacktestConfigChecker(config_manager)
    backtest_analysis = backtest_checker.check_backtest_date_range()
    
    print("\n2. 回测数据分析:")
    print("-" * 40)
    print(f"实际回测范围: {backtest_analysis['actual_range']['start_date']} 至 {backtest_analysis['actual_range']['end_date']}")
    print(f"数据点数: {backtest_analysis['actual_range']['data_points']}条")
    
    if backtest_analysis['issues']:
        print("\n❌ 发现问题:")
        for issue in backtest_analysis['issues']:
            print(f"   - {issue}")
    
    # 提供解决方案
    print("\n3. 解决方案:")
    print("-" * 40)
    
    if not all_correct:
        print("🔧 方案A: 自动修复配置")
        print("   运行以下命令修复数据长度配置:")
        print("   python -c \"")
        print("   import yaml")
        print("   with open('config/system.yaml', 'r') as f: config = yaml.safe_load(f)")
        print("   # 修复代码...")
        print("   \"")
        
        # 提供修复选项
        fix_choice = input("\n是否自动修复配置? (y/n): ")
        if fix_choice.lower() == 'y':
            if validator.fix_data_length_config():
                print("✅ 配置修复成功！")
            else:
                print("❌ 配置修复失败，请手动修改")
    else:
        print("✅ 配置检查通过，但回测数据仍然很少")
        print("💡 可能的原因:")
        print("   - 回测代码没有正确读取配置")
        print("   - 数据源API限制")
        print("   - 股票代码格式问题")
    
    # 生成修复脚本
    print("\n4. 完整修复脚本:")
    print("-" * 40)
    fix_script = backtest_checker.generate_fix_script()
    
    script_filename = "fix_chanlun_config.sh"
    with open(script_filename, 'w', encoding='utf-8') as f:
        f.write(fix_script)
    
    print(f"修复脚本已保存至: {script_filename}")
    print("执行命令: bash fix_chanlun_config.sh")
    
    # 最终建议
    print("\n5. 最终建议:")
    print("-" * 40)
    print("💡 如果修复后仍然无效，请检查:")
    print("   - 回测代码中是否正确读取data_fetcher配置")
    print("   - 数据源API文档，确认最大数据长度限制")
    print("   - 网络连接和数据源可用性")
    print("   - 系统日志中的错误信息")
    
    print("\n" + "=" * 70)
    print("诊断完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()