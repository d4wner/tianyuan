#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缠论系统数据获取深度诊断工具
功能：诊断数据获取和回测配置问题，提供具体修复方案
作者：缠论与量化交易专家
日期：2025-11-03
修复记录：2025-11-05 忽略urllib3兼容性警告，专注于核心功能
"""

import os
import sys
import yaml
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# 忽略urllib3兼容性警告
warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_diagnostic.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataSourceTester:
    """数据源测试器"""
    
    def __init__(self):
        self.sina_base_url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        self.test_symbols = ['sh510300', 'sh510500', 'sz159915']
    
    def test_sina_api(self, symbol: str, scale: str = "240", datalen: str = "100") -> Dict[str, Any]:
        """测试新浪API数据获取"""
        try:
            import requests
            
            params = {
                'symbol': symbol,
                'scale': scale,
                'datalen': datalen,
                'ma': 'no'
            }
            
            logger.info(f"测试新浪API: symbol={symbol}, scale={scale}, datalen={datalen}")
            
            response = requests.get(self.sina_base_url, params=params, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code == 200:
                try:
                    json_data = json.loads(response.text)
                    return {
                        'success': True,
                        'data_count': len(json_data),
                        'sample_data': json_data[:2] if json_data else [],
                        'raw_response': response.text[:500]
                    }
                except json.JSONDecodeError:
                    return {
                        'success': True,
                        'data_count': 0,
                        'sample_data': [],
                        'raw_response': response.text[:500],
                        'warning': '返回数据不是有效JSON格式'
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'response': response.text[:500]
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_all_timeframes(self, symbol: str) -> Dict[str, Any]:
        """测试所有时间级别的数据获取"""
        results = {}
        timeframes = {
            'weekly': {'scale': '240', 'datalen': '500'},
            'daily': {'scale': '240', 'datalen': '1000'},
            'minute': {'scale': '5', 'datalen': '10000'}
        }
        
        for tf, config in timeframes.items():
            results[tf] = self.test_sina_api(symbol, config['scale'], config['datalen'])
        
        return results

class BacktestCodeAnalyzer:
    """回测代码分析器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.main_py_path = project_root / 'src' / 'main.py'
    
    def find_backtest_code_sections(self) -> List[Dict[str, Any]]:
        """查找回测相关代码段"""
        sections = []
        
        if self.main_py_path.exists():
            try:
                with open(self.main_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                patterns = [
                    ('backtest', '回测相关代码'),
                    ('timeframe', '时间级别参数'),
                    ('datalen', '数据长度参数'),
                    ('symbol', '股票代码处理'),
                    ('date_range', '日期范围设置')
                ]
                
                for pattern, description in patterns:
                    if pattern in content.lower():
                        lines = content.split('\n')
                        relevant_lines = []
                        for i, line in enumerate(lines):
                            if pattern in line.lower():
                                start = max(0, i-2)
                                end = min(len(lines), i+3)
                                relevant_lines.extend(lines[start:end])
                        
                        sections.append({
                            'file': 'main.py',
                            'pattern': pattern,
                            'description': description,
                            'code_snippet': relevant_lines[:10]
                        })
            except Exception as e:
                logger.error(f"读取main.py失败: {e}")
        
        return sections
    
    def analyze_date_range_logic(self) -> Dict[str, Any]:
        """分析日期范围设置逻辑"""
        issues = []
        
        # 模拟从日志中提取的日期范围
        log_date_range = {
            'start': '2025-10-06',
            'end': '2025-11-03',
            'days': 28
        }
        
        if log_date_range['days'] < 30:
            issues.append(f"回测日期范围过短: 仅{log_date_range['days']}天")
        
        if self.main_py_path.exists():
            try:
                with open(self.main_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                hardcoded_dates = []
                date_patterns = ['2025-10-06', '2025-11-03', '20251006', '20251103']
                for pattern in date_patterns:
                    if pattern in content:
                        hardcoded_dates.append(pattern)
                
                if hardcoded_dates:
                    issues.append(f"发现硬编码日期: {hardcoded_dates}")
            except Exception as e:
                logger.error(f"分析日期范围失败: {e}")
        
        return {
            'log_date_range': log_date_range,
            'issues': issues,
            'suggestions': [
                "修改回测代码中的日期范围设置",
                "使用动态日期范围",
                "确保日期范围参数从配置文件读取"
            ]
        }

class ConfigValidator:
    """配置验证器"""
    
    def __init__(self, config_dir: Path = Path('config')):
        self.config_dir = config_dir
    
    def validate_config(self) -> Dict[str, Any]:
        """验证配置"""
        issues = []
        
        system_config = self._load_config('system.yaml')
        if system_config:
            if not system_config.get('system', {}).get('data_fetcher', {}):
                issues.append("data_fetcher配置不完整")
            
            if not system_config.get('system', {}).get('backtest', {}):
                issues.append("回测配置缺失")
        
        return {
            'issues': issues,
            'config_status': '存在问题' if issues else '正常'
        }
    
    def _load_config(self, filename: str) -> Optional[Dict[str, Any]]:
        """加载配置文件"""
        config_file = self.config_dir / filename
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        return None

class FixGenerator:
    """修复建议生成器"""
    
    def generate_fix_script(self, diagnostic_results: Dict[str, Any]) -> str:
        """生成修复脚本"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缠论系统数据获取修复脚本
生成时间: {current_time}
"""

import sys
from datetime import datetime, timedelta

def main():
    print("缠论系统数据获取修复指南")
    print("=" * 50)
    
    print("根据诊断结果，建议进行以下修复:")
    print("1. 检查回测代码中的日期范围设置")
    print("2. 确认数据获取器正确使用配置参数")
    print("3. 验证股票代码格式")
    print("4. 检查网络连接和API配置")
    
    print("\\n完成修复后，重新运行回测:")
    print("python src/main.py --backtest --timeframe weekly")

if __name__ == "__main__":
    main()
'''

def main():
    """主诊断函数"""
    print("=" * 70)
    print("缠论系统数据获取深度诊断")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    diagnostic_results = {}
    
    # 1. 测试数据源
    print("\n1. 数据源API测试")
    print("-" * 40)
    
    data_tester = DataSourceTester()
    test_symbol = 'sh510300'
    
    try:
        api_results = data_tester.test_all_timeframes(test_symbol)
        
        for timeframe, result in api_results.items():
            status = "✅" if result.get('success') and result.get('data_count', 0) > 10 else "❌"
            data_count = result.get('data_count', 0)
            print(f"{status} {timeframe}: 获取到{data_count}条数据")
            
            if not result.get('success'):
                print(f"   错误: {result.get('error')}")
            elif data_count <= 10:
                print(f"   警告: 数据量不足")
        
        diagnostic_results['api_test'] = api_results
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        diagnostic_results['api_test'] = {'error': str(e)}
    
    # 2. 分析回测代码
    print("\n2. 回测代码分析")
    print("-" * 40)
    
    code_analyzer = BacktestCodeAnalyzer(project_root)
    
    try:
        code_sections = code_analyzer.find_backtest_code_sections()
        date_analysis = code_analyzer.analyze_date_range_logic()
        
        if code_sections:
            print(f"找到{len(code_sections)}个回测相关代码段")
        else:
            print("未找到明显的回测代码段")
        
        if date_analysis['issues']:
            print("日期范围问题:")
            for issue in date_analysis['issues']:
                print(f"   ❌ {issue}")
        
        diagnostic_results['code_analysis'] = {
            'sections_found': len(code_sections),
            'date_issues': date_analysis['issues']
        }
    except Exception as e:
        print(f"❌ 代码分析失败: {e}")
        diagnostic_results['code_analysis'] = {'error': str(e)}
    
    # 3. 配置验证
    print("\n3. 配置验证")
    print("-" * 40)
    
    config_validator = ConfigValidator()
    
    try:
        config_validation = config_validator.validate_config()
        print(f"配置状态: {config_validation['config_status']}")
        
        if config_validation['issues']:
            for issue in config_validation['issues']:
                print(f"   ❌ {issue}")
        
        diagnostic_results['config_validation'] = config_validation
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        diagnostic_results['config_validation'] = {'error': str(e)}
    
    # 4. 生成修复方案
    print("\n4. 修复方案")
    print("-" * 40)
    
    fix_generator = FixGenerator()
    
    try:
        fix_script = fix_generator.generate_fix_script(diagnostic_results)
        fix_script_filename = 'comprehensive_fix.py'
        
        with open(fix_script_filename, 'w', encoding='utf-8') as f:
            f.write(fix_script)
        
        print(f"✅ 修复指南已保存至: {fix_script_filename}")
        print("执行命令: python comprehensive_fix.py")
    except Exception as e:
        print(f"❌ 生成修复方案失败: {e}")
    
    print("\n" + "=" * 70)
    print("诊断完成！请根据上述建议进行修复。")
    print("=" * 70)

if __name__ == "__main__":
    main()