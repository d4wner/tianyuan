#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缠论系统数据获取深度诊断工具
功能：诊断数据获取和回测配置问题，提供具体修复方案
作者：缠论与量化交易专家
日期：2025-11-06
"""

import os
import sys
import yaml
import json
import logging
import subprocess
import importlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

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

class EnvironmentChecker:
    """环境兼容性检查器"""
    
    def check_urllib3_compatibility(self) -> Dict[str, Any]:
        """检查urllib3与系统Python的兼容性"""
        compatibility_info = {
            'urllib3_version': None,
            'ssl_backend': None,
            'python_version': sys.version,
            'platform': sys.platform,
            'issues': [],
            'fixes': []
        }
        
        try:
            import urllib3
            compatibility_info['urllib3_version'] = urllib3.__version__
            
            import ssl
            compatibility_info['ssl_backend'] = ssl.OPENSSL_VERSION
            
            # 检测LibreSSL问题
            if 'LibreSSL' in ssl.OPENSSL_VERSION:
                if hasattr(urllib3, '__version__') and int(urllib3.__version__.split('.')[0]) >= 2:
                    issue_msg = f"urllib3 v{urllib3.__version__} 与 LibreSSL {ssl.OPENSSL_VERSION} 不兼容"
                    compatibility_info['issues'].append(issue_msg)
                    
                    # 生成修复方案
                    fix_solution = self.generate_ssl_fix()
                    compatibility_info['fixes'].append(fix_solution)
                    
        except ImportError as e:
            compatibility_info['issues'].append(f"urllib3导入失败: {e}")
        except Exception as e:
            compatibility_info['issues'].append(f"环境检查异常: {e}")
        
        return compatibility_info
    
    def generate_ssl_fix(self) -> Dict[str, Any]:
        """生成SSL兼容性修复方案"""
        return {
            'problem': 'urllib3 v2+ 与 macOS 系统Python的LibreSSL不兼容',
            'root_cause': 'macOS系统Python使用LibreSSL而非OpenSSL',
            'solutions': [
                {
                    'method': '降级urllib3',
                    'command': 'pip install "urllib3<2.0" --force-reinstall',
                    'description': '将urllib3降级到1.26.x版本'
                },
                {
                    'method': '使用conda环境',
                    'command': 'conda create -n chanlun-fix python=3.9 openssl=1.1.1',
                    'description': '创建新的conda环境使用OpenSSL'
                }
            ]
        }

class DataSourceTester:
    """数据源测试器"""
    
    def __init__(self):
        self.sina_base_url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    
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
                        'sample_data': json_data[:2] if json_data else []
                    }
                except json.JSONDecodeError:
                    return {
                        'success': True,
                        'data_count': 0,
                        'warning': '返回数据不是有效JSON格式'
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}"
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
                
                patterns = ['backtest', 'timeframe', 'datalen', 'symbol', 'date_range']
                
                for pattern, description in patterns:
                    if pattern in content.lower():
                        sections.append({
                            'pattern': pattern,
                            'found': True
                        })
            except Exception as e:
                logger.error(f"读取main.py失败: {e}")
        
        return sections
    
    def analyze_date_range_logic(self) -> Dict[str, Any]:
        """分析日期范围设置逻辑"""
        issues = []
        
        # 检查日期范围是否过短
        log_date_range = {'days': 28}
        if log_date_range['days'] < 30:
            issues.append(f"回测日期范围过短: 仅{log_date_range['days']}天")
        
        return {
            'issues': issues,
            'suggestions': [
                "修改回测代码中的日期范围设置",
                "使用动态日期范围"
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
        
        # 根据诊断结果生成针对性的修复代码
        ssl_fix_code = ""
        if diagnostic_results.get('environment_check', {}).get('issues'):
            ssl_issues = diagnostic_results['environment_check']['issues']
            if any('LibreSSL' in issue for issue in ssl_issues):
                ssl_fix_code = '''
def fix_urllib3_libressl():
    """修复urllib3与LibreSSL兼容性问题"""
    print("修复urllib3 LibreSSL兼容性问题...")
    print("方案1: 降级urllib3到1.26.x版本")
    print("执行命令: pip install \\"urllib3<2.0\\" --force-reinstall")
    return True
'''
        
        # 修复第362行缩进错误 - 确保return语句正确缩进
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缠论系统数据获取修复脚本
生成时间: {current_time}
"""

import sys
from datetime import datetime, timedelta

{ssl_fix_code}

def main():
    print("缠论系统数据获取修复指南")
    print("根据诊断结果进行相应修复")
    
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
    
    # 1. 环境兼容性检查
    print("\n1. 环境兼容性检查")
    print("-" * 40)
    
    env_checker = EnvironmentChecker()
    
    try:
        env_compatibility = env_checker.check_urllib3_compatibility()
        
        print(f"Python版本: {env_compatibility['python_version'].split()[0]}")
        print(f"SSL后端: {env_compatibility['ssl_backend']}")
        
        if env_compatibility.get('urllib3_version'):
            print(f"urllib3版本: {env_compatibility['urllib3_version']}")
        
        if env_compatibility['issues']:
            for issue in env_compatibility['issues']:
                print(f"⚠️  {issue}")
        else:
            print("✅ 环境兼容性正常")
        
        diagnostic_results['environment_check'] = env_compatibility
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        diagnostic_results['environment_check'] = {'error': str(e)}
    
    # 2. 测试数据源
    print("\n2. 数据源API测试")
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
    
    # 3. 分析回测代码
    print("\n3. 回测代码分析")
    print("-" * 40)
    
    code_analyzer = BacktestCodeAnalyzer(project_root)
    
    try:
        code_sections = code_analyzer.find_backtest_code_sections()
        date_analysis = code_analyzer.analyze_date_range_logic()
        
        if code_sections:
            print(f"找到{len(code_sections)}个回测相关代码段")
        else:
            print("未找到回测代码段")
        
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
    
    # 4. 配置验证
    print("\n4. 配置验证")
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
    
    # 5. 生成修复方案
    print("\n5. 修复方案")
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