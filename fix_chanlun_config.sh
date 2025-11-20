#!/bin/bash
# 缠论系统配置修复脚本
# 生成时间: 2025-11-03

echo "开始修复缠论系统配置..."

# 备份原配置文件
cp config/system.yaml config/system.yaml.backup.$(date +%Y%m%d_%H%M%S)

# 使用Python修复配置
python3 -c "
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
"

echo "修复完成！请重新运行回测程序。"

# 提示重新启动服务
echo "建议重启缠论系统服务:"
echo "1. 停止当前运行的系统"
echo "2. 重新启动: python src/main.py --backtest --timeframe weekly"
