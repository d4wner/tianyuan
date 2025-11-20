#!/usr/bin/env python3
import sys
import os

# 获取当前工作目录
current_dir = os.getcwd()
sys.path.append(current_dir)

try:
    from src.monitor import Monitor
    print('✅ 监控模块初始化成功')
    
    # 添加更多测试代码...
    monitor = Monitor()
    print(f'✅ 监控对象创建成功: {type(monitor)}')
    
except Exception as e:
    print(f'❌ 初始化失败: {e}')
    import traceback
    traceback.print_exc()