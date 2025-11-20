#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pytest
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logger = logging.getLogger('TestDataFetcher')
logger.setLevel(logging.INFO)

# 导入模块
from src.data_fetcher import StockDataAPI

class TestDataFetcher:
    """StockDataAPI测试类"""
    
    @pytest.fixture
    def api(self):
        """创建API实例"""
        return StockDataAPI(max_retries=1, timeout=5)
    
    def test_get_daily_data_success(self, api):
        """测试成功获取日线数据"""
        logger.info("测试获取日线数据")
        df = api.get_daily_data('510300', '20250101', '20251231')
        assert not df.empty
        assert 'date' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns
        assert 'symbol' in df.columns
        logger.info("日线数据获取测试通过")
    
    def test_get_daily_data_empty(self, api):
        """测试获取空日线数据"""
        logger.info("测试空日线数据场景")
        df = api.get_daily_data('invalid_symbol', '20250101', '20251231')
        assert df.empty
    
    def test_get_minute_data_success(self, api):
        """测试成功获取分钟数据"""
        logger.info("测试获取分钟数据")
        df = api.get_minute_data('510300', period='5m', days=3)
        assert not df.empty
        assert 'date' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns
        assert 'symbol' in df.columns
        logger.info("分钟数据获取测试通过")
    
    def test_get_minute_data_empty(self, api):
        """测试获取空分钟数据"""
        logger.info("测试空分钟数据场景")
        df = api.get_minute_data('invalid_symbol', period='5m', days=3)
        assert df.empty
    
    def test_health_check(self, api):
        """测试健康检查"""
        logger.info("测试健康检查")
        status = api.health_check()
        
        # 验证健康检查返回的所有必需字段
        assert 'status' in status
        assert 'version' in status
        assert 'last_updated' in status
        assert 'data_sources' in status
        
        # 验证具体值
        assert status['status'] == 'OK'
        assert isinstance(status['data_sources'], list)
        assert 'sina' in status['data_sources']
        
        logger.info("健康检查测试通过")

if __name__ == "__main__":
    # 配置基础日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行测试
    pytest.main([__file__, "-v"])