#!/usr/bin/env python3
"""
缠论系统配置文件修复工具
修复YAML配置文件中的语法错误
"""

import yaml
import os
import re
from pathlib import Path

def fix_etfs_config():
    """修复ETF配置文件"""
    etfs_content = """
bond:
  511010:
    black_swan_enabled: false
    commission: 0.0002
    enabled: true
    market: sh
    name: 国债ETF
    position_limit: 0.2
    stop_loss: 0.01
    type: bond
    volume_requirement: 200000000
  511260:
    black_swan_enabled: false
    commission: 0.0002
    enabled: true
    market: sh
    name: 十年国债ETF
    position_limit: 0.2
    stop_loss: 0.01
    type: bond
    volume_requirement: 200000000

broad:
  159915:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sz
    name: 创业板ETF
    position_limit: 0.4
    stop_loss: 0.03
    type: broad
    volume_requirement: 500000000
  510050:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sh
    name: 上证50ETF
    position_limit: 0.4
    stop_loss: 0.03
    type: broad
    volume_requirement: 1000000000
  510300:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sh
    name: 沪深300ETF
    position_limit: 0.4
    stop_loss: 0.03
    type: broad
    volume_requirement: 1000000000
  510500:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sh
    name: 中证500ETF
    position_limit: 0.4
    stop_loss: 0.03
    type: broad
    volume_requirement: 500000000
  588000:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sh
    name: 科创50ETF
    position_limit: 0.4
    stop_loss: 0.03
    type: broad
    volume_requirement: 500000000

commodity:
  159937:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sz
    name: 黄金基金
    position_limit: 0.3
    stop_loss: 0.04
    type: commodity
    volume_requirement: 300000000
  518880:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sh
    name: 黄金ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: commodity
    volume_requirement: 500000000

global:
  commission_default: 0.0003
  max_single_position: 0.4
  max_total_position: 0.7
  stop_loss_default: 0.03
  volume_thresholds:
    absolute:
      abnormal_volume: 180000000
      contraction_threshold: 70000000
      normal_breakthrough: 120000000
      strong_breakthrough: 150000000
    percentage:
      abnormal_volume: 180%
      contraction_threshold: 70%
      normal_breakthrough: 120%
      strong_breakthrough: 150%
    note: 绝对值为具体数值，百分比为相对变化率

hk:
  510900:
    black_swan_enabled: true
    commission: 0.0004
    enabled: true
    market: sh
    name: H股ETF
    position_limit: 0.3
    stop_loss: 0.035
    type: hk
    volume_requirement: 300000000
  513550:
    black_swan_enabled: false
    commission: 0.0004
    enabled: true
    market: sh
    name: 港股通50ETF
    position_limit: 0.3
    stop_loss: 0.035
    type: hk
    volume_requirement: 200000000

sector:
  512010:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sh
    name: 医药ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: sector
    volume_requirement: 300000000
  512660:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sh
    name: 军工ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: sector
    volume_requirement: 300000000
  512760:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sh
    name: 芯片ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: sector
    volume_requirement: 300000000
  512800:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sh
    name: 银行ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: sector
    volume_requirement: 300000000
  512880:
    black_swan_enabled: true
    commission: 0.0003
    enabled: true
    market: sh
    name: 证券ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: sector
    volume_requirement: 500000000

theme:
  159928:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sz
    name: 消费ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: theme
    volume_requirement: 300000000
  512690:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sh
    name: 酒ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: theme
    volume_requirement: 300000000
  515030:
    black_swan_enabled: false
    commission: 0.0003
    enabled: true
    market: sh
    name: 新能源车ETF
    position_limit: 0.3
    stop_loss: 0.04
    type: theme
    volume_requirement: 300000000

us:
  513100:
    black_swan_enabled: true
    commission: 0.0004
    enabled: true
    market: sh
    name: 纳指ETF
    position_limit: 0.3
    stop_loss: 0.035
    type: us
    volume_requirement: 300000000
  513500:
    black_swan_enabled: true
    commission: 0.0004
    enabled: true
    market: sh
    name: 标普500ETF
    position_limit: 0.3
    stop_loss: 0.035
    type: us
    volume_requirement: 300000000
"""
    
    # 修复缩进问题
    etfs_content = re.sub(r'^ {2}position_limit', '    position_limit', etfs_content, flags=re.MULTILINE)
    etfs_content = re.sub(r'^ {2}stop_loss', '    stop_loss', etfs_content, flags=re.MULTILINE)
    etfs_content = re.sub(r'^ {2}type', '    type', etfs_content, flags=re.MULTILINE)
    etfs_content = re.sub(r'^ {2}volume_requirement', '    volume_requirement', etfs_content, flags=re.MULTILINE)
    
    return etfs_content

def fix_system_config():
    """修复系统配置文件"""
    system_content = """
# 缠论系统配置文件 - 完整修复版
# 确保所有配置项完整且一致

# 计算器配置
calculator:
  weekly_pen_min_length: 3
  weekly_central_min_length: 3
  daily_pen_min_length: 5
  daily_central_min_length: 5
  minute_pen_min_length: 10
  minute_central_min_length: 10

# 风险管理配置
risk_management:
  black_swan_rules:
  - weekly_bottom_divergence_confirmed
  - weekly_volatility > 5%
  - weekly_volume_abnormal_increase > 150%
  - daily_confirmation_required
  volume_thresholds:
    absolute:
      abnormal_volume: 180000000
      contraction_threshold: 70000000
      normal_breakthrough: 120000000
      strong_breakthrough: 150000000
    percentage:
      abnormal_volume: 180%
      contraction_threshold: 70%
      normal_breakthrough: 120%
      strong_breakthrough: 150%
  stop_loss_settings:
    stop_loss_type: "dynamic"
    stop_loss_atr_period: 14
    stop_loss_atr_multiplier: 2.0
    stop_loss_default: 0.03
    trailing_stop_enabled: true
    trailing_stop_activation: 0.05
    trailing_stop_distance: 0.02
    emergency_stop_loss: 0.10

# 策略映射配置
strategy_mapping:
  daily: config/strategy_daily.yaml
  weekly: config/strategy_weekly.yaml
  minute: config/strategy_minute.yaml

# 策略配置
strategies:
  daily:
    buy_points:
      first_buy:
        confidence_level: high
        execution_timing:
        - minute_oversold
        - volume_contraction<0.7
        position_range:
        - 0.1
        - 0.15
        required_conditions:
        - weekly_bottom_divergence
        - daily_trend_confirmation
        stop_loss: 0.03
      second_buy:
        confidence_level: very_high
        execution_timing:
        - minute_golden_cross
        - volume_expansion>1.5
        position_range:
        - 0.4
        - 0.5
        required_conditions:
        - weekly_pullback_confirmation
        - daily_reversal_signal
        stop_loss: 0.02
      third_buy:
        confidence_level: medium
        execution_timing:
        - minute_breakout
        - volume>1.2
        position_range:
        - 0.2
        - 0.25
        required_conditions:
        - weekly_breakout_confirmation
        - daily_consolidation
        stop_loss: 0.025
    timeframe_priority:
      execution: minute
      primary: weekly
      secondary: daily
    version: 2025-10-16
  weekly:
    buy_points:
      first_buy:
        confidence_level: high
        execution_timing:
        - weekly_pullback
        - volume_contraction<0.7
        position_range:
        - 0.1
        - 0.15
        required_conditions:
        - monthly_trend_confirmation
        - weekly_divergence
        stop_loss: 0.03
      second_buy:
        confidence_level: very_high
        execution_timing:
        - weekly_golden_cross
        - volume_expansion>1.5
        position_range:
        - 0.4
        - 0.5
        required_conditions:
        - weekly_pullback_confirmation
        - daily_reversal_signal
        stop_loss: 0.02
      third_buy:
        confidence_level: medium
        execution_timing:
        - weekly_breakout
        - volume>1.2
        position_range:
        - 0.2
        - 0.25
        required_conditions:
        - weekly_breakout_confirmation
        - daily_consolidation
        stop_loss: 0.025
    timeframe_priority:
      execution: daily
      primary: weekly
      secondary: monthly
    version: 2025-10-16
  minute:
    buy_points:
      first_buy:
        confidence_level: medium
        execution_timing:
        - minute_breakout
        - volume_expansion>1.2
        position_range:
        - 0.05
        - 0.1
        required_conditions:
        - daily_trend_confirmation
        stop_loss: 0.02
      second_buy:
        confidence_level: high
        execution_timing:
        - minute_golden_cross
        - volume_expansion>1.5
        position_range:
        - 0.2
        - 0.3
        required_conditions:
        - daily_confirmation
        stop_loss: 0.015
    timeframe_priority:
      execution: minute
      primary: daily
      secondary: weekly
    version: 2025-10-16

# 系统核心配置
system:
  # 数据获取器配置
  data_fetcher:
    type_safety: true
    data_sources:
    - sina
    - tencent
    sina:
      symbol_format: with_prefix
      base_url: http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
      params:
        daily:
          scale: "240"
          ma: "no"
          datalen: "1000"
        weekly:
          scale: "week"
          ma: "no"
          datalen: "500"
        minute:
          scale: "5"
          ma: "no"
          datalen: "10000"
    tencent:
      symbol_format: with_prefix
      enabled: true
      weekly_url: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
      params:
        weekly:
          _var: "kline_week"
          param: "{symbol},week,,,320,qfq"
  
  # 自动风险控制
  auto_risk_control:
    enabled: true
    max_consecutive_loss: 3
    max_daily_loss: 0.05
    volatility_protection: true
    volume_anomaly_detection: true
    weekly_review: true
  
  # 回测配置
  backtest:
    initial_capital: 100000
    commission: 0.0003
    slippage: 0.0001
    risk_free_rate: 0.03
    max_position_per_trade: 0.5
    min_data_days: 90
    start_date: "2024-01-01"
    end_date: "2025-11-06"
    timeframe: weekly
  
  # 备份配置
  backup:
    auto_backup: true
    backup_dir: backups/data
    backup_interval_hours: 24
    max_backup_files: 7
    weekly_backup: true
  
  # 缠论参数配置
  chanlun:
    weekly:
      pen_min_length: 5
      segment_min_length: 5
      central_bank_min_bars: 3
      fractal_sensitivity: 3
      divergence_confirmation: 2
    daily:
      pen_min_length: 5
      segment_min_length: 8
      central_bank_min_bars: 5
      fractal_sensitivity: 3
      divergence_confirmation: 3
    minute:
      pen_min_length: 10
      segment_min_length: 15
      central_bank_min_bars: 10
      fractal_sensitivity: 5
      divergence_confirmation: 5
    life_line_period: 20
    ranging_threshold: 0.015
    confirm_break_days: 2
  
  # 数据保留天数
  data_retention_days: 90
  
  # 数据重试配置
  data_retry:
    max_retries: 3
    retry_delay: 1
    timeout: 10
    exponential_backoff: true
  
  # 数据源配置
  data_source:
    name: sina
    preferred_timeframes:
    - weekly
    - daily
    fallback_sources:
    - tencent
    max_retries: 3
    timeout: 10
    retry_attempts: 3
  
  # 数据存储配置
  data_storage:
    daily_data_dir: data/daily
    weekly_data_dir: data/weekly
    minute_data_dir: data/minute
    signals_dir: data/signals
    positions_path: data/signals/positions.json
    backup_dir: backups/data
    weekly_reports_dir: outputs/weekly_reports
  
  # 钉钉通知配置
  dingding:
    access_token: "1332588108dbdce65e6f5a8d196210517101ac53408e372fc1a6fa8944c20391"
    alert_levels:
      emergency: true
      warning: true
      info: false
    position_units: 50
    weekly_report: true
  
  # 数据导出配置
  export:
    output_dir: outputs/exports
    format: csv
    weekly_reports: true
    auto_clean_days: 30
  
  # 日志级别
  log_level: info
  
  # 运行模式
  mode: monitor
  
  # 监控配置
  monitoring:
    primary_interval: 86400
    secondary_interval: 3600
    realtime_alert: true
    end_of_day_no_new_positions: true
    last_trading_hour: 14
    weekly_scan_day: 5
    monthly_scan_day: 28
  
  # 性能配置
  performance:
    cache_enabled: true
    cache_expiry_hours: 24
    max_concurrent_requests: 5
    weekly_data_priority: true
  
  # 性能监控
  performance_monitor: true
  
  # 绘图配置
  plot:
    auto_save: true
    format: png
    resolution: 1920x1080
    output_dir: outputs/plots
    preferred_timeframes:
    - weekly
    - daily
    - minute
  
  # 实时监控配置
  realtime_monitoring:
    enabled: true
    scan_interval: 10
    signal_expiry_minutes: 5
    min_volume_threshold: 1000000
    weekly_scan_enabled: true
  
  # 风险管理配置
  risk_management:
    black_swan_rules:
    - weekly_bottom_divergence
    - volatility>5%
    - volume_abnormal_increase>150%
    consecutive_loss_control:
      enabled: true
      max_consecutive_losses: 3
      daily_loss_threshold: 0.02
      suspension_days: 7
    emergency_control: true
    market_conditions:
      black_swan: 0.4
      declining: 0.0
      ranging: 0.3
      trending: 0.7
    position_control:
      weekly_signal:
        base_position: 0.4
        daily_adjustment: ±0.2
      minute_execution:
        max_adjustment: 0.1
    trading_suspension_days: 7
    stop_loss_settings:
      stop_loss_type: "dynamic"
      stop_loss_atr_period: 14
      stop_loss_atr_multiplier: 2.0
      stop_loss_default: 0.03
      trailing_stop_enabled: true
      trailing_stop_activation: 0.05
      trailing_stop_distance: 0.02
      emergency_stop_loss: 0.10
  
  # 策略配置
  strategy:
    active: "缠论周线主导策略"
    buy_points:
      first_buy:
        confidence_level: high
        execution_timing:
        - minute_oversold
        - volume_contraction<0.7
        position_range:
        - 0.1
        - 0.15
        required_conditions:
        - weekly_bottom_divergence
        - daily_trend_confirmation
        stop_loss: 0.03
      second_buy:
        confidence_level: very_high
        execution_timing:
        - minute_golden_cross
        - volume_expansion>1.5
        position_range:
        - 0.4
        - 0.5
        required_conditions:
        - weekly_pullback_confirmation
        - daily_reversal_signal
        stop_loss: 0.02
      third_buy:
        confidence_level: medium
        execution_timing:
        - minute_breakout
        - volume>1.2
        position_range:
        - 0.2
        - 0.25
        required_conditions:
        - weekly_breakout_confirmation
        - daily_consolidation
        stop_loss: 0.025
    timeframe_priority:
      execution: minute
      primary: weekly
      secondary: daily
    version: "2025-10-16"
  
  # 时间级别配置
  timeframe_config:
    default_timeframe: weekly
    supported_timeframes:
    - weekly
    - daily
    - minute
    fallback_priority:
    - weekly
    - daily
    - minute
    sync_across_timeframes: true
    enabled: true
  
  # 交易时间配置
  trading_hours:
    morning_start: "09:30"
    morning_end: "11:30"
    afternoon_start: "13:00"
    afternoon_end: "15:00"
    weekly_analysis_start: "20:00"
    weekly_analysis_end: "22:00"
  
  # 周策略配置
  weekly_strategy:
    enabled: true
    min_weeks_data: 52
    scan_day: 5
    analysis_hours:
    - 20
    - 22
    divergence_confirmation: 2
    timeframe_fallback: daily
    trend_confirmation: true
    volume_analysis: true

# 自动启动配置
auto_start: true
"""
    
    # 修复乱码和语法错误
    system_content = re.sub(r'stop_loss: 极0\.03', 'stop_loss: 0.03', system_content)
    system_content = re.sub(r'position极_range', 'position_range', system_content)
    system_content = re.sub(r'required_极conditions', 'required_conditions', system_content)
    system_content = re.sub(r'极       - daily_trend_confirmation', '        - daily_trend_confirmation', system_content)
    
    return system_content

def validate_yaml(content, filename):
    """验证YAML语法"""
    try:
        yaml.safe_load(content)
        print(f"✓ {filename} YAML语法验证通过")
        return True
    except yaml.YAMLError as e:
        print(f"✗ {filename} YAML语法错误: {e}")
        return False

def save_config_file(content, filepath):
    """保存配置文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 配置文件已保存: {filepath}")
        return True
    except Exception as e:
        print(f"✗ 保存文件失败 {filepath}: {e}")
        return False

def main():
    """主函数"""
    print("开始修复缠论系统配置文件...")
    print("=" * 50)
    
    # 修复ETF配置文件
    print("修复ETF配置文件...")
    etfs_content = fix_etfs_config()
    
    # 修复系统配置文件
    print("修复系统配置文件...")
    system_content = fix_system_config()
    
    # 验证YAML语法
    print("\n验证配置文件语法...")
    etfs_valid = validate_yaml(etfs_content, "etfs.yaml")
    system_valid = validate_yaml(system_content, "system.yaml")
    
    if not etfs_valid or not system_valid:
        print("配置文件存在语法错误，请手动检查修复")
        return
    
    # 保存配置文件
    print("\n保存配置文件...")
    config_dir = "config"
    
    etfs_saved = save_config_file(etfs_content, os.path.join(config_dir, "etfs.yaml"))
    system_saved = save_config_file(system_content, os.path.join(config_dir, "system.yaml"))
    
    if etfs_saved and system_saved:
        print("\n" + "=" * 50)
        print("✓ 所有配置文件修复完成并保存成功！")
        print("配置文件路径:")
        print(f"  - {os.path.join(config_dir, 'etfs.yaml')}")
        print(f"  - {os.path.join(config_dir, 'system.yaml')}")
    else:
        print("\n✗ 配置文件保存失败，请检查目录权限")

if __name__ == "__main__":
    main()