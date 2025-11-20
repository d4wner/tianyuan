#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置文件检查与回测时间调整建议
功能：检查提供的YAML配置文件语法和逻辑错误，并给出拉长回测时间的建议
作者：缠论与量化交易专家（ISTJ）
日期：2025-11-03
"""

import yaml
import sys
from typing import Dict, Any, List

# 配置文件内容（从用户消息中提取）
ETFS_YAML = """
# config/etfs.yaml
# 宽基ETF配置 - 已修复量能单位不统一问题
broad:
  510300:
    name: 沪深300ETF
    type: broad
    black_swan_enabled: true
    volume_requirement: 1000000000  # 修复：统一为整数格式
    position_limit: 0.4             # 新增：单标仓位限制40%
    commission: 0.0003              # 新增：交易佣金0.03%
    stop_loss: 0.03                 # 新增：默认止损3%
    enabled: true                   # 新增：启用状态
    
  510500:
    name: 中证500ETF
    type: broad
    black_swan_enabled: true
    volume_requirement: 500000000   # 修复：统一为整数格式
    position_limit: 0.4
    commission: 0.0003
    stop_loss: 0.03
    enabled: true
    
  588000:
    name: 科创50ETF
    type: broad
    black_swan_enabled: true
    volume_requirement: 500000000    # 修复：统一为整数格式
    position_limit: 0.4
    commission: 0.0003
    stop_loss: 0.03
    enabled: true
    
  159915:
    name: 创业板ETF
    type: broad
    black_swan_enabled: true
    volume_requirement: 500000000    # 修复：统一为整数格式
    position_limit: 0.4
    commission: 0.0003
    stop_loss: 0.03
    enabled: true
    
  510050:
    name: 上证50ETF
    type: broad
    black_swan_enabled: true
    volume_requirement: 1000000000   # 修复：统一为整数格式
    position_limit: 0.4
    commission: 0.0003
    stop_loss: 0.03
    enabled: true

# 行业ETF配置 - 已调整黑天鹅规则
sector:
  512880:
    name: 证券ETF
    type: sector
    black_swan_enabled: true        # 修复：证券ETF波动率高，启用黑天鹅
    volume_requirement: 500000000   # 修复：统一为整数格式
    position_limit: 0.3             # 行业ETF仓位限制较低
    commission: 0.0003
    stop_loss: 0.04                 # 行业ETF止损放宽
    enabled: true
    
  512010:
    name: 医药ETF
    type: sector
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true
    
  512760:
    name: 芯片ETF
    type: sector
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true
    
  512800:
    name: 银行ETF
    type: sector
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true
    
  512660:
    name: 军工ETF
    type: sector
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true

# 主题ETF配置 - 已添加产品类型标识
theme:
  512690:
    name: 酒ETF
    type: theme
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true
    
  159928:
    name: 消费ETF
    type: theme
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true
    
  515030:
    name: 新能源车ETF
    type: theme
    black_swan_enabled: false
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true

# 港股ETF配置 - 已修复量能单位
hk:
  510900:
    name: H股ETF
    type: hk
    black_swan_enabled: true
    volume_requirement: 300000000    # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0004               # 港股交易佣金略高
    stop_loss: 0.035                 # 港股波动大，止损放宽
    enabled: true
    
  513550:
    name: 港股通50ETF
    type: hk
    black_swan_enabled: false
    volume_requirement: 200000000     # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0004
    stop_loss: 0.035
    enabled: true

# 美股ETF配置 - 已添加产品类型标识
us:
  513500:
    name: 标普500ETF
    type: us
    black_swan_enabled: true
    volume_requirement: 300000000     # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0004               # 美股交易佣金略高
    stop_loss: 0.035                 # 美股波动大，止损放宽
    enabled: true
    
  513100:
    name: 纳指ETF
    type: us
    black_swan_enabled: true
    volume_requirement: 300000000     # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0004
    stop_loss: 0.035
    enabled: true

# 债券ETF配置 - 已调整黑天鹅规则
bond:
  511010:
    name: 国债ETF
    type: bond
    black_swan_enabled: false        # 修复：债券ETF波动低，禁用黑天鹅
    volume_requirement: 200000000     # 修复：统一为整数格式
    position_limit: 0.2              # 债券ETF仓位限制较低
    commission: 0.0002               # 债券交易佣金较低
    stop_loss: 0.01                  # 债券止损较小
    enabled: true
    
  511260:
    name: 十年国债ETF
    type: bond
    black_swan_enabled: false        # 修复：债券ETF波动低，禁用黑天鹅
    volume_requirement: 200000000     # 修复：统一为整数格式
    position_limit: 0.2
    commission: 0.0002
    stop_loss: 0.01
    enabled: true

# 商品ETF配置 - 已添加产品类型标识
commodity:
  518880:
    name: 黄金ETF
    type: commodity
    black_swan_enabled: true
    volume_requirement: 500000000     # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04                  # 商品波动大，止损放宽
    enabled: true
    
  159937:
    name: 黄金基金
    type: commodity
    black_swan_enabled: true
    volume_requirement: 300000000     # 修复：统一为整数格式
    position_limit: 0.3
    commission: 0.0003
    stop_loss: 0.04
    enabled: true

# 全局配置
global:
  max_total_position: 0.7            # 总仓位上限70%
  max_single_position: 0.4           # 单标的上限40%
  stop_loss_default: 0.03            # 默认止损3%
  commission_default: 0.0003         # 默认佣金0.03%
  volume_thresholds:
    normal_breakthrough: 120000000   # 普通突破量能阈值
    strong_breakthrough: 150000000   # 强势突破量能阈值
    abnormal_volume: 180000000       # 异常量能阈值
    contraction_threshold: 70000000  # 量能萎缩阈值
"""

SYSTEM_YAML = """
# config/system.yaml
system:
  mode: monitor
  log_level: info
  auto_start: true
  data_retention_days: 90
  performance_monitor: true

# 策略配置 - 周线主导
strategy:
  active: 缠论周线主导策略
  version: 2025-10-16
  timeframe_priority:
    primary: weekly    # 核心判定级别：周线
    secondary: daily   # 辅助确认级别：日线
    execution: minute  # 执行级别：分钟线
  
  # 买点规则 - 周线主导
  buy_points:
    first_buy:
      position_range: [0.1, 0.15]
      stop_loss: 0.03
      confidence_level: high
      required_conditions:
        - weekly_bottom_divergence
        - daily_trend_confirmation
      execution_timing:
        - minute_oversold
        - volume_contraction<0.7
    
    second_buy:
      position_range: [0.4, 0.5]
      stop_loss: 0.02
      confidence_level: very_high
      required_conditions:
        - weekly_pullback_confirmation
        - daily_reversal_signal
      execution_timing:
        - minute_golden_cross
        - volume_expansion>1.5
    
    third_buy:
      position_range: [0.2, 0.25]
      stop_loss: 0.025
      confidence_level: medium
      required_conditions:
        - weekly_breakout_confirmation
        - daily_consolidation
      execution_timing:
        - minute_breakout
        - volume>1.2

# 回测配置
backtest:
  commission: 0.0003
  initial_capital: 100000
  slippage: 0.0001
  max_position_per_trade: 0.5
  risk_free_rate: 0.03
  timeframe: weekly  # 回测基于周线

# 缠论计算配置 - 周线参数优化
chanlun:
  # 周线参数
  weekly:
    central_bank_min_bars: 3
    fractal_sensitivity: 2
    pen_min_length: 3
    segment_min_length: 5
    divergence_confirmation: 2
  
  # 日线参数
  daily:
    central_bank_min_bars: 5
    fractal_sensitivity: 3
    pen_min_length: 5
    segment_min_length: 8
    divergence_confirmation: 3
  
  # 分钟线参数
  minute:
    central_bank_min_bars: 10
    fractal_sensitivity: 5
    pen_min_length: 10
    segment_min_length: 15
    divergence_confirmation: 5
  
  # 通用参数
  ranging_threshold: 0.015
  life_line_period: 20
  confirm_break_days: 2

# 数据源配置
data_source:
  name: sina
  retry_attempts: 3
  preferred_timeframes: [weekly, daily]
  fallback_sources: [tencent]
  timeout: 10
  max_retries: 3

# 数据获取器配置
data_fetcher:
  type_safety: true
  data_sources: [sina, tencent]
  sina:
    weekly_url: "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    daily_url: "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    minute_url: "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params:
      weekly:
        scale: "week"
        ma: "no"
        datalen: "200"
      daily:
        scale: "240"
        ma: "no"
        datalen: "100"
      minute:
        scale: "5"
        ma: "no"
        datalen: "1000"
  tencent:
    enabled: true
    fallback_only: true
    weekly_url: "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params:
      weekly:
        _var: "kline_week"
        param: "{symbol},week,,,320,qfq"
        r: "random"

# 钉钉通知配置
dingding:
  access_token: "1332588108dbdce65e6f5a8d196210517101ac53408e372fc1a6fa8944c20391"
  position_units: 50
  alert_levels:
    emergency: true
    warning: true
    info: false
  weekly_report: true

# 导出配置
export:
  format: csv
  output_dir: outputs/exports
  auto_clean_days: 30
  weekly_reports: true

# 监控配置
monitoring:
  primary_interval: 86400
  secondary_interval: 3600
  realtime_alert: true
  end_of_day_no_new_positions: true
  last_trading_hour: 14
  weekly_scan_day: 5
  monthly_scan_day: 28

# 绘图配置
plot:
  format: png
  output_dir: outputs/plots
  auto_save: true
  resolution: "1920x1080"
  preferred_timeframes: [weekly, daily, minute]

# 风险管理配置
risk_management:
  # 黑天鹅规则
  black_swan_rules:
    - weekly_bottom_divergence
    - volatility>5%
    - volume_abnormal_increase>150%
  
  # 仓位控制
  position_control:
    weekly_signal: 
      base_position: 0.4
      daily_adjustment: "±0.2"
    minute_execution:
      max_adjustment: 0.1
  
  # 紧急风控
  emergency_control: true
  trading_suspension_days: 7
  
  # 连续亏损控制
  consecutive_loss_control:
    enabled: true
    max_consecutive_losses: 3
    suspension_days: 7
    daily_loss_threshold: 0.02
  
  # 市场状态仓位限制
  market_conditions:
    trending: 0.7
    ranging: 0.3
    declining: 0.0
    black_swan: 0.4

# 交易时间配置
trading_hours:
  morning_start: "09:30"
  morning_end: "11:30"
  afternoon_start: "13:00"
  afternoon_end: "15:00"
  weekly_analysis_start: "20:00"
  weekly_analysis_end: "22:00"

# 实时监控配置
realtime_monitoring:
  enabled: true
  scan_interval: 10
  signal_expiry_minutes: 5
  min_volume_threshold: 1000000
  weekly_scan_enabled: true

# 自动化风控配置
auto_risk_control:
  enabled: true
  max_daily_loss: 0.05
  max_consecutive_loss: 3
  volatility_protection: true
  volume_anomaly_detection: true
  weekly_review: true

# 性能优化配置
performance:
  cache_enabled: true
  cache_expiry_hours: 24
  max_concurrent_requests: 5
  weekly_data_priority: true

# 备份与恢复配置
backup:
  auto_backup: true
  backup_interval_hours: 24
  max_backup_files: 7
  backup_dir: backups/data
  weekly_backup: true

# 数据存储配置
data_storage:
  positions_path: "data/signals/positions.json"
  daily_data_dir: "data/daily"
  weekly_data_dir: "data/weekly"
  minute_data_dir: "data/minute"
  signals_dir: "data/signals"
  backup_dir: "backups/data"
  weekly_reports_dir: "outputs/weekly_reports"

# 周线策略特殊配置
weekly_strategy:
  enabled: true
  scan_day: 5
  analysis_hours: [20, 22]
  min_weeks_data: 52
  divergence_confirmation: 2
  volume_analysis: true
  trend_confirmation: true
  timeframe_fallback: daily

# 数据获取重试配置
data_retry:
  max_retries: 3
  retry_delay: 1
  exponential_backoff: true
  timeout: 10

# 多时间级别配置框架
timeframe_config:
  enabled: true
  supported_timeframes: [weekly, daily, minute]
  default_timeframe: weekly
  fallback_priority: [weekly, daily, minute]
  sync_across_timeframes: true
"""

RISK_RULES_YAML = """
# config/risk_rules.yaml
# 黑天鹅事件规则 - 已优化为周线级别
black_swan_rules:
  - weekly_bottom_divergence_confirmed  # 周线底背驰确认
  - weekly_volatility > 5%              # 周线波动率>5%
  - weekly_volume_abnormal_increase > 150%  # 周线量能异常放大>150%
  - daily_confirmation_required         # 需要日线确认

# 三类买点仓位分配规则 - 已调整为周线主导
buy_point_rules:
  first_buy:
    position_size: [0.1, 0.15]  # 一买仓位10%-15%
    stop_loss: 0.03
    confidence_level: high
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_bottom_divergence_confirmed  # 周线底背驰确认
      - weekly_MACD_golden_cross           # 周线MACD金叉
      - weekly_break_trend_line             # 周线突破趋势线
      - weekly_volume_contraction < 70%     # 周线量能萎缩至70%以下
      - daily_confirmation                  # 日线确认信号
      
  second_buy:
    position_size: [0.4, 0.5]   # 二买仓位40%-50%
    stop_loss: 0.02
    confidence_level: very_high
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_pullback_above_prior_low     # 周线回踩不破前低
      - weekly_volume_contraction_then_expansion  # 周线量能先缩后放
      - weekly_volume > 150%                  # 周线量能放大150%以上
      - daily_confirmation                    # 日线确认信号
      
  third_buy:
    position_size: [0.2, 0.25]  # 三买仓位20%-25%
    stop_loss: 0.025
    confidence_level: medium
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_pullback_above_central_bank_upper  # 周线回踩不破中枢上沿
      - weekly_volume > 120%                       # 周线量能放大120%以上
      - daily_confirmation                         # 日线确认信号

# 三类卖点仓位分配规则 - 已调整为周线主导
sell_point_rules:
  first_sell:
    position_adjust: -0.5  # 减仓50%
    stop_loss: 0.02
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_top_divergence_confirmed        # 周线顶背驰确认
      - weekly_MACD_dead_cross                  # 周线MACD死叉
      - weekly_break_trend_line_down            # 周线跌破趋势线
      - weekly_volume_abnormal_increase > 150%  # 周线量能异常放大150%
      - daily_confirmation                     # 日线确认信号
      
  second_sell:
    position_adjust: -0.3  # 减仓30%
    stop_loss: 0.015
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_rebound_below_prior_high  # 周线反弹不过前高
      - weekly_volume_contraction        # 周线量能萎缩
      - weekly_daily_drop > 0.03         # 周内单日跌幅>3%
      - daily_confirmation               # 日线确认信号
      
  third_sell:
    position_adjust: -0.2  # 清仓剩余20%
    stop_loss: 0.01
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_break_life_line            # 周线跌破生命线
      - weekly_consecutive_break_days >= 2  # 周线连续2日跌破
      - weekly_volume > 130%              # 周线量能放大130%
      - daily_confirmation                # 日线确认信号

# 紧急风控规则（包含连续亏损暂停机制）- 已优化为周线级别
emergency_rules:
  immediate_sell_all:
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_black_swan_event_detected  # 周线黑天鹅事件检测
      - weekly_market_crash > 0.05         # 周线市场暴跌5%
      - weekly_volume > 200%               # 周线量能放大200%
    action: sell_all_positions
    
  trading_suspension:
    timeframe: weekly  # 周线级别信号
    conditions:
      - weekly_consecutive_loss_weeks >= 2  # 连续2周亏损
    action: suspend_trading_7_days  # 暂停交易7天

# 连续亏损暂停机制（新增独立规则）- 已优化为周线级别
consecutive_loss_control:
  enabled: true
  max_consecutive_losses: 3
  suspension_days: 7
  timeframe: weekly  # 周线级别信号
  conditions:
    - weekly_loss > 0.02               # 单周亏损>2%
    - weekly_consecutive_loss_weeks >= 2  # 连续2周亏损
  actions:
    - suspend_trading
    - reduce_position_limit 0.5  # 仓位限制降至50%
    - require_manual_review

# 生命线跌破标准（更新版）- 已优化为周线级别
life_line_rules:
  ma_periods: [20, 60]  # 20日和60日均线
  timeframe: weekly      # 周线级别信号
  break_definition: weekly_close_below_ma  # 周线收盘价低于均线
  consecutive_weeks: 2  # 连续2周跌破
  confirmation_required: true
  volume_threshold: 130%  # 周线量能放大130%以上
  actions:
    - trigger_third_sell
    - reduce_exposure 0.5  # 降低风险暴露50%
    - enable_extra_monitoring

# 量能触发标准 - 已优化为周线级别
volume_thresholds:
  normal_breakthrough: 120%  # 周线普通突破所需量能
  strong_breakthrough: 150%  # 周线强势突破所需量能
  abnormal_volume: 180%      # 周线异常量能阈值
  contraction_threshold: 70% # 周线量能萎缩阈值

# 时间相关规则 - 已优化为周线主导
time_rules:
  weekly_analysis_day: 5      # 周线分析日：周五
  weekly_analysis_hours: [20, 22]  # 周线分析时间：20:00-22:00
  avoid_end_of_week: true     # 避免周末交易
  last_day_no_new_positions: true  # 周五不开新仓
  morning_session_preferred: true   # 优先上午时段交易

# 周线背驰判定规则 - 新增专项规则
divergence_rules:
  # 底背驰判定
  bottom_divergence:
    required_timeframes: [weekly]  # 必须满足周线底背驰
    confirmations: [daily]         # 日线确认
    volume_threshold: 1.5          # 量能放大150%
    conditions:
      - price_new_low
      - indicator_higher_low
      - volume_expansion
      - break_downtrend_line
  
  # 顶背驰判定
  top_divergence:
    required_timeframes: [weekly]  # 必须满足周线顶背驰
    confirmations: [daily]         # 日线确认
    volume_threshold: 1.5          # 量能放大150%
    conditions:
      - price_new_high
      - indicator_lower_high
      - volume_divergence
      - break_uptrend_line
  
  # 背驰强度分级
  strength_levels:
    weak:
      conditions: [price_divergence]
      action: monitor_only
    medium:
      conditions: [price_divergence, volume_divergence]
      action: reduce_position 0.3
    strong:
      conditions: [price_divergence, volume_divergence, indicator_divergence]
      action: reduce_position 0.5

# 周线仓位控制调整 - 新增专项规则
position_control:
  weekly_signal: 
    base_position: 0.4  # 周线信号基础仓位40%
    daily_adjustment: ±0.2  # 日线微调幅度±20%
  minute_execution:
    max_adjustment: 0.1  # 分钟级最大调整10%
  timeframe_weights:
    weekly: 0.6   # 周线权重60%
    daily: 0.3    # 日线权重30%
    minute: 0.1   # 分钟线权重10%

# 周线主导工作流 - 新增专项规则
weekly_workflow:
  step_1: weekly_scan  # 周线扫描
  step_2: divergence_detection  # 背驰检测
  step_3: daily_confirmation  # 日线确认
  step_4: minute_execution  # 分钟执行
  step_5: position_adjustment  # 仓位调整
  timeframe_priority: [weekly, daily, minute]  # 时间级别优先级

# 信号生成规则 - 新增周线专项规则
signal_generation:
  buy:
    required_conditions:
      - weekly_bottom_divergence
      - daily_trend_confirmation
    execution_timing:
      - minute_oversold
      - volume_contraction<0.7
    timeframe: weekly  # 周线主导
  
  sell:
    required_conditions:
      - weekly_top_divergence
      - daily_trend_confirmation
    execution_timing:
      - minute_overbought
      - volume_expansion>1.5
    timeframe: weekly  # 周线主导

# 市场状态仓位限制 - 已优化为周线级别
market_conditions:
  trending: 
    condition: weekly_trending_up
    position_limit: 0.7
  ranging:
    condition: weekly_ranging
    position_limit: 0.3
  declining:
    condition: weekly_trending_down
    position_limit: 0.0
  black_swan:
    condition: weekly_black_swan_detected
    position_limit: 0.4
"""

def parse_yaml(yaml_str: str, config_name: str) -> Dict[str, Any]:
    """
    解析YAML字符串并返回字典，捕获语法错误。
    
    Args:
        yaml_str (str): YAML格式的字符串
        config_name (str): 配置文件名，用于错误提示
    
    Returns:
        Dict[str, Any]: 解析后的字典
    
    Raises:
        yaml.YAMLError: 如果YAML语法错误
    """
    try:
        data = yaml.safe_load(yaml_str)
        return data
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"配置文件 {config_name} 语法错误: {e}")

def check_etfs_config(data: Dict[str, Any]) -> List[str]:
    """
    检查ETF配置文件逻辑错误。
    
    Args:
        data (Dict[str, Any]): 解析后的ETF配置数据
    
    Returns:
        List[str]: 错误消息列表，空列表表示无错误
    """
    errors = []
    
    # 检查全局配置
    global_config = data.get('global', {})
    if not global_config:
        errors.append("缺失全局配置 'global' 部分")
    else:
        max_total = global_config.get('max_total_position')
        if max_total is None or max_total <= 0 or max_total > 1:
            errors.append("全局配置 max_total_position 必须为 (0,1] 之间的浮点数")
        
        max_single = global_config.get('max_single_position')
        if max_single is None or max_single <= 0 or max_single > 1:
            errors.append("全局配置 max_single_position 必须为 (0,1] 之间的浮点数")
        
        if max_total and max_single and max_single > max_total:
            errors.append("全局配置 max_single_position 不能大于 max_total_position")
    
    # 检查每个ETF配置
    etf_categories = ['broad', 'sector', 'theme', 'hk', 'us', 'bond', 'commodity']
    for category in etf_categories:
        etfs = data.get(category, {})
        for code, config in etfs.items():
            # 检查必要字段
            required_fields = ['name', 'type', 'black_swan_enabled', 'volume_requirement', 
                              'position_limit', 'commission', 'stop_loss', 'enabled']
            for field in required_fields:
                if field not in config:
                    errors.append(f"ETF {code} 缺失字段 '{field}'")
            
            # 检查数值范围
            pos_limit = config.get('position_limit')
            if pos_limit and (pos_limit <= 0 or pos_limit > 1):
                errors.append(f"ETF {code} position_limit 必须为 (0,1] 之间的浮点数")
            
            commission = config.get('commission')
            if commission and commission < 0:
                errors.append(f"ETF {code} commission 不能为负数")
            
            stop_loss = config.get('stop_loss')
            if stop_loss and stop_loss <= 0:
                errors.append(f"ETF {code} stop_loss 必须为正数")
            
            volume_req = config.get('volume_requirement')
            if volume_req and volume_req <= 0:
                errors.append(f"ETF {code} volume_requirement 必须为正整数")
    
    return errors

def check_system_config(data: Dict[str, Any]) -> List[str]:
    """
    检查系统配置文件逻辑错误。
    
    Args:
        data (Dict[str, Any]): 解析后的系统配置数据
    
    Returns:
        List[str]: 错误消息列表，空列表表示无错误
    """
    errors = []
    
    # 检查回测配置
    backtest = data.get('backtest', {})
    if backtest:
        capital = backtest.get('initial_capital')
        if capital and capital <= 0:
            errors.append("回测配置 initial_capital 必须为正数")
        
        commission = backtest.get('commission')
        if commission and commission < 0:
            errors.append("回测配置 commission 不能为负数")
        
        slippage = backtest.get('slippage')
        if slippage and slippage < 0:
            errors.append("回测配置 slippage 不能为负数")
    
    # 检查数据获取配置
    data_fetcher = data.get('data_fetcher', {})
    if data_fetcher:
        sina_params = data_fetcher.get('sina', {}).get('params', {})
        for timeframe in ['weekly', 'daily', 'minute']:
            params = sina_params.get(timeframe, {})
            datalen = params.get('datalen')
            if datalen and not datalen.isdigit():
                errors.append(f"数据获取配置 {timeframe} 的 datalen 必须为数字字符串")
    
    return errors

def check_risk_rules_config(data: Dict[str, Any]) -> List[str]:
    """
    检查风险规则配置文件逻辑错误。
    
    Args:
        data (Dict[str, Any]): 解析后的风险规则配置数据
    
    Returns:
        List[str]: 错误消息列表，空列表表示无错误
    """
    errors = []
    
    # 检查买点规则
    buy_rules = data.get('buy_point_rules', {})
    for buy_point, rules in buy_rules.items():
        pos_size = rules.get('position_size')
        if pos_size and (not isinstance(pos_size, list) or len(pos_size) != 2):
            errors.append(f"买点规则 {buy_point} position_size 必须为长度为2的列表")
        elif pos_size:
            min_pos, max_pos = pos_size
            if min_pos <= 0 or max_pos > 1 or min_pos > max_pos:
                errors.append(f"买点规则 {buy_point} position_size 范围无效")
        
        stop_loss = rules.get('stop_loss')
        if stop_loss and stop_loss <= 0:
            errors.append(f"买点规则 {buy_point} stop_loss 必须为正数")
    
    # 检查卖点规则
    sell_rules = data.get('sell_point_rules', {})
    for sell_point, rules in sell_rules.items():
        pos_adjust = rules.get('position_adjust')
        if pos_adjust and (pos_adjust > 0 or pos_adjust < -1):
            errors.append(f"卖点规则 {sell_point} position_adjust 必须为 [-1,0] 之间的浮点数")
        
        stop_loss = rules.get('stop_loss')
        if stop_loss and stop_loss <= 0:
            errors.append(f"卖点规则 {sell_point} stop_loss 必须为正数")
    
    return errors

def suggest_backtest_extension(system_data: Dict[str, Any]) -> str:
    """
    给出拉长回测时间的建议。
    
    Args:
        system_data (Dict[str, Any]): 解析后的系统配置数据
    
    Returns:
        str: 建议文本
    """
    suggestion = "拉长回测时间建议：\n"
    
    # 检查数据获取配置中的 datalen 参数
    data_fetcher = system_data.get('data_fetcher', {})
    sina_params = data_fetcher.get('sina', {}).get('params', {})
    
    for timeframe in ['weekly', 'daily', 'minute']:
        params = sina_params.get(timeframe, {})
        datalen = params.get('datalen', '未知')
        current_datalen = int(datalen) if datalen.isdigit() else 0
        
        suggestion += f"- {timeframe} 级别当前 datalen: {datalen} (约 {current_datalen} 条数据)\n"
        if timeframe == 'weekly':
            suggested_datalen = 500  # 建议增加到500周（约10年）
            suggestion += f"  建议修改为 {suggested_datalen} 以获取约10年数据\n"
        elif timeframe == 'daily':
            suggested_datalen = 1000  # 建议增加到1000天（约4年）
            suggestion += f"  建议修改为 {suggested_datalen} 以获取约4年数据\n"
        elif timeframe == 'minute':
            suggested_datalen = 10000  # 建议增加到10000分钟（约数周）
            suggestion += f"  建议修改为 {suggested_datalen} 以获取更长时间数据\n"
    
    suggestion += "\n修改方法：在 config/system.yaml 的 data_fetcher.sina.params 中调整 datalen 值。\n"
    suggestion += "注意：数据源可能限制最大获取长度，需根据实际调整。"
    
    return suggestion

def main():
    """
    主函数：检查所有配置文件并输出结果。
    """
    print("=" * 60)
    print("配置文件检查报告")
    print("=" * 60)
    
    all_errors = []
    configs = [
        ("config/etfs.yaml", ETFS_YAML, check_etfs_config),
        ("config/system.yaml", SYSTEM_YAML, check_system_config),
        ("config/risk_rules.yaml", RISK_RULES_YAML, check_risk_rules_config)
    ]
    
    parsed_data = {}
    
    for config_name, yaml_str, check_func in configs:
        print(f"\n检查配置文件: {config_name}")
        try:
            data = parse_yaml(yaml_str, config_name)
            parsed_data[config_name] = data
            errors = check_func(data)
            if errors:
                all_errors.extend([f"{config_name}: {error}" for error in errors])
                print(f"  ❌ 发现 {len(errors)} 个错误:")
                for error in errors:
                    print(f"     - {error}")
            else:
                print("  ✅ 无语法和逻辑错误")
        except yaml.YAMLError as e:
            all_errors.append(f"{config_name}: {e}")
            print(f"  ❌ YAML语法错误: {e}")
        except Exception as e:
            all_errors.append(f"{config_name}: 检查过程中出错: {e}")
            print(f"  ❌ 检查错误: {e}")
    
    print("\n" + "=" * 60)
    if all_errors:
        print("❌ 配置文件检查失败，发现以下错误:")
        for error in all_errors:
            print(f"   - {error}")
    else:
        print("✅ 所有配置文件检查通过，无错误。")
        
        # 提供回测时间调整建议
        system_data = parsed_data.get("config/system.yaml", {})
        suggestion = suggest_backtest_extension(system_data)
        print("\n" + "=" * 60)
        print(suggestion)
    
    print("=" * 60)

if __name__ == "__main__":
    main()