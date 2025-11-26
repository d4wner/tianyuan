import json
import datetime
import os

def find_november_buys(file_path, target_days=[24, 25]):
    print(f"分析文件: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            signals = json.load(f)
        
        print(f"文件包含 {len(signals)} 个信号")
        
        # 查找指定日期的买入信号
        found_buys = []
        
        for i, signal in enumerate(signals):
            try:
                # 转换时间戳
                timestamp = signal['date'] / 1000
                dt = datetime.datetime.fromtimestamp(timestamp)
                
                # 检查是否是2025年11月指定日期的买入信号
                if (dt.year == 2025 and dt.month == 11 and dt.day in target_days and 
                    signal.get('type') == 'buy'):
                    
                    found_buys.append({
                        'index': i,
                        'datetime': dt,
                        'price': signal.get('price'),
                        'strength': signal.get('strength'),
                        'reason': signal.get('reason')
                    })
            except Exception as e:
                print(f"处理信号 #{i+1} 时出错: {e}")
        
        # 显示找到的买点信号
        if found_buys:
            print(f"\n找到 {len(found_buys)} 个2025年11月{'-'.join(map(str, target_days))}日的买点信号:")
            for buy in found_buys:
                print(f"\n买点 #{buy['index']+1}:")
                print(f"  时间: {buy['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  价格: {buy['price']}")
                print(f"  强度: {buy['strength']}")
                print(f"  原因: {buy['reason']}")
                
                # 根据信号原因和强度推断级别
                # 基于monitor.py中的多级别联立分析逻辑
                if buy['strength'] > 0.5:
                    print("  可能的买点级别: 高可信度信号，可能是15分钟+5分钟复合级别")
                else:
                    print("  可能的买点级别: 标准信号，可能是5分钟或日线级别")
        else:
            print(f"\n未找到2025年11月{'-'.join(map(str, target_days))}日的买点信号")
            
            # 显示文件中所有2025年11月的信号
            print("\n2025年11月的所有信号:")
            nov_signals = []
            for signal in signals:
                try:
                    dt = datetime.datetime.fromtimestamp(signal['date'] / 1000)
                    if dt.year == 2025 and dt.month == 11:
                        nov_signals.append({
                            'datetime': dt,
                            'type': signal.get('type'),
                            'price': signal.get('price'),
                            'strength': signal.get('strength')
                        })
                except:
                    pass
            
            if nov_signals:
                for sig in nov_signals:
                    print(f"  {sig['datetime'].strftime('%Y-%m-%d %H:%M:%S')} | {sig['type']} | {sig['price']} | {sig['strength']}")
            else:
                print("  未找到2025年11月的任何信号")
                
                # 显示文件中信号的时间范围
                if signals:
                    first_dt = datetime.datetime.fromtimestamp(signals[0]['date'] / 1000)
                    last_dt = datetime.datetime.fromtimestamp(signals[-1]['date'] / 1000)
                    print(f"\n文件中信号的时间范围: {first_dt.strftime('%Y-%m-%d')} 至 {last_dt.strftime('%Y-%m-%d')}")
    
    except Exception as e:
        print(f"解析文件时出错: {e}")

if __name__ == "__main__":
    # 分析两个信号文件
    file_24 = "/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251124_120616.json"
    file_25 = "/Users/pingan/tools/trade/tianyuan/outputs/exports/sh512660_signals_20251125_084914.json"
    
    print("=== 分析11月24日回测信号 ===")
    find_november_buys(file_24)
    
    print("\n" + "="*60 + "\n")
    
    print("=== 分析11月25日回测信号 ===")
    find_november_buys(file_25)