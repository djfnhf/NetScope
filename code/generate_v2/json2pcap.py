import json
import argparse
from scapy.all import *
from scapy.layers.inet import IP, TCP

def create_pcap_from_header_payload(header_file, payload_file, output_file):
    """
    将header和payload数据合并并生成PCAP文件
    
    Args:
        header_file: header JSON文件路径
        payload_file: payload JSON文件路径
        output_file: 输出PCAP文件路径
    """
    try:
        # 读取header和payload文件
        with open(header_file, 'r') as f:
            header_data = json.load(f)
        
        with open(payload_file, 'r') as f:
            payload_data = json.load(f)
        
        # 创建空的包列表
        packets = []
        processed_count = 0
        
        # 遍历每一天的数据
        for day in header_data.keys():
            if day not in payload_data:
                print(f"Warning: Day '{day}' found in header but not in payload data")
                continue
                
            # 获取当天的header和payload
            day_headers = header_data[day]
            day_payloads = payload_data[day]
            
            # 确保header和payload数量匹配
            if len(day_headers) != len(day_payloads):
                print(f"Warning: Mismatch between headers ({len(day_headers)}) and payloads ({len(day_payloads)}) for {day}")
                min_count = min(len(day_headers), len(day_payloads))
                day_headers = day_headers[:min_count]
                day_payloads = day_payloads[:min_count]
            
            # 处理每个数据包
            for i in range(len(day_headers)):
                try:
                    # 解析header信息
                    header_str = day_headers[i].replace("'", "\"")
                    header_info = json.loads(header_str)
                    
                    # 获取payload并转换为字节
                    payload_hex = day_payloads[i]
                    payload_bytes = bytes.fromhex(payload_hex)
                    
                    # 创建IP和TCP层
                    ip_layer = IP(src=header_info['src'], dst=header_info['dst'])
                    tcp_layer = TCP(sport=header_info['sport'], dport=header_info['dport'])
                    
                    # 创建完整的数据包
                    packet = ip_layer / tcp_layer / Raw(load=payload_bytes)
                    
                    # 添加到包列表
                    packets.append(packet)
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for packet {i} in {day}: {e}")
                except ValueError as e:
                    print(f"Error processing hex payload for packet {i} in {day}: {e}")
                except Exception as e:
                    print(f"Unexpected error processing packet {i} in {day}: {e}")
        
        # 写入PCAP文件
        if packets:
            wrpcap(output_file, packets)
            print(f"Successfully created {output_file} with {processed_count} packets")
        else:
            print("No packets were processed. Output file was not created.")
            
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Merge header and payload JSON files into a PCAP file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python merge_pcap.py -h header.json -p payload.json -o output.pcap
  python merge_pcap.py --header CIC2017_header.json --payload CIC2017_payload.json --output result.pcap
        '''
    )
    
    # 添加命令行参数
    parser.add_argument('-H', '--header', 
                       required=True, 
                       help='Input header JSON file path')
    
    parser.add_argument('-P', '--payload', 
                       required=True, 
                       help='Input payload JSON file path')
    
    parser.add_argument('-o', '--output', 
                       default='output1.pcap', 
                       help='Output PCAP file path (default: output.pcap)')
    
    parser.add_argument('-v', '--verbose', 
                       action='store_true', 
                       help='Enable verbose output')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 显示输入参数（如果启用详细输出）
    if args.verbose:
        print(f"Header file: {args.header}")
        print(f"Payload file: {args.payload}")
        print(f"Output file: {args.output}")
        print("Starting PCAP file creation...")
    
    # 调用主函数
    create_pcap_from_header_payload(args.header, args.payload, args.output)

if __name__ == "__main__":
    main()