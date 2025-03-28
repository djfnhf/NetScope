import random
from scapy.all import rdpcap, wrpcap
import argparse
import os
from glob import glob

def simulate_network_errors(input_file, output_file, retransmission_prob=0.2, delta_time=0.1, shuffle=True):
    """模拟网络错误（乱序和重传）的单个PCAP文件处理器"""
    try:
        # 读取原始pcap文件
        packets = rdpcap(input_file)
        
        # 处理重传
        new_packets = []
        for pkt in packets:
            new_packets.append(pkt)
            if random.random() < retransmission_prob:
                # 复制数据包并调整时间戳
                retrans_pkt = pkt.copy()
                retrans_pkt.time += delta_time
                new_packets.append(retrans_pkt)
        
        # 处理乱序
        if shuffle:
            random.shuffle(new_packets)
        
        # 写入新pcap文件
        wrpcap(output_file, new_packets)
        print(f"Processed {input_file} -> {output_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def batch_process(input_dir, output_dir, retransmission_prob=0.2, delta_time=0.1, shuffle=True):
    """批量处理文件夹中的所有PCAP文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有PCAP文件
    pcap_files = glob(os.path.join(input_dir, "*.pcap")) + glob(os.path.join(input_dir, "*.pcapng"))
    
    if not pcap_files:
        print(f"No PCAP files found in {input_dir}")
        return
    
    for pcap_file in pcap_files:
        # 创建输出文件路径
        base_name = os.path.basename(pcap_file)
        output_file = os.path.join(output_dir, f"modified_{base_name}")
        
        # 处理每个文件
        simulate_network_errors(
            pcap_file,
            output_file,
            retransmission_prob,
            delta_time,
            shuffle
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模拟网络错误（乱序和重传）的PCAP文件批量处理器")
    parser.add_argument("--input_dir", required=True, help="输入的PCAP文件夹路径")
    parser.add_argument("--output_dir", required=True, help="输出的PCAP文件夹路径")
    parser.add_argument("--prob", type=float, default=0.2, 
                        help="重传概率（0-1，默认0.2）")
    parser.add_argument("--delta", type=float, default=0.1, 
                        help="重传时间增量（秒，默认0.1）")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle",
                        help="禁用乱序功能")
    
    args = parser.parse_args()
    
    batch_process(
        args.input_dir,
        args.output_dir,
        args.prob,
        args.delta,
        args.shuffle
    )