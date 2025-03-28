from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
import random
import argparse
import os
from glob import glob

def modify_packet(packet):
    """核心修改函数，实现对抗性扰动"""
    if IP in packet:
        # IP层对抗性修改
        packet[IP].src = "192.168.{}.{}".format(random.randint(1,254), random.randint(1,254))
        packet[IP].dst = "10.0.{}.{}".format(random.randint(1,254), random.randint(1,254))
        packet[IP].ttl = random.choice([64, 128, 255])
        
    if TCP in packet:
        packet[TCP].sport = random.randint(1024, 65535)
        packet[TCP].dport = random.randint(1024, 65535)
        packet[TCP].flags = random.choice(["S", "PA", "RA", "FA"])
        
    if UDP in packet:
        packet[UDP].sport = random.randint(1024, 65535)
        packet[UDP].dport = random.randint(1024, 65535)
        
    if Raw in packet:
        original_payload = packet[Raw].load
        modified_payload = bytearray(original_payload)
        
        # 添加随机字节扰动（3%的字节变异）
        for i in range(len(modified_payload)):
            if random.random() < 0.03:
                modified_payload[i] ^= 0xFF
                
        # 插入特定模式（例如shellcode特征混淆）
        if len(modified_payload) > 16:
            modified_payload[8:16] = b"\x90\x90\x90\x90\x90\x90\x90\x90"
        
        packet[Raw].load = bytes(modified_payload)
    
    return packet

def process_pcap(input_path, output_dir, count=100):
    """处理单个PCAP文件并生成对抗样本"""
    try:
        packets = rdpcap(input_path)
        modified_packets = []
        
        for pkt in packets[:count]:
            modified_pkt = modify_packet(pkt.copy())
            modified_packets.append(modified_pkt)
            
        # 创建输出文件路径
        base_name = os.path.basename(input_path)
        output_file = os.path.join(output_dir, f"adversarial_{base_name}")
        wrpcap(output_file, modified_packets)
        print(f"Processed {input_path} -> {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def batch_process(input_dir, output_dir, count=100):
    """批量处理文件夹中的所有PCAP文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有PCAP文件
    pcap_files = glob(os.path.join(input_dir, "*.pcap")) + glob(os.path.join(input_dir, "*.pcapng"))
    
    if not pcap_files:
        print(f"No PCAP files found in {input_dir}")
        return
    
    for pcap_file in pcap_files:
        process_pcap(pcap_file, output_dir, count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCAP对抗样本批量生成器 v2.0")
    parser.add_argument("--input_dir", required=True, help="输入PCAP文件夹路径")
    parser.add_argument("--output_dir", required=True, help="输出对抗样本文件夹路径")
    parser.add_argument("--count", type=int, default=100, help="每个文件处理数据包数量（默认100）")
    
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output_dir, args.count)