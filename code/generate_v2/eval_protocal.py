import scapy.all as scapy
from scapy.layers import http, dns
import ipaddress
import re
from collections import defaultdict
import argparse
import sys
import os

class ProtocolComplianceChecker:
    def __init__(self):
        self.results = {
            'summary': defaultdict(int),
            'violations': [],
            'statistics': defaultdict(int),
            'compliant_packets': 0  # 记录合规数据包数量
        }
        
    def check_ip_compliance(self, packet):
        """检查IP协议合规性"""
        violations = []
        
        if not packet.haslayer(scapy.IP):
            return violations
            
        ip_layer = packet[scapy.IP]
        
        # 检查IP版本
        if ip_layer.version != 4:
            violations.append(f"IP版本异常: {ip_layer.version}")
            
        # 检查头部长度
        if ip_layer.ihl < 5 or ip_layer.ihl > 15:
            violations.append(f"IP头部长度异常: {ip_layer.ihl}")
            
        # 检查TTL
        if ip_layer.ttl == 0:
            violations.append("TTL为0")
            
        # 检查源地址和目的地址
        try:
            src_ip = ipaddress.ip_address(ip_layer.src)
            if src_ip.is_multicast or src_ip.is_unspecified:
                violations.append(f"异常源IP地址: {ip_layer.src}")
                
            dst_ip = ipaddress.ip_address(ip_layer.dst)
            if dst_ip.is_unspecified:
                violations.append(f"异常目的IP地址: {ip_layer.dst}")
        except ValueError:
            violations.append(f"无效IP地址格式")
            
        return violations
    
    def check_tcp_compliance(self, packet):
        """检查TCP协议合规性"""
        violations = []
        
        if not packet.haslayer(scapy.TCP):
            return violations
            
        tcp_layer = packet[scapy.TCP]
        
        # 检查端口号
        if tcp_layer.sport == 0 or tcp_layer.dport == 0:
            violations.append("TCP端口号为0")
            
        # 检查标志位组合
        flags = tcp_layer.flags
        if flags & 0x01 and flags & 0x04:  # FIN和RST同时设置
            violations.append("FIN和RST标志位同时设置")
            
        if flags & 0x02 and flags & 0x10:  # SYN和ACK同时设置但不是握手包
            # 检查序列号确认号关系
            if tcp_layer.ack == 0:
                violations.append("SYN-ACK包中ACK号为0")
                
        # 检查窗口大小
        if tcp_layer.window == 0:
            violations.append("TCP窗口大小为0")
            
        return violations
    
    def check_udp_compliance(self, packet):
        """检查UDP协议合规性"""
        violations = []
        
        if not packet.haslayer(scapy.UDP):
            return violations
            
        udp_layer = packet[scapy.UDP]
        
        # 检查端口号
        if udp_layer.sport == 0 or udp_layer.dport == 0:
            violations.append("UDP端口号为0")
            
        # 检查长度字段
        if udp_layer.len < 8:
            violations.append(f"UDP长度字段异常: {udp_layer.len}")
            
        return violations
    
    def check_http_compliance(self, packet):
        """检查HTTP协议合规性"""
        violations = []
        
        if not packet.haslayer(scapy.TCP) or not packet.haslayer(http.HTTPRequest):
            return violations
            
        http_layer = packet[http.HTTPRequest]
        
        # 检查HTTP方法
        method = http_layer.Method.decode('utf-8', errors='ignore') if http_layer.Method else ""
        valid_methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]
        if method and method not in valid_methods:
            violations.append(f"无效HTTP方法: {method}")
            
        # 检查HTTP版本
        if hasattr(http_layer, 'Http_Version'):
            version = http_layer.Http_Version.decode('utf-8', errors='ignore') if http_layer.Http_Version else ""
            if version and not re.match(r'HTTP/\d\.\d', version):
                violations.append(f"无效HTTP版本: {version}")
                
        # 检查Host头
        if hasattr(http_layer, 'Host'):
            host = http_layer.Host.decode('utf-8', errors='ignore') if http_layer.Host else ""
            if not host:
                violations.append("缺少Host头部")
                
        return violations
    
    def check_dns_compliance(self, packet):
        """检查DNS协议合规性"""
        violations = []
        
        if not packet.haslayer(scapy.UDP) or not packet.haslayer(dns.DNS):
            return violations
            
        dns_layer = packet[dns.DNS]
        
        # 检查DNS事务ID
        if dns_layer.id == 0:
            violations.append("DNS事务ID为0")
            
        # 检查QR位
        if dns_layer.qr not in [0, 1]:
            violations.append(f"DNS QR位异常: {dns_layer.qr}")
            
        # 检查响应码
        if dns_layer.qr == 1 and dns_layer.rcode > 5:  # 响应包且响应码超出标准范围
            violations.append(f"DNS响应码异常: {dns_layer.rcode}")
            
        return violations
    
    def analyze_packet(self, packet):
        """分析单个数据包的协议合规性"""
        packet_violations = []
        
        # 记录协议统计
        if packet.haslayer(scapy.IP):
            self.results['statistics']['IP'] += 1
        if packet.haslayer(scapy.TCP):
            self.results['statistics']['TCP'] += 1
        if packet.haslayer(scapy.UDP):
            self.results['statistics']['UDP'] += 1
        if packet.haslayer(http.HTTPRequest):
            self.results['statistics']['HTTP'] += 1
        if packet.haslayer(dns.DNS):
            self.results['statistics']['DNS'] += 1
            
        # 执行各层协议检查
        packet_violations.extend(self.check_ip_compliance(packet))
        packet_violations.extend(self.check_tcp_compliance(packet))
        packet_violations.extend(self.check_udp_compliance(packet))
        packet_violations.extend(self.check_http_compliance(packet))
        packet_violations.extend(self.check_dns_compliance(packet))
        
        # 记录违规
        if packet_violations:
            self.results['summary']['total_violations'] += len(packet_violations)
            self.results['violations'].append({
                'packet_number': len(self.results['violations']) + 1,
                'timestamp': packet.time,
                'violations': packet_violations,
                'summary': f"发现 {len(packet_violations)} 个合规性问题"
            })
        else:
            # 如果没有违规，则这是一个合规数据包
            self.results['compliant_packets'] += 1
    
    def analyze_pcap(self, pcap_file):
        """分析整个pcap文件"""
        try:
            print(f"正在分析文件: {pcap_file}")
            packets = scapy.rdpcap(pcap_file)
            self.results['summary']['total_packets'] = len(packets)
            print(f"发现 {len(packets)} 个数据包，开始分析...")
            
            for i, packet in enumerate(packets):
                if (i + 1) % 1000 == 0:
                    print(f"已分析 {i + 1} 个数据包...")
                self.analyze_packet(packet)
                
            # 计算合规率：合规数据包数 / 总数据包数
            if self.results['summary']['total_packets'] > 0:
                self.results['summary']['compliance_rate'] = (
                    self.results['compliant_packets'] / self.results['summary']['total_packets']
                )
            else:
                self.results['summary']['compliance_rate'] = 1.0
            
            print("分析完成!")
            
        except Exception as e:
            print(f"分析pcap文件时出错: {e}")
            return False
            
        return True
    
    def generate_report(self):
        """生成合规性报告"""
        report = []
        report.append("=" * 60)
        report.append("网络流量协议合规性评估报告")
        report.append("=" * 60)
        report.append(f"分析数据包总数: {self.results['summary']['total_packets']}")
        report.append(f"合规数据包数量: {self.results['compliant_packets']}")
        report.append(f"违规数据包数量: {len(self.results['violations'])}")
        report.append(f"协议合规率: {self.results['summary']['compliance_rate']:.2%}")
        report.append("")
        
        report.append("协议统计:")
        for protocol, count in self.results['statistics'].items():
            report.append(f"  {protocol}: {count} 个数据包")
        report.append("")
        
        if self.results['violations']:
            report.append("详细违规信息:")
            for violation in self.results['violations']:
                report.append(f"数据包 #{violation['packet_number']} (时间: {violation['timestamp']}):")
                for v in violation['violations']:
                    report.append(f"  - {v}")
                report.append("")
        else:
            report.append("未发现协议合规性问题!")
            
        return "\n".join(report)
    
    def save_report(self, filename):
        """保存报告到文件"""
        report = self.generate_report()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {filename}")
        except Exception as e:
            print(f"保存报告时出错: {e}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="网络流量协议合规性评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s traffic.pcap
  %(prog)s traffic.pcap -o report.txt
  %(prog)s traffic.pcap --output detailed_report.txt --verbose
        """
    )
    
    parser.add_argument(
        'pcap_file',
        help='要分析的PCAP文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='compliance_report.txt',
        help='输出报告文件名 (默认: compliance_report.txt)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细分析过程'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查文件是否存在
    if not os.path.exists(args.pcap_file):
        print(f"错误: 文件 '{args.pcap_file}' 不存在!")
        sys.exit(1)
    
    # 检查文件扩展名
    if not args.pcap_file.lower().endswith(('.pcap', '.cap', '.pcapng')):
        print(f"警告: 文件 '{args.pcap_file}' 可能不是标准的PCAP文件格式")
    
    # 创建检查器实例
    checker = ProtocolComplianceChecker()
    
    # 分析pcap文件
    if checker.analyze_pcap(args.pcap_file):
        # 生成并显示报告
        report = checker.generate_report()
        print("\n" + report)
        
        # 保存报告到文件
        checker.save_report(args.output)
        
        # 如果有违规且启用了详细模式，显示更多信息
        if args.verbose and checker.results['violations']:
            print(f"\n发现 {len(checker.results['violations'])} 个有问题的数据包")
    else:
        print("PCAP文件分析失败")
        sys.exit(1)

if __name__ == "__main__":
    main()