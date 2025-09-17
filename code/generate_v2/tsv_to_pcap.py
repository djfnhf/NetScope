#!/usr/bin/python3
# -*- coding:utf-8 -*-
import argparse
import csv
import os
import sys
import binascii
from typing import List

from scapy.all import Raw, Ether, wrpcap, PcapWriter


def parse_hex_words_to_bytes(text_a: str) -> bytes:
	"""
	将以空格分隔的十六进制字节串恢复为bytes。
	示例："45 00 00 28 ..." -> b"\x45\x00\x00\x28..."
	忽略空字符串，遇到非十六进制字符将抛出ValueError。
	"""
	words: List[str] = [w for w in text_a.strip().split(' ') if w != '']
	hex_str = ''.join(words)
	if len(hex_str) % 2 != 0:
		raise ValueError("hex length is not even; cannot convert to bytes")
	try:
		return binascii.unhexlify(hex_str)
	except binascii.Error as e:
		raise ValueError(f"invalid hex data: {e}")


def tsv_to_packets(tsv_path: str) -> List[bytes]:
	"""
	读取形如[label, text_a]列的TSV文件，返回每行payload对应的原始字节序列列表。
	不使用标签信息，严格按text_a恢复。
	"""
	packets: List[bytes] = []
	with open(tsv_path, newline='') as f:
		reader = csv.reader(f, delimiter='\t')
		rows = list(reader)
		if not rows:
			return packets
		# 跳过表头
		start_idx = 1 if rows[0] and len(rows[0]) >= 2 and rows[0][0] == 'label' and rows[0][1] == 'text_a' else 0
		for row in rows[start_idx:]:
			if not row or len(row) < 2:
				continue
			text_a = row[1]
			if not text_a:
				continue
			try:
				pkt_bytes = parse_hex_words_to_bytes(text_a)
				packets.append(pkt_bytes)
			except ValueError:
				# 跳过非法行
				continue
	return packets


def write_pcap_from_packets(packets: List[bytes], out_path: str) -> None:
    """
    将原始字节序列写入PCAP，并避免链路层类型不一致导致的错误：
    - 若存在可解析的以太网帧：统一写为以太网(linktype=1)。对无法解析为以太网的负载，封装为虚拟以太网帧(Ether(type=0xFFFF)/Raw)。
    - 若全部为原始负载：使用固定linktype=147(USER0)写入，避免Raw触发KeyError。
    """
    ether_packets = []  # 已解析为Ether的包
    raw_payloads = []   # 仅原始负载的包

    for pkt_bytes in packets:
        if not pkt_bytes:
            continue
        try:
            p = Ether(pkt_bytes)
            _ = p.type  # 触发解析校验
            ether_packets.append(p)
        except Exception:
            raw_payloads.append(pkt_bytes)

    if ether_packets:
        # 将所有原始负载封装成虚拟以太网帧，避免出现混合linktype
        for payload in raw_payloads:
            fake_eth = Ether(dst="00:00:00:00:00:00", src="00:00:00:00:00:00", type=0xFFFF) / Raw(load=payload)
            ether_packets.append(fake_eth)
        wrpcap(out_path, ether_packets)
        return

    # 若没有任何以太网帧，全部是原始负载：用PcapWriter指定linktype=147 (LINKTYPE_USER0)
    if raw_payloads:
        writer = PcapWriter(out_path, linktype=147, sync=True)
        try:
            for payload in raw_payloads:
                # 直接写入Raw或bytes都可；指定了linktype不会再查询l2映射
                writer.write(Raw(load=payload))
        finally:
            writer.close()


def main():
	parser = argparse.ArgumentParser(description='Convert TSV (label\ttext_a) back to PCAP')
	parser.add_argument('--tsv', required=True, type=str, help='输入TSV路径，例如 train_dataset.tsv')
	parser.add_argument('--out', required=False, type=str, help='输出PCAP路径，默认同名改为.pcap')
	args = parser.parse_args()

	tsv_path = args.tsv
	if not os.path.exists(tsv_path):
		print(f"TSV不存在: {tsv_path}")
		sys.exit(1)

	out_path = args.out
	if not out_path:
		base, _ = os.path.splitext(tsv_path)
		out_path = base + '.pcap'

	packets = tsv_to_packets(tsv_path)
	if not packets:
		print('没有可写入的包（解析失败或为空）。')
		# 仍然写一个空文件以对齐行为
		open(out_path, 'wb').close()
		return

	write_pcap_from_packets(packets, out_path)
	print(f"已写出PCAP: {out_path}, 包数量: {len(packets)}")


if __name__ == '__main__':
	main()


