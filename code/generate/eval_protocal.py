import argparse
import pandas as pd
from typing import Dict, Any

# 增强协议规范检查规则
PROTOCOL_RULES = {
    6: {  # TCP
        "valid_ports": lambda x: 1 <= x <= 65535,  # 排除0端口
        "required_flags": ["SYN", "ACK", "RST", "FIN"],
        "allowed_flags": ["SYN", "ACK", "RST", "FIN", "PSH", "URG"],
        "header_length": lambda x: x >= 20
    },
    17: {  # UDP
        "valid_ports": lambda x: 0 <= x <= 65535,
        "disallowed_flags": ["SYN", "ACK", "RST", "FIN"],
        "header_length": lambda x: x == 8
    },
    0: {  # 处理实例中的ICMP协议号为0的情况
        "valid_ports": lambda x: x is None,
        "required_fields": ["ICMP Type", "ICMP Code"]
    }
}

def validate_protocol(row: Dict[str, Any]) -> Dict[str, str]:
    """增强型协议合规性验证"""
    errors = []
    proto = row["Protocol"]
    
    # 协议号有效性验证
    if proto not in PROTOCOL_RULES:
        return {"Flow ID": row["Flow ID"], "valid": False, "errors": [f"未知协议号: {proto}"]}
    
    rules = PROTOCOL_RULES[proto]
    
    try:
        # 端口验证
        if "valid_ports" in rules:
            for port_field in ["Src Port", "Dst Port"]:
                port = row[port_field]
                if not rules["valid_ports"](port):
                    errors.append(f"无效{port_field}: {port}")

        # 标志位验证
        if proto == 6:  # TCP
            for flag in rules["required_flags"]:
                if row[f"{flag} Flag Count"] < 0:
                    errors.append(f"非法TCP标志: {flag}")
                    
        elif proto in [17, 0]:  # UDP和ICMP
            for flag in rules.get("disallowed_flags", []):
                if row[f"{flag} Flag Count"] > 0:
                    errors.append(f"协议{proto}不允许{flag}标志")

        # 头部长度验证
        if "header_length" in rules:
            total_header = row["Fwd Header Length"] + row["Bwd Header Length"]
            if not rules["header_length"](total_header):
                errors.append(f"头部长度{total_header}不符合协议规范")

        # ICMP专用验证
        if proto in [0, 1]:
            for field in rules["required_fields"]:
                if pd.isna(row[field]) or row[field] < 0:
                    errors.append(f"缺失或无效{field}")

    except KeyError as e:
        errors.append(f"缺失关键字段: {str(e)}")
    
    return {
        "Flow ID": row["Flow ID"],
        "valid": len(errors) == 0,
        "errors": errors
    }

def evaluate_compliance(test_file: str, gen_file: str) -> pd.DataFrame:
    """执行合规性评估"""
    try:
        # 读取CSV数据集
        test_df = pd.read_csv(test_file)
        gen_df = pd.read_csv(gen_file)
        
        # 执行验证
        results = []
        for df in [test_df, gen_df]:
            df_results = df.apply(validate_protocol, axis=1).tolist()
            results.append(pd.DataFrame(df_results))
            
        return pd.concat([
            results[0].add_prefix("Test_"), 
            results[1].add_prefix("Gen_")
        ], axis=1)
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='网络流量协议合规性评估')
    parser.add_argument('--test', type=str, required=True, help='测试集CSV文件路径')
    parser.add_argument('--gen', type=str, required=True, help='生成数据集CSV文件路径')
    args = parser.parse_args()

    report = evaluate_compliance(args.test, args.gen)
    
    # 输出报告
    print("\n=== 合规性评估报告 ===")
    print(f"测试集通过率: {report['Test_valid'].mean():.2%}")
    print(f"生成集通过率: {report['Gen_valid'].mean():.2%}")
    
    print("\n错误详情（前10条）:")
    print(report[report['Gen_valid'] == False].head(10))