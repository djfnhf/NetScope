import pandas as pd
import numpy as np
import argparse
import os

def sample_and_save(input_paths, output_dir, sample_sizes):
    """
    从三个CSV文件中随机抽取指定行数并保存为新文件
    
    参数：
        input_paths (list): 输入文件路径列表（三个路径）
        output_dir (str): 输出目录
        sample_sizes (list): 各文件需抽取的行数，例如 [4_000_000, 2_000_000, 2_000_000]
    """
    # 定义输出文件名
    output_files = ["train.csv", "val.csv", "test.csv"]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(3):
        input_path = input_paths[i]
        sample_size = sample_sizes[i]
        output_path = os.path.join(output_dir, output_files[i])
        
        print(f"Process the file: {input_path} -> Extract {sample_size} rows -> save to {output_path}")
        
        # 读取CSV文件
        try:
            # 获取文件总行数（跳过标题）
            with open(input_path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1  # 减去标题行
            
            # 检查行数是否足够
            if total_lines < sample_size:
                raise ValueError(f"The {input_path} file has only {total_lines} lines and cannot extract {sample_size} lines")
            
            # 随机抽取行（跳过标题）
            skip = sorted(np.random.choice(range(1, total_lines + 1), total_lines - sample_size, replace=False))
            
            # 读取并保存
            df = pd.read_csv(input_path, skiprows=skip)
            df.to_csv(output_path, index=False)
            print(f"The {len(df)} row was saved successfully")
            
        except Exception as e:
            print(f"Error processing {input_path} : {str(e)}")
            continue

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="从三个CSV文件中随机抽取数据")
    parser.add_argument("--input1", type=str, required=True, help="第一个输入文件路径")
    parser.add_argument("--input2", type=str, required=True, help="第二个输入文件路径")
    parser.add_argument("--input3", type=str, required=True, help="第三个输入文件路径")
    parser.add_argument("--output_dir", type=str, default="sampled_data", help="输出目录")
    args = parser.parse_args()
    
    # 调用函数
    sample_and_save(
        input_paths=[args.input1, args.input2, args.input3],
        output_dir=args.output_dir,
        sample_sizes=[4000000, 2000000, 2000000]
    )

"""
调用方法
python sample_data.py --input1 path/to/file1.csv --input2 path/to/file2.csv --input3 path/to/file3.csv --output_dir my_output_folder
"""