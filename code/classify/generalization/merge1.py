import os
import pandas as pd
import argparse

def merge_csv_in_folders(folder_paths):
    """
    将三个文件夹中的所有CSV文件分别合并，并保存为 tmp_1.csv, tmp_2.csv, tmp_3.csv
    
    参数：
        folder_paths (list): 三个文件夹路径列表
    """
    for i, folder_path in enumerate(folder_paths):
        # 获取文件夹中所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        
        if not csv_files:
            print(f"No CSV files in {folder_path}, skip")
            continue
        
        # 初始化一个空列表，用于存储每个CSV文件的数据
        dfs = []
        
        # 处理第一个文件
        first_file = csv_files[0]
        first_file_path = os.path.join(folder_path, first_file)
        try:
            df_first = pd.read_csv(first_file_path)
            dfs.append(df_first)
            print(f"Loaded first file: {first_file_path} ({len(df_first)} lines)")
        except Exception as e:
            print(f"Error reading first file {first_file_path}: {e}")
            continue  # 跳过整个文件夹的处理
        
        # 处理剩余文件
        for csv_file in csv_files[1:]:
            file_path = os.path.join(folder_path, csv_file)
            try:
                # 跳过标题行并使用第一个文件的列名
                df = pd.read_csv(file_path, skiprows=1, header=None, names=df_first.columns)
                dfs.append(df)
                print(f"Loaded file: {file_path} ({len(df)} lines)")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # 合并所有DataFrame
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            output_path = os.path.join(folder_path, f"tmp_{i+1}.csv")
            merged_df.to_csv(output_path, index=False)
            print(f"Concatenated {len(dfs)} files, saved to {output_path} ({len(merged_df)} lines)")
        else:
            print(f"No valid data in {folder_path}, skip")

if __name__ == "__main__":
    # 定义三个文件夹路径
    parser = argparse.ArgumentParser(description="从三个CSV文件中随机抽取数据")
    parser.add_argument("--folder1", type=str, required=True, help="第一个文件夹路径")
    parser.add_argument("--folder2", type=str, required=True, help="第二个文件夹路径")
    parser.add_argument("--folder3", type=str, required=True, help="第三个文件夹路径")
    args = parser.parse_args()
    folder_paths = [
        args.folder1,  # 替换为实际路径
        args.folder2,  # 替换为实际路径
        args.folder3   # 替换为实际路径
    ]
    
    # 调用函数
    merge_csv_in_folders(folder_paths)

"""
调用方法
python merge.py -- folder1 /path/to/folder1 -- folder2 /path/to/folder2 -- folder3 path/to/folder3
"""