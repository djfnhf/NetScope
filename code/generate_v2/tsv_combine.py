import argparse
import pandas as pd
import sys

def concatenate_tsv_files(file1, file2, output_file, ignore_index=True):
    """
    横向拼接两个TSV文件（按列方向拼接）
    
    参数:
    file1: 第一个TSV文件路径
    file2: 第二个TSV文件路径
    output_file: 输出文件路径
    ignore_index: 是否重置行索引
    """
    try:
        # 读取TSV文件
        print(f"正在读取文件: {file1}")
        df1 = pd.read_csv(file1, sep='\t')
        print(f"正在读取文件: {file2}")
        df2 = pd.read_csv(file2, sep='\t')
        
        print(f"文件1 '{file1}' 的形状: {df1.shape[0]} 行 × {df1.shape[1]} 列")
        print(f"文件2 '{file2}' 的形状: {df2.shape[0]} 行 × {df2.shape[1]} 列")
        print(f"文件1 列名: {list(df1.columns)}")
        print(f"文件2 列名: {list(df2.columns)}")
        
        # 检查行数是否一致（可选，但建议检查）
        if df1.shape[0] != df2.shape[0]:
            print(f"警告: 两个文件行数不一致 (文件1: {df1.shape[0]}, 文件2: {df2.shape[0]})")
            print("将按照较少的行数进行拼接，多余的行将被丢弃")
            
            # 取两个文件的最小行数
            min_rows = min(df1.shape[0], df2.shape[0])
            if min_rows < df1.shape[0]:
                df1 = df1.head(min_rows)
                print(f"已截断文件1至 {min_rows} 行")
            if min_rows < df2.shape[0]:
                df2 = df2.head(min_rows)
                print(f"已截断文件2至 {min_rows} 行")
        
        # 横向拼接（按列方向）
        print("正在横向拼接文件...")
        concatenated_df = pd.concat([df1, df2], axis=1, ignore_index=ignore_index)
        
        # 如果重置了索引，需要重新设置列名
        if ignore_index:
            # 生成新的列名
            new_columns = []
            for i, col in enumerate(df1.columns):
                new_columns.append(f"{col}_1")
            for i, col in enumerate(df2.columns):
                new_columns.append(f"{col}_2")
            concatenated_df.columns = new_columns
        else:
            # 保留原始列名，如果有重复会添加后缀
            pass
        
        print(f"拼接后文件形状: {concatenated_df.shape[0]} 行 × {concatenated_df.shape[1]} 列")
        print(f"拼接后列名: {list(concatenated_df.columns)}")
        
        # 保存结果
        concatenated_df.to_csv(output_file, sep='\t', index=False)
        print(f"拼接完成！结果已保存到: {output_file}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='横向拼接两个TSV文件（按列方向拼接）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python concat_tsv.py file1.tsv file2.tsv -o concatenated.tsv
  python concat_tsv.py data1.tsv data2.tsv -o result.tsv --keep-index
        '''
    )
    
    # 添加必需参数
    parser.add_argument('file1', help='第一个TSV文件路径')
    parser.add_argument('file2', help='第二个TSV文件路径')
    
    # 添加可选参数
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    parser.add_argument('--keep-index', action='store_false', dest='ignore_index',
                       help='保留原始行索引（默认会重置索引）')
    parser.add_argument('--no-check-rows', action='store_true',
                       help='不检查行数是否一致，直接拼接（可能导致数据错位）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行拼接
    success = concatenate_tsv_files(
        file1=args.file1,
        file2=args.file2,
        output_file=args.output,
        ignore_index=args.ignore_index
    )
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()