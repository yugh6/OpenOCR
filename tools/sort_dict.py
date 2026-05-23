import os

def sort_dictionary(input_path, output_path):
    print(f"正在读取文件: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    # 1. 读取并去重
    unique_chars = set()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除行尾的换行符
            char = line.strip('\r\n')
            # 只有非空字符才加入（如果你需要保留空格，请根据情况修改逻辑）
            unique_chars.add(char)
    
    print(f"去重后字符总数: {len(unique_chars)}")

    # 2. 按照 Unicode 编码排序
    # Python 的默认 sort 就是按 Unicode 编码点 (Code Point) 排序的
    sorted_chars = sorted(list(unique_chars))

    # 3. 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')

    print(f"排序完成！已保存至: {output_path}")
    
    # 4. 打印预览（头和尾）
    print("-" * 30)
    print("【预览】")
    print(f"前5个字符: {sorted_chars[:5]}")
    print(f"后5个字符: {sorted_chars[-5:]}")
    print("-" * 30)

if __name__ == "__main__":
    # --- 配置 ---
    # 这里填你之前生成的那个乱序的字典文件名
    INPUT_FILE = "final_train_dict.txt" 
    
    # 这里填排序后要保存的文件名
    OUTPUT_FILE = "final_train_dict_sorted.txt"
    # -----------
    
    sort_dictionary(INPUT_FILE, OUTPUT_FILE)