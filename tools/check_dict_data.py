import lmdb
import os

def get_chars_from_lmdb(lmdb_path):
    """从单个 LMDB 数据集中提取所有出现的字符"""
    chars = set()
    # readonly=True, lock=False 是为了在服务器上安全读取，不干扰别人
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_samples_ptr = txn.get(b'num-samples')
            if num_samples_ptr is None:
                print(f"  [跳过] {lmdb_path}: 未找到 num-samples")
                return chars
            
            num_samples = int(num_samples_ptr.decode())
            print(f"  [处理中] {lmdb_path}: 样本数 {num_samples}")

            for i in range(1, num_samples + 1):
                label_key = f'label-{i:09d}'.encode()
                label = txn.get(label_key)
                if label:
                    label_str = label.decode('utf-8')
                    for char in label_str:
                        chars.add(char)
                
                if i % 100000 == 0:
                    print(f"    已读取 {i}/{num_samples}...")
        env.close()
    except Exception as e:
        print(f"  [出错] 无法读取 {lmdb_path}: {e}")
    return chars

def main():
    # 1. 路径设置
    dict_path = os.path.expanduser("~/ppocr_keys_v1.txt")
    root_data_dir = "/data/yxs/wh_lmdb/benchmark_bctr/train"
    
    # 2. 读取字典
    dict_chars = set()
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip('\n\r')
                if char:
                    dict_chars.add(char)
    else:
        print(f"错误: 找不到字典文件 {dict_path}")
        return

    print(f"字典载入成功，总字符数: {len(dict_chars)}")

    # 3. 遍历子文件夹并汇总数据字符
    all_data_chars = set()
    sub_folders = ['document_train', 'handwriting_train', 'scene_train', 'web_train']
    
    for sub in sub_folders:
        full_path = os.path.join(root_data_dir, sub)
        if os.path.exists(full_path):
            print(f"开始扫描子数据集: {sub}")
            sub_chars = get_chars_from_lmdb(full_path)
            all_data_chars.update(sub_chars)
        else:
            print(f"提示: 路径不存在 {full_path}")

    print(f"\n所有数据扫描完成，实际出现字符总数: {len(all_data_chars)}")

    # 4. 进行集合运算对比
    # 在字典但数据里没有
    only_in_dict = dict_chars - all_data_chars
    # 在数据但字典里没有
    only_in_data = all_data_chars - dict_chars

    # 5. 输出分析报告
    print("\n" + "="*30)
    print("对齐统计分析报告")
    print("="*30)
    
    print(f"1. 数据集中存在但【字典缺失】的字符数: {len(only_in_data)}")
    if only_in_data:
        res_data = "".join(sorted(list(only_in_data)))
        print(f"   预览: {res_data[:50]}...")
        with open('missing_in_dict.txt', 'w', encoding='utf-8') as f:
            f.write(res_data)
        print("   完整列表已存至: missing_in_dict.txt")

    print(f"\n2. 字典中存在但【数据从未出现】的字符数: {len(only_in_dict)}")
    if only_in_dict:
        # 这里会有很多汉字，因为 Union14M 主要是英文/数字
        with open('unused_in_dict.txt', 'w', encoding='utf-8') as f:
            f.write("".join(sorted(list(only_in_dict))))
        print(f"   数量较多，完整列表已存至: unused_in_dict.txt")
    
    print("\n任务完成！")

if __name__ == "__main__":
    main()