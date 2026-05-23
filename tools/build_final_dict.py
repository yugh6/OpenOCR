import lmdb
import os

def get_chars_from_lmdb(lmdb_path):
    """从单个 LMDB 数据集中提取所有出现的字符"""
    chars = set()
    if not os.path.exists(lmdb_path):
        print(f"  [跳过] 路径不存在: {lmdb_path}")
        return chars
    
    print(f"  [处理中] {lmdb_path}...")
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_samples_ptr = txn.get(b'num-samples')
            if num_samples_ptr:
                num_samples = int(num_samples_ptr.decode())
                for i in range(1, num_samples + 1):
                    label = txn.get(f'label-{i:09d}'.encode())
                    if label:
                        for char in label.decode('utf-8'):
                            chars.add(char)
                    if i % 200000 == 0:
                        print(f"    已处理 {i}/{num_samples}...")
        env.close()
    except Exception as e:
        print(f"  [出错] {e}")
    return chars

def load_dict(path):
    """读取字典文件返回字符集合和原始列表(保持顺序)"""
    chars_list = []
    chars_set = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip('\n\r')
                if char and char not in chars_set:
                    chars_list.append(char)
                    chars_set.add(char)
    return chars_set, chars_list

def main():
    # 1. 配置路径（请根据你的实际路径修改）
    V1_PATH = os.path.expanduser("~/ppocr_keys_v1.txt")
    V5_PATH = os.path.expanduser("~/ppocrv5_dict.txt")
    
    # 所有的 LMDB 数据集路径列表（包含中英文、训练、测试）
    LMDB_PATHS = [
        # 英文训练集
        "/data/yxs/wh_lmdb/Union14M-L-LMDB-Filtered/filter_train_challenging",
        "/data/yxs/wh_lmdb/Union14M-L-LMDB-Filtered/filter_train_easy",
        "/data/yxs/wh_lmdb/Union14M-L-LMDB-Filtered/filter_train_hard",
        "/data/yxs/wh_lmdb/Union14M-L-LMDB-Filtered/filter_train_medium",
        "/data/yxs/wh_lmdb/Union14M-L-LMDB-Filtered/filter_train_normal",
        # 英文测试集
        "/data/yxs/wh_lmdb/parseq-test/IC13_857",
        "/data/yxs/wh_lmdb/parseq-test/CUTE80",
        "/data/yxs/wh_lmdb/parseq-test/IC15_1811",
        "/data/yxs/wh_lmdb/parseq-test/IIIT5k",
        "/data/yxs/wh_lmdb/parseq-test/SVT",
        "/data/yxs/wh_lmdb/parseq-test/SVTP",
        "/data/yxs/wh_lmdb/u14m-test/artistic",
        "/data/yxs/wh_lmdb/u14m-test/contextless",
        "/data/yxs/wh_lmdb/u14m-test/curve",
        "/data/yxs/wh_lmdb/u14m-test/general",
        "/data/yxs/wh_lmdb/u14m-test/multi_oriented",
        "/data/yxs/wh_lmdb/u14m-test/multi_words",
        "/data/yxs/wh_lmdb/u14m-test/salient",
        # 中文训练集
        "/data/yxs/wh_lmdb/benchmark_bctr/train/document_train", 
        "/data/yxs/wh_lmdb/benchmark_bctr/train/handwriting_train", 
        "/data/yxs/wh_lmdb/benchmark_bctr/train/scene_train", 
        "/data/yxs/wh_lmdb/benchmark_bctr/train/web_train", 
        # 中文测试集
        "/data/yxs/wh_lmdb/benchmark_bctr/test/document_test",
        "/data/yxs/wh_lmdb/benchmark_bctr/test/handwriting_test",
        "/data/yxs/wh_lmdb/benchmark_bctr/test/scene_test",
        "/data/yxs/wh_lmdb/benchmark_bctr/test/scene_test_1k",
        "/data/yxs/wh_lmdb/benchmark_bctr/test/web_test",
    ]

    # 2. 加载基准字典
    print("正在加载基准字典...")
    v1_set, v1_list = load_dict(V1_PATH)
    v5_set, _ = load_dict(V5_PATH)
    print(f"v1 字典大小: {len(v1_set)}, v5 字典大小: {len(v5_set)}")

    # 3. 统计全量数据中的字符
    print("\n正在扫描全量数据集（中英文+训测）...")
    data_chars = set()
    for path in LMDB_PATHS:
        data_chars.update(get_chars_from_lmdb(path))
    print(f"数据集实际出现总字符数: {len(data_chars)}")

    # 4. 执行学长的合并逻辑
    # 规则 1: v1 字典里的字符全部保留
    final_chars_list = list(v1_list) 
    
    # 规则 2 & 3: 数据中有、v1没有、但在v5有的，加入
    to_add = []
    to_discard = []
    
    for char in data_chars:
        if char not in v1_set:
            if char in v5_set:
                to_add.append(char)
            else:
                to_discard.append(char)
    
    # 将新增字符排序后追加到末尾
    to_add.sort()
    final_chars_list.extend(to_add)

    # 5. 保存结果
    output_path = "final_train_dict.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in final_chars_list:
            f.write(char + "\n")

    # 6. 打印总结报告
    print("\n" + "="*40)
    print("最终字表生成报告")
    print("="*40)
    print(f"1. v1 原始字符数: {len(v1_set)}")
    print(f"2. 从 v5 中额外补充的字符数: {len(to_add)}")
    print(f"3. 最终字表总数 (v1 + 补充): {len(final_chars_list)}")
    print(f"4. 被舍弃的无效字符数 (数据中有但v1/v5皆无): {len(to_discard)}")
    if to_discard:
        print(f"   舍弃预览: {''.join(list(to_discard)[:30])}")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()