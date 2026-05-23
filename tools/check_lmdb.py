import lmdb
import sys

# 替换成你的LMDB路径
lmdb_path = "/data/yxs/wh_lmdb/BCTR_Filter/train/web_train"

def check_lmdb_integrity(lmdb_path):
    """
    检查 LMDB 数据集的完整性，确保每个样本都包含 image、wh、label 和 score 四个字段
    """
    print(f"正在打开 LMDB: {lmdb_path}")
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"无法打开 LMDB 环境: {e}")
        return

    with env.begin() as txn:
        # 获取总样本数
        num_samples_buf = txn.get(b"num-samples")
        if not num_samples_buf:
            print("错误: 找不到 'num-samples' 键，可能不是标准的 OCR LMDB 格式。")
            env.close()
            return
            
        num_samples = int(num_samples_buf.decode())
        print(f"数据集总样本数: {num_samples}")
        print("开始检查所有样本完整性...\n")

        error_count = 0
        missing_image_count = 0
        missing_wh_count = 0
        missing_label_count = 0
        missing_score_count = 0

        for i in range(1, num_samples + 1):
            # 格式化字符串，确保是9位数字
            index_str = f"{i:09d}"
            
            img_key = f"image-{index_str}".encode()
            wh_key = f"wh-{index_str}".encode()
            label_key = f"label-{index_str}".encode()
            score_key = f"score-{index_str}".encode()

            # 获取数据
            img_buf = txn.get(img_key)
            wh_buf = txn.get(wh_key)
            label_buf = txn.get(label_key)
            score_buf = txn.get(score_key)

            has_error = False
            missing_fields = []

            if not img_buf:
                missing_fields.append("image")
                missing_image_count += 1
                has_error = True
            if not wh_buf:
                missing_fields.append("wh")
                missing_wh_count += 1
                has_error = True
            if not label_buf:
                missing_fields.append("label")
                missing_label_count += 1
                has_error = True
            if not score_buf:
                missing_fields.append("score")
                missing_score_count += 1
                has_error = True
            if has_error:
                error_count += 1
                print(f"[错误] 样本 {i} ({index_str}): 缺失字段 {missing_fields}")

        print("\n" + "-"*30)
        print("检查完成！统计结果:")
        print(f"总样本数: {num_samples}")
        print(f"存在错误的样本数: {error_count}")
        print(f"缺失 image 的总数: {missing_image_count}")
        print(f"缺失 wh 的总数: {missing_wh_count}")
        print(f"缺失 label 的总数: {missing_label_count}")
        print(f"缺失 score 的总数: {missing_score_count}")
        print("-"*30)

    env.close()

def preview_lmdb_samples(lmdb_path, n=10):
    """
    预览 LMDB 数据集的前 n 个样本的 label 和 score
    
    Args:
        lmdb_path (str): LMDB 数据库路径
        n (int): 要预览的样本数量，默认为 10
    """
    print(f"正在打开 LMDB: {lmdb_path}")
    try:
        # 设置 readahead=False 和 meminit=False 以提高随机读取性能
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"无法打开 LMDB 环境: {e}")
        return

    with env.begin() as txn:
        # 获取总样本数
        num_samples_buf = txn.get(b"num-samples")
        if not num_samples_buf:
            print("错误: 找不到 'num-samples' 键，可能不是标准的 OCR LMDB 格式。")
            env.close()
            return
            
        num_samples = int(num_samples_buf.decode())
        
        # 确定实际要遍历的数量，不能超过总样本数
        count_to_check = min(n, num_samples)
        print(f"数据集总样本数: {num_samples}")
        print(f"正在预览前 {count_to_check} 个样本的 Label 和 Score:\n")

        for i in range(1, count_to_check + 1):
            # 格式化字符串，确保是9位数字
            index_str = f"{i:09d}"
            
            label_key = f"label-{index_str}".encode()
            score_key = f"score-{index_str}".encode()

            label_buf = txn.get(label_key)
            score_buf = txn.get(score_key)

            print(f"--- 样本 {i} ({index_str}) ---")
            
            # 处理 Label
            if label_buf:
                try:
                    # 尝试去除可能的空白字符或换行符
                    label_content = label_buf.decode('utf-8').strip()
                    print(f"Label: {label_content}")
                except UnicodeDecodeError:
                    print(f"Label: (非 UTF-8 编码二进制数据, 长度: {len(label_buf)})")
            else:
                print("Label: <缺失>")

            # 处理 Score
            if score_buf:
                try:
                    score_content = score_buf.decode('utf-8').strip()
                    print(f"Score: {score_content}")
                except UnicodeDecodeError:
                    print(f"Score: (非 UTF-8 编码二进制数据, 长度: {len(score_buf)})")
            else:
                print("Score: <缺失>")
            
            print() # 空行分隔

    env.close()


if __name__ == "__main__":
    # check_lmdb_integrity(lmdb_path)
    preview_lmdb_samples(lmdb_path, n=10)