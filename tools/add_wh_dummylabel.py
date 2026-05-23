import lmdb
from PIL import Image
import io
from tqdm import tqdm
import os

# ================= 配置区 =================
raw_lmdb_path = "/data/yxs/ygh/unlabeled/english/book32_lmdb"
# 建议输出到新路径，不要直接覆盖原数据，以防万一
output_lmdb_path = "/data/yxs/ygh/unlabeled/english/book32_lmdb_std"
map_size = 1099511627776  # 1TB 空间预留
write_frequency = 5000     # 每5000张图提交一次，平衡速度与内存
# =========================================

# 1. 读取原数据
env_raw = lmdb.open(raw_lmdb_path, readonly=True, lock=False)
txn_raw = env_raw.begin()
total_num = int(txn_raw.get(b"num-samples").decode())
print(f"开始处理，总样本数: {total_num}")

# 2. 检查输出路径是否存在，避免重复处理
if os.path.exists(output_lmdb_path):
    print(f"错误: 输出路径 {output_lmdb_path} 已存在，请删除或修改路径后重试。")
    exit()

# 3. 创建新 LMDB
env_out = lmdb.open(output_lmdb_path, map_size=map_size)
txn_out = env_out.begin(write=True)

pbar = tqdm(total=total_num, desc="格式转换进度")
success_count = 0
fail_count = 0

for idx in range(1, total_num + 1):
    img_key = f"image-{idx:09d}".encode()
    img_buf = txn_raw.get(img_key)

    if not img_buf:
        fail_count += 1
        pbar.update(1)
        continue

    # 核心逻辑：解析图片获取宽高 (wh)
    try:
        img = Image.open(io.BytesIO(img_buf)).convert("RGB")
        w, h = img.size
        wh_buf = f"{w}_{h}".encode()
    except Exception as e:
        # 遇到损坏的图片直接跳过，记录日志
        fail_count += 1
        if fail_count <= 10: # 只打印前10个错误，避免刷屏
            print(f"Warning: 样本 {idx} 解析失败: {e}")
        pbar.update(1)
        continue

    # 写入新 LMDB
    txn_out.put(img_key, img_buf)
    txn_out.put(f"label-{idx:09d}".encode(), b"dummy_label") 
    txn_out.put(f"wh-{idx:09d}".encode(), wh_buf)
    
    success_count += 1

    # 批量提交事务
    if idx % write_frequency == 0:
        txn_out.commit()
        txn_out = env_out.begin(write=True)
    
    pbar.update(1)

# 4. 收尾工作
txn_out.put(b"num-samples", str(success_count).encode())
txn_out.commit()
env_raw.close()
env_out.close()
pbar.close()

print(f"\n处理完成！")
print(f"  成功样本数: {success_count}")
print(f"  失败/损坏样本数: {fail_count}")
print(f"  新数据集已保存至: {output_lmdb_path}")