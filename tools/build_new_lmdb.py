import json
import lmdb
import os
import io
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
json_path = "/remote-home/yexingsun/OpenOCR/output/json/pseudo_labels_subdir2.json" 
old_lmdb_path = "/data/yxs/ygh/unlabeled/chinese/subdir2_std" # 老的无标注数据集
new_lmdb_path = "/data/yxs/ygh/labeled/first/chinese/subdir22" # 新建的纯净数据集路径 
THRESHOLD = 0.90 
# ==========================================

print("1. 读取并过滤 JSON 文件...")
with open(json_path, 'r', encoding='utf-8') as f:
    try: data = json.load(f)
    except:
        f.seek(0)
        data = [json.loads(line) for line in f]

valid_data = [item for item in data if item['score'] >= THRESHOLD]
print(f"过滤完成，保留了 {len(valid_data)} 条高质量伪标签。")

print(f"\n2. 准备创建全新纯净数据集: {new_lmdb_path}")
if not os.path.exists(new_lmdb_path):
    os.makedirs(new_lmdb_path)

env_in = lmdb.open(old_lmdb_path, readonly=True, lock=False)
txn_in = env_in.begin()

env_out = lmdb.open(new_lmdb_path, map_size=1099511627776) 
txn_out = env_out.begin(write=True)

new_index = 1 
success_count = 0

print("3. 正在提取图片、标签和 WH 信息...")
for item in tqdm(valid_data):
    old_image_key_str = item['image_id'] # 如 "image-000000005"
    label_text = item['label']
    
    # 获取老库的图片数据
    img_buf = txn_in.get(old_image_key_str.encode('utf-8'))
    if img_buf is None:
        print(f"图片缺失: {old_image_key_str}")
        continue
    if not label_text or len(label_text.strip()) == 0:
        print(f"标签非法: {old_image_key_str}, label={label_text}")
        continue
        
    # 构造新库的 Key
    new_image_key = f"image-{new_index:09d}".encode('utf-8')
    new_label_key = f"label-{new_index:09d}".encode('utf-8')
    new_wh_key = f"wh-{new_index:09d}".encode('utf-8')
    
    # 获取老库的 WH 数据 
    old_wh_key_str = old_image_key_str.replace("image-", "wh-")
    wh_buf = txn_in.get(old_wh_key_str.encode('utf-8'))
    
    if wh_buf is not None:
        # 如果老库里有，直接复制过来 (假设你老库已经是正确的格式了)
        txn_out.put(new_wh_key, wh_buf)
    else:
        # 如果老库碰巧缺了这个键，当场解码图片算出来，确保新库 100% 完美
        logging.warning(f"WH 信息缺失，正在解码图片计算 WH: {old_image_key_str}")
        try:
            img = Image.open(io.BytesIO(img_buf))
            w, h = img.size
            txn_out.put(new_wh_key, f"{w}_{h}".encode('utf-8'))
        except Exception as e:
            print(f"图片解码失败: {old_image_key_str}, 错误: {str(e)}")
            print(f"跳过图片: {old_image_key_str}")
            continue # 图片实在坏了就算了
    
    # 写入新库 (图片 + 标签)
    txn_out.put(new_image_key, img_buf)
    txn_out.put(new_label_key, label_text.encode('utf-8'))
    
    new_index += 1
    success_count += 1

# 写入总数护身符
num_samples_key = b'num-samples'
txn_out.put(num_samples_key, str(success_count).encode('utf-8'))

# 提交保存并关门
txn_out.commit()
env_in.close()
env_out.close()

print(f"\n完成！包含 Image, Label 和 WH 的全功能纯净数据集已生成。")
print(f"总计写入了 {success_count} 条数据。")