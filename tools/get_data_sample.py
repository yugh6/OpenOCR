# import lmdb
# import cv2
# import numpy as np
# import os

# lmdb_path = "/data/yxs/ygh/labeled/UCTI-11M/hand"

# def export_samples(lmdb_path, output_dir="samples", start=1, num=10):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     env = lmdb.open(lmdb_path, readonly=True, lock=False)
#     with env.begin(write=False) as txn:
#         for i in range(start, start+num):
#             img_buf = txn.get(f'image-{i:09d}'.encode())
#             label = txn.get(f'label-{i:09d}'.encode()).decode('utf-8')
#             score = txn.get(f'score-{i:09d}'.encode()).decode('utf-8')

#             # 解码图片
#             img_array = np.frombuffer(img_buf, dtype=np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
#             # 保存图片，文件名包含标签内容
#             # 注意：文件名不能包含特殊字符，简单起见用序号
#             filename = f"sample_{i}_{label}_{score}.jpg"
#             cv2.imwrite(os.path.join(output_dir, filename), img)
#             print(f"已导出: {filename}")
#     env.close()

# if __name__ == "__main__":
#     export_samples(lmdb_path, output_dir="samples", start=1, num=10)



import lmdb
import cv2
import numpy as np
import os
import sys

# 配置路径
lmdb_path = "/data/yxs/ygh/labeled/UCTI-11M/hand"  # 替换为你的 LMDB 路径
output_dir = f"./samples/{os.path.basename(lmdb_path)}"  # 输出目录

def export_vertical_samples(lmdb_path, output_dir, ratio_threshold):
    """
    从 LMDB 中提取所有 h < ratio_threshold * w 的图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"正在打开 LMDB: {lmdb_path}")
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"无法打开 LMDB: {e}")
        return

    with env.begin(write=False) as txn:
        # 1. 获取总样本数
        num_samples_buf = txn.get(b"num-samples")
        if not num_samples_buf:
            print("错误: 找不到 'num-samples' 键。")
            env.close()
            return
            
        num_samples = int(num_samples_buf.decode())
        print(f"数据集总样本数: {num_samples}")
        print(f"正在筛选高宽比 < {ratio_threshold} 的图片...\n")

        count_exported = 0
        
        for i in range(1, num_samples + 1):
            # 格式化索引
            index_str = f"{i:09d}"
            
            # 2. 尝试快速获取宽高 (通过 wh- key)
            wh_key = f"wh-{index_str}".encode()
            wh_buf = txn.get(wh_key)
            
            w, h = None, None
            
            if wh_buf:
                try:
                    # 格式是 "width_height"
                    wh_str = wh_buf.decode('utf-8').strip()
                    parts = wh_str.split('_') 
                    if len(parts) == 2:
                        w = int(parts[0])
                        h = int(parts[1])
                except:
                    pass
            
            # 3. 如果 wh 获取失败，则解码图片获取宽高
            if w is None or h is None:
                img_key = f"image-{index_str}".encode()
                img_buf = txn.get(img_key)
                if not img_buf:
                    continue
                
                try:
                    img_array = np.frombuffer(img_buf, dtype=np.uint8)
                    # IMREAD_UNCHANGED 或 IMREAD_COLOR 都可以获取尺寸
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                except Exception as e:
                    # print(f"样本 {i} 解码失败: {e}")
                    continue

            # 4. 判断条件: h < 1.5 * w
            if h < ratio_threshold * w:
                # 重新获取图片和标签用于保存
                img_key = f"image-{index_str}".encode()
                label_key = f"label-{index_str}".encode()
                
                img_buf = txn.get(img_key)
                label_buf = txn.get(label_key)
                
                if img_buf:
                    try:
                        img_array = np.frombuffer(img_buf, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        label_text = ""
                        if label_buf:
                            try:
                                label_text = label_buf.decode('utf-8')
                                # 限制长度防止文件名过长
                                label_text = label_text[:50] 
                            except:
                                label_text = "unknown_label"
                        
                        filename = f"v_sample_{i}_h{h}_w{w}_{label_text}.jpg"
                        save_path = os.path.join(output_dir, filename)
                        cv2.imwrite(save_path, img)
                        
                        count_exported += 1
                        if count_exported % 100 == 0:
                            print(f"已导出 {count_exported} 张竖排/高宽比异常图片... (当前索引: {i})")
                            
                    except Exception as e:
                        print(f"保存样本 {i} 时出错: {e}")

        print("\n" + "="*30)
        print(f"筛选完成！")
        print(f"总共导出满足 h < {ratio_threshold}*w 的图片数量: {count_exported}")
        print(f"保存路径: {os.path.abspath(output_dir)}")
        print("="*30)

    env.close()

if __name__ == "__main__":
    # 可以在这里修改 threshold
    export_vertical_samples(lmdb_path, output_dir, ratio_threshold=0.25)