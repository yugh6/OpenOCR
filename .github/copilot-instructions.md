# OpenOCR Copilot 使用说明

## 仓库概览
- 复旦团队的通用 OCR 工具箱（检测、识别、公式、表格、文档解析），基于 PaddleOCR/MMOCR。核心组件：OpenOCR（检测+识别）、UniRec-0.1B（统一识别）、OpenDoc-0.1B（版面+UniRec）。
- 关键目录：模型与数据配置在 [configs](../configs)；检测代码在 [opendet](../opendet)，识别代码在 [openrec](../openrec)，训练/评测/导出/推理脚本在 [tools](../tools)，操作说明在 [docs](../docs)，演示脚本在仓库根目录。

## 配置与命令习惯
- 脚本统一使用 `--c` 指定 YAML，`--o key=value` 追加/覆盖（PaddleOCR 风格）。`Global.*` 常用字段：device、backend、infer_img、onnx_model_path 等。
- 推理后端：Torch 默认；ONNX 需 `Global.backend=onnx`，设备由 `Global.device` 控制（cpu 或 CUDA 编号）。
- 模型配置入口：识别在 [configs/rec](../configs/rec)，检测在 [configs/det](../configs/det)。典型配置：UniRec 用 [configs/rec/unirec/focalsvtr_ardecoder_unirec.yml](../configs/rec/unirec/focalsvtr_ardecoder_unirec.yml)；OpenOCR 移动端用识别 [configs/rec/svtrv2/repsvtr_ch.yml](../configs/rec/svtrv2/repsvtr_ch.yml) + 检测 [configs/det/dbnet/repvit_db.yml](../configs/det/dbnet/repvit_db.yml)。

## 环境速记
- Torch 推理：`pip install -r requirements.txt` 后安装 torch >=1.13（推荐 2.2.0 + CUDA 11.8），再从 release/HF/ModelScope 下载权重。ONNX 推理：仅需 `openocr-python` + `onnxruntime`。
- OpenDoc 额外依赖 Paddle：`paddlepaddle-gpu==3.2.0`、`paddlex`、`pypdfium2`、`opencv-contrib-python`，torch 版本文档推荐 2.6.0（见 [docs/opendoc.md](../docs/opendoc.md)）。

## 常用流程
- OpenOCR 全流程推理：`python tools/infer_e2e.py --img_path=PATH [--backend=onnx --device=cpu --onnx_det_model_path=... --onnx_rec_model_path=...]`
- 仅检测：`python tools/infer_det.py --c configs/det/dbnet/repvit_db.yml --o Global.infer_img=PATH`
- 仅识别：`python tools/infer_rec.py --c configs/rec/svtrv2/repsvtr_ch.yml --o Global.infer_img=PATH`（替换为其他模型配置即可）
- UniRec Torch 推理：`python tools/infer_rec.py --c configs/rec/unirec/focalsvtr_ardecoder_unirec.yml --o Global.infer_img=PATH`
- UniRec ONNX 推理：下载 `unirec_0_1b_onnx` 后运行 `python tools/depolyment/unirec_onnx/infer_onnx.py --image PATH`
- OpenDoc 文档解析：`python tools/infer_doc.py --input_path PATH --output_path ./output --gpus -1|0,1 --is_save_vis_img --is_save_json --is_save_markdown --pretty`（版面 PP-DocLayoutV2 + 识别 UniRec）
- 训练示例（UniRec）：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=23333 --nproc_per_node=8 tools/train_rec.py --c configs/rec/unirec/focalsvtr_ardecoder_unirec.yml`；数据根路径在 YAML 的 `Train.dataset.root_path` 等字段。
- 导出 ONNX：`python tools/toonnx.py --c CONFIG --o Global.device=cpu`，输出位于 `output/.../export_*/*.onnx`。
- Gradio 演示：`demo_gradio.py`（OpenOCR），`demo_unirec.py`（UniRec），`demo_opendoc.py`（OpenDoc）；需安装 `gradio==4.20.0`，按文档解压示例数据。

## 约定与提示
- 数据多为 LMDB，生成工具见 [tools/create_lmdb_dataset.py](../tools/create_lmdb_dataset.py)；UniRec40M 需先合并 `data.mdb.part_*`。
- 权重存放在 GitHub releases / HuggingFace / ModelScope；配置默认使用相对路径，可用 `--o Global.pretrained_model=PATH` 覆盖。
- 依赖栈混合：Paddle（版面）、PyTorch（识别）、ONNX 可选；排查环境问题时保持独立环境更容易定位。
- 检测基础类在 [opendet/modeling/base_detector.py](../opendet/modeling/base_detector.py)，识别基础类在 [openrec/modeling/base_recognizer.py](../openrec/modeling/base_recognizer.py)；各自的 backbone/neck/head/encoder/decoder 在子目录中注册。
- 新增模型时沿用现有配置与注册方式（`openrec/modeling/*` 或 `opendet/modeling/*`），并尽量复用 `Global` 配置键，保持脚本兼容性。

如果有不清楚或缺失的信息，请告诉我以便补充和调整。
