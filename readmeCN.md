# 中国建筑语义分割项目说明

本仓库提供了一个基于 TensorFlow 2 的动态图卷积神经网络（Dynamic Graph CNN，DGCNN）并结合注意力聚合模块，用于对中国古建点云数据执行语义分割。

## 仓库结构

| 文件 | 说明 |
| ---- | ---- |
| `config.py` | 汇总超参数（类别配色、数据路径、优化器设置等）的配置文件。 |
| `data_utils.py` | 将原始 `.txt` 点云转换为模型可用 `.h5` 数据集的工具脚本，包含切块、采样与特征格式化。 |
| `provider.py` | 加载 `.h5` 文件、管理文件路径、执行基础增强与统计的辅助函数。 |
| `model_Att.py` | 使用 `tf.keras` 实现带注意力聚合的 DGCNN 模型与损失函数。 |
| `train.py` | 负责端到端训练：准备数据集、构建模型、记录日志与保存权重。 |
| `application.py` | 针对单个房间/区域进行推理与评估的脚本，使用训练好的权重。 |

## 环境配置

1. **（可选）创建并激活虚拟环境：**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **安装 Python 依赖：**
   ```bash
   pip install --upgrade pip
   pip install tensorflow==2.3 numpy h5py
   ```
   *模型基于 TensorFlow 2.3 开发。理论上 2.x 版本大多兼容，但使用固定版本可以避免 API 不一致。*
3. **（可选）启用 GPU 加速：** 安装与你 CUDA/cuDNN 版本匹配的 TensorFlow GPU 发行版，可显著加速训练。

## 数据准备流程

1. **整理原始数据：**
   - 将每个房间的原始点云 `.txt` 文件放入 `./data/`。文件需包含以空格分隔的 `X Y Z R G B L` 列，其中 `L` 为语义标签。
   - 在 `./data/rooms.txt` 中逐行写入房间文件名，脚本会根据此列表读取数据。
2. **生成训练块与数据集：**
   ```bash
   python data_utils.py
   ```
   该命令将执行：
   - 遍历 `rooms.txt` 中的所有房间，将 XYZ 坐标归一化到从原点开始。
   - 对每个房间做滑动窗口切块，随机采样固定数量的点，并输出中间的 `*_blocked.h5`/`.txt` 文件（含颜色编码）。
   - 将绝对坐标与颜色转换为 9 维特征（中心化 XYZ + 归一化 XYZRGB），并汇总到 `CABDataset.h5`，同时生成 `rooms_name.txt` 记录块标识。
3. **检查输出：** 运行结束后你应看到：
   - `data/CABDataset.h5`：包含形如 `data`（B × N × 9 特征）与 `label`（B × N）的数组。
   - `data/rooms_name.txt`：存储各块名称，供训练/测试划分使用。
   - `data/all_files.txt`：列出训练时需加载的 `.h5` 文件路径（如单行 `data/CABDataset.h5`）。
   - 项目根目录下的 `<room>_blocked.h5` 与 `.txt` 文件，便于可视化检查。

## 数据格式与自定义

### 原始数据格式要求

- 预处理脚本默认读取**无表头的纯文本 `.txt` 点云文件**，每一行必须严格按照 `X Y Z R G B L` 的顺序提供七列数据。
- 坐标 `X Y Z` 视作以米为单位的浮点数，颜色 `R G B` 需为 `[0, 255]` 范围内的整数，标签 `L` 必须是 `[0, 11]` 的整型类别编号，对应 `config.NUM_CLASSES` 与 `config.COLOR_DICT`。
- 若存在空行或额外列，将在 `data_utils.txt2Matrix()` 中触发解析错误。运行 `python data_utils.py` 前请清理这些异常或先调整解析逻辑。

### 格式约束所在位置

- **解析阶段：** [`data_utils.txt2Matrix()`](data_utils.py) 会按行拆分 `rooms.txt` 中列出的文件，并默认最后一列为标签。如原始数据格式不同（例如包含法线、强度等列），请在此函数中修改读取顺序。
- **特征构建：** [`data_utils.formatAttribute()`](data_utils.py) 将 XYZRGB 转换为存储在 `CABDataset.h5` 中的 9 维特征；若需要加入更多属性，请同步扩展该函数，确保后续网络输入维度正确。
- **运行时加载：** [`provider.getDataFiles()` 与 `provider.loadH5Files()`](provider.py) 会根据 `data/all_files.txt` 查找生成的 `.h5` 数据。若数据路径或文件命名有变，可调整 `config.DATA_PATH`、`config.DATA_FILE` 或相应辅助函数。

### 自定义流程建议

1. 修改 `config.DATA_FILE` / `config.DATA_PATH` 以指向新的房间列表或数据目录。
2. 如果原始扫描包含额外列，需同步更新 `txt2Matrix()`、`formatAttribute()` 以及模型输入通道（如 `model_Att.py`）以保持维度一致。
3. 当生成多个 `.h5` 分片时，将文件名逐行写入 `data/all_files.txt`（或改写 `provider.getDataFiles()` 以自动遍历），再运行 `train.py`。
4. 变更任何格式后请重新执行 `python data_utils.py`，以保证训练/推理使用的新结构数据。

## 运行逻辑

1. **数据集加载：** `train.py` 会读取 `data/all_files.txt` 中列出的所有 `.h5`，合并后根据 `rooms_name.txt` 与 `--test_area` 指定的子串划分训练/测试集。
2. **类别权重与增强：** 训练阶段依据类别频率计算权重，并在优化时应用；同时随机对批次绕 Z 轴旋转以增强几何多样性。
3. **模型结构：** `model_Att.py` 构建带注意力的 DGCNN，通过 KNN 获取边特征，堆叠注意力聚合层、全局上下文池化，并使用 1×1 卷积预测每个点的 logits。
4. **训练循环：** 每个 epoch 遍历 `tf.data` 管线，向 TensorBoard 记录指标，按配置频率在保留区域上评估，并将最佳权重保存到 `log/`。
5. **推理流程：** `application.py` 载入训练好的 DGCNN，选取 `config.TEST_AREA` 对应的块执行分割，计算每类 OA/mIoU，并导出带颜色编码的 `.txt` 预测文件与标签对比。

## 训练

在完成数据准备后运行训练脚本：
```bash
python train.py --test_area test --log_dir log --epoch 150 --batch_size 4
```
关键参数：
- `--test_area`：用于匹配 `rooms_name.txt` 中的块名称以划分验证集。
- `--learning_rate`、`--decay_steps`、`--decay_rate`：优化器的学习率与衰减策略（默认值见 `config.py`）。
- `--log_dir`：存放 TensorBoard 日志与检查点的目录。
- `--test_frequency` / `--save_frequency`：评估与保存模型的频率。

使用 TensorBoard 监控训练：
```bash
tensorboard --logdir log
```

## 评估 / 推理

要在特定区域上评估训练好的模型：
1. 确保 `config.TEST_AREA` 与目标房间标识一致，并且存在对应的 `*_blocked.h5` 文件。
2. 运行：
   ```bash
   python application.py
   ```
   脚本会输出 OA/mIoU，写入 `精度.txt`，并导出 `{area}_predicted.txt` / `{area}_labeled.txt` 点云，使用 `config.COLOR_DICT` 进行颜色编码，便于可视化对比。

## 使用建议

- 根据实验需求在 `config.py` 中调整 `BATCH_SIZE`、`LEARNING_RATE`、`ZOOM_FACTOR` 等超参数。
- 若新增语义类别，记得更新 `config.NUM_CLASSES` 并扩展 `config.COLOR_DICT` 的颜色映射。
- 建议备份原始 `.txt` 与生成的 `.h5` 数据集；对于大规模扫描，重新生成数据集耗时较长。
