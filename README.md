# DICOM 影像批处理服务（匿名化 / 重标识 / RLE 转码）

影像数据管理员将导出的 DICOM 整理为供研究数据湖关联使用的副本。本服务提供网页批量上传
未压缩 DICOM 与扫描仪导出的单帧 16 位灰度 RLE Lossless CT、统一换号处理、
将 RLE 实例转码为未压缩影像，并下载整批处理结果（zip）的能力。

## 处理规则

对每个上传的 DICOM 文件：

| 项目 | 处理方式 |
| --- | --- |
| `PatientName`（含嵌套 PN） | **清空**（保留元素，值置空） |
| `PatientID`（含嵌套） | **替换**为确定性派生的 12 位数字新 ID（`P` + 12 位） |
| `StudyInstanceUID` / `SeriesInstanceUID` / `SOPInstanceUID` | **统一换号** |
| 上述三类 UID 的**全部嵌套引用**（序列内、文件元信息内） | 同步换号 |
| 文件元信息 `MediaStorageSOPInstanceUID` | 与新 `SOPInstanceUID` 同步 |
| SOP Class UID、Implementation Class UID | **保持原样** |
| `FrameOfReferenceUID` 等其他非身份 UID | **保持原样** |
| 未压缩实例的像素数据（PixelData） | **逐字节保持原样** |
| 未压缩实例的传输语法 | 仅接受 Implicit/Explicit VR LE、Explicit VR BE，**保持原样** |
| RLE Lossless CT 的像素数据 | **解码后以 Explicit VR Little Endian 重新封装**（OW），解码像素值逐值保留 |
| RLE 实例的传输语法 | 文件元信息 `TransferSyntaxUID` 同步改为 `1.2.840.10008.1.2.1` |
| 位深 / 符号 / CT 缩放 | `BitsAllocated/BitsStored/HighBit`、`PixelRepresentation`、`RescaleSlope/Intercept` 全部保留 |
| 其他压缩传输语法（JPEG、JPEG-LS、JPEG 2000、deflated 等） | 整批拒绝 |

### RLE Lossless CT 受理范围与转码

网页上传允许 RLE 实例与现有未压缩 DICOM **混传**，但 RLE 实例必须同时满足：

- 传输语法为 RLE Lossless（`1.2.840.10008.1.2.5`）；
- SOP Class 为 CT Image Storage；
- `BitsAllocated = 16`、`SamplesPerPixel = 1`、灰度（`MONOCHROME1/2`）；
- 单帧（无 `NumberOfFrames` 或其值为 1）。

不满足任一条件的 RLE 实例整批拒绝。转码由 pydicom 内置 RLE 处理器（仅依赖 numpy）
逐件完成：解码数组在转码写回后立即释放，**解码像素的内存占用只与当前实例大小有关，
不随批内实例数增长**。任一 RLE 实例解码或转码失败时：

- 批次详情页与上传提示指出**出错文件名**与原因；
- 整批标记 `failed`，不提供下载（下载返回 409），也不产生 zip；
- 不新增任何实例、UID/患者映射（换号）记录——所有转码与对象落盘都先于数据库写入。

### UID 换号如何保证一致

- 新 UID 由 `(角色, 原 UID)` 通过 SHA-256 **确定性派生**，结构为
  `<机构前缀>.0.<31 位十进制>`，满足 DICOM UID 规范（仅数字与点、每段无前导零、总长 ≤ 64）。
- 因为是纯函数派生，**同一原标识在任何批次、任何时间、任何并发 worker/进程下都得到同一新标识**；
  “引用先上传、目标实例后上传”也天然一致，不依赖写入顺序。
- 派生结果同时持久化到 SQLite（`uid_mappings` / `patient_mappings`），作为审计与数据湖关联依据。
- 身份 UID 的角色优先级高于标签提示：例如 `ReferencedStudySequence` 中承载 Study UID 的
  `ReferencedSOPInstanceUID` 会按 study 角色而非 sop 角色换号（见合成样例 `sr_001.dcm`）。

## 数据完整性与可靠性保证

1. **同字节复用**：同一 `SOPInstanceUID` 且原文件 SHA-256 相同，直接复用既有处理结果
   （RLE 实例不会重复解码转码）。
2. **字节冲突整批拒绝**：同一 `SOPInstanceUID` 但字节不同，整个批次标记 `failed`，
   不产生任何可下载产物。并发首次提交的竞态由 IMMEDIATE 事务内的哈希比对兜底。
3. **无半批可下载**：批次只有在全部文件校验、RLE 转码、入库、zip 打包并原子 rename
   成功后才置为 `ready`；zip 先写临时文件再 `os.replace`，失败/中断的批次下载返回 409。
4. **转码失败零写入**：RLE 解码/转码与对象落盘全部成功之前不写任何业务表，
   失败批次不新增实例、不新增换号记录。
5. **重启恢复**：进程启动时将遗留的 `processing` 批次标记 `failed`；已成功入库的实例与映射
   **保留不丢**，重新上传即可复用且标识不变。
6. **重试不变号**：确定性派生 + 映射表只插入不更新（`INSERT OR IGNORE`），重试绝不改变标识。
7. **先落盘后入库**：对象文件 `fsync` + 原子 rename 后才写数据库记录。
8. **清单可追溯**：`manifest.csv` 持续记录**上传原件**的 SHA-256，并逐件注明
   `source_transfer_syntax`（源传输语法）与 `delivery_transfer_syntax`（交付传输语法）。

## 目录结构

```
app/
  config.py        环境变量配置（数据目录、UID 前缀、上传大小、SQLite 超时）
  db.py            SQLite 连接、WAL、建表、传输语法列迁移、中断恢复
  identifiers.py   确定性 UID / PatientID 派生
  dicom_proc.py    DICOM 校验、RLE CT 受理校验与转码、标识收集、映射、匿名化重标识、对象落盘
  services.py      批次编排、zip 原子打包、CSV 清单、状态机
  web.py           Flask 路由与应用工厂
  wsgi.py          gunicorn 入口
  templates/       上传页、批次详情页
samples/
  generate_samples.py  合成样例生成器（含跨文件引用与 RLE Lossless CT）
  data/                生成的 5 个样例 DICOM
tests/             pytest 测试（含多进程并发、中断恢复与 RLE 转码）
Dockerfile
docker-compose.yml
```

## 快速开始（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

健康检查：`GET /healthz`。

数据持久化在命名卷 `dicom_data`（容器内 `/data`），包含：

- `dicom.db`（SQLite，WAL 模式）
- `objects/`（内容寻址的处理后 DICOM）
- `tmp/`（上传暂存，按批次隔离，结束即清理）
- `batches/`（最终 zip，原子写入）

## 本地开发（无 Docker）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 生成含跨文件引用与 RLE CT 的合成样例
python samples/generate_samples.py

# 运行测试（33 个，含多进程并发、RLE 转码与解码内存约束）
pytest -q

# 本地启动
DICOM_DATA_DIR=./data python -m flask --app app.wsgi:app run --debug
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DICOM_DATA_DIR` | `/data` | 数据根目录 |
| `DICOM_UID_PREFIX` | `1.2.826.0.1.3680043.10.9999`（IANA 测试弧） | 派生 UID 的组织前缀，**生产必须替换**为机构合法前缀，≤30 字符 |
| `MAX_CONTENT_LENGTH` | `2147483648`（2GB） | 单请求上传上限（字节） |
| `SQLITE_BUSY_TIMEOUT_MS` | `30000` | SQLite 写锁等待 |
| `FLASK_SECRET_KEY` | 开发占位 | 会话密钥，生产必须替换 |

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 上传表单与批次列表 |
| POST | `/upload` | multipart 多文件上传并同步处理，重定向到批次详情 |
| GET | `/batches/<id>` | 批次状态、错误信息、文件清单 |
| GET | `/batches/<id>/download` | 下载整批 zip（仅 `ready` 可用，否则 409） |
| GET | `/healthz` | 健康检查 |

下载的 zip 内含处理后的 DICOM（以新 SOP UID 命名）与 `manifest.csv`，
清单列出每个文件的原/新 PatientID、三级 UID 映射、**上传原件** SHA-256，
以及源/交付传输语法，供数据湖关联与转码审计。

## 合成样例说明

`samples/generate_samples.py` 生成 5 个样例文件：

- `ct_001.dcm` / `ct_002.dcm`：两张 64×64 带像素的未压缩 CT，共享 Study/Series；
- `ct_rle_001.dcm`：单帧 16 位有符号 RLE Lossless CT（含 `RescaleSlope/Intercept`），
  上传后转码为 Explicit VR Little Endian；
- `rs_001.dcm`：RT Structure Set，经 `ReferencedFrameOfReferenceSequence`
  → `RTReferencedStudySequence` → `ContourImageSequence` 跨文件引用两张 CT 的 SOP UID；
- `sr_001.dcm`：Enhanced SR，含 `ReferencedStudySequence`、`ReferencedSeriesSequence`、
  `CurrentRequestedProcedureEvidenceSequence` 三级嵌套引用。
