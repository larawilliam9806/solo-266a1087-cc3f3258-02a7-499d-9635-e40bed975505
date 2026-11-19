"""DICOM 批处理核心：校验、标识收集、映射、匿名化重标识、RLE 转码、落盘。

不变量：
- 仅清空 PatientName、替换 PatientID；
- 仅对 Study/Series/SOP 三类 UID 及其嵌套引用换号；
- SOP Class UID、FrameOfReferenceUID 等其他 UID 不动；
- 未压缩实例的像素数据与传输语法原样保留；
- 单帧 16 位灰度 RLE Lossless CT 转码为 Explicit VR Little Endian：
  解码像素值、位深、符号与 CT 缩放（RescaleSlope/Intercept）全部保留，
  文件元信息 TransferSyntaxUID 同步更新，PixelData 以显式 OW 落盘；
- RLE 实例逐件解码、转码后立即释放解码数组，解码内存不随批内实例数增长；
- 同字节实例复用既有结果，字节冲突整批拒绝；
- 任一实例校验/解码/转码失败都在任何数据库写入之前抛出，整批拒绝。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.dataelem import DataElement
from pydicom.tag import Tag
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RLELossless,
)

from . import config, identifiers
from .db import get_db, immediate_tx

# 未压缩传输语法白名单（deflated 1.2.1.99 属于压缩，拒绝）
UNCOMPRESSED_TS = {
    "1.2.840.10008.1.2",      # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",    # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",    # Explicit VR Big Endian
}

# 唯一接受的压缩传输语法：RLE Lossless（仅限单帧 16 位灰度 CT，见 _validate_rle_ct）
RLE_TS = str(RLELossless)
# RLE 转码后的交付传输语法
DELIVERY_TS = str(ExplicitVRLittleEndian)
TAG_PIXEL_DATA = Tag(0x7FE0, 0x0010)

# 三类身份标识标签
TAG_STUDY_UID = Tag(0x0020, 0x000D)
TAG_SERIES_UID = Tag(0x0020, 0x000E)
TAG_SOP_UID = Tag(0x0008, 0x0018)
TAG_SOP_CLASS = Tag(0x0008, 0x0016)
TAG_PATIENT_ID = Tag(0x0010, 0x0020)
TAG_PATIENT_NAME = Tag(0x0010, 0x0010)

# 非身份 UI 标签的角色提示（仅当值不在本批身份集合中时作为兜底）
ROLE_HINT_TAGS: dict[Tag, str] = {
    Tag(0x0002, 0x0003): "sop",   # Media Storage SOP Instance UID（文件元信息同步）
    Tag(0x0008, 0x1155): "sop",   # Referenced SOP Instance UID
}

# 明确不换号的 UI 标签：类 UID、传输语法、帧参考 UID 等。
# TransferSyntaxUID 不参与身份换号：未压缩实例原样保留，RLE 实例由
# _transcode_rle_to_evrle 在转码时显式改写为 Explicit VR Little Endian。
EXCLUDED_UI_TAGS = {
    Tag(0x0002, 0x0002),  # Media Storage SOP Class UID
    Tag(0x0002, 0x0010),  # Transfer Syntax UID
    Tag(0x0002, 0x0012),  # Implementation Class UID
    Tag(0x0008, 0x0016),  # SOP Class UID
    Tag(0x0008, 0x1150),  # Referenced SOP Class UID
    Tag(0x0020, 0x0052),  # Frame of Reference UID
    Tag(0x0020, 0x9162),  # Synchronization Frame of Reference UID
    Tag(0x0020, 0x9165),  # Concatenation Frame of Reference UID
}

ROLES = ("study", "series", "sop")


class BatchRejected(Exception):
    """整批拒绝类错误（任何实例失败都不产出可下载结果）。"""


@dataclass
class PlannedFile:
    tmp_path: Path
    orig_name: str
    sha256: str
    study_uid: str
    series_uid: str
    sop_uid: str
    patient_id: str
    ds: pydicom.Dataset
    source_ts: str                 # 上传原件的传输语法（清单记录用）
    delivery_ts: str               # 交付产物的传输语法（RLE 转 EVRLE，其余原样）
    is_rle: bool = False           # 是否为需要转码的 RLE Lossless CT
    reused: bool = False           # 命中既有实例，直接复用对象存储文件
    object_relpath: str = ""


@dataclass
class BatchPlan:
    files: list[PlannedFile] = field(default_factory=list)
    # 全局值 -> 角色（身份优先，标签提示兜底），跨整个批次统一
    value_roles: dict[str, str] = field(default_factory=dict)
    # 全局值 -> 新 UID
    uid_remap: dict[str, str] = field(default_factory=dict)
    patient_remap: dict[str, str] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_ui_elements(ds: pydicom.Dataset):
    """遍历数据集与文件元信息中的全部 UI 元素（含序列嵌套）。"""
    seen: set[int] = set()

    def callback(_ds, elem):
        if elem.VR == "UI":
            key = id(elem)
            if key not in seen:
                seen.add(key)
                yield_elems.append(elem)

    yield_elems: list = []
    ds.walk(callback, recursive=True)
    file_meta = getattr(ds, "file_meta", None)
    if file_meta:
        file_meta.walk(callback, recursive=True)
    return yield_elems


def _ui_values(elem) -> list[str]:
    """取 UI 元素的全部值（兼容单值与多值 VM）。"""
    value = elem.value
    if value is None:
        return []
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return [v for v in value if v]
    return [value]


def _read_and_validate(path: Path) -> tuple[pydicom.Dataset, str, bool]:
    """读取并校验上传文件。

    返回 (数据集, 源传输语法 UID, 是否 RLE)。接受：
    - 未压缩传输语法（Implicit/Explicit VR LE、Explicit VR BE）；
    - 单帧、BitsAllocated=16、灰度的 RLE Lossless CT（后续转 EVRLE）。
    其余压缩语法或不符合限定条件的 RLE 一律整批拒绝。
    """
    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception as exc:  # noqa: BLE001 - 统一转成批次拒绝
        raise BatchRejected(f"无法解析为 DICOM 文件 {path.name}: {exc}") from exc

    if not ds.file_meta or "TransferSyntaxUID" not in ds.file_meta:
        raise BatchRejected(f"{path.name} 缺少文件元信息或 TransferSyntaxUID")
    ts = str(ds.file_meta.TransferSyntaxUID)
    if ts in UNCOMPRESSED_TS:
        return ds, ts, False
    if ts == RLE_TS:
        _validate_rle_ct(path.name, ds)
        return ds, ts, True
    raise BatchRejected(
        f"{path.name} 使用不支持的压缩传输语法 {ts}，"
        "本服务仅接受未压缩 DICOM 与单帧 16 位灰度 RLE Lossless CT"
    )


def _validate_rle_ct(name: str, ds: pydicom.Dataset) -> None:
    """校验 RLE 实例严格落在“单帧 16 位灰度 CT”的受理范围内。"""
    sop_class = str(ds.SOPClassUID) if TAG_SOP_CLASS in ds else ""
    meta_class = (
        str(ds.file_meta.MediaStorageSOPClassUID)
        if "MediaStorageSOPClassUID" in ds.file_meta
        else ""
    )
    if CTImageStorage not in (sop_class, meta_class):
        raise BatchRejected(
            f"{name} 为 RLE Lossless 但不是 CT Image Storage 实例"
            f"（SOPClassUID={sop_class or meta_class or '缺失'}），整批拒绝"
        )
    if int(getattr(ds, "BitsAllocated", 0)) != 16:
        raise BatchRejected(
            f"{name} 的 BitsAllocated={getattr(ds, 'BitsAllocated', '缺失')}，"
            "本服务仅接受 BitsAllocated=16 的 RLE CT，整批拒绝"
        )
    if int(getattr(ds, "SamplesPerPixel", 0)) != 1:
        raise BatchRejected(
            f"{name} 的 SamplesPerPixel={getattr(ds, 'SamplesPerPixel', '缺失')}，"
            "本服务仅接受单样本灰度 RLE CT，整批拒绝"
        )
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise BatchRejected(
            f"{name} 的 PhotometricInterpretation={photometric or '缺失'}，"
            "本服务仅接受灰度（MONOCHROME1/2）RLE CT，整批拒绝"
        )
    nr_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if nr_frames != 1:
        raise BatchRejected(
            f"{name} 为 {nr_frames} 帧 RLE 实例，本服务仅接受单帧 RLE CT，整批拒绝"
        )
    for tag, label in (
        (Tag(0x0028, 0x0010), "Rows"),
        (Tag(0x0028, 0x0011), "Columns"),
        (TAG_PIXEL_DATA, "PixelData"),
    ):
        if tag not in ds or ds[tag].value in (None, b""):
            raise BatchRejected(f"{name} 缺少 {label}，无法转码，整批拒绝")
    if int(getattr(ds, "PixelRepresentation", -1)) not in (0, 1):
        raise BatchRejected(
            f"{name} 的 PixelRepresentation={getattr(ds, 'PixelRepresentation', '缺失')}，"
            "无法确定像素符号，整批拒绝"
        )


def transcode_rle_to_evrle(ds: pydicom.Dataset, name: str) -> None:
    """把单帧 RLE Lossless CT 原地转码为 Explicit VR Little Endian。

    - 经 pydicom RLE 处理器解码，像素值与 PixelRepresentation 决定的符号保持；
    - 位深（BitsAllocated/Stored、HighBit）、CT 缩放（RescaleSlope/Intercept）
      等像素相关属性一概不触碰；
    - 文件元信息 TransferSyntaxUID 同步为 1.2.840.10008.1.2.1；
    - PixelData 以显式小端 OW 元素写回，解码数组立即释放（不保留在数据集缓存中），
      使解码像素的内存占用只与当前实例有关，不随批内实例数增长。
    """
    try:
        arr = ds.pixel_array  # RLE 解码（单帧 -> 2D ndarray，dtype 随 PixelRepresentation）
    except Exception as exc:  # noqa: BLE001 - 解码失败统一整批拒绝
        raise BatchRejected(f"{name} RLE 像素解码失败: {exc}") from exc
    try:
        expected_bytes = int(ds.Rows) * int(ds.Columns) * int(ds.BitsAllocated) // 8
        if arr.shape != (int(ds.Rows), int(ds.Columns)):
            raise ValueError(
                f"解码帧形状 {arr.shape} 与 Rows/Columns "
                f"({ds.Rows},{ds.Columns}) 不一致"
            )
        if arr.dtype.itemsize * 8 != int(ds.BitsAllocated):
            raise ValueError(
                f"解码位深 {arr.dtype.itemsize * 8} 与 BitsAllocated "
                f"{ds.BitsAllocated} 不一致"
            )
        # 显式小端序列化；dtype.kind（u/i）保留无符号/二补码符号语义
        raw = arr.astype(f"<{arr.dtype.kind}{arr.dtype.itemsize}").tobytes()
        if len(raw) != expected_bytes:
            raise ValueError(
                f"解码字节数 {len(raw)} 与期望 {expected_bytes} 不一致"
            )
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds[TAG_PIXEL_DATA] = DataElement(TAG_PIXEL_DATA, "OW", raw)
        # 立即释放解码缓存，避免逐件转码时所有解码像素在数据集上累积
        for attr in ("_pixel_array", "_pixel_id"):
            if hasattr(ds, attr):
                delattr(ds, attr)
        ds.is_decompressed = False
    except BatchRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - 转码失败统一整批拒绝
        raise BatchRejected(f"{name} RLE 转码失败: {exc}") from exc
    finally:
        del arr  # 引用计数即时回收解码数组


def _object_relpath(new_sop_uid: str) -> str:
    # 以新 SOP UID 做两级分片，字符仅含数字与点，天然文件名安全
    return str(Path(new_sop_uid[:2]) / new_sop_uid[2:4] / f"{new_sop_uid}.dcm")


def plan_batch(tmp_files: list[tuple[str, Path]]) -> BatchPlan:
    """读取并校验整批文件，构建全局映射计划；任何冲突直接抛 BatchRejected。"""
    if not tmp_files:
        raise BatchRejected("批次为空")

    plan = BatchPlan()
    seen_sop: dict[str, str] = {}       # 原 SOP UID -> sha256
    conn = get_db()
    try:
        for orig_name, tmp_path in tmp_files:
            ds, source_ts, is_rle = _read_and_validate(tmp_path)
            digest = sha256_file(tmp_path)
            study_uid = str(ds[TAG_STUDY_UID].value)
            series_uid = str(ds[TAG_SERIES_UID].value)
            sop_uid = str(ds[TAG_SOP_UID].value)
            patient_id = str(ds[TAG_PATIENT_ID].value) if TAG_PATIENT_ID in ds else ""

            # 批内同 SOP 去重/冲突
            if sop_uid in seen_sop:
                if seen_sop[sop_uid] != digest:
                    raise BatchRejected(
                        f"批次内 SOPInstanceUID {sop_uid} 对应文件字节不一致，整批拒绝"
                    )
                continue  # 完全重复文件，只处理一次

            # 与库内既有实例比对：同 UID 同字节复用，字节不同整批拒绝
            row = conn.execute(
                "SELECT sha256, object_path, new_sop_uid FROM instances WHERE sop_uid=?",
                (sop_uid,),
            ).fetchone()
            reused = False
            object_relpath = ""
            if row is not None:
                if row["sha256"] != digest:
                    raise BatchRejected(
                        f"SOPInstanceUID {sop_uid} 已存在但文件字节不同，整批拒绝"
                    )
                reused = True
                object_relpath = row["object_path"]

            pf = PlannedFile(
                tmp_path=tmp_path,
                orig_name=orig_name,
                sha256=digest,
                study_uid=study_uid,
                series_uid=series_uid,
                sop_uid=sop_uid,
                patient_id=patient_id,
                ds=ds,
                source_ts=source_ts,
                # RLE 实例交付为 EVRLE；未压缩实例传输语法保持原样
                delivery_ts=DELIVERY_TS if is_rle else source_ts,
                is_rle=is_rle,
                reused=reused,
                object_relpath=object_relpath,
            )
            plan.files.append(pf)
            seen_sop[sop_uid] = digest

            # 1) 身份 UID 优先登记角色
            for uid, role in (
                (study_uid, "study"),
                (series_uid, "series"),
                (sop_uid, "sop"),
            ):
                existing = plan.value_roles.get(uid)
                if existing is not None and existing != role:
                    raise BatchRejected(
                        f"UID {uid} 在批次中同时被当作 {existing} 与 {role}，整批拒绝"
                    )
                plan.value_roles[uid] = role

            # 2) 其余 UI 值按标签提示登记角色，冲突拒绝
            for elem in _iter_ui_elements(ds):
                if elem.tag in EXCLUDED_UI_TAGS:
                    continue
                hint = ROLE_HINT_TAGS.get(elem.tag)
                for value in _ui_values(elem):
                    if value in plan.value_roles:
                        continue
                    if hint is not None:
                        plan.value_roles[value] = hint
                    # 无提示且非身份的 UI（如 StorageMediaFileSetUID）保持原值

        # 3) 与库内历史映射核对角色，防止跨批次角色漂移
        if plan.value_roles:
            placeholders = ",".join("?" * len(plan.value_roles))
            rows = conn.execute(
                f"SELECT role, original_uid FROM uid_mappings "
                f"WHERE original_uid IN ({placeholders})",
                tuple(plan.value_roles),
            ).fetchall()
            for row in rows:
                wanted = plan.value_roles.get(row["original_uid"])
                if wanted is not None and wanted != row["role"]:
                    raise BatchRejected(
                        f"UID {row['original_uid']} 历史角色为 {row['role']}，"
                        f"本批为 {wanted}，整批拒绝"
                    )
                plan.value_roles[row["original_uid"]] = row["role"]

        # 4) 确定性派生全部新 UID 与新 PatientID
        for original_uid, role in plan.value_roles.items():
            plan.uid_remap[original_uid] = identifiers.derive_uid(role, original_uid)
        patient_ids = {f.patient_id for f in plan.files}
        for pid in patient_ids:
            plan.patient_remap[pid] = identifiers.derive_patient_id(pid)
        return plan
    finally:
        conn.close()


def _transform_dataset(ds: pydicom.Dataset, plan: BatchPlan) -> None:
    """按计划原地改写数据集（含文件元信息），像素与传输语法不触碰。"""

    def callback(_ds, elem):
        tag = elem.tag
        if tag == TAG_PATIENT_NAME:
            elem.value = ""  # 仅清空，保留元素与 VR
            return
        if tag == TAG_PATIENT_ID:
            # 对任何遇到的 PatientID（含计划外的嵌套值）都确定性派生，
            # 避免未登记的患者 ID 原样泄露
            elem.value = plan.patient_remap.get(
                str(elem.value),
                identifiers.derive_patient_id(str(elem.value)),
            )
            return
        if elem.VR != "UI" or tag in EXCLUDED_UI_TAGS:
            return
        raw = elem.value
        if raw is None:
            return
        if isinstance(raw, (list, pydicom.multival.MultiValue)):
            elem.value = [plan.uid_remap.get(v, v) if v else v for v in raw]
        else:
            elem.value = plan.uid_remap.get(raw, raw)

    ds.walk(callback, recursive=True)
    file_meta = getattr(ds, "file_meta", None)
    if file_meta:
        file_meta.walk(callback, recursive=True)


def _atomic_write_object(ds: pydicom.Dataset, object_relpath: str) -> Path:
    objects_dir = config.get_objects_dir()
    final_path = objects_dir / object_relpath
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        # 并发下另一 worker 已写出同一确定性结果，直接复用
        return final_path
    tmp_path = objects_dir / f".{os.getpid()}-{id(ds)}.dcm.tmp"
    # write_like_original=False 保证带前导与文件元信息（含换号后的
    # Media Storage SOP Instance UID）按标准 DICOM 文件格式落盘
    ds.save_as(str(tmp_path), write_like_original=False)
    with open(tmp_path, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def persist_mappings(plan: BatchPlan) -> None:
    """持久化 UID/患者映射（只插入不更新；确定性派生保证值恒定）。"""
    conn = get_db()
    try:
        with immediate_tx(conn):
            for original_uid, role in plan.value_roles.items():
                conn.execute(
                    "INSERT OR IGNORE INTO uid_mappings(role, original_uid, new_uid) "
                    "VALUES(?,?,?)",
                    (role, original_uid, plan.uid_remap[original_uid]),
                )
            for orig_pid, new_pid in plan.patient_remap.items():
                conn.execute(
                    "INSERT OR IGNORE INTO patient_mappings(original_patient_id, new_patient_id) "
                    "VALUES(?,?)",
                    (orig_pid, new_pid),
                )
    finally:
        conn.close()


def process_plan(plan: BatchPlan, batch_id: str) -> None:
    """执行计划：RLE 转码 -> 写对象存储 -> 映射/实例入库 -> 登记批次文件。

    先落盘后入库，且所有像素解码/转码与对象写入都在任何数据库写入之前完成：
    任一 RLE 实例解码或转码失败都会在这里抛出 BatchRejected，
    不新增任何实例、映射或换号记录，整批不可下载。

    RLE 逐件转码并即时释放解码数组，解码像素内存只与当前实例大小有关，
    不随批内实例数增长。

    实例登记在 IMMEDIATE 事务内做“插入或哈希比对”，堵住两个进程同时首次
    提交同一 SOP 不同字节时 plan 阶段都未见历史记录的竞态窗口。
    """
    # 阶段 1：转码 + 对象落盘（无任何数据库写入，失败可干净整批拒绝）
    for pf in plan.files:
        if pf.reused:
            continue
        _transform_dataset(pf.ds, plan)
        if pf.is_rle:
            transcode_rle_to_evrle(pf.ds, pf.orig_name)
        new_sop = plan.uid_remap[pf.sop_uid]
        pf.object_relpath = _object_relpath(new_sop)
        _atomic_write_object(pf.ds, pf.object_relpath)

    # 阶段 2：全部转码/落盘成功后，才持久化映射与实例
    persist_mappings(plan)

    conn = get_db()
    try:
        for pf in plan.files:
            with immediate_tx(conn):
                existing = conn.execute(
                    "SELECT sha256 FROM instances WHERE sop_uid=?",
                    (pf.sop_uid,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO instances"
                        "(sop_uid, sha256, new_sop_uid, object_path, "
                        "source_transfer_syntax, delivery_transfer_syntax) "
                        "VALUES(?,?,?,?,?,?)",
                        (
                            pf.sop_uid,
                            pf.sha256,
                            plan.uid_remap[pf.sop_uid],
                            pf.object_relpath,
                            pf.source_ts,
                            pf.delivery_ts,
                        ),
                    )
                elif existing["sha256"] != pf.sha256:
                    # 并发竞态兜底：另一批次已提交同 SOP 但字节不同
                    raise BatchRejected(
                        f"SOPInstanceUID {pf.sop_uid} 并发提交了不同字节的文件，整批拒绝"
                    )
                conn.execute(
                    "INSERT OR IGNORE INTO batch_files(batch_id, sop_uid, orig_name, "
                    "original_patient_id, original_study_uid, original_series_uid) "
                    "VALUES(?,?,?,?,?,?)",
                    (batch_id, pf.sop_uid, pf.orig_name,
                     pf.patient_id, pf.study_uid, pf.series_uid),
                )
    finally:
        conn.close()
