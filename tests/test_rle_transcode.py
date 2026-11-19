"""RLE Lossless CT 受理与转码测试。

覆盖：
- RLE CT 与未压缩 DICOM 混传，下载包中转为 Explicit VR Little Endian；
- 解码像素值、位深、符号、CT 缩放参数逐值保留，文件元信息同步；
- 清单记录原件 SHA-256，并逐件注明源/交付传输语法；
- 同字节重传复用、异字节整批拒绝的规则对 RLE 实例同样成立；
- 任一 RLE 解码/转码失败：页面指出出错文件、整批不可下载、
  不新增实例也不新增换号记录；
- 转码逐件释放解码像素，解码内存不随批内实例数增长；
- 受理范围限定：单帧、BitsAllocated=16、灰度、CT Image Storage。
"""
from __future__ import annotations

import csv
import io
import weakref
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.encaps import encapsulate
from pydicom.tag import Tag
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RLELossless,
    generate_uid,
)
from werkzeug.datastructures import MultiDict

from samples import generate_samples as gen
from app import dicom_proc
from app.db import get_db
from pydicom.pixel_data_handlers import rle_handler


def _upload_batch(client, paths):
    data = MultiDict()
    handles = []
    for p in paths:
        fh = open(p, "rb")
        handles.append(fh)
        data.add("dicom_files", (fh, Path(p).name, "application/dicom"))
    resp = client.post("/upload", data=data, content_type="multipart/form-data",
                       follow_redirects=False)
    for fh in handles:
        fh.close()
    return resp


def _batch_id(resp) -> str:
    assert resp.status_code == 302
    return resp.headers["Location"].rstrip("/").split("/")[-1]


def _zip(client, batch_id) -> zipfile.ZipFile:
    resp = client.get(f"/batches/{batch_id}/download")
    assert resp.status_code == 200
    return zipfile.ZipFile(io.BytesIO(resp.data))


def _manifest_rows(zf) -> list[dict]:
    text = zf.read("manifest.csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


# ----------------------------------------------------------- 辅助：构造 RLE CT
def build_rle_ct(
    path: Path,
    *,
    sop_uid: str | None = None,
    signed: bool = True,
    rows: int = 32,
    cols: int = 32,
    seed: int = 1,
    bits_allocated: int = 16,
    samples_per_pixel: int = 1,
    photometric: str = "MONOCHROME2",
    number_of_frames: int | None = None,
    sop_class: str = CTImageStorage,
    pixels: np.ndarray | None = None,
) -> tuple[pydicom.Dataset, np.ndarray]:
    """生成单帧 RLE Lossless CT，返回 (写出的数据集, 原始像素数组)。"""
    sop_uid = sop_uid or generate_uid()
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = RLELossless
    meta.ImplementationClassUID = generate_uid()
    ds = pydicom.dataset.FileDataset(
        str(path), pydicom.dataset.Dataset(), file_meta=meta, preamble=b"\0" * 128
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "PID-RLE-0001"
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 1 if signed else 0
    ds.SamplesPerPixel = samples_per_pixel
    ds.PhotometricInterpretation = photometric
    ds.RescaleIntercept = -1024.0
    ds.RescaleSlope = 1.0
    if pixels is None:
        rng = np.random.default_rng(seed)
        if samples_per_pixel == 3:
            dtype = np.uint8 if bits_allocated == 8 else np.uint16
            pixels = rng.integers(0, 200, size=(rows, cols, 3)).astype(dtype)
        elif bits_allocated == 8:
            pixels = rng.integers(0, 255, size=(rows, cols), dtype=np.uint8)
        else:
            dtype = np.int16 if signed else np.uint16
            pixels = rng.integers(-900, 900, size=(rows, cols), dtype=np.int16).astype(dtype)
    if samples_per_pixel == 3 or (pixels is not None and pixels.dtype.kind == "u"):
        ds.PixelRepresentation = 0
    if number_of_frames is not None:
        ds.NumberOfFrames = number_of_frames
    ds.compress(RLELossless, pixels)
    ds.save_as(str(path), write_like_original=False)
    return ds, pixels


# =========================================================== 1. 基本转码正确性
def test_rle_ct_transcoded_to_evrle_with_pixels_preserved(client, tmp_path):
    rle_path = tmp_path / "rle_ct.dcm"
    _, pixels = build_rle_ct(rle_path, signed=True)

    batch_id = _batch_id(_upload_batch(client, [rle_path]))
    zf = _zip(client, batch_id)
    rows = _manifest_rows(zf)
    assert len(rows) == 1
    row = rows[0]

    # 清单：源 RLE，交付 EVRLE
    assert row["source_transfer_syntax"] == str(RLELossless)
    assert row["delivery_transfer_syntax"] == str(ExplicitVRLittleEndian)

    out = pydicom.dcmread(io.BytesIO(zf.read(row["new_sop_uid"] + ".dcm")))

    # 文件元信息同步为 EVRLE，数据集本身也是显式小端
    assert str(out.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian)
    assert out.is_implicit_VR is False and out.is_little_endian is True
    # 16 位灰度像素以显式 OW 编码
    assert out[Tag(0x7FE0, 0x0010)].VR == "OW"

    # 解码像素值逐值保留（含二补码有符号语义）
    assert out.pixel_array.dtype == np.int16
    assert np.array_equal(out.pixel_array, pixels)

    # 位深、符号、CT 缩放参数保留
    assert out.BitsAllocated == 16 and out.BitsStored == 16 and out.HighBit == 15
    assert out.PixelRepresentation == 1
    assert float(out.RescaleSlope) == 1.0
    assert float(out.RescaleIntercept) == -1024.0
    hu_out = out.pixel_array.astype(np.float64) * float(out.RescaleSlope) \
        + float(out.RescaleIntercept)
    assert np.array_equal(hu_out, pixels.astype(np.float64) - 1024.0)

    # SOP Class UID 保持
    assert str(out.SOPClassUID) == str(CTImageStorage)
    assert str(out.file_meta.MediaStorageSOPClassUID) == str(CTImageStorage)
    # 身份 UID 已换号且文件元信息同步
    assert str(out.SOPInstanceUID) == str(out.file_meta.MediaStorageSOPInstanceUID)
    assert str(out.SOPInstanceUID) != row["original_sop_uid"]
    assert str(out.PatientName) == ""


def test_unsigned_rle_ct_preserves_unsigned_dtype(client, tmp_path):
    rle_path = tmp_path / "rle_unsigned.dcm"
    _, pixels = build_rle_ct(rle_path, signed=False)
    batch_id = _batch_id(_upload_batch(client, [rle_path]))
    row = _manifest_rows(_zip(client, batch_id))[0]
    out = pydicom.dcmread(io.BytesIO(
        _zip(client, batch_id).read(row["new_sop_uid"] + ".dcm")))
    assert out.pixel_array.dtype == np.uint16
    assert np.array_equal(out.pixel_array, pixels)
    assert out.PixelRepresentation == 0


# =========================================================== 2. 与未压缩混传
def test_rle_mixed_with_uncompressed_batch(client, sample_files, tmp_path):
    rle_path = tmp_path / "rle_ct.dcm"
    build_rle_ct(rle_path, sop_uid=generate_uid())
    paths = sorted(sample_files.glob("*.dcm")) + [rle_path]

    batch_id = _batch_id(_upload_batch(client, paths))
    zf = _zip(client, batch_id)
    rows = _manifest_rows(zf)
    assert len(rows) == len(paths)

    by_source: dict[str, dict] = {}
    for r in rows:
        by_source.setdefault(r["source_transfer_syntax"], r)
    # 两种源语法都在批内
    assert str(RLELossless) in by_source
    assert str(ExplicitVRLittleEndian) in by_source

    for r in rows:
        member = pydicom.dcmread(io.BytesIO(zf.read(r["new_sop_uid"] + ".dcm")))
        # 所有交付件都是 EVRLE（未压缩原件本身即 EVRLE，保持不变）
        assert str(member.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian)
        assert r["delivery_transfer_syntax"] == str(ExplicitVRLittleEndian)

    # 未压缩原件的像素逐字节保持（取 ct_001）
    ct1_row = next(r for r in rows if r["original_filename"] == "ct_001.dcm")
    orig = pydicom.dcmread(str(sample_files / "ct_001.dcm"))
    delivered = pydicom.dcmread(io.BytesIO(
        zf.read(ct1_row["new_sop_uid"] + ".dcm")))
    assert delivered.PixelData == orig.PixelData
    assert ct1_row["source_transfer_syntax"] == str(ExplicitVRLittleEndian)


# =========================================================== 3. 清单 SHA-256 与传输语法
def test_manifest_keeps_original_sha256(client, tmp_path):
    rle_path = tmp_path / "rle_ct.dcm"
    build_rle_ct(rle_path)
    import hashlib
    original_digest = hashlib.sha256(rle_path.read_bytes()).hexdigest()

    batch_id = _batch_id(_upload_batch(client, [rle_path]))
    row = _manifest_rows(_zip(client, batch_id))[0]
    # 清单记录的是上传原件的 SHA-256，而非转码后产物
    assert row["sha256"] == original_digest

    conn = get_db()
    try:
        inst = conn.execute(
            "SELECT sha256, source_transfer_syntax, delivery_transfer_syntax "
            "FROM instances WHERE sop_uid=?",
            (row["original_sop_uid"],),
        ).fetchone()
        assert inst["sha256"] == original_digest
        assert inst["source_transfer_syntax"] == str(RLELossless)
        assert inst["delivery_transfer_syntax"] == str(ExplicitVRLittleEndian)
    finally:
        conn.close()


# =========================================================== 4. 同字节复用 / 异字节拒绝
def test_rle_identical_retransmit_reuses_result(client, tmp_path):
    rle_path = tmp_path / "rle_ct.dcm"
    build_rle_ct(rle_path, sop_uid="1.2.3.4005.1")
    b1 = _batch_id(_upload_batch(client, [rle_path]))
    z1 = _zip(client, b1)
    r1 = _manifest_rows(z1)[0]

    # 同 SOP 同字节重传：复用，产物字节一致
    b2 = _batch_id(_upload_batch(client, [rle_path]))
    z2 = _zip(client, b2)
    r2 = _manifest_rows(z2)[0]
    assert r2["new_sop_uid"] == r1["new_sop_uid"]
    assert z2.read(r2["new_sop_uid"] + ".dcm") == z1.read(r1["new_sop_uid"] + ".dcm")

    conn = get_db()
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM instances WHERE sop_uid=?",
            ("1.2.3.4005.1",),
        ).fetchone()["c"] == 1
    finally:
        conn.close()


def test_rle_same_uid_different_bytes_rejects_batch(client, tmp_path):
    good = tmp_path / "good.dcm"
    build_rle_ct(good, sop_uid="1.2.3.4006.1", seed=7)
    b1 = _batch_id(_upload_batch(client, [good]))
    assert client.get(f"/batches/{b1}/download").status_code == 200

    evil_dir = tmp_path / "evil"
    evil_dir.mkdir()
    evil = evil_dir / "evil.dcm"
    build_rle_ct(evil, sop_uid="1.2.3.4006.1", seed=9)  # 同 SOP，不同像素/字节

    other = tmp_path / "other.dcm"
    build_rle_ct(other, sop_uid="1.2.3.4006.2")

    b2 = _batch_id(_upload_batch(client, [evil, other]))
    detail = client.get(f"/batches/{b2}")
    assert b"failed" in detail.data
    assert "字节不同".encode() in detail.data
    assert client.get(f"/batches/{b2}/download").status_code == 409

    # 异字节批次未带入任何新实例
    conn = get_db()
    try:
        assert conn.execute(
            "SELECT COUNT(*) c FROM instances WHERE sop_uid=?",
            ("1.2.3.4006.2",),
        ).fetchone()["c"] == 0
    finally:
        conn.close()


# =========================================================== 5. 解码/转码失败整批拒绝
def _corrupt_rle(path: Path) -> None:
    """把 RLE 帧负载替换为非法字节，保留合法封装与元信息。"""
    ds = pydicom.dcmread(str(path))
    ds.PixelData = encapsulate([b"\x01" * 40])  # 非法 RLE 头/段
    ds.save_as(str(path), write_like_original=False)


def test_corrupt_rle_fails_whole_batch_and_names_file(client, tmp_path):
    good = tmp_path / "good_rle.dcm"
    build_rle_ct(good, sop_uid="1.2.3.5001.1")
    bad = tmp_path / "broken_rle.dcm"
    build_rle_ct(bad, sop_uid="1.2.3.5001.2")
    _corrupt_rle(bad)

    batch_id = _batch_id(_upload_batch(client, [good, bad]))
    detail = client.get(f"/batches/{batch_id}")
    assert b"failed" in detail.data
    # 页面指出出错文件
    assert "broken_rle.dcm".encode() in detail.data
    assert client.get(f"/batches/{batch_id}/download").status_code == 409

    # 不新增实例、不新增换号记录
    conn = get_db()
    try:
        assert conn.execute("SELECT COUNT(*) c FROM instances").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM uid_mappings").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM patient_mappings").fetchone()["c"] == 0
        # 批次文件也不登记
        assert conn.execute(
            "SELECT COUNT(*) c FROM batch_files WHERE batch_id=?", (batch_id,)
        ).fetchone()["c"] == 0
    finally:
        conn.close()

    # 失败后重新上传完好批次：标识派生不受影响，正常可下载
    retry = _batch_id(_upload_batch(client, [good]))
    zf = _zip(client, retry)
    assert zf.testzip() is None


def test_rle_decode_error_wrapped_with_filename(tmp_path):
    bad = tmp_path / "broken_rle.dcm"
    build_rle_ct(bad)
    _corrupt_rle(bad)
    ds = pydicom.dcmread(str(bad))
    with pytest.raises(dicom_proc.BatchRejected) as exc_info:
        dicom_proc.transcode_rle_to_evrle(ds, "broken_rle.dcm")
    assert "broken_rle.dcm" in str(exc_info.value)


# =========================================================== 6. 解码内存不随批内实例数增长
def test_decoded_pixel_memory_does_not_accumulate(client, monkeypatch, tmp_path):
    """转码 N 个 RLE 实例期间，任一时刻存活的解码数组不超过 1 个。"""
    paths = []
    for i in range(6):
        p = tmp_path / f"rle_{i:02d}.dcm"
        build_rle_ct(p, sop_uid=f"1.2.3.6000.{i}", rows=64, cols=64, seed=i)
        paths.append(p)

    live: list[weakref.ref] = []
    max_live = {"value": 0}
    original_get_pixeldata = rle_handler.get_pixeldata

    def tracking_get_pixeldata(ds, *args, **kwargs):
        arr = original_get_pixeldata(ds, *args, **kwargs)
        ref = weakref.ref(arr)
        live.append(ref)

        def _on_collect(_arg, _live=live, _ref=ref):
            if _ref in _live:
                _live.remove(_ref)

        weakref.finalize(arr, _on_collect, None)
        max_live["value"] = max(max_live["value"], sum(r() is not None for r in live))
        return arr

    # pydicom 通过模块属性查找处理器函数
    monkeypatch.setattr(rle_handler, "get_pixeldata", tracking_get_pixeldata)

    batch_id = _batch_id(_upload_batch(client, paths))
    assert client.get(f"/batches/{batch_id}/download").status_code == 200

    # 处理完全部实例后，无解码数组残留
    assert all(r() is None for r in live)
    # 峰值同时存活数为 1：逐件解码、转码后立即释放
    assert max_live["value"] == 1


# =========================================================== 7. 受理范围限定
def test_rle_non_ct_sop_class_rejected(client, tmp_path):
    from pydicom.uid import EnhancedSRStorage
    p = tmp_path / "rle_sr.dcm"
    build_rle_ct(p, sop_class=EnhancedSRStorage)
    batch_id = _batch_id(_upload_batch(client, [p]))
    detail = client.get(f"/batches/{batch_id}")
    assert b"failed" in detail.data
    assert "CT Image Storage".encode() in detail.data
    assert client.get(f"/batches/{batch_id}/download").status_code == 409


def test_rle_bits_allocated_not_16_rejected(client, tmp_path):
    p = tmp_path / "rle_8bit.dcm"
    build_rle_ct(p, bits_allocated=8)
    batch_id = _batch_id(_upload_batch(client, [p]))
    detail = client.get(f"/batches/{batch_id}")
    assert b"failed" in detail.data
    assert "BitsAllocated=16".encode() in detail.data
    assert client.get(f"/batches/{batch_id}/download").status_code == 409


def test_rle_multiframe_rejected(client, tmp_path):
    p = tmp_path / "rle_mf.dcm"
    # 手工构造多帧 RLE
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = RLELossless
    ds = pydicom.dataset.FileDataset(
        str(p), pydicom.dataset.Dataset(), file_meta=meta, preamble=b"\0" * 128
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Rows = 8
    ds.Columns = 8
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = 2
    arr = np.zeros((2, 8, 8), dtype=np.uint16)
    ds.compress(RLELossless, arr)
    ds.save_as(str(p), write_like_original=False)

    batch_id = _batch_id(_upload_batch(client, [p]))
    detail = client.get(f"/batches/{batch_id}")
    assert b"failed" in detail.data
    assert "单帧".encode() in detail.data
    assert client.get(f"/batches/{batch_id}/download").status_code == 409


def test_rle_rgb_rejected(client, tmp_path):
    p = tmp_path / "rle_rgb.dcm"
    build_rle_ct(p, samples_per_pixel=3, photometric="RGB", rows=8, cols=8)
    batch_id = _batch_id(_upload_batch(client, [p]))
    detail = client.get(f"/batches/{batch_id}")
    assert b"failed" in detail.data
    assert client.get(f"/batches/{batch_id}/download").status_code == 409


def test_other_compressed_syntax_still_rejected(client, sample_files, tmp_path):
    # 既有规则不回退：JPEG 等非 RLE 压缩语法仍整批拒绝
    ds = pydicom.dcmread(str(sample_files / "sr_001.dcm"))
    ds.file_meta.TransferSyntaxUID = pydicom.uid.JPEGBaseline8Bit
    bad = tmp_path / "jpeg.dcm"
    ds.save_as(str(bad), write_like_original=False)
    batch_id = _batch_id(_upload_batch(client, [bad]))
    detail = client.get(f"/batches/{batch_id}")
    assert b"failed" in detail.data
    assert "压缩传输语法".encode() in detail.data
    assert client.get(f"/batches/{batch_id}/download").status_code == 409


def test_rle_monochrome1_accepted_and_preserved(client, tmp_path):
    """MONOCHROME1 同样属于受理灰度范围，像素值与光度解释原样保留。"""
    p = tmp_path / "rle_mono1.dcm"
    _, pixels = build_rle_ct(p, photometric="MONOCHROME1", signed=False)
    batch_id = _batch_id(_upload_batch(client, [p]))
    row = _manifest_rows(_zip(client, batch_id))[0]
    out = pydicom.dcmread(io.BytesIO(
        _zip(client, batch_id).read(row["new_sop_uid"] + ".dcm")))
    assert str(out.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian)
    assert str(out.PhotometricInterpretation) == "MONOCHROME1"
    assert np.array_equal(out.pixel_array, pixels)


def test_implicit_vr_uncompressed_keeps_original_ts(client, tmp_path):
    """与 RLE 混传时，Implicit VR Little Endian 原件的传输语法与像素保持不变。"""
    from pydicom.uid import ImplicitVRLittleEndian
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    impl_path = tmp_path / "impl.dcm"
    ds = pydicom.dataset.FileDataset(
        str(impl_path), pydicom.dataset.Dataset(), file_meta=meta, preamble=b"\0" * 128
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "A"
    ds.PatientID = "PID-IMPL"
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    pixels = np.arange(16, dtype=np.uint16)
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(impl_path), write_like_original=False)

    rle_path = tmp_path / "rle.dcm"
    build_rle_ct(rle_path, sop_uid=generate_uid())

    batch_id = _batch_id(_upload_batch(client, [impl_path, rle_path]))
    zf = _zip(client, batch_id)
    rows = {r["original_filename"]: r for r in _manifest_rows(zf)}

    impl_row = rows["impl.dcm"]
    assert impl_row["source_transfer_syntax"] == str(ImplicitVRLittleEndian)
    assert impl_row["delivery_transfer_syntax"] == str(ImplicitVRLittleEndian)
    impl_out = pydicom.dcmread(
        io.BytesIO(zf.read(impl_row["new_sop_uid"] + ".dcm")))
    assert str(impl_out.file_meta.TransferSyntaxUID) == str(ImplicitVRLittleEndian)
    assert impl_out.is_implicit_VR is True
    assert np.array_equal(impl_out.pixel_array.reshape(-1), pixels)


# =========================================================== 8. 样例 RLE 端到端
def test_sample_rle_ct_end_to_end(client, sample_files):
    rle_path = sample_files / "ct_rle_001.dcm"
    orig = pydicom.dcmread(str(rle_path))
    assert str(orig.file_meta.TransferSyntaxUID) == str(RLELossless)
    orig_pixels = orig.pixel_array

    batch_id = _batch_id(_upload_batch(client, [rle_path]))
    row = _manifest_rows(_zip(client, batch_id))[0]
    zf = _zip(client, batch_id)
    out = pydicom.dcmread(io.BytesIO(zf.read(row["new_sop_uid"] + ".dcm")))

    assert str(out.file_meta.TransferSyntaxUID) == str(ExplicitVRLittleEndian)
    assert np.array_equal(out.pixel_array, orig_pixels)
    assert out.pixel_array.dtype == orig_pixels.dtype
    assert row["source_transfer_syntax"] == str(RLELossless)
    assert row["delivery_transfer_syntax"] == str(ExplicitVRLittleEndian)
    # 样例 SOP 出现在清单
    assert row["original_sop_uid"] == gen.SOP_CT_RLE
