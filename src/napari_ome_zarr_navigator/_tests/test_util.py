from botocore.exceptions import ClientError
from zarr.storage import FsspecStore

from napari_ome_zarr_navigator import util


def test_make_store_s3_anonymous_without_credentials(monkeypatch):
    monkeypatch.setattr(util, "_has_aws_credentials", lambda: False)
    store = util.make_store("S3", "s3://bucket/image.zarr")
    assert isinstance(store, FsspecStore)
    assert store.fs.anon is True  # type: ignore[attr-defined]
    assert store.read_only


def test_make_store_s3_uses_credentials_when_present(monkeypatch):
    monkeypatch.setattr(util, "_has_aws_credentials", lambda: True)
    assert util.make_store("S3", "s3://bucket/image.zarr") == "s3://bucket/image.zarr"


def test_make_store_file_returns_path():
    assert util.make_store("File", "/data/image.zarr") == "/data/image.zarr"


def _s3fs_error(exc_type, msg):
    """Mimic s3fs: a builtin OSError subclass raised from a botocore error."""
    cause = ClientError({"Error": {"Code": "400", "Message": msg}}, "HeadObject")
    try:
        raise exc_type(msg) from cause
    except exc_type as exc:
        return exc


def test_access_hint_detects_s3fs_errors(monkeypatch):
    monkeypatch.setattr(util, "_has_aws_credentials", lambda: True)
    assert util.access_hint(_s3fs_error(OSError, "Bad Request"))
    assert util.access_hint(_s3fs_error(PermissionError, "Forbidden"))


def test_access_hint_ignores_non_s3_errors():
    assert util.access_hint(PermissionError("local file")) == ""
    assert util.access_hint(ValueError("other")) == ""
