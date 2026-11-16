"""Offline tests for the Zenodo download helper (network calls are mocked)."""

import hashlib
import logging

import pytest
import requests

from napari_ome_zarr_navigator import _sample_data

_PAYLOAD = b"fake zarr zip bytes" * 1000
_KEY = "operetta_plate.zarr.zip"
# Real Zenodo file links end in /content, not in the file name.
_LINK = f"https://zenodo.org/api/records/8322727/files/{_KEY}/content"
_RECORD = {
    "metadata": {
        "doi": "10.5281/zenodo.8322727",
        "title": "Test record",
        "publication_date": "2023-09-06",
    },
    "files": [
        {
            "key": _KEY,
            "size": len(_PAYLOAD),
            "checksum": "md5:" + hashlib.md5(_PAYLOAD).hexdigest(),
            "links": {"self": _LINK},
        }
    ],
}


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json = json_data
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_zenodo(monkeypatch):
    calls = []

    def fake_get(url, params=None, stream=False, timeout=None):
        calls.append({"url": url, "params": params})
        if url.startswith("https://zenodo.org/api/records/") and url.endswith(
            "/content"
        ):
            return _FakeResponse(content=_PAYLOAD)
        return _FakeResponse(json_data=_RECORD)

    monkeypatch.setattr(_sample_data.requests, "get", fake_get)
    return calls


@pytest.fixture
def pkg_caplog(caplog, monkeypatch):
    """caplog for the package logger, which has propagate=False (see util.py)."""
    monkeypatch.setattr(
        logging.getLogger("napari_ome_zarr_navigator"), "propagate", True
    )
    caplog.set_level(logging.DEBUG, logger=_sample_data.logger.name)
    return caplog


def test_download_saves_under_file_key_with_valid_checksum(
    tmp_path, fake_zenodo, pkg_caplog
):
    target = tmp_path / "record"
    _sample_data.download_from_zenodo("10.5281_zenodo.8322727", directory=target)

    assert sorted(p.name for p in target.iterdir()) == [_KEY]
    assert (target / _KEY).read_bytes() == _PAYLOAD
    infos = [r for r in pkg_caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1, "expected exactly one user-facing notification"
    assert "Test record" in infos[0].getMessage()
    assert any("Checksum OK" in r.getMessage() for r in pkg_caplog.records)
    assert not [r for r in pkg_caplog.records if r.levelno >= logging.WARNING]


def test_checksum_mismatch_logs_warning(tmp_path, monkeypatch, pkg_caplog):
    def fake_get(url, params=None, stream=False, timeout=None):
        if url == _LINK:
            return _FakeResponse(content=b"corrupted")
        return _FakeResponse(json_data=_RECORD)

    monkeypatch.setattr(_sample_data.requests, "get", fake_get)
    _sample_data.download_from_zenodo(
        "10.5281_zenodo.8322727", directory=tmp_path / "record"
    )
    warnings = [r for r in pkg_caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Checksum mismatch" in warnings[0].getMessage()


def test_existing_directory_skips_download_with_warning(
    tmp_path, fake_zenodo, pkg_caplog
):
    target = tmp_path / "record"
    target.mkdir()
    _sample_data.download_from_zenodo("10.5281_zenodo.8322727", directory=target)

    assert list(target.iterdir()) == []
    assert not [c for c in fake_zenodo if c["url"] == _LINK]
    assert any(
        r.levelno == logging.WARNING and "already exists" in r.getMessage()
        for r in pkg_caplog.records
    )


def test_access_token_only_sent_when_given(tmp_path, fake_zenodo):
    _sample_data.download_from_zenodo(
        "10.5281_zenodo.8322727", directory=tmp_path / "a"
    )
    _sample_data.download_from_zenodo(
        "10.5281_zenodo.8322727", directory=tmp_path / "b", access_token="secret"
    )
    file_calls = [c for c in fake_zenodo if c["url"] == _LINK]
    assert [c["params"] for c in file_calls] == [None, {"access_token": "secret"}]


def test_http_error_on_file_download_raises(tmp_path, monkeypatch):
    def fake_get(url, params=None, stream=False, timeout=None):
        if url == _LINK:
            return _FakeResponse(status_code=404)
        return _FakeResponse(json_data=_RECORD)

    monkeypatch.setattr(_sample_data.requests, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        _sample_data.download_from_zenodo(
            "10.5281_zenodo.8322727", directory=tmp_path / "record"
        )
