import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests


def load_downloader():
    path = Path(__file__).resolve().parents[1] / "code/scripts/download_checkpoint.py"
    spec = importlib.util.spec_from_file_location("checkpoint_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, url, data=b"", status=200, headers=None):
        self.url = url
        self.content = data
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield self.content


@pytest.mark.parametrize("permanent_failure", [False, True])
def test_verified_resume_and_retention_of_independent_ranges(
    tmp_path, permanent_failure
):
    module = load_downloader()
    data = b"01234567" * (256 * 1024) + b"last bytes"
    output = tmp_path / "model"
    parts = output / ".range_parts"
    parts.mkdir(parents=True)
    manifest = {
        "sha": "pinned",
        "siblings": [
            {
                "rfilename": "weights.safetensors",
                "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
            }
        ],
    }
    (parts / "manifest.json").write_text(json.dumps(manifest))
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  name: test/model\n  revision: pinned\n")
    record = tmp_path / "verified.json"

    class Session:
        def get(self, url, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            if permanent_failure and start == 0:
                raise requests.exceptions.SSLError("simulated TLS interruption")
            return Response(
                url,
                data[start : end + 1],
                206,
                {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            )

        def close(self):
            pass

    argv = [
        "download",
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--record-path",
        str(record),
        "--workers",
        "2",
        "--chunk-mib",
        "1",
        "--transfer-retries",
        "1",
        "--weight-base-url",
        "https://example.com/official-model",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(module.requests, "Session", Session),
        patch.object(
            module.requests,
            "request",
            side_effect=lambda method, url, **kwargs: Response(url),
        ),
    ):
        if permanent_failure:
            with pytest.raises(RuntimeError, match="1 ranges incomplete"):
                module.main()
            assert (parts / "weights.safetensors/000001").is_file()
            assert (parts / "weights.safetensors/000002").is_file()
            assert (parts / "failed_ranges.json").is_file()
            assert not record.exists()
        else:
            module.main()
            assert (output / "weights.safetensors").read_bytes() == data
            assert json.loads(record.read_text())["all_shard_sha256_verified"] is True
