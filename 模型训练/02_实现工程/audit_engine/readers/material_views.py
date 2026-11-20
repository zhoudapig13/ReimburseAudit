"""Whole-image and whole-page views, always bound to one original material."""

from __future__ import annotations

import hashlib
import math
import uuid
from pathlib import Path


def image_views(material, source: Path, cache_root: Path, *, pdf_dpi: int = 120):
    if material.type == "image":
        from PIL import Image, ImageOps

        with Image.open(source) as original:
            picture = ImageOps.exif_transpose(original)
            width, height = picture.size
            if height <= 3 * width:
                yield ({"kind": "image", "source_sha256": material.source_sha256, "page": None}, source)
                return
            # Automatic overlapping views preserve every source pixel. These
            # windows are preprocessing, not manually labelled image regions.
            cache_root.mkdir(parents=True, exist_ok=True)
            count = math.ceil(height / (1.5 * width))
            stride = math.ceil(height / count)
            overlap = min(96, stride // 8)
            for index in range(count):
                start, end = max(0, index * stride - overlap), min(height, (index + 1) * stride + overlap)
                key = hashlib.sha256(f"{material.source_sha256}:long-v1:{start}:{end}".encode()).hexdigest()
                target = cache_root / (key + ".png")
                if not target.is_file():
                    temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    picture.crop((0, start, width, end)).convert("RGB").save(temporary, format="PNG")
                    temporary.replace(target)
                yield ({"kind": "image", "source_sha256": material.source_sha256, "page": None,
                        "view_mode": "automatic_long_image_tile_v1", "view_index": index + 1,
                        "view_count": count, "source_rows": [start, end]}, target)
    elif material.type == "pdf":
        import pymupdf

        cache_root.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(source) as document:
            for index, page in enumerate(document):
                key = hashlib.sha256(
                    f"{material.source_sha256}:{pdf_dpi}:{index}".encode()
                ).hexdigest()
                target = cache_root / (key + ".png")
                if not target.is_file():
                    pixmap = page.get_pixmap(dpi=pdf_dpi, alpha=False)
                    temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    temporary.write_bytes(pixmap.tobytes("png"))
                    temporary.replace(target)
                yield (
                    {
                        "kind": "image",
                        "source_sha256": material.source_sha256,
                        "page": index + 1,
                        "pages": len(document),
                        "dpi": pdf_dpi,
                    },
                    target,
                )
