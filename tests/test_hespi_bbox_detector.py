import json
from pathlib import Path

import pytest

from services.hespi_bbox_coco_service import HespiBboxCocoService, SHEET_COMPONENT_CATEGORIES

IMAGE_SAMPLE_DIR = Path(__file__).resolve().parent / "image-sample"

SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")


def _find_sample_image() -> Path | None:
    if not IMAGE_SAMPLE_DIR.exists():
        return None
    for pattern in SUPPORTED_EXTENSIONS:
        files = sorted(IMAGE_SAMPLE_DIR.glob(pattern))
        if files:
            return files[0]
    return None


@pytest.fixture
def sample_image():
    image = _find_sample_image()
    if image is None:
        pytest.skip(
            f"No sample image in {IMAGE_SAMPLE_DIR}. "
            "Place a herbarium sheet scan there (jpg/png/tif) and re-run."
        )
    return image


def test_coco_json_structure(sample_image):
    """Run bbox detection on a sample herbarium sheet and validate COCO output."""
    service = HespiBboxCocoService()
    coco = service.detect_from_file(str(sample_image))

    output = json.dumps(coco, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("COCO JSON output:")
    print("=" * 70)
    print(output)
    print("=" * 70)

    assert "info" in coco
    assert "licenses" in coco
    assert "categories" in coco
    assert "images" in coco
    assert "annotations" in coco

    assert len(coco["categories"]) == 11
    category_names = {c["name"] for c in coco["categories"]}
    for name in SHEET_COMPONENT_CATEGORIES:
        assert name in category_names

    assert len(coco["images"]) == 1
    img = coco["images"][0]
    assert img["width"] > 0
    assert img["height"] > 0

    for ann in coco["annotations"]:
        assert len(ann["bbox"]) == 4
        assert all(isinstance(v, int) for v in ann["bbox"])
        assert 0 <= ann["category_id"] <= 10
        assert 0.0 <= ann["score"] <= 1.0
        assert ann["area"] > 0
        assert ann["iscrowd"] == 0
        assert isinstance(ann["segmentation"], list)

    print(f"\nDetected {len(coco['annotations'])} bounding box(es).")
