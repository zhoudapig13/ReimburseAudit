import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf
from docx import Document
from openpyxl import Workbook
from PIL import Image

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "normalize_pilot_evidence.py"
)


def load_normalization_module():
    spec = importlib.util.spec_from_file_location(
        "normalize_pilot_evidence", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load normalization module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(evidence_type, filename):
    return {
        "evidence_id": "e001",
        "raw_ref": f"workflow/1.1.1/{filename}",
        "type": evidence_type,
        "role_hint": "background",
        "owners": [],
        "text": None,
        "segments": [],
        "views": [],
        "parse_status": "source_available",
    }


class SourcePathTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsafe raw_ref"):
                normalizer.resolve_source_path(raw_root, "../secret.pdf")


class EvidenceParserTests(unittest.TestCase):
    def test_image_records_dimensions_without_inventing_text(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow" / "1.1.1"
            workflow.mkdir(parents=True)
            image_path = workflow / "receipt.png"
            Image.new("RGB", (32, 18), color="white").save(image_path)

            parsed = normalizer.parse_evidence(
                evidence("image", "receipt.png"),
                raw_root=root,
                views_root=root / "views",
                workflow_id="1.1.1",
            )

            self.assertEqual(parsed["parse_status"], "visual_only")
            self.assertIsNone(parsed["text"])
            self.assertEqual(parsed["metadata"]["width_px"], 32)
            self.assertIn("multimodal_reading_required", parsed["warnings"])

    def test_docx_preserves_paragraph_table_and_embedded_image(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow" / "1.1.1"
            workflow.mkdir(parents=True)
            image_path = workflow / "stamp.png"
            Image.new("RGB", (20, 20), color="blue").save(image_path)
            document_path = workflow / "policy.docx"
            document = Document()
            document.add_paragraph("Travel policy")
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Item"
            table.cell(0, 1).text = "Limit"
            table.cell(1, 0).text = "Hotel"
            table.cell(1, 1).text = "220"
            document.add_picture(str(image_path))
            document.save(document_path)

            parsed = normalizer.parse_evidence(
                evidence("docx", "policy.docx"),
                raw_root=root,
                views_root=root / "views",
                workflow_id="1.1.1",
            )

            kinds = {segment["kind"] for segment in parsed["segments"]}
            self.assertIn("docx_paragraph", kinds)
            self.assertIn("docx_table_row", kinds)
            self.assertEqual(parsed["metadata"]["embedded_image_count"], 1)
            self.assertTrue(Path(parsed["views"][0]["file_path"]).is_file())

    def test_xlsx_preserves_cells_merges_and_unresolved_formula(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow" / "1.1.1"
            workflow.mkdir(parents=True)
            workbook_path = workflow / "limits.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Policy"
            sheet.append(["Item", "Limit", "Double"])
            sheet.append(["Hotel", 220, "=B2*2"])
            sheet.merge_cells("A3:C3")
            sheet["A3"] = "Note"
            workbook.save(workbook_path)

            parsed = normalizer.parse_evidence(
                evidence("xlsx", "limits.xlsx"),
                raw_root=root,
                views_root=root / "views",
                workflow_id="1.1.1",
            )

            self.assertEqual(parsed["parse_status"], "parsed_with_warnings")
            self.assertEqual(
                parsed["metadata"]["sheets"][0]["merged_ranges"], ["A3:C3"]
            )
            formula_cell = next(
                cell
                for segment in parsed["segments"]
                for cell in segment["cells"]
                if cell["coordinate"] == "C2"
            )
            self.assertEqual(formula_cell["formula"], "=B2*2")
            self.assertIsNone(formula_cell["cached_value"])
            self.assertIn("formula_cache_missing", parsed["warnings"])

    def test_pdf_extracts_text_and_renders_textless_page(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow" / "1.1.1"
            workflow.mkdir(parents=True)
            pdf_path = workflow / "notice.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Conference notice")
            document.new_page()
            document.save(pdf_path)
            document.close()

            parsed = normalizer.parse_evidence(
                evidence("pdf", "notice.pdf"),
                raw_root=root,
                views_root=root / "views",
                workflow_id="1.1.1",
            )

            self.assertIn("Conference notice", parsed["text"])
            self.assertEqual(parsed["metadata"]["textless_pages"], [2])
            self.assertEqual(parsed["views"][0]["page_number"], 2)
            self.assertTrue(Path(parsed["views"][0]["file_path"]).is_file())
            self.assertIn("multimodal_reading_required", parsed["warnings"])

    def test_normalized_bundle_does_not_add_private_targets(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow" / "1.1.1"
            workflow.mkdir(parents=True)
            source_path = workflow / "source.json"
            source_path.write_text('{"field":"value"}', encoding="utf-8")
            bundle = {
                "schema_version": "1.0",
                "input_version": "bundle_v1",
                "workflow_id": "1.1.1",
                "source_family_id": "sf_001",
                "input": {"application": "Trip"},
                "evidence": [evidence("json", "source.json")],
            }

            normalized = normalizer.normalize_bundle(
                bundle,
                raw_root=root,
                views_root=root / "views",
            )

            serialized = json.dumps(normalized)
            self.assertEqual(normalized["input_version"], "normalized_bundle_v1")
            self.assertNotIn('"target"', serialized)
            self.assertNotIn("reference_answer", serialized)

    def test_json_decimal_is_not_rounded_through_binary_float(self):
        normalizer = load_normalization_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow" / "1.1.1"
            workflow.mkdir(parents=True)
            source_path = workflow / "source.json"
            source_path.write_text('{"amount":12.30}', encoding="utf-8")

            parsed = normalizer.parse_evidence(
                evidence("json", "source.json"),
                raw_root=root,
                views_root=root / "views",
                workflow_id="1.1.1",
            )

            self.assertEqual(parsed["segments"][0]["value"], "12.30")


if __name__ == "__main__":
    unittest.main()
