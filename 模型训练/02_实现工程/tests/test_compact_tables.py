import copy
import unittest

from audit_engine.readers.compact_tables import (
    cell_projection,
    pack_table,
    unpack_table,
)


class CompactTableTests(unittest.TestCase):
    def fixture(self):
        return {
            "type": "xlsx",
            "segments": [
                {
                    "kind": "xlsx_row",
                    "segment_id": "e001:t001",
                    "source_locator": {"sheet": "Policy", "row": 4},
                    "cells": [
                        {
                            "coordinate": "B4",
                            "value": "Professor",
                            "value_type": "s",
                            "number_format": "General",
                        },
                        {
                            "coordinate": "D4",
                            "value": 220,
                            "value_type": "n",
                            "number_format": "$0.00",
                        },
                        {
                            "coordinate": "AA4",
                            "value": "2024-07-19T00:00:00",
                            "value_type": "d",
                            "number_format": "mm-dd-yy",
                        },
                    ],
                }
            ],
        }

    def test_roundtrip_preserves_sparse_coordinates_values_types_formats(self):
        evidence = self.fixture()
        self.assertEqual(unpack_table(pack_table(evidence)), cell_projection(evidence))

    def test_modified_amount_fails_digest(self):
        packed = pack_table(self.fixture())
        packed["sheets"][0]["r"][0][2][3] = 440
        with self.assertRaises(ValueError):
            unpack_table(packed)

    def test_duplicate_coordinate_and_unrepresented_metadata_rejected(self):
        evidence = self.fixture()
        evidence["segments"][0]["cells"].append(
            copy.deepcopy(evidence["segments"][0]["cells"][-1])
        )
        with self.assertRaises(ValueError):
            pack_table(evidence)
        evidence = self.fixture()
        evidence["segments"][0]["cells"][0]["comment"] = "exception rule"
        with self.assertRaises(ValueError):
            pack_table(evidence)
