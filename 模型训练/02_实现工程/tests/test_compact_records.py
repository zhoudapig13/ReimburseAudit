import unittest

from audit_engine.readers.compact_records import pack_records, unpack_records


class CompactRecordTests(unittest.TestCase):
    def test_duplicates_order_and_leading_zeros_preserved(self):
        original = [
            {"invoice_number": "0017"},
            {"invoice_number": "9324"},
            {"invoice_number": "0017"},
        ]
        self.assertEqual(unpack_records(pack_records(original)), original)

    def test_multiple_field_types_and_key_order(self):
        original = [
            {"amount": 12.5, "invoice_number": "0017", "flag": False},
            {"flag": None, "amount": 0, "invoice_number": "0001"},
        ]
        self.assertEqual(unpack_records(pack_records(original)), original)

    def test_omission_or_value_change_detected(self):
        packed = pack_records([{"invoice_number": "0017"}, {"invoice_number": "0020"}])
        packed["rows"].pop()
        with self.assertRaises(ValueError):
            unpack_records(packed)

    def test_nonuniform_records_rejected(self):
        with self.assertRaises(ValueError):
            pack_records([{"invoice_number": "0017"}, {"other": "0017"}])
