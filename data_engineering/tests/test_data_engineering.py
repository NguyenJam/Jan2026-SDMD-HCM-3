"""Parser edge cases plus end-to-end checks against the locked source."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from cmudict_data import Lexicon, ParseError, split_phone
from fetch_data import ROOT, load_lock, verify_sources
from run_analysis import analyze, check_regressions


class ParserTests(unittest.TestCase):
    def test_comments_blanks_and_hash_entry(self):
        lexicon = Lexicon.from_text(";;; comment\n\n#HASH  HH AE1 SH\n")
        self.assertEqual(lexicon.lookup("#hash")["selected"]["phones"], ["HH", "AE1", "SH"])
        self.assertEqual(lexicon.entries[0].source_line, 3)

    def test_variants_are_not_overwritten_or_auto_selected(self):
        lexicon = Lexicon.from_text("READ  R EH1 D\nREAD(1)  R IY1 D\n")
        result = lexicon.lookup("  rEaD ")
        self.assertTrue(result["selection_required"])
        self.assertIsNone(result["selected"])
        self.assertEqual([p["raw_label"] for p in result["pronunciations"]], ["READ", "READ(1)"])
        self.assertEqual(lexicon.lookup("read", 1)["selected"]["phones"], ["R", "IY1", "D"])

    def test_alternate_number_is_retained_even_without_unsuffixed_entry(self):
        lexicon = Lexicon.from_text("WORD(2)  W ER1 D\n")
        self.assertEqual(lexicon.lookup("word")["selected"]["variant_id"], 2)

    def test_duplicate_key_is_rejected_instead_of_losing_an_entry(self):
        with self.assertRaisesRegex(ParseError, "duplicate"):
            Lexicon.from_text("TEST  T EH1 S T\ntest  T EH0 S T\n")

    def test_identical_sequences_with_distinct_ids_are_retained(self):
        lexicon = Lexicon.from_text("TEST  T EH1 S T\nTEST(1)  T EH1 S T\n")
        self.assertEqual(len(lexicon.lookup("test")["pronunciations"]), 2)

    def test_missing_pronunciation_is_rejected(self):
        with self.assertRaisesRegex(ParseError, "Line 1"):
            Lexicon.from_text("TEST\n")

    def test_empty_lexicon_is_rejected(self):
        with self.assertRaises(ParseError):
            Lexicon.from_text(";;; only a comment\n")

    def test_unknown_phones_and_illegal_stress_are_rejected(self):
        for symbol in ("AX", "AH3", "T1", "eh1"):
            with self.subTest(symbol=symbol), self.assertRaises(ParseError):
                Lexicon.from_text(f"TEST  {symbol}\n")

    def test_stress_and_declared_bare_vowel(self):
        self.assertEqual(split_phone("AH0"), ("AH", 0))
        self.assertEqual(split_phone("OW2"), ("OW", 2))
        self.assertEqual(split_phone("T"), ("T", None))
        self.assertEqual(split_phone("AA"), ("AA", None))

    def test_zero_variant_suffix_is_rejected(self):
        with self.assertRaises(ParseError):
            Lexicon.from_text("TEST(0)  T EH1 S T\n")

    def test_punctuation_is_not_removed(self):
        lexicon = Lexicon.from_text("DON'T  D OW1 N T\n(LEFT-PAREN  L EH1 F T\n")
        self.assertEqual(lexicon.lookup("don't")["status"], "known")
        self.assertEqual(lexicon.lookup("dont")["status"], "oov")
        self.assertEqual(lexicon.lookup("(left-paren")["status"], "known")

    def test_latin1_is_decoded_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dict"
            path.write_bytes("D\u00c9J\u00c0  D EY1 ZH AA2\n".encode("latin-1"))
            self.assertEqual(Lexicon.from_file(path).lookup("d\u00e9j\u00e0")["status"], "known")

    def test_oov_and_invalid_input_never_return_a_selected_pronunciation(self):
        lexicon = Lexicon.from_text("TEST  T EH1 S T\n")
        for query, status in (("unknownxyz", "oov"), ("", "invalid_input"), ("two words", "invalid_input")):
            with self.subTest(query=query):
                result = lexicon.lookup(query)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["selected"])
                self.assertEqual(result["pronunciations"], [])

    def test_invalid_variant_never_falls_back(self):
        lexicon = Lexicon.from_text("TEST  T EH1 S T\n")
        for value, status in ((99, "invalid_variant"), (-1, "invalid_input"), (True, "invalid_input"), ("0", "invalid_input")):
            with self.subTest(value=value):
                result = lexicon.lookup("test", value)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["selected"])

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            content = b"TEST  T EH1 S T\n"
            (root / "source" / "dict").write_bytes(content + b"modified")
            lock = {"files": [{"name": "dict", "sha256": hashlib.sha256(content).hexdigest()}]}
            (root / "source_lock.json").write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify_sources(root)


class PinnedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon, cls.profile, _, _ = analyze()

    def test_sources_match_lock(self):
        self.assertEqual(len(verify_sources()), 3)

    def test_profile_counts_match_independent_source_count(self):
        # Count from raw whitespace-delimited labels, without using parser results.
        path = ROOT / "source" / "cmudict-0.7b"
        text = path.read_text(encoding=load_lock()["dictionary_encoding"])
        labels = [line.split()[0] for line in text.splitlines() if line.strip() and not line.startswith(";;;")]
        words = Counter(re.sub(r"\([0-9]+\)$", "", label) for label in labels)
        counts = self.profile["counts"]
        self.assertEqual(len(labels), counts["pronunciation_entries"])
        self.assertEqual(len(words), counts["base_spellings"])
        self.assertEqual(sum(n > 1 for n in words.values()), counts["words_with_multiple_entries"])
        self.assertEqual((len(labels), len(words)), (133854, 125074))
        self.assertEqual(counts["words_with_multiple_entries"], 8154)

    def test_known_variants_stress_oov_and_encoding_cases(self):
        self.assertEqual(len(check_regressions(self.lexicon)), 14)

    def test_inventory_distinguishes_base_phones_and_symbols(self):
        counts = self.profile["counts"]
        self.assertEqual(counts["observed_base_phones"], 39)
        self.assertEqual(counts["observed_phone_symbols"], 69)
        self.assertEqual(counts["declared_phone_symbols"], 84)
        self.assertEqual(self.profile["quality_checks"]["unsupported_phone_symbols"], [])

    def test_all_rows_are_accounted_for(self):
        c = self.profile["counts"]
        self.assertEqual(c["total_lines"], c["pronunciation_entries"] + c["comment_lines"] + c["blank_lines"])
        self.assertEqual(sum(int(n) * count for n, count in self.profile["entry_count_distribution"].items()), len(self.lexicon.entries))


if __name__ == "__main__":
    unittest.main()
