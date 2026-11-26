"""CMUdict 0.7b parsing and single-token lookup; no speech synthesis."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
BASE_PHONES = frozenset(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH".split()
)
# The companion symbols file declares bare vowels as well as stressed vowels.
LEGAL_SYMBOLS = BASE_PHONES | frozenset(v + d for v in VOWELS for d in "012")
VARIANT_SUFFIX = re.compile(r"\(([1-9][0-9]*)\)$")


class ParseError(ValueError):
    """A source row cannot be accepted without changing or losing information."""


def normalize_word(word):
    """Trim surrounding space and uppercase; retain accents and punctuation."""
    return word.strip().upper()


def split_phone(symbol):
    """Return the base phone and stress, without deleting stress from the entry."""
    if symbol not in LEGAL_SYMBOLS:
        raise ValueError(f"Unsupported CMUdict symbol: {symbol}")
    if symbol[-1:] in {"0", "1", "2"}:
        return symbol[:-1], int(symbol[-1])
    return symbol, None


@dataclass(frozen=True)
class Pronunciation:
    raw_label: str
    word: str
    variant_id: int  # 0 = unsuffixed entry; n = literal source suffix (n).
    phones: tuple[str, ...]
    source_line: int

    def as_dict(self):
        return {
            "raw_label": self.raw_label,
            "word": self.word,
            "variant_id": self.variant_id,
            "phones": list(self.phones),
            "source_line": self.source_line,
        }


class Lexicon:
    def __init__(self, entries):
        self.entries = tuple(entries)
        index = defaultdict(list)
        for entry in self.entries:
            index[entry.word].append(entry)
        # Retain source order. The first entry is not a context-aware prediction.
        self.by_word = {word: tuple(values) for word, values in index.items()}

    @classmethod
    def from_file(cls, path, encoding="latin-1"):
        return cls.from_text(Path(path).read_text(encoding=encoding))

    @classmethod
    def from_text(cls, text):
        entries = []
        seen = set()
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(";;;"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ParseError(f"Line {line_number}: missing pronunciation")
            label, *phones = fields
            match = VARIANT_SUFFIX.search(label)
            variant_id = int(match.group(1)) if match else 0
            spelling = label[:match.start()] if match else label
            if not spelling:
                raise ParseError(f"Line {line_number}: empty base spelling")
            # Reject numeric suffixes outside the documented positive-ID format.
            if not match and re.search(r"\([+-]?[0-9]+\)$", label):
                raise ParseError(f"Line {line_number}: invalid variant suffix {label}")
            word = normalize_word(spelling)
            invalid = sorted(set(phones) - LEGAL_SYMBOLS)
            if invalid:
                raise ParseError(f"Line {line_number}: unsupported symbols {invalid}")
            key = word, variant_id
            if key in seen:
                raise ParseError(f"Line {line_number}: duplicate word/variant {key}")
            seen.add(key)
            entries.append(Pronunciation(label, word, variant_id, tuple(phones), line_number))
        if not entries:
            raise ParseError("No pronunciation entries found")
        return cls(entries)

    def lookup(self, query, variant_id=None):
        """Return all choices; never guess a pronunciation or silently choose a variant."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        word = normalize_word(query)
        result = {
            "query": query,
            "normalized_word": word,
            "status": "invalid_input",
            "pronunciations": [],
            "selected": None,
            "selection_required": False,
        }
        if not word or any(char.isspace() for char in word):
            result["message"] = "Supply one dictionary token; phrase tokenization belongs to integration."
            return result
        if variant_id is not None and (type(variant_id) is not int or variant_id < 0):
            result["message"] = "variant_id must be a non-negative integer or None."
            return result
        choices = self.by_word.get(word, ())
        result["pronunciations"] = [entry.as_dict() for entry in choices]
        if not choices:
            result["status"] = "oov"
            result["message"] = "This word is not in the pinned dictionary; no pronunciation is generated."
            return result
        if variant_id is not None:
            selected = next((entry for entry in choices if entry.variant_id == variant_id), None)
            if selected is None:
                result["status"] = "invalid_variant"
                result["message"] = "Requested variant does not exist; choose an available variant_id."
                return result
        else:
            selected = choices[0] if len(choices) == 1 else None
        result["status"] = "known"
        result["selected"] = selected.as_dict() if selected else None
        result["selection_required"] = selected is None
        return result
