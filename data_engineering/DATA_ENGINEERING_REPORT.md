# CMUdict data profile and parsing design

Data engineer: Vũ Lê Hoàng (104774231)<br>
Group: HCM-3<br>
Unit: COS40007 Artificial Intelligence Engineering<br>
Analysis date: 27 October 2026

The pronunciation component can use the pinned CMUdict 0.7b snapshot to return English dictionary pronunciations while preserving alternatives and stress. The verified snapshot contains **133,854 pronunciation entries for 125,074 base spellings**. It contains pronunciation text and no audio. A local parser and regression suite now provide reproducible evidence for the data design; a TTS model and application interface have not been integrated in this package.

## Data source and reproducibility

The dictionary and its companion phone and symbol tables were downloaded from the official CMUSphinx Subversion repository at **revision 12857**. Each URL embeds that revision. `source_lock.json` records the three URLs and SHA-256 checksums. The dictionary is 3,716,714 bytes and has SHA-256:

```text
209a8b4cd265013e96f4658632a9878103b0c5abf62b50d4ef3ae1be226b29e4
```

The analysis verifies all source checksums before parsing. The source bytes remain unchanged, including the copyright notice and redistribution conditions. `LICENSE.CMUdict.txt` reproduces the source header. The other assets in a future speech system, including model weights and voices, require their own license review.

This snapshot must be decoded as **Latin-1**. Strict UTF-8 decoding fails, and the entry `DÉJÀ` contains non-ASCII bytes. Latin-1 preserves that spelling without replacement characters. This is a property of the pinned file, not a claim that every CMUdict distribution uses this encoding.

## Verified data profile

| Measure | Result | Design implication |
|---|---:|---|
| Pronunciation entries | 133,854 | Retain every source pronunciation row. |
| Base spellings after removing numeric variant suffixes | 125,074 | Index by normalized base spelling. |
| Base spellings with multiple entries | 8,154 | Return alternatives for explicit user selection. |
| Alternate entries with numeric suffixes | 8,780 | Store the source variant ID separately. |
| Observed base phones | 39 | Require an explicit mapping or error path for each base phone. |
| Observed phone symbols including stress digits | 69 | Preserve vowel stress during lookup and model adaptation. |
| Symbols declared in the companion table | 84 | Distinguish the legal inventory from observed tokens. |
| Audio recordings | 0 | Use a separate pretrained synthesis model. |
| Phone sequence length | 1–32 | Avoid assuming a constant pronunciation length. |
| Maximum entries for one spelling | 4 | The interface must allow more than two choices. |

The 84 declared symbols comprise 24 consonants and 15 vowels in four forms: the bare vowel and forms with stress digits 0, 1 and 2. The dictionary uses the 45 stressed vowel forms and 24 consonants, giving 69 observed symbols. The 15 bare vowel forms are legal according to the companion table but are not observed in this file. Stress 0 denotes no lexical stress, 1 primary stress and 2 secondary stress.

| Entries per base spelling | Number of base spellings |
|---|---:|
| 1 | 116,920 |
| 2 | 7,672 |
| 3 | 338 |
| 4 | 144 |

All 133,910 source lines are accounted for: 56 comment lines and 133,854 entries. Parsing found no malformed rows, duplicate normalized word/variant keys or symbols outside the declared inventory. There were no repeated phone sequences within the same base spelling. These checks establish structural consistency; they do not establish that every pronunciation is linguistically correct.

## Parsing and lookup design

1. Verify the dictionary, phone-table and symbol-table hashes against the lock file.
2. Decode the dictionary using Latin-1. Skip blank lines and lines beginning with `;;;`. Do not treat `#` as a comment prefix: `#HASH-MARK` is a real dictionary label.
3. Split a data row into its label and whitespace-separated phone tokens. Reject missing pronunciations and unsupported symbols with a source line number.
4. Remove only a trailing positive numeric variant suffix, such as `(1)`, from the lookup spelling. Keep `raw_label`, the numeric `variant_id`, the phone sequence and `source_line`. Use ID 0 for an unsuffixed entry; this is a local API convention.
5. Normalize lookup keys by trimming surrounding whitespace and uppercasing. Preserve apostrophes, hyphens, accents and punctuation. No accent removal, spelling correction or pronunciation guessing occurs.
6. Index each word to a collection of entries. Reject duplicate word/variant keys rather than silently overwriting a row. Preserve stress tokens such as `AH0`, `EH1` and `AA2` exactly.

For example, the source contains:

```text
READ     R EH1 D
READ(1)  R IY1 D
```

`lookup("read")` returns both entries with `selection_required: true` and `selected: null`. `lookup("read", variant_id=1)` selects `R IY1 D`. Variant IDs identify source entries; they do not identify tense or meaning, and the dictionary does not resolve sentence context.

The API returns `known`, `oov`, `invalid_input` or `invalid_variant`. An unknown word returns no pronunciation. A nonexistent variant never falls back to another entry. A word with one entry can be selected automatically; a word with multiple entries requires a choice. The synthesizer should receive a pronunciation only when `status` is `known` and `selected` is present.

## Scope and quality boundaries

This component accepts **one dictionary token per call**. Phrase segmentation, handling punctuation in prose, expanding numbers and deciding whether a whole phrase should fail on one OOV token belong to integration. Internal whitespace is rejected by this lookup API so those policies remain explicit. Dictionary labels can themselves contain punctuation, so integration should preserve contractions and hyphenated words when appropriate.

The dictionary primarily covers US English and cannot guarantee coverage of names, dialects or new vocabulary. It cannot supply recordings, duration, speaker characteristics or a complete prosody model, and cannot train an acoustic TTS model on its own. There is no acoustic training/validation/test split for this package. If regression cases are later split, keep variants of the same base spelling in the same partition.

One candidate source anomaly illustrates the limits of schema validation: `#HASH-MARK` has `HH AE1 M AA2 R K`, with no `SH` token. The parser faithfully retains this sequence. It should be reviewed as a possible dictionary error before using that entry in a demonstration. Corrections should be recorded as separate, versioned patches and should not alter the locked original.

## Validation and handover

`python verify_work.py` runs the full analysis, 14 fixed lookup cases and 20 unit/integration tests. The cases cover known words, capitalization, variants, stress, OOV, invalid inputs, punctuation, a contraction and the Latin-1 entry. Tests also check source tampering, duplicate keys, malformed rows, invalid stress, independent raw-file counts and preservation of every source row. Actual results are saved in `outputs/test_results.txt`, `outputs/test_summary.json` and `outputs/lookup_evidence.json`.

The handover to Nguyễn Tuấn Khanh consists of the 39 base phones, 69 observed stress-bearing symbols, 84 declared symbols, their occurrence counts and the selected-pronunciation API. Khanh still needs to validate compatibility with the chosen TTS model. Lê Đức Duy receives the lookup response contract and regression cases for integration and QA. No audio-quality, latency or TTS compatibility result is claimed here.

The original group draft lists 134,373 entries, 125,770 base spellings and 8,016 spellings with multiple entries. The originally profiled file was not present in this workspace, so its identity and those counts could not be reproduced. The group should use the values above if it adopts this locked snapshot, or provide and separately profile its original bytes. The dataset name alone is insufficient to reconcile the difference.

## Sources

- [CMUSphinx CMUdict 0.7b, SVN revision 12857](https://svn.code.sf.net/p/cmusphinx/code/!svn/bc/12857/trunk/cmudict/cmudict-0.7b). The raw file supplies the pronunciations, stress examples, copyright and terms.
- [Companion phone table](https://svn.code.sf.net/p/cmusphinx/code/!svn/bc/12857/trunk/cmudict/cmudict-0.7b.phones) and [symbol table](https://svn.code.sf.net/p/cmusphinx/code/!svn/bc/12857/trunk/cmudict/cmudict-0.7b.symbols). The inventory was checked against both files.
- [CMUSphinx announcement of CMUdict 0.7b](https://cmusphinx.github.io/2014/11/cmudict-0-7b-update/). Background on the release and dictionary coverage limitations.
