# Text for the group report

These passages describe the snapshot and local evidence included in this package. They are ready to incorporate into the group report when using this snapshot. Keep other members' work and the distinction between a design and an integrated system intact.

## Replace the local data profile

We profiled CMUdict 0.7b from the official CMUSphinx Subversion repository at revision 12857. The exact source is pinned by SHA-256 checksum and decoded using Latin-1 to preserve the non-ASCII dictionary entry. The snapshot contains 133,854 pronunciation entries covering 125,074 base spellings; 8,154 base spellings have multiple entries. We observed 39 base phones and 69 phone symbols when stress digits are retained, while the companion symbol table declares 84 legal symbols. The supplied assets contain no audio recordings. These results apply to this pinned snapshot and can be reproduced with the accompanying analysis script.

| Local profile of pinned CMUdict 0.7b | Count / observation | Design implication |
|---|---:|---|
| Pronunciation entries | 133,854 | Preserve each entry, including alternate forms. |
| Base spellings after removing variant suffixes | 125,074 | Normalize lookup keys and retain source variant IDs. |
| Base words with multiple entries | 8,154 | Expose alternatives rather than overwriting entries. |
| Base phones / observed symbols including stress | 39 / 69 | Validate model phone mapping and preserve vowel stress. |
| Declared legal symbols / audio recordings | 84 / 0 | Validate the legal inventory; use a separate TTS model. |

## Add the parsing design

The data component skips `;;;` comments, separates each spelling label from its ARPAbet sequence, and removes only a trailing numeric variant suffix when constructing the lookup key. Each entry retains its original label, variant ID, phone tokens and source line. Keys are trimmed and uppercased while apostrophes, hyphens, accents and stress digits are preserved. A normalized word maps to all its entries. Duplicate word/variant keys and invalid symbols cause explicit validation errors. A word with multiple entries requires an explicit variant choice; an unsupported word returns an OOV response without a fabricated pronunciation. Phrase tokenization remains an integration responsibility.

## Update the implementation status

The project remains at the system-design stage for TTS and application integration. A standalone local data-engineering component now provides source verification, dictionary parsing, single-token lookup and reproducible data profiling. It has passed 20 unit/integration tests and 14 fixed lookup regression cases. No TTS model has been integrated or evaluated by this component, and it does not produce audio.

Use this status consistently wherever the draft currently says that no code has been written. Planned group CI workflows, TTS mapping and speech output should remain described as planned unless separately implemented by their owners.

## Replace Vũ Lê Hoàng contribution statement

Prepared a reproducible profile of CMUdict 0.7b, including pronunciation entries, alternate forms, phone inventories and data limitations. Pinned the dictionary and companion tables by source revision and checksum, documented the Latin-1 encoding requirement, and prepared a standalone parser that preserves variants and stress. Produced data and lookup evidence, including 20 passing unit/integration tests and 14 passing regression cases, and documented the handover to speech synthesis and integration. Reconciled the reproduced snapshot's counts with the differing figures in the earlier draft.

## Update the AI declaration to reflect this package

Generative AI was used to assist with user stories, report revision, data-analysis scripts, parser implementation and test drafting. The accompanying package records the pinned source files, generated statistics and automated verification results. Each group member is responsible for reviewing their submitted contribution and following the unit's declaration requirements.

## Jira task text for the data engineer

| Task | Story | Evidence / acceptance check | Local status |
|---|---|---|---|
| Pin CMUdict source and document encoding and terms | US1, US2, US5 | Three source hashes verify; source revision, encoding and copyright retained. | Completed locally |
| Profile entries, base spellings, variants and symbols | US1, US2 | Counts reproduce from the pinned bytes; observed and declared inventories distinguished. | Completed locally |
| Prepare dictionary parser and token lookup | US1, US2 | All entries preserved; explicit choices for variants; stress retained; unsupported inputs explicit. | Completed locally |
| Provide regression cases and phone inventory | US2, US3, US4 | 14 fixed cases pass; symbol occurrence CSV and response examples available. | Completed locally |
| Connect lookup to the group application and chosen TTS model | US1–US4 | Team validates tokenization, variant choice, phone adapter and audio output end to end. | Pending integration |

This file records local work only. No Jira or Confluence changes have been posted.
