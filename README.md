# CMUdict Guided Pronunciation to Audio

Group HCM-3 | COS40007 Artificial Intelligence Engineering | Applied Project Task 1

## Project status

A standalone data-engineering component is implemented in [`data_engineering/`](data_engineering/): pinned source verification, CMUdict parsing, single-token lookup, profiling and regression tests. TTS model selection, phone mapping, phrase tokenization, audio output, UI and group CI remain planned. The dictionary component does not synthesize audio.

## Project overview

The group plans to design a small English pronunciation tool that accepts a supported word or short phrase, looks up its pronunciation in CMUdict, and passes the resulting phone sequence to a compatible pretrained text-to-speech (TTS) system. The intended output is a playable WAV file.

CMUdict is a pronouncing dictionary, not a collection of recordings. It can provide pronunciation entries for lookup, but it cannot train an acoustic speech model by itself. A separate TTS model and vocoder will need to be selected after checking phone-set support, licensing, runtime, and hardware requirements.

## Dataset

The data component pins **CMUdict 0.7b, SVN revision 12857**, with all three source files and SHA-256 hashes recorded in [`source_lock.json`](data_engineering/source_lock.json). The dictionary must be decoded as **Latin-1** to preserve `DÉJÀ`. The parser preserves pronunciation variants and stress markers and returns an explicit OOV response rather than guessing a pronunciation.

| Verified measure | Count |
|---|---:|
| Pronunciation entries | 133,854 |
| Base spellings | 125,074 |
| Base spellings with multiple entries | 8,154 |
| Base phones / observed symbols / declared symbols | 39 / 69 / 84 |
| Audio recordings | 0 |

Counts refer to the pinned bytes; differing counts from earlier drafts must not be mixed with this snapshot. See the [data-engineering report](data_engineering/DATA_ENGINEERING_REPORT.md) for provenance, parsing rules and limitations.

- Pinned source: [CMUdict 0.7b at SVN revision 12857](https://svn.code.sf.net/p/cmusphinx/code/!svn/bc/12857/trunk/cmudict/cmudict-0.7b); [current upstream project](https://github.com/cmusphinx/cmudict).
- Data boundary: CMUdict provides pronunciations; it does not include audio recordings for model training.
- The original dictionary notice and terms are retained in the source and [`LICENSE.CMUdict.txt`](data_engineering/LICENSE.CMUdict.txt). Future model and voice assets require their own source/version and license checks.

## Run the data component

Requires **Python 3.10+** and only the standard library. From the repository root:

```bash
python data_engineering/verify_work.py
```

The three pinned files are included, so this check runs offline. It verifies their hashes, profiles the full dictionary, checks 14 lookup regression cases and runs 20 unit/integration tests. Evidence is written to [`data_engineering/outputs/`](data_engineering/outputs/). If a source file is missing, `python data_engineering/fetch_data.py` downloads only bytes matching the lock; an existing modified file is not overwritten.

For an interactive lookup from the repository root:

```bash
python data_engineering/run_analysis.py --lookup read
python data_engineering/run_analysis.py --lookup read --variant 1
```

The first returns both pronunciations without selecting one; the second selects source variant 1. The API accepts one dictionary token per call. Phrase tokenization and the choice of context-appropriate pronunciation remain integration responsibilities.

## Repository layout and handover

```text
data_engineering/
  cmudict_data.py            # parser and single-token lookup contract
  fetch_data.py             # pinned source fetch/verification
  run_analysis.py            # profiling and lookup CLI
  verify_work.py             # full offline verification
  source_lock.json           # exact revision, URLs and hashes
  source/                    # unchanged dictionary and companion tables
  tests/                     # parser and pinned-data tests
  outputs/                   # profiles, inventory, lookup evidence and test results
  DATA_ENGINEERING_REPORT.md # source, design and validation report
  GROUP_REPORT_UPDATES.md    # proposed report, contribution and Jira text
```

- **Khanh:** use the [phone inventory](data_engineering/outputs/phone_inventory.csv) and selected phone sequence to develop the model adapter. Compatibility with a particular TTS model has not been checked.
- **Duy:** use the [regression cases](data_engineering/regression_cases.json) and [lookup evidence](data_engineering/outputs/lookup_evidence.json) for integration. Do not send a pronunciation to synthesis unless `status == "known"` and `selected` is present.
- **Đăng:** [GROUP_REPORT_UPDATES.md](data_engineering/GROUP_REPORT_UPDATES.md) contains suggested report and planning updates; no external Jira or Confluence changes are claimed.

See the [component README](data_engineering/README.md) for the lookup API and detailed commands.

## Proposed processing flow

1. Accept a word or short phrase and normalize the text for lookup.
2. Look up each supported word in the pinned CMUdict version.
3. Keep stress markers and show alternate entries when the dictionary lists them.
4. Return an explicit message for an out-of-vocabulary word.
5. Map CMUdict phones to the selected TTS model's input inventory, reporting any unsupported symbols.
6. Send the mapped sequence to the selected pretrained TTS system and save the generated audio as WAV.

The model and phone mapping are still to be chosen. The first version is intended to cover dictionary-supported English words and short phrases; names, dialect variation, and context-dependent pronunciations may fall outside that coverage.

## Validation

The data component has passed 20 unit/integration tests and 14 fixed lookup regression cases. These cover source hashes and tampering, independent raw counts, malformed/duplicate entries, variants, stress, punctuation, Latin-1 decoding, OOV and invalid inputs. Actual evidence is in [verification_summary.md](data_engineering/outputs/verification_summary.md), [test_summary.json](data_engineering/outputs/test_summary.json) and [test_results.txt](data_engineering/outputs/test_results.txt).

TTS and end-to-end checks remain planned: phone mapping must account for each input symbol or fail explicitly; audio checks must verify WAV readability, duration and samples. A small, opt-in listening review may inform pronunciation/clarity assessment. No audio-quality, latency or TTS compatibility result is claimed for the data component.

## Team workflow

- **Messenger:** day-to-day group communication and blocker updates.
- **Jira:** planned backlog, priorities, sprint tasks, and review status.
- **Confluence:** shared project notes, decisions, risks, and design documentation.
- **GitHub:** source control and pull-request review of proposed code changes.

## Contributions

The first three contributions below describe the blueprint and planning submission. The data-engineering contribution also includes the standalone implementation and evidence now supplied for review.

- **Nguyễn Tuấn Khanh:** Drafted the proposed dictionary-to-audio system flow and the high-level data, model lifecycle, monitoring, release, and risk plan.
- **Nguyễn Minh Đăng:** Prepared the Jira planning content, including user stories and sprint tasks, and outlined the Confluence page for shared project documentation.
- **Lê Đức Duy:** Drafted the group working agreement and coordination plan, including role allocation, meeting cadence, Messenger updates, response expectations, and blocker resolution.
- **Vũ Lê Hoàng:** Pinned the CMUdict sources by revision and hash, documented Latin-1 and redistribution terms, implemented the parser and single-token lookup, and produced reproducible profiles, phone inventory and test evidence. The data component preserves variants and stress and handles unsupported inputs explicitly. TTS/application integration is pending.

## GenAI assistance

Generative AI assisted report revision, data-analysis scripts, parser implementation and test drafting for the data-engineering component. The source lock and automated evidence document what was checked. Each group member remains responsible for reviewing their contribution and following the unit's declaration requirements.

## References

Carnegie Mellon University Speech Group. (n.d.). *CMUdict 0.7b* [Pronouncing dictionary]. https://github.com/cmusphinx/cmudict
