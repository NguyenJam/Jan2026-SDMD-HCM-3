# CMUdict Guided Pronunciation to Audio

Group HCM-3 | COS40007 Artificial Intelligence Engineering | Applied Project Task 1

## Project status

This repository is at the system-blueprint and project-planning stage. The workflow below describes the proposed system; there is no executable implementation, selected speech model, or CI workflow yet. Setup and run instructions will be added when those components are developed.

## Project overview

The group plans to design a small English pronunciation tool that accepts a supported word or short phrase, looks up its pronunciation in CMUdict, and passes the resulting phone sequence to a compatible pretrained text-to-speech (TTS) system. The intended output is a playable WAV file.

CMUdict is a pronouncing dictionary, not a collection of recordings. It can provide pronunciation entries for lookup, but it cannot train an acoustic speech model by itself. A separate TTS model and vocoder will need to be selected after checking phone-set support, licensing, runtime, and hardware requirements.

## Dataset

The planned lookup source is CMUdict 0.7b. Its entries use ARPAbet phone symbols and stress markers, and some words have alternate pronunciations. The planned parser will preserve those alternatives rather than dropping them. Words without a dictionary entry will receive a clear unsupported-word message instead of guessed pronunciation.

- Source: [CMUdict on GitHub](https://github.com/cmusphinx/cmudict)
- Data boundary: CMUdict provides pronunciations; it does not include audio recordings for model training.
- Before redistributing dictionary files or model assets, the group will review the applicable terms and record the source and version.

## Proposed processing flow

1. Accept a word or short phrase and normalize the text for lookup.
2. Look up each supported word in the pinned CMUdict version.
3. Keep stress markers and show alternate entries when the dictionary lists them.
4. Return an explicit message for an out-of-vocabulary word.
5. Map CMUdict phones to the selected TTS model's input inventory, reporting any unsupported symbols.
6. Send the mapped sequence to the selected pretrained TTS system and save the generated audio as WAV.

The model and phone mapping are still to be chosen. The first version is intended to cover dictionary-supported English words and short phrases; names, dialect variation, and context-dependent pronunciations may fall outside that coverage.

## Planned checks

When development begins, the group intends to check that the parser handles dictionary entries, stress markers, alternate pronunciations, and unsupported words. The phone-mapping check should account for every CMUdict symbol or return a clear error. Audio checks will verify that the WAV file can be opened, has a valid duration, and contains samples. A small, opt-in listening review can provide feedback on pronunciation and clarity; it will not be treated as a broad performance claim.

These checks are plans, not completed test results.

## Team workflow

- **Messenger:** day-to-day group communication and blocker updates.
- **Jira:** planned backlog, priorities, sprint tasks, and review status.
- **Confluence:** shared project notes, decisions, risks, and design documentation.
- **GitHub:** source control and, later, review of proposed code changes.

## Planning contributions

The contributions below refer to the blueprint and planning submission; they do not describe completed software implementation.

- **Nguyễn Tuấn Khanh:** Drafted the proposed dictionary-to-audio system flow and the high-level data, model lifecycle, monitoring, release, and risk plan.
- **Nguyễn Minh Đăng:** Prepared the Jira planning content, including user stories and sprint tasks, and outlined the Confluence page for shared project documentation.
- **Lê Đức Duy:** Drafted the group working agreement and coordination plan, including role allocation, meeting cadence, Messenger updates, response expectations, and blocker resolution.
- **Vũ Lê Hoàng:** Profiled CMUdict entries, variants, and phone symbols, then connected those findings to the planned lookup, alternate-pronunciation, and unsupported-word requirements.

## References

Carnegie Mellon University Speech Group. (n.d.). *CMUdict 0.7b* [Pronouncing dictionary]. https://github.com/cmusphinx/cmudict
