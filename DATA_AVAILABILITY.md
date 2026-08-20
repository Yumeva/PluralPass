# Data availability and release boundary

## Public upstream football data

PluralPass uses StatsBomb Open Data with 360 freeze frames. The analysis is fixed to upstream commit `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`.

The match data are not copied into this repository. `pluralpass download` retrieves the exact revision from `statsbomb/open-data` and writes a local manifest containing the resolved commit, selected matches, file sizes and SHA-256 hashes. Users are responsible for complying with StatsBomb's terms and attribution requirements.

## Generated football-analysis artifacts

Processed graph records, split manifests, trained weights and prediction files are generated locally and ignored by Git. An archival release may deposit selected non-restricted artifacts separately; any DOI will be added only after a deposit exists.

## Human evaluation

The coach evaluation was approved by the Guangzhou Sport University ethics committee (`2024LCLL-71`), and informed consent was obtained before participation.

This public repository does **not** contain:

- names, contact details or institutional identifiers;
- signed consent forms or ethics correspondence;
- identity keys linking participant codes to people;
- raw individual-level responses or identifying free text;
- working spreadsheets carrying obsolete template labels.

Analysis utilities accept de-identified controlled exports. Aggregate or suitably anonymized source data may be released only after ethics, consent and disclosure-risk review.

The completed study was not externally preregistered before outcome access. Internal dated materials are therefore not described as prospective preregistration.

