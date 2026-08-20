# Data directory

Only metadata and empty directory markers are tracked in Git.

- `raw/`: version-pinned StatsBomb JSON files downloaded by `pluralpass download`;
- `interim/`: temporary, non-authoritative transformation outputs;
- `processed/`: pass graphs, audit tables and split manifests;
- `metadata/`: schema and feature definitions safe for public release.

Raw and processed match data are ignored because they originate from a third-party dataset. Human-participant records are not stored under this directory. See [`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md).
