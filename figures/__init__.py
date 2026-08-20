"""Submission figure source package (Python only)."""

import matplotlib

# Figure sources are batch-rendered in CI and archival environments without a display server.
matplotlib.use("Agg", force=True)
