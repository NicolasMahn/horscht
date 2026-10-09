# Horscht v0.1 release

The release uses Qwen3-4B, with the final transformer block and a 265-output
linear decision head trained. See [the paper](../docs/paper/paper.md),
[model card](model/README.md), and [dataset card](dataset/README.md).

Git stores code, documentation, notices, provenance and aggregate evidence.
The 417.9 MB model package and approximately 128.4 MB dataset package are
prepared locally for Hugging Face. They contain no external benchmark items
or sealed internal test/calibration rows.

Verify the prepared payloads without uploading:

```sh
.venv/bin/python scripts/publish_release.py --dry-run
```

Authenticate locally with a repository-write token, then publish both packages
under the authenticated account (or supply `--namespace ORGANIZATION`):

```sh
.venv/bin/hf auth login
.venv/bin/python scripts/publish_release.py
```

The publisher creates public model `horscht-qwen3-4b-v0.1` and dataset
`horscht-decisions-v0.1` repositories. Use `--private` to create private
repositories instead. It uploads only an explicit file allowlist, checks local
artifact hashes first, and records remote commit revisions and local file
checksums in `publication-receipt.json`. It never uploads from the raw data or
benchmark directories. An existing repository's visibility is not changed.

Do not describe prepared artifacts as uploaded until that receipt records both
remote commits. GitHub's existing repository remains private; pushing this
release branch does not change its visibility.
