# swe_images

Mirror the per-instance container images for SWE-bench and R2E-Gym to
`ghcr.io/<org>` (default org: **`swe-images`**) and make them publicly pullable.

Supported datasets:

| `--dataset`          | HuggingFace source                | image source                                          |
|----------------------|-----------------------------------|-------------------------------------------------------|
| `swe-bench-verified` | `SWE-bench/SWE-bench_Verified`    | `docker.io/swebench/sweb.eval.x86_64.<id>:latest`     |
| `swe-bench`          | `SWE-bench/SWE-bench`             | `docker.io/swebench/sweb.eval.x86_64.<id>:latest`     |
| `r2e-gym-v1`         | `R2E-Gym/R2E-Gym-V1`             | `docker.io/<row.docker_image>` (e.g. `namanjain12/…`) |
| `r2e-gym-lite`       | `R2E-Gym/R2E-Gym-Lite`           | `docker.io/<row.docker_image>`                        |

Target naming drops the source registry + first namespace component, so
overlapping datasets collapse onto the same target and de-duplicate:

```
swebench/sweb.eval.x86_64.<id>:latest -> ghcr.io/swe-images/sweb.eval.x86_64.<id>:latest
namanjain12/aiohttp_final:<commit>    -> ghcr.io/swe-images/aiohttp_final:<commit>
```

## ⚠️ Scale

These are full benchmark image sets — **thousands of multi-GB images**
(SWE-bench full ≈ 2.3k instances, R2E-Gym-V1 ≈ 8k+). A complete mirror moves
**terabytes**. Always smoke-test with `--limit` first, and prefer the `skopeo`
engine (registry→registry, no local disk).

## Requirements

- A copy engine — auto-detected in this order:
  - `crane` (`go-containerregistry`) — registry→registry, no daemon, **and** can
    set labels (required for `--repo-link`). `go install github.com/google/go-containerregistry/cmd/crane@latest`
  - `skopeo` — registry→registry, no daemon, but **cannot** set labels.
  - `docker` — needs a running daemon + local disk for the largest image.
- `pip install datasets swebench` (already in this repo's deps).
- A GitHub token with `write:packages` (+ `read:packages` for the visibility
  check). Source images on Docker Hub are public — no pull auth needed.

```bash
export GHCR_USER=<your-github-username>
export GHCR_TOKEN=<PAT with write:packages,read:packages>
```

## Usage

```bash
cd tools/swe_images

# 1. Dry run — list targets, write the source→ghcr mapping, copy nothing
python publish.py --dataset swe-bench-verified --dry-run --mapping-out map.json

# 2. Smoke test — mirror 5 images, then verify they're public
python publish.py --dataset swe-bench-verified --limit 5

# 3. Full mirror, 6 concurrent copies, resumable
python publish.py --dataset all --jobs 6

# 4. Re-check visibility only (no copying)
python publish.py --dataset all --verify-only

# 5. Mirror AND link every image to the GitHub repo swe-images/swe-images
python publish.py --dataset all --repo-link swe-images/swe-images
```

## Linking images to a repository

`--repo-link swe-images/swe-images` (or a full URL) stamps the
`org.opencontainers.image.source` label onto each image config. GHCR reads that
label and **automatically links** the package to the repo — the package page
then shows the repo's README/source, and the package can inherit the repo's
access settings. This is the only programmatic way to link a package; there is
no link-an-existing-package API.

- Labeling rewrites the image config, so it needs `crane` (preferred) or
  `docker`; `skopeo` cannot and is skipped when `--repo-link` is set.
- Images already mirrored **without** the label are re-pushed when you add
  `--repo-link` (the manifest records the link, so a label-less entry no longer
  counts as done). `--skip-existing` is ignored for linked targets since a
  present image may lack the label.

Runs are **resumable**: every success is appended to `.local/manifest.jsonl`
and re-runs skip recorded targets. `--skip-existing` also probes the
destination registry so a fresh checkout still avoids re-pushing.

## Making the images public

This is the hard part, and the constraints are GitHub's, not ours. Findings
([docs](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility),
[bulk discussion #55094](https://github.com/orgs/community/discussions/55094)):

1. **A freshly pushed ghcr package is always private**, regardless of any org
   setting. There is no "push it public" path.
2. **No working API for org-owned packages.** The undocumented
   `PATCH /orgs/{org}/packages/container/{name}` with `{"visibility":"public"}`
   returns **404 for organizations** (it only works for *personal*-account
   packages via `/user/packages/...`). `gh api` hits the same 404.
3. **Linking to a repo does _not_ make a package public.** A package linked via
   the `org.opencontainers.image.source` label (our `--repo-link`) inherits the
   repo's *access permissions* but **not its visibility** — a public linked repo
   still leaves the package private.
4. The org **"Package creation"** setting only controls which visibilities are
   *allowed*; it does **not** change the default or flip existing packages.

### Recommended: mirror from a GitHub Actions workflow (`--set-public`)

The key, from [SO 77092191](https://stackoverflow.com/questions/77092191/use-github-to-change-visibility-of-ghcr-io-package):
the undocumented visibility PATCH **404s for packages pushed with an external
PAT** (they're org-owned and not connected to any repo). Pushing instead with a
workflow's `${{ secrets.GITHUB_TOKEN }}` **connects each package to that repo**
and gives the token admin on it — the configuration where the flip can succeed.

So the reliable, scalable recipe:

1. **One-time prerequisite (org owner):** enable public packages —
   *Org → Settings → Packages → "Package creation" → check **Public***.
   This only *allows* public; it does **not** make pushes public by itself.
2. **Run the mirror from inside the `swe-images/swe-images` repo** via Actions,
   authenticating to ghcr with `GITHUB_TOKEN` (not a PAT), and pass
   `--set-public`. The script pushes, then best-effort PATCHes each private
   package to public and reports per-package status. A ready-to-copy workflow is
   in [`mirror.github-actions.yml`](./mirror.github-actions.yml).

```bash
# locally, the same flag works if your token has admin on repo-connected packages:
python publish.py --dataset swe-bench-verified --limit 5 --set-public
```

### Fallbacks if a package stays private

- **Per-package UI:** package → *Package settings → Danger Zone → Change
  visibility → Public* (public → private is **not** reversible).
- **Bulk via an authenticated browser session** (internal web endpoint, CSRF +
  `user_session` cookie, not a PAT) — unofficial and brittle; ask if you want a
  helper.

### Caveats

- **Container packages are always granular (org/user-owned), never
  repo-scoped** — so visibility is **never** inherited from a linked repo, even
  via `GITHUB_TOKEN` or the `--repo-link` label. Linking inherits *access
  permissions* only. The repo connection matters here because it's what lets
  `--set-public` target the package, **not** because it changes visibility.
- The script always finishes with a visibility check
  (`GET /orgs/{org}/packages/container/{name}`, needs `read:packages`) and lists
  everything still private, so you know exactly what's left.

## Files

- `publish.py` — CLI: enumerate → copy (crane/skopeo/docker) → link → verify → `--set-public`.
- `sources.py` — per-dataset image enumeration and source→target naming.
- `mirror.github-actions.yml` — drop-in workflow for the `swe-images/swe-images` repo.
