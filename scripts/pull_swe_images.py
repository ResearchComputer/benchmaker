#!/usr/bin/env python3
"""Write public GHCR image refs from the swe-images GitHub Packages org."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API = "https://api.github.com"
DEFAULT_ORG = "swe-images"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / ".local" / "images.txt"


class GitHubAPIError(RuntimeError):
    def __init__(self, status: int, url: str, detail: str):
        super().__init__(f"GitHub API request failed: HTTP {status} {url}\n{detail}")
        self.status = status


def github_json(url: str, token: str | None) -> tuple[Any, str | None]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "benchmaker-swe-image-list",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
            return json.loads(body), resp.headers.get("Link")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise GitHubAPIError(e.code, url, detail) from e


def github_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "benchmaker-swe-image-list"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"GitHub page request failed: HTTP {e.code} {url}\n{detail}") from e


def next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        url_part, _, rel_part = part.strip().partition(";")
        if 'rel="next"' in rel_part:
            return url_part.strip()[1:-1]
    return None


def paged_json(url: str, token: str | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    while url:
        page, link = github_json(url, token)
        if not isinstance(page, list):
            raise SystemExit(f"expected list from GitHub API, got {type(page).__name__}")
        items.extend(page)
        url = next_link(link)
    return items


def api_package_names(org: str, token: str | None) -> list[str]:
    url = f"{API}/orgs/{org}/packages?package_type=container&per_page=100"
    packages = paged_json(url, token)
    names = [str(pkg["name"]) for pkg in packages if "name" in pkg]
    return sorted(set(names))


def scrape_package_names(org: str) -> list[str]:
    names: set[str] = set()
    total_pages = 1
    page = 1
    pattern = re.compile(rf'href="/orgs/{re.escape(org)}/packages/container/package/([^"?#]+)')
    while page <= total_pages:
        url = f"https://github.com/orgs/{org}/packages?page={page}"
        text = github_html(url)
        for match in pattern.finditer(text):
            names.add(html.unescape(urllib.parse.unquote(match.group(1))))
        total_match = re.search(r'data-total-pages="(\d+)"', text)
        if total_match:
            total_pages = int(total_match.group(1))
        if page == 1 and not total_match and not names:
            break
        page += 1
    return sorted(names)


def package_tags(org: str, package: str, token: str | None) -> list[str]:
    encoded = urllib.parse.quote(package, safe="")
    url = f"{API}/orgs/{org}/packages/container/{encoded}/versions?per_page=100"
    versions = paged_json(url, token)
    tags: set[str] = set()
    for version in versions:
        metadata = version.get("metadata") if isinstance(version, dict) else None
        container = metadata.get("container") if isinstance(metadata, dict) else None
        raw_tags = container.get("tags") if isinstance(container, dict) else None
        if isinstance(raw_tags, list):
            tags.update(str(tag) for tag in raw_tags if tag)
    return sorted(tags)


def api_image_refs(org: str, token: str | None) -> list[str]:
    refs: list[str] = []
    for package in api_package_names(org, token):
        tags = package_tags(org, package, token)
        refs.extend(f"ghcr.io/{org}/{package}:{tag}" for tag in tags)
    return sorted(set(refs))


def scraped_image_refs(org: str) -> list[str]:
    return [f"ghcr.io/{org}/{package}:latest" for package in scrape_package_names(org)]


def image_refs(org: str, token: str | None) -> list[str]:
    try:
        return api_image_refs(org, token)
    except GitHubAPIError as e:
        if e.status != 401 or token:
            raise SystemExit(str(e)) from e
        print(
            "GitHub Packages API requires authentication; falling back to public "
            "HTML package listing and assuming :latest tags.",
            file=sys.stderr,
        )
        return scraped_image_refs(org)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch ghcr.io image refs from an org's public container packages."
    )
    parser.add_argument("--org", default=DEFAULT_ORG, help=f"GitHub org (default: {DEFAULT_ORG})")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"output text file (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GHCR_TOKEN"),
        help="optional GitHub token; defaults to GITHUB_TOKEN/GHCR_TOKEN",
    )
    args = parser.parse_args(argv)

    refs = image_refs(args.org, args.token)
    if not refs:
        raise SystemExit(f"no image refs found for org {args.org!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(refs) + "\n")
    print(f"wrote {len(refs)} image refs to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
