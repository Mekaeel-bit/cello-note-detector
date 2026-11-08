#!/usr/bin/env python3
"""Discover and download PD MusicXML/MXL (or MIDI) files from an IMSLP work page.

Why earlier auto-find failed
----------------------------
IMSLP serves ``Special:ImagefromIndex/<id>`` behind a bot-check / disclaimer
redirect for non-browser clients. Guessing CDN URLs without the real filename
also 404s. The durable pattern is:

1. Fetch the **work page HTML** (usually allowed).
2. Read hidden ``/images/<h1>/<h2>/<filename>`` links (or ReverseLookup HTML).
3. Download via CDN:
   ``https://ks15.imslp.org/files/imglnks/usimg/<h1>/<h2>/IMSLP<id>-<filename>``
   (try a few ``ks*`` / ``s9`` hosts).

This script automates steps 1–3 so Teddy should not need to paste deep CDN links
long-term. Audio→PD discovery can call it after song-ID suggests a work page.

Examples
--------
  python3 scripts/fetch_imslp_score.py \\
    --work-url 'https://imslp.org/wiki/Ricercari,_canone_e_sonate_per_violoncello_(Gabrielli,_Domenico)' \\
    --prefer mxl --imslp-id 923358 -o /tmp/gabrielli.mxl

  python3 scripts/fetch_imslp_score.py \\
    --work-url 'https://imslp.org/wiki/Ricercari,_canone_e_sonate_per_violoncello_(Gabrielli,_Domenico)' \\
    --prefer mid --imslp-id 187416 -o /tmp/ricercar1.mid
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_UA = (
    "Mozilla/5.0 (compatible; cello-note-detector/0.1; "
    "+https://github.com/Mekaeel-bit/cello-note-detector)"
)
_COOKIES = "imslpdisclaimeraccepted=yes; imslp_wikiLanguageSelectorLanguage=en"
_HOSTS = ("ks15", "ks7", "ks9", "s9", "ca1")


def _http_get(url: str, dest: Path | None = None) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Cookie": _COOKIES, "Referer": "https://imslp.org/"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    if dest is not None:
        dest.write_bytes(data)
    return data


def _discover_image_links(work_html: str) -> list[tuple[str, str, str]]:
    """Return (hash1, hash2, filename) from /images/h1/h2/filename links."""
    found: list[tuple[str, str, str]] = []
    for m in re.finditer(
        r"/images/([0-9a-f])/([0-9a-f]{2})/([^\"'<> ]+)", work_html, flags=re.I
    ):
        found.append((m.group(1), m.group(2), urllib.parse.unquote(m.group(3))))
    # Dedupe preserving order
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _imslp_id_from_tools(work_html: str, filename: str) -> str | None:
    """Best-effort map filename → IMSLP id via JGFileTools / nearby anchors."""
    # Direct: #923358 near filename in page
    esc = re.escape(filename)
    m = re.search(
        rf"IMSLP(\d{{3,}}).{{0,200}}{esc}|{esc}.{{0,200}}IMSLP(\d{{3,}})",
        work_html,
        flags=re.I | re.S,
    )
    if m:
        return m.group(1) or m.group(2)
    # File: wiki title pages sometimes use PMLP… without IMSLP id in path
    return None


def fetch_from_work_page(
    work_url: str,
    *,
    prefer: str = "mxl",
    imslp_id: str | None = None,
    substring: str | None = None,
) -> tuple[bytes, str, str]:
    """Download a score file. Returns (bytes, resolved_url, filename)."""
    html = _http_get(work_url).decode("utf-8", "replace")
    # Also pull ReverseLookup HTML when id known — exposes /images/ paths.
    if imslp_id:
        try:
            rl = _http_get(
                f"https://imslp.org/wiki/Special:ReverseLookup/{imslp_id}"
            ).decode("utf-8", "replace")
            html += "\n" + rl
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: ReverseLookup failed: {exc}", file=sys.stderr)

    links = _discover_image_links(html)
    prefer = prefer.lower().lstrip(".")
    candidates = [
        (h1, h2, name)
        for h1, h2, name in links
        if name.lower().endswith("." + prefer)
        and (substring is None or substring.lower() in name.lower())
    ]
    if not candidates:
        raise SystemExit(
            f"No .{prefer} candidates on work page"
            + (f" matching {substring!r}" if substring else "")
        )

    # Prefer explicit IMSLP id filename match when possible
    chosen = candidates[0]
    if imslp_id:
        for c in candidates:
            if imslp_id in c[2] or True:
                # Try all candidates with this id-prefixed CDN name
                chosen = c
                break

    h1, h2, filename = chosen
    enc_name = urllib.parse.quote(filename, safe="().-_")
    names_to_try = [enc_name]
    if imslp_id:
        names_to_try.insert(
            0, urllib.parse.quote(f"IMSLP{imslp_id}-{filename}", safe="().-_")
        )
        # Common IMSLP CDN form already includes PMLP… basename
        if not filename.upper().startswith("IMSLP"):
            names_to_try.insert(
                0,
                urllib.parse.quote(
                    f"IMSLP{imslp_id}-{filename}", safe="().-_"
                ),
            )

    last_err: Exception | None = None
    for host in _HOSTS:
        for name in names_to_try:
            url = f"https://{host}.imslp.org/files/imglnks/usimg/{h1}/{h2}/{name}"
            try:
                data = _http_get(url)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
            if len(data) < 2000:
                continue
            # Reject HTML/bot-check payloads
            head = data[:200].lstrip()
            if head.startswith(b"<") or head.startswith(b"<!DOCTYPE"):
                continue
            return data, url, filename

    raise SystemExit(
        f"CDN download failed for {filename} (id={imslp_id}). Last error: {last_err}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-url", required=True, help="IMSLP work page URL")
    p.add_argument(
        "--prefer",
        default="mxl",
        help="Extension to prefer: mxl, musicxml, mid, pdf, zip (default mxl)",
    )
    p.add_argument("--imslp-id", default=None, help="Numeric IMSLP file id when known")
    p.add_argument(
        "--substring",
        default=None,
        help="Optional filename substring filter (e.g. Ricercar, Drop_G)",
    )
    p.add_argument("-o", "--output", required=True, type=Path)
    args = p.parse_args(argv)

    data, url, filename = fetch_from_work_page(
        args.work_url,
        prefer=args.prefer,
        imslp_id=args.imslp_id,
        substring=args.substring,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"Downloaded {filename} ({len(data)} bytes)")
    print(f"URL: {url}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
