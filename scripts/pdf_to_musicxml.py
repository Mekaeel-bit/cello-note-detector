#!/usr/bin/env python3
"""Convert an engraved PD IMSLP PDF → MusicXML via Audiveris OMR.

Prefer **clean engraved / typeset** single-movement PDFs (Hamrefors Caprice
No.2, modern MuseScore exports). Manuscripts and complete Caprices books are
rejected by default — MATCH should attach only the played movement/clip.

Examples
--------
  # Capriccio No.2 (Hamrefors IMSLP #618567) — 2-page engraved PDF
  python3 scripts/fetch_imslp_score.py \\
    --work-url "https://imslp.org/wiki/11_Capricci_for_Cello_(Dall'Abaco,_Joseph)" \\
    --prefer pdf --imslp-id 618567 --substring Caprice_No._2 \\
    -o /tmp/caprice2.pdf

  python3 scripts/pdf_to_musicxml.py /tmp/caprice2.pdf -o /tmp/caprice2_omr

  # Then register in the local catalog (any cello PD piece):
  python3 scripts/add_catalog_piece.py \\
    --id dallabaco_capriccio2 \\
    --musicxml /tmp/caprice2_omr/caprice2.musicxml \\
    --title "Dall'Abaco Capriccio No.2 in G minor" \\
    --composer "Joseph Marie Clément Dall'Abaco" \\
    --edition "Hamrefors 2020 IMSLP #618567 via Audiveris OMR" \\
    --license "Public Domain (composition); engraving non-commercial copying welcome" \\
    --imslp "https://imslp.org/wiki/11_Capricci_for_Cello_(Dall'Abaco,_Joseph)" \\
    --key "G minor" --typical-duration-sec 176 --min-confidence 0.35

Requires Audiveris 5.x (``/opt/audiveris/bin/Audiveris`` or ``AUDIVERIS_BIN``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cello_note_detector.identify.omr import (  # noqa: E402
    assess_pdf_quality,
    find_audiveris,
    pdf_to_musicxml,
)
from cello_note_detector.identify.musicxml_edition import (  # noqa: E402
    parse_musicxml_fingerprint,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", type=Path, help="Engraved PD PDF (not a manuscript)")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for .musicxml / .mxl / Audiveris log",
    )
    p.add_argument("--stem", default=None, help="Output basename (default: PDF stem)")
    p.add_argument(
        "--allow-manuscript",
        action="store_true",
        help="Override manuscript rejection (OMR quality will be poor)",
    )
    p.add_argument(
        "--audiveris",
        default=None,
        help="Path to Audiveris binary (else AUDIVERIS_BIN / PATH / /opt/…)",
    )
    p.add_argument(
        "--fingerprint",
        action="store_true",
        help="Also write fingerprint.json beside the MusicXML",
    )
    args = p.parse_args(argv)

    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")

    quality = assess_pdf_quality(pdf)
    print(f"PDF quality: score={quality.score:.2f} engraved={quality.engraved_likely} "
          f"manuscript={quality.manuscript_likely} collection={quality.collection_likely}")
    print(f"  reasons: {', '.join(quality.reasons) or '(none)'}")

    bin_path = find_audiveris(args.audiveris)
    if bin_path is None:
        raise SystemExit(
            "Audiveris not found. Install from "
            "https://github.com/Audiveris/audiveris/releases "
            "(Ubuntu .deb) or set AUDIVERIS_BIN."
        )
    print(f"Audiveris: {bin_path}")

    result = pdf_to_musicxml(
        pdf,
        args.output_dir,
        audiveris_bin=bin_path,
        stem=args.stem,
        allow_manuscript=args.allow_manuscript,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if result.status != "ok" or not result.musicxml_path:
        return 1

    if args.fingerprint:
        fp = parse_musicxml_fingerprint(Path(result.musicxml_path))
        fp_path = Path(args.output_dir) / f"{Path(result.musicxml_path).stem}_fingerprint.json"
        fp_path.write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")
        print(f"Fingerprint: {fp_path} ({fp.get('num_notes')} notes, "
              f"{fp.get('num_measures')} measures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
