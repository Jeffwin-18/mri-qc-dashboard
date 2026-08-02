"""
Regenerate the labeled reference feature CSV from raw volumes.

Use this once you have the raw .hdr/.img (or .nii/.nii.gz) volumes that
correspond to a labeled dataset (e.g. the ones that originally produced
final_artifact_features.csv). It re-runs the *current* feature extraction
pipeline — including the new BiasQuadrantRange / BiasGradientMagnitude
features — so the output CSV can be dropped straight into the dashboard's
"Reference (labeled) data" slot to retrain with the richer feature set.

Requires a manifest CSV mapping each scan to its label:

    filename,Artifact,Level
    subj001_blur_low,blur,low
    subj001_bias_high,bias,high
    subj002_original,original,none
    ...

`filename` should match the .hdr/.nii(.gz) filename stem (no extension).
Any extra manifest columns (e.g. MRI_ID) are carried through untouched.

Usage:
    python regenerate_reference_csv.py \
        --volumes-dir /path/to/raw/volumes \
        --manifest /path/to/manifest.csv \
        --output final_artifact_features_v2.csv \
        --max-slices 64
"""

import argparse
import sys

import pandas as pd

from feature_extraction import find_hdr_img_pairs, extract_scan_features


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes-dir", required=True, help="Folder containing raw .hdr/.img or .nii(.gz) volumes")
    ap.add_argument("--manifest", required=True, help="CSV with filename,Artifact,Level (+ any extra columns)")
    ap.add_argument("--output", default="reference_features_regenerated.csv")
    ap.add_argument("--max-slices", type=int, default=64)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if "filename" not in manifest.columns or "Artifact" not in manifest.columns:
        print("Manifest must have at least 'filename' and 'Artifact' columns.", file=sys.stderr)
        sys.exit(1)
    manifest["_key"] = manifest["filename"].astype(str).str.lower()

    pairs, warnings = find_hdr_img_pairs(args.volumes_dir)
    for w in warnings:
        print(f"[warn] {w}")

    pair_by_name = {p["name"].lower(): p for p in pairs}

    rows = []
    missing, failed = [], []
    for _, row in manifest.iterrows():
        key = row["_key"]
        pair = pair_by_name.get(key)
        if pair is None:
            missing.append(row["filename"])
            continue
        try:
            feats = extract_scan_features(pair["hdr"], max_slices=args.max_slices)
        except Exception as e:
            failed.append(f"{row['filename']}: {e}")
            continue
        clean = {k: v for k, v in feats.items() if not k.startswith("_")}
        out_row = {**row.drop("_key").to_dict(), **clean}
        rows.append(out_row)
        print(f"  extracted: {row['filename']}")

    if missing:
        print(f"\n{len(missing)} manifest entries had no matching volume in --volumes-dir:")
        for m in missing:
            print(f"  - {m}")
    if failed:
        print(f"\n{len(failed)} volumes failed to process:")
        for f in failed:
            print(f"  - {f}")

    if not rows:
        print("\nNo scans were successfully processed — nothing written.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.output}")
    print("Columns:", list(out_df.columns))


if __name__ == "__main__":
    main()
