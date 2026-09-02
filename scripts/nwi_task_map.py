"""Flatten per-partition batch manifests into one Slurm array task map.

The statewide run is ~62 partitions of unequal size (Lyon's batches ranged from
14 to 93 fields, a 10x spread in runtime). Submitting one array per partition
would leave the schedule hostage to whichever partition finishes last. Mapping
every (partition, batch) pair onto a single flat array index instead lets Slurm
pack the whole statewide workload against the free-partition core budget, so
the tail is one batch rather than one partition.

Writes task_map.csv with columns: task_id, label, batch_id, n_fields.

Run after the prep stage, once every partition has a batch_manifest.csv.
"""

import argparse
from pathlib import Path

import pandas as pd


def build_task_map(labels, run_root):
    """Return a DataFrame mapping array task ids to (label, batch_id)."""
    run_root = Path(run_root)
    rows = []
    for label in labels:
        manifest_path = run_root / label / "pestrun" / "batch_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No batch manifest for {label} at {manifest_path}; "
                "the prep stage has not run (or failed) for this partition."
            )
        manifest = pd.read_csv(manifest_path)
        counts = manifest.groupby("batch_id").size()
        for batch_id, n_fields in counts.items():
            rows.append({"label": label, "batch_id": int(batch_id), "n_fields": int(n_fields)})

    if not rows:
        raise ValueError("No batches found across any partition.")

    df = pd.DataFrame(rows)
    # Largest batches first: long tasks start early, so the last wave is short
    # ones and the array drains evenly instead of ending on a 93-field batch.
    df = df.sort_values("n_fields", ascending=False, ignore_index=True)
    df.insert(0, "task_id", range(len(df)))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        required=True,
        help="Text file of partition labels, one per line (blank lines and # comments ignored)",
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Directory containing <label>/pestrun/batch_manifest.csv for each label",
    )
    parser.add_argument("--out", required=True, help="Path to write task_map.csv")
    args = parser.parse_args()

    labels = [
        line.strip()
        for line in Path(args.labels).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    df = build_task_map(labels, args.run_root)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out}: {len(df)} tasks across {df['label'].nunique()} partition(s)")
    print(f"  total fields: {df['n_fields'].sum()}")
    print(f"  batch size: min {df['n_fields'].min()}, max {df['n_fields'].max()}")
    # Parsed by the submitting script to size --array=0-(N-1).
    print(f"TASK_COUNT={len(df)}")


if __name__ == "__main__":
    main()
