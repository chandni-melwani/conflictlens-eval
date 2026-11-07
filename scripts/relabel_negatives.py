import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

FORBIDDEN_BASENAMES = {
    "negative_labels.jsonl",
    "negative_labels_round1.jsonl",
    "gold.jsonl",
}


def validate_output_path(out_path: str):
    resolved = os.path.realpath(out_path)
    base = os.path.basename(resolved).lower()
    if base in FORBIDDEN_BASENAMES:
        sys.exit(f"Error: Refusing to run: output path '{out_path}' resolves to forbidden file '{base}'.")

    # Also check against paths of existing workspace forbidden files
    for forbidden in [
        "negative_labels.jsonl",
        "negative_labels_round1.jsonl",
        "gold.jsonl",
        os.path.join("data", "negative_labels.jsonl"),
        os.path.join("data", "negative_labels_round1.jsonl"),
        os.path.join("data", "gold.jsonl"),
    ]:
        if os.path.exists(forbidden):
            try:
                if os.path.samefile(resolved, forbidden):
                    sys.exit(f"Error: Refusing to run: output path resolves to forbidden file '{forbidden}'.")
            except (FileNotFoundError, OSError):
                pass
            if resolved.lower() == os.path.realpath(forbidden).lower():
                sys.exit(f"Error: Refusing to run: output path resolves to forbidden file '{forbidden}'.")


def load_and_validate_sample(sample_path: str):
    if not os.path.exists(sample_path):
        sys.exit(f"Error: Sample file not found at '{sample_path}'.")

    records = []
    with open(sample_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Stop with an error if any record has a field name containing "label"
            for key in rec.keys():
                if "label" in key.lower():
                    sys.exit(
                        f"Error: Record on line {line_num} (id: {rec.get('id')}) has forbidden field name containing 'label': '{key}'"
                    )

            # Verify records have only id, passage1, and passage2
            if set(rec.keys()) != {"id", "passage1", "passage2"}:
                sys.exit(
                    f"Error: Record on line {line_num} (id: {rec.get('id')}) has unexpected fields: {list(rec.keys())}. Must have only id, passage1, passage2."
                )

            records.append(rec)

    return records


def load_labeled_ids(out_path: str):
    labeled_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "id" in entry:
                        labeled_ids.add(entry["id"])
                except json.JSONDecodeError:
                    pass
    return labeled_ids


def main():
    parser = argparse.ArgumentParser(description="Relabel negative sample for intra-annotator reliability.")
    parser.add_argument(
        "--sample",
        default=os.path.join("data", "relabel_sample.jsonl"),
        help="Path to relabel sample jsonl",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "relabel_labels.jsonl"),
        help="Path to relabel output jsonl",
    )
    args = parser.parse_args()

    validate_output_path(args.output)
    records = load_and_validate_sample(args.sample)
    labeled_ids = load_labeled_ids(args.output)

    total_items = len(records)
    remaining_items = [r for r in records if r["id"] not in labeled_ids]

    if not remaining_items:
        print(f"All {total_items} items already labeled in '{args.output}'.")
        return

    label_map = {
        "c": "COMPATIBLE",
        "x": "CONTRADICT",
        "u": "UNSURE",
    }

    for k, item in enumerate(records, 1):
        if item["id"] in labeled_ids:
            continue

        print(f"\nitem {k} of {total_items}")
        print(f"Passage 1: {item['passage1']}")
        print(f"Passage 2: {item['passage2']}")

        t0 = time.time()
        while True:
            try:
                cmd = input("[c] COMPATIBLE, [x] CONTRADICT, [u] UNSURE, [q] quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            if cmd == "q":
                print("Exiting without saving.")
                return
            elif cmd in label_map:
                sec_spent = round(time.time() - t0, 2)
                decision = {
                    "id": item["id"],
                    "label": label_map[cmd],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "seconds_spent": sec_spent,
                }
                out_dir = os.path.dirname(args.output)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.output, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(decision, ensure_ascii=False) + "\n")
                    f_out.flush()
                labeled_ids.add(item["id"])
                break
            else:
                print("Invalid input. Keys: c COMPATIBLE, x CONTRADICT, u UNSURE, q quit")

    print(f"\nFinished labeling all {total_items} items.")


if __name__ == "__main__":
    main()

