"""Import the normalized configs/bench.md format and preserve explicit follow-ups."""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import ROOT, digest, validate_rows, write_json, write_jsonl


HEADING = re.compile(r"^(\d+\.\d+)（(.+)）$")
DIMENSION = re.compile(r"^维度[一二三四五六七八九十]+[｜|](.+)$")
PARENTS = {"1.2": "1.1", "1.4": "1.3", "1.6": "1.5", "1.7": "1.6"}
SESSIONS = {"1.1": "Q3", "1.2": "Q3", "1.3": "Q4", "1.4": "Q4",
            "1.5": "Q6", "1.6": "Q6", "1.7": "Q6"}


def parse(source):
    source = Path(source)
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[1].strip() != "Global 指令":
        raise ValueError("Missing the expected Global 指令 block.")
    global_instruction = lines[2].strip()
    rows = []
    current = None
    dimension = ""

    def finish():
        nonlocal current
        if current is None:
            return
        body = "\n".join(current.pop("lines")).strip()
        if not body:
            raise ValueError("Empty question: " + current["number"])
        number = current["number"]
        case_id = "CURRENT-" + number.replace(".", "-")
        parent = PARENTS.get(number)
        metadata = {
            "is_placeholder": False,
            "synthetic": True,
            "dimension": current["dimension"],
            "source_question_id": number,
            "source_heading": current["label"],
            "source_document": "configs/bench.md",
            "global_instruction_applied": True,
        }
        if number in SESSIONS:
            metadata["session_id"] = SESSIONS[number]
        if parent:
            metadata["parent_case_id"] = "CURRENT-" + parent.replace(".", "-")
        rows.append({
            "schema_version": "0.1.0",
            "case_id": case_id,
            "disease": "mixed_cardiovascular",
            "stage": "current_bench",
            "risk_level": "unspecified",
            "prompt_type": "neutral",
            "prompt": global_instruction + "\n\n" + body,
            "mirror_pair_id": None,
            "expected_treatment_roles": [],
            "metadata": metadata,
        })
        current = None

    for line in lines[3:]:
        stripped = line.strip()
        dim = DIMENSION.match(stripped)
        if dim:
            finish()
            dimension = dim.group(1)
            continue
        head = HEADING.match(stripped)
        if head:
            finish()
            current = {"number": head.group(1), "label": head.group(2),
                       "dimension": dimension, "lines": []}
            continue
        if current is not None:
            current["lines"].append(line)
    finish()

    validate_rows(rows, ROOT / "datasets/schemas/case_schema.json")
    ids = [row["case_id"] for row in rows]
    if len(rows) != 28 or len(ids) != len(set(ids)):
        raise ValueError(f"Expected 28 unique questions; found {len(rows)}.")
    for index, row in enumerate(rows):
        parent = row["metadata"].get("parent_case_id")
        if parent and parent not in ids[:index]:
            raise ValueError("Missing or late parent: " + parent)
    return rows, global_instruction


def import_file(source, output):
    rows, instruction = parse(source)
    output = Path(output)
    write_jsonl(output, rows)
    write_json(output.with_suffix(".audit.json"), {
        "source_sha256": digest(source),
        "question_count": len(rows),
        "global_instruction": instruction,
        "follow_up_edges": PARENTS,
        "mirror_questions_are_independent": True,
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "configs/bench.md")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/cases/current_bench.jsonl")
    args = parser.parse_args()
    rows = import_file(args.source, args.output)
    print(f"Imported {len(rows)} questions; {len(PARENTS)} follow-up edges.")


if __name__ == "__main__":
    main()
