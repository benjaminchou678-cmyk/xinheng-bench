import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected a configuration object: " + str(path))
    return value


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path=None):
    config = read_yaml(Path(path) if path else ROOT / "configs/benchmark.yaml")
    if config.get("phase") not in (0, "pilot"):
        raise ValueError("Supported phases: 0 or pilot.")
    return config


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be 1–80 letters, digits, hyphens or underscores.")
    return run_id


def artifact(config, category, run_id, filename):
    validate_run_id(run_id)
    return project_path(config["output_dir"]) / category / run_id / filename


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def validate_rows(rows, schema_path):
    schema = read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for index, row in enumerate(rows, 1):
        errors = list(validator.iter_errors(row))
        if errors:
            raise ValueError(f"{schema_path.name}: row {index}: {errors[0].message}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
