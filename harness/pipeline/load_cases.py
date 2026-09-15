from pipeline.common import project_path, read_jsonl, validate_rows


def load_cases(config):
    rows = read_jsonl(project_path(config["cases"]))
    case_limit = config.get("case_limit")
    if case_limit is not None:
        if not isinstance(case_limit, int) or isinstance(case_limit, bool) or case_limit < 1:
            raise ValueError("case_limit must be a positive integer.")
        rows = rows[:case_limit]
    validate_rows(rows, project_path(config["case_schema"]))
    identifiers = [row["case_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate case_id in dataset.")
    if config.get("phase") == 0 and any(not row["metadata"]["is_placeholder"] for row in rows):
        raise ValueError("Phase 0 accepts only explicitly marked placeholder cases.")
    if config.get("phase") == "pilot":
        pilot_case_limit = config.get("pilot_case_limit", 6)
        if pilot_case_limit is not None:
            if not isinstance(pilot_case_limit, int) or isinstance(pilot_case_limit, bool) or pilot_case_limit < 1:
                raise ValueError("pilot_case_limit must be a positive integer or null.")
            if len(rows) > pilot_case_limit and config.get("dataset_type") != "a4_exploration":
                raise ValueError(f"Initial pilot is limited to {pilot_case_limit} cases.")
        if any(row["metadata"].get("synthetic") is not True for row in rows):
            raise ValueError("Pilot cases must explicitly declare synthetic: true.")
    return rows
