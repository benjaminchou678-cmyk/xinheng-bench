from collections import Counter


def summarize(annotations):
    """Operational counts only. Never interpret missing scores as zero."""
    result = {
        "annotation_count": len(annotations),
        "status_counts": dict(Counter(a["status"] for a in annotations)),
        "scored_count": sum(a["status"] == "ok" and a["score"] is not None for a in annotations),
        "response_quality": None,
        "treatment_representation": None,
        "treatment_role_matching": None,
        "bias_conclusion": None,
    }
    scored = [a["score"] for a in annotations if a["status"] == "ok" and a["score"] is not None]
    simple = [a["score"] for a in annotations if a["evaluator"] == "simple_rubric" and a["status"] == "ok" and a["score"] is not None]
    if simple:
        result["simple_rubric"] = {"mean": sum(simple) / len(simple), "max": 10, "count": len(simple)}
    a4 = [a for a in annotations if a["evaluator"] == "a4_ai_judge" and a["status"] == "ok"]
    if a4:
        capability = [a["details"].get("capability_score") for a in a4
                      if a["details"].get("capability_score") is not None]
        symmetry = [a["details"].get("symmetry", {}).get("score") for a in a4
                    if a["details"].get("symmetry") and a["details"]["symmetry"].get("score") is not None]
        flags = Counter(key for a in a4 for key, value in a["details"].get("flags", {}).items() if value == 1)
        result["a4_ai_judge"] = {
            "capability_mean": round(sum(capability) / len(capability), 1) if capability else None,
            "capability_count": len(capability),
            "symmetry_mean": round(sum(symmetry) / len(symmetry), 1) if symmetry else None,
            "symmetry_count": len(symmetry),
            "flag_counts": dict(flags),
        }
    return result
