#!/usr/bin/env python3
"""Fail-closed evaluator for the canonical TT-GMG eight-gate contract."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def evaluate(contract: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    measurements = evidence.get("measurements", {})
    results: list[dict[str, Any]] = []
    for gate in contract["performance_gates"]:
        gate_id = gate["id"]
        item = measurements.get(gate_id, {})
        raw_samples = item.get("samples", [])
        samples = [float(value) for value in raw_samples if isinstance(value, (int, float))]
        tags = set(item.get("tags", []))
        missing_tags = [tag for tag in gate["required_tags"] if tag not in tags]
        minimum_samples = int(gate["minimum_samples"])
        median = statistics.median(samples) if samples else None
        evidence_path = item.get("evidence")
        blockers: list[str] = []
        if len(samples) < minimum_samples:
            blockers.append(f"needs {minimum_samples} sample(s), has {len(samples)}")
        if missing_tags:
            blockers.append("missing evidence tags: " + ", ".join(missing_tags))
        if not evidence_path:
            blockers.append("missing durable evidence path")
        if median is not None and median > float(gate["maximum"]):
            blockers.append(f"median {median:.9g} exceeds {gate['maximum']:.9g} {gate['metric']}")
        status = "green" if not blockers else ("red" if samples and median is not None and median > float(gate["maximum"]) else "open")
        results.append(
            {
                "id": gate_id,
                "label": gate["label"],
                "status": status,
                "budget_maximum": gate["maximum"],
                "metric": gate["metric"],
                "sample_count": len(samples),
                "samples": samples,
                "median": median,
                "evidence": evidence_path,
                "note": item.get("note", ""),
                "blockers": blockers,
            }
        )
    green = sum(result["status"] == "green" for result in results)
    return {
        "schema": "tt_gmg_gate_report_v1",
        "contract_schema": contract["schema"],
        "target": contract["target"],
        "performance_score": {"green": green, "total": len(results)},
        "performance_gates": results,
        "correctness": evidence.get("correctness", {}),
        "acceptance_complete": green == len(results)
        and evidence.get("correctness", {}).get("reduced_dump", {}).get("status") == "green"
        and evidence.get("correctness", {}).get("full_ccx", {}).get("status") == "green"
        and evidence.get("correctness", {}).get("safety_path", {}).get("status") == "green",
    }


def render_markdown(report: dict[str, Any]) -> str:
    score = report["performance_score"]
    lines = [
        "# TT-GMG gate report",
        "",
        f"Performance score: **{score['green']}/{score['total']} green**.",
        f"Acceptance complete: **{str(report['acceptance_complete']).lower()}**.",
        "",
        "| Gate | Status | Median | Budget | Evidence/blocker |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["performance_gates"]:
        median = "—" if item["median"] is None else f"{item['median']:.9g}"
        reason = "; ".join(item["blockers"]) or item["evidence"] or ""
        lines.append(
            f"| {item['id']} | {item['status']} | {median} | ≤ {item['budget_maximum']:.9g} {item['metric']} | {reason} |"
        )
    lines.extend(["", "## Correctness/integration", "", "```json", json.dumps(report.get("correctness", {}), indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=here / "gate_contract.json")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    report = evaluate(_load(args.contract), _load(args.evidence))
    rendered = render_markdown(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(rendered, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
