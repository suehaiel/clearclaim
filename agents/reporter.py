import json
import os


def build_final_report(enriched_papers):
    """
    Agent 4: Aggregates per-paper verification outputs into one report.
    """
    total_claims = 0
    supported_claims = 0

    for paper in enriched_papers:
        claim_checks = paper.get("claim_checks", [])
        total_claims += len(claim_checks)
        supported_claims += sum(
            1 for c in claim_checks if c.get("verdict") == "supported"
        )

    support_rate = (
        round((supported_claims / total_claims), 4)
        if total_claims
        else 0.0
    )

    report = {
        "summary": {
            "papers_processed": len(enriched_papers),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "support_rate": support_rate,
        },
        "papers": enriched_papers,
    }
    return report


def save_final_report(report, output_path="data/final_report.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _verdict_emoji(verdict):
    if verdict == "supported":
        return "✅"
    if verdict in {"overstated", "vague"}:
        return "⚠️"
    if verdict in {"contradicted", "not_found"}:
        return "❌"
    return "⚪"


def save_markdown_report(report, output_path="data/final_report.md"):
    papers = report.get("papers", [])

    verdict_counts = {
        "supported": 0,
        "overstated": 0,
        "vague": 0,
        "contradicted": 0,
        "not_found": 0,
    }

    total_claims = 0
    for paper in papers:
        for check in paper.get("claim_checks", []):
            verdict = check.get("verdict", "not_found")
            if verdict not in verdict_counts:
                verdict = "not_found"
            verdict_counts[verdict] += 1
            total_claims += 1

    lines = []
    lines.append("# Academic Paper Fact-Check Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Papers processed: {len(papers)}")
    lines.append(f"- Total claims: {total_claims}")
    lines.append("- Verdict breakdown:")
    lines.append(f"  - supported: {verdict_counts['supported']}")
    lines.append(f"  - overstated: {verdict_counts['overstated']}")
    lines.append(f"  - vague: {verdict_counts['vague']}")
    lines.append(f"  - contradicted: {verdict_counts['contradicted']}")
    lines.append(f"  - not_found: {verdict_counts['not_found']}")
    lines.append("")

    lines.append("## Papers")
    lines.append("")
    for paper in papers:
        title = paper.get("title", "Untitled")
        arxiv_id = paper.get("arxiv_id", "unknown")
        abstract_match = paper.get("abstract_match", False)
        lines.append(f"### {title}")
        lines.append(f"- arxiv_id: {arxiv_id}")
        lines.append(f"- abstract_match: {abstract_match}")
        lines.append("")

        claim_checks = paper.get("claim_checks", [])
        if not claim_checks:
            lines.append("No claims extracted or verified.")
            lines.append("")
            continue

        for idx, check in enumerate(claim_checks, start=1):
            claim = check.get("claim", "")
            verdict = check.get("verdict", "not_found")
            confidence = check.get("confidence", "low")
            evidence_quote = check.get("evidence_quote", "not found")
            reasoning = check.get("reasoning", "")
            emoji = _verdict_emoji(verdict)

            lines.append(f"#### Claim {idx}")
            lines.append(f"- Claim: {claim}")
            lines.append(f"- Verdict: {emoji} {verdict}")
            lines.append(f"- Confidence: {confidence}")
            lines.append(f"- Evidence quote: \"{evidence_quote}\"")
            lines.append(f"- Reasoning: {reasoning}")
            lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
