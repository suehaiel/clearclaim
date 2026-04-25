from fetcher import fetch_arxiv_papers, save_papers
from loader import run_agent1_5
from agents.extractor import extract_claims_from_abstract
from agents.verifier import verify_claims
from agents.reporter import (
    build_final_report,
    save_final_report,
    save_markdown_report,
)


def main():
    print("Fetching papers...")

    papers = fetch_arxiv_papers("LLM benchmarks", 3)

    enriched_papers = []

    for paper in papers:
        print(f"Processing: {paper['title']}")

        full_paper = run_agent1_5(paper)
        if not full_paper:
            print("  SKIP — no extracted content")
            continue

        claims = extract_claims_from_abstract(full_paper.get("abstract", ""))
        claim_checks = verify_claims(
            claims,
            full_paper.get("results_section", ""),
        )

        full_paper["claims"] = claims
        full_paper["claim_checks"] = claim_checks
        enriched_papers.append(full_paper)

    save_papers(enriched_papers, "data/papers_with_text.json")
    final_report = build_final_report(enriched_papers)
    save_final_report(final_report, "data/final_report.json")
    save_markdown_report(final_report, "data/final_report.md")

    print(
        "Done. Saved papers_with_text.json, "
        "final_report.json, and final_report.md"
    )


if __name__ == "__main__":
    main()
