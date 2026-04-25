import requests
import pdfplumber
import os
import re


def download_pdf(pdf_url, save_path="paper.pdf"):
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except Exception as e:
        print(f"    PDF download failed: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"    PDF extraction failed: {e}")
    return text


def extract_results_section(full_text):
    """
    Tries to find the results/experiments section.
    Falls back to last 40% of paper if not found.
    """
    section_headers = [
        r'\bresults\b',
        r'\bexperiments\b',
        r'\bevaluation\b',
        r'\bfindings\b',
    ]

    best_pos = -1
    for pattern in section_headers:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        for match in reversed(matches):
            pos = match.start()
            # must be past first 30% — avoids table of contents hits
            if pos > len(full_text) * 0.3:
                best_pos = pos
                break
        if best_pos > 0:
            break

    if best_pos > 0:
        return full_text[best_pos:best_pos + 4000]
    else:
        # fallback: last 40% of paper
        start = int(len(full_text) * 0.6)
        return full_text[start:]


def extract_abstract_from_pdf(full_text):
    """
    Attempts to isolate the abstract section from extracted PDF text.
    """
    if not full_text:
        return ""

    lowered = full_text.lower()
    abstract_match = re.search(r"\babstract\b", lowered)
    if not abstract_match:
        return ""

    start = abstract_match.end()
    after_abstract = full_text[start: start + 3000]

    end_patterns = [
        r"\n\s*1\s+[A-Z][A-Za-z ]{2,}",
        r"\n\s*I\.?\s+[A-Z][A-Za-z ]{2,}",
        r"\n\s*introduction\b",
        r"\n\s*keywords?\b",
    ]

    end_pos = len(after_abstract)
    for pattern in end_patterns:
        m = re.search(pattern, after_abstract, flags=re.IGNORECASE)
        if m:
            end_pos = min(end_pos, m.start())

    abstract_text = after_abstract[:end_pos].strip()
    return re.sub(r"\s+", " ", abstract_text)


def abstracts_roughly_match(metadata_abstract, pdf_abstract):
    """
    Lightweight consistency check between feed metadata and PDF abstract.
    """
    if not metadata_abstract or not pdf_abstract:
        return False

    norm_meta = re.sub(r"\s+", " ", metadata_abstract).strip().lower()
    norm_pdf = re.sub(r"\s+", " ", pdf_abstract).strip().lower()

    meta_tokens = set(norm_meta.split())
    pdf_tokens = set(norm_pdf.split())
    if not meta_tokens or not pdf_tokens:
        return False

    overlap = len(meta_tokens & pdf_tokens)
    ratio = overlap / max(1, len(meta_tokens))
    return ratio >= 0.5


def run_agent1_5(paper):
    arxiv_id_clean = paper["arxiv_id"].replace("/", "_")
    pdf_path = f"{arxiv_id_clean}.pdf"

    print(f"  Downloading: {paper['title'][:60]}...")

    downloaded = download_pdf(paper["pdf_link"], save_path=pdf_path)

    if not downloaded:
        print("  SKIP — could not download PDF")
        return None

    full_text = extract_text_from_pdf(pdf_path)

    if len(full_text) < 500:
        print(
            f"  WARNING — extracted text is very short "
            f"({len(full_text)} chars)"
        )

    # Clean up PDF file after extraction
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    results_section = extract_results_section(full_text)
    pdf_abstract = extract_abstract_from_pdf(full_text)
    metadata_abstract = paper.get("abstract", "")
    abstract_match = abstracts_roughly_match(metadata_abstract, pdf_abstract)

    print(f"  OK — {len(full_text)} chars extracted, "
          f"{len(results_section)} chars in results section")
    if pdf_abstract:
        print(f"  Abstract check — {'MATCH' if abstract_match else 'DIFF'}")
    else:
        print("  Abstract check — PDF abstract not detected")

    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": metadata_abstract,
        "pdf_abstract": pdf_abstract,
        "abstract_match": abstract_match,
        "full_text": full_text[:20000],
        "results_section": results_section,
    }
