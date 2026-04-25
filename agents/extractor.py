import json
import importlib
import os
import re
import time

from dotenv import load_dotenv

try:
    google_genai = importlib.import_module("google.genai")
except Exception:  # pragma: no cover - runtime dependency failure path
    google_genai = None


load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"


def _clean_json_text(raw_text):
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _generate_with_gemini(prompt, api_key):
    """Return model text using the current Gemini SDK."""
    if google_genai is not None:
        try:
            client = google_genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=_MODEL_NAME,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""
        except Exception:
            pass

    return ""


def _fallback_extract_claims(abstract, max_claims):
    """Heuristic fallback so pipeline still returns claims when API fails."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", abstract or "")
        if s.strip()
    ]

    score_terms = (
        "achieve",
        "outperform",
        "improve",
        "results",
        "accuracy",
        "f1",
        "auc",
        "benchmark",
        "correlation",
        "pass rate",
        "%",
    )

    scored = []
    for sentence in sentences:
        lower = sentence.lower()
        score = sum(1 for term in score_terms if term in lower)
        if re.search(r"\d", sentence):
            score += 2
        if score > 0:
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    claims = []
    seen = set()
    for _, sentence in scored:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(sentence)
        if len(claims) >= max_claims:
            break

    if not claims:
        claims = sentences[:max_claims]
    return claims


def _gemini_extract_claims(abstract, max_claims):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return []

    prompt = (
        "Extract only specific, falsifiable claims from the abstract below. "
        "A good claim is concrete and verifiable (e.g., includes measured "
        "improvements, benchmark names, numbers, or explicit comparisons). "
        "Exclude vague statements and background context.\n\n"
        f"Return ONLY a JSON array of strings with at most "
        f"{max_claims} items. "
        "No markdown, no explanation, no extra keys.\n\n"
        f"Abstract:\n{abstract}"
    )

    try:
        raw_text = _generate_with_gemini(prompt, api_key=api_key)
        text = _clean_json_text(raw_text)
        if not text:
            return []
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []

        claims = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                claims.append(item.strip())
            if len(claims) >= max_claims:
                break
        return claims
    except Exception:
        return []
    finally:
        # Free-tier rate-limit protection.
        time.sleep(1)


def extract_claims_from_abstract(abstract, max_claims=5):
    """Agent 2: Extract specific, falsifiable claims from an abstract."""
    if not abstract:
        return []

    claims = _gemini_extract_claims(abstract, max_claims=max_claims)
    if claims:
        return claims

    return _fallback_extract_claims(abstract, max_claims=max_claims)
