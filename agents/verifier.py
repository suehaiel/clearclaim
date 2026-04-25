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

from agents.embedder import claim_similarity, find_best_matching_chunk


load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"
_VALID_VERDICTS = {
    "supported",
    "overstated",
    "vague",
    "contradicted",
    "not_found",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def _clean_json_text(raw_text):
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _best_sentence_from_chunk(claim, chunk):
    claim_tokens = _tokenize(claim)
    claim_terms = {token for token in claim_tokens if len(token) > 4}

    def _score_candidates(candidates):
        best_candidate = ""
        best_score = -1
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) < 20:
                continue
            candidate_tokens = _tokenize(candidate)
            overlap = len(claim_terms & candidate_tokens)
            if overlap <= 0:
                continue
            score = overlap / max(1.0, len(candidate_tokens) ** 0.5)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    line_candidates = [
        line.strip()
        for line in re.split(r"[\r\n]+", chunk or "")
        if line.strip()
    ]
    line_choice = _score_candidates(line_candidates)
    if line_choice:
        return line_choice

    sentence_candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", chunk or "")
        if sentence.strip()
    ]
    sentence_choice = _score_candidates(sentence_candidates)
    if sentence_choice:
        return sentence_choice

    chunk_text = re.sub(r"\s+", " ", chunk or "").strip()
    if not chunk_text:
        return "not found"

    ordered_tokens = sorted(
        [token for token in claim_terms],
        key=len,
        reverse=True,
    )
    for token in ordered_tokens:
        index = chunk_text.lower().find(token)
        if index != -1:
            start = max(0, index - 120)
            end = min(len(chunk_text), index + 240)
            return chunk_text[start:end].strip()

    return chunk_text[:360]


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


def _default_verdict(claim, best_chunk, similarity):
    if similarity >= 0.55:
        verdict = "supported"
    elif similarity >= 0.35:
        verdict = "overstated"
    elif similarity >= 0.2:
        verdict = "vague"
    else:
        verdict = "not_found"

    return {
        "verdict": verdict,
        "confidence": "low",
        "evidence_quote": _best_sentence_from_chunk(claim, best_chunk),
        "reasoning": (
            "Heuristic verdict based on similarity because the model call "
            "did not return a usable response."
        ),
    }


def _gemini_verdict(claim, best_chunk, similarity):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return _default_verdict(claim, best_chunk, similarity)

    prompt = (
        "You are verifying whether a paper claim is supported by a results "
        "excerpt. Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "verdict": "supported|overstated|vague|contradicted|not_found",\n'
        '  "confidence": "high|medium|low",\n'
        '  "evidence_quote": "exact quote from the excerpt or not found",\n'
        '  "reasoning": "1-2 sentences"\n'
        "}\n\n"
        f"Claim:\n{claim}\n\n"
        f"Best Matching Results Excerpt:\n{best_chunk or 'not found'}\n\n"
        "Rules:\n"
        "- If no direct evidence appears, use verdict not_found.\n"
        "- evidence_quote must be verbatim from the excerpt or 'not found'.\n"
        "- No markdown, no extra keys, no extra commentary."
    )

    try:
        raw_text = _generate_with_gemini(prompt, api_key=api_key)
        text = _clean_json_text(raw_text)
        if not text:
            return _default_verdict(claim, best_chunk, similarity)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return _default_verdict(claim, best_chunk, similarity)

        verdict = str(parsed.get("verdict", "not_found")).strip().lower()
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        evidence_quote = str(parsed.get("evidence_quote", "not found")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()

        if verdict not in _VALID_VERDICTS:
            verdict = "not_found"
        if confidence not in _VALID_CONFIDENCE:
            confidence = "low"
        if not evidence_quote:
            evidence_quote = "not found"
        if evidence_quote != "not found" and best_chunk:
            if evidence_quote not in best_chunk:
                evidence_quote = _best_sentence_from_chunk(claim, best_chunk)
        elif best_chunk and similarity >= 0.2:
            evidence_quote = _best_sentence_from_chunk(claim, best_chunk)
        if not reasoning:
            reasoning = "Insufficient structured reasoning returned by model."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "evidence_quote": evidence_quote,
            "reasoning": reasoning,
        }
    except Exception:
        return _default_verdict(claim, best_chunk, similarity)
    finally:
        # Free-tier rate-limit protection between Gemini calls.
        time.sleep(1)


def verify_claims(claims, results_section):
    """Agent 3b: Verify claims against the most relevant results chunk."""
    verifications = []

    for claim in claims or []:
        similarity = claim_similarity(claim, results_section)
        best_chunk = find_best_matching_chunk(claim, results_section)
        verdict_info = _gemini_verdict(claim, best_chunk, similarity)

        verifications.append(
            {
                "claim": claim,
                "similarity": round(float(similarity), 4),
                "verdict": verdict_info["verdict"],
                "confidence": verdict_info["confidence"],
                "evidence_quote": verdict_info["evidence_quote"],
                "reasoning": verdict_info["reasoning"],
            }
        )

    return verifications
