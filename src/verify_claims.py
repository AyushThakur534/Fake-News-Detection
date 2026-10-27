"""Retrieve evidence links and existing fact-checks; do not auto-infer truth."""
import os
import requests
from ddgs import DDGS

FACT_CHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_fact_checks(claim, limit=5):
    key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    if not key:
        return []
    try:
        response = requests.get(
            FACT_CHECK_API,
            params={"query": claim, "key": key, "pageSize": limit},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []
    output = []
    for item in payload.get("claims", []):
        for review in item.get("claimReview", []):
            output.append({
                "title": review.get("title") or item.get("text", "Fact-check review"),
                "snippet": (
                    f"Claim: {item.get('text', '')}\n"
                    f"Publisher: {review.get('publisher', {}).get('name', 'Unknown')}\n"
                    f"Rating: {review.get('textualRating', 'Not provided')}"
                ),
                "url": review.get("url", ""),
                "source": review.get("publisher", {}).get("name", ""),
            })
    return output[:limit]

def search_web_evidence(claim, limit=8):
    output = []
    with DDGS() as ddgs:
        for item in ddgs.text(claim, max_results=limit, safesearch="moderate"):
            output.append({
                "title": item.get("title", "Untitled result"),
                "snippet": item.get("body", ""),
                "url": item.get("href", ""),
                "source": item.get("source", ""),
            })
    return output

def verify_claim(claim, limit=8):
    claim = claim.strip()
    if len(claim) < 12:
        raise ValueError("Enter a specific claim of at least 12 characters.")
    fact_checks = search_fact_checks(claim, limit=5)
    web_results = search_web_evidence(claim, limit=limit)
    if fact_checks:
        status = "Existing fact-check reviews found — inspect the review and its rating."
    elif web_results:
        status = "Relevant search results found — manual verification required."
    else:
        status = "Insufficient evidence retrieved."
    return {
        "claim": claim,
        "status": status,
        "fact_checks": fact_checks,
        "web_results": web_results,
        "notice": (
            "Search results are leads, not proof. Open the sources, verify the "
            "date and context, and compare exact names and numbers. No automatic "
            "TRUE/FALSE verdict is inferred from snippets."
        ),
    }
