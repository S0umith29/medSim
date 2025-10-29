from typing import List, Dict
import json
import requests

from src.config import OLLAMA_ENDPOINT, OLLAMA_MODEL, ETHICS_POLICY


def _ollama_generate(prompt: str, temperature: float = 0.1) -> str:
    resp = requests.post(
        f"{OLLAMA_ENDPOINT.rstrip('/')}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "temperature": temperature, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def build_context_block(contexts: List[Dict]) -> str:
    return "\n\n".join([f"[Source: {c.get('pmc_id')}]\n{c.get('text')}" for c in contexts])


def evaluate_question(question: str, contexts: List[Dict]) -> Dict:
    """
    Evaluate a student's question against a rubric using the provided retrieval contexts.
    Returns a dict with scores, reasoning, phase_guess, risk_flags.
    (No guardrails; pure feedback only.)
    """
    context_block = build_context_block(contexts)
    prompt = f"""System: You are a clinical educator evaluating a medical student's question to a virtual patient.
Use the RAG context (case snippets) and the policy below. Score 0–5 for each criterion.
Explain briefly why. Return ONLY JSON following the schema.

Policy:
{ETHICS_POLICY}

Case context:
{context_block}

Student question:
{question}

JSON schema (fill all fields with appropriate values):
{{
  "scores": {{
    "relevance": 0,
    "diagnostic_utility": 0,
    "clarity_specificity": 0,
    "empathy_professionalism": 0,
    "hipaa_ethics": 0
  }},
  "reasoning": {{
    "relevance": "",
    "diagnostic_utility": "",
    "clarity_specificity": "",
    "empathy_professionalism": "",
    "hipaa_ethics": ""
  }},
  "phase_guess": "",
  "risk_flags": []
}}
"""
    raw = _ollama_generate(prompt, temperature=0.1)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "scores": {k: 3 for k in [
                "relevance","diagnostic_utility","clarity_specificity","empathy_professionalism","hipaa_ethics"
            ]},
            "reasoning": {k: "Auto-fallback (could not parse JSON)." for k in [
                "relevance","diagnostic_utility","clarity_specificity","empathy_professionalism","hipaa_ethics"
            ]},
            "phase_guess": "hpi",
            "risk_flags": ["parse_error"]
        }
