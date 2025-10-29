from typing import List, Dict, Optional
import requests
import os

from src.config import OLLAMA_ENDPOINT, OLLAMA_MODEL, SYSTEM_PROMPT, TEMPERATURE, ETHICS_POLICY


class OllamaLLM:
    def __init__(self, endpoint: str = OLLAMA_ENDPOINT, model: str = OLLAMA_MODEL, temperature: float = TEMPERATURE):
        self.endpoint = endpoint.rstrip('/')
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, contexts: List[Dict]) -> str:
        context_block = "\n\n".join([
            f"[Source: {c.get('pmc_id')}]\n{c.get('text')}" for c in contexts
        ])

        prompt = (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"Policy: {ETHICS_POLICY}\n\n"
            f"Context:\n{context_block}\n\n"
            f"User question: {question}\n\n"
            f"Answer concisely. Include citations like [PMC_id] where relevant."
        )

        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def generate_patient_reply(self, user_utterance: str, contexts: List[Dict], persona: Optional[Dict] = None) -> str:
        """
        Generate a roleplay patient's reply. Persona may include age, sex, name, baseline traits.
        """
        context_block = "\n\n".join([
            f"[Source: {c.get('pmc_id')}]\n{c.get('text')}" for c in contexts
        ])

        persona_desc = ""
        if persona:
            p = []
            if persona.get("name"): p.append(f"Name: {persona['name']}")
            if persona.get("age"): p.append(f"Age: {persona['age']}")
            if persona.get("sex"): p.append(f"Sex: {persona['sex']}")
            if persona.get("notes"): p.append(f"Notes: {persona['notes']}")
            persona_desc = "\n".join(p)

        prompt = (
            f"System: You are roleplaying as a cooperative patient. Answer only with subjective information a patient would say.\n"
            f"If the clinician asks for data you wouldn't know (labs, imaging), respond with uncertainty or what you were told.\n"
            f"If not sure, say you don't know. Keep responses concise and natural.\n"
            f"Policy: {ETHICS_POLICY}\n\n"
            f"Patient persona (optional):\n{persona_desc}\n\n"
            f"Relevant case snippets for consistency (not shown to user):\n{context_block}\n\n"
            f"Clinician says: {user_utterance}\n\n"
            f"Patient reply:"
        )

        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": max(0.2, self.temperature),
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
