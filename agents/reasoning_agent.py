"""
Reasoning Agent using GPT-5.2 via OpenRouter
Processes research results and generates final responses
"""

import os
import httpx
from dataclasses import dataclass
from typing import Optional, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ReasoningResult:
    """Result of reasoning operation"""
    original_query: str
    research_response: str
    final_response: str
    reasoning_steps: str


class ReasoningAgent:
    """Agent for processing research results and generating final answers using GPT-5.2"""

    SYSTEM_PROMPT = """Du bist ein intelligenter Assistent, der Rechercheergebnisse verarbeitet und präzise Antworten generiert.

## Deine Aufgaben:
1. Analysiere die Rechercheergebnisse
2. Extrahiere die relevantesten Informationen
3. Strukturiere die Antwort klar und verständlich
4. Füge bei Bedarf eigene Schlussfolgerungen hinzu
5. Weise auf Unsicherheiten oder fehlende Informationen hin

## Ausgabeformat:
- Beginne mit einer kurzen Zusammenfassung (2-3 Sätze)
- Gib dann die detaillierte Antwort
- Verwende Markdown für bessere Lesbarkeit
- Füge am Ende Empfehlungen oder nächste Schritte hinzu, wenn relevant

## Wichtig:
- Bleibe bei den Fakten aus der Recherche
- Erfinde keine Informationen
- Sei transparent über die Grenzen der verfügbaren Daten"""

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        # Create HTTP client that bypasses SSL verification (for corporate networks)
        http_client = httpx.AsyncClient(verify=False)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            default_headers={
                "HTTP-Referer": "https://theo-demo.local",
                "X-Title": "theo Research Assistant"
            }
        )
        self.model = os.getenv("GPT_MODEL", "openai/gpt-5.2")

    async def process_and_respond(
            self,
            original_query: str,
            research_response: str,
            mask_mapping: Optional[Dict[str, str]] = None,
            additional_context: Optional[str] = None
    ) -> ReasoningResult:
        """
        Process research results and generate final response

        Args:
            original_query: The user's original question (may contain masked terms)
            research_response: The response from external research
            mask_mapping: Optional mapping to restore original terms in response
            additional_context: Optional additional context

        Returns:
            ReasoningResult with final processed response
        """
        user_message = f"""## Ursprüngliche Frage des Nutzers:
{original_query}

## Rechercheergebnis:
{research_response}

---

Bitte verarbeite diese Informationen und erstelle eine finale, gut strukturierte Antwort für den Nutzer."""

        if additional_context:
            user_message = f"Zusätzlicher Kontext: {additional_context}\n\n{user_message}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.4,
                max_tokens=4096
            )

            final_response = response.choices[0].message.content

            # Restore original values if mask mapping provided
            if mask_mapping:
                for mask, original in mask_mapping.items():
                    final_response = final_response.replace(mask, original)

            return ReasoningResult(
                original_query=original_query,
                research_response=research_response,
                final_response=final_response,
                reasoning_steps="Analysis -> Synthesis -> Response Generation"
            )

        except Exception as e:
            print(f"Reasoning Error: {type(e).__name__}: {str(e)}")
            raise Exception(f"Reasoning failed: {str(e)}")

    async def generate_requirements(
            self,
            research_response: str,
            context: str
    ) -> str:
        """
        Generate requirements or action items from research results

        Args:
            research_response: The research response to analyze
            context: Context about what kind of requirements to generate

        Returns:
            Structured requirements as markdown
        """
        user_message = f"""Basierend auf folgenden Rechercheergebnissen, erstelle eine strukturierte Liste von Anforderungen oder Handlungsempfehlungen:

## Kontext:
{context}

## Rechercheergebnis:
{research_response}

---

Erstelle eine priorisierte Liste mit:
1. **Must-Have** - Kritische Anforderungen
2. **Should-Have** - Wichtige Anforderungen
3. **Nice-to-Have** - Optionale Verbesserungen

Formatiere als Markdown mit Checkboxen."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Du bist ein Requirements Engineer, der aus Rechercheergebnissen strukturierte Anforderungen ableitet."},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=2048
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Requirements Error: {type(e).__name__}: {str(e)}")
            raise Exception(f"Requirements generation failed: {str(e)}")
