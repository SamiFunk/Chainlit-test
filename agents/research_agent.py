"""
Research Agent using Perplexity Sonar Pro Search via OpenRouter
Performs external web research with sanitized queries
"""

import os
import re
import httpx
from dataclasses import dataclass, field
from typing import Optional, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ResearchResult:
    """Result of research operation"""
    query: str
    response: str
    sources: List[str] = field(default_factory=list)
    model_used: str = ""


class ResearchAgent:
    """Agent for external research using Perplexity Sonar Pro Search"""

    SYSTEM_PROMPT = """Du bist ein Recherche-Assistent. Deine Aufgabe ist es, präzise und faktenbasierte Recherchen durchzuführen.

## Deine Aufgaben:
1. Beantworte die Frage basierend auf aktuellen Web-Quellen
2. Gib strukturierte, gut formatierte Antworten
3. Zitiere relevante Quellen wenn möglich
4. Sei objektiv und faktenbasiert
5. Wenn du etwas nicht weißt, sage es ehrlich

## Ausgabeformat:
- Verwende Markdown für Formatierung
- Strukturiere lange Antworten mit Überschriften
- Liste Quellen am Ende auf wenn verfügbar
- Halte die Antwort prägnant aber vollständig"""

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
        self.model = os.getenv("PERPLEXITY_MODEL", "perplexity/sonar-pro-search")

    async def research(
            self,
            query: str,
            context: Optional[str] = None,
            language: str = "de"
    ) -> ResearchResult:
        """
        Perform research on a sanitized query

        Args:
            query: The sanitized query to research (should not contain PII/IP)
            context: Optional context about what kind of information is needed
            language: Response language (default: German)

        Returns:
            ResearchResult with response and sources
        """
        user_message = query
        if context:
            user_message = f"Kontext: {context}\n\nFrage: {query}"

        if language == "de":
            user_message += "\n\nBitte antworte auf Deutsch."

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=4096
            )

            response_text = response.choices[0].message.content

            # Extract sources if present in response
            sources = self._extract_sources(response_text)

            return ResearchResult(
                query=query,
                response=response_text,
                sources=sources,
                model_used=self.model
            )

        except Exception as e:
            print(f"Research Error: {type(e).__name__}: {str(e)}")
            raise Exception(f"Research failed: {str(e)}")

    def _extract_sources(self, text: str) -> List[str]:
        """Extract source URLs from response text"""
        url_pattern = r'https?://[^\s\)\]>]+'
        urls = re.findall(url_pattern, text)
        return list(set(urls))  # Remove duplicates

    async def research_with_followup(
            self,
            query: str,
            previous_response: str,
            followup_question: str
    ) -> ResearchResult:
        """
        Continue research with a follow-up question

        Args:
            query: Original query
            previous_response: Previous research response
            followup_question: Follow-up question

        Returns:
            ResearchResult with continued research
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": previous_response},
            {"role": "user", "content": followup_question}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=4096
            )

            response_text = response.choices[0].message.content
            sources = self._extract_sources(response_text)

            return ResearchResult(
                query=followup_question,
                response=response_text,
                sources=sources,
                model_used=self.model
            )

        except Exception as e:
            print(f"Follow-up Research Error: {type(e).__name__}: {str(e)}")
            raise Exception(f"Follow-up research failed: {str(e)}")
