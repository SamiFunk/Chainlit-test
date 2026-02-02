"""
Internal Agent - Direct Claude responses without external research
"""
import os
import httpx
from dataclasses import dataclass
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InternalResult:
    """Result of internal query"""
    query: str
    response: str
    model_used: str = ""


class InternalAgent:
    """Agent for direct responses without masking or external research"""

    SYSTEM_PROMPT = """Du bist theo, ein intelligenter interner Assistent.

## Aufgaben:
- Beantworte Fragen direkt und präzise
- Nutze dein Wissen für hilfreiche Antworten
- Sei transparent wenn du etwas nicht weißt

## Ausgabeformat:
- Strukturiere Antworten klar und übersichtlich
- Verwende Markdown für bessere Lesbarkeit
- Beginne mit einer kurzen Zusammenfassung bei komplexen Themen

## Wichtig:
- Dies ist der INTERNE Modus - keine externe Recherche
- Sensible Daten werden NICHT nach extern gesendet
- Bei Bedarf für aktuelle/externe Informationen, empfehle den externen Modus"""

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

    async def respond(self, query: str, context: str = None) -> InternalResult:
        """
        Generate direct response without masking or external research

        Args:
            query: The user's question
            context: Optional additional context

        Returns:
            InternalResult with the response
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        user_msg = query
        if context:
            user_msg = f"Kontext: {context}\n\n{query}"

        messages.append({"role": "user", "content": user_msg})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_tokens=4096
            )

            return InternalResult(
                query=query,
                response=response.choices[0].message.content,
                model_used=self.model
            )

        except Exception as e:
            print(f"Internal Agent Error: {type(e).__name__}: {str(e)}")
            raise Exception(f"Internal response failed: {str(e)}")
