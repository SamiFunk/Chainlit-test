"""
PII/IP Masking Agent using GPT-5.2 via OpenRouter
Detects and masks personal identifiable information (PII) and intellectual property (IP)
"""

import os
import json
import httpx
from dataclasses import dataclass, field
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class MaskingResult:
    """Result of PII/IP masking operation"""
    original_text: str
    masked_text: str
    detected_entities: dict = field(default_factory=dict)
    mask_mapping: dict = field(default_factory=dict)
    confidence: float = 0.0


class MaskingAgent:
    """Agent for detecting and masking PII and IP using GPT-5.2"""

    SYSTEM_PROMPT = """# KRITISCHE AUFGABE: Datenschutz-Analyse

Du bist ein Datenschutz-Experte mit der Aufgabe, ALLE sensiblen Informationen zu erkennen und zu maskieren.

## WICHTIG: Im Zweifel IMMER maskieren!
Wenn du unsicher bist ob etwas sensibel ist → MASKIERE ES.

## KONKRETE ERKENNUNGSMUSTER:

### PII (Personenbezogene Daten) - MUSS erkannt werden:

**PERSON** - Alle Personennamen:
- Mit Titel: "Dr. Thomas Weber", "Prof. Maria Schmidt"
- Ohne Titel: "Hans Müller", "Jennifer Miller"
- Ausländisch: "John Smith", "Pierre Dupont"
→ Maskiere als: [PERSON_1], [PERSON_2], etc.

**EMAIL** - Alle E-Mail-Adressen:
- Format: name@domain.de, name.nachname@firma.com
- Beispiele: "thomas.weber@siemens-energy.de", "h.mueller@company.de"
→ Maskiere als: [EMAIL_1], [EMAIL_2], etc.

**PHONE** - Alle Telefonnummern:
- Deutsch: "+49 171 8834521", "0171-8834521", "+49 (0)171 8834521"
- International: "+1 555 123 4567"
→ Maskiere als: [PHONE_1], [PHONE_2], etc.

**IBAN** - Bankverbindungen:
- Format: "DE89 3704 0044 0532 0130 00" oder ohne Leerzeichen
- Auch BIC: "COBADEFFXXX"
→ Maskiere als: [IBAN_1], [BIC_1]

**ADDRESS** - Physische Adressen:
- Straßen: "Petuelring 130", "Hauptstraße 42"
- Mit Ort: "München, Petuelring 130"
→ Maskiere als: [ADDRESS_1], [ADDRESS_2]

**ID_NUMBER** - Personalnummern, Ausweisnummern:
- Beispiel: "Personalnummer: 48291-SE"
→ Maskiere als: [ID_NUMBER_1]

### IP (Vertrauliche Geschäftsdaten) - MUSS erkannt werden:

**COMPANY** - Firmennamen (eigene und Partner):
- Beispiele: "Siemens Energy AG", "BMW AG", "Volkswagen AG", "Goldman Sachs"
- Auch Abkürzungen: "VW", "MB" im Firmenkontext
→ Maskiere als: [COMPANY_1], [COMPANY_2], etc.

**PROJECT** - Projektnamen und Codenamen:
- Beispiele: "Projekt Phoenix", "Codename: Aurora", "Initiative Alpha"
→ Maskiere als: [PROJECT_1], [PROJECT_2]

**FINANCIAL** - Finanzielle Zahlen mit Kontext:
- Budgets: "2.500.000 EUR", "Budget: 5 Mio"
- Umsätze: "Umsatzvolumen: 850.000 EUR"
- NICHT maskieren: Allgemeine Zahlen ohne Finanzkontext
→ Maskiere als: [FINANCIAL_1], [FINANCIAL_2]

**CLIENT** - Kundennamen (wenn als Kunde identifiziert):
- "Unser Kunde BMW AG", "Beteiligte Kunden: ..."
→ Maskiere als: [CLIENT_1], [CLIENT_2]

## AUSGABEFORMAT (STRIKT JSON):

Du MUSST exakt dieses JSON-Format zurückgeben:

```json
{
    "masked_text": "Der komplette Text mit allen [KATEGORIE_N] Platzhaltern",
    "entities": [
        {
            "original": "Dr. Thomas Weber",
            "masked": "[PERSON_1]",
            "category": "PERSON",
            "confidence": 0.95
        },
        {
            "original": "thomas.weber@siemens-energy.de",
            "masked": "[EMAIL_1]",
            "category": "EMAIL",
            "confidence": 0.99
        }
    ],
    "overall_confidence": 0.92,
    "total_entities_found": 15
}
```

## REGELN:
1. Gib NUR valides JSON zurück - KEINE Erklärungen davor oder danach
2. Bei JEDEM Namen, JEDER E-Mail, JEDER Telefonnummer → MASKIEREN
3. Gleiche Werte = gleicher Platzhalter (wenn "Dr. Thomas Weber" 3x vorkommt → immer [PERSON_1])
4. Nummerierung pro Kategorie: [PERSON_1], [PERSON_2], [EMAIL_1], [EMAIL_2], etc.
5. "total_entities_found" MUSS die Anzahl aller gefundenen Entitäten sein

## WARNUNG:
Wenn du in einem Dokument mit Tabellen, Namen und E-Mails KEINE Entitäten findest,
hast du einen Fehler gemacht! Analysiere erneut."""

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

    async def analyze_and_mask(self, text: str, context: Optional[str] = None) -> MaskingResult:
        """
        Analyze text for PII/IP and return masked version

        Args:
            text: The text to analyze and mask
            context: Optional context about the text (e.g., "internal email", "customer data")

        Returns:
            MaskingResult with original, masked text and entity mappings
        """
        user_message = f"Analysiere und maskiere folgenden Text:\n\n{text}"
        if context:
            user_message = f"Kontext: {context}\n\n{user_message}"

        try:
            print(f"[DEBUG] Using model: {self.model}")
            print(f"[DEBUG] Input text length: {len(text)} chars")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Increased from 0.1 for better detection
                max_tokens=16384  # Increased for longer documents with full JSON response
            )

            result_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Raw response (first 500 chars): {result_text[:500]}")

            # Parse JSON response
            # Handle markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result_data = json.loads(result_text)

            # Build mask mapping
            mask_mapping = {}
            detected_entities = {}

            for entity in result_data.get("entities", []):
                mask_mapping[entity["masked"]] = entity["original"]
                category = entity["category"]
                if category not in detected_entities:
                    detected_entities[category] = []
                detected_entities[category].append({
                    "original": entity["original"],
                    "masked": entity["masked"],
                    "confidence": entity.get("confidence", 0.9)
                })

            return MaskingResult(
                original_text=text,
                masked_text=result_data.get("masked_text", text),
                detected_entities=detected_entities,
                mask_mapping=mask_mapping,
                confidence=result_data.get("overall_confidence", 0.9)
            )

        except json.JSONDecodeError as e:
            # Log the actual response for debugging
            print(f"JSON Parse Error: {e}")
            print(f"Raw LLM response (first 1000 chars): {result_text[:1000]}")

            # Try to extract masked_text even from truncated JSON
            try:
                # Look for masked_text field
                import re
                masked_match = re.search(r'"masked_text":\s*"(.*?)(?:"\s*,\s*"entities"|$)', result_text, re.DOTALL)
                if masked_match:
                    extracted_masked = masked_match.group(1)
                    # Unescape JSON string
                    extracted_masked = extracted_masked.replace('\\"', '"').replace('\\n', '\n')
                    print(f"[RECOVERY] Extracted masked_text from truncated JSON ({len(extracted_masked)} chars)")

                    # Try to extract entities from the masked text itself
                    entity_pattern = r'\[([A-Z_]+)_(\d+)\]'
                    found_placeholders = re.findall(entity_pattern, extracted_masked)

                    detected_entities = {}
                    mask_mapping = {}
                    for category, num in found_placeholders:
                        placeholder = f"[{category}_{num}]"
                        if category not in detected_entities:
                            detected_entities[category] = []
                        # We don't have the original values, but we know they were masked
                        if not any(e['masked'] == placeholder for e in detected_entities.get(category, [])):
                            detected_entities[category].append({
                                'original': f'<{category.lower()}_{num}>',
                                'masked': placeholder,
                                'confidence': 0.8
                            })
                            mask_mapping[placeholder] = f'<{category.lower()}_{num}>'

                    return MaskingResult(
                        original_text=text,
                        masked_text=extracted_masked,
                        detected_entities=detected_entities,
                        mask_mapping=mask_mapping,
                        confidence=0.7  # Lower confidence due to recovery
                    )
            except Exception as recovery_error:
                print(f"[RECOVERY FAILED] {recovery_error}")

            # Return with warning - original text unchanged
            return MaskingResult(
                original_text=text,
                masked_text=text,
                detected_entities={},
                mask_mapping={},
                confidence=0.0
            )
        except Exception as e:
            print(f"[ERROR] Masking Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Masking failed: {str(e)}")

    def unmask_text(self, masked_text: str, mask_mapping: dict) -> str:
        """
        Restore original values from masked text

        Args:
            masked_text: Text with [CATEGORY_N] placeholders
            mask_mapping: Dictionary mapping placeholders to original values

        Returns:
            Text with original values restored
        """
        result = masked_text
        for mask, original in mask_mapping.items():
            result = result.replace(mask, original)
        return result

    def get_entity_summary(self, detected_entities: dict) -> str:
        """Generate a human-readable summary of detected entities"""
        if not detected_entities:
            return "Keine sensiblen Daten erkannt."

        lines = ["**Erkannte sensible Daten:**\n"]

        category_labels = {
            "PERSON": "Personen",
            "EMAIL": "E-Mail-Adressen",
            "PHONE": "Telefonnummern",
            "ADDRESS": "Adressen",
            "COMPANY": "Firmen",
            "PROJECT": "Projekte",
            "PRODUCT": "Produkte",
            "FINANCIAL": "Finanzdaten",
            "TECHNICAL": "Technische Daten",
            "CLIENT": "Kunden",
            "HEALTH": "Gesundheitsdaten",
            "ID_NUMBER": "ID-Nummern",
            "IBAN": "Bankverbindungen",
            "BIC": "BIC-Codes",
            "DATE_OF_BIRTH": "Geburtsdaten",
            "STRATEGY": "Strategien"
        }

        for category, entities in detected_entities.items():
            label = category_labels.get(category, category)
            count = len(entities)
            lines.append(f"\n**{label}:** ({count} gefunden)")
            for entity in entities:
                lines.append(f"  - `{entity['original']}` -> `{entity['masked']}`")

        return "\n".join(lines)
