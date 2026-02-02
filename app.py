"""
theo Research Assistant - Demo Prototype
A privacy-first AI assistant that masks PII/IP before external research

Workflow:
1. User submits query (+ optional files)
2. GPT-5.2 identifies and masks sensitive data (PII/IP)
3. User reviews and approves sanitized query
4. Perplexity Sonar Pro performs external research
5. GPT-5.2 processes response and delivers final answer
"""

import os
import warnings
import urllib3

# Suppress SSL warnings (for corporate networks with SSL inspection)
warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)

import chainlit as cl
from chainlit.input_widget import Select, Switch
from dotenv import load_dotenv

from agents import MaskingAgent, ResearchAgent, ReasoningAgent, InternalAgent

# Load environment variables
load_dotenv()

# Initialize agents
masking_agent = MaskingAgent()
research_agent = ResearchAgent()
reasoning_agent = ReasoningAgent()
internal_agent = InternalAgent()

# Mode constants
MODE_INTERNAL = "internal"
MODE_EXTERNAL = "external"


@cl.on_chat_start
async def start():
    """Initialize chat session"""
    # Store session state
    cl.user_session.set("workflow_state", "idle")
    cl.user_session.set("pending_query", None)
    cl.user_session.set("mask_mapping", {})
    cl.user_session.set("original_query", None)
    cl.user_session.set("mode", MODE_INTERNAL)  # Default to internal mode

    # ChatSettings with mode toggle
    settings = await cl.ChatSettings([
        Switch(
            id="external_mode",
            label="Externer Modus",
            initial=False,
            description="Aktiviere Perplexity-Recherche mit PII-Maskierung"
        ),
        Select(
            id="research_model",
            label="Recherche-Modell",
            values=["perplexity/sonar-pro", "perplexity/sonar-reasoning-pro"],
            initial_index=0,
        ),
    ]).send()

    cl.user_session.set("settings", settings)

    # Welcome message
    await cl.Message(
        content="""# 👋 Willkommen bei theo

**Aktueller Modus:** Intern (keine externe Recherche)

| Modus | Beschreibung |
|-------|-------------|
| **Intern** | Direkte Antworten, keine Daten nach extern |
| **Extern** | Web-Recherche mit automatischer PII-Maskierung |

**Modus wechseln:** Klicke auf das Zahnrad-Symbol oben rechts.

---

**Starte jetzt** indem du deine Frage eingibst! 👇"""
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes"""
    cl.user_session.set("settings", settings)
    is_external = settings.get("external_mode", False)
    new_mode = MODE_EXTERNAL if is_external else MODE_INTERNAL
    old_mode = cl.user_session.get("mode", MODE_INTERNAL)
    cl.user_session.set("mode", new_mode)

    # Update research model
    research_agent.model = settings.get("research_model", "perplexity/sonar-pro")

    if new_mode != old_mode:
        if new_mode == MODE_EXTERNAL:
            await cl.Message(
                content="## 🔄 **Externer Modus** aktiviert\n\n🔒 PII-Maskierung → ✅ Freigabe → 🌐 Perplexity"
            ).send()
        else:
            await cl.Message(
                content="## 🔄 **Interner Modus** aktiviert\n\n🔐 Direkte Antworten, keine externe Recherche"
            ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Main message handler"""
    workflow_state = cl.user_session.get("workflow_state", "idle")
    mode = cl.user_session.get("mode", MODE_INTERNAL)

    # Handle file attachments
    files_content = ""
    if message.elements:
        files_content = await process_files(message.elements)

    user_input = message.content
    if files_content:
        user_input = f"{message.content}\n\n--- Dateiinhalt ---\n{files_content}"

    # Route based on mode
    if mode == MODE_INTERNAL:
        await handle_internal_query(user_input)
    else:
        # External mode - existing masking workflow
        if workflow_state != "idle":
            cl.user_session.set("workflow_state", "idle")
        await start_masking_workflow(user_input)


async def handle_internal_query(user_query: str):
    """Internal mode - direct response without masking"""
    cl.user_session.set("original_query", user_query)

    async with cl.Step(name="🧠 Verarbeitung", type="llm") as step:
        step.input = "Generiere Antwort..."
        try:
            result = await internal_agent.respond(user_query)
            step.output = f"Fertig ({len(result.response)} Zeichen)"
        except Exception as e:
            step.output = f"Fehler: {str(e)}"
            await cl.Message(content=f"❌ **Fehler:** {str(e)}").send()
            return

    await cl.Message(content=result.response).send()

    # Quick switch action
    await cl.Message(
        content="**Weitere Aktionen:**",
        actions=[
            cl.Action(name="switch_to_external", label="🌐 Externe Recherche starten"),
            cl.Action(name="new_query", label="🔄 Neue Anfrage")
        ]
    ).send()


@cl.action_callback("switch_to_external")
async def on_switch_to_external(action: cl.Action):
    """Switch to external mode and start masking workflow"""
    await action.remove()
    original_query = cl.user_session.get("original_query", "")
    if not original_query:
        await cl.Message(content="❌ Keine Anfrage gefunden.").send()
        return

    cl.user_session.set("mode", MODE_EXTERNAL)
    await cl.Message(content="## 🔄 Wechsle zu externer Recherche...").send()
    await start_masking_workflow(original_query)


async def process_files(elements) -> str:
    """Extract text content from uploaded files"""
    content_parts = []

    for element in elements:
        if hasattr(element, 'path') and element.path:
            try:
                # Read text files
                if element.mime and 'text' in element.mime:
                    with open(element.path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_parts.append(f"**{element.name}:**\n```\n{content[:5000]}\n```")
                else:
                    content_parts.append(f"**{element.name}:** [Datei hochgeladen - {element.mime}]")
            except Exception as e:
                content_parts.append(f"**{element.name}:** [Fehler beim Lesen: {str(e)}]")

    return "\n\n".join(content_parts)


async def start_masking_workflow(user_query: str):
    """Step 1: Analyze and mask PII/IP in user query"""

    # Store original query
    cl.user_session.set("original_query", user_query)

    # Show processing indicator
    processing_msg = cl.Message(content="")
    await processing_msg.send()

    async with cl.Step(name="🔒 Datenschutz-Analyse", type="tool") as step:
        step.input = "Analysiere Text auf sensible Daten..."

        try:
            # Call masking agent
            result = await masking_agent.analyze_and_mask(user_query)

            # Store results in session
            cl.user_session.set("pending_query", result.masked_text)
            cl.user_session.set("mask_mapping", result.mask_mapping)
            cl.user_session.set("detected_entities", result.detected_entities)

            step.output = f"Analyse abgeschlossen (Konfidenz: {result.confidence:.0%})"

        except Exception as e:
            step.output = f"Fehler: {str(e)}"
            await cl.Message(content=f"❌ **Fehler bei der Analyse:** {str(e)}").send()
            return

    # Remove processing indicator
    await processing_msg.remove()

    # Check if any entities were detected
    detected_entities = cl.user_session.get("detected_entities", {})

    if detected_entities:
        # Show what was detected and masked
        entity_summary = masking_agent.get_entity_summary(detected_entities)

        await cl.Message(
            content=f"""## 🔍 Datenschutz-Analyse abgeschlossen

{entity_summary}

---

### 📤 Bereinigte Anfrage für externe Recherche:

> {result.masked_text}

---

**⚠️ Bitte prüfe die bereinigte Anfrage oben.**
Die maskierten Platzhalter (z.B. `[PERSON_1]`, `[COMPANY_1]`) ersetzen sensible Daten."""
        ).send()
    else:
        # Warning for long documents with no detections
        if len(user_query) > 500:
            await cl.Message(
                content=f"""## ⚠️ Warnung: Keine sensiblen Daten erkannt

Das Dokument ist **{len(user_query)} Zeichen** lang, aber es wurden keine sensiblen Daten gefunden.

**Bitte prüfe manuell** ob dies korrekt ist, bevor du fortfährst!

Bei Dokumenten mit Namen, E-Mails oder Firmendaten sollten normalerweise Daten erkannt werden.

### 📤 Anfrage für externe Recherche:

> {result.masked_text[:500]}{'...' if len(result.masked_text) > 500 else ''}"""
            ).send()
        else:
            await cl.Message(
                content=f"""## ✅ Keine sensiblen Daten erkannt

Deine Anfrage enthält keine erkennbaren personenbezogenen oder vertraulichen Daten.

### 📤 Anfrage für externe Recherche:

> {result.masked_text}"""
            ).send()

    # Ask for approval
    cl.user_session.set("workflow_state", "awaiting_approval")

    actions = [
        cl.Action(
            name="approve",
            payload={"action": "approve"},
            label="✅ Freigeben & Recherchieren"
        ),
        cl.Action(
            name="edit",
            payload={"action": "edit"},
            label="✏️ Bearbeiten"
        ),
        cl.Action(
            name="cancel",
            payload={"action": "cancel"},
            label="❌ Abbrechen"
        )
    ]

    await cl.Message(
        content="**Wie möchtest du fortfahren?**",
        actions=actions
    ).send()


@cl.action_callback("approve")
async def on_approve(action: cl.Action):
    """Handle approval - proceed with research"""
    await action.remove()

    masked_query = cl.user_session.get("pending_query")
    mask_mapping = cl.user_session.get("mask_mapping", {})
    original_query = cl.user_session.get("original_query")

    if not masked_query:
        await cl.Message(content="❌ Keine ausstehende Anfrage gefunden.").send()
        return

    # Show research in progress
    await cl.Message(content="## 🔍 Starte externe Recherche...").send()

    async with cl.Step(name="🌐 Perplexity Recherche", type="llm") as step:
        step.input = masked_query

        try:
            # Call research agent
            research_result = await research_agent.research(masked_query)
            step.output = f"Recherche abgeschlossen ({len(research_result.response)} Zeichen)"

        except Exception as e:
            step.output = f"Fehler: {str(e)}"
            await cl.Message(content=f"❌ **Recherche fehlgeschlagen:** {str(e)}").send()
            cl.user_session.set("workflow_state", "idle")
            return

    # Show raw research result
    await cl.Message(
        content=f"""## 📥 Rechercheergebnis (roh)

{research_result.response[:2000]}{'...' if len(research_result.response) > 2000 else ''}"""
    ).send()

    # Process with reasoning agent
    async with cl.Step(name="🧠 Verarbeitung & Reasoning", type="llm") as step:
        step.input = "Verarbeite Rechercheergebnis..."

        try:
            # Call reasoning agent
            final_result = await reasoning_agent.process_and_respond(
                original_query=original_query,
                research_response=research_result.response,
                mask_mapping=mask_mapping
            )
            step.output = "Verarbeitung abgeschlossen"

        except Exception as e:
            step.output = f"Fehler: {str(e)}"
            await cl.Message(content=f"❌ **Verarbeitung fehlgeschlagen:** {str(e)}").send()
            cl.user_session.set("workflow_state", "idle")
            return

    # Send final response
    await cl.Message(
        content=f"""## ✨ Finale Antwort

{final_result.final_response}"""
    ).send()

    # Reset state
    cl.user_session.set("workflow_state", "idle")
    cl.user_session.set("pending_query", None)

    # Offer follow-up actions
    followup_actions = [
        cl.Action(
            name="generate_requirements",
            payload={"action": "requirements"},
            label="📋 Anforderungen generieren"
        ),
        cl.Action(
            name="new_query",
            payload={"action": "new"},
            label="🔄 Neue Anfrage"
        )
    ]

    await cl.Message(
        content="**Weitere Aktionen:**",
        actions=followup_actions
    ).send()


@cl.action_callback("edit")
async def on_edit(action: cl.Action):
    """Handle edit request"""
    await action.remove()

    masked_query = cl.user_session.get("pending_query", "")

    await cl.Message(
        content=f"""## ✏️ Bearbeite die Anfrage

Aktuelle bereinigte Anfrage:
```
{masked_query}
```

**Gib deine bearbeitete Version ein:**"""
    ).send()

    cl.user_session.set("workflow_state", "editing")


@cl.action_callback("cancel")
async def on_cancel(action: cl.Action):
    """Handle cancellation"""
    await action.remove()

    cl.user_session.set("workflow_state", "idle")
    cl.user_session.set("pending_query", None)

    await cl.Message(content="❌ **Anfrage abgebrochen.** Du kannst jederzeit eine neue Frage stellen.").send()


@cl.action_callback("generate_requirements")
async def on_generate_requirements(action: cl.Action):
    """Generate requirements from research results"""
    await action.remove()

    original_query = cl.user_session.get("original_query", "")

    await cl.Message(content="## 📋 Generiere Anforderungen...").send()

    async with cl.Step(name="📋 Requirements Engineering", type="llm") as step:
        try:
            # Get the last research response from session if available
            requirements = await reasoning_agent.generate_requirements(
                research_response="(Basierend auf der letzten Recherche)",
                context=original_query
            )
            step.output = "Anforderungen generiert"

            await cl.Message(content=f"## 📋 Generierte Anforderungen\n\n{requirements}").send()

        except Exception as e:
            step.output = f"Fehler: {str(e)}"
            await cl.Message(content=f"❌ **Fehler:** {str(e)}").send()


@cl.action_callback("new_query")
async def on_new_query(action: cl.Action):
    """Start fresh with a new query"""
    await action.remove()

    cl.user_session.set("workflow_state", "idle")
    cl.user_session.set("pending_query", None)

    await cl.Message(content="🔄 **Bereit für eine neue Anfrage!** Gib deine Frage ein.").send()


@cl.on_chat_end
async def end():
    """Clean up on session end"""
    pass


@cl.on_stop
async def stop():
    """Handle stop request"""
    cl.user_session.set("workflow_state", "idle")
    await cl.Message(content="⏹️ Workflow gestoppt.").send()
