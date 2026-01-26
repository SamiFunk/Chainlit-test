# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run development server with auto-reload:**
```bash
chainlit run app.py -w
```

**Run production server:**
```bash
chainlit run app.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Reinitialize config:**
```bash
chainlit init
```

## Architecture

This is the **theo Research Assistant** - a privacy-first AI assistant that masks PII/IP before external research.

### Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. User submits question (+ optional files)                        │
│                          ↓                                          │
│  2. Claude identifies & masks sensitive data (PII/IP)              │
│     • Names → [PERSON_1]                                            │
│     • Companies → [COMPANY_1]                                       │
│     • etc.                                                          │
│                          ↓                                          │
│  3. Sanitized query shown to user for approval                     │
│     (Human-in-the-Loop)                                             │
│                          ↓                                          │
│  4. Perplexity performs external web research                      │
│                          ↓                                          │
│  5. Claude processes response & delivers final answer              │
└─────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
theo-demo/
├── app.py                    # Main Chainlit application (workflow orchestration)
├── agents/                   # AI Agent modules
│   ├── __init__.py
│   ├── masking_agent.py      # Claude-based PII/IP detection & masking
│   ├── research_agent.py     # Perplexity-based external research
│   └── reasoning_agent.py    # Claude-based response processing
├── utils/                    # Utility functions
│   └── __init__.py
├── public/                   # Static assets
│   └── css/
│       └── notion-theme.css  # Notion-inspired UI theme
├── .chainlit/
│   ├── config.toml           # Chainlit configuration
│   └── translations/         # Multi-language support
├── .env                      # API keys (OpenRouter)
├── requirements.txt          # Python dependencies
└── chainlit.md               # Welcome screen content
```

### Key Components

**Agents (via OpenRouter API):**
- `MaskingAgent` - Uses Claude to detect and mask PII/IP categories
- `ResearchAgent` - Uses Perplexity for web research
- `ReasoningAgent` - Uses Claude to process and finalize responses

**Chainlit Patterns:**
- `@cl.on_message` - Main message handler
- `@cl.on_chat_start` - Session initialization
- `@cl.action_callback` - Handle button clicks (approve/edit/cancel)
- `cl.user_session` - Store workflow state per session
- `cl.Step` - Visual workflow progress indicators
- `cl.Action` - Interactive buttons for Human-in-the-Loop

### PII/IP Categories Detected

**Personal Data (PII):**
- PERSON, EMAIL, PHONE, ADDRESS
- DATE_OF_BIRTH, ID_NUMBER, BANK_ACCOUNT, HEALTH

**Intellectual Property (IP):**
- COMPANY, PROJECT, PRODUCT
- FINANCIAL, STRATEGY, TECHNICAL, CLIENT

### Configuration

Environment variables in `.env`:
- `OPENROUTER_API_KEY` - API key for OpenRouter
- `CLAUDE_MODEL` - Model for masking/reasoning (default: anthropic/claude-3.5-sonnet)
- `PERPLEXITY_MODEL` - Model for research (default: perplexity/llama-3.1-sonar-large-128k-online)

### UI/UX

- Notion-inspired clean theme (custom CSS)
- German language interface
- Step-by-step workflow visualization
- Action buttons for user control
