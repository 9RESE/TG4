### DEV NOTES DO NOT EDIT
Update the docs based on all recent git changes. Review git log and diff to identify what changed, then update relevant documentation (architecture, features, ADRs, etc.) following the CLAUDE.md standards (Arc42, Diataxis, C4). Commit the documentation updates.


# Task: Begin TripleGain Phase 1 Implementation 
## Context
Read these files first:
1. CLAUDE.md - Project memory and constraints
2. docs/development/TripleGain-implementation-plan/01-phase-1-foundation.md - Phase 1 spec
3. docs/development/TripleGain-master-design/README.md - System design overview
## Existing Infrastructure (DO NOT recreate)
- TimescaleDB with 5-9 years historical data
- Data collectors in data/kraken_db/ (websocket_db_writer.py, gap_filler.py, etc.)
- Ollama with Qwen 2.5 7B at /media/rese/2tb_drive/ollama_config/
## Phase 1 Deliverables
1. Indicator Library (src/data/indicator_library.py)
2. Market Snapshot Builder (src/data/market_snapshot.py)
3. Prompt Template System (src/llm/prompt_builder.py)
4. Database tables for agent outputs
## Constraints
- Leverage existing TimescaleDB schema and collectors
- No code duplication - reuse data layer
- Test-first approach (pytest)
- Target structure: triplegain/src/
Begin by exploring the existing data/kraken_db/ code, then create the project structure and start with the indicator library