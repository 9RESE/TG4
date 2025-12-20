**DEV NOTES**
**DO NOT EDIT**
Update the docs based on all recent git changes. Review git log and diff to identify what changed, then update relevant documentation (architecture, features, ADRs, etc.) following the CLAUDE.md standards (Arc42, Diataxis, C4). Commit Git.


do a deep code and logic review of the implementation of docs/development/TripleGain-implementation-plan/07-phase-7-sentiment-analysis.md Provide a document of your findings and recommendations in this folder docs/development/reviews/phase-7/. ultrathink

address all issues and implement the fixes/recommendations outlined in docs/development/reviews/phase-7/deep-review.md ultrathink

do a deep code and logic review of the implementation of docs/development/TripleGain-implementation-plan/08-phase-8-hodl-bag-system.md and the implementation of the fixes from the reviews in docs/development/reviews/phase-8/ Provide a document of your findings and recommendations in this folder docs/development/reviews/phase-8/. ultrathink

address all issues and implement the fixes/recommendations outlined in docs/development/reviews/phase-8/deep-review-2025-12-20.md ultrathink
- **COMPLETED 2025-12-20**: Phase 8 Deep Review v2 Fixes Implemented
  - H1/H2/H3: Coordinator + paper trading integration (coordinator.py, run_paper_trading.py)
  - M1: Force accumulation now bypasses threshold (hodl_bag.py)
  - M2: Retry logic with configurable attempts (hodl_bag.py)
  - M3: Thread-safe price cache with separate lock (hodl_bag.py)
  - M5: Added is_paper column to hodl_bags (migrations/010_hodl_bags_is_paper.sql)
  - L1: Updated fallback prices to 2025 values (hodl.yaml)
  - L2: Implemented daily snapshot creation (hodl_bag.py)
  - L6: Completed API route tests with 27 tests (test_routes_hodl.py)
  - DEFERRED: M4 (integration tests), L3 (withdrawal), L4 (DCA queue), L5 (slippage), L7 (portfolio rebalance)


what issues do we have with any test?
what are the pre-existing issue with the OpenAI client test
will local qwen running every minute override the api llm trading strategies that run every 30 min?
what prompts are being sent to which llms and when. What are the responses used for?