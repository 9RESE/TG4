# Core Agents Review - Issue Checklist

**Review Date**: 2025-12-19
**Total Issues**: 20 (3 P0, 5 P1, 7 P2, 5 P3)
**Status**: 🔴 Blockers present - fix before production

---

## Priority 0 - CRITICAL (Blockers)

### 🔴 P0-12: Trading Decision Split Logic Error
- **File**: `triplegain/src/agents/trading_decision.py:594-607`
- **Issue**: 3-way tie picks winner alphabetically (2 BUY, 2 SELL, 2 HOLD → BUY wins with 33%)
- **Impact**: Could execute trades with minority support
- **Fix**: Force HOLD if consensus_strength ≤ 0.5
- **Effort**: 10 minutes
- **Test**: Add 3-way tie test case
- [ ] Code fixed
- [ ] Test added
- [ ] PR reviewed
- [ ] Deployed to staging

---

### 🔴 P0-13: Risk Engine Integration Unclear
- **File**: Trading Decision Agent flow
- **Issue**: Design shows Risk validation in agent, implementation separates it
- **Impact**: Unclear responsibility, potential validation bypass
- **Options**:
  - [ ] Option A: Add Risk.validate_trade() to Trading Agent (2 hours)
  - [ ] Option B: Update design spec to show orchestration responsibility (30 min)
- **Decision**: _____________
- [ ] Implemented
- [ ] Documentation updated
- [ ] Team agreement

---

### 🔴 P0-17: Portfolio DCA Rounding Overflow
- **File**: `triplegain/src/agents/portfolio_rebalance.py:496-522`
- **Issue**: Rounding can cause batches to exceed original amount ($99.99 → $100.02)
- **Impact**: Over-leverage, position size violations
- **Fix**: Use ROUND_DOWN, allocate remainder to first batch
- **Effort**: 20 minutes
- **Test**: Add edge case tests ($99.99/6, $0.01/6, $1000.01/6)
- [ ] Code fixed
- [ ] Tests added
- [ ] Edge cases validated
- [ ] PR reviewed

---

## Priority 1 - HIGH (Fix Before Beta)

### 🟠 P1-04: TA Fallback Confidence Too High
- **File**: `triplegain/src/agents/technical_analysis.py:443`
- **Current**: 0.4 (could trigger trades)
- **Target**: 0.2 (clearly insufficient)
- **Effort**: 2 minutes
- [ ] Fixed
- [ ] Test updated

---

### 🟠 P1-08: Regime Parameters Inconsistency
- **File**: `triplegain/src/agents/regime_detection.py:335-350`
- **Issue**: LLM adjustments partially override defaults (mixed sources)
- **Fix**: Use all LLM or all defaults, add _is_complete_adjustment()
- **Effort**: 15 minutes
- [ ] Fixed
- [ ] Logic validated

---

### 🟠 P1-14: Risk Engine Flow Documentation
- **Related to**: P0-13
- **Action**: Update design doc OR add integration
- [ ] Design updated
- [ ] Flow diagram corrected

---

### 🟠 P1-18: Hodl Bag Validation Weak
- **File**: `triplegain/src/agents/portfolio_rebalance.py:345-362`
- **Issue**: Warns but clamps negative to 0 (silent failure)
- **Fix**: Raise ValueError if hodl_bags > balance
- **Effort**: 5 minutes
- [ ] Fixed
- [ ] Exception test added

---

### 🟠 P1-21: Missing Risk Integration Test
- **File**: New integration test needed
- **Test**: TA → Regime → Trading → Risk → Execution
- **Effort**: 2 hours
- [ ] Test created
- [ ] Passing

---

## Priority 2 - MEDIUM (Fix Before Release)

### 🟡 P2-03: SQL Interpolation Risk
- **File**: `triplegain/src/agents/base_agent.py:252`
- **Fix**: Use parameterized query for interval
- **Effort**: 5 minutes
- [ ] Fixed

---

### 🟡 P2-05: TA Indicator Fallback Too Simple
- **File**: `triplegain/src/agents/technical_analysis.py:407-412`
- **Enhancement**: Add regime check or disable fallback trades
- **Effort**: 30 minutes
- [ ] Enhanced
- [ ] Tests updated

---

### 🟡 P2-09: Regime State Not Persisted
- **File**: `triplegain/src/agents/regime_detection.py:305-310`
- **Issue**: Restart loses regime duration tracking
- **Fix**: Persist to database
- **Effort**: 1 hour
- [ ] DB migration
- [ ] Persistence added
- [ ] Tests updated

---

### 🟡 P2-10: Regime Fallback Choppiness Priority
- **File**: `triplegain/src/agents/regime_detection.py:517-520`
- **Issue**: ADX > 25 overrides choppiness > 60
- **Fix**: Check choppiness first
- **Effort**: 10 minutes
- [ ] Fixed
- [ ] Logic tested

---

### 🟡 P2-15: Timeout Models Not Tracked
- **File**: `triplegain/src/agents/trading_decision.py:407-414`
- **Enhancement**: Add ModelDecision with timeout error
- **Effort**: 20 minutes
- [ ] Added
- [ ] Comparison table includes timeouts

---

### 🟡 P2-19: Portfolio Price Cache Race
- **File**: `triplegain/src/agents/portfolio_rebalance.py:617-622`
- **Issue**: Concurrent access to price cache (no lock)
- **Fix**: Add asyncio.Lock
- **Effort**: 10 minutes
- [ ] Lock added
- [ ] Concurrent test added

---

### 🟡 P2-XX: Add Missing Test Scenarios
- [ ] Concurrent process() calls (all agents)
- [ ] Database partial write failures
- [ ] All 6 models timeout cascade
- [ ] Low confidence + split + choppy combo
- **Effort**: 4 hours total

---

## Priority 3 - LOW (Enhancement)

### 🟢 P3-01: Document cache_ttl_seconds
- **File**: Config schema documentation
- [ ] Documented

---

### 🟢 P3-02: _parse_stored_output Override Warning
- **File**: `triplegain/src/agents/base_agent.py:293`
- [ ] Docstring added

---

### 🟢 P3-06: Support/Resistance from Candles
- **File**: `triplegain/src/agents/technical_analysis.py:432-434`
- [ ] Enhanced (optional)

---

### 🟢 P3-07: Reject Invalid Validation Critically
- **File**: `triplegain/src/agents/technical_analysis.py:257`
- [ ] Strict mode added (optional)

---

### 🟢 P3-11: Historical ATR Volatility
- **File**: `triplegain/src/agents/regime_detection.py:524-535`
- [ ] Percentile-based volatility (optional)

---

### 🟢 P3-16: Improve Text Extraction
- **File**: `triplegain/src/agents/trading_decision.py:547-554`
- [ ] NLP enhancement (optional)

---

### 🟢 P3-20: Move Mock Data to Tests
- **File**: `triplegain/src/agents/portfolio_rebalance.py:607-640`
- [ ] Refactored (optional)

---

## Progress Tracker

### By Priority
- **P0 (Critical)**: 0/3 fixed 🔴
- **P1 (High)**: 0/5 fixed 🟠
- **P2 (Medium)**: 0/7 fixed 🟡
- **P3 (Low)**: 0/5 fixed 🟢

### By Status
- [ ] **All P0 fixed** (blocks production)
- [ ] **All P1 fixed** (blocks beta)
- [ ] **All P2 fixed** (blocks release)
- [ ] **P3 enhancements** (optional)

### Estimated Effort
- **P0 fixes**: ~3 hours
- **P1 fixes**: ~4 hours
- **P2 fixes**: ~7 hours
- **P3 enhancements**: ~3 hours
- **Total**: ~17 hours

### Milestones
- [ ] **Stage 1**: P0 fixed → Deploy to staging (1 day)
- [ ] **Stage 2**: P1 fixed → Beta release (1 week)
- [ ] **Stage 3**: P2 fixed → Production candidate (2 weeks)
- [ ] **Stage 4**: P3 done → Full production (3 weeks)

---

## Sign-off

- [ ] **Developer**: All code fixes implemented
- [ ] **QA**: All new tests passing
- [ ] **Architect**: Design alignment confirmed
- [ ] **Security**: No new vulnerabilities
- [ ] **Product**: Acceptance criteria met

**Approved for production**: _______________
**Date**: _______________
**Signature**: _______________

---

## Notes

_Add comments, blockers, or questions here:_

<br/><br/><br/>

---

**Full Review**: `docs/development/reviews/core-agents-deep-review-2025-12-19.md`
**Summary**: `docs/development/reviews/core-agents-review-summary.md`
