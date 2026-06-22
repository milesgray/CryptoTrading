---
description: Reformulate a plan after criticized
---


## Phase 5: Plan Reformulation

### 5.1 Integrate Evidence
Update each section of the plan with findings:
- Strengthen weak architectural decisions
- Add missing tasks identified through research
- Adjust risk assessments based on real-world issues
- Refine performance/security considerations

### 5.2 Recalculate Plan Score
Re-evaluate plan quality after revisions:
- **Excellent Plan**: 9-10 (comprehensive, well-researched, mitigated risks)
- **Good Plan**: 7-8 (solid foundation, minor gaps)
- **Acceptable Plan**: 5-6 (viable but needs refinement)
- **Weak Plan**: <5 (significant risks or gaps - rework needed)

### 5.3 User Decision Points
Identify areas needing user input and format as decisions:

```markdown
## 8. Decision Points for User

### Decision 1: [Choice Topic]
**Question**: [What decision needs to be made?]

**Option A**: [Approach description]
- Pros: [Benefits]
- Cons: [Tradeoffs]
- Estimate: [Time/complexity impact]

**Option B**: [Approach description]
- Pros: [Benefits]
- Cons: [Tradeoffs]
- Estimate: [Time/complexity impact]

**Recommendation**: [Which option aligns with architecture/constraints]

**User Input Needed**: Yes/No
```

---

## Phase 6: Architectural Alignment Check

### 6.1 Pattern Validation
For each architectural pattern used:
- ✓ Confirm pattern exists in `systemPatterns.md`
- ✓ Verify pattern is applicable to this feature type
- ✓ Check for conflicts with other patterns
- ✓ Confirm team is trained on this pattern (or document learning curve)

### 6.2 Tech Stack Validation
- ✓ All technologies listed in `techContext.md`
- ✓ Version compatibility checked (if applicable)
- ✓ Dependencies don't conflict with existing stack
- ✓ Performance impact assessed vs constraints

### 6.3 Deviation Justification
If deviating from established patterns:
- Clearly document the pattern being deviated from
- Provide evidence-based justification (from web search or architectural analysis)
- Detail risk mitigation strategy
- Get explicit user approval before proceeding

---
