---
description: Act as a critic for the recent output or plan
---

---

# Evidence-Based Critique

## 1. Search for Validation
Conduct web searches to validate planning decisions:

**Search 1**: `[feature-type] best practices [tech-stack] 2026`
- Example: "real-time notifications React best practices 2026"
- Capture: Recommended architectural patterns, performance tips

**Search 2**: `[feature-type] common pitfalls [tech-stack]`
- Example: "WebSocket memory leaks Node.js"
- Capture: Known issues, mitigation strategies

**Search 3**: `[specific-algorithm-or-pattern] performance [constraints]`
- If applicable (e.g., "batch processing large datasets Redis")
- Capture: Benchmarks, optimization techniques

## 2. Critique Against Evidence
For each search result, evaluate:
- **Alignment**: Does our plan match industry best practice?
- **Gaps**: What did we miss or underestimate?

## 3. Document Critique
Update `.ai-context/plans/[feature-slug]-plan.md` section 7:
```markdown
## 7. Critique & Revisions

### Critique Process
- [Search 1 result]: Findings & alignment
- [Search 2 result]: Findings & alignment
- [Search 3 result]: Findings & alignment

### Score Before Revision
- **Plan Score**: [0-10] - [reasoning]

### Identified Gaps
- Gap 1: [What we missed]
- Gap 2: [What we missed]
- Gap 3: [What we missed]

### Architectural Pattern Deviation (if any)
- **Pattern**: [Which pattern being extended/modified]
- **Reason**: [Why deviation is necessary]
- **Evidence**: [Link to search results justifying deviation]
- **Risk Mitigation**: [How we handle the deviation safely]
```

call /replan