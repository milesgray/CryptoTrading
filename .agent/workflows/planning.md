---
description: Generate a detailed plan for a proposed change
---


## Phase 1: Detailed Planning

### 1.1 Create Initial Plan
Generate a detailed `.ai-context/plans/[feature-slug]-plan.md` with:

```markdown
# Feature Plan: [Feature Name]

## 1. Feature Overview
- **Description**: [User input description]
- **Goals**: [Derived success criteria]
- **Scope Boundaries**: [What's IN and OUT]
- **User Value**: [Why this matters]

## 2. Architecture & Design
- **Pattern(s) Used**: [From systemPatterns.md]
- **Component Structure**: [High-level breakdown]
- **Data Models**: [Entity/schema design]
- **Integration Points**: [How it connects to existing system]
- **Dependencies**: [Internal & external]

## 3. Technical Implementation
- **Tech Stack**: [Which technologies from techContext.md]
- **Key Algorithms/Patterns**: [Specific implementation approach]
- **Performance Considerations**: [If applicable]
- **Security Considerations**: [If applicable]

## 4. Acceptance Criteria
- [ ] Criterion 1: [Specific, measurable]
- [ ] Criterion 2: [Specific, measurable]

## 5. Task Breakdown
- **Task 1**: [Specific deliverable]
- **Task 2**: [Specific deliverable]

## 6. Risk Assessment
- **Risk 1**: [Potential issue & mitigation]
- **Risk 2**: [Potential issue & mitigation]

## 7. Critique & Revisions
*[Updated during Phase 4]*
- **Initial Plan Score**: [Will be set after critique]
- **Evidence-Based Revisions**: [Changes based on research]

## 8. Decision Points for User
*[Listed when ready for user feedback]*
- **Decision 1**: [Option A] vs [Option B] - [Tradeoffs]
- **Decision 2**: [Option A] vs [Option B] - [Tradeoffs]
```

### 1.2 Apply Known Patterns
- Use systemPatterns.md templates for component structure
- Apply established naming conventions from codebase
- Follow tech stack best practices from techContext.md

---

## Phase 2: User Review & Feedback Loop

### 2.1 Present Plan to User
Display the following in compressed format:
```
✓ Feature: [Name]
✓ Plan Score: [X/10]
✓ Status: Ready for Implementation
⚠ Decision Points: [Number requiring user input]

[Summary of major architectural decisions]

[List any deviations from established patterns with justifications]

[Decision points presented as numbered options]
```

### 2.2 Gather User Decisions
Present decision points one at a time:
- Show both options clearly
- Include pros/cons
- Highlight recommendation
- Wait for user selection before proceeding

### 2.3 Update Plan Based on Feedback
- Integrate user decisions into plan
- Recalculate affected tasks/timeline
- Update activeContext.md with finalized decisions

---

run /replan

produce finalized `.ai-context/plans/[feature-slug]-plan.md` 