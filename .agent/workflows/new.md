---
description: Initialize a new feature with full memory system integration, architectural validation, and evidence-based planning
---

## Phase 1: Memory System Initialization

### 1.1 Parse Input
- Extract feature description from `/new "..."` command
- Validate input is non-empty and descriptive (50+ chars recommended)
- If invalid, request clarification

### 1.2 Initialize Memory Session
- **Create task log entry**: `task-log_YYYY-MM-DD-HH-MM_feature-init.md`
- **Set working context**: Update `.ai-context/core/activeContext.md` with:
  - Feature name (slugified for branch naming)
  - Feature description (raw user input)
  - Session start timestamp
  - Memory reset point reference
- **Create feature plan file**: `.ai-context/plans/[feature-slug]-plan.md` (empty template)

### 1.3 Update Memory Index
- Add entry to `.ai-context/memory-index.md`:
  ```
  ## [Feature Name]
  - **Status**: Planning Phase
  - **Description**: [Feature Description]
  - **Plan File**: `.ai-context/plans/[feature-slug]-plan.md`
  - **Task Log**: `task-logs/task-log_YYYY-MM-DD-HH-MM_feature-init.md`
  - **Branch**: `feature/[feature-slug]`
  - **Initialized**: YYYY-MM-DD HH:MM
  ```

---

## Phase 2: Feature Context Extraction

### 2.1 Read Existing Architecture
- **Read** `.ai-context/core/systemPatterns.md` (existing patterns & architecture)
- **Read** `.ai-context/core/techContext.md` (tech stack & constraints)
- **Read** `.ai-context/core/productContext.md` (product vision & user needs)
- **Store in ephemeral context** for cross-reference during planning

### 2.2 Identify Architectural Fit
- Document which existing patterns apply to this feature
- Flag potential pattern conflicts or extensions needed
- Note tech stack constraints that affect feature design

---

## Phase 3: Branch Creation & Context Preparation

### 3.1 Create Feature Branch
```bash
git checkout -b feature/[feature-slug]
```
- Branch naming: `feature/[kebab-case-description]`
- Ensure clean commit history from main

### 3.2 Subagent Fan out
- run /planning
- creates `.ai-context/plans/[slug]-plan.md`

### 3.2 Prepare Implementation Context File
Create `.ai-context/core/activeContext.md` with compressed, actionable info based on `.ai-context/plans/[slug]-plan.md`:

```markdown
# Active Context: [Feature Name]

## Quick Reference
- **Feature**: [Name]
- **Branch**: `feature/[slug]`
- **Plan File**: `.ai-context/plans/[slug]-plan.md`
- **Status**: Ready for Implementation

## Executive Summary
[2-3 sentence summary of what's being built]

## Architecture Overview
[Simplified diagram or text description of major components]

## Tech Stack for This Feature
- [Technology 1]: [Purpose]
- [Technology 2]: [Purpose]

## Key Files to Create/Modify
- File path 1: [What it contains]
- File path 2: [What it contains]
- File path 3: [What it contains]

## Critical Implementation Details
1. [Most important architectural decision]
2. [Most important performance consideration]
3. [Most important integration point]

## Acceptance Criteria (Copy from Plan)
- [ ] Criterion 1
- [ ] Criterion 2

## Known Risks & Mitigations
- Risk 1: [How to avoid]
- Risk 2: [How to avoid]

## Next Prompt
Read `.ai-context/plans/[slug]-plan.md` for detailed implementation plan.
Then proceed with Phase 1 of [specific implementation approach].
```

### 3.3 Update Memory System Files
- **progress.md**: Add feature to roadmap with "Planning Complete" status
- **projectbrief.md**: Update if feature scope expands project understanding
- **memory-index.md**: Update status from "Planning Phase" to "Ready for Implementation"

---




## Phase 4: Implementation Handoff

### 4.1 Generate Summary Markdown
Create a single compressed markdown file containing:

```markdown
# Implementation Handoff: [Feature Name]

## Execution Summary
- **Feature**: [Feature Name]
- **Branch**: `feature/[slug]`
- **Plan Score**: [X/10]
- **Estimated Effort**: [X days]
- **Critical Path**: [Major tasks in sequence]

## Architecture Decision Map
[Visual or structured representation of key decisions and their reasoning]

## File-by-File Implementation Guide
### File: [path/to/file1.ext]
- **Purpose**: [What this file does]
- **Changes**: [What needs to be added/modified]
- **Integration Points**: [Where it connects to other files]
- **Key Patterns**: [Specific patterns to follow]

### File: [path/to/file2.ext]
- [Same structure]

## Implementation Sequence
1. [Task 1]: [Why it's first]
2. [Task 2]: [Dependencies]

## Validation Checklist
- [ ] [Acceptance criterion 1]
- [ ] [Acceptance criterion 2]

## Performance Targets
- [Metric 1]: [Target value]
- [Metric 2]: [Target value]

## Emergency Rollback Plan
- If [major issue], roll back with: [specific commands]
- Contact points: [Team leads, architecture owner]

## Reference Links
- Full Plan: `.ai-context/plans/[slug]-plan.md`
- Architecture Patterns: `.ai-context/core/systemPatterns.md`
- Tech Stack Details: `.ai-context/core/techContext.md`
```

### 4.2 Save Handoff Document
- Save to: `.ai-context/plans/[feature-slug]-handoff.md`
- Update task log with completion status and performance score
- Commit all memory files and handoff document to feature branch

### 4.3 Clear for Implementation
- Confirm all memory files are synchronized
- Verify branch is checked out and clean
- Display handoff markdown to user
- Prompt user to proceed with implementation using the handoff document

---

## Success Criteria for Workflow

A successful `/new` workflow execution results in:

- ✓ Complete, evidence-based feature plan (score ≥ 7/10)
- ✓ All memory system files updated
- ✓ Feature branch created and ready
- ✓ All user decisions documented
- ✓ Architectural patterns validated or justified deviations
- ✓ Handoff document ready for implementation phase
- ✓ Zero ambiguity in implementation approach
- ✓ Risk mitigation strategies in place

**Workflow Score Calculation**:
- Plan Quality: /10 (from Phase 5)
- Memory System Update: +2 (if complete)
- User Engagement: +1 (if all decisions made)
- Architectural Alignment: +2 (if validated or justified)
- **Max Score: 15** (Excellent execution at ≥13)

---

## Error Handling & Recovery

### If Research Finds Major Architectural Conflicts
1. **Document conflict**: Add to plan section 7
2. **Propose solutions**: Present 2-3 evidence-based alternatives
3. **Engage user**: Get decision on how to proceed
4. **Recalculate effort**: Update timeline if architecture changes

### If User Can't Decide on Options
1. **Provide strong recommendation** based on architecture alignment and evidence
2. **Explain rationale**: Why this option fits better
3. **Document as decision**: Record the choice in plan
4. **Proceed with recommendation** unless user explicitly overrides

### If Feature Scope Expands During Planning
1. **Flag scope creep**: Ask user if expansion is intentional
2. **Split into phases**: Primary feature + future enhancements
3. **Create separate plans**: One for MVP, document future work
4. **Update memory**: Adjust projectbrief.md and progress.md accordingly

### If Architecture Pattern Not Found in systemPatterns.md
1. **Research the pattern**: Conduct web search for best practices
2. **Document as extension**: Create pattern in systemPatterns.md
3. **Justify addition**: Link to evidence supporting the pattern
4. **Get user approval**: Confirm before using new pattern

---

## Quick Reference: File Locations

| File | Purpose | Phase Updated |
|------|---------|---|
| `.ai-context/core/activeContext.md` | Current feature context | 1, 7 |
| `.ai-context/plans/[slug]-plan.md` | Detailed feature plan | 3, 4, 5 |
| `.ai-context/plans/[slug]-handoff.md` | Implementation guide | 9 |
| `.ai-context/memory-index.md` | Master index | 1, 8 |
| `.ai-context/core/systemPatterns.md` | Architecture patterns | 2 (read) |
| `.ai-context/core/techContext.md` | Tech stack info | 2 (read) |
| `.ai-context/core/progress.md` | Project roadmap | 7 |
| `task-logs/task-log_*_feature-init.md` | Execution log | 1, 9 |

---

## Integration with Cascade Memories

After workflow completion, consider creating a Cascade Memory:

```
Create a memory of:
Feature: [Name]
Architecture: [Key patterns used]
Tech Stack: [Technologies deployed]
Critical Decisions: [Major choices made]
Risk Mitigations: [How risks are handled]
Reference: .ai-context/plans/[slug]-plan.md
```

This ensures the feature context persists across context window resets during implementation.

---

## Workflow State Machine

```
START
  ↓
[Phase 1: Memory Init] → Success?
  ├─ No → Request clarification
  └─ Yes ↓
[Phase 2: Context Extraction] → Success?
  ├─ No → Re-read architecture
  └─ Yes ↓
[Phase 3: Planning] → Success?
  ├─ No → Rework plan
  └─ Yes ↓
[Phase 4: Handoff] → Complete?
  ├─ No → Finalize documents
  └─ Yes ↓
END: Ready for Implementation
```