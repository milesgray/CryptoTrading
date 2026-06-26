---
auto_execution_mode: 3
description: Extract a skill from a completed conversation or PR
---
# Workflow: Extract Skill from Conversation or PR

**Purpose:**  Analyze a completed conversation and/or PR to determine whether a reusable skill should be extracted, updated, or left alone. Defaults to **no action** unless there is a clear, non-duplicative reason to act.

---

## Step 1 — Gather Inputs

Collect the artifacts to analyze:

- [ ] Full conversation transcript (or a summary of the key exchanges)
- [ ] PR diff and/or PR description, if applicable
- [ ] List of any error messages, wrong turns, or corrections made during the work
- [ ] Final working solution or output

> **Note:** If you only have a PR without a conversation, skip to Step 2 using the PR diff as the source of truth.

---

## Step 2 — Characterize the Work

Read through the inputs and answer these questions. Keep answers brief.

1. **What task was being accomplished?** (one sentence)
2. **What tools, libraries, or APIs were central to the solution?**
3. **Were there any non-obvious steps, gotchas, workarounds, or environment-specific constraints?**
4. **Was there significant back-and-forth or correction before arriving at a working solution?** (If no, this is a signal that no skill may be needed.)
5. **Is the output format/structure reusable, or highly specific to this one-off case?**

If questions 3 and 4 are both "no" / "none", stop here. **No skill action needed.** Document why and exit.

---

## Step 3 — Audit Existing Skills

Before creating anything, scan all available skills to check for overlap.

```
.agent/skills/
.claude/skills/
```

For each candidate skill found, assess:

| Existing Skill | Overlap Level | Notes |
|---|---|---|
| `<skill-name>` | None / Partial / High | What overlaps, what differs |

**Decision rules:**

- **High overlap** → Update the existing skill. Do NOT create a new one.
- **Partial overlap** → Evaluate carefully. If the new knowledge fits naturally as a section or note in the existing skill, update it. If it's genuinely a different domain, consider a new skill.
- **No overlap** → A new skill is a candidate, but only if the threshold in Step 4 is met.

---

## Step 4 — Apply the "Worth Capturing" Threshold

A skill earns its existence by saving meaningful future effort. Before proceeding, the work must clear **at least two** of the following bars:

- [ ] **Reoccurrence** — This type of task is likely to come up again (not a one-off edge case)
- [ ] **Non-obvious** — The solution required knowledge that isn't readily available from docs or general reasoning alone
- [ ] **Pitfall-rich** — There were real mistakes, dead ends, or environment quirks that future work should avoid
- [ ] **Multi-step** — The workflow has enough steps that a prompt-level reminder would genuinely help Claude execute it correctly
- [ ] **Time-costly** — Getting it wrong wastes significant time (e.g., environment setup, API auth flows, build configs)

If fewer than two boxes are checked, **stop. No skill action needed.**

---

## Step 5 — Draft the Skill Content

Only proceed here if Steps 3–4 justify action.

### 5a. If Updating an Existing Skill

1. Copy the existing `SKILL.md` to a writable location:
   ```bash
   cp .agent/skills/<path>/SKILL.md /tmp/<skill-name>/SKILL.md
   ```
2. Identify exactly what to add:
   - New pitfall or gotcha → add to a `## Known Issues / Pitfalls` section
   - New step or variant → add to the relevant section of the workflow
   - New environment constraint → add to `## Compatibility` or a notes block
3. Make the minimal change that captures the new knowledge. Do not restructure the whole skill.
4. Preserve the original `name` frontmatter field and directory name exactly.

### 5b. If Creating a New Skill

Follow the structure from the `skill-creator` skill:

```
<skill-name>/
├── SKILL.md          ← required
└── references/       ← optional, for large supporting docs
```

**SKILL.md frontmatter (required fields):**
```yaml
---
name: <skill-name>
description: >
  <What it does. When to trigger it. Key contexts or phrases that should
  activate this skill. Be slightly "pushy" — mention trigger contexts
  explicitly to avoid undertriggering.>
---
```

**Body must include:**
- The core workflow or steps, in order
- Any environment-specific constraints or dependencies
- A `## Known Pitfalls` section capturing every wrong turn from the source conversation/PR
- Expected inputs and outputs
- Keep under 500 lines; offload large reference material to `references/`

---

## Step 6 — Sanity Check Before Saving

Before writing the final file, answer these questions:

- [ ] Does this duplicate anything already captured in an existing skill?
- [ ] Is the `description` specific enough that Claude won't confuse it with another skill?
- [ ] Does the `description` mention the contexts and phrases that should trigger it?
- [ ] Are the pitfalls from the original conversation/PR explicitly called out?
- [ ] Is this scoped narrowly enough to be unambiguous, but broadly enough to be reusable?

If any box is unchecked, revise before saving.

---

## Step 7 — Final Output

### If No Action Was Taken

Write a brief note explaining the decision:

```
## Skill Extraction Decision

**Date:** <date>
**Source:** <conversation title or PR link>
**Decision:** No action taken

**Reason:** <e.g., "Task was highly one-off with no reusable pattern"
             or "Existing `docx` skill already covers this domain" >
```

### If Skill Was Updated

```
## Skill Extraction Decision

**Date:** <date>
**Source:** <conversation title or PR link>
**Decision:** Updated existing skill `<skill-name>`

**Changes made:**
- <bullet: what was added or changed>
- <bullet>

**Pitfalls captured:**
- <bullet: specific mistake or gotcha from the source work>
```

### If New Skill Was Created

```
## Skill Extraction Decision

**Date:** <date>
**Source:** <conversation title or PR link>
**Decision:** Created new skill `<skill-name>`

**Justification:**
- Threshold bars cleared: <list which ones>
- No existing skill covers: <explain gap>

**Pitfalls captured:**
- <bullet>
```

---

## Guiding Principles

| Principle | What It Means |
|---|---|
| **Default to nothing** | No action is correct far more often than action. Err heavily on the side of not creating. |
| **Update over create** | A new skill that partially overlaps an existing one creates confusion at trigger time. Merge first. |
| **Pitfalls are the payload** | The most valuable part of any extracted skill is the *wrong turns* — what to avoid. Don't skip this. |
| **Descriptions drive triggering** | A skill that never triggers is useless. The description is more important than the body. |
| **Minimal footprint** | Skills should be focused and tightly scoped. A skill that tries to cover too much becomes noise. |
