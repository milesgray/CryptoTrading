---
description: Close out a completed task — run tests, update docs & memory, commit, push, and open a PR
---

You are a senior engineer performing a disciplined task closeout. The current task is considered **functionally complete**. Your job is to verify that completion, capture all knowledge, and ship it cleanly.

Execute the following steps in order. Do not skip a step unless it explicitly does not apply to this project.

---
## Step 0 - Git Branch

Each feature should be developed in its own branch. If the current branch is not a feature branch, create one now.

- This will result in a PR to merge the feature branch into main.
- Remember through all of the steps that you are working towards getting this ready to be a new PR

## Step 1 — Verify Task Completion

Before touching anything, confirm the task is actually done:

1. Read the active task description from the project's task tracker (check `TODO.md`, open issue, linear ticket, Jira card, or the user's last instruction — whichever is present).
2. Summarize what was implemented in 2–4 sentences.
3. List any known remaining gaps or TODOs introduced during this task (if none, say so explicitly).
4. If the task is **not** complete, stop and report what is missing. Do not proceed.

---

## Step 2 — Run the Test Suite

1. Detect the test runner by checking in order: `package.json` scripts, `pytest.ini` / `pyproject.toml`, `Makefile`, `Dockerfile`, or CI config files (`.github/workflows/`, `circle.yml`, `.gitlab-ci.yml`).
2. Run the full test suite. Use the appropriate command, for example:
   - Python: `pytest` or `python -m pytest`
   - Node/JS: `npm test`, `yarn test`, or `pnpm test`
   - Other: infer from project structure
3. If tests fail:
   - Attempt to fix **only** test failures that are a direct consequence of the changes made in this task.
   - Do not refactor unrelated code to force tests to pass.
   - If a pre-existing test is broken by design (the task intentionally changed behavior), update the test and note it explicitly.
   - Re-run until green. If you cannot fix a failure, stop and report it — do not proceed with broken tests.
4. Report final test results: total passed, failed, skipped.

---

## Step 3 — Run Linters and Type Checkers (if applicable)

1. Check for linter/formatter configs: `.eslintrc`, `pyproject.toml` / `setup.cfg` (flake8, ruff, black), `.prettierrc`, `mypy.ini`, `tsconfig.json`, etc.
2. Run all applicable checks:
   - Python: `ruff check .` `black --check .`
   - JS/TS: `eslint .` and/or `tsc --noEmit`
3. Auto-fix anything that is safe to auto-fix (`ruff --fix`, `black .`, `eslint --fix`).
4. For remaining lint errors: fix only those introduced by this task. Do not do a repo-wide lint cleanup unless that was the task.
5. Report what was fixed and what (if anything) was left.

---

## Step 4 — Update Documentation

1. Identify all documentation that is affected by this task's changes. Look in:
   - `README.md` (feature lists, usage examples, env vars, install steps)
   - `docs/` directory (architecture docs, API references, guides)
   - Inline docstrings / JSDoc for any modified functions, classes, or modules
   - `CHANGELOG.md` or `HISTORY.md` if present
   - `openapi.yaml` / `swagger.json` if API surface changed
2. Update each affected document. Be concise and accurate. Do not pad.
3. If a `CHANGELOG.md` exists, prepend a new entry under an `[Unreleased]` section following the Keep a Changelog format:

   ```
   ## [Unreleased]
   ### Added / Changed / Fixed / Removed
   - <one-line description of this task's change>
   ```

4. Report all files updated.

---

## Step 5 — Update the Memory / Context System

Update the project's persistent memory so future sessions start with accurate context.

1. Check for a memory file. Look for: `.agent/memory.md`, `AGENTS.md`, `CONTEXT.md`, `docs/memory.md`, `.cursor/memory.md`, or any file the project uses as an LLM context anchor.
2. If a memory file exists, update it to reflect:
   - The task that was just completed (one-line summary)
   - Any new environment variables, secrets, or config values introduced
   - Any new dependencies added (`requirements.txt`, `package.json`, etc.)
   - Any architectural decisions made during this task
   - Current known blockers or next steps (if any)
3. If no memory file exists, create `.agent/memory.md` with the above fields using this template:

   ```markdown
   # Project Memory

   ## Last Completed Task
   <summary>

   ## Architecture Notes
   <any decisions made>

   ## Environment / Config
   <new env vars or secrets>

   ## Dependencies Added
   <new packages>

   ## Known Blockers / Next Steps
   <if any>
   ```

4. Do not bloat the memory file. Replace stale entries rather than appending duplicates.

---

## Step 6 — Stage and Review Changes

1. Run `git status` to see all modified, added, and deleted files.
2. Run `git diff --stat` to get a high-level change summary.
3. Review the diff for any files that should **not** be committed:
   - Build artifacts (`dist/`, `__pycache__/`, `.pyc`, `node_modules/`)
   - Local env files (`.env`, `.env.local`) — verify `.gitignore` covers these
   - Scratch files, debug prints, or temporary test code left in accidentally
4. Remove or un-stage any such files.
5. Stage all remaining changes: `git add -A` (or selectively stage if appropriate).

---

## Step 7 — Write the Commit Message

Write a commit message following the **Conventional Commits** spec:

```
<type>(<scope>): <short imperative summary>

<body: what changed and why — omit if obvious>

<footer: closes #<issue>, breaking changes, co-authors>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `perf`

Rules:

- Subject line ≤ 72 characters, imperative mood ("add X", not "added X")
- Body lines ≤ 100 characters
- Reference the issue/ticket number if one exists (`Closes #123`)
- If this is a breaking change, add `BREAKING CHANGE:` in the footer

Commit: `git commit -m "<message>"`

---

## Step 8 — Push to Remote

1. Identify the current branch: `git branch --show-current`
2. If on `main` / `master` directly, **stop** and warn the user — work should be on a feature branch. Ask how to proceed.
3. Push the branch: `git push origin <branch>` (use `--set-upstream` if the branch is new)
4. Report the push result including the remote URL printed by git.

---

## Step 9 — Open a Pull Request

1. Detect the git host by inspecting `git remote get-url origin`:
   - `github.com` → use `gh` CLI if available, otherwise print the PR URL
   - `gitlab.com` → use `glab` CLI if available, otherwise print the MR URL
   - `bitbucket.org` → print the PR URL
2. If the `gh` CLI is available, create the PR:

   ```
   gh pr create \
     --title "<Conventional Commits subject line from Step 7>" \
     --body "<generated PR body — see below>" \
     --base main \
     --draft
   ```

   Open as **draft** by default. The user can mark it ready to review.
3. PR body template (generate this from the task context):

   ```markdown
   ## Summary
   <2–4 sentence description of what this PR does>

   ## Changes
   - <bullet per logical change>

   ## Testing
   - [ ] Tests pass locally (`<test command>`)
   - [ ] Linting clean
   - [ ] Docs updated

   ## Related Issues
   Closes #<issue number if known>
   ```

4. If no CLI is available, output the fully pre-filled PR body so the user can paste it into the browser.

---

## Step 10 — Final Report

Print a clean closeout summary:

```
✅ Task Closeout Summary
========================
Task:        <one-line task description>
Branch:      <branch name>
Commit:      <short SHA> — <commit subject>
Tests:       <X passed, Y skipped, 0 failed>
Lint:        <clean / N issues fixed>
Docs:        <files updated>
Memory:      <file updated or created>
PR:          <URL or "created as draft">
Blockers:    <none / list>
```

If any step was skipped (e.g., no test suite exists), note it explicitly in the summary so the user knows it was considered and not just forgotten.