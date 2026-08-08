---
description: Work through every open review comment on a PR — fix, commit, reply, and resolve each thread — until the PR is clean.
---

---
argument-hint: [pr_number_or_url] [--no-resolve] [--dry-run]
allowed-tools: Bash(gh:*), Bash(git:*), Read, Grep, Glob, Edit
---

You are addressing PR review feedback end-to-end. GitHub PRs have three comment types —
**issue comments** (general, bottom of PR), **review comments** (line-specific, your target),
and **review summary comments** (the approve/request-changes wrapper). This command targets
line-specific review comments on **unresolved threads**, plus scans issue comments for anything
actionable a human reviewer left as a top-level note.

If `$ARGUMENTS` contains `--no-resolve`, fix and reply but leave every thread open for the
reviewer to close themselves. If it contains `--dry-run`, do steps 1–2 only and print a plan —
make no commits, replies, or resolutions.

## Steps

### 1. Identify the PR and repo context

```bash
gh pr view $ARGUMENTS --json number,title,url,headRefName,baseRefName,state 2>/dev/null \
  || gh pr view --json number,title,url,headRefName,baseRefName,state
```

If no PR number/URL is given in `$ARGUMENTS`, use the PR for the current branch. Note the
`owner`, `repo`, and `pr_number` — every call below needs them. Confirm `headRefName` matches
your current checked-out branch; if not, `git fetch` and `git checkout` it before touching code.

### 2. Gather every unresolved thread

Unlike single-comment resolution, this command needs the *full* set of open threads, deduplicated
by thread (a thread can have several comments in it — only reply to/act on the latest one, but
read the whole thread for context).

```bash
gh api graphql -f query='
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 50) {
            nodes {
              id
              databaseId
              body
              author { login }
              createdAt
            }
          }
        }
      }
    }
  }
}' -F owner={owner} -F repo={repo} -F number={pr_number} \
  | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)]'
```

Also pull general issue comments that read like review feedback (not bot noise, not your own
prior replies):

```bash
gh api repos/{owner}/{repo}/issues/{pr_number}/comments \
  | jq '[.[] | select(.user.type != "Bot")]'
```

Build a numbered worklist: one entry per unresolved thread + any actionable issue comments.
Skip threads that are `isOutdated: true` AND clearly refer to code already removed — note these
for the summary instead of acting on them (see step 6).

If `--dry-run` was passed: print the worklist (file, line, author, summarized ask) and **stop
here**.

### 3. Work each item in the list, one at a time

For each thread/comment, in order:

a. **Read for real understanding** — the comment body, the surrounding thread (earlier replies
   may add constraints), and the actual file/lines via `Read`, not just the diff hunk. Use `Grep`
   if the concern is about a pattern that might recur elsewhere in the same PR.

b. **Decide, don't just comply** — reviewer suggestions are sometimes wrong, already handled
   elsewhere, or based on a misread. Think it through. If you disagree, don't silently skip it —
   you still reply (step 3d), just with a reasoned pushback instead of a fix, and you do **not**
   resolve that thread even if `--no-resolve` wasn't passed (leave disagreements for a human).

c. **Implement the fix** — make the smallest change that actually addresses the concern. Don't
   bundle unrelated cleanup into the same commit.

d. **Commit** with a message that references what was addressed, e.g.:
   ```bash
   git add -A
   git commit -m "fix: <what changed>

   Addresses review comment on {path}:{line} from @{author}"
   ```
   One commit per logical fix is fine; don't force a strict 1:1 commit-to-comment ratio if two
   comments are actually the same root cause — just say so in both replies.

e. **Push** once all fixes for this pass are committed (batching pushes is fine, don't push after
   every single commit if you're about to make three more in the next 30 seconds):
   ```bash
   git push
   ```

f. **Reply on the thread** (not a generic PR comment):
   ```bash
   gh api -X POST repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies \
     -f body="Fixed in commit {short_sha}. {one-line description of what changed}."
   ```
   If pushing back instead of fixing, reply with the reasoning instead — no commit reference.

### 4. Run the project's actual checks before resolving anything

Don't resolve threads on faith. Detect and run what the repo actually uses (check for
`package.json` scripts, `pytest`/`tox.ini`, `Makefile`, etc.) — tests, lint, type-check, build.
If something fails because of your change, fix it before moving on; don't resolve a thread whose
fix broke CI.

### 5. Resolve threads (skip entirely if `--no-resolve`)

Only resolve threads where you actually implemented the fix (not the pushback cases from 3b).

```bash
gh api graphql -f query='
mutation {
  resolveReviewThread(input: {threadId: "{thread_id}"}) {
    thread { isResolved }
  }
}'
```

`{thread_id}` is the GraphQL node `id` from step 2 — **not** the `databaseId`/comment ID used in
step 3f. Don't conflate them; they're different objects.

### 6. Post one wrap-up summary comment on the PR

Not one comment per fix — a single roll-up so the reviewer isn't hunting through the thread list.

```bash
gh pr comment {pr_number} --body "$(cat <<'EOF'
## Review feedback addressed

**Fixed & resolved:**
- {path}:{line} — {one-liner} (commit {short_sha})
- ...

**Replied, left open for your call:**
- {path}:{line} — {why: disagreement / needs your input / outdated}

**Not touched:**
- {path}:{line} — {why: thread refers to code no longer present, etc.}

All checks passing: {yes/no, and what you ran}
EOF
)"
```

### 7. Final report to the user (in chat, not the PR)

Give a short summary: how many threads resolved, how many left open and why, what commits were
made, whether CI/checks pass, and anything that needs the user's judgment call before merging.

## Notes

- Thread IDs (GraphQL `id`, used to resolve) and comment IDs (`databaseId`, used to reply) are
  different values from the same API response — grabbing the wrong one is the most common failure
  mode here.
- Only accounts with write access to the repo can resolve review threads; if resolution calls
  fail with a permissions error, say so plainly rather than retrying blindly.
- Never resolve a thread you didn't actually address — an unresolved-but-replied-to thread is a
  perfectly fine outcome when the reviewer needs to have the final word.
- If a review left 20+ comments (common with AI reviewers like Copilot/CodeRabbit), still process
  them individually — don't batch-fix without reading each one, that's how real bugs slip past a
  review meant to catch them.