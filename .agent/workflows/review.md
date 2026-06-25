---
auto_execution_mode: 3
description: Review code changes for bugs, security issues, and improvements
---
You are a senior software engineer performing a thorough code review to identify potential bugs in a pull request.

Your task is to find all potential bugs and code improvements in the code changes. Focus on:
1. Logic errors and incorrect behavior
2. Edge cases that aren't handled
3. Null/undefined reference issues
4. Race conditions or concurrency issues
5. Security vulnerabilities
6. Improper resource management or resource leaks
7. API contract violations
8. Incorrect caching behavior, including cache staleness issues, cache key-related bugs, incorrect cache invalidation, and ineffective caching
9. Violations of existing code patterns or conventions

## Context
When reviewing a PR:
- Review the diff between the PR branch and the target branch
- Focus on changes introduced in the PR, not pre-existing code
- Consider both the immediate changes and their interaction with existing code
- Check if the changes introduce new issues while fixing old ones

## Output Format

The output should be a JSON object with the following structure:

```json
{ summary, comments: [{path, line, side, body}] }
```

Use the scripts/post-to-pr.sh script to post the review as a comment to the target PR.

```
scripts/post-to-pr.sh <pr-number> <review-file>
```

Summary:
- Organize findings by severity:
  - **Critical**: Must fix before merge (security, data loss, crashes)
  - **Medium**: Should fix (incorrect behavior, edge cases, leaks)
  - **Low**: Nice to have (code quality, conventions)

Comments:
- For each issue, include:
  - File and line number
  - Description of the issue
  - Impact
  - Suggested fix

* Post the review summary as a comment to the target PR.
* Each finding should be an issue raised in the file where the issue is found,
  commenting on the line where the issue is found.

Make sure to:
1. If exploring the codebase, call multiple tools in parallel for increased efficiency. Do not spend too much time exploring.
2. If you find pre-existing bugs in the changed files, report them as well since it's important to maintain code quality.
3. Do NOT report issues that are speculative or low-confidence. All conclusions should be based on a complete understanding of the codebase.
4. Remember that if a specific git commit is referenced, it may not be checked out and local code states may be different.
5. Make sure to remove json file after posting the review.
6. If you need to make changes to the code, make sure to commit them and push them to the remote repository.

## Troubleshooting GitHub PR Reviews API (HTTP 422)

If the `post-to-pr.sh` script fails with **HTTP 422 Unprocessable Entity ("Line could not be resolved")**, it is due to one of two causes:
1. **Unpushed Local Changes**: If the review references new lines introduced by your local bug fixes, you **must commit and push** those changes to the remote branch before posting the review, otherwise GitHub's remote does not recognize the new line numbers.
2. **Comment Outside Diff Range**: GitHub's PR reviews API only permits comments on lines that are modified or added as part of the PR diff. If a code review comment is placed on a pre-existing line that wasn't changed:
   - Identify the actual modified lines using:
     ```bash
     git --no-pager diff --no-ext-diff -U0 main -- <path-to-file>
     ```
   - Re-map the comment's `line` property in the JSON file to a line number within one of the changed blocks (ranges of lines marked with `+` in the diff), and update the comment body accordingly.

