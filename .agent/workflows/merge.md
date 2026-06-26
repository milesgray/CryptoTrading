---
auto_execution_mode: 3
---
# PR Merge Workflow

## Quick Merge (CLI merge, keep local branch)
```bash
git checkout main
git pull origin main
git merge --ff-only <branch-name>
git push origin main
```

## Full Merge with Cleanup
```bash
# 1. Ensure branch is clean
git status

# 2. Switch to main and sync
git checkout main
git pull origin main

# 3. Merge the feature branch
git merge --ff-only <branch-name>

# 4. Push merged changes
git push origin main

# 5. Delete local branch
git branch -d <branch-name>

# 6. Delete remote branch
git push origin --delete <branch-name>
```

## Merge via GitHub/GitLab UI + Local Sync
If you merge through the web UI, sync locally:
```bash
git checkout main
git pull origin main
git branch -d <branch-name>  # clean up local
git remote prune origin      # clean up stale refs
```

## Variables
- `<branch-name>`: Your feature/fix branch (e.g., `feature/cache-optimization`)
- `--ff-only`: Fast-forward only (fails if non-linear history—safer)
- Use `--no-ff` if you want to preserve merge commits

## agent Integration
Add to your existing workflow chain. Full sequence:
1. **commit** → staged changes
2. **lint** → code quality
3. **test** → validation
4. **pr-create** → open PR
5. **pr-merge** → (this workflow)

## Optional: After Merge
- Update other branches if needed: `git checkout develop && git pull origin develop && git merge main`
- Verify: `git log --oneline -n 5` on main
- Check: `git branch -a` to see cleanup results
