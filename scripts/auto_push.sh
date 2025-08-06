#!/usr/bin/env bash
set -euo pipefail

git add -A
msg=${1:-"Auto-push: integrate MegaSystem runner & logging"}
branch=$(git rev-parse --abbrev-ref HEAD)
if git diff --cached --quiet; then
  echo "Nothing to commit." >&2
else
  git commit -m "$msg"
fi

git push -u origin "$branch"