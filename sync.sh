#!/bin/bash
# AFA MCP 一键同步脚本
# 用法: bash sync.sh "commit message"
# 自动: git add → commit → push → 提示重装 MCP

set -e
cd /Users/jand/Projects/afa-mcp-server

MSG="${1:-chore: sync MCP changes}"

echo "📦 Staging changes..."
git add src/afa_mcp/server.py lobehub-plugin.json

if git diff --cached --quiet; then
    echo "✅ Nothing to commit — already up to date."
    exit 0
fi

echo "📝 Committing: $MSG"
git commit -m "$MSG"

echo "🚀 Pushing to GitHub..."
git push origin main

TOOLS=$(grep -c '@mcp.tool()' src/afa_mcp/server.py)
echo "✅ Done! $TOOLS tools pushed. Now reinstall the MCP plugin in LobeChat."
