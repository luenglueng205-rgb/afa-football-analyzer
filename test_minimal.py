"""Minimal MCP test"""
import json, math
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("afa-test")

@mcp.tool()
def ping() -> str:
    return "pong"

if __name__ == "__main__":
    mcp.run()
