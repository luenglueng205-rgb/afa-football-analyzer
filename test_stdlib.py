"""AFA MCP — stdio transport"""
import json, math, sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("afa-test")

@mcp.tool()
def ping() -> str:
    return "pong"

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
