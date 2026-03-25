"""
MCP Client Manager — Lazy connections to backend verification servers.

Manages StdioTransport connections to:
  - hipai-montague  (T1: deontological constraint checking)
  - mcp-logic       (T2: formal verification via Prover9/Mace4)
  - sheaf-consistency-enforcer (T3/L5: cross-layer coherence)

Connections are lazy (connect on first use) and resilient
(graceful fallback if a backend is unavailable).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

logger = logging.getLogger("conscience-servitor.mcp_clients")

BACKENDS_PATH = Path(__file__).parent / "backends.json"


class BackendConnection:
    """Single lazy MCP backend connection."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self._client: Client | None = None
        self._connected = False

    def _make_transport(self) -> StdioTransport:
        """Build StdioTransport from config."""
        return StdioTransport(
            command=self.config["command"],
            args=self.config.get("args", []),
            env=self.config.get("env"),
        )

    async def connect(self) -> bool:
        """Lazily connect to the backend. Returns True on success."""
        if self._connected and self._client is not None:
            return True
        try:
            transport = self._make_transport()
            self._client = Client(transport)
            await self._client.__aenter__()
            self._connected = True
            logger.info("Connected to backend: %s", self.name)
            return True
        except Exception as e:
            logger.warning("Failed to connect to %s: %s", self.name, e)
            self._client = None
            self._connected = False
            return False

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call a tool on this backend. Returns None on failure."""
        if not await self.connect():
            return None
        try:
            assert self._client is not None
            result = await self._client.call_tool(tool_name, arguments)
            # FastMCP call_tool returns CallToolResult; extract data
            if hasattr(result, "data"):
                data = result.data
                # If data is a string that looks like JSON, parse it
                if isinstance(data, str):
                    try:
                        return json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        return {"raw": data}
                if isinstance(data, dict):
                    return data
                return {"raw": str(data)}
            # Fallback: extract from content blocks
            if hasattr(result, "content") and result.content:
                text_parts = [
                    c.text for c in result.content
                    if hasattr(c, "text")
                ]
                combined = "\n".join(text_parts)
                try:
                    return json.loads(combined)
                except (json.JSONDecodeError, ValueError):
                    return {"raw": combined}
            return {"raw": str(result)}
        except Exception as e:
            logger.warning("Tool call %s.%s failed: %s", self.name, tool_name, e)
            return None

    async def disconnect(self):
        """Cleanly shut down the connection."""
        if self._client is not None and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._connected = False
            self._client = None


class MCPClientManager:
    """Manages connections to all backend MCP servers.

    Usage:
        manager = MCPClientManager()
        manager.load_config()

        # In eval_engine:
        result = await manager.hipai("check_action", {
            "subject_id": "Agent",
            "relation": "HARMS",
            "object_id": "User"
        })
    """

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or BACKENDS_PATH
        self.backends: dict[str, BackendConnection] = {}

    def load_config(self):
        """Load backend configurations from JSON."""
        if not self.config_path.exists():
            logger.warning("No backends.json found at %s", self.config_path)
            return

        with open(self.config_path, encoding="utf-8") as f:
            configs = json.load(f)

        for name, config in configs.items():
            self.backends[name] = BackendConnection(name, config)
            logger.info("Registered backend: %s", name)

    async def hipai(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call a hipai-montague tool."""
        backend = self.backends.get("hipai-montague")
        if backend is None:
            logger.warning("hipai-montague backend not configured")
            return None
        return await backend.call_tool(tool_name, arguments)

    async def logic(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call an mcp-logic tool."""
        backend = self.backends.get("mcp-logic")
        if backend is None:
            logger.warning("mcp-logic backend not configured")
            return None
        return await backend.call_tool(tool_name, arguments)

    async def sheaf(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call a sheaf-consistency-enforcer tool."""
        backend = self.backends.get("sheaf-consistency-enforcer")
        if backend is None:
            logger.warning("sheaf-consistency-enforcer backend not configured")
            return None
        return await backend.call_tool(tool_name, arguments)

    async def shutdown(self):
        """Disconnect all backends."""
        for backend in self.backends.values():
            await backend.disconnect()
        logger.info("All backend connections closed")
