# Lightweight agent-role mapping + helpers (no external deps).
# Update as needed if new agents arrive.

from typing import List, Dict
import re

# Normalize agent names: keep alphanumerics, uppercase (e.g., "KAY/O" -> "KAYO")
def normalize_agent(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = re.sub(r"[^A-Za-z0-9]", "", name).upper()
    # common alias
    if s in {"KAYO", "KAY0"}:
        s = "KAYO"
    return s

# Core mapping (covers commonly used agents in pro play)
AGENT_ROLE: Dict[str, str] = {
    # Controllers
    "BRIMSTONE": "Controller",
    "OMEN": "Controller",
    "VIPER": "Controller",
    "ASTRA": "Controller",
    "HARBOR": "Controller",
    "CLOVE": "Controller",
    # Initiators
    "SOVA": "Initiator",
    "SKYE": "Initiator",
    "BREACH": "Initiator",
    "FADE": "Initiator",
    "GEKKO": "Initiator",
    "KAYO": "Initiator",
    # Sentinels
    "KILLJOY": "Sentinel",
    "CYPHER": "Sentinel",
    "SAGE": "Sentinel",
    "CHAMBER": "Sentinel",
    "DEADLOCK": "Sentinel",
    # Duelists
    "JETT": "Duelist",
    "RAZE": "Duelist",
    "REYNA": "Duelist",
    "PHOENIX": "Duelist",
    "YORU": "Duelist",
    "NEON": "Duelist",
    "ISO": "Duelist",
}

ROLES = ["Controller", "Initiator", "Sentinel", "Duelist"]

def parse_agent_list(s: str) -> List[str]:
    """Split 'A;B;C;D;E' into normalized agent names."""
    if not isinstance(s, str):
        return []
    parts = [normalize_agent(x) for x in re.split(r"[;,|]", s) if x.strip()]
    return [p for p in parts if p]

def role_counts(agents: List[str]) -> Dict[str, int]:
    counts = {r: 0 for r in ROLES}
    for a in agents:
        role = AGENT_ROLE.get(normalize_agent(a), None)
        if role in counts:
            counts[role] += 1
    return counts

# Canonical list of agent names (normalized) for feature engineering
AGENTS = sorted(AGENT_ROLE.keys())

