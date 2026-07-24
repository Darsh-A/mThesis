#!/usr/bin/env python3
"""
Zotero Dashboard Generator
===========================
Queries the local Zotero SQLite database and generates an Obsidian-friendly
Zotero_Papers.md dashboard table in the thesis vault.

Usage:
    python zotero_dashboard.py              # generate once
    python zotero_dashboard.py --watch      # watch for changes, regenerate
    python zotero_dashboard.py --set-action <key> <action>  # set paper action
    python zotero_dashboard.py --list-actions                 # list all actions

Dependencies: pyyaml (for reading Summary frontmatter), stdlib only otherwise.
"""

import sqlite3
import os
import sys
import time
import json
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# --- Paths ---
ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"
SCRIPT_DIR = Path(__file__).resolve().parent
VAULT_ROOT = SCRIPT_DIR.parent / "thesis_vault"
OUTPUT_FILE = VAULT_ROOT / "Zotero_Papers.md"
SUMMARIES_DIR = VAULT_ROOT / "Summaries"
CONCEPTS_DIR = SUMMARIES_DIR / "Concepts"
ACTIONS_FILE = SCRIPT_DIR / "paper_actions.json"

# --- Constants ---
PAPER_ITEM_TYPES = {
    2: "artwork", 3: "attachment", 4: "audioRecording", 5: "bill",
    6: "blogPost", 7: "book", 8: "bookSection", 9: "case",
    10: "computerProgram", 11: "conferencePaper", 12: "dataset",
    14: "document", 15: "email", 16: "encyclopediaArticle", 17: "film",
    18: "forumPost", 19: "hearing", 20: "instantMessage", 21: "interview",
    22: "journalArticle", 23: "letter", 24: "magazineArticle",
    25: "manuscript", 26: "map", 27: "newspaperArticle",
    29: "patent", 30: "podcast", 31: "preprint", 32: "presentation",
    33: "radioBroadcast", 34: "report", 35: "standard",
    37: "thesis", 38: "tvBroadcast", 39: "videoRecording", 40: "webpage",
}

# Items to include as "papers" (exclude annotations, notes, attachments, etc.)
INCLUDE_TYPES = {
    "journalArticle", "preprint", "conferencePaper", "book", "bookSection",
    "thesis", "report", "webpage", "blogPost", "manuscript", "presentation",
    "computerProgram", "document",
}
# These are Zotero itemTypeIDs that are notes/attachments/annotations — skip them
SKIP_TYPE_IDS = {1, 3, 14, 28}  # annotation, attachment, document(note), note

ACTION_EMOJIS = {
    "want to read": "📖",
    "Not Reading": "🚫",
    "Completed": "✅",
    "To review again": "🔄",
    None: "⬜",
    "": "⬜",
}

# Valid action values
VALID_ACTIONS = {"want to read", "Not Reading", "Completed", "To review again"}

# Only include papers from this root collection (and its subcollections).
# Set to None to include all collections / unfiled papers.
ROOT_COLLECTION = "Thesis"

# --- Zotero DB helpers ---


def _open_zotero_db(path: Path) -> sqlite3.Connection:
    """Open zotero.sqlite read-only. If locked, copy to /tmp and open that."""
    uri = f"file:{path}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        con.execute("SELECT 1")
        return con
    except sqlite3.OperationalError:
        tmp = Path("/tmp") / "zotero_dashboard_copy.sqlite"
        shutil.copy2(path, tmp)
        con = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        return con


def _field_value(cur, item_id: int, field_id: int) -> Optional[str]:
    cur.execute(
        """SELECT idv.value FROM itemData id
           JOIN itemDataValues idv ON id.valueID = idv.valueID
           WHERE id.itemID = ? AND id.fieldID = ?""",
        (item_id, field_id),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _get_collection_hierarchy(cur) -> dict:
    """
    Build collection tree: {collectionID: {'name': ..., 'parentID': ..., 'path': ...}}.
    The 'path' is a human-readable slash-separated string from root to this collection.
    """
    cur.execute(
        "SELECT collectionID, collectionName, parentCollectionID FROM collections"
    )
    raw = cur.fetchall()
    col_map = {}
    for cid, name, parent in raw:
        col_map[cid] = {"name": name, "parentID": parent, "path": None}

    def _build_path(cid, visited=None):
        if visited is None:
            visited = set()
        if cid in visited:
            return f"[circular:{cid}]"
        visited.add(cid)
        info = col_map.get(cid)
        if not info:
            return str(cid)
        if info["path"] is not None:
            return info["path"]
        if info["parentID"] is None:
            info["path"] = info["name"]
        else:
            info["path"] = _build_path(info["parentID"], visited) + " / " + info["name"]
        return info["path"]

    for cid in col_map:
        _build_path(cid)

    return col_map


def _get_descendant_ids(cur, root_name: str, col_map: dict) -> set[int]:
    """
    Find a collection by name and return the set of its ID and all descendant IDs.
    Returns empty set if the root collection is not found.
    """
    # Find root collection ID by name (first match wins)
    root_id = None
    for cid, info in col_map.items():
        if info["name"] == root_name:
            root_id = cid
            break

    if root_id is None:
        return set()

    # BFS / DFS to collect all descendants
    descendant_ids = {root_id}
    stack = [root_id]
    while stack:
        parent = stack.pop()
        for cid, info in col_map.items():
            if info["parentID"] == parent and cid not in descendant_ids:
                descendant_ids.add(cid)
                stack.append(cid)

    return descendant_ids


def _get_item_collections(cur, item_id: int, col_map: dict) -> list[str]:
    """Return sorted list of collection paths this item belongs to."""
    cur.execute(
        "SELECT collectionID FROM collectionItems WHERE itemID = ?", (item_id,)
    )
    paths = []
    for (cid,) in cur.fetchall():
        info = col_map.get(cid)
        if info and info["path"]:
            paths.append(info["path"])
    return sorted(set(paths))


def _get_authors(cur, item_id: int) -> str:
    """Return comma-separated author names."""
    cur.execute(
        """SELECT c.firstName, c.lastName
           FROM itemCreators ic
           JOIN creators c ON ic.creatorID = c.creatorID
           WHERE ic.itemID = ? AND ic.creatorTypeID = 10  -- 'author'
           ORDER BY ic.orderIndex""",
        (item_id,),
    )
    authors = cur.fetchall()
    if not authors:
        # Try any creator type if no 'author' type found
        cur.execute(
            """SELECT c.firstName, c.lastName
               FROM itemCreators ic
               JOIN creators c ON ic.creatorID = c.creatorID
               WHERE ic.itemID = ?
               ORDER BY ic.orderIndex""",
            (item_id,),
        )
        authors = cur.fetchall()

    parts = []
    for first, last in authors:
        if first and last:
            parts.append(f"{last}, {first}")
        elif last:
            parts.append(last)
        elif first:
            parts.append(first)
    return "; ".join(parts)


def _get_last_read(cur, item_id: int) -> Optional[str]:
    """Return the most recent lastRead timestamp as a readable date (max across all attachments)."""
    cur.execute(
        "SELECT MAX(lastRead) FROM itemAttachments WHERE parentItemID = ? AND lastRead IS NOT NULL",
        (item_id,),
    )
    row = cur.fetchone()
    if row and row[0]:
        # lastRead is a Unix timestamp in milliseconds
        try:
            ts = int(row[0])
            if ts > 1_000_000_000_000:  # already milliseconds
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            return None
    return None


def _get_year(cur, item_id: int) -> str:
    """Extract year from the 'date' field (fieldID=6)."""
    date_str = _field_value(cur, item_id, 6)
    if date_str:
        # Various formats: "2018", "2018-04-10", "2018-04-10 2018-04-10"
        match = re.search(r"(\d{4})", date_str)
        if match:
            return match.group(1)
    return ""


def _get_paper_type_name(cur, item_id: int) -> str:
    cur.execute(
        """SELECT it.typeName FROM items i
           JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
           WHERE i.itemID = ?""",
        (item_id,),
    )
    row = cur.fetchone()
    return row[0] if row else "unknown"


# --- Summaries frontmatter helpers ---


def _parse_frontmatter(filepath: Path) -> dict:
    """Extract YAML frontmatter from a markdown file. Returns dict or {}."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    # Match YAML frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}

    try:
        import yaml

        return yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}


def _load_summaries_index() -> dict:
    """
    Scan Summaries/ (and Concepts/) for .md files with YAML frontmatter.
    Returns {citekey: {"action": ..., "file": relative_path}}.
    """
    index = {}
    for md_file in SUMMARIES_DIR.rglob("*.md"):
        fm = _parse_frontmatter(md_file)
        citekey = fm.get("citekey", "").strip() if fm.get("citekey") else ""
        if not citekey:
            continue

        rel = md_file.relative_to(VAULT_ROOT)
        entry = {"file": str(rel), "action": None}

        action = fm.get("action", "").strip() if fm.get("action") else ""
        if action in VALID_ACTIONS:
            entry["action"] = action

        index[citekey] = entry
    return index


# --- Action persistence ---


def _load_actions() -> dict:
    """Load paper_actions.json -> {zotero_key: action_string}."""
    if ACTIONS_FILE.exists():
        try:
            return json.loads(ACTIONS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_actions(actions: dict) -> None:
    ACTIONS_FILE.write_text(json.dumps(actions, indent=2, sort_keys=True) + "\n")


def _resolve_action(
    zotero_key: str,
    summary_index: dict,
    actions_json: dict,
) -> Optional[str]:
    """
    Resolve action priority: actions_json > summaries frontmatter > None.
    Returns one of the VALID_ACTIONS strings, or None.
    """
    # 1. Check explicit actions file
    if zotero_key in actions_json:
        val = actions_json[zotero_key]
        if val in VALID_ACTIONS:
            return val
        if val is None or val == "":
            return None

    # 2. Check Summaries frontmatter
    summary = summary_index.get(zotero_key)
    if summary and summary.get("action") in VALID_ACTIONS:
        return summary["action"]

    return None


# --- Markdown generation ---


def _generate_markdown(
    papers: list[dict],
    summary_index: dict,
    actions_json: dict,
) -> str:
    """Build the Zotero_Papers.md content."""
    lines = [
        "# 📚 Zotero Papers Dashboard",
        "",
        f"> Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Total papers: {len(papers)}",
        "",
        "| # | Title | Location | Author | Year | Last Read | Action | Summary |",
        "|---|-------|----------|--------|------|-----------|--------|---------|",
    ]

    for i, paper in enumerate(papers, 1):
        key = paper["key"]
        title = paper["title"] or "(untitled)"
        # Escape pipes in title for markdown table
        title_escaped = title.replace("|", "\\|")

        location = paper["location"] or "—"
        author = paper["author"] or "—"
        year = paper["year"] or "—"
        last_read = paper["lastRead"] or "—"

        action = _resolve_action(key, summary_index, actions_json)
        action_emoji = ACTION_EMOJIS.get(action, ACTION_EMOJIS[None])

        # Summary link
        summary_info = summary_index.get(key)
        if summary_info:
            summary_link = f"[[{summary_info['file']}]]"
        else:
            summary_link = "—"

        lines.append(
            f"| {i} | {title_escaped} | {location} | {author} | {year} | {last_read} | {action_emoji} | {summary_link} |"
        )

    # Footer
    lines.extend(
        [
            "",
            "---",
            "## Legend",
            "",
            "| Emoji | Meaning |",
            "|-------|---------|",
            f"| {ACTION_EMOJIS['want to read']} | Want to read |",
            f"| {ACTION_EMOJIS['Not Reading']} | Not reading |",
            f"| {ACTION_EMOJIS['Completed']} | Completed |",
            f"| {ACTION_EMOJIS['To review again']} | To review again |",
            f"| {ACTION_EMOJIS[None]} | No status |",
            "",
            "---",
            "*Dashboard auto-generated by `Scripts/zotero_dashboard.py`*",
        ]
    )

    return "\n".join(lines) + "\n"


# --- Main logic ---


def fetch_papers() -> list[dict]:
    """Query Zotero DB and return sorted list of paper dicts."""
    con = _open_zotero_db(ZOTERO_DB)
    cur = con.cursor()

    # Build collection map
    col_map = _get_collection_hierarchy(cur)

    # Determine allowed collection IDs if ROOT_COLLECTION is set
    allowed_collection_ids = None
    if ROOT_COLLECTION:
        allowed_collection_ids = _get_descendant_ids(cur, ROOT_COLLECTION, col_map)

    # Get all items that are papers
    cur.execute(
        """SELECT i.itemID, i.key, i.itemTypeID, i.dateAdded, i.dateModified
           FROM items i
           WHERE i.itemTypeID NOT IN ({})
           ORDER BY i.dateAdded DESC""".format(
            ",".join(str(t) for t in SKIP_TYPE_IDS)
        )
    )

    papers = []
    for item_id, key, type_id, date_added, date_modified in cur.fetchall():
        type_name = PAPER_ITEM_TYPES.get(type_id, "unknown")
        if type_name not in INCLUDE_TYPES:
            continue

        title = _field_value(cur, item_id, 1)
        if not title or not title.strip():
            continue  # Skip untitled / stub entries
        author = _get_authors(cur, item_id)
        year = _get_year(cur, item_id)
        last_read = _get_last_read(cur, item_id)
        collections = _get_item_collections(cur, item_id, col_map)

        # Filter: only include items in the allowed collection tree
        if allowed_collection_ids is not None:
            cur.execute(
                "SELECT collectionID FROM collectionItems WHERE itemID = ?",
                (item_id,),
            )
            item_col_ids = {row[0] for row in cur.fetchall()}
            if not (item_col_ids & allowed_collection_ids):
                continue

        papers.append(
            {
                "key": key,
                "title": title,
                "author": author,
                "year": year,
                "lastRead": last_read,
                "location": "; ".join(collections) if collections else None,
                "type": type_name,
                "dateAdded": date_added,
                "dateModified": date_modified,
            }
        )

    con.close()

    # Sort: filed papers first, then unfiled; within each group, alphabetically by title
    papers.sort(key=lambda p: (0 if p["location"] else 1,
                                (p["title"] or "").lower()))

    return papers


def generate() -> None:
    """Generate Zotero_Papers.md."""
    papers = fetch_papers()
    summary_index = _load_summaries_index()
    actions_json = _load_actions()
    md = _generate_markdown(papers, summary_index, actions_json)
    OUTPUT_FILE.write_text(md, encoding="utf-8")
    print(f"✅ Generated {OUTPUT_FILE} with {len(papers)} papers")


def set_action(zotero_key: str, action: Optional[str]) -> None:
    """Set the action status for a paper."""
    actions = _load_actions()

    if action and action not in VALID_ACTIONS:
        print(
            f"❌ Invalid action: '{action}'. Valid: {', '.join(sorted(VALID_ACTIONS))}"
        )
        sys.exit(1)

    if action is None:
        actions.pop(zotero_key, None)
        print(f"🗑️  Cleared action for {zotero_key}")
    else:
        actions[zotero_key] = action
        print(f"📝 Set {zotero_key} → {action} {ACTION_EMOJIS.get(action, '')}")

    _save_actions(actions)
    generate()


def list_actions() -> None:
    """List all papers with their current action status."""
    papers = fetch_papers()
    actions = _load_actions()
    summary_index = _load_summaries_index()

    print(f"{'Key':<10s} {'Action':<20s} {'Title'}")
    print("-" * 80)
    for p in papers:
        action = _resolve_action(p["key"], summary_index, actions)
        emoji = ACTION_EMOJIS.get(action, ACTION_EMOJIS[None])
        title = (p["title"] or "(untitled)")[:50]
        print(f"{p['key']:<10s} {emoji} {action or '(none)':<16s} {title}")


def watch_mode(poll_interval: int = 10) -> None:
    """Poll the Zotero DB mtime and regenerate when changed."""
    print(f"👀 Watching {ZOTERO_DB} for changes (polling every {poll_interval}s)...")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Press Ctrl+C to stop")

    last_mtime = 0
    last_summaries_mtime = 0

    try:
        while True:
            changed = False

            # Check Zotero DB
            if ZOTERO_DB.exists():
                db_mtime = ZOTERO_DB.stat().st_mtime
                if db_mtime > last_mtime:
                    last_mtime = db_mtime
                    changed = True

            # Check Summaries/ directory recursively
            if SUMMARIES_DIR.exists():
                max_mtime = 0
                for f in SUMMARIES_DIR.rglob("*.md"):
                    mtime = f.stat().st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                if max_mtime > last_summaries_mtime:
                    last_summaries_mtime = max_mtime
                    changed = True

            # Check actions file
            if ACTIONS_FILE.exists():
                amtime = ACTIONS_FILE.stat().st_mtime
                if amtime > last_mtime:  # reuse db_mtime check
                    changed = True

            if changed:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Regenerating...")
                generate()

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n👋 Stopped.")


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Zotero Dashboard Generator for Obsidian"
    )
    parser.add_argument(
        "--watch", "-w", action="store_true", help="Watch for changes and regenerate"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=10,
        help="Poll interval in seconds for --watch (default: 10)",
    )
    parser.add_argument("--set-action", nargs=2, metavar=("KEY", "ACTION"),
                        help="Set action for a paper (use 'null' to clear)")
    parser.add_argument("--list-actions", action="store_true",
                        help="List all papers and their action status")

    args = parser.parse_args()

    if args.set_action:
        key, action = args.set_action
        if action.lower() == "null":
            set_action(key, None)
        else:
            set_action(key, action)
    elif args.list_actions:
        list_actions()
    elif args.watch:
        # Run once first, then watch
        generate()
        watch_mode(args.interval)
    else:
        generate()


if __name__ == "__main__":
    main()
