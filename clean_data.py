#!/usr/bin/env python3
"""
clean_data.py

Usage:
  python clean_data.py --infile data.json --outfile data_clean.json
  python clean_data.py --infile data.json --outfile data_clean.json --lowercase
  python clean_data.py --infile data.json --outfile data_clean.json --remove-urls
  python clean_data.py --infile data.json --outfile data_clean.json --lowercase --remove-urls
"""

import argparse
import json
import re
from typing import Dict, List, Any, Tuple

# Matches URLs (http(s) + www + bare domains)
URL_RE = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s]+
      | www\.[^\s]+
      | (?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?
    )\b
    """
)

# Remove common "deleted/removed" variants
DELETED_EXACT = {"[deleted]", "[removed]"}
DELETED_RE = re.compile(r"(?i)^\s*\[(deleted|removed)\]\s*$")

# Heuristic "garbage" patterns (tune if needed)
GARBAGE_RE_LIST = [
    re.compile(r"(?i)\bautomoderator\b"),
    re.compile(r"(?i)\bi am a bot\b"),
    re.compile(r"(?i)\bthis action was performed automatically\b"),
    re.compile(r"(?i)\bplease contact the moderators\b"),
    re.compile(r"(?i)\bread the rules\b"),
    re.compile(r"(?i)\bcheck the faq\b"),
    re.compile(r"(?i)\bdiscord\.gg\b"),
    re.compile(r"(?i)\bsalary sharing\b"),
    re.compile(r"(?i)\bsurvey\b"),
    re.compile(r"(?i)\bmoderator\b.*\bteam\b"),
    re.compile(r"(?i)\bsave3rdpartyapps\b"),
    re.compile(r"(?i)\bsubreddit\b.*\brules\b"),
]

# Optional: remove very low-content posts (adjust threshold if you want)
MIN_WORDS_DEFAULT = 10


def normalize_text(text: str, lowercase: bool, remove_urls: bool) -> str:
    # Normalize whitespace a bit
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()

    if remove_urls:
        text = URL_RE.sub("", text)
        # clean up extra whitespace created by url removal
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if lowercase:
        text = text.lower()

    return text


def is_deleted_or_empty(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip()
    if t.lower() in DELETED_EXACT:
        return True
    if DELETED_RE.match(t):
        return True
    return False


def looks_like_garbage(text: str) -> bool:
    # Apply heuristic patterns
    for rx in GARBAGE_RE_LIST:
        if rx.search(text):
            return True
    return False


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def clean_dataset(
    data: Dict[str, List[Dict[str, Any]]],
    lowercase: bool,
    remove_urls: bool,
    min_words: int,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, int]]]:
    cleaned: Dict[str, List[Dict[str, str]]] = {}
    stats: Dict[str, Dict[str, int]] = {}

    for subreddit, posts in data.items():
        kept = []
        removed_empty = 0
        removed_garbage = 0
        removed_short = 0
        removed_missing_fields = 0

        for p in posts:
            pid = p.get("id")
            text = p.get("text")

            if not isinstance(pid, str) or not isinstance(text, str):
                removed_missing_fields += 1
                continue

            if is_deleted_or_empty(text):
                removed_empty += 1
                continue

            # Normalize *after* empty/deleted check (faster)
            text_norm = normalize_text(text, lowercase=lowercase, remove_urls=remove_urls)

            if is_deleted_or_empty(text_norm):
                removed_empty += 1
                continue

            if looks_like_garbage(text_norm):
                removed_garbage += 1
                continue

            if word_count(text_norm) < min_words:
                removed_short += 1
                continue

            kept.append({"id": pid, "text": text_norm})

        cleaned[subreddit] = kept
        stats[subreddit] = {
            "original": len(posts),
            "kept": len(kept),
            "removed_empty_or_deleted": removed_empty,
            "removed_garbage": removed_garbage,
            "removed_too_short": removed_short,
            "removed_missing_fields": removed_missing_fields,
        }

    return cleaned, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to raw data.json")
    ap.add_argument("--outfile", required=True, help="Path to write cleaned json")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase all text")
    ap.add_argument("--remove-urls", action="store_true", help="Remove URLs from text")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS_DEFAULT, help="Drop posts with fewer than this many words (default: 10)")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object: {subreddit: [posts...]}")

    cleaned, stats = clean_dataset(
        data=data,
        lowercase=args.lowercase,
        remove_urls=args.remove_urls,
        min_words=args.min_words,
    )

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # Print a quick summary
    print("Wrote:", args.outfile)
    print("\nPer-subreddit stats:")
    for sub, s in stats.items():
        print(f"- {sub}: {s['kept']}/{s['original']} kept | "
              f"empty {s['removed_empty_or_deleted']}, garbage {s['removed_garbage']}, "
              f"short {s['removed_too_short']}, missing {s['removed_missing_fields']}")


if __name__ == "__main__":
    main()
