#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence Splitting for ASR Text (Multilingual, Robust) + Optional CKIP Transformers Word Segmentation

New: Optional CKIP-Transformers integration to improve Chinese segmentation precision.
- If enabled, CJK long segments will be wrapped at word boundaries (CKIP WS) rather than raw char counts.
- Fallback gracefully when CKIP is unavailable.

Usage:
  python sentence_splitter.py "raw text here"
  echo "raw text" | python sentence_splitter.py
  python sentence_splitter.py -f input.txt > sentences.txt

Options:
  -f/--file: read input text from file
  -m/--max-len: preferred maximum sentence length for fallback splitting (default: 120)
  --json: output JSON array instead of lines
  --use-ckip: enable CKIP Transformers word segmentation for Chinese
  --ckip-ws-model: CKIP WS model name (default: albert_tiny). Examples: albert_tiny, bert_base

Note:
- This splitter is heuristic-based (regex + unicode punctuation) with optional CKIP assistance.
- CKIP is used only when explicitly enabled. When unavailable, the program falls back automatically.
"""

import argparse
import json
import re
import sys
from typing import List, Optional

# Optional CKIP imports (lazy)
CKIP_WS = None
CKIP_WS_MODEL = "bert-base"

def try_init_ckip(ws_model: str = "albert_tiny"):
    global CKIP_WS
    if CKIP_WS is not None:
        return CKIP_WS
    try:
        from ckip_transformers.nlp import CkipWordSegmenter
        CKIP_WS = CkipWordSegmenter(model=ws_model)
    except Exception as e:
        print(f"[CKIP] Unable to init CKIP WordSegmenter: {e}. Fallback to default heuristics.", file=sys.stderr)
        CKIP_WS = None
    return CKIP_WS


def ckip_tokenize(text: str) -> List[str]:
    if CKIP_WS is None:
        return []
    try:
        # CKIP WS returns list of lists for batch; we pass single item
        tokens = CKIP_WS([text])[0]
        return tokens
    except Exception as e:
        print(f"[CKIP] Tokenization error: {e}.", file=sys.stderr)
        return []

# Unicode-aware sentence terminators across languages
SENT_END_PUNCT = set(
    list(".!?;:")
    + list("。！？；：")
    + [
        "؟",  # Arabic question mark
        "٫",  # Arabic decimal separator (rare)
        "؛",  # Arabic semicolon
        "۔",  # Urdu period
        "।",  # Devanagari danda
        "॥",  # Devanagari double danda
    ]
)

# Trailing quotes/brackets that may appear right after end punctuations
TRAILING_CLOSE = set(list('"\'”’）】》〉』」]') + [")", "]", "}", "›", "»"])

# Common abbreviations to protect (period should not end sentence)
ABBREVIATIONS = set([
    # English
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "e.g.", "i.e.", "etc.",
    "u.s.", "u.k.", "ph.d.", "m.d.", "inc.", "ltd.", "co.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    # Spanish
    "sr.", "sra.", "d.", "ud.", "uds.", "p.ej.",
    # German
    "z.b.", "usw.", "bzw.", "ca.",
    # French
    "ex.", "env.", "prof.",
    # Portuguese
    "sr.", "sra.", "p.ex.",
    # Italian
    "c.a.", "p.es.",
])

# Regex for initials like "A." or "J.R.R." that shouldn't end sentences
INITIALS_PATTERN = re.compile(r"(?i)^(?:[A-Z]\.){2,4}$")

# Safe spaces normalization
WS_RE = re.compile(r"[^\S\n]+")

# CJK heuristics
def contains_cjk(s: str) -> bool:
    return re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", s) is not None

# Detect if token is an abbreviation (case-insensitive)

def is_abbrev(token: str) -> bool:
    t = token.strip().lower()
    if t in ABBREVIATIONS:
        return True
    if INITIALS_PATTERN.match(t):
        return True
    return False


def normalize_text(text: str) -> str:
    # Normalize whitespace and standardize quotes
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = WS_RE.sub(" ", text).strip()
    return text


# Core splitting logic

def split_by_punctuation(text: str) -> List[str]:
    # We iterate chars and build sentences respecting abbreviations and trailing close symbols
    sentences = []
    buf = []

    def flush():
        s = ''.join(buf).strip()
        if s:
            sentences.append(s)
        buf.clear()

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)

        if ch in SENT_END_PUNCT:
            # Lookahead: consume trailing quotes/brackets
            j = i + 1
            while j < n and text[j] in TRAILING_CLOSE:
                buf.append(text[j])
                j += 1
            # Special handling: consecutive initials like U.S. or J.R.R.
            if ch == '.' and j + 1 < n:
                # If previous token is single-letter initial and next forms "X." sequence, keep aggregating
                prev_text = ''.join(buf)
                if re.search(r"[A-Za-z]\.$", prev_text):
                    if text[j].isalpha() and (j + 1 < n and text[j+1] == '.'):
                        i = j
                        continue
            # Peek previous token to avoid splitting on abbreviations
            prev = ''.join(buf).rstrip()
            m = re.search(r"([\w\.]+)[^\w\.]*$", prev)
            token = m.group(1) if m else ""
            if token:
                if token.endswith('.') and len(token) > 1:
                    pass
                token_l = token.lower()
                if is_abbrev(token_l):
                    i = j
                    continue
            flush()
            i = j
            continue

        elif ch == '\n':
            flush()

        i += 1

    flush()
    return sentences


def smart_resplit_long(segment: str, max_len: int, use_ckip: bool = False) -> List[str]:
    s = segment.strip()
    if not s or len(s) <= max_len:
        return [s] if s else []

    # If CJK and CKIP enabled and available, split by CKIP tokens with max_len constraint
    if use_ckip and contains_cjk(s) and CKIP_WS is not None:
        tokens = ckip_tokenize(s)
        if tokens:
            acc = []
            buf = ""
            for tok in tokens:
                candidate = (buf + tok).strip()
                if len(candidate) <= max_len:
                    buf = candidate + ""
                else:
                    if buf:
                        acc.append(buf)
                        buf = tok
                    else:
                        # Single token longer than max_len; fall back to hard wrap
                        wrapped = hard_wrap(tok, max_len)
                        acc.extend(wrapped[:-1])
                        buf = wrapped[-1] if wrapped else ""
            if buf:
                acc.append(buf)
            # Final pass: ensure segments not exceeding max_len
            final = []
            for seg in acc:
                if len(seg) > max_len:
                    final.extend(hard_wrap(seg, max_len))
                else:
                    final.append(seg)
            return [x for x in final if x]

    # Prefer split by commas and conjunctions (multilingual)
    zh_marks = r"，|、|；|：|然後|而且|但是|所以|因為|如果|以及"
    comma_split = re.split(rf"({zh_marks}|,|;|—|–|-)", s)

    acc = []
    buf = ""
    for part in comma_split:
        if not part:
            continue
        candidate = (buf + part).strip()
        if not candidate:
            continue
        if len(candidate) <= max_len:
            buf = candidate + " "
        else:
            if buf.strip():
                acc.append(buf.strip())
                buf = part.strip() + " "
            else:
                hard = hard_wrap(candidate, max_len)
                acc.extend(hard[:-1])
                buf = (hard[-1] + " ") if hard else ""
    if buf.strip():
        acc.append(buf.strip())

    final = []
    for seg in acc:
        if len(seg) > max_len:
            final.extend(hard_wrap(seg, max_len))
        else:
            final.append(seg)
    return [x for x in final if x]


def hard_wrap(text: str, max_len: int) -> List[str]:
    # Wrap by nearest space; if CJK (no spaces), wrap by char length
    parts = []
    t = text.strip()
    while len(t) > max_len:
        cut = t.rfind(' ', 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(t[:cut].strip())
        t = t[cut:].strip()
    if t:
        parts.append(t)
    return parts


def split_text(text: str, max_len: int = 120, use_ckip: bool = False, ckip_ws_model: Optional[str] = None) -> List[str]:
    """Split raw ASR text into sentences.

    Steps:
    - Normalize whitespace
    - Split by multilingual punctuation, protecting abbreviations
    - Fallback: for segments still too long, resplit by commas/conjunctions/hard wrap
    - Optional: use CKIP word segmentation to improve CJK fallback splitting
    """
    text = normalize_text(text)
    if not text:
        return []

    # Initialize CKIP if requested
    if use_ckip:
        model = ckip_ws_model or CKIP_WS_MODEL
        try_init_ckip(ws_model=model)

    sentences = split_by_punctuation(text)

    # Merge extremely short fragments (e.g., due to stray punctuation)
    merged = []
    for seg in sentences:
        seg = seg.strip()
        if not seg:
            continue
        # Drop stray period-only segments (e.g., after absorbing closing bracket following ?)
        if seg in {'.', '。'}:
            continue
        if merged and len(seg) < 4:
            merged[-1] = (merged[-1] + ' ' + seg).strip()
        else:
            merged.append(seg)

    # Fallback splitting for long segments
    final = []
    for seg in merged:
        if len(seg) > max_len:
            final.extend(smart_resplit_long(seg, max_len, use_ckip=use_ckip))
        else:
            final.append(seg)
    return final


def main():
    parser = argparse.ArgumentParser(description="Sentence splitting for ASR text (multilingual, robust) with optional CKIP WS")
    parser.add_argument("text", nargs="?", help="Raw text to split (omit to read stdin)")
    parser.add_argument("-f", "--file", help="Read text from file path")
    parser.add_argument("-m", "--max-len", type=int, default=120, help="Preferred maximum sentence length for fallback splitting")
    parser.add_argument("--json", action="store_true", help="Output JSON array instead of lines")
    parser.add_argument("--use-ckip", action="store_true", help="Enable CKIP Transformers word segmentation for Chinese")
    parser.add_argument("--ckip-ws-model", default="bert-base", help="CKIP WS model name (default: bert-base)")
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text is not None:
        raw = args.text
    else:
        raw = sys.stdin.read()

    if args.use_ckip:
        try_init_ckip(ws_model=args.ckip_ws_model)

    sentences = split_text(raw, max_len=args.max_len, use_ckip=args.use_ckip, ckip_ws_model=args.ckip_ws_model)
    if args.json:
        print(json.dumps(sentences, ensure_ascii=False))
    else:
        for s in sentences:
            print(s)


if __name__ == "__main__":
    main()
