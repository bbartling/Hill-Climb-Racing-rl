#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter

PRIMITIVES = (str, int, float, bool, type(None))


def type_name(v):
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        return "number"
    if v is None:
        return "null"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def truncate_str(s, max_len):
    if max_len is None or len(s) <= max_len:
        return s
    return s[:max_len] + f"… ({len(s)-max_len} more chars)"


def compile_patterns(items):
    pats = []
    for it in items:
        if len(it) >= 2 and it.startswith("/") and it.endswith("/"):
            pats.append(re.compile(it[1:-1]))
        else:
            pats.append(re.compile(re.escape(it)))
    return pats


def match_any(key, patterns):
    for p in patterns:
        if p.fullmatch(key) or p.search(key):
            return True
    return False


def summarize_array(arr):
    types = [type_name(x) for x in arr]
    counts = Counter(types)
    obj_example = next((x for x in arr if isinstance(x, dict)), None)
    return counts, len(arr), obj_example


def print_schema(
    data,
    indent=0,
    key_name=None,
    max_string=100,
    redact_patterns=(),
    exclude_patterns=(),
    visited_ids=None,
):
    if visited_ids is None:
        visited_ids = set()
    pad = "  " * indent
    kprefix = f"{key_name}: " if key_name is not None else ""
    oid = id(data)
    if oid in visited_ids:
        print(f"{pad}{kprefix}<cycle>")
        return
    visited_ids.add(oid)

    if isinstance(data, dict):
        if key_name:
            print(f"{pad}{key_name} (object)")
        for k, v in data.items():
            if match_any(k, exclude_patterns):
                continue
            if match_any(k, redact_patterns):
                print(f"{pad}  {k} (redacted)")
                continue
            print_schema(
                v,
                indent + 1,
                k,
                max_string,
                redact_patterns,
                exclude_patterns,
                visited_ids,
            )
    elif isinstance(data, list):
        counts, length, obj_example = summarize_array(data)
        types_desc = ", ".join(f"{t}×{c}" for t, c in counts.items())
        print(f"{pad}{kprefix}array (length={length}; elements: {types_desc or '—'})")
        if obj_example:
            print(f"{pad}  └─ element schema (object):")
            print_schema(
                obj_example,
                indent + 2,
                None,
                max_string,
                redact_patterns,
                exclude_patterns,
                visited_ids,
            )
    elif isinstance(data, PRIMITIVES):
        t = type_name(data)
        val = (
            data
            if isinstance(data, (int, float, bool))
            else truncate_str(str(data), max_string)
        )
        print(f"{pad}{kprefix}{t} example={val}")
    else:
        print(f"{pad}{kprefix}{type_name(data)}")


def main():
    parser = argparse.ArgumentParser(description="Print JSON key/type schema.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="hcr_config.json",
        help="Path to JSON file (default: ./hcr_config.json)",
    )
    parser.add_argument(
        "--redact",
        nargs="*",
        default=["ref_img_b64", "/.*_b64$/"],
        help="Keys or regex to redact (print key but hide value).",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="Keys or regex to exclude entirely."
    )
    parser.add_argument(
        "--max-string",
        type=int,
        default=100,
        help="Max string length for preview values.",
    )
    args = parser.parse_args()

    redact_pats = compile_patterns(args.redact)
    exclude_pats = compile_patterns(args.exclude)

    if not os.path.exists(args.json_path):
        print(f"⚠️  JSON file not found: {args.json_path}")
        return

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Schema for {os.path.abspath(args.json_path)}:\n")
    print_schema(data, 0, None, args.max_string, redact_pats, exclude_pats)


if __name__ == "__main__":
    main()
