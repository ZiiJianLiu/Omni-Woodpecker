#!/usr/bin/env python3
import json
import sys
from typing import List, Dict

ORDER = [
    ("Single anchor", [
        ("single_L20", r"\quad anchor $L{=}20$"),
        ("single_L26", r"\quad anchor $L{=}26$"),
        ("single_L30", r"\quad anchor $L{=}30$"),
    ]),
    ("Multi-layer", [
        ("multi_equal_alpha", r"\quad equal $\alpha$"),
        ("multi_uniform_svd", r"\quad uniform SVD"),
        ("multi_no_complement", r"\quad no complement"),
        ("multi_fixed_strengths", r"\quad fixed strengths"),
        ("multi_no_gating", r"\quad no gating"),
        ("multi_only_residual", r"\quad only residual shrink"),
        ("multi_only_anti_prior", r"\quad only anti-prior shrink"),
    ]),
    (None, [
        ("regular", "Regular"),
    ])
]

def load_rows(path: str) -> Dict[str, Dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows[obj["name"]] = obj
    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: format_ablation_table.py SUMMARY.jsonl")
        sys.exit(1)
    rows = load_rows(sys.argv[1])
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  {\small")
    print(r"  \setlength{\tabcolsep}{5pt}")
    print(r"  \begin{tabular}{lcc}")
    print(r"    \hline")
    print(r"    Variant & $\mathrm{CHAIR}_i \downarrow$ & $\mathrm{CHAIR}_s \downarrow$ \\")
    print(r"    \hline")
    for group, items in ORDER:
        if group:
            print(f"    \\multicolumn{{3}}{{l}}{{{group}}} \\\\")
        for key, label in items:
            r = rows.get(key, {})
            chair_i = r.get("CHAIR_i", 0.0)
            chair_s = r.get("CHAIR_s", 0.0)
            print(f"    {label} & {chair_i:.2f} & {chair_s:.2f}  \\\\")
        if group:
            print(r"    \hdashline")
    print(r"    \hline")
    print(r"  \end{tabular}}")
    print(r"  \caption{Component ablations (LLaVA-1.5-7B). Orthogonality and adaptivity are both important; weighted SVD improves stability and headroom.}")
    print(r"  \label{tab:ablation_components}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()


