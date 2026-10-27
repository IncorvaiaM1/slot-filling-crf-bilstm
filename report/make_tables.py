"""Build the LaTeX tables and number macros for the A1a report.

Reads only files already in the repo (no training, no torch needed):
  figures/data_efficiency.json, figures/data_efficiency_dev.json,
  figures/data_efficiency_plateau_lr.json, figures/crf_feature_ablation.json,
  figures/error_examples.json, crf_predictions.jsonl, bilstm_predictions.jsonl,
  and the ATIS data via data_loader.py.

Writes report/tables/*.tex and report/numbers.tex. Rerun after retraining:
    python report/make_tables.py
"""

from __future__ import annotations
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_loader  # noqa: E402
from seqeval.metrics import classification_report  # noqa: E402
from seqeval.scheme import IOB2  # noqa: E402

OUT = ROOT / "report"
TABLES = OUT / "tables"
FIG = ROOT / "figures"
TAGS = ["0.05", "0.10", "0.25", "1.00"]
WORD = {"0.05": "Five", "0.10": "Ten", "0.25": "TwentyFive", "1.00": "Full"}
N_SENT = {"0.05": 224, "0.10": 448, "0.25": 1120, "1.00": 4478}


def tex(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def load(name: str) -> dict:
    return json.loads((FIG / name).read_text(encoding="utf-8"))


def spans(tags: list[str]) -> list[tuple[str, int, int]]:
    out, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            typ, j = tags[i][2:], i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            out.append((typ, i, j))
            i = j
        else:
            i += 1
    return out


def f3(x: float) -> str:
    return f"{x:.3f}"


macros: list[tuple[str, str]] = []


def macro(name: str, value: str) -> None:
    macros.append((name, value))


# ---------------------------------------------------------------- Part A ----
def table_ablation() -> None:
    abl = load("crf_feature_ablation.json")
    rows = [
        ("identity", "word identity only (current token)"),
        ("+affix", "\\quad + prefixes/suffixes (1--3 chars)"),
        ("+ortho", "\\quad + casing, digit, shape, length"),
        ("+ctx1", "\\quad + all of the above for tokens $\\pm1$, bigrams"),
        ("+ctx2", "\\quad + lower-cased words at $\\pm2$"),
        ("full", "\\quad + gazetteers / time regex (\\textbf{submitted})"),
        ("full-gaz", "full $-$ gazetteers"),
        ("full-affix", "full $-$ affixes"),
        ("full-ortho", "full $-$ orthographic"),
        ("full-ctx", "full $-$ all context"),
    ]
    abl5_path = FIG / "crf_feature_ablation_0.05.json"
    abl5 = load(abl5_path.name) if abl5_path.is_file() else {}
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Trained on 100\% (4,478 sent.)} & 5\% (224) \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-6}",
        r"Feature set & \#feats & Dev F1 & Test F1 & Role errs & Dev F1 \\",
        r"\midrule",
    ]
    for key, label in rows:
        r = abl[key]
        dev = f3(r["dev_span_f1"])
        if key == "full":
            dev = r"\textbf{" + dev + "}"
        role = str(r["dev_role_only"]) if "dev_role_only" in r else "--"
        d5 = f3(abl5[key]["dev_span_f1"]) if key in abl5 else "--"
        if key == "full" and d5 != "--":
            d5 = r"\textbf{" + d5 + "}"
        lines.append(
            f"{label} & {r['n_state_features']:,} & {dev} & {f3(r['test_span_f1'])} & {role} & {d5} \\\\"
        )
        if key == "full":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{3pt}\parbox{0.86\linewidth}{\scriptsize \#feats = state features with"
        r" nonzero weight after training; the L1 penalty ($c_1{=}0.1$) prunes features made redundant"
        r" by others, so the count can drop when a group is added. Role errs = dev spans with the"
        r" right boundaries and the right slot but the wrong role (e.g.\ \texttt{toloc} vs"
        r" \texttt{fromloc}), out of 1,709 gold spans; -- = not measured.}",
    ]
    (TABLES / "crf_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macro("crfRoleIdentity", str(abl["identity"].get("dev_role_only", "")))
    macro("crfWrongTypeIdentity", str(abl["identity"].get("dev_wrong_type", "")))
    macro("crfRoleCtxOne", str(abl["+ctx1"].get("dev_role_only", "")))
    macro("crfRoleFull", str(abl["full"].get("dev_role_only", "")))
    if abl5:
        macro("crfDevFiveFull", f3(abl5["full"]["dev_span_f1"]))
        macro("crfDevFiveNoGaz", f3(abl5["full-gaz"]["dev_span_f1"]))
        macro("crfDevFiveNoAffix", f3(abl5["full-affix"]["dev_span_f1"]))
        macro("crfDevFiveNoOrtho", f3(abl5["full-ortho"]["dev_span_f1"]))
        macro("crfDevFiveIdentity", f3(abl5["identity"]["dev_span_f1"]))
        macro("crfDevFiveNoCtx", f3(abl5["full-ctx"]["dev_span_f1"]))
    macro("crfDevIdentity", f3(abl["identity"]["dev_span_f1"]))
    macro("crfDevCtxOne", f3(abl["+ctx1"]["dev_span_f1"]))
    macro("crfDevNoGaz", f3(abl["full-gaz"]["dev_span_f1"]))
    macro("crfDevNoCtx", f3(abl["full-ctx"]["dev_span_f1"]))
    macro("crfDevFull", f3(abl["full"]["dev_span_f1"]))


# ---------------------------------------------------------------- Part C ----
def seed_stats(payload: dict) -> dict:
    """{variant: {tag: (mean, min, max, n)}} including the CRF (n=1)."""
    out: dict = {"crf": {}, "bilstm": {}, "bilstm-softmax": {}}
    for tag in TAGS:
        out["crf"][tag] = (payload["metrics"]["crf"][tag]["span_f1"],) * 3 + (1,)
    table: dict = {"bilstm": {}, "bilstm-softmax": {}}
    for tag, m in payload["metrics"]["bilstm"].items():
        table["bilstm"].setdefault(tag, {})["42"] = m["span_f1"]
    for variant, by_tag in payload.get("extras", {}).items():
        for tag, by_seed in by_tag.items():
            table[variant].setdefault(tag, {}).update(by_seed)
    for variant, by_tag in table.items():
        for tag, by_seed in by_tag.items():
            v = list(by_seed.values())
            out[variant][tag] = (statistics.mean(v), min(v), max(v), len(v))
    return out


def cell(stat: tuple, best: bool) -> str:
    mean, lo, hi, n = stat
    s = f3(mean)
    if best:
        s = r"\textbf{" + s + "}"
    if n > 1:
        s += r"\,{\scriptsize$\pm$" + f"{(hi - lo) / 2:.3f}" + "}"
    return s


def table_curve() -> None:
    test = seed_stats(load("data_efficiency.json"))
    dev = seed_stats(load("data_efficiency_dev.json"))
    old = load("data_efficiency_plateau_lr.json")["span_f1"]
    lines = [
        r"\begin{tabular}{@{}rr ccc c@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Test span F1} & Dev span F1 \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-6}",
        r"Frac. & Sent. & CRF & BiLSTM-CRF & BiLSTM+softmax & CRF / BiLSTM-CRF \\",
        r"\midrule",
    ]
    for tag in TAGS:
        best = max(test[v][tag][0] for v in test)
        cells = [cell(test[v][tag], test[v][tag][0] == best) for v in ("crf", "bilstm", "bilstm-softmax")]
        d = f"{f3(dev['crf'][tag][0])} / {f3(dev['bilstm'][tag][0])}"
        lines.append(f"{int(float(tag) * 100)}\\% & {N_SENT[tag]:,} & " + " & ".join(cells) + f" & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "data_efficiency.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for tag in TAGS:
        w = WORD[tag]
        macro(f"crfTest{w}", f3(test["crf"][tag][0]))
        macro(f"lstmTest{w}", f3(test["bilstm"][tag][0]))
        macro(f"smTest{w}", f3(test["bilstm-softmax"][tag][0]))
        macro(f"crfDev{w}Frac", f3(dev["crf"][tag][0]))
        macro(f"lstmDev{w}", f3(dev["bilstm"][tag][0]))
        macro(f"smDev{w}", f3(dev["bilstm-softmax"][tag][0]))
        macro(f"oldLstmTest{w}", f3(old["test"][tag]))
        macro(f"oldLstmDev{w}", f3(old["dev"][tag]))
        macro(f"nSent{w}", f"{N_SENT[tag]:,}")


# ---------------------------------------------------------------- Part D ----
def table_errors() -> None:
    splits = data_loader.load_atis()
    train_vocab = {t for ex in splits["train"] for t in ex.tokens}
    gold = [ex.slots for ex in splits["test"]]
    toks = [ex.tokens for ex in splits["test"]]
    preds = {
        m: [json.loads(l)["slots"] for l in (ROOT / f"{m}_predictions.jsonl").open(encoding="utf-8")]
        for m in ("crf", "bilstm")
    }

    # Span-level error types per model.
    kinds = ["wrong type", "wrong boundary", "type + boundary", "missed", "spurious"]
    counts = {m: Counter() for m in preds}
    oov = {m: Counter() for m in preds}
    for i, g in enumerate(gold):
        gs = spans(g)
        for m, P in preds.items():
            ps = spans(P[i])
            pset = set(ps)
            for typ, a, b in gs:
                has_oov = any(toks[i][k] not in train_vocab for k in range(a, b))
                key = "oov" if has_oov else "inv"
                oov[m][key + "_total"] += 1
                if (typ, a, b) in pset:
                    oov[m][key + "_hit"] += 1
                    continue
                overl = [p for p in ps if p[1] < b and a < p[2]]
                if any(p[1:] == (a, b) for p in overl):
                    counts[m]["wrong type"] += 1
                elif any(p[0] == typ for p in overl):
                    counts[m]["wrong boundary"] += 1
                elif overl:
                    counts[m]["type + boundary"] += 1
                else:
                    counts[m]["missed"] += 1
            for p in ps:
                if not any(q[1] < p[2] and p[1] < q[2] for q in gs):
                    counts[m]["spurious"] += 1

    ex = load("error_examples.json")
    n_gold = sum(len(spans(g)) for g in gold)
    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Span errors on test (" + f"{n_gold:,}" + r" gold spans) & CRF & BiLSTM-CRF \\",
        r"\midrule",
    ]
    for k in kinds:
        lines.append(f"{k} & {counts['crf'][k]} & {counts['bilstm'][k]} \\\\")
    lines.append(r"\midrule")
    for key, label in (("oov", "recall, spans w/ OOV token"), ("inv", "recall, in-vocab spans")):
        c, l = oov["crf"], oov["bilstm"]
        lines.append(
            f"{label} (n={c[key + '_total']}) & {c[key + '_hit'] / c[key + '_total']:.3f} "
            f"& {l[key + '_hit'] / l[key + '_total']:.3f} \\\\"
        )
    lines.append(r"\midrule")
    ec = ex["exact_match_counts"]
    lines.append(r"sentences wrong (this model only) & " + f"{ec.get('crf_wrong_only', 0)} & {ec.get('bilstm_wrong_only', 0)} \\\\")
    lines.append(r"sentences wrong for both & \multicolumn{2}{c}{" + f"{ec.get('both_wrong', 0)}" + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "error_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macro("nCrfWrongOnly", str(ec.get("crf_wrong_only", 0)))
    macro("nLstmWrongOnly", str(ec.get("bilstm_wrong_only", 0)))
    macro("nBothWrong", str(ec.get("both_wrong", 0)))
    macro("nBothRight", str(ec.get("both_correct", 0)))
    macro("oovSpans", str(oov["crf"]["oov_total"]))
    macro("crfOovRecall", f"{oov['crf']['oov_hit'] / oov['crf']['oov_total']:.3f}")
    macro("lstmOovRecall", f"{oov['bilstm']['oov_hit'] / oov['bilstm']['oov_total']:.3f}")

    # Per-slot-type F1 where the models differ most (support >= 5).
    rep = {
        m: classification_report(gold, P, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
        for m, P in preds.items()
    }
    rows = []
    for typ, r in rep["crf"].items():
        if typ.endswith("avg") or r["support"] < 5:
            continue
        diff = rep["bilstm"][typ]["f1-score"] - r["f1-score"]
        rows.append((abs(diff), diff, typ, r["support"], r["f1-score"], rep["bilstm"][typ]["f1-score"]))
    rows.sort(reverse=True)
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Slot type & $n$ & CRF & BiLSTM-CRF & $\Delta$ \\",
        r"\midrule",
    ]
    for _, diff, typ, sup, fc, fl in sorted(rows[:10], key=lambda r: r[1]):
        lines.append(f"\\texttt{{{tex(typ)}}} & {sup} & {f3(fc)} & {f3(fl)} & {diff:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "per_slot.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The ten examples: show only the spans where gold / CRF / BiLSTM disagree.
    bucket_name = {"crf_wrong_only": "CRF", "bilstm_wrong_only": "LSTM", "both_wrong": "both"}
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}l l X@{}}",
        r"\toprule",
        r"\# & Wrong & Utterance, then the disputed spans (G = gold, C = CRF, L = BiLSTM-CRF) \\",
        r"\midrule",
    ]
    for k, rec in enumerate(ex["examples"], 1):
        t, g, c, l = rec["tokens"], rec["gold"], rec["crf"], rec["bilstm"]
        sg, sc, sl = set(spans(g)), set(spans(c)), set(spans(l))
        disputed = (sg ^ sc) | (sg ^ sl)
        hot = {i for _, a, b in disputed for i in range(a, b)}
        utt = " ".join(
            (r"\textbf{" + tex(w) + "}" if i in hot else tex(w))
            + (r"$^{\dagger}$" if w not in train_vocab else "")
            for i, w in enumerate(t)
        )

        def show(S: set) -> str:
            items = sorted(s for s in S if s in disputed or any(s[1] < d[2] and d[1] < s[2] for d in disputed))
            if not items:
                return "--"
            return "; ".join(f"\\texttt{{{tex(typ)}}}={tex(' '.join(t[a:b]))}" for typ, a, b in items)

        detail = f"G: {show(sg)} \\quad C: {show(sc)} \\quad L: {show(sl)}"
        lines.append(f"D{k} ({rec['index']}) & {bucket_name[rec['bucket']]} & {utt} \\newline {{\\footnotesize {detail}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (TABLES / "error_examples.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import shutil

    TABLES.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for name in ("data_efficiency.pdf", "data_efficiency_dev.pdf"):
        shutil.copy2(FIG / name, OUT / "figures" / name)
    table_ablation()
    table_curve()
    table_errors()
    body = ["% Auto-generated by report/make_tables.py -- do not edit by hand."]
    body += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in macros]
    (OUT / "numbers.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"Wrote {len(list(TABLES.glob('*.tex')))} tables and {len(macros)} macros to {OUT}")


if __name__ == "__main__":
    main()
