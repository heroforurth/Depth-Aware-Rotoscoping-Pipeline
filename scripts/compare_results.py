"""Compare YOUR re-run (results/) with the reference numbers in the report (reference_results/).

    python scripts/compare_results.py

For every table you re-ran, each metric's average is compared with the reference.
MATCH  = within 2% (or 0.002) of the reference: GPU maths differ slightly between computers.
DIFFER = a real difference: check your setup (versions, settings) or the data.
Speed tables are only shown, never judged: they depend on your computer.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF, MINE = ROOT / "reference_results", ROOT / "results"
REL_TOL, ABS_TOL = 0.02, 0.002
NOT_METRICS = {"frames"}            # numeric columns that are counts, not results
MACHINE_DEPENDENT = {"speed.csv"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def split_columns(rows):
    cols = list(rows[0])
    metrics = [c for c in cols if c not in NOT_METRICS and all(is_number(r[c]) for r in rows if r[c] != "")]
    keys = [c for c in cols if c not in metrics and c not in NOT_METRICS]
    return keys, metrics


def mean(rows, col):
    vals = [float(r[col]) for r in rows if is_number(r.get(col))]
    return sum(vals) / len(vals) if vals else None


def compare(name):
    ref, mine = load(REF / name), load(MINE / name)
    keys, metrics = split_columns(ref)
    # compare the same rows only (a partial re-run is fine)
    index = {tuple(r[k] for k in keys): r for r in mine}
    pairs = [(r, index[tuple(r[k] for k in keys)]) for r in ref if tuple(r[k] for k in keys) in index]
    print(f"\n== {name}: {len(pairs)} of {len(ref)} reference rows found in your run")
    if not pairs:
        return 0, 0
    ok = bad = 0
    groups = sorted({r[keys[-1]] for r, _ in pairs}) if keys else [None]
    for g in groups:
        sel = [(a, b) for a, b in pairs if g is None or a[keys[-1]] == g]
        for m in metrics:
            ra, rb = mean([a for a, _ in sel], m), mean([b for _, b in sel], m)
            if ra is None or rb is None:
                continue
            diff = rb - ra
            close = abs(diff) <= max(ABS_TOL, REL_TOL * abs(ra))
            label = "shown only" if name in MACHINE_DEPENDENT else ("MATCH" if close else "DIFFER")
            if name not in MACHINE_DEPENDENT:
                ok, bad = ok + close, bad + (not close)
            prefix = f"{g:>22s} " if g is not None else ""
            print(f"  {prefix}{m:>14s}  reference {ra:10.4f}   yours {rb:10.4f}   diff {diff:+.4f}   {label}")
    return ok, bad


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not MINE.exists():
        raise SystemExit("No results/ folder yet. Run reproduce_results.bat (or an evaluation script) first.")
    names = sorted(p.name for p in REF.glob("*.csv") if (MINE / p.name).exists())
    if not names:
        raise SystemExit("results/ has no tables that match reference_results/. Run an evaluation first.")
    ok = bad = 0
    for n in names:
        a, b = compare(n)
        ok, bad = ok + a, bad + b
    missing = sorted(p.name for p in REF.glob("*.csv") if not (MINE / p.name).exists())
    print(f"\nSUMMARY: {ok} metrics MATCH, {bad} DIFFER.  Not re-run: {', '.join(missing) or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
