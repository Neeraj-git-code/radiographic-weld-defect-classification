"""
check_leakage.py  --  RUN THIS FIRST. It takes about 10 seconds and needs no
GPU and no TensorFlow.

It answers the one question your 100.00% baseline accuracy raises: are the
training and testing partitions of your RIAWELC copy actually independent?

RIAWELC images are PATCHES cropped from larger parent radiographs. The
filename encodes both:

    RRT-101R_Img1_A80_S4_[16][1].png
    |_________________|  |__||_|
      parent radiograph  row  col

If patches from the SAME parent radiograph appear in both training and
testing, the test set is not measuring generalisation to unseen welds. It is
measuring whether the model can recognise a weld it has already seen from a
slightly different crop. That inflates accuracy toward 100%.

This script reports, per class and overall:
  1. distinct parent radiographs in each split
  2. parents shared between train and test  <-- THE HEADLINE NUMBER
  3. how many test patches have a same-parent patch in training
  4. exact duplicate files by content hash (MD5) across splits
  5. spatially adjacent crops across splits (|row diff| <= 1 and |col diff| <= 1)

Usage:
    python check_leakage.py
    python check_leakage.py --root "path/to/RIAWELC_dataset/DB - Copy"
"""

import os, re, json, hashlib, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="RIAWELC_dataset/DB - Copy")
ap.add_argument("--hash", action="store_true",
                help="also compute MD5 duplicates (slower, a few minutes)")
ap.add_argument("--out", default="leakage_audit.json")
args = ap.parse_args()

SPLITS = ["training", "validation", "testing"]
PATCH_RE = re.compile(r"^(?P<parent>.+?)_\[(?P<row>\d+)\]\[(?P<col>\d+)\]"
                      r"(?P<suffix>.*)\.png$", re.IGNORECASE)

# ---------------------------------------------------------------- scan
index = {s: [] for s in SPLITS}          # (class, parent, row, col, path)
unparsed = {s: [] for s in SPLITS}

for split in SPLITS:
    sdir = os.path.join(args.root, split)
    if not os.path.isdir(sdir):
        raise SystemExit(f"Not found: {sdir}\nPass the right --root.")
    for cls in sorted(os.listdir(sdir)):
        cdir = os.path.join(sdir, cls)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if not fn.lower().endswith(".png"):
                continue
            m = PATCH_RE.match(fn)
            path = os.path.join(cdir, fn)
            if m:
                index[split].append((cls, m.group("parent"),
                                     int(m.group("row")), int(m.group("col")),
                                     path, m.group("suffix")))
            else:
                unparsed[split].append(path)

print("=" * 66)
print("PATCH COUNTS")
for s in SPLITS:
    print(f"  {s:11s} {len(index[s]):6d} patches"
          f"   ({len(unparsed[s])} filenames did not match the patch pattern)")
total = sum(len(index[s]) for s in SPLITS) + sum(len(unparsed[s]) for s in SPLITS)
print(f"  {'TOTAL':11s} {total:6d}")

# ------------------------------------------------------- parent overlap
parents = {s: defaultdict(list) for s in SPLITS}
for s in SPLITS:
    for cls, par, r, c, path, suf in index[s]:
        parents[s][par].append((cls, r, c, path))

tr, va, te = (set(parents[s]) for s in SPLITS)
shared_tr_te = sorted(tr & te)
shared_tr_va = sorted(tr & va)
shared_va_te = sorted(va & te)

test_patches_leaked = sum(len(parents["testing"][p]) for p in shared_tr_te)
n_test = len(index["testing"])

print("\n" + "=" * 66)
print("PARENT-RADIOGRAPH OVERLAP")
print(f"  distinct parents  train={len(tr)}  val={len(va)}  test={len(te)}")
print(f"  parents in BOTH train and test : {len(shared_tr_te)}")
print(f"  parents in BOTH train and val  : {len(shared_tr_va)}")
print(f"  parents in BOTH val and test   : {len(shared_va_te)}")
print(f"\n  >>> test patches whose parent radiograph is also in TRAINING: "
      f"{test_patches_leaked} / {n_test} "
      f"({100*test_patches_leaked/max(n_test,1):.1f}% of the test set)")

if test_patches_leaked > 0:
    print("\n  VERDICT: the split is PATCH-LEVEL, not radiograph-level.")
    print("  Test accuracy on this partition does NOT measure generalisation")
    print("  to unseen welds. Report it as in-distribution patch accuracy and")
    print("  build a radiograph-level split for any generalisation claim.")
else:
    print("\n  VERDICT: no parent radiograph is shared between train and test.")

# ------------------------------------------- spatially adjacent crops
adj_pairs, adj_test = 0, set()
train_cells = defaultdict(set)
for cls, par, r, c, path, suf in index["training"]:
    train_cells[par].add((r, c))
for cls, par, r, c, path, suf in index["testing"]:
    if par in train_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in train_cells[par]:
                    adj_pairs += 1
                    adj_test.add(path)
print("\n" + "=" * 66)
print("SPATIALLY ADJACENT CROPS ACROSS SPLITS")
print(f"  test patches with a directly neighbouring crop in training: "
      f"{len(adj_test)} ({100*len(adj_test)/max(n_test,1):.1f}% of test)")
print(f"  total adjacent (test, train) crop pairs: {adj_pairs}")
print("  Neighbouring crops of the same radiograph overlap in texture, film")
print("  grain and often in the defect itself, so these are near-duplicates.")

# ------------------------------------------- suspicious filename suffixes
suspicious = defaultdict(list)
for s in SPLITS:
    for cls, par, r, c, path, suf in index[s]:
        if suf.strip():
            suspicious[s].append(os.path.basename(path))
if any(suspicious.values()):
    print("\n" + "=" * 66)
    print("FILENAMES WITH AN EXTRA SUFFIX AFTER THE PATCH COORDINATES")
    print("  (e.g. ' - Copia' / ' - Copy' marks a duplicated file)")
    for s in SPLITS:
        if suspicious[s]:
            print(f"  {s}: {len(suspicious[s])}")
            for f in suspicious[s][:8]:
                print("      ", f)

# ------------------------------------------------------- MD5 duplicates
dup_report = {}
if args.hash:
    print("\n" + "=" * 66)
    print("EXACT DUPLICATE FILES BY CONTENT (MD5) -- this takes a few minutes")
    digest = defaultdict(list)
    for s in SPLITS:
        for cls, par, r, c, path, suf in index[s]:
            with open(path, "rb") as fh:
                digest[hashlib.md5(fh.read()).hexdigest()].append((s, path))
    cross = {h: v for h, v in digest.items() if len({s for s, _ in v}) > 1}
    within = {h: v for h, v in digest.items()
              if len(v) > 1 and len({s for s, _ in v}) == 1}
    print(f"  identical images appearing in MORE THAN ONE split: {len(cross)}")
    print(f"  identical images duplicated within a single split: {len(within)}")
    for h, v in list(cross.items())[:5]:
        print("      ", [f"{s}:{os.path.basename(p)}" for s, p in v])
    dup_report = {"cross_split_duplicate_groups": len(cross),
                  "within_split_duplicate_groups": len(within)}
else:
    print("\n(Skipping MD5 duplicate check. Re-run with --hash to include it.)")

# ------------------------------------------------------------- save
report = {
    "root": args.root,
    "patch_counts": {s: len(index[s]) for s in SPLITS},
    "total_images": total,
    "distinct_parents": {"training": len(tr), "validation": len(va),
                         "testing": len(te)},
    "parents_shared_train_test": len(shared_tr_te),
    "parents_shared_train_val": len(shared_tr_va),
    "parents_shared_val_test": len(shared_va_te),
    "test_patches_with_parent_in_training": test_patches_leaked,
    "test_patches_with_parent_in_training_pct":
        round(100 * test_patches_leaked / max(n_test, 1), 2),
    "test_patches_with_adjacent_crop_in_training": len(adj_test),
    "adjacent_cross_split_pairs": adj_pairs,
    "filenames_with_extra_suffix":
        {s: len(v) for s, v in suspicious.items()},
    **dup_report,
    "example_shared_parents": shared_tr_te[:25],
}
with open(args.out, "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved:", args.out)
