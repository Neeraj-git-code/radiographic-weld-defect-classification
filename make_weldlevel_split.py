"""
make_weldlevel_split.py
=======================
Rebuilds RIAWELC as a WELD-LEVEL (parent-radiograph) split.

WHY
---
RIAWELC images are tiles cropped from larger parent radiographs. The filename
records both:

    RRT-101R_Img1_A80_S4_[16][1].png
    |_________________|  |__||_|
      parent radiograph  row  col

The distributed split shuffles TILES, so tiles from one weld land in training
AND testing. A model then only has to recognise a weld it already memorised
from a neighbouring crop, which is why baselines reach ~100%. This script
assigns every tile of a given parent radiograph to exactly ONE split, so the
test set contains welds the model has genuinely never seen.

TWO COMPLICATIONS THIS HANDLES
------------------------------
1. A parent radiograph can contain tiles of MORE THAN ONE class (84 of 398
   test parents do). The whole parent must still move as one unit, so exact
   per-class ratios are impossible. The script uses a greedy assignment that
   minimises per-class deviation from the target ratios.
2. Tile counts per parent are very uneven (1 to 89 in the test split alone).
   Parents are therefore assigned largest-first, which is the standard
   bin-packing heuristic and gives much tighter balance than random order.

OUTPUT
------
    DB_weldlevel/
        training/   Difetto1/ Difetto2/ Difetto4/ NoDifetto/
        validation/ ...
        testing/    ...
    split_manifest.json   which parent went to which split, plus the
                          resulting class distribution and deviations

Files are HARD-LINKED by default, so the new folder costs almost no disk
space. Use --copy if your filesystem does not support hard links.

USAGE
-----
    python make_weldlevel_split.py --src "RIAWELC_dataset/DB - Copy"
    python make_weldlevel_split.py --src "..." --ratios 0.65 0.25 0.10
    python make_weldlevel_split.py --src "..." --copy --seed 42

Then point the training scripts at it:
    python weld_def1_baseline.py --root DB_weldlevel --tag weldlevel
    python weld_def2_hybrid.py   --root DB_weldlevel --tag weldlevel
"""

import os, re, json, shutil, random, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="RIAWELC_dataset/DB - Copy",
                help="existing dataset root containing training/validation/testing")
ap.add_argument("--dst", default="DB_weldlevel", help="output root")
ap.add_argument("--ratios", nargs=3, type=float, default=[0.65, 0.25, 0.10],
                metavar=("TRAIN", "VAL", "TEST"))
ap.add_argument("--copy", action="store_true",
                help="copy files instead of hard-linking")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dry-run", action="store_true",
                help="report the split without writing any files")
ap.add_argument("--dedup", action="store_true",
                help="drop byte-identical duplicate images (keeps one copy). "
                     "Recommended: the distributed RIAWELC copy contains the "
                     "same crop in more than one split.")
args = ap.parse_args()

random.seed(args.seed)
SPLITS = ["training", "validation", "testing"]
TARGET = dict(zip(SPLITS, args.ratios))
assert abs(sum(args.ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

# the trailing part after [row][col] may carry junk like " - Copia"
PATCH_RE = re.compile(r"^(?P<parent>.+?)_\[(?P<row>\d+)\]\[(?P<col>\d+)\]"
                      r"(?P<suffix>.*)\.png$", re.IGNORECASE)

# ------------------------------------------------------------------ 1. scan
# parent -> class -> [absolute source paths]
parent_files = defaultdict(lambda: defaultdict(list))
unparsed = []
classes = set()

for split in SPLITS:
    sdir = os.path.join(args.src, split)
    if not os.path.isdir(sdir):
        raise SystemExit(f"Not found: {sdir}\nCheck --src.")
    for cls in sorted(os.listdir(sdir)):
        cdir = os.path.join(sdir, cls)
        if not os.path.isdir(cdir):
            continue
        classes.add(cls)
        for fn in os.listdir(cdir):
            if not fn.lower().endswith(".png"):
                continue
            path = os.path.join(cdir, fn)
            m = PATCH_RE.match(fn)
            if m:
                parent_files[m.group("parent")][cls].append(path)
            else:
                # no parseable tile index: treat the stem as its own parent so
                # it is never silently dropped
                unparsed.append(path)
                parent_files[os.path.splitext(fn)[0]][cls].append(path)

CLASSES = sorted(classes)

# ------------------------------------------------- 1b. optional dedup
dropped_dups = 0
if args.dedup:
    import hashlib
    print("Hashing every image to find byte-identical duplicates ...")
    seen = {}
    for parent, per_class in parent_files.items():
        for cls, files in per_class.items():
            keep = []
            for path in files:
                with open(path, "rb") as fh:
                    h = hashlib.md5(fh.read()).hexdigest()
                if h in seen:
                    dropped_dups += 1
                else:
                    seen[h] = path
                    keep.append(path)
            per_class[cls] = keep
    # clear out any class entry that became empty
    for parent in list(parent_files):
        for cls in list(parent_files[parent]):
            if not parent_files[parent][cls]:
                del parent_files[parent][cls]
        if not parent_files[parent]:
            del parent_files[parent]
    print(f"  dropped {dropped_dups} byte-identical duplicate images")

total_tiles = sum(len(v) for p in parent_files.values() for v in p.values())

print("=" * 70)
print("SOURCE SCAN")
print(f"  classes            : {CLASSES}")
print(f"  parent radiographs : {len(parent_files)}")
print(f"  tiles              : {total_tiles}")
print(f"  filenames without a [row][col] index: {len(unparsed)}")
multi = [p for p, c in parent_files.items() if len(c) > 1]
print(f"  parents spanning >1 class: {len(multi)}")

class_totals = defaultdict(int)
for p in parent_files.values():
    for cls, files in p.items():
        class_totals[cls] += len(files)

# --------------------------------------------------- 2. greedy assignment
# Assign each parent whole. Pick the split that leaves per-class proportions
# closest to target. Largest parents first so the big lumps land early.
assigned = {}
cur = {s: defaultdict(int) for s in SPLITS}      # split -> class -> tiles

order = sorted(parent_files.items(),
               key=lambda kv: -sum(len(f) for f in kv[1].values()))

# quota[split][class] = how many tiles of that class this split should end up with
quota = {s: {c: TARGET[s] * class_totals[c] for c in CLASSES} for s in SPLITS}

def fill_if(split, counts):
    """How full `split` would become, relative to its quota, if this parent
    were added. Measured as the WORST per-class fill ratio among the classes
    this parent actually contributes, so a parent is sent to whichever split
    is furthest below quota for the classes it carries. Ties break on overall
    fill, which keeps the totals in line too."""
    worst = 0.0
    for cls, n in counts.items():
        q = quota[split][cls]
        if q <= 0:
            return float("inf")          # this split wants none of this class
        worst = max(worst, (cur[split][cls] + n) / q)
    total_q = sum(quota[split].values())
    overall = (sum(cur[split].values()) + sum(counts.values())) / total_q
    return worst + 1e-6 * overall

for parent, per_class in order:
    counts = {cls: len(f) for cls, f in per_class.items()}
    best = min(SPLITS, key=lambda s: fill_if(s, counts))
    assigned[parent] = best
    for cls, n in counts.items():
        cur[best][cls] += n

# ------------------------------------------------------------- 3. report
print("\n" + "=" * 70)
print("RESULTING SPLIT")
hdr = f"  {'class':<11}" + "".join(f"{s[:5]:>10}" for s in SPLITS) + f"{'total':>10}"
print(hdr)
rows = {}
for cls in CLASSES:
    tot = class_totals[cls]
    line = f"  {cls:<11}"
    rows[cls] = {}
    for s in SPLITS:
        n = cur[s][cls]
        rows[cls][s] = n
        line += f"{n:>10}"
    line += f"{tot:>10}"
    print(line)

print(f"\n  {'share':<11}" + "".join(f"{s[:5]:>10}" for s in SPLITS))
max_dev = 0.0
for cls in CLASSES:
    tot = class_totals[cls]
    line = f"  {cls:<11}"
    for s in SPLITS:
        share = cur[s][cls] / tot if tot else 0
        max_dev = max(max_dev, abs(share - TARGET[s]))
        line += f"{share*100:>9.1f}%"
    print(line)
print(f"  target      " + "".join(f"{TARGET[s]*100:>9.1f}%" for s in SPLITS))
print(f"\n  worst class-share deviation from target: {max_dev*100:.2f} "
      f"percentage points")

n_parents = {s: sum(1 for p, a in assigned.items() if a == s) for s in SPLITS}
n_tiles = {s: sum(cur[s].values()) for s in SPLITS}
print(f"\n  parents per split: " +
      ", ".join(f"{s}={n_parents[s]}" for s in SPLITS))
print(f"  tiles   per split: " +
      ", ".join(f"{s}={n_tiles[s]}" for s in SPLITS))

# ------------------------------------------------------- 4. sanity check
overlap = False
for a in range(len(SPLITS)):
    for b in range(a + 1, len(SPLITS)):
        pa = {p for p, s in assigned.items() if s == SPLITS[a]}
        pb = {p for p, s in assigned.items() if s == SPLITS[b]}
        shared = pa & pb
        if shared:
            overlap = True
            print(f"  !! {len(shared)} parents shared "
                  f"{SPLITS[a]}<->{SPLITS[b]}")
print("\n  parent overlap between splits:",
      "FOUND (bug)" if overlap else "NONE  <-- this is the point of the script")

if n_tiles["testing"] == 0 or n_parents["testing"] == 0:
    raise SystemExit("\nTest split is empty. Adjust --ratios.")

# ------------------------------------------------------------- 5. write
manifest = {
    "source": args.src, "destination": args.dst, "seed": args.seed,
    "ratios": dict(zip(SPLITS, args.ratios)),
    "classes": CLASSES,
    "parents_total": len(parent_files),
    "parents_multiclass": len(multi),
    "tiles_total": total_tiles,
    "deduplicated": bool(args.dedup),
    "duplicate_images_dropped": dropped_dups,
    "parents_per_split": n_parents,
    "tiles_per_split": n_tiles,
    "class_counts_per_split": {s: {c: cur[s][c] for c in CLASSES}
                               for s in SPLITS},
    "worst_class_share_deviation_pp": round(max_dev * 100, 3),
    "parent_assignment": assigned,
}

if args.dry_run:
    print("\n(dry run: no files written)")
    with open("split_manifest_dryrun.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved: split_manifest_dryrun.json")
    raise SystemExit(0)

print("\n" + "=" * 70)
print(f"WRITING {args.dst}  ({'copying' if args.copy else 'hard-linking'})")
for s in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(args.dst, s, cls), exist_ok=True)

written, collisions = 0, 0
for parent, per_class in parent_files.items():
    split = assigned[parent]
    for cls, files in per_class.items():
        outdir = os.path.join(args.dst, split, cls)
        for src in files:
            name = os.path.basename(src)
            dst = os.path.join(outdir, name)
            if os.path.exists(dst):
                # identical filename under the same class from a different
                # source split: keep both, they may be different images
                stem, ext = os.path.splitext(name)
                k = 1
                while os.path.exists(dst):
                    dst = os.path.join(outdir, f"{stem}__dup{k}{ext}")
                    k += 1
                collisions += 1
            try:
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)      # fallback if hard link fails
            written += 1
    if written % 4000 < 20:
        print(f"    {written}/{total_tiles} ...")

print(f"  wrote {written} files ({collisions} filename collisions renamed)")

with open(os.path.join(args.dst, "split_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  saved {os.path.join(args.dst, 'split_manifest.json')}")

print("\nNEXT:")
print(f"  python check_leakage.py --root {args.dst}      # expect 0 shared parents")
print(f"  python weld_def1_baseline.py --root {args.dst} --tag weldlevel")
print(f"  python weld_def2_hybrid.py   --root {args.dst} --tag weldlevel")
