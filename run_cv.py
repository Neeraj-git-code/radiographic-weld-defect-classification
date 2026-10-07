"""
run_cv.py  --  weld-level grouped k-fold cross-validation for both models.

This is point 3 of your guide's list. It must be GROUPED k-fold, not ordinary
k-fold: every tile of a parent radiograph stays inside one fold, otherwise
each fold reproduces exactly the leakage the paper is about.

HOW IT WORKS
  * scans the ORIGINAL dataset once, de-duplicates by content hash
  * groups tiles by parent radiograph
  * GroupKFold over welds, so each fold tests on welds never trained on
  * for every fold, trains the baseline and the hybrid from scratch
  * writes cv_results/fold_<i>_baseline.json and fold_<i>_hybrid.json
  * no files are copied: it uses flow_from_dataframe

READ THIS BEFORE STARTING
  On your laptop CPU one baseline run took 367 min and one hybrid run 475
  min. Five folds x two models is therefore about 70 hours. Options:

   (a) Google Colab with a GPU. Typically 10-20x faster, so roughly 4-7
       hours for the whole thing. This is what I would do.
   (b) --max-epochs 12 --patience 4 caps the cost. Accuracies come out a
       little lower but the COMPARISON stays fair, because both models get
       the same budget. Say so in the paper.
   (c) --folds 5 rather than 10. Note that Wilcoxon on 5 folds can never
       give p < 0.05: the smallest possible two-sided p with n=5 is 0.0625.
       If your guide wants a significant Wilcoxon across folds, use 10.

USAGE
    python run_cv.py --src "/path/to/DB - Copy" --folds 5
    python run_cv.py --src "..." --folds 10 --max-epochs 12 --patience 4
    python run_cv.py --src "..." --folds 5 --only hybrid     # resume/split up

Then:
    python paired_tests.py --folds cv_results
"""

import os, re, json, time, hashlib, random, argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import (GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Concatenate, BatchNormalization, Dense,
                                     Dropout, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, log_loss
from sklearn.utils.class_weight import compute_class_weight

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="RIAWELC_dataset/DB - Copy")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--out", default="cv_results")
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--warmup-epochs", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--only", choices=["baseline", "hybrid"], default=None)
ap.add_argument("--dedup", action="store_true", default=True)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

PATCH_RE = re.compile(r"^(?P<parent>.+?)_\[\d+\]\[\d+\].*\.png$", re.IGNORECASE)

# ------------------------------------------------------------ 1. index
rows, seen = [], {}
dups = 0
for split in ["training", "validation", "testing"]:
    sdir = os.path.join(args.src, split)
    if not os.path.isdir(sdir):
        continue
    for cls in sorted(os.listdir(sdir)):
        cdir = os.path.join(sdir, cls)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith(".png"):
                continue
            path = os.path.join(cdir, fn)
            if args.dedup:
                with open(path, "rb") as fh:
                    h = hashlib.md5(fh.read()).hexdigest()
                if h in seen:
                    dups += 1
                    continue
                seen[h] = path
            m = PATCH_RE.match(fn)
            parent = m.group("parent") if m else os.path.splitext(fn)[0]
            rows.append({"filepath": path, "cls": cls, "weld": parent})

data = pd.DataFrame(rows)
CLASSES = sorted(data.cls.unique())
print(f"images {len(data):,}   welds {data.weld.nunique()}   "
      f"classes {CLASSES}   duplicates removed {dups:,}")

# ------------------------------------------------------- 2. generators
train_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                               rotation_range=20, zoom_range=0.20,
                               width_shift_range=0.10, height_shift_range=0.10,
                               horizontal_flip=True)
eval_aug = ImageDataGenerator(preprocessing_function=preprocess_input)


def gen(df, aug, shuffle):
    return aug.flow_from_dataframe(df, x_col="filepath", y_col="cls",
                                   classes=CLASSES, target_size=(args.img, args.img),
                                   batch_size=args.batch, class_mode="categorical",
                                   shuffle=shuffle, seed=args.seed, validate_filenames=False)


def build(kind):
    base = EfficientNetB0(weights="imagenet", include_top=False,
                          input_tensor=Input(shape=(args.img, args.img, 3)))
    if kind == "baseline":
        for l in base.layers:
            l.trainable = True
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(512, activation="relu")(x)
    else:
        x = Concatenate()([GlobalAveragePooling2D()(base.output),
                           GlobalMaxPooling2D()(base.output)])
        x = BatchNormalization()(x)
        x = Dense(512, activation="swish")(x)
    x = Dropout(0.5)(x)
    return base, Model(base.input, Dense(len(CLASSES), activation="softmax")(x))


def evaluate(model, g, tag, fold, secs, epochs):
    g.reset()
    t0 = time.time()
    probs = model.predict(g, verbose=0)
    infer = time.time() - t0
    y = g.classes[:len(probs)]
    pred = probs.argmax(1)
    rep = classification_report(y, pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    out = {"fold": fold, "model": tag,
           "accuracy": float(rep["accuracy"]),
           "macro_precision": float(rep["macro avg"]["precision"]),
           "macro_recall": float(rep["macro avg"]["recall"]),
           "macro_f1": float(rep["macro avg"]["f1-score"]),
           "weighted_f1": float(rep["weighted avg"]["f1-score"]),
           "log_loss": float(log_loss(y, probs, labels=list(range(len(CLASSES))))),
           "per_class_f1": {c: float(rep[c]["f1-score"]) for c in CLASSES},
           "n_test_tiles": int(len(y)),
           "train_seconds": round(secs, 1), "epochs": epochs,
           "inference_ms_per_image": round(1000 * infer / max(len(y), 1), 3)}
    json.dump(out, open(os.path.join(args.out, f"fold_{fold}_{tag}.json"), "w"), indent=2)
    pd.DataFrame({"filepath": g.filenames[:len(pred)],
                  "true_class": [CLASSES[i] for i in y],
                  "predicted_class": [CLASSES[i] for i in pred],
                  "confidence": probs.max(1)}).to_csv(
        os.path.join(args.out, f"fold_{fold}_{tag}_predictions.csv"), index=False)
    print(f"    {tag}: acc {100*out['accuracy']:.2f}%  macroF1 {out['macro_f1']:.4f}"
          f"  ({epochs} epochs, {secs/60:.0f} min)")
    return out


# ------------------------------------------------------- 3. the folds
gkf = GroupKFold(n_splits=args.folds)
summary = []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(data, groups=data.weld), start=1):
    tr_all, te = data.iloc[tr_idx], data.iloc[te_idx]
    # carve a validation set out of TRAINING welds only
    welds = tr_all.weld.unique()
    rng = np.random.default_rng(args.seed + fold)
    rng.shuffle(welds)
    n_val = max(1, int(round(0.2 * len(welds))))
    val_w = set(welds[:n_val])
    va = tr_all[tr_all.weld.isin(val_w)]
    tr = tr_all[~tr_all.weld.isin(val_w)]
    assert not (set(tr.weld) & set(te.weld)), "weld leaked into test"
    assert not (set(va.weld) & set(te.weld)), "weld leaked into test"
    print(f"\n=== FOLD {fold}/{args.folds} ===")
    print(f"  train {len(tr):,} tiles / {tr.weld.nunique()} welds | "
          f"val {len(va):,} / {va.weld.nunique()} | "
          f"test {len(te):,} / {te.weld.nunique()}")

    g_tr, g_va, g_te = gen(tr, train_aug, True), gen(va, eval_aug, False), gen(te, eval_aug, False)
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=g_tr.classes)
    cw = {i: float(w) for i, w in enumerate(cw)}

    for kind in (["baseline", "hybrid"] if args.only is None else [args.only]):
        tf.keras.backend.clear_session()
        tf.random.set_seed(args.seed + fold)
        base, model = build(kind)
        cbs = [EarlyStopping(monitor="val_loss", patience=args.patience,
                             restore_best_weights=True, verbose=0),
               ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                 patience=max(2, args.patience // 2), verbose=0)]
        t0 = time.time(); ep = 0
        if kind == "baseline":
            model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy",
                          metrics=["accuracy"])
            h = model.fit(g_tr, validation_data=g_va, epochs=args.max_epochs,
                          callbacks=cbs, verbose=1)
            ep = len(h.history["loss"])
        else:
            for l in base.layers:
                l.trainable = False
            model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy",
                          metrics=["accuracy"])
            h1 = model.fit(g_tr, validation_data=g_va, epochs=args.warmup_epochs,
                           callbacks=[EarlyStopping(monitor="val_loss", patience=4,
                                                    restore_best_weights=True)],
                           verbose=1)
            # backbone BatchNorm stays frozen -- the fix from the main paper
            for l in base.layers:
                l.trainable = not isinstance(l, BatchNormalization)
            model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy",
                          metrics=["accuracy"])
            h2 = model.fit(g_tr, validation_data=g_va, epochs=args.max_epochs,
                           callbacks=cbs, class_weight=cw, verbose=1)
            ep = len(h1.history["loss"]) + len(h2.history["loss"])
        summary.append(evaluate(model, g_te, kind, fold, time.time() - t0, ep))

# ------------------------------------------------------------ 4. report
if summary:
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.out, "cv_summary.csv"), index=False)
    print("\n" + "=" * 60)
    for kind, grp in s.groupby("model"):
        print(f"{kind:9s} accuracy {100*grp.accuracy.mean():.2f}% "
              f"+/- {100*grp.accuracy.std(ddof=1):.2f}   "
              f"macroF1 {grp.macro_f1.mean():.4f} +/- {grp.macro_f1.std(ddof=1):.4f}")
    print("=" * 60)
    print(f"wrote {args.out}/cv_summary.csv")
    print("next:  python paired_tests.py --folds", args.out)
