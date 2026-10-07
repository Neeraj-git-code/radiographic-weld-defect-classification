"""
EXPERIMENT 3 - HYBRID-POOLING EfficientNetB0  (corrected version)
=================================================================
Changes vs. the original weld_def2.py:

  [H1] *** THE IMPORTANT ONE ***  Backbone BatchNormalization layers are kept
       frozen (inference mode) when the backbone is unfrozen for Phase 2.
       In the original run every BN layer was set trainable at the phase
       boundary, so the backbone switched from ImageNet moving statistics to
       mini-batch statistics in one step. That is what produced the collapse
       from 96.05% validation accuracy at the end of Phase 1 to 80.64% in the
       first Phase-2 epoch, and it cost 17 epochs to climb back above 99%.
       Freezing backbone BN is the standard Keras fine-tuning recipe.

  [H2] ABLATION MODE (--ablation). Experiment 3 changed five things at once:
       pooling, head normalization, head activation, training schedule and
       class weighting. That confounds the comparison and the paper has to
       admit it. Running with --ablation keeps ONLY the hybrid pooling and
       matches the baseline on everything else (ReLU head, no BN, single
       phase, lr 1e-4, no class weights), which isolates the pooling effect.

  [H3] Wall-clock training and per-image inference time are measured.
  [H4] Per-class ROC-AUC and average precision saved to metrics.json.
  [H5] Selective-classification sweep saved to coverage_risk.csv, so the 0.98
       threshold is chosen from evidence rather than asserted.
  [H6] Publication-quality figures, no overlapping suptitles.
  [H7] EPOCHS_EXECUTED recorded explicitly per phase.
  [H8] --seed lets you run repeated trials for the significance test the
       paper's Future Work section calls for.

Usage:
    python weld_def2_hybrid_fixed.py                 # corrected hybrid
    python weld_def2_hybrid_fixed.py --ablation      # pooling-only ablation
    python weld_def2_hybrid_fixed.py --seed 7
"""

import os, json, time, random, re, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0, preprocess_input
)
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate,
    BatchNormalization, Dense, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from sklearn.metrics import (
    classification_report, confusion_matrix, log_loss,
    roc_curve, auc, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 1. CONFIGURATION
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("--ablation", action="store_true",
                help="[H2] isolate hybrid pooling; match baseline elsewhere")
ap.add_argument("--root", default="RIAWELC_dataset/DB - Copy",
                help="dataset root. Use DB_weldlevel for the weld-level split.")
ap.add_argument("--tag", default="",
                help="suffix for the output folder, e.g. --tag weldlevel")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_ROOT = args.root
TRAIN_DIR = os.path.join(DATA_ROOT, "training")
VAL_DIR   = os.path.join(DATA_ROOT, "validation")
TEST_DIR  = os.path.join(DATA_ROOT, "testing")

TAG = "ablation" if args.ablation else "hybrid"
SUFFIX = (f"_{args.tag}" if args.tag else "") + \
         (f"_seed{SEED}" if SEED != 42 else "")
OUTPUT_DIR = f"{TAG}_results{SUFFIX}"
PAPER_DIR  = f"paper_figures{(f'_{args.tag}' if args.tag else '')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR,  exist_ok=True)

IMG_SIZE, BATCH_SIZE = 224, 16
PHASE1_EPOCHS, PHASE1_LR = 10,  1e-3
PHASE2_EPOCHS, PHASE2_LR = 40,  1e-5
ABLATION_EPOCHS, ABLATION_LR = 40, 1e-4            # identical to baseline

IEEE_COL = 3.45
plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def save(fig, *paths):
    for p in paths:
        fig.savefig(p)
    plt.close(fig)

def paper(name):
    """Only the canonical seed-42 hybrid run writes the paper figures."""
    return [os.path.join(PAPER_DIR, name)] \
        if (not args.ablation and SEED == 42) else []

# ============================================================
# 2. DATA GENERATORS  (identical to the baseline script)
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, zoom_range=0.20,
    width_shift_range=0.10, height_shift_range=0.10,
    horizontal_flip=True,
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, seed=SEED)
val_gen = eval_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False)
test_gen = eval_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False)

class_indices = train_gen.class_indices
CLASS_NAMES = [c for c, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]
NUM_CLASSES = len(CLASS_NAMES)
with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=2)

dist = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Training":   np.bincount(train_gen.classes, minlength=NUM_CLASSES),
    "Validation": np.bincount(val_gen.classes,   minlength=NUM_CLASSES),
    "Testing":    np.bincount(test_gen.classes,  minlength=NUM_CLASSES),
})
dist.to_csv(os.path.join(OUTPUT_DIR, "dataset_distribution.csv"), index=False)
print(dist.to_string(index=False))
print("TOTAL IMAGES:", dist[["Training", "Validation", "Testing"]].values.sum())

# ============================================================
# 3. CLASS WEIGHTS
# ============================================================
weights = compute_class_weight("balanced",
                               classes=np.arange(NUM_CLASSES),
                               y=train_gen.classes)
class_weights = {i: float(w) for i, w in enumerate(weights)}
with open(os.path.join(OUTPUT_DIR, "class_weights.json"), "w") as f:
    json.dump({CLASS_NAMES[i]: w for i, w in class_weights.items()}, f, indent=2)
print("Class weights:", class_weights)

# ============================================================
# 4. MODEL  (hybrid pooling head)
# ============================================================
base_model = EfficientNetB0(
    weights="imagenet", include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

avg_pool = GlobalAveragePooling2D()(base_model.output)
max_pool = GlobalMaxPooling2D()(base_model.output)
x = Concatenate()([avg_pool, max_pool])

if args.ablation:
    # [H2] head matched to the baseline: no BN, ReLU. Only pooling differs.
    x = Dense(512, activation="relu")(x)
else:
    x = BatchNormalization()(x)
    x = Dense(512, activation="swish")(x)

x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(base_model.input, output)

total_params = int(model.count_params())

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        (logs if logs is not None else {})["learning_rate"] = \
            float(tf.keras.backend.get_value(lr))

def make_callbacks(ckpt, es_patience, lr_patience):
    return [
        LearningRateLogger(),
        EarlyStopping(monitor="val_loss", patience=es_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                          patience=lr_patience, verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, ckpt),
                        monitor="val_loss", save_best_only=True, verbose=1),
    ]

# ============================================================
# 5. TRAINING
# ============================================================
train_seconds = 0.0

if args.ablation:
    # ---------- single phase, baseline-matched ----------
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=ABLATION_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    t0 = time.time()
    h = model.fit(train_gen, validation_data=val_gen, epochs=ABLATION_EPOCHS,
                  callbacks=make_callbacks("best_ablation.keras", 10, 5),
                  verbose=1)                      # no class_weight: matched
    train_seconds = time.time() - t0
    hist = pd.DataFrame(h.history)
    hist.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    P1_EXEC, P2_EXEC = 0, len(hist)
    combined = hist

else:
    # ---------- Phase 1: frozen backbone, head warm-up ----------
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=PHASE1_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    print("\n=== PHASE 1: HEAD WARM-UP (backbone frozen) ===")
    t0 = time.time()
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=PHASE1_EPOCHS,
                   callbacks=make_callbacks("phase1_best.keras", 4, 3),
                   verbose=1)
    train_seconds += time.time() - t0
    hist1 = pd.DataFrame(h1.history)
    hist1.to_csv(os.path.join(OUTPUT_DIR, "phase1_history.csv"), index=False)
    P1_EXEC = len(hist1)

    # ---------- Phase 2: full fine-tuning ----------
    # [H1] THE FIX: unfreeze everything EXCEPT backbone BatchNorm layers.
    n_bn = 0
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False              # stay in inference mode
            n_bn += 1
        else:
            layer.trainable = True
    print(f"\n=== PHASE 2: FULL FINE-TUNING "
          f"({n_bn} backbone BatchNorm layers kept frozen) ===")

    model.compile(optimizer=Adam(learning_rate=PHASE2_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=PHASE2_EPOCHS,
                   callbacks=make_callbacks("best_hybrid.keras", 6, 3),
                   class_weight=class_weights, verbose=1)
    train_seconds += time.time() - t0
    hist2 = pd.DataFrame(h2.history)
    hist2.to_csv(os.path.join(OUTPUT_DIR, "phase2_history.csv"), index=False)
    P2_EXEC = len(hist2)
    combined = pd.concat([hist1, hist2], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "combined_history.csv"),
                    index=False)

EPOCHS_EXECUTED = P1_EXEC + P2_EXEC                                   # [H7]
print(f"\nEPOCHS EXECUTED: phase1={P1_EXEC} phase2={P2_EXEC} "
      f"total={EPOCHS_EXECUTED}")
print(f"TRAINING TIME: {train_seconds/60:.1f} min")

# ------------------------------------------------------------
# Figs. 4 and 5 - combined accuracy and loss
# ------------------------------------------------------------
eh = np.arange(1, len(combined) + 1)

fig, ax = plt.subplots(figsize=(IEEE_COL, 1.95))
ax.plot(eh, combined["accuracy"], label="Training", color="#2c6fbb")
ax.plot(eh, combined["val_accuracy"], label="Validation", color="#e08214")
if P1_EXEC:
    ax.axvline(P1_EXEC + 0.5, color="0.45", ls="--", lw=0.9)
    ax.text(P1_EXEC / 2, 1.005, "Phase 1", fontsize=6.5, ha="center", color="0.3")
    ax.text(P1_EXEC + P2_EXEC / 2, 1.005, "Phase 2", fontsize=6.5,
            ha="center", color="0.3")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)
save(fig, os.path.join(OUTPUT_DIR, "02_combined_accuracy.png"),
          *paper("fig04_hyb_accuracy.png"))

fig, ax = plt.subplots(figsize=(IEEE_COL, 1.95))
ax.plot(eh, combined["loss"], label="Training", color="#2c6fbb")
ax.plot(eh, combined["val_loss"], label="Validation", color="#e08214")
if P1_EXEC:
    ax.axvline(P1_EXEC + 0.5, color="0.45", ls="--", lw=0.9)
ax.set_yscale("log")
ax.set_xlabel("Epoch"); ax.set_ylabel("Categorical cross-entropy")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3, which="both")
save(fig, os.path.join(OUTPUT_DIR, "03_combined_loss.png"),
          *paper("fig05_hyb_loss.png"))

# ============================================================
# 6. TEST EVALUATION
# ============================================================
test_gen.reset()
t0 = time.time()
probs = model.predict(test_gen, verbose=1)
infer_seconds = time.time() - t0

y_true = test_gen.classes
y_pred = np.argmax(probs, axis=1)
conf = probs.max(axis=1)
correct = (y_true == y_pred)

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                               digits=4, zero_division=0)
print(report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                            output_dict=True, zero_division=0)

y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
per_class_auc, per_class_ap = {}, {}
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    per_class_auc[name] = float(auc(fpr, tpr))
    per_class_ap[name] = float(average_precision_score(y_bin[:, i], probs[:, i]))

metrics = {
    "run_type": TAG, "seed": SEED, "data_root": DATA_ROOT, "tag": args.tag,
    "accuracy":        float(rep["accuracy"]),
    "macro_precision": float(rep["macro avg"]["precision"]),
    "macro_recall":    float(rep["macro avg"]["recall"]),
    "macro_f1":        float(rep["macro avg"]["f1-score"]),
    "weighted_f1":     float(rep["weighted avg"]["f1-score"]),
    "log_loss":        float(log_loss(y_true, probs,
                                      labels=list(range(NUM_CLASSES)))),
    "per_class_auc": per_class_auc, "per_class_ap": per_class_ap,      # [H4]
    "macro_auc": float(np.mean(list(per_class_auc.values()))),
    "macro_ap": float(np.mean(list(per_class_ap.values()))),
    "total_params": total_params,
    "epochs_phase1": P1_EXEC, "epochs_phase2": P2_EXEC,                # [H7]
    "epochs_executed": EPOCHS_EXECUTED,
    "train_seconds": round(train_seconds, 1),                          # [H3]
    "seconds_per_epoch": round(train_seconds / max(EPOCHS_EXECUTED, 1), 2),
    "test_inference_seconds": round(infer_seconds, 2),
    "inference_ms_per_image": round(1000 * infer_seconds /
                                    len(test_gen.filenames), 3),
    "mean_conf_correct":   float(conf[correct].mean()),
    "mean_conf_incorrect": float(conf[~correct].mean()) if (~correct).any()
                           else None,
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print(json.dumps(metrics, indent=2))

pd.DataFrame({
    "filename": test_gen.filenames,
    "true_class":      [CLASS_NAMES[i] for i in y_true],
    "predicted_class": [CLASS_NAMES[i] for i in y_pred],
    "confidence": conf,
}).to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# ============================================================
# 7. [H5] SELECTIVE-CLASSIFICATION COVERAGE-RISK SWEEP
# ============================================================
rows = []
for tau in np.round(np.arange(0.50, 1.0001, 0.01), 2):
    keep = conf >= tau
    rows.append({
        "threshold": tau,
        "deferred": int((~keep).sum()),
        "deferred_pct": round(100 * (~keep).mean(), 3),
        "coverage_pct": round(100 * keep.mean(), 3),
        "errors_deferred": int((~keep & ~correct).sum()),
        "errors_retained": int((keep & ~correct).sum()),
        "retained_accuracy": round(float(correct[keep].mean()), 6)
                             if keep.any() else None,
    })
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(OUTPUT_DIR, "coverage_risk.csv"), index=False)
clean = cov[cov.errors_retained == 0]
if not clean.empty:
    best = clean.iloc[0]
    print(f"\n[H5] Lowest threshold giving 100% retained accuracy: "
          f"tau={best.threshold} deferring {best.deferred} images "
          f"({best.deferred_pct}%)")

# ------------------------------------------------------------
# Figs. 7, 8 - confusion matrices
# ------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
for norm, fname, papername in [
        (False, "09_confusion_matrix.png", "fig07_hyb_confusion.png"),
        (True, "10_normalized_confusion_matrix.png",
         "fig08_hyb_confusion_norm.png")]:
    M = cm / cm.sum(1, keepdims=True) if norm else cm
    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.55))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if norm else M.max())
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            txt = f"{M[i, j]:.4f}" if norm else f"{int(M[i, j]):d}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.6 if norm else 7.5,
                    color="white" if M[i, j] > M.max() * 0.55 else "0.15",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    save(fig, os.path.join(OUTPUT_DIR, fname), *paper(papername))

# ------------------------------------------------------------
# Fig. 9 - misclassified examples
# ------------------------------------------------------------
from tensorflow.keras.preprocessing.image import load_img, img_to_array
wrong = np.where(~correct)[0]
if len(wrong):
    n = min(len(wrong), 6); ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(IEEE_COL, 1.28 * nrow))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k < n:
            i = wrong[k]
            img = load_img(os.path.join(TEST_DIR, test_gen.filenames[i]),
                           target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{CLASS_NAMES[y_true[i]]} $\\rightarrow$ "
                         f"{CLASS_NAMES[y_pred[i]]}\nconf. {conf[i]:.2f}",
                         fontsize=6.2, pad=2)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.22, hspace=0.45)
    save(fig, os.path.join(OUTPUT_DIR, "16_misclassified_examples.png"),
              *paper("fig09_hyb_misclassified.png"))

# ------------------------------------------------------------
# Fig. 10 - per-class metrics
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(IEEE_COL, 1.95))
xs = np.arange(NUM_CLASSES); w = 0.26
for i, key in enumerate(["precision", "recall", "f1-score"]):
    vals = [rep[c][key] for c in CLASS_NAMES]
    b = ax.bar(xs + (i - 1) * w, vals, w, label=key.capitalize())
    ax.bar_label(b, fmt="%.3f", fontsize=4.8, padding=1)
ax.set_xticks(xs); ax.set_xticklabels(CLASS_NAMES, rotation=15)
ax.set_ylabel("Score"); ax.set_ylim(0.95, 1.008)
ax.legend(frameon=False, ncol=3, loc="lower center")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
save(fig, os.path.join(OUTPUT_DIR, "11_per_class_metrics.png"),
          *paper("fig10_hyb_per_class.png"))

# ------------------------------------------------------------
# Fig. 11 - ROC curves
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(IEEE_COL, 2.4))
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    ax.plot(fpr, tpr, lw=1.1, label=f"{name} (AUC={per_class_auc[name]:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)
save(fig, os.path.join(OUTPUT_DIR, "14_roc_curves.png"),
          *paper("fig11_hyb_roc.png"))

# ------------------------------------------------------------
# Fig. 12 - confidence distribution (log scale so errors are visible)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(IEEE_COL, 1.95))
bins = np.linspace(min(0.45, conf.min()), 1.0, 30)
ax.hist(conf[correct], bins=bins, color="#2c6fbb",
        label=f"Correct ({int(correct.sum()):,})")
if (~correct).any():
    ax.hist(conf[~correct], bins=bins, color="#c0392b",
            label=f"Incorrect ({int((~correct).sum())})")
ax.axvline(0.98, color="0.25", ls="--", lw=0.9)
ax.set_yscale("log")
ax.set_xlabel("Maximum softmax probability")
ax.set_ylabel("Test images (log scale)")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3, which="both")
save(fig, os.path.join(OUTPUT_DIR, "13_confidence_distribution.png"),
          *paper("fig12_hyb_confidence.png"))

# ============================================================
# 8. GRAD-CAM  ([H6] no overlapping suptitle)
# ============================================================
target_layer = next(l for l in reversed(model.layers)
                    if len(getattr(l, "output_shape", ())) == 4)
print("Grad-CAM target layer:", target_layer.name)
grad_model = Model(model.inputs, [target_layer.output, model.output])

def make_gradcam(img_array, class_idx):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

N_CAM = 2
idxs = np.linspace(0, len(test_gen.filenames) - 1, N_CAM).astype(int)
fig, axes = plt.subplots(N_CAM, 2, figsize=(IEEE_COL, 1.72 * N_CAM))
for row, k in enumerate(idxs):
    raw = img_to_array(load_img(os.path.join(TEST_DIR, test_gen.filenames[k]),
                                target_size=(IMG_SIZE, IMG_SIZE)))
    cam = make_gradcam(preprocess_input(np.expand_dims(raw.copy(), 0)),
                       int(y_pred[k]))
    axes[row, 0].imshow(raw.astype("uint8"))
    axes[row, 1].imshow(cam, cmap="jet")
    for c in (0, 1):
        axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    if row == 0:
        axes[row, 0].set_title("Input radiograph", fontsize=7.5, pad=3)
        axes[row, 1].set_title("Hybrid Grad-CAM", fontsize=7.5, pad=3)
    axes[row, 0].set_ylabel(f"Sample {row+1}", fontsize=7)
fig.subplots_adjust(wspace=0.05, hspace=0.06)
save(fig, os.path.join(OUTPUT_DIR, "18_gradcam.png"),
          *paper("fig14_hyb_gradcam.png"))

print("\nDONE. Outputs in", OUTPUT_DIR)
