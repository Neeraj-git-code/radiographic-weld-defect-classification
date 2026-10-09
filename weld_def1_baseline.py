import os, json, time, random, re, argparse
from collections import defaultdict

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
    GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from sklearn.metrics import (
    classification_report, confusion_matrix, log_loss,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# 1. CONFIGURATION

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="RIAWELC_dataset/DB - Copy",
                help="dataset root. Use DB_weldlevel for the weld-level split.")
ap.add_argument("--tag", default="",
                help="suffix for the output folder, e.g. --tag weldlevel")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_ROOT  = args.root
TRAIN_DIR  = os.path.join(DATA_ROOT, "training")
VAL_DIR    = os.path.join(DATA_ROOT, "validation")
TEST_DIR   = os.path.join(DATA_ROOT, "testing")

SUFFIX     = (f"_{args.tag}" if args.tag else "") + \
             (f"_seed{SEED}" if SEED != 42 else "")
OUTPUT_DIR = f"baseline_results{SUFFIX}"
PAPER_DIR  = f"paper_figures{SUFFIX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR,  exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 16
MAX_EPOCHS = 40
INIT_LR    = 1e-4

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

# 2. DATA GENERATORS  (augmentation on TRAIN only)

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
print("Classes:", class_indices)

with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=2)

# Dataset distribution table + Fig. 1

def counts(gen):
    c = np.bincount(gen.classes, minlength=NUM_CLASSES)
    return c

dist = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Training":   counts(train_gen),
    "Validation": counts(val_gen),
    "Testing":    counts(test_gen),
})
dist.to_csv(os.path.join(OUTPUT_DIR, "dataset_distribution.csv"), index=False)
print(dist.to_string(index=False))
print("TOTAL IMAGES:", dist[["Training", "Validation", "Testing"]].values.sum())

fig, ax = plt.subplots(figsize=(IEEE_COL, 2.05))
x = np.arange(NUM_CLASSES); w = 0.27
for i, (col, colr) in enumerate(zip(["Training", "Validation", "Testing"],
                                    ["#2c6fbb", "#e08214", "#3f8f3f"])):
    b = ax.bar(x + (i - 1) * w, dist[col], w, label=col, color=colr)
    ax.bar_label(b, fmt="%d", fontsize=5, padding=1)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("Number of images")
ax.set_ylim(0, dist[["Training"]].values.max() * 1.18)
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
save(fig, os.path.join(OUTPUT_DIR, "01_dataset_distribution.png"),
          os.path.join(PAPER_DIR,  "fig01_dataset_distribution.png"))

# 3. [C4] RADIOGRAPH-LEVEL LEAKAGE AUDIT

PARENT_RE = re.compile(r"_\[\d+\]\[\d+\]\.png$", re.IGNORECASE)

def parents(gen):
    out = defaultdict(set)
    for fn in gen.filenames:
        base = os.path.basename(fn)
        out[PARENT_RE.sub("", base)].add(fn)
    return out

p_tr, p_va, p_te = parents(train_gen), parents(val_gen), parents(test_gen)
audit = {
    "patches":  {"train": len(train_gen.filenames),
                 "val": len(val_gen.filenames),
                 "test": len(test_gen.filenames)},
    "distinct_parent_radiographs": {"train": len(p_tr), "val": len(p_va),
                                    "test": len(p_te)},
    "shared_parents_train_test": sorted(set(p_tr) & set(p_te)),
    "shared_parents_train_val":  sorted(set(p_tr) & set(p_va)),
    "shared_parents_val_test":   sorted(set(p_va) & set(p_te)),
}
audit["n_shared_train_test"] = len(audit["shared_parents_train_test"])
audit["test_patches_with_shared_parent"] = sum(
    len(p_te[k]) for k in audit["shared_parents_train_test"])
with open(os.path.join(OUTPUT_DIR, "leakage_audit.json"), "w") as f:
    json.dump(audit, f, indent=2)

print("\n===== RADIOGRAPH-LEVEL LEAKAGE AUDIT =====")
print("distinct parent radiographs:", audit["distinct_parent_radiographs"])
print("parents shared train<->test:", audit["n_shared_train_test"])
print("test patches whose parent also appears in training:",
      audit["test_patches_with_shared_parent"],
      f'({100*audit["test_patches_with_shared_parent"]/len(test_gen.filenames):.1f}% of test)')
print("==========================================\n")

# 4. MODEL  (unchanged architecture)

base_model = EfficientNetB0(
    weights="imagenet", include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
for layer in base_model.layers:
    layer.trainable = True                      # single-phase full fine-tuning

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(base_model.input, output)

model.compile(optimizer=Adam(learning_rate=INIT_LR),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

total_params     = int(model.count_params())
trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))

# 5. TRAIN  ([C1] timed)

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        logs = logs or {}
        logs["learning_rate"] = float(tf.keras.backend.get_value(lr))

callbacks = [
    LearningRateLogger(),
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_baseline.keras"),
                    monitor="val_loss", save_best_only=True, verbose=1),
]

t0 = time.time()
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=MAX_EPOCHS, callbacks=callbacks, verbose=1)
train_seconds = time.time() - t0

hist = pd.DataFrame(history.history)
hist.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
EPOCHS_EXECUTED = len(hist)                                          # [C5]
BEST_EPOCH = int(hist["val_loss"].idxmin()) + 1
print(f"\nEPOCHS EXECUTED: {EPOCHS_EXECUTED}   BEST EPOCH: {BEST_EPOCH}")
print(f"TRAINING TIME: {train_seconds/60:.1f} min "
      f"({train_seconds/EPOCHS_EXECUTED:.1f} s/epoch)")

# Figs. 2 and 3 - accuracy and loss

ep = np.arange(1, EPOCHS_EXECUTED + 1)

fig, ax = plt.subplots(figsize=(IEEE_COL, 1.95))
ax.plot(ep, hist["accuracy"], label="Training", color="#2c6fbb")
ax.plot(ep, hist["val_accuracy"], label="Validation", color="#e08214")
ax.axvline(BEST_EPOCH, color="0.45", ls=":", lw=0.9)
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)
save(fig, os.path.join(OUTPUT_DIR, "02_accuracy_curve.png"),
          os.path.join(PAPER_DIR,  "fig02_base_accuracy.png"))

fig, ax = plt.subplots(figsize=(IEEE_COL, 1.95))
ax.plot(ep, hist["loss"], label="Training", color="#2c6fbb")
ax.plot(ep, hist["val_loss"], label="Validation", color="#e08214")
ax.set_yscale("log")
ax.set_xlabel("Epoch"); ax.set_ylabel("Categorical cross-entropy")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.3, which="both")
save(fig, os.path.join(OUTPUT_DIR, "03_loss_curve.png"),
          os.path.join(PAPER_DIR,  "fig03_base_loss.png"))

# 6. TEST EVALUATION  ([C1] inference timed)

test_gen.reset()
t0 = time.time()
probs = model.predict(test_gen, verbose=1)
infer_seconds = time.time() - t0
ms_per_image = 1000.0 * infer_seconds / len(test_gen.filenames)

y_true = test_gen.classes
y_pred = np.argmax(probs, axis=1)

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                               digits=4, zero_division=0)
print(report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                            output_dict=True, zero_division=0)

# [C2] per-class ROC-AUC and average precision
y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
per_class_auc, per_class_ap = {}, {}
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    per_class_auc[name] = float(auc(fpr, tpr))
    per_class_ap[name] = float(average_precision_score(y_bin[:, i], probs[:, i]))

metrics = {
    "data_root": DATA_ROOT, "tag": args.tag, "seed": SEED,
    "accuracy":        float(rep["accuracy"]),
    "macro_precision": float(rep["macro avg"]["precision"]),
    "macro_recall":    float(rep["macro avg"]["recall"]),
    "macro_f1":        float(rep["macro avg"]["f1-score"]),
    "weighted_f1":     float(rep["weighted avg"]["f1-score"]),
    "log_loss":        float(log_loss(y_true, probs,
                                      labels=list(range(NUM_CLASSES)))),
    "per_class_auc":   per_class_auc,                                  # [C2]
    "per_class_ap":    per_class_ap,                                   # [C2]
    "macro_auc":       float(np.mean(list(per_class_auc.values()))),
    "macro_ap":        float(np.mean(list(per_class_ap.values()))),
    "total_params":     total_params,                                  # [C1]
    "trainable_params": trainable_params,
    "epochs_executed":  EPOCHS_EXECUTED,                               # [C5]
    "best_epoch":       BEST_EPOCH,
    "train_seconds":       round(train_seconds, 1),                    # [C1]
    "seconds_per_epoch":   round(train_seconds / EPOCHS_EXECUTED, 2),
    "test_inference_seconds": round(infer_seconds, 2),
    "inference_ms_per_image": round(ms_per_image, 3),
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print(json.dumps(metrics, indent=2))

pd.DataFrame({
    "filename":  test_gen.filenames,
    "true_class":      [CLASS_NAMES[i] for i in y_true],
    "predicted_class": [CLASS_NAMES[i] for i in y_pred],
    "confidence":      probs.max(axis=1),
}).to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# Fig. 6 - confusion matrix

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(IEEE_COL, 2.55))
im = ax.imshow(cm, cmap="Blues")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=7.5,
                color="white" if cm[i, j] > cm.max() * 0.55 else "0.15",
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=20)
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.ax.tick_params(labelsize=6)
save(fig, os.path.join(OUTPUT_DIR, "05_confusion_matrix.png"),
          os.path.join(PAPER_DIR,  "fig06_base_confusion.png"))

# 7. GRAD-CAM  ([C3] no overlapping suptitle)

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

from tensorflow.keras.preprocessing.image import load_img, img_to_array
N_CAM = 2
idxs = np.linspace(0, len(test_gen.filenames) - 1, N_CAM).astype(int)

fig, axes = plt.subplots(N_CAM, 2, figsize=(IEEE_COL, 1.72 * N_CAM))
for row, k in enumerate(idxs):
    path = os.path.join(TEST_DIR, test_gen.filenames[k])
    raw = img_to_array(load_img(path, target_size=(IMG_SIZE, IMG_SIZE)))
    arr = preprocess_input(np.expand_dims(raw.copy(), 0))
    cam = make_gradcam(arr, int(y_pred[k]))

    axes[row, 0].imshow(raw.astype("uint8"))
    axes[row, 1].imshow(cam, cmap="jet")
    for c in (0, 1):
        axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
    if row == 0:
        axes[row, 0].set_title("Input radiograph", fontsize=7.5, pad=3)
        axes[row, 1].set_title("Baseline Grad-CAM", fontsize=7.5, pad=3)
    axes[row, 0].set_ylabel(f"Sample {row+1}", fontsize=7)
fig.subplots_adjust(wspace=0.05, hspace=0.06)   # no suptitle -> no overlap
save(fig, os.path.join(OUTPUT_DIR, "14_gradcam.png"),
          os.path.join(PAPER_DIR,  "fig13_base_gradcam.png"))

print("\nDONE. Outputs in", OUTPUT_DIR, "and", PAPER_DIR)
