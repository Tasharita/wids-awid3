import sys, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             ConfusionMatrixDisplay)

TARGET = sys.argv[1] if len(sys.argv) > 1 else "label_bin"   # or "label"
FEATS = ["frames","deauth","disassoc","beacon","probe_req","assoc","unprot_mgmt",
         "uniq_dst","bcast_frac","mean_len","std_len","rssi_mean","rssi_std"]
# Ablation run: delete "unprot_mgmt" from FEATS above and run again.

TEST_PARTS = ["data/Deauth/Deauth_23.csv", "data/Deauth/Deauth_31.csv",
              "data/Deauth/Deauth_14.csv", "data/Disass/Disas_30.csv",
              "data/Disass/Disas_40.csv", "data/Disass/Disas_24.csv"]

df = pd.read_csv("dataset.csv"); df[FEATS] = df[FEATS].fillna(0)
test = df[df.part.isin(TEST_PARTS)]; train = df[~df.part.isin(TEST_PARTS)]
test.to_csv("test_windows.csv", index=False)
print("TRAIN:\n", train[TARGET].value_counts().to_string())
print("TEST:\n", test[TARGET].value_counts().to_string())

models = {
  "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=0),
  "Random Forest": RandomForestClassifier(200, class_weight="balanced", n_jobs=-1, random_state=0),
  "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
  "Naive Bayes": GaussianNB(),
}
results = []
def report(name, pred):
    acc = accuracy_score(test[TARGET], pred)
    p, r, f, _ = precision_recall_fscore_support(test[TARGET], pred, average=("binary" if TARGET=="label_bin" else "macro"), zero_division=0)
    results.append((name, acc, p, r, f))
    print(f"{name}: acc={acc:.4f} precision={p:.3f} recall={r:.3f} f1={f:.3f}")
    ConfusionMatrixDisplay.from_predictions(test[TARGET], pred, cmap="Blues")
    plt.title(name); plt.tight_layout()
    plt.savefig(f"cm_{TARGET}_{name.replace(' ','_').replace(':','')}.png", dpi=200); plt.close()

for name, m in models.items():
    m.fit(train[FEATS], train[TARGET]); report(name, m.predict(test[FEATS]))

if TARGET == "label_bin":
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, contamination=0.02, random_state=0))
    iso.fit(train.loc[train.label_bin == 0, FEATS])            # normal traffic only
    report("Isolation Forest", (iso.predict(test[FEATS]) == -1).astype(int))
    report("Rule deauth+disassoc>=10 per s", ((test.deauth + test.disassoc) >= 10).astype(int))

rf = models["Random Forest"]
pd.Series(rf.feature_importances_, FEATS).sort_values().plot.barh()
plt.tight_layout(); plt.savefig(f"importance_{TARGET}.png", dpi=200); plt.close()
print(export_text(models["Decision Tree"], feature_names=FEATS))
pd.DataFrame(results, columns=["model","accuracy","precision","recall","f1"]).to_csv(f"results_{TARGET}.csv", index=False)
joblib.dump((rf, FEATS, TARGET), f"wids_{TARGET}.joblib")
