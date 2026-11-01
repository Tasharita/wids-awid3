import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
FEATS = ["frames","deauth","disassoc","beacon","probe_req","assoc","unprot_mgmt",
         "uniq_dst","bcast_frac","mean_len","std_len","rssi_mean","rssi_std"]
df = pd.read_csv("dataset.csv"); df[FEATS] = df[FEATS].fillna(0)
rf = RandomForestClassifier(200, class_weight="balanced", random_state=0, n_jobs=-1)
pred = cross_val_predict(rf, df[FEATS], df.label_bin, groups=df.part, cv=GroupKFold(n_splits=5))
print(classification_report(df.label_bin, pred, target_names=["normal","attack"], digits=3))
print(confusion_matrix(df.label_bin, pred))
