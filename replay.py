# usage: python replay.py <normal file> <attack file> ...
import sys, time, joblib, pandas as pd
model, FEATS, TARGET = joblib.load("wids_label.joblib")   # multi-class model
df = pd.read_csv("test_windows.csv"); df[FEATS] = df[FEATS].fillna(0)
alerts = 0
for part in sys.argv[1:]:
    print(f"--- replaying {part} ---")
    d = df[df.part == part].sort_values("win")
    for win, g in d.groupby("win"):
        pred = model.predict(g[FEATS])
        flagged = pred != "normal"
        stamp = time.strftime("%H:%M:%S", time.localtime(win))
        if not flagged.any():
            print(f"[{stamp}] OK    {len(g)} devices, no alerts")
        for (_, r), kind in zip(g[flagged].iterrows(), pred[flagged]):
            msg = f"[{stamp}] ALERT {kind.upper()} attack from {r.src} (deauth={int(r.deauth)}, disassoc={int(r.disassoc)})"
            print(msg); open("alerts.log", "a").write(msg + "\n"); alerts += 1
        time.sleep(1)
print(f"Done. {alerts} alerts logged to alerts.log")
