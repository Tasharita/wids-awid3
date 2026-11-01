import glob, numpy as np, pandas as pd

COL = dict(time="frame.time_epoch", src="wlan.ta", dst="wlan.da",
           ftype="wlan.fc.type", subtype="wlan.fc.subtype",
           rssi="wlan_radio.signal_dbm", length="frame.len",
           protected="wlan.fc.protected", label="Label")
FOLDERS = ["Deauth", "Disass"]          # later add "Kr00k", "Normal" if you get them
NORMAL_NAMES = {"normal"}

def num(s):
    return pd.to_numeric(s, errors="coerce")

def first_rssi(s):                       # '-57-69-57' -> -57 (first antenna)
    return pd.to_numeric(s.astype(str).str.extract(r"^(-?\d+)")[0], errors="coerce")

def prep(c):
    d = pd.DataFrame()
    d["t"] = num(c[COL["time"]]);  d["win"] = np.floor(d["t"])
    typ = num(c[COL["ftype"]]);    sub = num(c[COL["subtype"]])
    mgmt = typ.eq(0)
    d["is_deauth"] = mgmt & sub.eq(12);  d["is_disassoc"] = mgmt & sub.eq(10)
    d["is_beacon"] = mgmt & sub.eq(8);   d["is_probe"] = mgmt & sub.eq(4)
    d["is_assoc"] = mgmt & sub.isin([0, 2])
    d["unprot_mgmt"] = mgmt & num(c[COL["protected"]]).eq(0)
    d["src"] = c[COL["src"]];  d["dst"] = c[COL["dst"]]
    d["is_bcast"] = d["dst"].eq("ff:ff:ff:ff:ff:ff")
    d["len"] = num(c[COL["length"]]);  d["rssi"] = first_rssi(c[COL["rssi"]])
    lab = c[COL["label"]].astype(str).str.strip().str.lower()
    d["atk"] = np.where(lab.isin(NORMAL_NAMES), "", lab)
    return d.dropna(subset=["t", "src"])

def windows(d):
    g = d.groupby(["win", "src"])
    out = g.agg(frames=("t", "size"), deauth=("is_deauth", "sum"),
                disassoc=("is_disassoc", "sum"), beacon=("is_beacon", "sum"),
                probe_req=("is_probe", "sum"), assoc=("is_assoc", "sum"),
                unprot_mgmt=("unprot_mgmt", "sum"), uniq_dst=("dst", "nunique"),
                bcast_frac=("is_bcast", "mean"), mean_len=("len", "mean"),
                std_len=("len", "std"), rssi_mean=("rssi", "mean"),
                rssi_std=("rssi", "std"))
    lab = d[d["atk"] != ""].groupby(["win", "src"])["atk"].first().rename("label")
    out = out.join(lab).reset_index()
    out["label"] = out["label"].fillna("normal")
    return out

def process_file(path):
    carry, parts = None, []
    for ch in pd.read_csv(path, usecols=list(COL.values()),
                          chunksize=500_000, low_memory=False):
        ch = prep(ch)
        if carry is not None: ch = pd.concat([carry, ch])
        last = ch["win"].max()
        carry = ch[ch["win"] == last]
        parts.append(windows(ch[ch["win"] < last]))
    if carry is not None: parts.append(windows(carry))
    return pd.concat(parts)

frames = []
for folder in FOLDERS:
    for path in sorted(glob.glob(f"data/{folder}/*.csv")):
        w = process_file(path); w["file"] = folder; w["part"] = path
        frames.append(w)
        print(path, len(w), "windows,", int(w.win.max() - w.win.min()), "s long,",
              int((w.label != "normal").sum()), "attack windows")
ds = pd.concat(frames, ignore_index=True)
ds["label_bin"] = (ds["label"] != "normal").astype(int)
ds.to_csv("dataset.csv", index=False)
print(ds["label"].value_counts())
