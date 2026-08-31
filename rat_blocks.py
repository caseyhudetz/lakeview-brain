"""East Lakeview rodent complaints: persistent blocks vs one-off spikes. Throwaway.

Runs the whole thing in one shot:
  Step 1  prints the recon (real sr_type values, real field names, null counts,
          duplicate-flag usage) so the filters are verified at runtime, not guessed
  Step 2  pulls 2021-01-01 to present, analyzed both with duplicates in and collapsed
  Step 3  buckets to 100-blocks, reports what it had to drop
  Step 4  heatmap PNG, top 40 blocks by volume, months as columns
  Step 5  PERSISTENCE and SPIKINESS ranked side by side, plus how much each is
          just volume wearing a hat

Usage:
    python3 rat_blocks.py                  # pull from Socrata
    python3 rat_blocks.py --csv rats.csv   # or work from a local extract
    python3 rat_blocks.py --dupes          # chart the duplicates-included cut instead
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET = "v6vf-nfxy"          # 311 Service Requests, Dec 2018 onward
BASE = f"https://data.cityofchicago.org/resource/{DATASET}.json"
START = "2021-01-01"
COMMUNITY_AREA = "6"           # Lake View
PAGE = 50000

# ---------------------------------------------------------------------------
# PLACEHOLDER GEOGRAPHY. "East Lakeview" has no official boundary and this repo
# had no existing filter to reuse. This box is Diversey -> Irving Park, Halsted
# -> the lake. If the Rats of Lakeview map draws it differently, change this --
# it will change which blocks rank.
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 41.9320, 41.9560
LNG_MIN, LNG_MAX = -87.6520, -87.6200

TOP_N_CHART = 40
TOP_N_LIST = 15


# --------------------------------------------------------------------- fetch
def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def soql(**params):
    p = {f"${k}": v for k, v in params.items() if v is not None}
    return _get(BASE + "?" + urllib.parse.urlencode(p))


def rodent_types():
    """Step 1a: ask the data which sr_type values exist, don't guess a string."""
    rows = soql(
        select="sr_type, count(1) as n",
        where=(f"created_date >= '{START}T00:00:00' AND "
               "(upper(sr_type) like '%RODENT%' OR upper(sr_type) like '%RAT%')"),
        group="sr_type", order="n DESC", limit=100)
    print("=" * 72)
    print("STEP 1a  sr_type values matching rodent/rat (citywide, since " + START + ")")
    print("=" * 72)
    for r in rows:
        print(f"  {int(r['n']):>9,}  {r['sr_type']}")
    if not rows:
        sys.exit("  none matched -- the type name changed, widen the LIKE")
    return [r["sr_type"] for r in rows]


def fetch_all(types):
    clause = " OR ".join("sr_type = '%s'" % t.replace("'", "''") for t in types)
    where = (f"created_date >= '{START}T00:00:00' AND "
             f"community_area = '{COMMUNITY_AREA}' AND ({clause})")
    frames, offset = [], 0
    while True:
        chunk = soql(where=where, limit=PAGE, offset=offset)
        frames.extend(chunk)
        if len(chunk) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(frames)


# ------------------------------------------------------------------- columns
def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def recon(df):
    """Step 1b-d: real field names, null rates, duplicate flagging."""
    print("\n" + "=" * 72)
    print(f"STEP 1b  field names present ({len(df):,} rows, Lake View, since {START})")
    print("=" * 72)
    print("  " + ", ".join(sorted(df.columns)))

    cols = {
        "street number": pick(df, "street_number", "street_num", "address_street_number"),
        "direction":     pick(df, "street_direction", "street_dir", "address_street_direction"),
        "street name":   pick(df, "street_name", "address_street_name"),
        "street type":   pick(df, "street_type", "street_suffix", "address_street_suffix"),
        "latitude":      pick(df, "latitude", "lat"),
        "longitude":     pick(df, "longitude", "lon", "lng"),
    }
    print("\n  resolved:")
    for label, c in cols.items():
        print(f"    {label:<15} -> {c if c else 'NOT FOUND'}")

    print("\n" + "=" * 72)
    print("STEP 1c  null counts")
    print("=" * 72)
    n = len(df)
    for label, c in cols.items():
        if not c:
            continue
        miss = int(df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum())
        print(f"  {label:<15} ({c:<18}) null/blank {miss:>7,}  ({miss / n:.1%})")
    both_geo = [c for c in (cols["latitude"], cols["longitude"]) if c]
    if both_geo:
        nogeo = int(df[both_geo].isna().any(axis=1).sum())
        print(f"  {'no lat AND/OR lng':<15} {'':<20} {nogeo:>18,}  ({nogeo / n:.1%})")

    print("\n" + "=" * 72)
    print("STEP 1d  duplicate flagging")
    print("=" * 72)
    dup_c = pick(df, "duplicate", "is_duplicate", "duplicate_of")
    par_c = pick(df, "parent_sr_number", "parent_request_number", "parent_sr")
    if dup_c:
        flags = df[dup_c].astype(str).str.lower().isin(["true", "1", "y", "yes"])
        print(f"  {dup_c:<22} true on {int(flags.sum()):,} of {n:,} ({flags.mean():.1%})")
    else:
        print("  no boolean duplicate field found")
    if par_c:
        has = df[par_c].notna() & (df[par_c].astype(str).str.strip() != "")
        print(f"  {par_c:<22} set on {int(has.sum()):,} of {n:,} ({has.mean():.1%})")
    else:
        print("  no parent-request field found")
    if not dup_c and not par_c:
        print("  -> will fall back to collapsing same-block-same-day repeats")
    return cols, dup_c, par_c


# -------------------------------------------------------------- transform
def geo_filter(df, cols):
    lat_c, lng_c = cols["latitude"], cols["longitude"]
    if not lat_c or not lng_c:
        print("\n  !! no coordinate columns; keeping all of Lake View (too wide)")
        return df, 0, 0
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lng = pd.to_numeric(df[lng_c], errors="coerce")
    nogeo = int(lat.isna().sum() | 0) + 0
    nogeo = int((lat.isna() | lng.isna()).sum())
    keep = (lat.between(LAT_MIN, LAT_MAX) & lng.between(LNG_MIN, LNG_MAX))
    outside = int((~keep & ~(lat.isna() | lng.isna())).sum())
    return df[keep].copy(), nogeo, outside


def to_blocks(df, cols):
    """Step 3: 1247 N BROADWAY -> '1200 N BROADWAY'."""
    num = pd.to_numeric(df[cols["street number"]], errors="coerce")
    ok_num = num.notna() & (num > 0)

    def clean(c):
        if not c:
            return pd.Series([""] * len(df), index=df.index)
        return df[c].fillna("").astype(str).str.strip().str.upper()

    direction, name, stype = clean(cols["direction"]), clean(cols["street name"]), clean(cols["street type"])
    ok = ok_num & (name != "")
    hundred = (num // 100 * 100)

    dropped = int((~ok).sum())
    out = df.loc[ok].copy()
    h = hundred.loc[ok].astype(np.int64).astype(str)
    parts = pd.DataFrame({"h": h, "d": direction.loc[ok],
                          "n": name.loc[ok], "t": stype.loc[ok]}, index=out.index)
    out["block"] = [" ".join(p for p in row if p) for row in parts.itertuples(index=False)]
    return out, dropped


def collapse(df, dup_c, par_c):
    """Drop rows the city itself marks as duplicates of another request."""
    mask = pd.Series(False, index=df.index)
    if dup_c:
        mask |= df[dup_c].astype(str).str.lower().isin(["true", "1", "y", "yes"])
    if par_c:
        mask |= df[par_c].notna() & (df[par_c].astype(str).str.strip() != "")
    if not dup_c and not par_c:
        keep = ~df.duplicated(subset=["block", "month"], keep="first")
        return df[keep].copy(), int((~keep).sum())
    return df[~mask].copy(), int(mask.sum())


# --------------------------------------------------------------- analysis
def matrix(df):
    m = (df.groupby(["block", "month"]).size().unstack(fill_value=0))
    full = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    m = m.reindex(columns=full, fill_value=0)
    return m.loc[m.sum(axis=1).sort_values(ascending=False).index]


def scores(m):
    total = m.sum(axis=1)
    return pd.DataFrame({
        "total": total,
        "persistence": (m > 0).sum(axis=1) / m.shape[1],
        "spikiness": m.max(axis=1) / total.replace(0, np.nan),
        "peak_month": m.idxmax(axis=1).astype(str),
    })


def spearman(a, b):
    """Rank correlation without pulling in scipy."""
    a, b = a.rank(), b.rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def report(label, m, s):
    print("\n" + "=" * 72)
    print(f"STEP 5  {label}   ({m.shape[0]:,} blocks x {m.shape[1]} months, "
          f"{int(m.values.sum()):,} complaints)")
    print("=" * 72)
    pers = s.sort_values(["persistence", "total"], ascending=False).head(TOP_N_LIST)
    spik = s[s["total"] >= 5].sort_values(["spikiness", "total"], ascending=False).head(TOP_N_LIST)

    print(f"\n{'PERSISTENCE (share of months w/ >=1)':<44}   {'SPIKINESS (biggest month / total)':<44}")
    print(f"{'#':<3}{'block':<26}{'mo%':>6}{'tot':>6}   {'#':<3}{'block':<26}{'max%':>6}{'tot':>6}")
    print("-" * 94)
    for i in range(TOP_N_LIST):
        L = R = " " * 41
        if i < len(pers):
            r = pers.iloc[i]
            L = f"{i+1:<3}{pers.index[i][:25]:<26}{r['persistence']:>5.0%}{int(r['total']):>6}"
        if i < len(spik):
            r = spik.iloc[i]
            R = f"{i+1:<3}{spik.index[i][:25]:<26}{r['spikiness']:>5.0%}{int(r['total']):>6}"
        print(f"{L}   {R}")

    print("\n  -- are these just volume in disguise? --")
    sub = s[s["total"] >= 5]
    print(f"  spearman(persistence, total) = {spearman(sub['persistence'], sub['total']):+.2f}"
          "   (near +1.0 => persistence is a volume proxy)")
    print(f"  spearman(spikiness,   total) = {spearman(sub['spikiness'], sub['total']):+.2f}"
          "   (near -1.0 => spikiness is just 'low volume')")
    print(f"  spearman(persistence, spikiness) = "
          f"{spearman(sub['persistence'], sub['spikiness']):+.2f}")
    overlap = set(pers.index) & set(spik.index)
    print(f"  blocks in BOTH top-{TOP_N_LIST} lists: {len(overlap)}"
          + (f"  {sorted(overlap)}" if overlap else "  (fully disjoint)"))

    structural = pers[pers.index.isin(s[s["spikiness"] <= s["spikiness"].median()].index)]
    if len(structural):
        print(f"\n  STRUCTURAL (high persistence, below-median spikiness):")
        for b, r in structural.head(8).iterrows():
            print(f"    {b:<30} {r['persistence']:>4.0%} of months, {int(r['total']):>4} total")
    event = spik[spik.index.isin(s[s["persistence"] <= s["persistence"].median()].index)]
    if len(event):
        print(f"\n  EVENT (high spikiness, below-median persistence):")
        for b, r in event.head(8).iterrows():
            print(f"    {b:<30} {r['spikiness']:>4.0%} in {r['peak_month']}, {int(r['total']):>4} total")
    return pers, spik


def heatmap(m, label, path):
    top = m.head(TOP_N_CHART)
    fig, ax = plt.subplots(figsize=(max(10, m.shape[1] * 0.22), 11))
    # Rat complaints are heavily skewed -- one spike month otherwise flattens
    # every other cell to white. Clip the scale at the 98th percentile of
    # non-zero cells so the mid-range stays readable; spikes just top out.
    nz = top.values[top.values > 0]
    vmax = float(np.percentile(nz, 98)) if nz.size else 1.0
    vmax = max(vmax, 1.0)
    clipped = int((top.values > vmax).sum())
    im = ax.imshow(top.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{b}  ({int(t)})" for b, t in zip(top.index, top.sum(axis=1))], fontsize=8)
    labels = [str(p) for p in top.columns]
    step = max(1, len(labels) // 30)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=90, fontsize=7)
    ax.set_xlabel("month")
    ax.set_title(f"East Lakeview rodent complaints by 100-block and month\n"
                 f"top {len(top)} blocks by volume, {label}", fontsize=11)
    cb_label = ("complaints" if not clipped
                else f"complaints (scale clipped at {vmax:.0f}; "
                     f"{clipped} cell(s) above)")
    fig.colorbar(im, ax=ax, label=cb_label, extend="max" if clipped else "neither",
                 fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\n  wrote {path}")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="local extract instead of the Socrata pull")
    ap.add_argument("--dupes", action="store_true",
                    help="chart the duplicates-included cut (default: collapsed)")
    ap.add_argument("--out", default="rat_blocks.png")
    ap.add_argument("--merge-street-type", action="store_true",
                    help="fold 'N CLARK ST'/'N CLARK AVE'/'N CLARK' into one block")
    a = ap.parse_args()

    if a.csv:
        df = pd.read_csv(a.csv, dtype=str, low_memory=False)
        print(f"loaded {len(df):,} rows from {a.csv}")
    else:
        df = fetch_all(rodent_types())
        print(f"\npulled {len(df):,} rows")
    if df.empty:
        sys.exit("no rows")

    cols, dup_c, par_c = recon(df)

    date_c = pick(df, "created_date", "creation_date", "requested_datetime")
    df["month"] = pd.to_datetime(df[date_c], errors="coerce").dt.to_period("M")
    nodate = int(df["month"].isna().sum())
    df = df[df["month"].notna()]

    df, nogeo, outside = geo_filter(df, cols)
    df, dropped = to_blocks(df, cols)
    if a.merge_street_type:
        df["block"] = df["block"].str.replace(
            r"\s+(ST|AVE|RD|BLVD|DR|PL|PKWY|CT|TER|LN|WAY)$", "", regex=True)
        print("  (street_type folded into the block key)")

    print("\n" + "=" * 72)
    print("STEPS 2-3  what got dropped")
    print("=" * 72)
    print(f"  unparseable created_date          {nodate:>7,}")
    print(f"  null lat/lng (can't place)        {nogeo:>7,}")
    print(f"  outside the East Lakeview box     {outside:>7,}")
    print(f"  no usable street number/name      {dropped:>7,}")
    print(f"  -> {len(df):,} rows across {df['block'].nunique():,} blocks")

    stem = df["block"].str.replace(r"\s+(ST|AVE|RD|BLVD|DR|PL|PKWY|CT|TER|LN|WAY)$", "",
                                   regex=True)
    frag = (df.assign(stem=stem).groupby("stem")["block"].nunique())
    frag = frag[frag > 1]
    if len(frag):
        print(f"\n  !! {len(frag)} block(s) fragmented by inconsistent street_type -- "
              "same block spelled several ways:")
        for st in frag.head(8).index:
            variants = sorted(df.loc[stem == st, "block"].unique())
            print(f"     {st:<26} -> {variants}")
        print("     these split one real block's counts across rows and will DEFLATE its")
        print("     persistence score. Re-run with --merge-street-type to fold them together.")

    kept, n_dup = collapse(df, dup_c, par_c)
    print(f"  duplicates collapsed away          {n_dup:>7,}  ({n_dup / max(len(df),1):.1%})")

    m_all, m_col = matrix(df), matrix(kept)
    s_all, s_col = scores(m_all), scores(m_col)
    p_all, k_all = report("DUPLICATES INCLUDED", m_all, s_all)
    p_col, k_col = report("DUPLICATES COLLAPSED", m_col, s_col)

    print("\n" + "=" * 72)
    print("DOES COLLAPSING DUPLICATES CHANGE THE ANSWER?")
    print("=" * 72)
    for nm, x, y in (("persistence", p_all, p_col), ("spikiness", k_all, k_col)):
        same = len(set(x.index) & set(y.index))
        print(f"  top-{TOP_N_LIST} {nm:<12} {same}/{TOP_N_LIST} blocks in common"
              f"{'  -- SAME STORY' if same >= TOP_N_LIST - 2 else '  -- RANKING MOVES, use collapsed'}")
        moved = [b for b in x.index if b not in y.index]
        if moved:
            print(f"      dropped out when collapsed: {moved}")

    heatmap(*( (m_all, "duplicates included") if a.dupes else (m_col, "duplicates collapsed") ), a.out)


if __name__ == "__main__":
    main()
