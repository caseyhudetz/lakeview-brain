"""Step 1 recon for the East Lakeview rodent question. Throwaway.

Answers, against live data rather than guesses:
  1. which sr_type values actually match rodent/rat
  2. the real field names for street number / direction / name / type
  3. how many rows have null lat/lng or null address parts
  4. whether duplicate_of / parent_request_number exist, and how many rows use them

Run:  python3 explore_311.py
"""
import json
import sys
import urllib.parse
import urllib.request

DATASET = "v6vf-nfxy"  # Chicago 311 Service Requests, Dec 2018 onward
BASE = f"https://data.cityofchicago.org/resource/{DATASET}.json"
META = f"https://data.cityofchicago.org/api/views/{DATASET}.json"

# Lake View is community area 6. "East Lakeview" is a sub-area of it with no
# official boundary -- narrowing that is an open question, see the report.
AREA = "6"
SINCE = "2021-01-01T00:00:00"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def soql(**params):
    q = "&".join(f"${k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    return get(f"{BASE}?{q}")


def main():
    print("=" * 70)
    print("1. COLUMN INVENTORY")
    print("=" * 70)
    cols = get(META)["columns"]
    names = [c["fieldName"] for c in cols]
    for c in cols:
        print(f"  {c['fieldName']:<28} {c.get('dataTypeName','?'):<12} {c.get('name','')}")

    addressish = [n for n in names if any(
        k in n.lower() for k in ("street", "address", "block", "direction", "suffix"))]
    geoish = [n for n in names if any(
        k in n.lower() for k in ("lat", "lon", "location", "x_coord", "y_coord"))]
    dupish = [n for n in names if any(
        k in n.lower() for k in ("duplicate", "parent", "sr_number"))]
    print(f"\n  address-ish fields : {addressish}")
    print(f"  geo-ish fields     : {geoish}")
    print(f"  duplicate-ish      : {dupish}")

    print("\n" + "=" * 70)
    print("2. sr_type VALUES MATCHING rodent/rat")
    print("=" * 70)
    rows = soql(
        select="sr_type, count(1) as n",
        where=f"created_date >= '{SINCE}' AND "
              "(upper(sr_type) like '%RODENT%' OR upper(sr_type) like '%RAT%')",
        group="sr_type", order="n DESC", limit=100)
    for r in rows:
        print(f"  {int(r['n']):>8,}  {r['sr_type']}")
    if not rows:
        print("  (none -- widen the LIKE, the type name has changed)")

    types = [r["sr_type"] for r in rows]
    if not types:
        sys.exit("no matching sr_type; stop here")
    type_clause = " OR ".join(f"sr_type = '{t}'" for t in types)
    scope = (f"created_date >= '{SINCE}' AND community_area = '{AREA}' "
             f"AND ({type_clause})")

    print("\n" + "=" * 70)
    print(f"3. COMPLETENESS  (community_area {AREA}, {SINCE[:10]} onward)")
    print("=" * 70)
    total = int(soql(select="count(1) as n", where=scope)[0]["n"])
    print(f"  total rows: {total:,}")

    for f in sorted(set(addressish + geoish)):
        try:
            n = int(soql(select="count(1) as n",
                         where=f"{scope} AND {f} IS NULL")[0]["n"])
        except Exception as e:
            print(f"  {f:<28} (query failed: {e})")
            continue
        print(f"  null {f:<24} {n:>8,}  ({n / total:.1%})" if total else f"  null {f}: {n}")

    print("\n" + "=" * 70)
    print("4. DUPLICATE FLAGGING")
    print("=" * 70)
    if not dupish:
        print("  no duplicate/parent-style field present")
    for f in dupish:
        try:
            nn = int(soql(select="count(1) as n",
                          where=f"{scope} AND {f} IS NOT NULL")[0]["n"])
        except Exception as e:
            print(f"  {f:<28} (query failed: {e})")
            continue
        print(f"  {f:<28} non-null on {nn:,} of {total:,} rows ({nn / total:.1%})"
              if total else f"  {f}: {nn}")

    print("\n  sample rows:")
    for r in soql(where=scope, limit=3):
        print("   ", json.dumps(r, indent=6)[:1200])


if __name__ == "__main__":
    main()
