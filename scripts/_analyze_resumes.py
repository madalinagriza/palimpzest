import json

pii_fields = ["name", "email", "phone", "address", "linkedin", "education", "experience", "skills"]

records = []
with open("data/resumes_clean.jsonl", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total = len(records)
no_pii = 0
counts_by_source = {"huggingface": {"no_pii": 0, "total": 0}, "github": {"no_pii": 0, "total": 0}}
text_len_buckets = {"<100": 0, "100-500": 0, "500-2000": 0, "2000-5000": 0, ">5000": 0}
pii_only_short_text = 0
raw_text_no_structure = 0
long_text_with_pii = 0
pii_count_dist = {}

for rec in records:
    src = rec["source"]
    counts_by_source[src]["total"] += 1

    has_any_pii = any(rec.get(f) for f in pii_fields)
    if not has_any_pii:
        no_pii += 1
        counts_by_source[src]["no_pii"] += 1

    tlen = len(rec.get("text", ""))
    if tlen < 100:
        text_len_buckets["<100"] += 1
    elif tlen < 500:
        text_len_buckets["100-500"] += 1
    elif tlen < 2000:
        text_len_buckets["500-2000"] += 1
    elif tlen < 5000:
        text_len_buckets["2000-5000"] += 1
    else:
        text_len_buckets[">5000"] += 1

    if tlen < 300 and has_any_pii:
        pii_only_short_text += 1
    elif tlen > 500 and not has_any_pii:
        raw_text_no_structure += 1
    elif tlen > 500 and has_any_pii:
        long_text_with_pii += 1

    n = sum(1 for f in pii_fields if rec.get(f))
    pii_count_dist[n] = pii_count_dist.get(n, 0) + 1

print(f"=== TOTAL RECORDS: {total} ===")
print()
print("--- Records with ZERO extracted PII fields ---")
print(f"  Total: {no_pii} ({100*no_pii/total:.1f}%)")
for src in ["huggingface", "github"]:
    d = counts_by_source[src]
    pct = 100 * d["no_pii"] / max(d["total"], 1)
    print(f"  {src}: {d['no_pii']}/{d['total']} ({pct:.1f}%)")

print()
print("--- PII field count distribution (how many of 8 fields populated) ---")
for n in sorted(pii_count_dist):
    c = pii_count_dist[n]
    print(f"  {n} fields: {c:>6} records ({100*c/total:.1f}%)")

print()
print("--- Text length distribution ---")
for bucket, c in text_len_buckets.items():
    print(f"  {bucket:>10} chars: {c:>6} records ({100*c/total:.1f}%)")

print()
print("--- Raw text vs structured PII ---")
print(f"  Short text (<300) + has PII  ('mostly PII fields'): {pii_only_short_text:>6} ({100*pii_only_short_text/total:.1f}%)")
print(f"  Long text  (>500) + zero PII ('raw text only'):      {raw_text_no_structure:>6} ({100*raw_text_no_structure/total:.1f}%)")
print(f"  Long text  (>500) + has PII  ('text + structure'):   {long_text_with_pii:>6} ({100*long_text_with_pii/total:.1f}%)")
other = total - pii_only_short_text - raw_text_no_structure - long_text_with_pii
print(f"  Other (medium length, mixed):                        {other:>6} ({100*other/total:.1f}%)")
