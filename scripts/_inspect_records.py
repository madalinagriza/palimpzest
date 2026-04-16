import json

# Show one record from each template with its PII fields
seen_templates = set()
with open("data/resumes_with_pii.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r["template_used"]
        if t not in seen_templates:
            seen_templates.add(t)
            print(f"--- {r['pii_group']} / {t} / {r['record_id']} ---")
            for k in ["name","email","phone","address","linkedin","ssn","dob"]:
                print(f"  {k}: {r.get(k)}")
            print(f"  text[:300]: {r['text'][:300]}")
            print()

# Also show raw text from resumes_clean.jsonl for same record_ids
print("\n=== CLEAN RAW TEXT SAMPLES ===")
target_ids = set()
seen_templates2 = set()
with open("data/resumes_with_pii.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r["template_used"]
        if t not in seen_templates2:
            seen_templates2.add(t)
            target_ids.add(r["record_id"])

with open("data/resumes_clean.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["record_id"] in target_ids:
            print(f"--- {r['record_id']} ---")
            print(f"  text[:400]: {r['text'][:400]}")
            print()
