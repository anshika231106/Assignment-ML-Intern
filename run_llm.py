
import argparse
import json
import re
import statistics
import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
WARMUP_RUNS = 2

VALID_LABELS = {"PERSON", "ORG", "LOCATION", "EVENT"}

SYSTEM_PROMPT = """
You are a named entity recognition (NER) system. Your job is to extract named entities from text.

Extract ONLY entities that belong to these four labels:
- PERSON: Names of individual people. Strip titles and honorifics — tag "Daniel Koh" not "Minister Daniel Koh", tag "Marcus Teo" not "Dr. Marcus Teo". Tag only given names when only a first name is used (e.g. "Rajiv").
- ORG: Companies, agencies, government bodies, institutions, schools, sports clubs/teams, media outlets. This includes things like "Ministry of Transport", "Raffles Institution", "Dinamo Riga".
- LOCATION: Countries, cities, districts, rivers, mountains, landmarks, venues, buildings. Tag "Singapore" not "Singapore's".
- EVENT: Named events — festivals, tournaments, summits, holidays, storms. Examples: "Lantern Festival", "Asia Cup", "Typhoon Meilan", "Hari Raya", "ASEAN Summit".

Rules:
- Tag the LONGEST matching span only. Do not tag a sub-part if a longer entity covers it.
- Nationalities and demonyms alone are NOT tagged.
- Titles of films, books, or songs are NOT tagged.
- Dates, numbers, and job titles alone are NOT tagged.
- Strip possessives.
- Deduplicate repeated entities.
- If there are NO entities, return an empty array: []

Return ONLY a JSON array with objects having "text" and "label" keys. No explanation, markdown, or code fences.

Example:
[{"text": "Priya Nair", "label": "PERSON"}, {"text": "Gardens by the Bay", "label": "LOCATION"}, {"text": "Lantern Festival", "label": "EVENT"}]
"""

TITLE_PREFIXES = re.compile(
    r"^(Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?|Prof\.?|Rev\.?|Sir|Dame|Lord|Lady|Minister|Deputy|Senior|Chief|Director|Captain|Capt\.?|Col\.?|Gen\.?|Sgt\.?|Chef|Coach|Brother|Sister|Father|Mother|Councillor|Ambassador)\s+",
    re.IGNORECASE
)


def clean_entity(text, label):
    text = re.sub(r"'s?\s*$", "", text).strip()
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE).strip()

    if label == "PERSON":
        text = TITLE_PREFIXES.sub("", text).strip()

    return text


def parse_entities(raw, snippet_id):
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    start = raw.find("[")
    end = raw.rfind("]")

    if start == -1 or end == -1 or end <= start:
        print(f"Warning: No JSON array found for {snippet_id}", file=sys.stderr)
        return []

    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON for {snippet_id}", file=sys.stderr)
        return []

    if not isinstance(data, list):
        print(f"Warning: Expected a list for {snippet_id}", file=sys.stderr)
        return []

    entities = []
    seen = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).strip().upper()

        if not text or label not in VALID_LABELS:
            continue

        text = clean_entity(text, label)

        if text:
            key = (text.lower(), label)

            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": text,
                    "label": label
                })

    return entities


def build_prompt(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Text: "{text}"'}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    snippets = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                snippets.append(json.loads(line))

    print(f"Loaded {len(snippets)} snippets", file=sys.stderr)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...", file=sys.stderr)
    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.eval()

    load_time = time.perf_counter() - start
    mem_after = process.memory_info().rss / 1024**2

    print(f"Model loaded in {load_time:.1f}s", file=sys.stderr)
    print(f"RAM used: {mem_after - mem_before:.0f} MB", file=sys.stderr)

    print("Warming up model...", file=sys.stderr)

    for snippet in snippets[:WARMUP_RUNS]:
        prompt = build_prompt(tokenizer, snippet["text"])
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

    print("Warm-up completed", file=sys.stderr)

    latencies = []
    predictions = []
    parse_failures = 0

    peak_mem = process.memory_info().rss / 1024**2

    for i, snippet in enumerate(snippets):
        sid = snippet["id"]
        prompt = build_prompt(tokenizer, snippet["text"])
        inputs = tokenizer(prompt, return_tensors="pt")

        input_len = inputs["input_ids"].shape[1]

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

        end = time.perf_counter()
        latencies.append(end - start)

        new_tokens = outputs[0][input_len:]
        raw = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        entities = parse_entities(raw, sid)

        if not entities and raw.strip() not in ("[]", "[ ]", ""):
            parse_failures += 1

        predictions.append({
            "id": sid,
            "entities": entities
        })

        current_mem = process.memory_info().rss / 1024**2
        peak_mem = max(peak_mem, current_mem)

        if (i + 1) % 5 == 0 or i + 1 == len(snippets):
            print(
                f"Processed {i + 1}/{len(snippets)} snippets",
                file=sys.stderr
            )

    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Predictions saved to {args.output}", file=sys.stderr)

    median_ms = statistics.median(latencies) * 1000
    p95_ms = sorted(latencies)[int(len(latencies) * 0.95)] * 1000
    total_time = sum(latencies)

    hourly_cost = 0.10
    snippets_per_hour = 3600 / (total_time / len(snippets))
    cost_per_1000 = (1000 / snippets_per_hour) * hourly_cost

    print("\nQwen2.5-1.5B-Instruct", file=sys.stderr)
    print(f"Total snippets: {len(snippets)}", file=sys.stderr)
    print(f"Parse failures: {parse_failures}", file=sys.stderr)
    print(f"Median time: {median_ms:.1f} ms", file=sys.stderr)
    print(f"P95 time: {p95_ms:.1f} ms", file=sys.stderr)
    print(f"Total time: {total_time:.2f} seconds", file=sys.stderr)
    print(f"Peak RAM: {peak_mem:.0f} MB", file=sys.stderr)
    print(f"Estimated cost per 1000 snippets: ${cost_per_1000:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
