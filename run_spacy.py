
import argparse
import json
import re
import statistics
import sys
import time

import psutil
import spacy


DEFAULT_MODEL = "en_core_web_trf"

LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "EVENT": "EVENT"
}

TITLE_PREFIXES = re.compile(
    r"^(Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?|Prof\.?|Rev\.?|Sir|Dame|Lord|Lady|Minister|Deputy|Senior|Chief|Director|Captain|Capt\.?|Col\.?|Gen\.?|Sgt\.?)\s+",
    re.IGNORECASE
)


def clean_entity(text, label):
    text = re.sub(r"'s?\s*$", "", text).strip()
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE).strip()

    if label == "PERSON":
        text = TITLE_PREFIXES.sub("", text).strip()

    return text


def predict_one(nlp, text):
    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        label = LABEL_MAP.get(ent.label_)

        if label is None:
            continue

        name = clean_entity(ent.text, label)

        if name:
            key = (name.lower(), label)

            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": name,
                    "label": label
                })

    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    snippets = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                snippets.append(json.loads(line))

    print(f"Loaded {len(snippets)} snippets", file=sys.stderr)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    print(f"Loading model: {args.model}", file=sys.stderr)

    start = time.perf_counter()

    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Model {args.model} not found. Install it using:", file=sys.stderr)
        print(f"python -m spacy download {args.model}", file=sys.stderr)
        sys.exit(1)

    load_time = time.perf_counter() - start
    mem_after = process.memory_info().rss / 1024**2

    print(f"Model loaded in {load_time:.1f}s", file=sys.stderr)
    print(f"RAM used: {mem_after - mem_before:.0f} MB", file=sys.stderr)

    for snippet in snippets[:2]:
        predict_one(nlp, snippet["text"])

    print("Warm-up completed", file=sys.stderr)

    latencies = []
    predictions = []

    peak_mem = process.memory_info().rss / 1024**2

    for snippet in snippets:
        start = time.perf_counter()

        entities = predict_one(nlp, snippet["text"])

        end = time.perf_counter()
        latencies.append(end - start)

        current_mem = process.memory_info().rss / 1024**2
        peak_mem = max(peak_mem, current_mem)

        predictions.append({
            "id": snippet["id"],
            "entities": entities
        })

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

    print(f"\nspaCy Model: {args.model}", file=sys.stderr)
    print(f"Total snippets: {len(snippets)}", file=sys.stderr)
    print(f"Median time: {median_ms:.1f} ms", file=sys.stderr)
    print(f"P95 time: {p95_ms:.1f} ms", file=sys.stderr)
    print(f"Total time: {total_time:.2f} seconds", file=sys.stderr)
    print(f"Peak RAM: {peak_mem:.0f} MB", file=sys.stderr)
    print(f"Estimated cost per 1000 snippets: ${cost_per_1000:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
