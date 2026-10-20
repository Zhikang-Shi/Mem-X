"""Lightweight room-type evidence derived from contained objects."""

from __future__ import annotations

from collections import Counter

ROOM_TYPE_RULES = {
    "kitchen": {"refrigerator": 4, "microwave": 3.5, "stove": 4, "dishwasher": 3.5, "kettle": 2.5, "coffee machine": 2.5, "sink": 1, "cabinet": 1.2, "counter": 2},
    "bathroom": {"toilet": 5, "bathtub": 4.5, "shower": 4.5, "towel": 2.8, "soap dispenser": 3, "mirror": 1.6, "sink": 2},
    "bedroom": {"bed": 5, "pillow": 3.5, "blanket": 2.5, "nightstand": 4, "wardrobe": 2.5, "dresser": 3},
    "living room": {"sofa": 4, "tv": 2.8, "coffee table": 3, "fireplace": 3, "armchair": 2, "speaker": 1.6, "ottoman": 2, "piano": 2},
    "dining room": {"dining table": 4.5, "dining chair": 3.5, "table": 1.5, "chair": 0.8, "plate": 1.2},
    "office": {"desk": 4, "office chair": 3.5, "monitor": 3, "computer": 3, "laptop": 3, "printer": 3},
    "laundry room": {"washing machine": 5, "washer": 5, "dryer": 4.5, "laundry basket": 3.5, "iron": 2.5},
    "closet": {"wardrobe": 3, "clothes": 3, "shoe": 2, "bag": 1.5, "box": 1},
    "entryway": {"umbrella": 2.5, "shoe": 1.8, "bag": 1.5, "coat": 2, "rug": 1.2},
    "garage": {"bicycle": 3, "tool": 3, "box": 1.5, "luggage": 1},
}


def infer_room(labels: list[str]) -> tuple[str, float, list[str], dict[str, float]]:
    counts = Counter(labels)
    scores, evidence = {}, {}
    for room, rule in ROOM_TYPE_RULES.items():
        matched = [term for term in rule if counts[term]]
        score = sum(rule[term] * (1 + min(counts[term] - 1, 3) * 0.15) for term in matched)
        if score:
            scores[room], evidence[room] = round(score, 3), matched
    if not scores:
        return "unknown", 0.0, [], {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best, best_score = ranked[0]
    confidence = best_score / sum(scores.values())
    margin = best_score - (ranked[1][1] if len(ranked) > 1 else 0)
    if best_score < 2 or (confidence < 0.38 and margin < 1):
        return "unknown", round(confidence, 3), evidence[best], scores
    return best, round(confidence, 3), evidence[best], scores
