"""Rule extraction and canonical vocabulary for retrieval queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

DEFAULT_ROOM_ALIASES = {
    "lounge": "living room", "sitting room": "living room", "livingroom": "living room",
    "restroom": "bathroom", "washroom": "bathroom", "bed room": "bedroom",
    "dining area": "dining room", "kitchen area": "kitchen", "study": "office",
    "workspace": "office", "hall": "hallway", "corridor": "hallway",
}

DEFAULT_OBJECT_ALIASES = {
    "bath": "bathtub", "bathroom cabinet": "cabinet", "couch": "sofa",
    "dining chair": "chair", "dining table": "table", "flowerpot": "pot",
    "kitchen cabinet": "cabinet", "kitchen sink": "sink", "kitchen table": "table",
    "television": "tv", "tv monitor": "tv", "fridge": "refrigerator",
    "coffee maker": "coffee machine", "countertop": "counter", "worktop": "counter",
    "washbasin": "sink", "bookshelf": "shelf", "doormat": "rug", "door mat": "rug",
    "floor mat": "rug", "bath mat": "rug", "mat": "rug", "carpet": "rug",
}


def clean_phrase(value: object) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())


def phrase_in_text(phrase: str, text: str) -> bool:
    return f" {clean_phrase(phrase)} " in f" {clean_phrase(text)} "


class TaskEntityExtractor:
    def __init__(
        self, object_terms: Iterable[str], *, room_terms: Iterable[str] = (),
        object_aliases: Mapping[str, str] | None = None,
        room_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.object_aliases = {**DEFAULT_OBJECT_ALIASES, **(object_aliases or {})}
        self.room_aliases = {**DEFAULT_ROOM_ALIASES, **(room_aliases or {})}
        self.object_terms = {
            clean_phrase(term) for term in [*object_terms, *self.object_aliases, *self.object_aliases.values()]
            if clean_phrase(term)
        }
        defaults = {"kitchen", "bathroom", "bedroom", "living room", "dining room", "office", "laundry room", "closet", "entryway", "garage", "hallway"}
        self.room_terms = defaults | {clean_phrase(term) for term in room_terms}

    def canonical_object(self, value: object) -> str:
        cleaned = clean_phrase(value)
        return clean_phrase(self.object_aliases.get(cleaned, cleaned))

    def canonical_room(self, value: object) -> str | None:
        cleaned = clean_phrase(value)
        return clean_phrase(self.room_aliases.get(cleaned, cleaned)) or None

    def extract(self, task_text: str) -> tuple[str | None, str | None]:
        text = clean_phrase(task_text)
        room = next((term for term in sorted(self.room_terms, key=len, reverse=True) if phrase_in_text(term, text)), None)
        if room is None:
            room = next((canonical for alias, canonical in self.room_aliases.items() if phrase_in_text(alias, text)), None)
        obj = next((term for term in sorted(self.object_terms, key=len, reverse=True) if phrase_in_text(term, text)), None)
        return self.canonical_room(room), self.canonical_object(obj) or None
