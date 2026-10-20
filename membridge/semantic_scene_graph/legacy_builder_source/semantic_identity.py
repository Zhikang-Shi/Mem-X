"""Pure helpers for conservative object identity and alias handling."""


GENERIC_LABELS = {"object", "item", "thing", "stuff", "unknown", "entity"}


def normalize_semantic_label(value):
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def canonical_semantic_label(value, aliases=None):
    label = normalize_semantic_label(value)
    alias_map = {
        normalize_semantic_label(alias): normalize_semantic_label(canonical)
        for alias, canonical in (aliases or {}).items()
    }
    return alias_map.get(label, label)


def semantic_labels_compatible(
    left, right, aliases=None, generic_labels=None
):
    generic = {
        normalize_semantic_label(value)
        for value in (generic_labels or GENERIC_LABELS)
    }
    left_canonical = canonical_semantic_label(left, aliases)
    right_canonical = canonical_semantic_label(right, aliases)
    if not left_canonical or not right_canonical:
        return False
    if left_canonical in generic or right_canonical in generic:
        return left_canonical == right_canonical
    return left_canonical == right_canonical
