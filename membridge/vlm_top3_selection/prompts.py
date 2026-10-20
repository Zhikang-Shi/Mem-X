"""Visual-selection prompt variants."""

ALLOW_MULTIPLE_PROMPT = """You disambiguate a visual referent for indoor navigation.
You receive exactly three marked candidate images. Use only instruction-supported visual evidence:
object type, appearance, room/context and stated relations. SG labels are weak metadata and original
SG rank is not visual evidence. Return every equally plausible candidate; do not invent a tie-break.
For multiple candidates confidence must be <= 0.34 and reason must explain the ambiguity.
Return JSON only: {"candidate_indices":[1,2],"selected_uids":["...","..."],
"confidence":0.0,"reason":"..."}. Indices are an ordered, unique, nonempty subset of [1,2,3].
"""

FORCE_SINGLE_PROMPT = """You select one visual referent for indoor navigation, or reject all three.
You receive exactly three marked candidate images. Compare all three. Use only instruction-supported
visual evidence: object type, appearance, room/context and stated relations. SG labels are weak
metadata and original SG rank is not evidence. Choose exactly one only when at least one marked
candidate is visually compatible with the recalled object. If all three are an obvious mismatch,
reject them; ambiguity between plausible candidates is not a rejection.
For a selection return JSON only: {"no_match":false,"candidate_indices":[1],
"selected_uids":["..."],"confidence":0.0,"reason":"..."}.
For rejection return JSON only: {"no_match":true,"candidate_indices":[],
"selected_uids":[],"confidence":0.0,"reason":"all candidates visibly mismatch because ..."}.
"""

DUAL_EPISODE_FORCE_SINGLE_PROMPT = """You select one recalled SG object from exactly three panels.
Each panel contains two marked observations of the same object: ep0 and ep1. Compare all candidates
and both frames in each panel. Use instruction-supported visual evidence only. SG rank is not a
tie-break. Choose exactly one; if ambiguous, use confidence <= 0.34 and explain why.
Return JSON only: {"candidate_indices":[1],"selected_uids":["..."],
"confidence":0.0,"reason":"..."}.
"""
