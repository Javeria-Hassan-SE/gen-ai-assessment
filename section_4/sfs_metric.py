# pip install editdistance pyenchant
# pyenchant needs the enchant C lib: brew install enchant / apt install libenchant-2-dev
# falls back to a built-in word list if enchant isn't available

from __future__ import annotations

import editdistance
from typing import Literal

# try pyenchant, fall back to a hardcoded common-word set
try:
    import enchant as _enchant
    _dict = _enchant.Dict("en_US")
    _USE_ENCHANT = True
except (ImportError, Exception):
    _USE_ENCHANT = False
    # ~500 most common English words as a lightweight fallback
    _COMMON_WORDS: frozenset[str] = frozenset({
        "a", "able", "about", "above", "across", "act", "action", "actually",
        "add", "after", "again", "age", "ago", "agree", "air", "all", "allow",
        "almost", "alone", "along", "already", "also", "always", "am", "among",
        "an", "and", "another", "answer", "any", "area", "are", "around", "as",
        "ask", "asked", "at", "away", "back", "bad", "bag", "ball", "bank",
        "be", "became", "because", "become", "been", "before", "began", "behind",
        "being", "below", "best", "better", "between", "big", "bit", "black",
        "blue", "body", "book", "both", "boy", "break", "broke", "broken",
        "brought", "build", "built", "but", "buy", "by", "call", "called",
        "came", "can", "cannot", "car", "care", "carry", "case", "cat",
        "cause", "certain", "change", "check", "child", "children", "city",
        "clear", "close", "cold", "come", "coming", "could", "country",
        "course", "cut", "dark", "day", "dead", "did", "different", "do",
        "does", "dog", "done", "door", "down", "draw", "during", "each",
        "early", "end", "enough", "even", "ever", "every", "example",
        "eye", "eyes", "face", "fact", "fall", "family", "far", "father",
        "feel", "felt", "few", "field", "find", "fire", "first", "follow",
        "food", "for", "form", "found", "free", "from", "front", "full",
        "get", "getting", "give", "given", "go", "god", "gone", "good",
        "got", "great", "green", "ground", "group", "grow", "had", "hand",
        "happen", "hard", "has", "have", "he", "head", "hear", "heard",
        "heart", "help", "her", "here", "him", "his", "hold", "home",
        "hope", "hot", "house", "how", "however", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "keep", "kind",
        "knew", "know", "land", "large", "last", "later", "laugh", "learn",
        "leave", "left", "let", "life", "light", "like", "line", "list",
        "listen", "little", "live", "long", "look", "low", "made", "make",
        "man", "many", "may", "me", "mean", "meet", "men", "might", "more",
        "most", "move", "much", "must", "my", "name", "need", "never",
        "new", "next", "night", "no", "nor", "not", "nothing", "now",
        "number", "of", "off", "often", "old", "on", "once", "only",
        "open", "or", "other", "our", "out", "over", "own", "part",
        "past", "people", "perhaps", "place", "play", "point", "put",
        "ran", "read", "real", "red", "right", "road", "room", "run",
        "said", "same", "saw", "say", "school", "see", "seem", "set",
        "she", "short", "show", "side", "since", "small", "so", "some",
        "soon", "stand", "start", "still", "stop", "story", "such", "sun",
        "take", "talk", "tell", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "thing", "think", "this", "those",
        "though", "thought", "through", "time", "to", "today", "together",
        "told", "too", "took", "top", "toward", "turn", "two", "under",
        "until", "up", "upon", "us", "use", "very", "walk", "want", "was",
        "watch", "way", "we", "well", "went", "were", "what", "when",
        "where", "which", "while", "who", "why", "will", "with", "woke",
        "word", "work", "world", "would", "write", "year", "yes", "yet",
        "you", "young", "your", "hello", "early", "broke",
    })


def is_standard_english(word: str) -> bool:
    """True if word is in the standard English dictionary (case-insensitive)."""
    w = word.lower().strip()
    if not w:
        return False
    if _USE_ENCHANT:
        return _dict.check(w)
    return w in _COMMON_WORDS


def _classify_edit(gt_token: str, pred_token: str) -> Literal["correct", "normalisation", "noise"]:
    """Classify one token-level diff as correct, normalisation, or noise.

    normalisation: pred is valid English but doesn't match gt — could be a
    silent correction of an intentional misspelling. Weight 1.0 in SFS because
    it corrupts verbatim fidelity without any visible signal.

    noise: pred isn't valid English — garbled output. Weight 0.5 because it's
    at least detectable.

    Known limitation: when both gt and pred are valid English words (e.g. "ran"
    vs "run"), this returns 'normalisation' even though it may just be a tense
    error. SFS can't tell the difference in that case.
    """
    if gt_token == pred_token:
        return "correct"
    if is_standard_english(pred_token):
        return "normalisation"
    return "noise"


def sfs(ground_truth: str, predicted: str) -> float:
    """Source-Fidelity Score — measures verbatim transcription fidelity.

    Unlike CER/WER, SFS weights edit types by their impact on fidelity:
      - normalisation edits (gt non-standard, pred standard): weight 1.0
      - noise edits (pred not standard English): weight 0.5

    Formula: SFS = 1 - (norm_penalty + noise_penalty) / max(len(gt_tokens), 1)
    Clamped to [0.0, 1.0]. Higher is better.

    Tokens are whitespace-split and lowercased. Alignment is positional;
    extra or missing tokens on either side count as noise (0.5 each).
    """
    gt_tokens = ground_truth.lower().split()
    pred_tokens = predicted.lower().split()

    n_gt = len(gt_tokens)
    if n_gt == 0:
        return 1.0

    norm_penalty = 0.0
    noise_penalty = 0.0

    for gt_tok, pred_tok in zip(gt_tokens, pred_tokens):
        edit = _classify_edit(gt_tok, pred_tok)
        if edit == "normalisation":
            norm_penalty += 1.0
        elif edit == "noise":
            noise_penalty += 0.5

    # unmatched tokens (length difference) all count as noise
    noise_penalty += abs(len(pred_tokens) - n_gt) * 0.5

    return max(0.0, min(1.0, 1.0 - (norm_penalty + noise_penalty) / n_gt))


def cer(ground_truth: str, predicted: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    return editdistance.eval(ground_truth, predicted) / max(len(ground_truth), 1)


if __name__ == "__main__":
    # Example 1: silent normalisation
    # CER is low (small character diff) but SFS catches that the engine
    # silently "fixed" intentional child misspellings.
    gt1 = "I siad hello and wokeup eerly"
    pr1 = "I said hello and woke up early"
    print("Example 1 - Silent Normalisation")
    print(f"  GT   : {gt1}")
    print(f"  Pred : {pr1}")
    print(f"  CER  : {cer(gt1, pr1):.4f}  <- low, looks fine to standard metrics")
    print(f"  SFS  : {sfs(gt1, pr1):.4f}  <- low, normalisation edits caught")
    print()

    # Example 2: noise errors
    # Both CER and SFS flag this as bad, but SFS uses noise weight (0.5)
    # not normalisation weight (1.0) — garbled output is a different failure
    # mode from silent correction.
    gt2 = "he got scord and ran"
    pr2 = "he g2[ xkqz and r@n"
    print("Example 2 - Noise Errors")
    print(f"  GT   : {gt2}")
    print(f"  Pred : {pr2}")
    print(f"  CER  : {cer(gt2, pr2):.4f}  <- high, both metrics agree it's bad")
    print(f"  SFS  : {sfs(gt2, pr2):.4f}  <- lower than 1.0 but noise penalty only")
