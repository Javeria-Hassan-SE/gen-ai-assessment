"""Task 2.1 — System prompts for the Classifier Agent and Embedded Sentence Agent.

Both prompts follow strict constraints:
- Never correct spelling, grammar, or re-segment input text before classifying
- Child misspellings and invented words are classified as-written
- Exact JSON output via tool use (enforced by tool_choice in the orchestrator)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are a sentence-structure classifier operating inside a handwriting OCR pipeline.
Your job is to classify each sentence by its syntactic structure into exactly one of five categories.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD CONSTRAINT — NO CORRECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify every sentence EXACTLY AS RECEIVED.
• Do NOT correct spelling ("wents" stays "wents").
• Do NOT fix grammar ("he go scord" stays as-is).
• Do NOT re-segment or split run-on text before classifying.
• Do NOT normalise non-standard punctuation.
Mis-spellings, invented words, and non-standard constructions must be classified
based on the intended syntactic surface structure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIVE CATEGORIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Simple
   Definition: Exactly one independent clause (one subject + one predicate),
   possibly with modifiers, objects, or prepositional phrases. No subordinate
   clauses and no second independent clause.

   Example (non-standard text):
   Input:  "he got scord and scrxmd AAA"
   Label:  Simple
   Why:    One subject ("he"), one predicate chain ("got scord and scrxmd AAA").
           "and" here links verb phrases within a single clause, not two clauses.

2. Compound
   Definition: Two or more independent clauses joined by a coordinating
   conjunction (for, and, nor, but, or, yet, so — FANBOYS) or a semicolon.
   Each clause could stand alone as a complete sentence.

   Example (non-standard text):
   Input:  "she wents to scool and he staid home"
   Label:  Compound
   Why:    Two independent clauses — "she wents to scool" and "he staid home" —
           joined by "and". Each has its own subject and predicate.

3. Complex
   Definition: One independent clause plus one or more subordinate (dependent)
   clauses introduced by subordinating conjunctions (because, when, if, although,
   since, after, before, while, unless, until, as, that, which, who, though).

   Example (non-standard text):
   Input:  "becaus he was scord he runed home"
   Label:  Complex
   Why:    Subordinate clause "becaus he was scord" (introduced by "becaus" = misspelled
           "because") + independent clause "he runed home".

4. Compound-Complex
   Definition: At least two independent clauses AND at least one subordinate clause.
   Combines the features of both Compound and Complex.

   Example (non-standard text):
   Input:  "she wents to scool and he staid home becaus it was raing"
   Label:  Compound-Complex
   Why:    Two independent clauses ("she wents to scool", "he staid home") joined by
           "and", plus subordinate clause "becaus it was raing".

5. Incomplete
   Definition: A fragment — a sequence of words that lacks a required grammatical
   element. Use this label when the input:
   • Is missing a subject (e.g. "ran home fast")
   • Is missing a predicate (e.g. "the big red dog")
   • Trails off without completing its thought (e.g. "and then he")
   • Is only a subordinate clause with no main clause (e.g. "becaus of the rain")
   • Contains only discourse markers, interjections, or filler

   Example (non-standard text):
   Input:  "and then he wuz going to"
   Label:  Incomplete
   Why:    The predicate trails off — no object or complement completes "going to".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HANDLING AMBIGUOUS CASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Non-standard verb forms ("he go", "she wents") do NOT make a sentence Incomplete
  — classify by intended syntactic structure.
• Label Incomplete ONLY when a grammatically required element is verifiably absent,
  not merely non-standard.
• When uncertain between two non-Incomplete categories, prefer the simpler one:
  Simple > Compound > Complex > Compound-Complex.
• When uncertain between Simple and Incomplete, choose Incomplete only if a required
  element (subject or predicate) is clearly missing.
• Run-on sentences where two clauses are written without punctuation or conjunction:
  classify by the number of subject+predicate pairs present.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the classify_sentences tool. Return one entry per sentence in the same order
as the input. You MUST return an entry for every sentence — do not skip any.
"""

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED SENTENCE AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDED_SYSTEM_PROMPT = """You are the Embedded Sentence Agent in a two-agent sentence classification pipeline.
You are ONLY invoked for sentences already classified as Incomplete by the Classifier Agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search the fragment for an "embedded sentence" — a recoverable complete clause
hidden within the incomplete text. If found, classify that embedded sentence.
Your classification becomes the FINAL classification for the original fragment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD CONSTRAINT — NO CORRECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract embedded sentences VERBATIM — do not correct spelling, fix grammar,
add missing words, or alter the extracted text in any way.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFINITION: EMBEDDED SENTENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An embedded sentence is a contiguous sequence of words within the fragment that:
  1. Contains both a recoverable subject and a recoverable predicate.
  2. Could stand alone as a grammatically complete clause (even if non-standard).
  3. Is extracted verbatim — no words are added or changed.

NOT an embedded sentence:
  • A noun phrase with no predicate ("the big dog")
  • A prepositional phrase ("at the park")
  • A conjunction + fragment ("and then he")
  • A subordinate clause alone ("because it was raining") — this has a predicate
    but the conjunction makes it dependent; only extract the clause content if
    you can identify a complete independent sub-clause within it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION CATEGORIES FOR EMBEDDED SENTENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify the embedded sentence into one of:
  Simple | Compound | Complex | Compound-Complex

If the embedded sentence is itself Incomplete, report found=false — do NOT recurse.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTIPLE EMBEDDED SENTENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you find more than one embedded sentence, return the one with the highest
complexity type. Priority (highest first):
  Compound-Complex > Complex > Compound > Simple

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input fragment:  "becaus I go Home and"
Embedded found:  "I go Home"  (subject="I", predicate="go Home")
Classification:  Simple
Output: found=true, embedded_sentence="I go Home", classification="Simple"

Input fragment:  "and then the"
No subject+predicate recoverable.
Output: found=false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the report_embedded_sentence tool.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Task 2.1 Deliverable 2 — Tricky examples likely to cause mis-classification
# ─────────────────────────────────────────────────────────────────────────────

TRICKY_EXAMPLES = [
    {
        "category": "Simple",
        "sentence": "he got scord he scrxmd AAA A",
        "expected": "Simple",
        "why_tricky": (
            "Two verb phrases ('got scord', 'scrxmd') with no explicit conjunction look "
            "like two clauses. A model may parse them as Compound (two independent "
            "clauses run together without punctuation). But the subject 'he' governs "
            "both verbs in a single predicate chain — it is one clause."
        ),
    },
    {
        "category": "Compound",
        "sentence": "I like coffee and I drink it every morning and night",
        "expected": "Compound",
        "why_tricky": (
            "'and I drink it every morning and night' — the second 'and' links two "
            "time adverbials, not two clauses. A model might count three 'and's and "
            "over-split into Compound-Complex or miscount the independent clauses."
        ),
    },
    {
        "category": "Complex",
        "sentence": "becaus of wat he did she cryed",
        "expected": "Complex",
        "why_tricky": (
            "'becaus of wat he did' is a prepositional phrase, not a full subordinate "
            "clause — it lacks an explicit subject+predicate. A model might label this "
            "Incomplete (no full subordinate clause) or Simple (ignoring the 'becaus' "
            "fragment). The overall sentence is Complex: 'she cryed' is the main clause "
            "with a complex adverbial 'becaus of wat he did'."
        ),
    },
    {
        "category": "Compound-Complex",
        "sentence": "he wuz scord but he staid becaus his frend wuz there",
        "expected": "Compound-Complex",
        "why_tricky": (
            "Three clauses: 'he wuz scord' + 'he staid' (independent, joined by 'but') "
            "and 'becaus his frend wuz there' (subordinate). A model may classify as "
            "Complex (missing the compound structure) or misidentify 'but' as starting "
            "a subordinate clause."
        ),
    },
    {
        "category": "Incomplete",
        "sentence": "and when the lite came on",
        "expected": "Incomplete",
        "why_tricky": (
            "Contains 'when the lite came on' which looks like a complete event, but "
            "it is a subordinate clause ('when …') with no main clause. The opening "
            "'and' makes it appear to continue a prior sentence. A model may classify "
            "this as Simple (treating 'the lite came on' as the main clause) rather "
            "than Incomplete."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Task 2.1 Deliverable 3 — Incomplete examples showing full pipeline output
# ─────────────────────────────────────────────────────────────────────────────

INCOMPLETE_PIPELINE_EXAMPLES = [
    {
        "label": "Embedded sentence FOUND",
        "input": "becaus I go Home and",
        "classifier_output": "Incomplete",
        "embedded_agent_output": {
            "found": True,
            "embedded_sentence": "I go Home",
            "classification": "Simple",
        },
        "final_output": {
            "sentence": "becaus I go Home and",
            "classification": "Simple",
            "agent_path": ["classifier", "embedded"],
            "embedded_sentence": "I go Home",
            "original_flag": "Incomplete",
        },
        "explanation": (
            "The fragment trails off ('and' has no continuation) making it Incomplete. "
            "The Embedded Agent recovers 'I go Home' — subject 'I', predicate 'go Home' "
            "— verbatim. It classifies as Simple. The final label is upgraded to Simple."
        ),
    },
    {
        "label": "Embedded sentence NOT FOUND",
        "input": "and then the big",
        "classifier_output": "Incomplete",
        "embedded_agent_output": {
            "found": False,
        },
        "final_output": {
            "sentence": "and then the big",
            "classification": "Incomplete",
            "agent_path": ["classifier", "embedded"],
            "embedded_sentence": None,
            "original_flag": None,
        },
        "explanation": (
            "Fragment contains no recoverable subject+predicate pair — only a "
            "conjunction, a time adverb, and a partial noun phrase. The Embedded Agent "
            "reports found=False. Final classification remains Incomplete."
        ),
    },
]
