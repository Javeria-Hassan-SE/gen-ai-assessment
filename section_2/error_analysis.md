# Task 2.3 — Error Analysis: Sentence Classification

## 1. Confusion Matrix Interpretation

A confusion matrix for the five-class classifier would have rows = true labels and
columns = predicted labels.  The cells most likely to be non-zero (off-diagonal) are:

| True \ Predicted  | Simple | Compound | Complex | Compound-Complex | Incomplete |
|-------------------|--------|----------|---------|-----------------|------------|
| **Simple**        | ✓      | ↑        | –       | –               | ↑          |
| **Compound**      | ↑      | ✓        | –       | ↑               | –          |
| **Complex**       | –      | –        | ✓       | ↑               | ↑          |
| **Compound-Complex** | –  | ↑        | ↑       | ✓               | –          |
| **Incomplete**    | ↑      | –        | ↑       | –               | ✓          |

`✓` = correct diagonal, `↑` = common confusion direction, `–` = rare.

---

## 2. Most Common Error Patterns

### A. Simple → Compound (over-splitting)

**Pattern:** The model sees two verb phrases connected by "and" and incorrectly
assigns two independent clauses.

**Example:**  
`"he got scord and scrxmd AAA A"` → predicted **Compound**, true **Simple**

**Root cause:** "and" is canonically a coordinating conjunction that joins independent
clauses in Compound sentences.  When the subject is implicit (elided) in the second
verb phrase — "he got scord **and** [he] scrxmd" — the model fails to recognise it as
verb-phrase coordination rather than clause coordination.  This is especially frequent
with non-standard spelling that disrupts part-of-speech inference.

**Fix / Mitigation:**  
The system prompt explicitly addresses this: *"'and' here links verb phrases within a
single clause, not two clauses."* The prompt instructs the model to look for separate
explicit subjects, not just coordinating conjunctions.

---

### B. Incomplete → Simple (under-flagging fragments)

**Pattern:** A fragment that has a surface subject+verb structure is labelled Simple
when a required complement is missing or the predicate trails off.

**Example:**  
`"and when the lite came on"` → predicted **Simple**, true **Incomplete**

**Root cause:** The subordinate clause "when the lite came on" contains a full
subject-predicate pair.  The model conflates having a subject and verb with being a
complete sentence, ignoring that (a) the whole thing is a dependent clause and (b) the
leading "and" signals continuation of prior text.

**Fix / Mitigation:**  
The system prompt defines Incomplete to include *"only a subordinate clause with no
main clause"* and warns: *"When uncertain between Simple and Incomplete, choose
Incomplete only if a required element (subject or predicate) is clearly missing."*
Additionally, the Embedded Agent acts as a safety net: an incorrectly labelled Simple
sentence is NOT re-routed (routing is one-way — only Incomplete goes to the Embedded
Agent), so clear guidance in the Classifier prompt is critical.

---

### C. Complex → Compound-Complex (over-promotion)

**Pattern:** A sentence with one independent clause and one subordinate clause is
labelled Compound-Complex because the model over-counts "and" or other conjunctions as
creating an additional independent clause.

**Example:**  
`"she wents to scool and he staid home becaus it was raing"` is a true
Compound-Complex, but `"he wuz scord becaus his frend wuz there and he felt safe"` has
only one independent clause ("he wuz scord") with a complex adverbial — the model may
still predict Compound-Complex.

**Root cause:** The model greedily interprets every coordinating conjunction as a
clause boundary.  With non-standard spelling, syntactic parsing is degraded so the
model falls back on surface heuristics (conjunction count) rather than clause count.

**Fix / Mitigation:**  
The prompt requires the model to count *independent clauses* (those that could stand
alone), not conjunctions.  The example set includes a case where "and" links
adverbials, not clauses.

---

### D. Compound → Compound-Complex (spurious subordinate clause detection)

**Pattern:** A sentence with two independent clauses but no subordinate clause is
labelled Compound-Complex because the model misidentifies a prepositional phrase or
participial phrase as a subordinate clause.

**Example:**  
`"I like coffee and I drink it every morning and night"` → predicted
**Compound-Complex**, true **Compound**

**Root cause:** "every morning and night" is an adverbial time expression.  The model
may parse "and night" as a conjunction introducing a new clause, or treat the
prepositional phrase as a subordinate clause.

**Fix / Mitigation:**  
The prompt instructs the model that subordinate clauses are introduced by specific
subordinating conjunctions (because, when, if, although…) and that adverbials/
prepositional phrases do not count.

---

### E. Complex → Incomplete (over-fragmentation of truncated subordinators)

**Pattern:** A sentence beginning with a misspelled subordinating conjunction is
labelled Incomplete because the model cannot recognise the word as a subordinator.

**Example:**  
`"becaus of wat he did she cryed"` → predicted **Incomplete** (treating "becaus of"
as an incomplete conjunction phrase), true **Complex**

**Root cause:** OCR errors and child misspellings produce tokens like "becaus", "wen",
"becuz", "altho" that the model's tokeniser doesn't align with known subordinators.
Without recognising the subordinator, the model may treat the opening phrase as a
fragment.

**Fix / Mitigation:**  
The system prompt explicitly lists "becaus" as a misspelled "because" in the Complex
example.  The hard constraint section states: *"Mis-spellings… must be classified based
on the intended syntactic surface structure"* — so the model is directed to infer
intent, not require exact spelling.

---

## 3. Role of the Embedded Sentence Agent in Reducing Errors

The Embedded Agent does not reduce the errors above — those all happen *within* the
Classifier Agent's decision space (categories 1–4 and Incomplete).

Its role is narrower: for sentences that *are genuinely Incomplete* (fragments), it
recovers any complete clause hiding inside the fragment and promotes the classification
to reflect that clause's structure.  This increases precision on the Incomplete class
by separating:

- Pure fragments (nothing recoverable → stays Incomplete)  
- Fragments containing a hidden clause (e.g., `"becaus I go Home and"` → promoted to
  Simple because "I go Home" is recoverable)

This matters because child handwriting frequently contains trailing conjunctions, false
starts, or partially written sentences that embed a complete thought.  Without the
Embedded Agent, all of these would be discarded as Incomplete, losing structural
information that the downstream application (e.g., a writing-complexity dashboard)
could use.

---

## 4. Summary Table

| Error Type | Direction | Cause | Prompt Defence |
|---|---|---|---|
| Verb-phrase "and" splits clause | Simple → Compound | Treats VP coord as clause coord | Explicit "and links VPs" note |
| Dependent clause taken as complete | Incomplete → Simple | Surface S+V present | Subordinator + "and" continuation rule |
| "and" over-counted | Complex → Compound-Complex | Conjunction ≠ clause | Count independent clauses, not conjunctions |
| PP/adverb treated as subordinate | Compound → Compound-Complex | Heuristic clause detection | Requires specific subordinating conjunctions |
| Misspelled subordinator not recognised | Complex → Incomplete | Tokeniser miss-alignment | Classify by intended structure, not exact tokens |
