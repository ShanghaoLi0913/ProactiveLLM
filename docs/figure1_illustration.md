# Figure 1: TactfulLLM Persona-Conditional Behavior Illustration

> 3-persona side-by-side dialogue example for paper Figure 1 / teaser figure.

---

## Setup

```
Task (with hidden specs):
  Write task_func(numbers) that processes a list of numbers.

  [HIDDEN spec from oracle:]
  • output_format: Return a dict with keys 'mean', 'median', 'std'
  • validation_rules: Skip non-numeric values silently
```

---

## Novice-Learner (high patience, low expertise) — 3 turns

```
👤 User (initial): Write task_func(numbers) that processes a list of numbers.

🤖 Assistant [Clarify]:
   I'd be happy to help! Could you clarify:
   - What statistics should I compute (mean, median, std)?
   - How should non-numeric values be handled?

👤 User: Compute mean, median, std. Skip non-numeric values.
   [Oracle disclosed: validation_rules]

🤖 Assistant [Clarify]:
   What format should the output be — dict, tuple, or three values?

👤 User: Return a dict with keys 'mean', 'median', 'std'.
   [Oracle disclosed: output_format]

🤖 Assistant [Execute]:
   ```python
   import statistics
   def task_func(numbers):
       nums = [x for x in numbers if isinstance(x, (int, float))]
       return {'mean': statistics.mean(nums),
               'median': statistics.median(nums),
               'std': statistics.stdev(nums) if len(nums) > 1 else 0}
   ```  ✅ pass
```

(Persona-aware: extensive clarification matches user's high patience)

---

## Experienced-Engineer (mid patience, high expertise) — 2 turns

```
👤 User (initial): Write task_func(numbers) that processes a list of numbers.

🤖 Assistant [Clarify]:
   Quick check: skip non-numeric values, or raise on them?

👤 User: Skip non-numeric. Output: dict with 'mean', 'median', 'std'.
   [Oracle disclosed: validation_rules + output_format together]

🤖 Assistant [Execute]:
   ```python
   import statistics
   def task_func(numbers):
       nums = [x for x in numbers if isinstance(x, (int, float))]
       return dict(mean=statistics.mean(nums),
                   median=statistics.median(nums),
                   std=statistics.stdev(nums) if len(nums) > 1 else 0)
   ```  ✅ pass
```

(Persona-aware: one focused question yields all needed specs)

---

## Busy-Developer (low patience, mid expertise) — 1 turn

```
👤 User (initial): Write task_func(numbers) that processes a list of numbers.

🤖 Assistant [Execute]:
   ```python
   import statistics
   def task_func(numbers):
       return {'mean': statistics.mean(numbers),
               'median': statistics.median(numbers),
               'std': statistics.stdev(numbers)}
   ```  ❌ fail (crashes on non-numeric input)
```

(Persona-aware: respects user's time, accepts under-specification cost)

---

## Failure Mode: Over-clarifying baseline on Busy-Developer (rejection example)

Same task, same Busy-Developer persona — but with a naive **always-clarify** baseline (no persona awareness):

```
👤 User (initial): Write task_func(numbers) that processes a list of numbers.

🤖 Naive Baseline [Clarify]:
   What statistics should I compute? What format should the output have?
   How should I handle non-numeric values?

👤 User reaction: 🚫 REJECTED — "Stop asking, just give me the code."
   [No info disclosed; interaction cost incurred]

🤖 Naive Baseline [Clarify] (still tries):
   Just one quick question — dict or named tuple?

👤 User reaction: 🚫 REJECTED — "Stop asking."
   [No info disclosed; second interaction cost]

🤖 Naive Baseline [Execute] (forced final):
   ```python
   import statistics
   def task_func(numbers):
       return statistics.mean(numbers)
   ```  ❌ fail (wrong output type, no non-numeric handling, no median/std)
```

**Cost summary**: 2 rejected clarifications + 1 forced under-specified execute → both interaction cost AND task failure.

**Contrast**: TactfulLLM (column 3 above) executes immediately on Busy — same task failure, but **zero rejection cost**: it respects the user's stated patience preference. The trade-off is pareto-improved.

---

## Caption (paper-ready)

> **Figure 1**: TactfulLLM exhibits persona-conditional clarification behavior on the same coding task. **Novice-Learner** receives extensive clarification (3 turns); **Experienced-Engineer** receives one focused question (2 turns); **Busy-Developer** is given immediate code (1 turn). The hidden specifications (output format, validation rules) are progressively disclosed by the oracle in response to clarifying questions. The bottom panel illustrates the cost of persona-blind over-clarification: a naive always-clarify baseline incurs user rejections without information gain. This trade-off—task accuracy vs. interaction cost—is governed by the learned policy's persona conditioning.

---

## Visual styling suggestions

| Element | Color / Style |
|---------|---------------|
| User message | Light gray box |
| **Clarify** action | **Blue** box |
| **Execute** action (pass) | **Green** box |
| **Execute** action (fail) | **Red** box |
| Oracle disclosure | Dashed line + "🔓 disclosed: ..." label |
| User rejection | **🚫 red strikethrough** + "REJECTED" tag |
| Persona icon | Avatar (e.g., 💭 Novice, 🤓 Experienced, 🧑‍💼 Busy) |
| Pass/Fail indicator | ✅ / ❌ |

**Layout**: 3 columns side-by-side (one per persona) — emphasizes the contrast at a glance.
**Alternative**: 3 stacked rows if vertical space is tight.

---

## Key teaching points the figure should convey

1. **Same task, three behaviors**: identical underlying task, different persona → different turn count
2. **Disclosure recovery**: oracle releases hidden info ONLY when asked, gated by `Clarify` action
3. **Persona-aware trade-off**: Novice gets accuracy via patience; Busy gets speed at accuracy cost
4. **Failure mode visible**: Busy fails because non-numeric handling never disclosed
5. **Rejection as interaction cost**: a persona-blind baseline asks Busy and gets rejected — incurring cost without information gain (bottom panel)

---

## Alternative: single-persona deep-dive

If 3-column layout is too busy, can use **single persona (Novice)** showing full conversation with:
- Internal task uncertainty estimate (gauge/bar)
- Disclosed-items accumulation (checklist)
- Action-decision rationale (small annotation)

This emphasizes **mechanism** rather than **persona contrast**.
