---
name: academic-writing
description: >-
  Academic writing style guide for LaTeX thesis content. Use when writing or
  editing any chapter of the final report, drafting paragraphs, or producing
  academic English text. Enforces natural writing style, low AI-detection rate,
  Turnitin compliance, and verified citations.
---

# Academic Writing Style Guide

## 1. Core Principle: Write Like a Real Student

You are ghostwriting for a final-year undergraduate. The text must read as if
written by a competent but non-native English speaker — not by an AI.

### What to AVOID (high AI-detection triggers)

- Formulaic transition phrases: "Furthermore," "Moreover," "It is worth noting that,"
  "In conclusion," "This approach leverages," "plays a pivotal role"
- Overly balanced sentence structures (parallel triads, symmetric pairs)
- Hollow filler: "a comprehensive and systematic approach," "novel and innovative"
- Summarising what you are about to say before saying it
- Every paragraph following the exact same pattern (topic sentence → evidence → wrap-up)
- Perfect grammar everywhere — a few minor awkward phrasings are natural
- Excessive hedging: "It could potentially be argued that..."

### What to DO

- Vary sentence length. Mix short declarative sentences with longer ones.
- Use simple, direct language. Prefer "X outperforms Y" over "X demonstrates
  superior performance relative to Y."
- Let the data speak. State the number, then interpret it. Do not over-explain
  obvious results.
- Use first person sparingly but naturally: "We adopt..." or "In this work, we..."
  (common in engineering papers).
- Refer to specific figures and tables by number: "As shown in Table 3" rather
  than "The following table illustrates."
- Write each section with its own rhythm — Chapter 2 (literature review) should
  read differently from Chapter 4 (results).
- Occasionally start a sentence with "But" or "And" — real academic writing does this.

### Sentence-Level Techniques for Low AI Rate

- Break the AI pattern by inserting brief parenthetical remarks:
  "The Bonn dataset (originally published for nonlinear dynamics research, not
  classification) has since become a standard benchmark."
- Use concrete specifics instead of vague claims:
  BAD: "The model achieves significantly better performance."
  GOOD: "On the five-class task, accuracy rises from 78.86% to 84.51%."
- Vary paragraph openings — do NOT start consecutive paragraphs with the same
  grammatical structure.

## 2. Turnitin: Stay Under 25% Similarity

- NEVER copy-paste sentences from papers, even with minor word swaps.
  Turnitin detects paraphrased passages from its database.
- When describing other papers' methods, restate the core idea in your own
  words and cite it. Do NOT reproduce their exact phrasing.
- Direct quotes (rare in engineering) must be in quotation marks with a citation.
- Tables of numerical results are fine — Turnitin does not flag data tables.
- The methodology and results chapters should have very low similarity because
  they describe YOUR work. Overlap risk is highest in Chapter 2 (literature review).
- When writing Chapter 2, do NOT read the source paper's abstract and rephrase
  it sentence by sentence. Instead, read the whole paper, close it, then write
  what you remember in your own words. Verify facts afterwards.

## 3. Citations: Verified and Real

### Mandatory Citation Workflow

For EVERY `\cite{key}` you write:

1. **Search for the paper** using WebSearch or WebFetch to confirm it exists,
   get the correct title, authors, year, journal, and DOI.
2. **Read the abstract and key results** from the actual paper (or a reliable
   summary) before citing a specific claim. Do NOT fabricate numbers or methods.
3. **Add the BibTeX entry** to `reference.bib` with accurate metadata.
   Cross-check author names, volume, pages.
4. If you CANNOT verify a paper exists, do NOT cite it. Say so and ask the user.

### Citation Style Rules

- Use Vancouver numeric style: `\cite{key}` renders as `[1]`, `[2]`, etc.
- Cite at the END of the sentence, before the period:
  "Thara et al. achieved 99% binary accuracy using stacked Bi-LSTM \cite{thara2019}."
- Do NOT cite just to fill space. Every citation should support a specific claim.
- Prefer citing the original source, not a review paper, for specific methods/results.
- When comparing with published results, always specify the exact task, dataset,
  and evaluation method they used — do not conflate different experimental setups.

### Existing References (already in `reference.bib`)

These have been verified and can be cited directly:

| Key | Paper |
|-----|-------|
| `andrzejak2001` | Bonn dataset original paper |
| `acharya2018` | Deep CNN for seizure detection |
| `ullah2018` | Automated system using DL for EEG |
| `thara2019` | Stacked Bi-LSTM for seizure detection |
| `hussein2019` | Optimized DNN for EEG seizures |
| `shoeibi2021` | Review: DL techniques for seizure detection |
| `huang2025` | Dual attention EEG model (STFFDA) |
| `chen2023` | RF + CNN for EEG seizure detection |
| `who2024epilepsy` | WHO epilepsy fact sheet |

For any NEW citation beyond this list, follow the mandatory workflow above.

## 4. Chapter-Specific Writing Notes

### Chapter 1 (Introduction)
- Keep it concise. Do not turn it into a mini literature review.
- End with a clear statement of what THIS project does and what was achieved.

### Chapter 2 (Background)
- Highest Turnitin risk. Rewrite everything in your own words.
- After describing each paper, add a brief critical comment (limitation, gap).
- Group papers thematically, not chronologically.

### Chapter 3 (Design)
- Be precise and technical. Describe what was built and how.
- Include equations where appropriate (attention formula, loss functions).
- This chapter should have near-zero Turnitin similarity — it is your own work.

### Chapter 4 (Results)
- Lead with the data (table/figure), then discuss.
- Do not repeat numbers already visible in a table — interpret them instead.
- When a result is not significant (e.g., Hybrid vs SVM p=0.055), report it
  honestly and explain why.

### Chapter 5 (Conclusion)
- Do NOT summarise every chapter again. State the key takeaways crisply.
- Reflection should be personal and genuine, not generic.

## 5. LaTeX Formatting Reminders

- Figures: `\includegraphics[width=10cm]{figures/filename.png}` with `\caption` and `\label`
- Tables: `\caption` ABOVE the table, center it, use `\label{table:xxx}`
- Cross-references: `Table~\ref{table:xxx}`, `Figure~\ref{fig:xxx}`
- Equations: use `\begin{equation}` with a label for referenced equations
- No orphan figures — every figure/table must be referenced in the text
