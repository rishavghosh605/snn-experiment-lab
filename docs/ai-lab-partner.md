# Using AI as a Research Lab Partner

**How to start fast, actually learn, and not just copy-paste answers.**

## Why this matters now

AI has moved past the "ask a chatbot" stage. Across higher education — including in **Germany** (IU International University (Duisburg)'s **Syntea** learning companion, TU Munich's **OneTutor** Socratic tutor serving tens of thousands of students) — the shift is toward using AI as an **active learning and research partner**: something that runs experiments, iterates, and walks you through results, rather than a search engine that hands back a paragraph.

The confusion most people feel isn't "is AI good?" — it's **"how do I actually use it to get good at something?"** This guide answers that.

## The mindset shift

Three ways people misuse AI for learning:

- **The copy-paste trap** — you get working code and a false sense of skill. If you can't re-explain it, you didn't learn it.
- **The oracle trap** — you ask "what is X?" once and move on. That's info, not understanding.
- **The vague-ask trap** — "teach me SNNs" produces either too-simple or too-deep text, because the AI never knew where you actually stand.

The fix in all three is the same: treat the AI as a **lab partner** — you do it *with* it, one small step at a time, and you are the one who explains the results.

## The method — how to start

A concrete loop for learning anything technical:

1. **State your level.** Exactly where you are ("I know Python, I'm new to neural networks"). This sets the AI's difficulty correctly.
2. **Pick one topic.** Not "SNNs" — *one* thing, like "single neuron behaviour."
3. **Define "done."** Say what success looks like ("I can draw a plot of membrane potential and explain every part of it").
4. **Do experiments, not lectures.** Ask the AI to set up a runnable experiment that produces a visible result, then explain the graph to you.
5. **Explain it back.** Re-explain the mechanism in your own words; have the AI check you.
6. **One variation.** Change one thing and predict what happens *before* running it. This is where real understanding forms.
7. **Escalate only when it clicks.** Move on only after step 6 — otherwise you're just collecting terms.

This is exactly why **[`AGENT_PROMPT.md`](./AGENT_PROMPT.md)** exists: it codifies the loop as a 10-step per-experiment routine (Question → Hypothesis → Build → Run → Visualize → Read the graph → Insight → Math → Challenge → Decide). Paste it into your agent and it becomes a partner that *makes you* reason, instead of an answer generator.

## Best practices that work

| Technique | How to apply it |
|:---|:---|
| **Role prompt** | "You are my lab partner. Guide me with questions before giving answers." |
| **Context first** | Tell it your toolchain, level, and constraints up front. |
| **Socratic mode** | Ask it to lead you to the answer with hints, not hand it over. |
| **Explain-back** | After any result, write your own explanation and have it critique it. |
| **Predict-run** | Guess the outcome, then run, then compare. The gap is the lesson. |
| **Ground it** | Ask for sources/references for anything you'll actually rely on. |

## Pitfalls to watch

- **Hallucination.** AI can be confidently wrong or cite outdated docs. Verify anything high-stakes against authoritative sources.
- **Shallow context.** A one-line ask gets a generic answer. Give it your level and your goal every time you frame a new topic.
- **Skipping the "why."** Always ask *why* a design/result is what it is. The why is the learning.
- **Letting it build blindly.** If the AI writes code you can't follow, stop and ask for it in smaller steps.

## Use it responsibly

The strongest finding from institutions like Stanford is that AI works best **human-in-the-loop** — as a partner that supports *you*, not a replacement for thinking. Colleges are adopting AI to help people learn more, and the ones getting value treat it as a collaborator: they verify, they question, and they use the AI to push their own understanding forward rather than substitute for it.

**In short:** use the AI to run the experiment, but *you* read the graph and say what it means. That's the difference between using AI and learning with AI.
