# How the agent policy works

![The agent loop](docs/agent-loop.svg)

**The idea.** A language model plays the role of a performance engineer: it
reads what has been measured, forms a hypothesis, and decides the next
experiments. Everything around it is ordinary code: running experiments,
keeping the books, choosing finalists. The platform, not the model, has the
final word.

## The loop

1. **What we know.** Before every decision the model gets a fresh summary of
   the night: every result, every failure and its cause, the noise, its own
   notes so far.
2. **The model decides.** It explains what it thinks is going on and proposes
   configs to test it.
3. **Experiments run.** The configs are launched and benchmarked, several at
   once on the night's GPU budget.
4. **The lab books** record the results, and the next summary is built from
   them. The model is woken when a result lands, not on a timer.

## Design choices

- **The summary is rebuilt every turn, not remembered.** A long conversation
  forgets or drifts; a summary rebuilt from the books cannot. Old chat can be
  dropped, the summary never is, so nothing learned is lost.
- **The model only gets experiment verbs.** Propose, check, read a log,
  measure again, write a note — no shell, no files. It spends its effort on
  tuning, not on operating a computer.
- **Code decides what is a finalist.** The model cannot write scores. Results
  are compared against production measured twice, and noise is taken into
  account, so one lucky measurement cannot win.
- **The platform gives the verdict.** At the end it measures the finalists and
  production itself, side by side. A good story from the model does not count.
- **Failures teach.** A failed launch is read and diagnosed, and becomes a
  constraint the model sees on every later turn, so the same mistake is not
  paid for twice.
- **Nights build on each other.** Lessons and constraints are saved and shown
  to the next night.

Built on [pi](https://github.com/earendil-works/pi), which runs the model and
its tool calls; everything above is this package.
