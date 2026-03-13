# autodata

Optimize training data for language model pre-training — both what the model sees (filtering, truncation, quality selection, deduplication, etc.) and how it's arranged (curriculum ordering, mixing, packing strategy, etc.) — to minimize val_bpb.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autodata/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autodata/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — fixed model architecture, optimizer, training loop. Do not modify.
   - `data.py` — the file you modify to process training data.
4. **Verify data exists**: Ask the human if they have already run `modal run modal_app.py --prepare`. If not, tell them to run it (this downloads data shards and trains the tokenizer on Modal).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation
Each experiment runs on a remote GPU via Modal. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `modal run modal_app.py`.

**What you CAN do:**
- Modify `data.py` — this is the only file you edit. Anything that changes what data the model sees or how it's arranged is fair game. You can rewrite the entire file if you want.

**What you CANNOT do:**
- Modify `train.py`. It is read-only. The model architecture, optimizer, and training loop are fixed.
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, base data loading, tokenizer, and training constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.
- Change the tokenizer. Vocab size and split pattern are frozen constants.

**The goal is simple: get the lowest val_bpb by improving the data pipeline.** The model sees exactly the same architecture and hyperparameters every run. The only variable is what data it trains on and how that data is processed. Since the time budget is fixed at 5 minutes, filtering out low-quality data means the model spends more of its budget on high-quality data.

**CPU cost matters.** The data pipeline runs on CPU and the timer includes data loading. Heavy processing eats into the 5-minute training budget, meaning fewer training steps.

**VRAM** should stay roughly constant since the model is fixed, but data pipeline changes (e.g. document length distribution) could affect it slightly.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is (with the default pass-through data pipeline).

## What's in data.py

The entire `data.py` file is yours to modify however you want. Rewrite it, restructure it, add new functions, delete existing ones — whatever gets the lowest val_bpb.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline (no filtering)
b2c3d4e	0.993200	44.0	keep	filter docs shorter than 100 chars
c3d4e5f	1.005000	44.0	discard	aggressive quality filter (too much data removed)
d4e5f6g	0.000000	0.0	crash	regex filter caused empty batches
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autodata/mar12`).

LOOP FOREVER:

1. Look at the git state and review `results.tsv` to understand what's been tried and what the current best is.
2. **Explore the data** when forming a new hypothesis, when stuck, when you haven't looked at the data recently, or simply find it useful to. Don't explore every single iteration — use your judgment. When you do explore, write whatever analysis code is useful for what you're curious about: sample documents, compute statistics, look at length distributions, spot junk, check for repetition, etc. Let what you actually see drive your hypothesis — don't just guess.
3. Modify `data.py` with a data idea informed by your exploration.
4. git commit
5. Run the experiment: `modal run modal_app.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
10. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (e.g. empty batch from over-filtering, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches, draw on best practices from papers and the ML community. The loop runs until the human interrupts you, period.
