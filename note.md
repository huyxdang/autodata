# Notes

## Decoupled dataloader time from training budget

The 5-minute training timer in `train.py` now only counts GPU compute + optimizer time. Dataloader `next()` calls for the next step happen outside the timer.

This means the agent can do expensive data processing (complex filters, dedup, etc.) in `data.py` without being penalized for CPU time. Previously, a slow dataloader would reduce the number of training steps, conflating "slow code" with "bad data."

**Caveat**: The inter-micro-step fetch (between grad accumulation steps) is still inside the timer, but it overlaps with GPU backward computation so it's negligible. If the dataloader is extremely slow, there could still be brief GPU idle time during this window. A background prefetch thread in the dataloader would fix this if it becomes an issue.
