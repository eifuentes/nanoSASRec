# Extended Details

Reference material that the [README](README.md) links to. You don't need it to follow the curriculum.

## Remote training on Modal

`remote.py` preprocesses the requested dataset, mounts the local training scripts and the `sasrec/` package, adds `--device cuda`, and keeps downloaded data on a _Modal Volume_.

Before running several ML-20M jobs at once, with `sweep.sh` or by hand, run one job first and let it finish preprocessing. That fills the shared data volume, so the other jobs don't all try to download and extract the same dataset at once. If dataset processing goes wrong, you may need to wipe the _Modal Volume_.

## Checkpoints

v4+ scripts save checkpoints so an interrupted run can pick up where it left off, on Modal or anywhere else.

- **Where and when.** Every 5 epochs by default, and after the last epoch, to `checkpoints/<script>-<dataset>-seed<N>-<hash>/`. Pass `--checkpoint-every N` to save every N epochs, or `--checkpoint-every 0` to stop saving them. The hash covers every argument that changes the weights.
- **Resuming.** After any interruption (a crash, Ctrl-C, a preemption), run the same command again. It restores the model, the optimizer and all random number generators from the latest checkpoint. On Modal, checkpoints live on their own Volume and Modal re-runs the command itself, so a preempted job picks up where it left off.
- **Extending a run.** `--epochs` is not part of the hash, so a finished run can be extended by re-running it with a larger `--epochs`. The learning rate is constant, so the result is the same as a single longer run.
- **Changing code.** The hash covers arguments, not code. After changing a training script, delete its run directory (or pass a new `--checkpoint-dir`) so the run starts fresh.

### When a resumed run matches exactly

A resumed run matches an uninterrupted one exactly on CPU (`tests/test_checkpoint.py`). On GPU, two things can break that:

- **Summation order.** v7+ look up a learned bias for every pair of positions, so millions of entries read the same 32 rows of each bias table. In the backward pass, CUDA adds their gradients in whatever order its threads finish. Floating-point addition depends on order, so the last bits change from run to run, and over training that grows to the third or fourth decimal of the loss. Pass `--deterministic` to make PyTorch use fixed-order versions of those operations. Then v7+ repeat exactly too, at about 20% more time per epoch (v7 on ML-20M, A10). v4-v6 have no bias tables and repeat exactly without it.
- **GPU model.** Different GPU models can run different kernels, so their results differ in the last bits even with `--deterministic`. Modal's `A10` can be either an A10 or an A10G, so a run preempted on one and resumed on the other won't match exactly.

This drift is noise of the same kind as changing the seed, and no larger. After 3 epochs of v7 on ML-20M it moved NDCG@10 by 0.0004 (0.1527 vs 0.1531) and left HR@10 unchanged, inside the ±0.0005 NDCG spread across seeds 42-44. It doesn't change any conclusion in this repo.
