# Failure-video filename convention

How rollout replay videos are named by the patched openpi LIBERO eval client, and
what downstream tooling relies on.

## The saving gate

```python
save_videos: str = "failures"   # CLI: --args.save-videos {all|failures|none}
...
replay_images.append(img)       # per-step, inside the try
...
suffix = "success" if done else "failure"
task_segment = task_description.replace(" ", "_")
if args.save_videos == "all" or (args.save_videos == "failures" and not done):
    imageio.mimwrite(
        pathlib.Path(args.video_out_path)
        / f"rollout_{args.task_suite_name}_task{task_id}_ep{episode_idx}_{task_segment}_{suffix}.mp4",
        [np.asarray(x) for x in replay_images], fps=10,
    )
```

## The naming logic, exactly

```python
suffix       = "success" if done else "failure"
task_segment = task_description.replace(" ", "_")
name         = f"rollout_{args.task_suite_name}_task{task_id}_ep{episode_idx}_{task_segment}_{suffix}.mp4"
path         = pathlib.Path(args.video_out_path) / name
```

Flat directory — no subdirectories at write time. All grouping happens later via
`organize_videos.py` / `arrange_videos_pi.py`.

Example:

```
rollout_libero_object_unseen_full_task936_ep3_pick_up_the_lamb_chop_and_place_it_in_the_basket_failure.mp4
```

## Provenance of each component

| Component | Source | Notes |
|---|---|---|
| `rollout_` | literal | fixed prefix, load-bearing for the consumer regexes |
| `{task_suite_name}` | `args.task_suite_name` (CLI `--args.task-suite-name`) | e.g. `libero_object_unseen_full`. **Contains underscores**, so it is not delimiter-safe |
| `task{task_id}` | loop var over `task_ids` | index into the suite, *not* into `--args.task-ids`. Passing `--args.task-ids "936 937"` still yields `task936`, `task937` |
| `ep{episode_idx}` | `range(args.num_trials_per_task)` | 0-based, resets per task |
| `{task_segment}` | `task.language` → spaces→underscores | the natural-language instruction, e.g. `pick_up_the_lamb_chop_and_place_it_in_the_basket`. Not `task.name` |
| `{suffix}` | `done` at loop exit | `success` \| `failure` |

## Parsing consequences

The only reliably-parseable anchors are the literal prefix, `_task<digits>_`,
`_ep<digits>_`, and the terminal `_(success|failure).mp4`. Suite name and description
both contain underscores, so **you cannot recover fields by splitting on `_`** — every
consumer uses an anchored regex instead:

```python
# organize_videos.py — suite name absorbed into an opaque prefix
r"^(?P<prefix>rollout_.+_task\d+)_ep\d+_.*_(?P<status>success|failure)\.mp4$"

# aggregate_unseen_eval.py:91 — glob, wildcards over suite-vs-description ambiguity
f"*{suite}_task{task['task_id']}_ep{ep.get('episode_idx')}_*failure.mp4"
```

`prefix` deliberately spans `rollout_…_task936` as one unit — the greedy `.+` swallows
the suite name because there is no way to delimit it. If you reimplement, keeping
`_task<N>_ep<K>_` intact and the status terminal is what preserves compatibility; the
middle is free.

## Collision behavior

`(suite, task_id, episode_idx)` is unique within a run, so there are no in-run
collisions. But the name carries **no model, seed, or trial-count**, so two lanes
writing to the same `video_out_path` silently overwrite each other. The lane scripts
separate them by directory (`full/videos_pi05` vs `full/videos_pi0fast`) rather than by
filename — preserve that, since the failure mode is silent.

Also: `success`/`failure` is embedded in the name, so re-running a task that flips
outcome leaves the stale opposite-suffix file behind rather than replacing it.

## Semantics worth getting right

- **`done` is the success flag**, not a termination flag. On success the loop `break`s
  immediately, so success videos are systematically shorter than failures (which always
  run the full `max_steps`).
- **Exceptions are filed as failures.** The `except: break` path leaves `done=False`, so
  a crashed episode produces a `_failure.mp4` indistinguishable from an honest policy
  failure. `episode_records` records `success: false` for it too — the aggregator cannot
  separate them either.
- **Frames are what the policy sees, not the raw render**: agentview rotated 180°
  (`[::-1, ::-1]`), `resize_with_pad` to 224², uint8. Padding bars are expected.
- **The `num_steps_wait` settling period is not recorded** — appending starts after it.
  `steps` in the results JSON is `max(0, t - num_steps_wait)`, matching.
- **Terminal frame is missing**: append happens *before* `env.step`, so the video ends
  one frame before the state that set `done`.
- Buffer is per-episode RAM, reset at episode start.

## Consumers (change these together with the name)

- `scripts/aggregate_unseen_eval.py:91` — emits a `video_hint` glob into `failure_index.csv`
- `/tmp2/leocheng/eval_results/organize_videos.py` — regroups to `{prefix}/{status}/`
- `/tmp2/leocheng/eval_results/arrange_videos_pi.py` — regroups to `{object}/{prefix}/{status}/`;
  its regex additionally hardcodes `pick_up_the_(.+)_and_place_it_in_the_basket`, so it
  only works for pick-place suites

## Reference run

The completed full-pool run used `--args.save-videos all` on both lanes, yielding 22,037
`_failure.mp4` files under `eval_results/full/videos_pi05` and `videos_pi0fast`. For
drill-down only, `failures` is the cheaper setting and is the code default.
