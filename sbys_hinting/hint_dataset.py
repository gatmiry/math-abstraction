"""
Custom dataset with on_batch_end hook for dynamic hint-level adjustment.

verl natively calls on_batch_end(batch) after each training step if the
dataset class has the method. This lets us inspect rewards and adjust
prompts (e.g., hint levels) for the next iteration.
"""

from collections import defaultdict, OrderedDict

from verl.utils.dataset.rl_dataset import RLHFDataset


class HintDataset(RLHFDataset):
    """RLHFDataset subclass with on_batch_end for dynamic prompt adjustment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-problem hint level: maps problem_string → current hint level (0 = no hints)
        self.hint_level = defaultdict(int)  # problem_str -> int

    def on_batch_end(self, batch):
        scores = batch.batch["token_level_scores"].sum(-1)  # [B*n] total reward per response
        uids = batch.non_tensor_batch["uid"]                # same uid for all n rollouts of a problem
        prompts = batch.non_tensor_batch["raw_prompt"]

        # Group scores by uid (batch may be reordered by balance_batch)
        groups = OrderedDict()  # uid -> {prompt, scores}
        for i, uid in enumerate(uids):
            if uid not in groups:
                groups[uid] = {"prompt": prompts[i], "scores": []}
            groups[uid]["scores"].append(scores[i].item())

        total_pass = 0.0
        for uid, g in groups.items():
            n = len(g["scores"])
            pr = sum(1 for s in g["scores"] if s > 0) / n
            total_pass += pr
            prob = g["prompt"]
            if pr == 0.0:
                self.hint_level[prob] = min(self.hint_level[prob] + 1, 5)
            elif pr == 1.0:
                self.hint_level[prob] = max(self.hint_level[prob] - 1, 0)

        mean_pr = total_pass / len(groups) if groups else 0.0
        nonzero = {k: v for k, v in self.hint_level.items() if v > 0}
        print(f"[HintDataset] on_batch_end: {len(groups)} problems, "
              f"mean_pass_rate={mean_pr:.3f}, "
              f"{len(nonzero)} problems with hints>0")
        print("--------------------------------")
        exit()
