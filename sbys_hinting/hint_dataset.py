"""
Custom dataset with on_batch_end hook for dynamic hint-level adjustment.

verl natively calls on_batch_end(batch) after each training step if the
dataset class has the method. This lets us inspect rewards and adjust
prompts (e.g., hint levels) for the next iteration.
"""

from verl.utils.dataset.rl_dataset import RLHFDataset


class HintDataset(RLHFDataset):
    """RLHFDataset subclass with on_batch_end for dynamic prompt adjustment."""

    def on_batch_end(self, batch):
        n = 4  # rollout.n — number of generations per prompt
        scores = batch.batch["token_level_scores"].sum(-1)  # [B*n]
        scores = scores.view(-1, n)  # [B, n] — 4 scores per prompt

        mean_score = scores.mean(dim=1)   # average reward per problem
        pass_rate = (scores > 0).float().mean(dim=1)  # fraction of correct generations

        # Problem with 0/4 correct → needs more hints
        # Problem with 4/4 correct → reduce hints
        print(f"[HintDataset] on_batch_end: {len(mean_score)} problems, "
              f"mean_score={mean_score.mean().item():.3f}, "
              f"mean_pass_rate={pass_rate.mean().item():.3f}")
        print(f"problem [0] is {batch.non_tensor_batch["raw_prompt"][0]} reward is {scores[0]}")
        print(f"problem [1] is {batch.non_tensor_batch["raw_prompt"][1]} reward is {scores[1]}")
        print(f"problem [2] is {batch.non_tensor_batch["raw_prompt"][2]} reward is {scores[2]}")
        print(f"problem [3] is {batch.non_tensor_batch["raw_prompt"][3]} reward is {scores[3]}")
        print("--------------------------------")
        exit()