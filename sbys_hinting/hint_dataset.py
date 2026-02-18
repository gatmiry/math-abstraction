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
        self.unable_index = defaultdict(int)
        self.able_index = defaultdict(int)
        self.try_index = defaultdict(int)
        self.guide_steps_count = defaultdict(int)
        for row in self.dataframe:
            prob = row["prompt"][1]['content']
            #self.hint_level[problem_str] = 0
            self.unable_index[prob] = 0
            self.able_index[prob] = len(row["sbys_solution"])
            self.try_index[prob] = 0
            self.guide_steps_count[prob] = len(row["sbys_solution"])

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        prompt_str = str(row_dict["raw_prompt"])  # or however you key it
        level = self.hint_level.get(prompt_str, 0)
        if level > 0:
            # Modify row_dict["raw_prompt"] to inject hints
            row_dict["raw_prompt"] = self._add_hints(row_dict["raw_prompt"], level)
        return row_dict

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



            prob = g["prompt"]
            n = len(g["scores"])
            pr = sum(1 for s in g["scores"] if s > 0) 
            ## if pr is not 0 or 4, continue to the next problem
            if pr not in [0, 4]:
                continue

            all_correct = pr == 4
            
            if self.try_index[prob] <= self.unable_index[prob] and all_correct:
                self.able_index[prob] = self.try_index[prob] - 1
                self.try_index[prob] = max(self.try_index[prob] - 1, 0)
            elif self.try_index[prob] >= self.able_index[prob] and not all_correct:
                self.unable_index[prob] = self.try_index[prob]
                self.try_index[prob] = min(self.try_index[prob] + 1, self.guide_steps_count[prob])
            else:
                if not all_correct:
                    self.unable_index[prob] = self.try_index[prob]
                    self.try_index[prob] = math.ceil((self.try_index[prob] + self.able_index[prob]) / 2)
                else:
                    self.able_index[prob] = self.try_index[prob]
                    self.try_index[prob] = math.floor((self.try_index[prob] + self.unable_index[prob]) / 2)
            
            
            
            
            
            total_pass += (pr / n)

            # if pr == 0.0:
            #     self.hint_level[prob] = min(self.hint_level[prob] + 1, 5)
            # elif pr == 1.0:
            #     self.hint_level[prob] = max(self.hint_level[prob] - 1, 0)

        mean_pr = total_pass / len(groups) if groups else 0.0
        nonzero = {k: v for k, v in self.try_index.items() if v > 0}
        print(f"[HintDataset] on_batch_end: {len(groups)} problems, "
              f"mean_pass_rate={mean_pr:.3f}, "
              f"{len(nonzero)} problems with hints>0")
        print("--------------------------------")
        #exit()
