import torch
import random

class Memory:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.tasks_memory = {}
        self.tasks_seen = {}       # Tracks total samples seen per task for Reservoir Sampling
        self.num_tasks = 0

    def _rebalance(self):
        task_quota = self.memory_size // self.num_tasks

        for task_id in self.tasks_memory:
            # Randomly drop samples until we meet the new quota for every task memory
            while len(self.tasks_memory[task_id]) > task_quota:
                idx = random.randrange(len(self.tasks_memory[task_id]))
                self.tasks_memory[task_id].pop(idx)

    def add_samples(self, task_id: int, samples: torch.Tensor):
        if task_id not in self.tasks_memory:
            self.tasks_memory[task_id] = []
            self.tasks_seen[task_id] = 0
            self.num_tasks += 1
            self._rebalance() # for making space for the new task

        task_quota = self.memory_size // self.num_tasks

        for sample in samples:
            self.tasks_seen[task_id] += 1

            # Fill memory up to the quota
            if len(self.tasks_memory[task_id]) < task_quota:
                self.tasks_memory[task_id].append(sample.clone())
            else:
                # Reservoir Sampling: Decreasing probability of overwrite over time
                j = random.randint(0, self.tasks_seen[task_id] - 1)
                if j < task_quota:
                    self.tasks_memory[task_id][j] = sample.clone()

    def get_samples(self, n_replay_samples: int):

        # 1. Determine how many samples to draw from each task
        quotas = [n_replay_samples // self.num_tasks] * self.num_tasks
        remainder = n_replay_samples % self.num_tasks

        # Distribute the remainder randomly across tasks
        task_indices = list(range(self.num_tasks))
        random.shuffle(task_indices)
        for i in range(remainder):
            quotas[task_indices[i]] += 1

        samples = []
        task_ids = list(self.tasks_memory.keys())

        # 2. Extract the samples based on the calculated quotas
        for task_id, quota in zip(task_ids, quotas):
            memory_samples = self.tasks_memory[task_id]
            if not memory_samples:
                continue

            n_samples = min(quota, len(memory_samples))
            samples_idx = torch.randperm(len(memory_samples))[:n_samples]

            for idx in samples_idx:
                samples.append(memory_samples[idx].unsqueeze(dim=0))

        if not samples:
             return torch.empty(0)

        return torch.cat(samples)