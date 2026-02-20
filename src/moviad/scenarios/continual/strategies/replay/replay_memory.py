import torch
import random

"""
class Memory:

    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.tasks_memory = {}      
        self.num_tasks = 0    
    
    def _rebalance(self):
        task_quota = self.memory_size // self.num_tasks

        for task_id in self.tasks_memory:
            while(len(self.tasks_memory[task_id]) > task_quota): 
                idx = random.randrange(len(self.tasks_memory[task_id]))
                self.tasks_memory[task_id].pop(idx)
        
    def add_samples(self, task_id: int, samples: torch.Tensor): 
        if task_id not in self.tasks_memory.keys():
            self.tasks_memory[task_id] = []
            self.num_tasks += 1
            self._rebalance()

        task_quota = self.memory_size // self.num_tasks

        for sample in samples:
            if len(self.tasks_memory[task_id]) < task_quota:
                self.tasks_memory[task_id].append(sample.clone())
            else:
                j = random.randint(0, len(self.tasks_memory[task_id]) - 1)
                if j < task_quota:
                    self.tasks_memory[task_id][j] = sample.clone()

    def get_samples(self, n_replay_samples: int):
        samples_per_task = n_replay_samples // self.num_tasks

        samples = []
        if n_replay_samples < self.num_tasks:
            for task_id, memory_samples in self.tasks_memory.items(): 
                # take one sample from each task until we reach n_replay_samples
                idx = random.randrange(len(memory_samples))
                samples.append(self.tasks_memory[task_id][idx].unsqueeze(dim=0))
        else:
            for task_id, memory_samples in self.tasks_memory.items(): 
                n_samples = min(samples_per_task, len(memory_samples))
                samples_idx = torch.randperm(len(memory_samples))[:n_samples]
                for idx in samples_idx:
                    samples.append(self.tasks_memory[task_id][idx].unsqueeze(dim=0))

        return torch.cat(samples)
"""


class Memory:

    def __init__(self, memory_size: int):
        if memory_size < 1:
            raise ValueError("memory_size must be at least 1.")
        self.memory_size = memory_size
        self.tasks_memory: dict[int, list[torch.Tensor]] = {}
        self.num_tasks: int = 0
        # Counts for Reservoir Sampling: how many samples each task had seen
        self._seen_counts: dict[int, int] = {}

    @property
    def task_quota(self) -> int:
        """Max number of samples per task, given the actual number of tasks"""
        if self.num_tasks == 0:
            return 0
        return self.memory_size // self.num_tasks

    def _rebalance(self) -> None:
        """
        Reduce each task at the current task_quota by removing random samples
        """
        quota = self.task_quota
        if quota == 0:
            # Caso limite: più task che slot di memoria — svuota tutto
            for tid in self.tasks_memory:
                self.tasks_memory[tid] = []
            return

        for tid in self.tasks_memory:
            buf = self.tasks_memory[tid]
            while len(buf) > quota:
                # Rimuove un elemento casuale in O(1) swap-with-last
                idx = random.randrange(len(buf))
                buf[idx] = buf[-1]
                buf.pop()

    def add_samples(self, task_id: int, samples: torch.Tensor) -> None:
        """
        Add samples to the buffer with reservoir sampling

        Args:
            task_id: task identifier
            samples:  tensor of shape (n, ...) with the samples to add
        """
        if task_id not in self.tasks_memory:
            self.tasks_memory[task_id] = []
            self._seen_counts[task_id] = 0
            self.num_tasks += 1
            self._rebalance()  # reduce other tasks to the new quote

        quota = self.task_quota
        buf = self.tasks_memory[task_id]

        for sample in samples:
            self._seen_counts[task_id] += 1
            n_seen = self._seen_counts[task_id]

            if len(buf) < quota:
                # Buffer is not full yet -> add directly
                buf.append(sample.clone())
            else:
                # Reservoir sampling: replace with probability quota/n_seen
                j = random.randrange(n_seen)
                if j < quota:
                    buf[j] = sample.clone()

    def get_samples(
        self, n_replay_samples: int, exclude_task_id: int | None = None
    ) -> torch.Tensor:
        """
        Return a samples batch from the memory without duplicates

        Samples are uniformally distributed between admissible tasks.
        If a task has less samples then quota, the remaining are distributed between other tasks

        Args:
            n_replay_samples: number of samples to return
            exclude_task_id:  task to exclude (usually the current task)

        Returns:
            Tensor of shape (n, ...) with the selected samples.

        Raises:
            ValueError: se la memoria ammissibile è vuota.
        """

        # Allowable task
        if exclude_task_id is None:
            task_ids = list(self.tasks_memory.keys())
        else:
            task_ids = [tid for tid in self.tasks_memory if tid != exclude_task_id]

        if not task_ids:
            raise ValueError("No allowable task in memory.")

        # Check total availability
        available_per_task = {tid: list(range(len(self.tasks_memory[tid]))) for tid in task_ids}
        total_available = sum(len(v) for v in available_per_task.values())

        if total_available == 0:
            raise ValueError("Allowable memory is empty.")

        n = min(n_replay_samples, total_available)

        # ----------------------------------------------------------------
        # Distribuzione equa con redistribuzione degli avanzi
        # ----------------------------------------------------------------
        # Calcola quanti campioni prendere per ogni task, rispettando la disponibilità effettiva e garantendo zero duplicati.

        # Assegna una quota iniziale uguale per tutti i task
        quota_per_task: dict[int, int] = {}
        remaining_n = n
        remaining_tasks = list(task_ids)
        """
        while remaining_tasks and remaining_n > 0:
            base = remaining_n // len(remaining_tasks)
            overflow_tasks = []

            for tid in remaining_tasks:
                avail = len(available_per_task[tid])
                take = min(base if base > 0 else 1, avail)
                quota_per_task[tid] = quota_per_task.get(tid, 0) + take
                remaining_n -= take
                if quota_per_task[tid] < avail:
                    overflow_tasks.append(tid)

            # Eventuali campioni ancora da assegnare vanno ai task con disponibilità
            remaining_tasks = overflow_tasks
            if base == 0 and not overflow_tasks:
                break  # nessun task ha più disponibilità
        """

        while remaining_tasks and remaining_n > 0:
            # IMPORTANT: randomizza l'ordine per non favorire sempre i primi task
            random.shuffle(remaining_tasks)

            base = remaining_n // len(remaining_tasks)
            overflow_tasks: list[int] = []

            for tid in remaining_tasks:
                avail = len(available_per_task[tid])

                # se base == 0, prova comunque ad assegnare 1 finché ci sono sample da assegnare
                take = min(base if base > 0 else 1, avail, remaining_n)

                if take <= 0:
                    continue

                quota_per_task[tid] = quota_per_task.get(tid, 0) + take
                remaining_n -= take

                # se il task ha ancora disponibilità, può ricevere altri campioni nei giri successivi
                if quota_per_task[tid] < avail:
                    overflow_tasks.append(tid)

                if remaining_n == 0:
                    break

            remaining_tasks = overflow_tasks
            if base == 0 and not overflow_tasks:
                break  # nessun task ha più disponibilità

        # ----------------------------------------------------------------
        # Campionamento senza rimessa (indici unici per task)
        # ----------------------------------------------------------------
        samples: list[torch.Tensor] = []

        for tid, take in quota_per_task.items():
            if take <= 0:
                continue
            mem = self.tasks_memory[tid]
            # torch.randperm garantisce indici unici
            chosen_indices = torch.randperm(len(mem))[:take].tolist()
            for idx in chosen_indices:
                samples.append(mem[idx].unsqueeze(0))

        if not samples:
            raise ValueError("Impossible to select samples from the memory.")

        return torch.cat(samples, dim=0)