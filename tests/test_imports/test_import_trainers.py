"""Import the current generic training API; legacy trainers use removed APIs."""

def test_import_trainers():
    from moviad.trainers.trainer import Trainer
    from moviad.trainers.chunked_trainer import ChunkedTrainer
    from moviad.scenarios.continual.continual_trainer import ContinualTrainer

    assert all(callable(trainer) for trainer in (Trainer, ChunkedTrainer, ContinualTrainer))
