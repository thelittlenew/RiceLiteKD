"""Keep every sample and update count while avoiding a final one-image batch."""
from torch.utils.data import BatchSampler
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class RedistributeSingletonBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        assert batch_size >= 3, 'Redistribution requires a batch size of at least three'
        assert not drop_last, 'This sampler must retain every image'
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is None:
                pending = batch
            elif len(batch) == 1:
                assert len(pending) >= 3
                yield pending[:-1]
                yield pending[-1:] + batch
                pending = None
            else:
                yield pending
                pending = batch
        if pending is not None:
            assert len(pending) >= 2, 'Training requires at least two images'
            yield pending
