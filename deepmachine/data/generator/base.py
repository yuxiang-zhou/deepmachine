import numpy as np
from pathlib import Path

from ... import utils, callbacks

class Generator(utils.Sequence, callbacks.Callback):
    def __init__(self, dirpath, data_process_fn, multi_output=True, glob='*.*', batch_size=32):
        super().__init__()

        if type(glob) is not list:
            glob = [glob]

        self.data_path = []
        for g in glob:
            self.data_path += list(Path(dirpath).rglob(g))
        
        self.batch_size = batch_size
        self.indexes = list(range(len(self.data_path)))
        self.multi_output = multi_output
        self._data_process_fn = data_process_fn
        np.random.shuffle(self.indexes)



    def on_epoch_end(self, *args, **kwargs):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.data_path) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_output = []
        for i in batch_indexes:
            output = self._data_process_fn(self.data_path[i])
            if output is not None:
                batch_output.append(output)

        if self.multi_output:
            batch_output = list(map(list, zip(*batch_output)))
            batch_output = list(map(np.array, batch_output))
        else:
            batch_output = np.array(batch_output)

        return batch_output