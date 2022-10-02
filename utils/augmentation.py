

import numpy as np
from keras.preprocessing.image import Iterator

#Create DataPreprocessor class

class CustomIterator(Iterator):

    def __init__(self, modalities, batch_size=6, shuffle=False, seed=None,
                 dim_ordering='tf'):
        self.modality1 = modalities[0]
        self.modality2 = modalities[1]
        self.dim_ordering = dim_ordering
        self.batch_size = batch_size
        super(CustomIterator, self).__init__(self.modality1.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_modality1 = np.zeros(tuple([len(index_array)] + list(self.modality1.shape[1:])))
        batch_modality2 = np.zeros(tuple([len(index_array)] + list(self.modality2.shape[1:])))

        for i, j in enumerate(index_array):
            _modality1 = self.modality1[j]
            _modality2 = self.modality2[j]
            batch_modality1[i]= _modality1
            batch_modality2[i] = _modality2

        return [batch_modality1, batch_modality2], [batch_modality1, batch_modality2]


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
