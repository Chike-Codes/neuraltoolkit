class Subset:
    """
    A view of a slice of a dataset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.x = self.dataset.x[self.indices]
        self.y = self.dataset.y[self.indices]      

    @property
    def size(self):
        return self.x.shape[0]