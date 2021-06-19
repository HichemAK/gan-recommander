from model_cfwgan import CFWGAN


class ReleaseModel:
    def __init__(self, model_path, num_items):
        self.model = CFWGAN.load_from_checkpoint(model_path, trainset=None, num_items=num_items, config='movielens-1m')

    def predict(self, user_vector):
        self.model.eval()
        return self.model.generator(user_vector)