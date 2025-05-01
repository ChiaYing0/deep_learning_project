from torch.utils.data import DataLoader

class Trainer:

    # [Train]
    # include check point: save, load model parameters


    def __init__(self, config):
        pass

    def train(self, model, train_dataset, valid_dataset):
        
        train_dataloader = DataLoader(train_dataset, batch_size=4)

        for batch in train_dataloader:

            print('do one batch')
            logit = model(batch)
            break

    def test(self, model, test_dataset):
        pass