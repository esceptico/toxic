from toxic.data.data_module import DataModule
from toxic.modelling.model import Model
from toxic.modelling.training import train


def main():
    data = DataModule('/home/enio/PycharmProjects/toxic/data/dataset.txt')
    model = Model(vocab_size=data.tokenizer.vocab_size,
                  token_emb_size=64, filters=[(1, 128), (2, 128), (3, 128)]).to('cuda')
    train(model, data.val_loader(), data.val_loader())


if __name__ == '__main__':
    main()
