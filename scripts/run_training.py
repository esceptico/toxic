from toxic.data.data_module import DataModule
from toxic.modelling.encoders import WideCNNEncoder
from toxic.modelling.models import ShallowSentenceClassifier
from toxic.training import train


def main():
    data = DataModule('/home/enio/PycharmProjects/toxic/data/dataset.txt')
    encoder = WideCNNEncoder(
        token_embedding_size=64,
        vocab_size=data.tokenizer.vocab_size,
        filters=[(1, 128), (2, 128), (3, 128)],
        dropout=0.2,
        projection_size=256
    )
    model = ShallowSentenceClassifier(
        encoder=encoder,
        dropout=0.2,
        projection_size=128,
        n_classes=4
    ).to('cuda')
    train(model, data.val_loader(), data.val_loader(), n_epoch=5)


if __name__ == '__main__':
    main()
