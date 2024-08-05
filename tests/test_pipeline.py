from squeeze_net.data import ImageNetDataLoader
from squeeze_net.model import SqueezeNet


def test_end2end():
    dataloader = ImageNetDataLoader(batch_size=1)
    model = SqueezeNet()
    for batch in dataloader:
        print(model(batch))
