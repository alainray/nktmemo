from typing import Callable, Sequence, Union
import functools
import flax.linen as nn

# Code taken from https://github.com/gortizji/linearized-networks from
# paper "What can Linearized Networks really tell us about generalization?" (NeurIPS 2021)



class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        # x = nn.log_softmax(x)
        return x


class LeNet(nn.Module):
    activation: Union[None, Callable] = nn.relu
    kernel_size: Sequence[int] = (5, 5)
    strides: Sequence[int] = (2, 2)
    window_shape: Sequence[int] = (2, 2)
    num_classes: int = 1
    features: Sequence[int] = (6, 16, 120, 84, 1)
    pooling: bool = True
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(nn.Conv, padding=self.padding)
        x = conv(features=self.features[0], kernel_size=tuple(self.kernel_size))(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pooling:
            x = nn.avg_pool(x, window_shape=tuple(self.window_shape), strides=tuple(self.strides))

        x = conv(features=self.features[1], kernel_size=tuple(self.kernel_size))(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pooling:
            x = nn.avg_pool(x, window_shape=tuple(self.window_shape), strides=tuple(self.strides))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.features[2])(x)
        if self.activation is not None:
            x = self.activation(x)
        x = nn.Dense(self.features[3])(x)
        if self.activation is not None:
            x = self.activation(x)

        x = nn.Dense(self.num_classes)(x)
        # x = nn.log_softmax(x)
        return x

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x


model_dict = {'fc': MLP, 'lenet': LeNet, 'cnn': CNN}
model_params = {'fc': {'features':[200,100,1]}, 'lenet': {'num_classes': 1, 'features': (6,16,120,84,1)}, 'cnn': {}}