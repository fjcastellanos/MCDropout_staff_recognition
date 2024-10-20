import torch
import torch.nn as nn


class SAE(torch.nn.Module):
  def __init__(self):
    with_batch_normalization = True
    dropout_probability = 0

    super().__init__()
    self.encoder = torch.nn.Sequential(
      nn.Conv2d(in_channels=1  , out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.MaxPool2d(kernel_size=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.MaxPool2d(kernel_size=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.MaxPool2d(kernel_size=(2, 2)),
    )

    self.decoder = torch.nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.Upsample(scale_factor=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.Upsample(scale_factor=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.Upsample(scale_factor=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=1  , kernel_size=(5, 5), padding=2),
      nn.Sigmoid(),
    )

  def set_dropout_probability(self, dropout_probability = 0.2):
    for module in self.modules():
      if 'Dropout' in type(module).__name__:
        module.p = dropout_probability

  def enable_eval_dropout(self):
    for module in self.modules():
      if 'Dropout' in type(module).__name__:
        print(module)
        module.train()

  def forward(self, x):
    # Checks if input if batched or not. If it's not batched, add a dimension
    addDimension = x.dim() == 3

    if addDimension:
      x = torch.stack([x])

    x = self.encoder(x)
    x = self.decoder(x)

    # If we added a dimension, remove it for the loss function
    if addDimension:
      x = x.squeeze(0)

    return x