import torch
import torch.nn as nn

class ModelCheckpoint:
  def __init__(self, filepath, model):
    """
      Initialises the class for a ModelCheckpoint callback.
      Arguments:
        filepath: string
          filename with .pt extension to save the model
        model: torch.nn
          model to save while training
    """
    self.filepath = filepath
    self.model: torch.nn.Module = model
    self.save()

  def save(self):
    torch.save(self.model, self.filepath)


class EarlyStopping:
  def __init__(self, patience, epochs, checkpoint, epsilon, searchFor = 'min', plotMsg = False):
    """
      Initialises the class for an Early Stopping callback.
      Arguments:
        patience: int
          number of epochs without improving the metric
        epochs: int
          total number of epochs
        checkpoint: ModelCheckpoint
          object for saving the best model
        epsilon: float
          threshold in order to update best result
        searchFor: str
          if best result is min or max
    """
    self.patience = patience
    self.epochs = epochs
    self.checkpoint = checkpoint
    self.stop_training = False
    self.actual_patience = 1
    self.best_result = None
    self.epsilon = epsilon
    self.plotMsg = plotMsg
    self.searchFor = searchFor

    if self.searchFor == 'min':
      self.best_result = float('inf')
    elif self.searchFor == 'max':
      self.best_result = 0.0

  def update(self, last_result, epoch):
    """
      Updates results after each epoch and saves the best model
      through ModelCheckpoint Object.
      Arguments:
        last_result: float
          result obtained on the selected metric
      Returns:
        Boolean
          True if the current patience has achieved maximum patience
          False otherwise
    """
    hasImproved = False

    if self.searchFor == 'min':
      hasImproved =  last_result < self.best_result
    elif self.searchFor == 'max':
      hasImproved =  last_result > self.best_result

    if hasImproved:
      # Improvement
      self.actual_patience = 1
      last_best = self.best_result
      self.best_result = last_result

      # Saving best model
      self.checkpoint.save()
      if self.plotMsg:
        print(f'Updating best result ({self.best_result}), last: {last_best}. Difference of {np.round(abs(last_best-self.best_result), 3)}.')
        print(f'Saving on {self.checkpoint.filepath}')

    else:
      # No improvement
      self.actual_patience += 1
      if self.actual_patience >= self.patience:
        self.stop_training = True
        if self.plotMsg:
          print(f'Training stopped with patience {self.patience}')
          print(f'Best result obtained: {np.round(self.best_result, 3)}')
      if self.plotMsg:
        print(f'Not updating best result ({self.best_result}), last: {last_result}.')


    # if self.plotMsg:
    print(f'\tEarly stopping: [{self.actual_patience}/{self.patience}] on epoch {epoch}/{self.epochs}')

    return self.stop_training

