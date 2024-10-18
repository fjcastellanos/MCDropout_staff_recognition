""" Drawing functions """

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import  Image as PILImage
import utilsParameters

def drawTensorImageColor(tensor_image):

  img = np.uint8(tensor_image.squeeze().permute(1, 2, 0).numpy() * 255)

  # display the image
  plt.imshow(img)
  plt.show()

def drawTensorImageGrayScale(tensor_image, plot=False, save=False, image_name=None):

  gray_image = np.uint8(tensor_image.squeeze().numpy() * 255)  # Assuming tensor values are in [0, 1] range

  if plot:
    img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # display the image
    plt.imshow(img)
    plt.show()

  if save:
    image = PILImage.fromarray(gray_image, mode='L')

    image.save(image_name)

def drawBoxesOnTensor(tensor_image, bboxes, plot=False, save=False, image_name=None) :

  gray_image = np.uint8(tensor_image.numpy() * 255).squeeze()  # Assuming tensor values are in [0, 1] range
  contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

  for b in bboxes:
    x1, y1, x2, y2 = [int(elem) for elem in b]
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

  # convert BGR to RGB
  img = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

  if plot:
    # display the image
    plt.imshow(img)
    plt.show()

  if save:
    # save the image
    cv2.imwrite(image_name, img)

def drawBoxesPredictedAndGroundTruth(tensor_image, bboxes_predicted, bboxes_groundtruth, is_normalized, plot, save, image_name):

  gray_image = np.uint8(tensor_image.numpy()).squeeze() if is_normalized else np.uint8(tensor_image.numpy() * 255).squeeze() # Assuming tensor values are in [0, 1] range
  contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
  green_color = (0, 255, 0)
  red_color   = (0, 0, 255)

  for b in bboxes_predicted:
    x1, y1, x2, y2 = [int(elem) for elem in b]
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), red_color, 2, cv2.LINE_AA)

  for b in bboxes_groundtruth:
    x1, y1, x2, y2 = [int(elem) for elem in b]
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), green_color, 1, cv2.LINE_AA)

  # convert BGR to RGB
  img = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

  if plot:
    # display the image
    plt.imshow(img)
    plt.show()

  if save:
    if utilsParameters.DEBUG_FLAG:
      print(f'Saving in {image_name}')
    cv2.imwrite(image_name, img)


def immediatDrawGrayImg(gray_image):
  img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
  plt.imshow(img)
  plt.show()


def saveTensorToImageGrayScale(tensor_image, image_name):

  gray_image = np.uint8(tensor_image.squeeze().numpy() * 255)  # Assuming tensor values are in [0, 1] range

  image = PILImage.fromarray(gray_image, mode='L')

  image.save(image_name)


def saveBoxesOnTensorToImage(tensor_image, bboxes, image_name):
  gray_image = np.uint8(tensor_image.numpy() * 255).squeeze()  # Assuming tensor values are in [0, 1] range
  contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

  for b in bboxes:
    x1, y1, x2, y2 = [int(elem) for elem in b]
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

  # convert BGR to RGB
  img = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

  # save the image
  cv2.imwrite(image_name, img)


def saveBoxesPredictedAndGroundTruth(tensor_image, bboxes_predicted, bboxes_groundtruth, image_name):
  # gray_image = np.uint8(tensor_image.numpy() * 255).squeeze()  # Assuming tensor values are in [0, 1] range
  gray_image = np.uint8(tensor_image.numpy() * 255).squeeze() # Assuming tensor values are in [0, 1] range
  contour_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
  green_color = (0, 255, 0)
  red_color   = (255, 0, 0)

  for b in bboxes_predicted:
    x1, y1, x2, y2 = [int(elem) for elem in b]
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), red_color, 1, cv2.LINE_AA)

  for b in bboxes_groundtruth:
    x1, y1, x2, y2 = [int(elem) for elem in b]
    cv2.rectangle(contour_image, (x1, y1), (x2, y2), green_color, 1, cv2.LINE_AA)

  # cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
  # convert BGR to RGB
  img = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
  cv2.imwrite(image_name, img)




def plotTrainval_losses(train_losses, val_losses):
  # Convert the loss history lists to numpy arrays
  train_loss_history = np.array(train_losses)
  val_loss_history = np.array(val_losses)

  # Plot the loss curves
  plt.plot(train_loss_history, label='Training loss')
  plt.plot(val_loss_history, label='Validation loss')
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

