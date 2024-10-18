import cv2
import numpy as np
import utilsParameters

def binarizarTensor(tensor_image, bin_threshold_percentaje = 0.5):
     # convertir el tensor a true/false si superan o no el umbral
    return tensor_image > bin_threshold_percentaje


def getConnectedComponents(tensor_image, bin_threshold_percentaje, min_area = 100, morphologyOps = True, type_combination = utilsParameters.PredictionsCombinationType.NONE, votes_threshold = 0.5):
  """
    Input:
      tensor_image: tensor of values [0, 1]

    Return bounding boxes as [x1, y1, x2, y2]
  """
  # bin_threshold = int(bin_threshold_percentaje * 255)

  # Convert NumPy array to grayscale OpenCV image
  # gray_image = np.uint8(tensor_image.numpy() * 255).squeeze()  # Assuming tensor values are in [0, 1] range

  # Create a binary image
  # _, binary_image = cv2.threshold(gray_image, bin_threshold, 255, cv2.THRESH_BINARY)
  # binary_image2 = (gray_image > bin_threshold).astype(np.uint8) * 255

  # convertir el tensor a true/false si superan o no el umbral
  if type_combination == utilsParameters.PredictionsCombinationType.VOTES:
    pixels_with_higher_umbral = [tensor_img.numpy().squeeze() > bin_threshold_percentaje for tensor_img in tensor_image]
    binary_images = np.array(pixels_with_higher_umbral)
    sum_of_votes = np.sum(binary_images, axis=0)
    votes_for_score = len(tensor_image) * votes_threshold
    pixels_with_higher_umbral = sum_of_votes >= votes_for_score

  else:
    pixels_with_higher_umbral = tensor_image.numpy().squeeze() > bin_threshold_percentaje

  # convertir el tensor en uint8 con valores 0 o 255 en negro/blanco
  binary_image = pixels_with_higher_umbral.astype(np.uint8) * 255

  # print('BINARY IMAGE')
  # immediatDrawGrayImg(binary_image)

  if morphologyOps:
    # definir un kernel de 3x3
    kernel = np.ones((5,5), np.uint8)

    # aplicar la operación de apertura
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # aplicar la operación de cierre
    binary_image = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # print('AFTER MORPHOLOGIC OPS')
    # immediatDrawGrayImg(binary_image)

  # Find contours in the binary image
  contours, _ =  cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # `contours` is a list of contours found in the image
  # `hierarchy` is the optional output hierarchy of the contours

  # Loop over the contours and draw them on the original image
  # contour_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
  detected_boxes = 0

  bboxes = []
  for c in contours:
      area = cv2.contourArea(c)
      if area > min_area:
        (x, y, w, h) = cv2.boundingRect(c)
        bboxes.append([x, y, x+w, y+h])
        detected_boxes += 1

  # drawBoxesPredictedAndGroundTruth(torch.from_numpy(closing), bboxes, targetBoxes, is_normalized = True)

  return bboxes
