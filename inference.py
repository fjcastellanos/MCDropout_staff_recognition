import torch
import utilsParameters
import DataLoader
import DataHelper
import ConnectedComponents
import utilsIO
import metrics
import SAE
import drawing

import gc
from PIL import  Image as PILImage
import torchvision.transforms as T
import numpy as np


def getPredictionModel(dataset_name):
  model = torch.load(f'SAE_{dataset_name}.pt', map_location=torch.device(utilsParameters.device))
  model.to(utilsParameters.device)
  return model

def TFMForward(
    dataset_name, image: PILImage, uses_redimension_vertical: bool, uses_redimension_horizontal: bool,
    bin_umbral: float, val_dropout: float, times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType
    ):
    """Forward de una imagen para obtener los bounding boxes

    Args:
        dataset_name (string): Nombre del dataset
        image (Image): Imagen a hacer forward
        uses_redimension_vertical (bool): Si el modelo usa redimensión vertical
        uses_redimension_horizontal (bool): Si el modelo usa redimensión horizontal
        bin_umbral (float): Umbral de binarización
        val_dropout (float): Dropout a aplicar en prediccion
        times_pass_model (int): Cantidad de predicciones a combinar en prediccion
        type_combination (PredictionsCombinationType): Tipo de combinacion a aplicar sobre las predicciones

    Returns:
        list: Bounding boxes extraídas de la imagen
    """
    # Create model
    model = getPredictionModel(dataset_name=dataset_name)

    if val_dropout > 0:
        model.enable_eval_dropout()
        model.set_dropout_probability(dropout_probability=val_dropout)

    image = image.resize(utilsParameters.SAE_IMAGE_SIZE)

    transforms = DataLoader.get_transform()
    transformedImage = transforms(image)

    with T.no_grad():
        result = DataHelper.forwardToModel(model=model,
                                image=transformedImage,
                                times_pass_model=times_pass_model,
                                type_combination=type_combination
                                )

        boxes = ConnectedComponents.getConnectedComponents(result, bin_threshold_percentaje=bin_umbral)

        if uses_redimension_vertical or uses_redimension_horizontal:
            vResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_vertical   else 1
            hResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_horizontal else 1
            boxes = [DataLoader.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

    return boxes


def TFMTest(
    dataset_name: str, bin_umbral_for_model:float, dropout_value: float, val_dropout: float,
    times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType,
    votes_threshold: float,
    uses_redimension_vertical: float, uses_redimension_horizontal: float,
    save_test_info: bool, save_test_img: bool
    ):
    # Crear cache
    gc.collect()
    torch.cuda.empty_cache()

    # Generate path of the saved model
    sae_file = DataHelper.getModelFileName (
        dataset_name=dataset_name,
        dropout_value=dropout_value,
        uses_redimension_vertical=uses_redimension_vertical,
        uses_redimension_horizontal=uses_redimension_horizontal
        )

    # Get folder to obtain best bin threshold

    # val_best_bin_log_file = f'{folder_validation}/bestBin/{sae_file}.bestBin'
    # bin_umbral_for_model = 0
    # with open(val_best_bin_log_file, 'r') as file:
    #     bin_umbral_for_model = float(file.readline())

    folder_test = DataHelper.getLogsTestFolder (
        val_dropout=val_dropout,
        times_pass_model=times_pass_model,
        type_combination=type_combination
        )

    test_img_folder = DataHelper.getImgsTestFolder (
        val_dropout=val_dropout,
        times_pass_model=times_pass_model,
        type_combination=type_combination,
        dataset_name=dataset_name,
        model_name=sae_file
        )

    print(f'Testing model {sae_file} with Bin umbral {bin_umbral_for_model}')

    utilsIO.makeDirIfNotExist(folder_test)
    utilsIO.makeDirIfNotExist(test_img_folder)

    path_model = f'{utilsParameters.DRIVE_MODELS_FOLDER}/{sae_file}.pt'

    test_best_bin_log_file = f'{folder_test}/{sae_file}.test'

    # Create model
    model: SAE.SAE = torch.load(path_model, map_location=torch.device(utilsParameters.device))
    model.to(utilsParameters.device)
    model.eval()

    if val_dropout > 0:
        model.enable_eval_dropout()
        model.set_dropout_probability(dropout_probability=val_dropout)

    # Create datasets to train and val
    data_loader_test = DataHelper.generateTestDatasetLoader (
        dataset_name=dataset_name
        )


    # Evaluation
    bin_F1score_sum = 0
    bin_IoUscore_sum = 0

    with torch.no_grad():
        # Iterate over each example of the eval dataset
        for iteration, batch in enumerate(data_loader_test):
            if utilsParameters.DEBUG_FLAG:
                print(f'Eval with batch {iteration}/{len(data_loader_test)}')

            # Get the inputs and labels from the batch
            image, target = batch
            targetBoxes = target.squeeze().numpy().tolist()

            # Forward pass
            result = DataHelper.forwardToModel(model=model,
                                    image=image,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination
                                    )

            # Creates a dictionary with the bin thresholds and F1 score for each
            if utilsParameters.DEBUG_FLAG:
                print(f'\r\tTesting with {bin_umbral_for_model} binarization umbral', end='')

            # Extract BB from prediction
            boxes = ConnectedComponents.getConnectedComponents(result,
                                           bin_threshold_percentaje=bin_umbral_for_model,
                                           type_combination=type_combination,
                                           votes_threshold=votes_threshold
                                           )
            if uses_redimension_vertical or uses_redimension_horizontal:  # If we're using a resized BB, resize it to original
                vResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_vertical   else 1
                hResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_horizontal else 1
                boxes = [DataLoader.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

            # Calculate F1 and IoU
            f1, matched_ious = metrics.calculate_F1(y_true=targetBoxes, y_pred=boxes, iou_threshold=0.5)

            # Accumulate F1 and IoU score in the bin umbral used for this concrete example
            bin_F1score_sum  += f1
            bin_IoUscore_sum += np.mean(matched_ious) if len(matched_ious) > 0 else 0

            if save_test_img:
                drawing.drawBoxesPredictedAndGroundTruth (
                    tensor_image=image,
                    bboxes_predicted=boxes,
                    bboxes_groundtruth=targetBoxes,
                    is_normalized=False,
                    image_name=f'{test_img_folder}/{iteration}.png',
                    plot=False,
                    save=save_test_img
                    )

            if utilsParameters.DEBUG_FLAG:
                print()

            torch.cuda.empty_cache()
            # gc.collect()


    # Calculate mean of F1 and IoU scores
    number_elements = len(data_loader_test)

    bin_F1score_sum /= number_elements
    bin_IoUscore_sum /= number_elements

    print(f'Tested {sae_file}: \t Bin Umbral {bin_umbral_for_model}, Combination {type_combination.value}, Times {times_pass_model} --> F1 - {bin_F1score_sum} | IoU - {bin_IoUscore_sum}')

    # Save in a file the F1 and IoU metrics
    if save_test_info:
        with open(test_best_bin_log_file, 'w') as f:
            f.write(f'F1: {bin_F1score_sum}\nIoU: {bin_IoUscore_sum}\n Test model {sae_file} with Bin Umbral {bin_umbral_for_model}, Combination {type_combination.value}, Times {times_pass_model}')

