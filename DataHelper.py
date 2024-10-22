import utilsParameters
import DataLoaderOwn
import DataHolderOwn
import torch
from torch.utils.data import DataLoader

def getModelFileName(dataset_name: str, dropout_value: float, uses_redimension_vertical: bool, uses_redimension_horizontal: bool):
    sae_file = 'SAE'

    sae_file += f'_D{int(dropout_value * 10)}'

    if uses_redimension_vertical or uses_redimension_horizontal:
        sae_file += '_R'
        if uses_redimension_vertical:
            sae_file += 'V'
        if uses_redimension_horizontal:
            sae_file += 'H'

    return f'{sae_file}_{dataset_name}'

def getLogsTrainingTrainFolder(dropout_value: float, uses_redimension_vertical: bool, uses_redimension_horizontal: bool):
    r_folder = 'R'
    if uses_redimension_vertical:
        r_folder += 'V'
    if uses_redimension_horizontal:
        r_folder += 'H'
    return f'{utilsParameters.DRIVE_TRAIN_LOGS_FOLDER}/trainLoss/D{dropout_value}/{r_folder}'

def getLogsTrainingValFolder(dropout_value: float, uses_redimension_vertical: bool, uses_redimension_horizontal: bool):
    r_folder = 'R'
    if uses_redimension_vertical:
        r_folder += 'V'
    if uses_redimension_horizontal:
        r_folder += 'H'
    return f'{utilsParameters.DRIVE_TRAIN_LOGS_FOLDER}/valLoss/D{dropout_value}/{r_folder}'

def getLogsValidationFolder(val_dropout: float, times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType):
    return f'{utilsParameters.DRIVE_VAL_LOGS_FOLDER}/{type_combination.name}/D{val_dropout}/T{times_pass_model}'

def getLogsTestFolder(val_dropout: float, times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType):
    return f'{utilsParameters.DRIVE_TEST_LOGS_FOLDER}/{type_combination.name}/D{val_dropout}/T{times_pass_model}'

def getImgsValidationFolder(val_dropout: float, times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType, dataset_name: str, model_name: str):
    return f'{utilsParameters.DRIVE_VAL_IMG_FOLDER}/{type_combination.name}/D{val_dropout}/T{times_pass_model}/{dataset_name}/{model_name}'

def getImgsTestFolder(val_dropout: float, times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType, dataset_name_train: str, dataset_name_test: str, model_name: str):
    return f'{utilsParameters.DRIVE_TEST_IMG_FOLDER}/{type_combination.name}/D{val_dropout}/T{times_pass_model}/{dataset_name_train}/{dataset_name_test}/{model_name}'


def generateTrainDatasetLoader(dataset_name: str, uses_redimension_vertical: bool, uses_redimension_horizontal: bool):
    batch_size = 1

    # Create datasets to train and val
    datasetLoader = DataLoaderOwn.DatasetLoader(dataset_name)

    # Load the datasets
    train_json, train_img = datasetLoader.loadTrainPaths()


    dataset_train = DataHolderOwn.TrainMusicDataset(
        name=dataset_name,
        jsonPaths=train_json,
        imagesPaths=train_img,
        box_resize_vertical=uses_redimension_vertical,
        box_resize_horizontal=uses_redimension_horizontal,
        resize_shape=utilsParameters.SAE_IMAGE_SIZE,
        transforms=DataLoaderOwn.get_transform()
        )

    # Training dataset
    data_loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return data_loader_train

def generateValDatasetLoaderForTrain(dataset_name, uses_redimension_vertical, uses_redimension_horizontal):
    batch_size = 1

    # Create datasets to train and val
    datasetLoader = DataLoaderOwn.DatasetLoader(dataset_name)

    # Load the datasets
    val_json, val_img = datasetLoader.loadValPaths()


    dataset_eval  = DataHolderOwn.TrainMusicDataset(
        name=dataset_name,
        jsonPaths=val_json,
        imagesPaths=val_img,
        box_resize_vertical=uses_redimension_vertical,
        box_resize_horizontal=uses_redimension_horizontal,
        resize_shape=utilsParameters.SAE_IMAGE_SIZE,
        transforms=DataLoaderOwn.get_transform()
        )


    # Validation  dataset (no redimension in order to calculate metrics)
    data_loader_eval = DataLoader(
        dataset_eval, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return data_loader_eval

def generateValDatasetLoaderForTest(dataset_name):
    batch_size = 1

    # Create datasets to train and val
    datasetLoader = DataLoaderOwn.DatasetLoader(dataset_name)

    # Load the datasets
    val_json, val_img = datasetLoader.loadValPaths()


    dataset_eval  = DataHolderOwn.TestMusicDataset(
        name=dataset_name,
        jsonPaths=val_json,
        imagesPaths=val_img,
        resize_shape=utilsParameters.SAE_IMAGE_SIZE,
        transforms=DataLoaderOwn.get_transform()
        )


    # Validation  dataset (no redimension in order to calculate metrics)
    data_loader_eval = DataLoader(
        dataset_eval, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return data_loader_eval


def generateTestDatasetLoader(dataset_name):
    batch_size = 1

    # Create datasets to train and val
    datasetLoader = DataLoaderOwn.DatasetLoader(dataset_name)

    # Load the datasets
    test_json, test_img = datasetLoader.loadTestPaths()


    dataset_test  = DataHolderOwn.TestMusicDataset(
        name=dataset_name,
        jsonPaths=test_json,
        imagesPaths=test_img,
        resize_shape=utilsParameters.SAE_IMAGE_SIZE,
        transforms=DataLoaderOwn.get_transform())


    # Validation  dataset (no redimension in order to calculate metrics)
    data_loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return data_loader_test

def forwardToModel(model, image, times_pass_model, type_combination):
    if type_combination == utilsParameters.PredictionsCombinationType.MEAN:
        result = model(image.to(utilsParameters.device))
        for i in range(times_pass_model):
            print(f'\r\tForward passing with mean result {i+1}', end='')
            result += model(image.to(utilsParameters.device))
        result /= times_pass_model
        print()
        return result.cpu()

    elif type_combination == utilsParameters.PredictionsCombinationType.MAX:
        result = model(image.to(utilsParameters.device))
        for i in range(times_pass_model):
            print(f'\r\tForward passing with max result {i+1}', end='')
            result = torch.max(result, model(image.to(utilsParameters.device)))
        print()
        return result.cpu()

    elif type_combination == utilsParameters.PredictionsCombinationType.VOTES:
        print(f'\r\tForward passing with voting result {times_pass_model}')
        return [model(image.to(utilsParameters.device)).cpu() for _ in range(times_pass_model)]

    else:
        return model(image.to(utilsParameters.device)).cpu()

