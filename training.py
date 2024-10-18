import gc
import torch
import DataHelper
import DataLoader
import utilsParameters
import utilsTraining
import SAE
import drawing
import utilsIO
import ConnectedComponents
import metrics
import numpy as np



def TFMTrainModel(dataset_name: str, dropout_value: float,
                  uses_redimension_vertical: bool, uses_redimension_horizontal: bool,
                  save_train_info: bool, plot_epochs: bool
                  ):
    # Crear cache
    gc.collect()
    torch.cuda.empty_cache()

    # Generate path to save model
    sae_file = DataHelper.getModelFileName(dataset_name=dataset_name, dropout_value=dropout_value, uses_redimension_vertical=uses_redimension_vertical, uses_redimension_horizontal=uses_redimension_horizontal)

    path_checkpoint = f'{utilsParameters.DRIVE_MODELS_FOLDER}/{sae_file}.pt'

    train_loss_log_folder = DataHelper.getLogsTrainingTrainFolder (
        dropout_value=dropout_value,
        uses_redimension_vertical=uses_redimension_vertical,
        uses_redimension_horizontal=uses_redimension_horizontal
        )
    val_loss_log_folder = DataHelper.getLogsTrainingValFolder (
        dropout_value=dropout_value,
        uses_redimension_vertical=uses_redimension_vertical,
        uses_redimension_horizontal=uses_redimension_horizontal
        )

    train_loss_log_file = f'{train_loss_log_folder}/{sae_file}.md'
    val_loss_log_file = f'{val_loss_log_folder}/{sae_file}.md'

    print(f'Training {sae_file}')

    if utilsIO.pathExist(val_loss_log_file) and utilsIO.pathExist(train_loss_log_file):
      print('SKIPPED')
      return

    utilsIO.makeDirIfNotExist(utilsParameters.DRIVE_MODELS_FOLDER)
    utilsIO.makeDirIfNotExist(train_loss_log_folder)
    utilsIO.makeDirIfNotExist(val_loss_log_folder)

    # Create model
    model = SAE.SAE()
    model.to(utilsParameters.device)

    model.set_dropout_probability(dropout_probability=dropout_value)

    # Create datasets to train and val
    data_loader_train = DataHelper.generateTrainDatasetLoader (
        dataset_name=dataset_name,
        uses_redimension_vertical=uses_redimension_vertical,
        uses_redimension_horizontal=uses_redimension_horizontal
        )
    data_loader_eval = DataHelper.generateValDatasetLoaderForTrain (
        dataset_name=dataset_name,
        uses_redimension_vertical=uses_redimension_vertical,
        uses_redimension_horizontal=uses_redimension_horizontal
        )

    # Set losses
    loss_function = torch.nn.MSELoss()
    val_loss_function = torch.nn.MSELoss()
    train_losses = []
    val_losses = []

    # Params and optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-5, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)



    # Callbacks
    modelCheckpoint = utilsTraining.ModelCheckpoint(path_checkpoint, model)
    earlyStopping = utilsTraining.EarlyStopping(patience=utilsParameters.TRAIN_PATIENTE, epochs=utilsParameters.TRAIN_NUM_EPOCHS, checkpoint=modelCheckpoint, epsilon=0.00001)

    # Num batches
    num_batches_train = len(data_loader_train)
    num_batches_val = len(data_loader_eval)


    for epoch in range(utilsParameters.TRAIN_NUM_EPOCHS):
        if utilsParameters.DEBUG_FLAG:
            print(f'Epoch: {epoch+1}/{utilsParameters.TRAIN_NUM_EPOCHS}')

        ### TRAIN ONE EPOCH
        # Set model to train
        model.train()

        epochTrainLoss = []
        for iteration, batch in enumerate(data_loader_train):
            image, info, targetImage = batch

            if utilsParameters.DEBUG_FLAG:
                print(f'\r\tTraining Batch {iteration+1} of {num_batches_train} ({info["name"]})', end='')

            # Get the inputs and labels from the batch
            image, targetImage = image.to(utilsParameters.device), targetImage.to(utilsParameters.device)

            # Forward pass
            reconstructed = model(image)
            loss = loss_function(reconstructed, targetImage)
            # reconstructed.detach(), image.detach(), targetImage.detach()

            # Backward pass and update parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Append training loss
            epochTrainLoss.append(loss.item())

        if utilsParameters.DEBUG_FLAG:
            print()

        # Optimizer step
        lr_scheduler.step()

        # clear GPU memory usage
        torch.cuda.empty_cache()
        gc.collect()

        ### VALIDATE ONE EPOCH
        # Validation
        model.eval()
        epochValLoss = []
        with torch.no_grad():
            for iteration, batch in enumerate(data_loader_eval):
                if utilsParameters.DEBUG_FLAG:
                    print(f'\r\tValidation Batch {iteration+1} of {num_batches_val}', end='')

                # Get the inputs and labels from the batch
                image, _, targetImage = batch
                image, targetImage = image.to(utilsParameters.device), targetImage.to(utilsParameters.device)

                # Forward pass
                reconstructed = model(image)

                # Get loss function
                valLoss = val_loss_function(reconstructed,  targetImage)

                # Save loss for single example
                epochValLoss.append(valLoss.item())

            if utilsParameters.DEBUG_FLAG:
                print()

        # Losses
        mean_train_loss = np.mean(epochTrainLoss)
        mean_val_loss = np.mean(epochValLoss)

        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)

        print(f'\tEpoch {epoch}: \t Train Loss - {round(mean_train_loss, 5)} | Val Loss - {round(mean_val_loss, 5)}')

        # Update EarlyStopping
        stopTraining = earlyStopping.update(mean_val_loss, epoch)

        # clear GPU memory usage
        torch.cuda.empty_cache()
        gc.collect()

        # If EarlyStopping says to stop, we stop training
        if stopTraining:
            print('Early Stopped!')
            break

    # Plot train val losses
    if plot_epochs:
        drawing.plotTrainval_losses(train_losses, val_losses)

    if save_train_info:
        # Save Log of the training
        with open(train_loss_log_file, 'w') as f:
            [f.write(f'{it}, {result}\n') for it, result in enumerate(train_losses)]

        with open(val_loss_log_file, 'w') as f:
            [f.write(f'{it}, {result}\n') for it, result in enumerate(val_losses)]


    return model


def TFMValidation(
    dataset_name: str, dropout_value: float, val_dropout: float,
    times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType,
    votes_threshold: float = 0.5,
    uses_redimension_vertical: bool = True, uses_redimension_horizontal: bool = True,
    save_val_info: bool = True, save_val_imgs: bool = False
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

    # Generate folder to save this validation process
    folder_validation = DataHelper.getLogsValidationFolder (
        val_dropout=val_dropout,
        times_pass_model=times_pass_model,
        type_combination=type_combination
        )

    utilsIO.makeDirIfNotExist(folder_validation)

    path_model = f'{utilsParameters.DRIVE_MODELS_FOLDER}/{sae_file}.pt'

    if save_val_info:
        val_best_bin_log_folder = f'{folder_validation}/bestBin'
        val_f1_log_folder = f'{folder_validation}/valF1'
        val_iou_log_folder = f'{folder_validation}/valIoU'

        val_best_bin_log_file = f'{val_best_bin_log_folder}/{sae_file}.bestBin'
        val_f1_log_file = f'{val_f1_log_folder}/{sae_file}.valF1'
        val_iou_log_file = f'{val_iou_log_folder}/{sae_file}.valIoU'

        utilsIO.makeDirIfNotExist(val_best_bin_log_folder)
        utilsIO.makeDirIfNotExist(val_f1_log_folder)
        utilsIO.makeDirIfNotExist(val_iou_log_folder)

    if save_val_imgs:
        val_img_folder = DataHelper.getImgsValidationFolder (
            val_dropout=val_dropout,
            times_pass_model=times_pass_model,
            type_combination=type_combination,
            dataset_name=dataset_name,
            model_name=sae_file
        )
        utilsIO.makeDirIfNotExist(val_img_folder)

    print(f'Evaluating {sae_file}')

    # Create model
    model: SAE = torch.load(path_model, map_location=torch.device(utilsParameters.device))
    model.to(utilsParameters.device)

    model.eval()
    if val_dropout > 0:
        model.enable_eval_dropout()
        model.set_dropout_probability(dropout_probability=val_dropout)

    # Create datasets to train and val
    data_loader_eval = DataHelper.generateValDatasetLoaderForTest (
        dataset_name=dataset_name
        )


    # Evaluation
    bin_F1score_map = {i: 0.0 for i in utilsParameters.BIN_UMBRALS}
    bin_IoUscore_map = {i: 0.0 for i in utilsParameters.BIN_UMBRALS}

    with torch.no_grad():
        # Iterate over each example of the eval dataset
        for iteration, batch in enumerate(data_loader_eval):
            if utilsParameters.DEBUG_FLAG:
                print(f'Eval with batch {iteration}/{len(data_loader_eval)}')

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
            for bin_umbral in utilsParameters.BIN_UMBRALS:
                if utilsParameters.DEBUG_FLAG:
                    print(f'\r\tTesting with {bin_umbral} binarization umbral', end='')

                # Extract BB from prediction
                boxes = ConnectedComponents.getConnectedComponents(result,
                                               bin_threshold_percentaje=bin_umbral,
                                               type_combination=type_combination,
                                               votes_threshold=votes_threshold
                                               )
                if uses_redimension_vertical or uses_redimension_horizontal:  # If we're using a resized BB, resize it to original
                    vResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_vertical   else 1
                    hResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_horizontal else 1
                    boxes = [DataLoader.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

                # Calculate F1 and IoU
                f1, matched_ious = metrics.calculate_F1(y_true = targetBoxes, y_pred = boxes, iou_threshold=0.5)

                # Accumulate F1 and IoU score in the bin umbral used for this concrete example
                bin_F1score_map[bin_umbral]  += f1
                bin_IoUscore_map[bin_umbral] += np.mean(matched_ious) if len(matched_ious) > 0 else 0

            if utilsParameters.DEBUG_FLAG:
                print()

            torch.cuda.empty_cache()
            # gc.collect()


    # Calculate mean of F1 and IoU scores
    number_elements = len(data_loader_eval)
    for key in bin_F1score_map:
        bin_F1score_map[key] /= number_elements

    for key in bin_IoUscore_map:
        bin_IoUscore_map[key] /= number_elements

    # Calculate mean of each bin theshold and decide max
    best_bin_threshold = max(bin_F1score_map, key=bin_F1score_map.get)

    # Check for dupes in best F1 score
    common_keys = []
    for key, value in bin_F1score_map.items():
        if value == bin_F1score_map[best_bin_threshold]:
            common_keys.append(key)

    # If there's dupes, select the one with most IoU
    if len(common_keys) > 1:
        for k in common_keys:
            best_bin_threshold = best_bin_threshold if bin_IoUscore_map[best_bin_threshold] > bin_IoUscore_map[k] else k


    # Create the "best umbral" info message
    stringBestMsg = f'Evaluated {sae_file}: \t Bin Threshold {best_bin_threshold} --> F1 - {bin_F1score_map[best_bin_threshold]} | IoU - {bin_IoUscore_map[best_bin_threshold]}'
    print(stringBestMsg)

    if save_val_imgs:
        images_data_loader_eval = DataHelper.generateValDatasetLoaderForTest (
            dataset_name=dataset_name
        )
        with torch.no_grad():
        # Iterate over each example of the eval dataset
            for iteration, batch in enumerate(images_data_loader_eval):
                image, target = batch
                targetBoxes = target.squeeze().numpy().tolist()
                result = DataHelper.forwardToModel(model=model,
                                    image=image,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination
                                    )
                boxes = ConnectedComponents.getConnectedComponents(result,
                                               bin_threshold_percentaje=best_bin_threshold,
                                               type_combination=type_combination,
                                               votes_threshold=votes_threshold
                                               )
                if uses_redimension_vertical or uses_redimension_horizontal:  # If we're using a resized BB, resize it to original
                    vResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_vertical   else 1
                    hResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_horizontal else 1
                    boxes = [DataLoader.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

                image_save_path = f'{val_img_folder}/{iteration}.png'
                print(f'Saving image in {image_save_path}')
                drawing.drawBoxesPredictedAndGroundTruth (
                    tensor_image=image,
                    bboxes_predicted=boxes,
                    bboxes_groundtruth=targetBoxes,
                    is_normalized=False,
                    image_name=image_save_path,
                    plot=False,
                    save=save_val_imgs
                    )

    # Save in a file the F1 and IoU metrics
    if save_val_info:
        with open(val_best_bin_log_file, 'w') as f:
            f.write(f'{best_bin_threshold}\n{stringBestMsg}')

        stringF1List = '\n'.join([f' {round(key, 2)} {value}' for key, value in bin_F1score_map.items()])
        with open(val_f1_log_file, 'w') as f:
            f.write(stringF1List)

        stringIoUList = '\n'.join([f' {round(key, 2)} {value}' for key, value in bin_IoUscore_map.items()])
        with open(val_iou_log_file, 'w') as f:
            f.write(stringIoUList)

    return best_bin_threshold