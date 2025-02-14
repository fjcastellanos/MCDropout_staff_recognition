import torch
import utilsParameters
import DataLoaderOwn
import DataHelper
import ConnectedComponents
import utilsIO
import metrics
import SAEModel
import drawing

import gc
from PIL import  Image as PILImage
import torchvision.transforms as T
import numpy as np
from mean_average_precision import MetricBuilder


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

    transforms = DataLoaderOwn.get_transform()
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
            boxes = [DataLoaderOwn.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

    return boxes


def TFMTest(
    dataset_name_train: str, dataset_name_test: str, bin_umbral_for_model:float, dropout_value: float, val_dropout: float,
    times_pass_model: int, type_combination: utilsParameters.PredictionsCombinationType,
    votes_threshold: float,
    uses_redimension_vertical: float, uses_redimension_horizontal: float,
    save_test_info: bool, save_test_img: bool,
    seed_value: int
    ):
    # Crear cache
    gc.collect()
    torch.cuda.empty_cache()
    utilsParameters.init(seed_value)

    # Generate path of the saved model
    sae_file = DataHelper.getModelFileName (
        dataset_name=dataset_name_train,
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
        dataset_name_train=dataset_name_train,
        dataset_name_test=dataset_name_test,
        model_name=sae_file
        )

    print(f'Testing model {sae_file} with Bin umbral {bin_umbral_for_model}')

    utilsIO.makeDirIfNotExist(folder_test)
    utilsIO.makeDirIfNotExist(test_img_folder)

    path_model = f'{utilsParameters.DRIVE_MODELS_FOLDER}/{sae_file}.pt'

    test_best_bin_log_file = f'{folder_test}/{sae_file}.test'

    # Create model
    model: SAEModel.SAE = torch.load(path_model, map_location=torch.device(utilsParameters.device))
    model.to(utilsParameters.device)
    model.eval()

    if val_dropout > 0:
        model.enable_eval_dropout()
        model.set_dropout_probability(dropout_probability=val_dropout)

    # Create datasets to train and val
    data_loader_test = DataHelper.generateTestDatasetLoader (
        dataset_name=dataset_name_test
        )


    # Evaluation
    bin_F1score_sum = 0
    bin_precision_sum = 0
    bin_recall_sum = 0
    bin_IoUscore_sum = 0
    matched_ious = []
    tp = 0
    fp = 0
    fn = 0
    total_gt_boxes = 0
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    list_f1_individual_pages = []
    list_img_GTs = []
    list_images = []

    list_results = []
    with torch.no_grad():
        # Iterate over each example of the eval dataset
        for iteration, batch in enumerate(data_loader_test):
            if utilsParameters.DEBUG_FLAG:
                print(f'Eval with batch {iteration}/{len(data_loader_test)}')

            # Get the inputs and labels from the batch
            image, target = batch
            targetBoxes = target.squeeze().numpy().tolist()
            total_gt_boxes += len(targetBoxes) 
            list_images.append(image)

            img_GT = np.zeros((image.shape))
            vResize = utilsParameters.BBOX_REDIMENSION if uses_redimension_vertical   else 1
            hResize = utilsParameters.BBOX_REDIMENSION if uses_redimension_horizontal else 1
            bboxes_GT_resized = [DataLoaderOwn.resize_box(box, vResize=vResize, hResize=hResize) for box in targetBoxes]
            for bbox_GT in bboxes_GT_resized:
                img_GT[0,0,int(bbox_GT[1]):int(bbox_GT[3]),int(bbox_GT[0]):int(bbox_GT[2])] = 1
            
            list_img_GTs.append(img_GT)

            # Forward pass
            result = DataHelper.forwardToModel(model=model,
                                    image=image,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination
                                    )
            list_results.append(result)
            # Creates a dictionary with the bin thresholds and F1 score for each
            if utilsParameters.DEBUG_FLAG:
                print(f'\r\tTesting with {bin_umbral_for_model} binarization umbral', end='')

            # Extract BB from prediction
            boxes, scores = ConnectedComponents.getConnectedComponents(result,
                                           bin_threshold_percentaje=bin_umbral_for_model,
                                           type_combination=type_combination,
                                           votes_threshold=votes_threshold
                                           )
            if uses_redimension_vertical or uses_redimension_horizontal:  # If we're using a resized BB, resize it to original
                vResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_vertical   else 1
                hResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_horizontal else 1
                boxes = [DataLoaderOwn.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

            # Calculate F1 and IoU
            f1, matched_ious_img, true_positives, false_positives, false_negatives = metrics.calculate_F1(y_true=targetBoxes, y_pred=boxes, iou_threshold=0.5)
            
            list_f1_individual_pages.append(f1)
            for iou_bbox in matched_ious_img:
                matched_ious.append(iou_bbox)
            
            tp += true_positives
            fp += false_positives
            fn += false_negatives
            
            idx_box_in_page = 0           
            detections = []     
            annotations = []
            for box in boxes:
                score = scores[idx_box_in_page]
                idx_box_in_page+=1
                detections.append([box[0],box[1],box[2],box[3], 0, score])
            for box in targetBoxes:
                # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                annotations.append([box[0],box[1],box[2],box[3], 0, 0, 0])
            
            metric_fn.add(np.array(detections), np.array(annotations))
                
                
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


    list_results_numpy = []
    for results_image in list_results:
        emptylist=[]
        list_results_numpy.append(emptylist)
        for results_prediction in results_image:
            list_results_numpy[len(list_results_numpy)-1].append(results_prediction.numpy())
            
    

    page = 5
    num_pred = 2
    
    gt = list_img_GTs[page][0,0,:,:]
    ROWS = list_results_numpy[0][0].shape[2]
    COLS = list_results_numpy[0][0].shape[3]  

    result = np.zeros((ROWS, COLS))
    import math
    print("Dumping results to a matrix...")
    for r in range(ROWS):
        print ("Average of the results..." + str(r) + "/" + str(ROWS))
        
        for c in range(COLS):
            list_predictions_pixel = []
            for prediction in list_results_numpy[page][0:num_pred]:
                prediction_th = (prediction[0,0,r,c] >=bin_umbral_for_model) *1
                if gt[r,c] == 1:
                    value = 1 - prediction_th
                else:
                    value = prediction_th

                list_predictions_pixel.append(value)
            
            value = np.count_nonzero(list_predictions_pixel)
            result[r, c] = value

    from matplotlib import pyplot as plt
    plt.imshow(result, cmap='Reds', vmin=np.amin(result), vmax=np.amax(result))
    plt.axis('off')  # Opcional: eliminar los ejes
    plt.savefig('pruebas/100preds_nobin.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    #import cv2
    #cv2.imwrite("pruebas/prueba-6.png", list_results_numpy[page][6][0,0,:,:]*255)

    # Calculate mean of F1 and IoU scores
    bin_F1score_sum, bin_precision_sum, bin_recall_sum= metrics.getF1_from_TP_FP_FN(tp,fp,fn)
    bin_IoUscore_sum += np.mean(matched_ious) if len(matched_ious) > 0 else 0

    mAP = metric_fn.value(iou_thresholds=0.5)['mAP']
    COCO = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    print("mAP:" + str(mAP))
    print(f"COCO mAP: " + str(COCO))
    

    tp_norm = tp / total_gt_boxes if total_gt_boxes > 0 else 0.
    fp_norm = fp / (tp + fp) if (tp + fp) > 0 else 0.
    fn_norm = fn / total_gt_boxes if total_gt_boxes > 0 else 0. 
    
    # Create the "best umbral" info message
    stringBestMsg = f'Test {sae_file}: \t Bin Threshold {bin_umbral_for_model}, Combination {type_combination.value}, Times {times_pass_model} --> F1 - {bin_F1score_sum} | IoU - {bin_IoUscore_sum}  | Prec - {bin_precision_sum}  | Recall - {bin_recall_sum} | TP - {tp}  | FP - {fp}  | FN - {fn} | TP-norm - {tp_norm}  | FP-norm - {fp_norm}  | FN-norm - {fn_norm}'    
    print(stringBestMsg)
    
    resize_str = ""
    if uses_redimension_vertical:
        resize_str += 'V'
    if uses_redimension_horizontal:
        resize_str += 'H'
        
    F1_avg_indiv = np.mean(list_f1_individual_pages) if len(list_f1_individual_pages) >0 else 0.
    F1_variance_indiv = np.std(list_f1_individual_pages) if len(list_f1_individual_pages) > 0 else 0.
    
    
    logs = "Source;Target;Partition;model;Resize;dropoutTrain;dropoutVal;Repetitions;Combination;VoteTH;BinTH;TotalGTBoxes;F1;IoU;Prec;Recall;TP;FP;FN;TP-norm;FP-norm;FN-norm;mAP;COCO\n"
    logs += f'{dataset_name_train};{dataset_name_test};test;{sae_file};{resize_str};{dropout_value};{val_dropout};{times_pass_model};{type_combination.value};{votes_threshold};{bin_umbral_for_model};{total_gt_boxes};{bin_F1score_sum};{bin_IoUscore_sum};{bin_precision_sum};{bin_recall_sum};{tp};{fp};{fn};{tp_norm};{fp_norm};{fn_norm};{mAP};{COCO};{F1_avg_indiv};{F1_variance_indiv}'    
    
    
    
    # Save in a file the F1 and IoU metrics
    if save_test_info:
        with open(test_best_bin_log_file, 'w') as f:
            f.write(f'F1: {bin_F1score_sum}\nIoU: {bin_IoUscore_sum}\n Test model {sae_file} with Bin Umbral {bin_umbral_for_model}, Combination {type_combination.value}, Times {times_pass_model}')

    return logs