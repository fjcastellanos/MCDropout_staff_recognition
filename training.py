import gc
import torch
import DataHelper
import DataLoaderOwn
import utilsParameters
import utilsTraining
from SAEModel import SAE
import drawing
import utilsIO
import ConnectedComponents
import metrics
import numpy as np
from mean_average_precision import MetricBuilder
from sklearn.metrics import f1_score, precision_score, recall_score


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
    utilsParameters.init()

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
    bin_precision_map = {i: 0.0 for i in utilsParameters.BIN_UMBRALS}
    bin_recall_map = {i: 0.0 for i in utilsParameters.BIN_UMBRALS}
    bin_IoUscore_map = {i: 0.0 for i in utilsParameters.BIN_UMBRALS}
    matched_ious_per_th = {}
    tp_per_th = {}
    fp_per_th = {}
    fn_per_th = {}
    total_gt_boxes = 0
    
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    dict_f1_individual_pages_per_threshold = {}
    list_results = []
    list_img_GTs = []
    with torch.no_grad():
        # Iterate over each example of the eval dataset
        
        idx_page = 0
        for iteration, batch in enumerate(data_loader_eval):
            if utilsParameters.DEBUG_FLAG:
                print(f'Eval with batch {iteration}/{len(data_loader_eval)}')

            # Get the inputs and labels from the batch
            image, target = batch
            targetBoxes = target.squeeze().numpy().tolist()
            total_gt_boxes += len(targetBoxes) 
            
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
            for bin_umbral in utilsParameters.BIN_UMBRALS:
                if utilsParameters.DEBUG_FLAG:
                    print(f'\r\tTesting with {bin_umbral} binarization umbral', end='')

                if bin_umbral not in matched_ious_per_th:
                    matched_ious_per_th[bin_umbral] = []
                
                if bin_umbral not in tp_per_th:
                    tp_per_th[bin_umbral] = 0
                    fp_per_th[bin_umbral] = 0
                    fn_per_th[bin_umbral] = 0
                
                # Extract BB from prediction
                boxes, scores = ConnectedComponents.getConnectedComponents(result,
                                               bin_threshold_percentaje=bin_umbral,
                                               type_combination=type_combination,
                                               votes_threshold=votes_threshold
                                               )
                if uses_redimension_vertical or uses_redimension_horizontal:  # If we're using a resized BB, resize it to original
                    vResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_vertical   else 1
                    hResize = utilsParameters.BBOX_REDIMENSIONED_RECOVER if uses_redimension_horizontal else 1
                    boxes = [DataLoaderOwn.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

                # Calculate F1 and IoU
                f1, matched_ious_img, true_positives, false_positives, false_negatives = metrics.calculate_F1(y_true = targetBoxes, y_pred = boxes, iou_threshold=0.5)
                
                if bin_umbral not in dict_f1_individual_pages_per_threshold:
                    dict_f1_individual_pages_per_threshold[bin_umbral] = []
                        
                dict_f1_individual_pages_per_threshold[bin_umbral].append((f1, matched_ious_img))
                
                for iou_bbox in matched_ious_img:
                    matched_ious_per_th[bin_umbral].append(iou_bbox)

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
                
                idx_page+=1
                
                
                tp_per_th[bin_umbral] += true_positives
                fp_per_th[bin_umbral] += false_positives
                fn_per_th[bin_umbral] += false_negatives
                    
                # Accumulate F1 and IoU score in the bin umbral used for this concrete example
                
            if utilsParameters.DEBUG_FLAG:
                print()

            torch.cuda.empty_cache()
            # gc.collect()
            
    for bin_umbral in utilsParameters.BIN_UMBRALS:
        bin_F1score_map[bin_umbral], bin_precision_map[bin_umbral], bin_recall_map[bin_umbral]  = metrics.getF1_from_TP_FP_FN(tp_per_th[bin_umbral], fp_per_th[bin_umbral], fn_per_th[bin_umbral])
        bin_IoUscore_map[bin_umbral] = np.mean(matched_ious_per_th[bin_umbral]) if len(matched_ious_per_th[bin_umbral]) > 0 else 0
        
    print ("F1 and IoU for each binarization threshold")
    for bin_umbral in utilsParameters.BIN_UMBRALS:
        print(str(bin_umbral) + ";" + str(bin_F1score_map[bin_umbral]) + ";" + str(bin_IoUscore_map[bin_umbral]))
    
    # Calculate mean of each bin theshold and decide max
    best_bin_threshold = max(bin_F1score_map, key=bin_F1score_map.get)

    idx_img = 0
    list_F1_pixelwise_mean = []
    list_prec_pixelwise_mean = []
    list_recall_pixelwise_mean = []
    list_IoU_pixelwise_mean = []
    
    list_F1_pixelwise_std = []
    list_prec_pixelwise_std = []
    list_recall_pixelwise_std = []
    list_IoU_pixelwise_std = []
    
    
    idx_img_doc = 1
    for list_results_doc in list_results:
        print(f'\nEvaluating image {idx_img_doc}')
        idx_img_doc+=1
        img_GT = list_img_GTs[idx_img]
        
        dictF1_Prec_Recall = evaluate_images(list_results_doc, img_GT, best_bin_threshold)
        f1_mean = np.mean(dictF1_Prec_Recall["f1"])
        f1_std = np.std(dictF1_Prec_Recall["f1"])
        
        prec_mean = np.mean(dictF1_Prec_Recall["precision"])
        prec_std = np.std(dictF1_Prec_Recall["precision"])
        
        recall_mean = np.mean(dictF1_Prec_Recall["recall"])
        recall_std = np.std(dictF1_Prec_Recall["recall"])
        
        IoU_mean = np.mean(dictF1_Prec_Recall["IoU"])
        IoU_std = np.std(dictF1_Prec_Recall["IoU"])
        
        list_F1_pixelwise_mean.append(f1_mean)
        list_F1_pixelwise_std.append(f1_std)
        list_prec_pixelwise_mean.append(prec_mean)
        list_prec_pixelwise_std.append(prec_std)
        list_recall_pixelwise_mean.append(recall_mean)
        list_recall_pixelwise_std.append(recall_std)
        list_IoU_pixelwise_mean.append(IoU_mean)
        list_IoU_pixelwise_std.append(IoU_std)
        
        idx_img += 1
        
    
    
        
    F1s_avg_indiv_alldocs = [item [0] for item in dict_f1_individual_pages_per_threshold[best_bin_threshold]]
    F1_avg_indiv = np.mean(F1s_avg_indiv_alldocs) if len(F1s_avg_indiv_alldocs) >0 else 0.
    F1_variance_indiv = np.std(F1s_avg_indiv_alldocs) if len(F1s_avg_indiv_alldocs) > 0 else 0.
    
    IoUs_avg_indiv_alldocs = [np.mean(item [1]) for item in dict_f1_individual_pages_per_threshold[best_bin_threshold]]
    IoU_avg_indiv = np.mean(IoUs_avg_indiv_alldocs) if len(IoUs_avg_indiv_alldocs) >0 else 0.
    IoU_variance_indiv = np.std(IoUs_avg_indiv_alldocs) if len(IoUs_avg_indiv_alldocs) > 0 else 0.
    
    
    str_F1_pixel = ""
    str_prec_pixel = ""
    str_recall_pixel = ""
    str_IoU_pixel = ""

    dict_confidence_mean_per_image = {}
    dict_confidence_std_per_image = {}
    
    list_results_numpy = []
    for results_image in list_results:
        emptylist=[]
        list_results_numpy.append(emptylist)
        for results_prediction in results_image:
            list_results_numpy[len(list_results_numpy)-1].append(results_prediction.numpy())
            
    
    for num_predictions in [2,5,25,75,250,500]:
        for results_image in list_results_numpy: #iteration of all images
            if num_predictions > len(results_image):
                continue
            confidence_matrix = np.mean(results_image[0:num_predictions], axis=0)
            std_matrix = np.std(results_image[0:num_predictions], axis=0)
            
            std_pixel_value = np.mean(std_matrix)
            confidence_value = np.mean(confidence_matrix)
            
            if num_predictions not in dict_confidence_mean_per_image:
                dict_confidence_mean_per_image[num_predictions] = []
                dict_confidence_std_per_image[num_predictions] = []
            dict_confidence_mean_per_image[num_predictions].append(confidence_value)
            dict_confidence_std_per_image[num_predictions].append(std_pixel_value)
        
    str_confidence_per_predictions = ""
    str_std_per_predictions = ""
    for num_predictions in [2,5,25,75,250,500]:
            if num_predictions > len(results_image):
                continue
            confidence_value_mean_all = np.mean(dict_confidence_mean_per_image[num_predictions])
            confidence_value_std_all = np.mean(dict_confidence_std_per_image[num_predictions])
            str_confidence_per_predictions += str(confidence_value_mean_all) + ";"
            str_std_per_predictions += str(confidence_value_std_all) + ";"
            
            
            F1_pixel_mean = np.mean(list_F1_pixelwise_mean[0:num_predictions])
            F1_pixel_std = np.mean(list_F1_pixelwise_std[0:num_predictions])
            prec_pixel_mean = np.mean(list_prec_pixelwise_mean[0:num_predictions])
            prec_pixel_std = np.mean(list_prec_pixelwise_std[0:num_predictions])
            recall_pixel_mean = np.mean(list_recall_pixelwise_mean[0:num_predictions])
            recall_pixel_std = np.mean(list_recall_pixelwise_std[0:num_predictions])
            IoU_pixel_mean = np.mean(list_IoU_pixelwise_mean[0:num_predictions])
            IoU_pixel_std = np.mean(list_IoU_pixelwise_std[0:num_predictions])
            
            str_F1_pixel += str(F1_pixel_mean)+";" + str(F1_pixel_std) + ";"
            str_prec_pixel += str(prec_pixel_mean)+";" + str(prec_pixel_std) + ";"
            str_recall_pixel += str(recall_pixel_mean)+";" + str(recall_pixel_std) + ";"
            str_IoU_pixel += str(IoU_pixel_mean)+";" + str(IoU_pixel_std) + ";"
    
    print ("Dataset;DropoutInference;STD@2;STD@5;STD@25;STD@75;STD@250;STD@500;CONF@2;CONF@5;CONF@25;CONF@75;CONF@250;CONF@500;F1-reg-page;F1-std-reg-page;IoU-reg-page;IoU-std-reg-page;F1-px-mean@2;F1-px-std@2;F1-px-mean@5;F1-px-std@5;F1-px-mean@25;F1-px-std@25;F1-px-mean@75;F1-px-std@75;F1-px-mean@250;F1-px-std@250;F1-px-mean@500;F1-px-std@500;Prec-px-mean@2;Prec-px-std@2;Prec-px-mean@5;Prec-px-std@5;Prec-px-mean@25;Prec-px-std@25;Prec-px-mean@75;Prec-px-std@75;Prec-px-mean@250;Prec-px-std@250;Prec-px-mean@500;Prec-px-std@500;Recall-px-mean@2;Recall-px-std@2;Recall-px-mean@5;Recall-px-std@5;Recall-px-mean@25;Recall-px-std@25;Recall-px-mean@75;Recall-px-std@75;Recall-px-mean@250;Recall-px-std@250;Recall-px-mean@500;Recall-px-std@500;IoU-px-mean@2;IoU-px-std@2;IoU-px-mean@5;IoU-px-std@5;IoU-px-mean@25;IoU-px-std@25;IoU-px-mean@75;IoU-px-std@75;IoU-px-mean@250;IoU-px-std@250;IoU-px-mean@500;IoU-px-std@500")
    print (f'{dataset_name};{val_dropout};{str_std_per_predictions}{str_confidence_per_predictions}{F1_avg_indiv};{F1_variance_indiv};{IoU_avg_indiv};{IoU_variance_indiv};{str_F1_pixel};{str_prec_pixel};{str_recall_pixel};{str_IoU_pixel}')
    # Check for dupes in best F1 score
    common_keys = []
    for key, value in bin_F1score_map.items():
        if value == bin_F1score_map[best_bin_threshold]:
            common_keys.append(key)


    mAP = metric_fn.value(iou_thresholds=0.5)['mAP']
    COCO = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    print("mAP:" + str(mAP))
    print(f"COCO mAP: " + str(COCO))
    
    
    # If there's dupes, select the one with most IoU
    if len(common_keys) > 1:
        for k in common_keys:
            best_bin_threshold = best_bin_threshold if bin_IoUscore_map[best_bin_threshold] > bin_IoUscore_map[k] else k

    
    tp_norm = tp_per_th[best_bin_threshold] / total_gt_boxes
    fp_norm = fp_per_th[best_bin_threshold] / (tp_per_th[best_bin_threshold] + fp_per_th[best_bin_threshold]) if ((tp_per_th[best_bin_threshold] + fp_per_th[best_bin_threshold]))> 0 else 0.
    fn_norm = fn_per_th[best_bin_threshold] / total_gt_boxes
    
    # Create the "best umbral" info message
    stringBestMsg = f'Validation {sae_file}: \t Bin Threshold {best_bin_threshold} --> F1 - {bin_F1score_map[best_bin_threshold]} | IoU - {bin_IoUscore_map[best_bin_threshold]}  | Prec - {bin_precision_map[best_bin_threshold]}  | Recall - {bin_recall_map[best_bin_threshold]} | TP - {tp_per_th[best_bin_threshold]}  | FP - {fp_per_th[best_bin_threshold]}  | FN - {fn_per_th[best_bin_threshold]} | TP-norm - {tp_norm}  | FP-norm - {fp_norm}  | FN-norm - {fn_norm}'    
    print(stringBestMsg)

    resize_str = ""
    if uses_redimension_vertical:
        resize_str += 'V'
    if uses_redimension_horizontal:
        resize_str += 'H'
        
    
    
    logs = "Source;Target;Partition;model;Resize;dropoutTrain;dropoutVal;Repetitions;Combination;VoteTH;BinTH;TotalGTBoxes;F1;IoU;Prec;Recall;TP;FP;FN;TP-norm;FP-norm;FN-norm;mAP;COCO;F1_avg_indiv;F1_variance\n"
    logs += f'{dataset_name};{dataset_name};val;{sae_file};{resize_str};{dropout_value};{val_dropout};{times_pass_model};{type_combination.value};{votes_threshold};{best_bin_threshold};{total_gt_boxes};{bin_F1score_map[best_bin_threshold]};{bin_IoUscore_map[best_bin_threshold]};{bin_precision_map[best_bin_threshold]};{bin_recall_map[best_bin_threshold]};{tp_per_th[best_bin_threshold]};{fp_per_th[best_bin_threshold]};{fn_per_th[best_bin_threshold]};{tp_norm};{fp_norm};{fn_norm};{mAP};{COCO};{F1_avg_indiv};{F1_variance_indiv}'    
    
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
                    boxes = [DataLoaderOwn.resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]

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

    return best_bin_threshold, logs


def calculate_iou_pixelwise(pred_mask, gt_mask):
    # Asegurarse de que ambas máscaras tengan el mismo tamaño
    assert pred_mask.shape == gt_mask.shape, "Las máscaras deben tener el mismo tamaño"
    
    # Calcular la intersección y la unión
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calcular IoU
    iou = intersection / union if union != 0 else 0

    iou_value = iou.item if isinstance(iou, torch.Tensor) else iou
    return iou_value

def evaluate_images(pred_images, gt, bin_threshold):
    f1_scores = []
    precisions = []
    recalls = []
    IoU_scores = []
    gt_int = gt>0.5*1
    gt_flat = gt_int.flatten()
    idx_page = 1
    print("")
    for pred in pred_images:
        print(f'\r\tEvaluating prediction {idx_page}', end='')
        pred_np = pred.numpy()
        idx_page+=1
        pred_flat = (pred_np.flatten() > bin_threshold)*1
        
        # Calcular precisión, recall y F1-score para cada par de imágenes
        precision = precision_score(gt_flat, pred_flat, average='binary', zero_division=0)
        recall = recall_score(gt_flat, pred_flat, average='binary', zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, average='binary', zero_division=0)
        IoU = calculate_iou_pixelwise(pred_flat, gt_flat)
        
        # Guardar los resultados
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        IoU_scores.append(IoU)
        
    return {
        "precision": precisions,
        "recall": recalls,
        "f1": f1_scores,
        "IoU": IoU_scores
    }
    
    
