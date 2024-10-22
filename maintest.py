
from SAEModel import SAE
import training
import inference
import utilsParameters

import argparse


def menu():
    parser = argparse.ArgumentParser(description='MonteCarlo dropout for staff retrieval')
    
    parser.add_argument('-db_train', required=True, choices=utilsParameters.DATASETS, help='Dataset name for training and validation')
    parser.add_argument('-db_test',  required=True, choices=utilsParameters.DATASETS, help='Dataset name for testing')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


def saveLogs(logs, filepath):
    with open(filepath, 'w') as myfile:
        myfile.write(logs)


def run_Base():
    # Models parameters
    dropout_value_list = [0]
    save_val_info = True
    save_val_imgs = True
    val_dropout = 0
    times_pass_model = 1
    type_combination = utilsParameters.PredictionsCombinationType.NONE

                                    # Mod1,  Mod2,  Mod3,  Mod4
    usesRedimensionVerticalList   = [False, False, True , True]
    usesRedimensionHorizontalList = [False, True , False, True]
    
    bin_th_list = []
    
    logs = ""

    for uses_redimension_vertical, uses_redimension_horizontal in zip(usesRedimensionVerticalList, usesRedimensionHorizontalList):
        for dataset_name in utilsParameters.DATASETS:
            for dropout_value in dropout_value_list:
                bin_th, logs_experiment = training.TFMValidation(dataset_name=dataset_name,
                            dropout_value=dropout_value,
                            val_dropout=val_dropout,
                            times_pass_model=times_pass_model,
                            type_combination=type_combination,
                            uses_redimension_vertical=uses_redimension_vertical,
                            uses_redimension_horizontal=uses_redimension_horizontal,
                            save_val_info=save_val_info,
                            save_val_imgs=save_val_imgs
                            )
                bin_th_list.append(bin_th)
                if logs == "":
                    logs = logs_experiment + "\n"
                else:
                    logs += logs_experiment.split("\n")[1] + "\n"


    
    
    #test
    # Models parameters
    dropout_value = 0
    save_test_info = True
    save_test_img = True
    val_dropout = 0
    times_pass_model = 1
    type_combination = utilsParameters.PredictionsCombinationType.NONE

                                    # Mod1,  Mod2,  Mod3,  Mod4
    usesRedimensionVerticalList   = [False, False, True , True]
    usesRedimensionHorizontalList = [False, True , False, True]
    
    idx_experiment = 0


    for uses_redimension_vertical, uses_redimension_horizontal in zip(usesRedimensionVerticalList, usesRedimensionHorizontalList):
        for dataset_name in utilsParameters.DATASETS:
            bin_th = bin_th_list[idx_experiment]
            logs_experiment = inference.TFMTest(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        val_dropout=val_dropout,
                        times_pass_model=times_pass_model,
                        type_combination=type_combination,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_test_info=save_test_info,
                        save_test_img=save_test_img,
                        bin_umbral_for_model=bin_th,
                        votes_threshold=0.5
                        )
            idx_experiment = idx_experiment+1
            if logs == "":
                logs = logs_experiment + "\n"
            else:
                logs += logs_experiment.split("\n")[1] + "\n"
    saveLogs(logs, "results_run_Base.txt")
    
    
    
def run_DropoutTrainBase():
    #dropout con entrenamiento
    
    logs = ""
    #umbral
    # Models parameters
    save_val_info = True
    save_val_imgs = True
    val_dropout = 0
    times_pass_model = 1
    type_combination = utilsParameters.PredictionsCombinationType.NONE
    votes_threshold = 0

    uses_redimension_vertical = True
    uses_redimension_horizontal = True
    
    bin_th_list = []

    for dropout_value in utilsParameters.DROPOUT_VALUES:
        for dataset_name in utilsParameters.DATASETS:
            bin_th, logs_experiment = training.TFMValidation(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        val_dropout=val_dropout,
                        times_pass_model=times_pass_model,
                        type_combination=type_combination,
                        votes_threshold=votes_threshold,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_val_info=save_val_info,
                        save_val_imgs=save_val_imgs
                        )
            bin_th_list.append(bin_th)
            if logs == "":
                logs = logs_experiment + "\n"
            else:
                logs += logs_experiment.split("\n")[1] + "\n"
                
    idx_experiment = 0
    save_test_img = True

    for dropout_value in utilsParameters.DROPOUT_VALUES:
        for dataset_name in utilsParameters.DATASETS:
            bin_th = bin_th_list[idx_experiment]
            logs_experiment = inference.TFMTest(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        val_dropout=val_dropout,
                        times_pass_model=times_pass_model,
                        type_combination=type_combination,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_test_info=save_test_info,
                        save_test_img=save_test_img,
                        bin_umbral_for_model=bin_th,
                        votes_threshold=0.5
                        )
            idx_experiment = idx_experiment+1
            if logs == "":
                logs = logs_experiment + "\n"
            else:
                logs += logs_experiment.split("\n")[1] + "\n"
                
    saveLogs(logs, "results_run_DropoutTrainBase.txt")
    
    
def run_DropoutCombination(config):
    #dropout en test
    logs = ""
    # Models parameters
    save_val_info = False
    save_val_imgs = False
    save_test_img = True
    save_test_info = False

    uses_redimension_vertical = True
    uses_redimension_horizontal = True
    
    MODELS_TO_TEST = [
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_CAPITAN,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.3,
            val_dropout=[.1,.2,.3,.4,.5],
            times_pass_model= [1,2,5,10,25,50,75,100,200,300,400,500,750,1000],
            type_combination=[utilsParameters.PredictionsCombinationType.MEAN, utilsParameters.PredictionsCombinationType.MAX, utilsParameters.PredictionsCombinationType.VOTES]
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            val_dropout=[.1,.2,.3,.4,.5],
            times_pass_model= [1,2,5,10,25,50,75,100,200,300,400,500,750,1000],
            type_combination=[utilsParameters.PredictionsCombinationType.MEAN, utilsParameters.PredictionsCombinationType.MAX, utilsParameters.PredictionsCombinationType.VOTES]
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            val_dropout=[.1,.2,.3,.4,.5],
            times_pass_model= [1,2,5,10,25,50,75,100,200,300,400,500,750,1000],
            type_combination=[utilsParameters.PredictionsCombinationType.MEAN, utilsParameters.PredictionsCombinationType.MAX, utilsParameters.PredictionsCombinationType.VOTES]
            )
    ]
    
    votes_threshold_list = [.25, .5, .75]
    bin_th_list = []
    idx_experiment = 0

    for TEST_PARAMETER in MODELS_TO_TEST:
        if config.db_train is not None and TEST_PARAMETER.dataset_name != config.db_train:
            continue
        for type_combination in TEST_PARAMETER.type_combination:
            for times_pass_model in TEST_PARAMETER.times_pass_model:
                for val_dropout_item in TEST_PARAMETER.val_dropout:
                    if type_combination == utilsParameters.PredictionsCombinationType.VOTES:
                        for votes_threshold in votes_threshold_list:
                            bin_th, logs_experiment = training.TFMValidation(dataset_name=TEST_PARAMETER.dataset_name,
                                    dropout_value=TEST_PARAMETER.train_dropout,
                                    val_dropout=val_dropout_item,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination,
                                    votes_threshold=votes_threshold,
                                    uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                    uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                    save_val_info=save_val_info,
                                    save_val_imgs=save_val_imgs
                                )
                            bin_th_list.append(bin_th)
                            if logs == "":
                                logs = logs_experiment + "\n"
                            else:
                                logs += logs_experiment.split("\n")[1] + "\n"

                            logs_experiment = inference.TFMTest(dataset_name_train=TEST_PARAMETER.dataset_name,
                                                                dataset_name_test=config.db_test,
                                    dropout_value=TEST_PARAMETER.train_dropout,
                                    val_dropout=val_dropout,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination,
                                    uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                    uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                    save_test_info=save_test_info,
                                    save_test_img=save_test_img,
                                    bin_umbral_for_model=bin_th,
                                    votes_threshold=votes_threshold
                                    )
                            logs += logs_experiment.split("\n")[1] + "\n"
                    else:
                        bin_th, logs_experiment = training.TFMValidation(dataset_name=TEST_PARAMETER.dataset_name,
                                    dropout_value=TEST_PARAMETER.train_dropout,
                                    val_dropout=val_dropout_item,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination,
                                    votes_threshold=0.,
                                    uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                    uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                    save_val_info=save_val_info,
                                    save_val_imgs=save_val_imgs
                                )
                        
                        bin_th_list.append(bin_th)
                        if logs == "":
                            logs = logs_experiment + "\n"
                        else:
                            logs += logs_experiment.split("\n")[1] + "\n"
                        logs_experiment = inference.TFMTest(dataset_name_train=TEST_PARAMETER.dataset_name,
                                                            dataset_name_test=config.db_test,
                                dropout_value=TEST_PARAMETER.train_dropout,
                                val_dropout=val_dropout_item,
                                times_pass_model=times_pass_model,
                                type_combination=type_combination,
                                uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                save_test_info=save_test_info,
                                save_test_img=save_test_img,
                                bin_umbral_for_model=bin_th,
                                votes_threshold=0
                                )
                        logs += logs_experiment.split("\n")[1] + "\n"
                    print("Experiment " + str(idx_experiment) + " finished")
                    idx_experiment = idx_experiment+1
                    saveLogs(logs, "results_run_DropoutCombination.txt")

    saveLogs(logs, "results_run_DropoutCombination.txt")
    
        
if __name__ == '__main__':
    
    config = menu()
    print (config)

    #run_Base()
    #run_DropoutTrainBase()
    
    run_DropoutCombination(config)

        
    
    
    
    #Fijar método de combinacion
    # Models parameters
    save_val_info = True
    save_val_imgs = True

    uses_redimension_vertical = True
    uses_redimension_horizontal = True

    MODELS_TO_TEST = [
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_CAPITAN,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.3,
            type_combination=utilsParameters.PredictionsCombinationType.VOTES,
            times_pass_model=63
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            type_combination=utilsParameters.PredictionsCombinationType.MEAN,
            times_pass_model=63
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            type_combination=utilsParameters.PredictionsCombinationType.VOTES,
            times_pass_model=15
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            type_combination=utilsParameters.PredictionsCombinationType.MEAN,
            times_pass_model=31
            )
    ]


    saveLogs(logs, "results.txt")
    votes_threshold = 0.5


    for TEST_PARAMETER in MODELS_TO_TEST:
        for val_dropout in utilsParameters.DROPOUT_VALUES:
            if val_dropout != TEST_PARAMETER.train_dropout:
                bin_th, logs_experiment = training.TFMValidation(dataset_name=TEST_PARAMETER.dataset_name,
                            dropout_value=TEST_PARAMETER.train_dropout,
                            val_dropout=val_dropout,
                            times_pass_model=TEST_PARAMETER.times_pass_model,
                            type_combination=TEST_PARAMETER.type_combination,
                            votes_threshold=votes_threshold,
                            uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                            uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                            save_val_info=save_val_info,
                            save_val_imgs=save_val_imgs
                        )   
                bin_th_list.append(bin_th)
                if logs == "":
                    logs = logs_experiment + "\n"
                else:
                    logs += logs_experiment.split("\n")[1] + "\n"




    #FINAL TEST
                
    FINAL_TEST_MODELS = [
        # Base Models
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_CAPITAN,
            uses_redimension_horizontal=False,
            uses_redimension_vertical=False,
            bin_umbral=0.65,
            train_dropout=0,
            val_dropout=0,
            type_combination=utilsParameters.PredictionsCombinationType.NONE,
            times_pass_model=1
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=False,
            uses_redimension_vertical=False,
            bin_umbral=0.8,
            train_dropout=0,
            val_dropout=0,
            type_combination=utilsParameters.PredictionsCombinationType.NONE,
            times_pass_model=1
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=False,
            uses_redimension_vertical=False,
            bin_umbral=0.75,
            train_dropout=0,
            val_dropout=0,
            type_combination=utilsParameters.PredictionsCombinationType.NONE,
            times_pass_model=1
            ),

        # Trained with Dropout but no val Dropout
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_CAPITAN,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.80,
            train_dropout=0.3,
            val_dropout=0,
            type_combination=utilsParameters.PredictionsCombinationType.NONE,
            times_pass_model=1
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.70,
            train_dropout=0.2,
            val_dropout=0,
            type_combination=utilsParameters.PredictionsCombinationType.NONE,
            times_pass_model=1
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.60,
            train_dropout=0.2,
            val_dropout=0,
            type_combination=utilsParameters.PredictionsCombinationType.NONE,
            times_pass_model=1
            ),

        # Trained with Dropout and val Dropout
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_CAPITAN,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.45,
            train_dropout=0.3,
            val_dropout=0.3,
            type_combination=utilsParameters.PredictionsCombinationType.VOTES,
            times_pass_model=63
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.60,
            train_dropout=0.2,
            val_dropout=0.1,
            type_combination=utilsParameters.PredictionsCombinationType.VOTES,
            times_pass_model=31
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.50,
            train_dropout=0.2,
            val_dropout=0.2,
            type_combination=utilsParameters.PredictionsCombinationType.MEAN,
            times_pass_model=31
            ),
    ]

    FINAL_TEST_MODELS = [
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=True,
            uses_redimension_vertical=True,
            bin_umbral=0.60,
            train_dropout=0.2,
            val_dropout=0.1,
            type_combination=utilsParameters.PredictionsCombinationType.VOTES,
            times_pass_model=31
            ),
    ]

    save_test_info = False
    save_test_imgs = True

    votes_threshold = 0.5

    for TEST_PARAMETER in FINAL_TEST_MODELS:
                inference.TFMTest(dataset_name=TEST_PARAMETER.dataset_name,
                        bin_umbral_for_model=TEST_PARAMETER.bin_umbral,
                        dropout_value=TEST_PARAMETER.train_dropout,
                        val_dropout=TEST_PARAMETER.train_dropout,
                        times_pass_model=TEST_PARAMETER.times_pass_model,
                        type_combination=TEST_PARAMETER.type_combination,
                        votes_threshold=votes_threshold,
                        uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                        uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                        save_test_info=save_test_info,
                        save_test_img=save_test_imgs
                    )
                
                
    saveLogs(logs, "results.txt")
    
    
    pass

