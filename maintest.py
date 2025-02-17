
from SAEModel import SAE
import training
import inference
import utilsParameters

import argparse
import os

def menu():
    parser = argparse.ArgumentParser(description='MonteCarlo dropout for staff retrieval')
    
    parser.add_argument('-db_train', required=True, choices=utilsParameters.DATASETS, help='Dataset name for training and validation')
    parser.add_argument('-db_test',  nargs='*', required=True, choices=utilsParameters.DATASETS, help='Dataset name for testing')
    parser.add_argument('-seed', default=145, type=int, help='Seed for reproducibility')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args
0

def saveLogs(logs, filepath):
    # Crear las carpetas si no existen
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as myfile:
        myfile.write(logs)


def run_Base(config):
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
            if dataset_name == utilsParameters.DATASET_CAPITAN:
                val_dropout = 0.3
            elif dataset_name == utilsParameters.DATASET_SEILS:
                val_dropout = 0.2
            elif dataset_name == utilsParameters.DATASET_FMT_C:
                val_dropout = 0.2
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

    saveLogs(logs, "results_run_Base_VAL_DROPOUT.txt")
    
    
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
                        votes_threshold=0.5,
                        seed_value = config.seed
                        )
            idx_experiment = idx_experiment+1
            if logs == "":
                logs = logs_experiment + "\n"
            else:
                logs += logs_experiment.split("\n")[1] + "\n"
    saveLogs(logs, "results/seed_" + str(config.seed) + "/results_run_Base.txt")
    
    
    
def run_DropoutTrainBase(config):
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
    save_test_info = True

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
                        votes_threshold=0.5,
                        seed_value = config.seed
                        )
            idx_experiment = idx_experiment+1
            if logs == "":
                logs = logs_experiment + "\n"
            else:
                logs += logs_experiment.split("\n")[1] + "\n"
                
    saveLogs(logs, "results/seed_" + str(config.seed) + "/results_run_DropoutTrainBase.txt")
    
    
def run_DropoutCombination(config):
    #dropout en test
    logs = ""
    # Models parameters
    save_val_info = False
    save_val_imgs = False
    save_test_img = False
    save_test_info = False
    
    path_results = "results/seed_" + str(config.seed) + "/results_run_DropoutCombination_" + config.db_train + "_color_temperature_matrix.txt"

    uses_redimension_vertical = True
    uses_redimension_horizontal = True
    all_combinations_experiment = [utilsParameters.PredictionsCombinationType.VOTES]
    num_repetitions_experiment = [75]
    MODELS_TO_TEST = [
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_CAPITAN,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.3,
            val_dropout=[.3], #[.3,.4,.5]
            times_pass_model= num_repetitions_experiment,
            type_combination=all_combinations_experiment,
            bin_umbral = 0.5
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            val_dropout=[.2],
            times_pass_model= num_repetitions_experiment,
            type_combination=all_combinations_experiment,
            bin_umbral = 0.5
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            val_dropout=[.2],
            times_pass_model= num_repetitions_experiment,
            type_combination=all_combinations_experiment,
            bin_umbral = 0.5
            )
    ]
    
    votes_threshold_list = [.25, .5, .75, 1.]
    bin_th_list = []
    idx_experiment = 0

    for TEST_PARAMETER in MODELS_TO_TEST:
        if config.db_train is not None and TEST_PARAMETER.dataset_name != config.db_train:
            continue
        bin_th = TEST_PARAMETER.bin_umbral
        for type_combination in TEST_PARAMETER.type_combination:
            for times_pass_model in TEST_PARAMETER.times_pass_model:
                for val_dropout_item in TEST_PARAMETER.val_dropout:
                    if type_combination == utilsParameters.PredictionsCombinationType.VOTES:
                        for votes_threshold in votes_threshold_list:
                            '''
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
                            '''
                            
                            if votes_threshold == 1:
                                times_pass_model = 2
                            #bin_th = 0.4 #quitar
                            logs_experiment = ""#quitar
                            bin_th_list.append(bin_th)
                            if logs == "":
                                if logs_experiment != "":
                                    logs = logs_experiment + "\n"
                            else:
                                if logs_experiment != "":
                                    logs += logs_experiment.split("\n")[1] + "\n"
                            saveLogs(logs, path_results)
                            
                            for db_test in config.db_test:
                                logs_experiment = inference.TFMTest(dataset_name_train=TEST_PARAMETER.dataset_name,
                                                                    dataset_name_test=db_test,
                                        dropout_value=TEST_PARAMETER.train_dropout,
                                        val_dropout=val_dropout_item,
                                        times_pass_model=times_pass_model,
                                        type_combination=type_combination,
                                        uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                        uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                        save_test_info=save_test_info,
                                        save_test_img=save_test_img,
                                        bin_umbral_for_model=bin_th,
                                        votes_threshold=votes_threshold,
                                        seed_value = config.seed
                                        )
                                logs += logs_experiment.split("\n")[1] + "\n"
                    else:
                        '''
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
                        '''
                        #bin_th = 0.4
                        logs_experiment = ""
                        bin_th_list.append(bin_th)
                        if logs == "":
                            if logs_experiment != "":
                                logs = logs_experiment + "\n"
                        else:
                            if logs_experiment != "":
                                logs += logs_experiment.split("\n")[1] + "\n"
                        saveLogs(logs, path_results)
                        
                        for db_test in config.db_test:
                            logs_experiment = inference.TFMTest(dataset_name_train=TEST_PARAMETER.dataset_name,
                                                                dataset_name_test=db_test,
                                    dropout_value=TEST_PARAMETER.train_dropout,
                                    val_dropout=val_dropout_item,
                                    times_pass_model=times_pass_model,
                                    type_combination=type_combination,
                                    uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                    uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                    save_test_info=save_test_info,
                                    save_test_img=save_test_img,
                                    bin_umbral_for_model=bin_th,
                                    votes_threshold=0,
                                    seed_value = config.seed
                                    )
                            logs += logs_experiment.split("\n")[1] + "\n"
                        
                    print("Experiment " + str(idx_experiment) + " finished")
                    idx_experiment = idx_experiment+1
                    saveLogs(logs, path_results)

    saveLogs(logs, path_results)
    print("Results saved in...")
    print(path_results)
    
        
if __name__ == '__main__':
    
    config = menu()
    print (config)

    #run_Base(config)
    #run_DropoutTrainBase(config)
    
    run_DropoutCombination(config)

        

    
    
    pass

