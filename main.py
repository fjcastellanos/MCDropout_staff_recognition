

import training
import inference
import utilsParameters

from enum import Enum



if __name__ == '__main__':
    

    # Models parameters
    dropout_value = 0
    save_train_info = True
    plot_epochs = False

                                    # Mod1,  Mod2,  Mod3,  Mod4
    usesRedimensionVerticalList   = [False, False, True , True]
    usesRedimensionHorizontalList = [False, True , False, True]

    for uses_redimension_vertical, uses_redimension_horizontal in zip(usesRedimensionVerticalList, usesRedimensionHorizontalList):
        for dataset_name in utilsParameters.DATASETS:
            training.TFMTrainModel(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_train_info=save_train_info,
                        plot_epochs=plot_epochs
                        )

    #Extraemos mejor umbral
    # Models parameters
    dropout_value = 0
    save_val_info = True
    save_val_imgs = True
    val_dropout = 0
    times_pass_model = 1
    type_combination = utilsParameters.PredictionsCombinationType.NONE

                                    # Mod1,  Mod2,  Mod3,  Mod4
    usesRedimensionVerticalList   = [False, False, True , True]
    usesRedimensionHorizontalList = [False, True , False, True]

    for uses_redimension_vertical, uses_redimension_horizontal in zip(usesRedimensionVerticalList, usesRedimensionHorizontalList):
        for dataset_name in utilsParameters.DATASETS:
            training.TFMValidation(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        val_dropout=val_dropout,
                        times_pass_model=times_pass_model,
                        type_combination=type_combination,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_val_info=save_val_info,
                        save_val_imgs=save_val_imgs
                        )


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

    for uses_redimension_vertical, uses_redimension_horizontal in zip(usesRedimensionVerticalList, usesRedimensionHorizontalList):
        for dataset_name in utilsParameters.DATASETS:
            inference.TFMTest(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        val_dropout=val_dropout,
                        times_pass_model=times_pass_model,
                        type_combination=type_combination,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_test_info=save_test_info,
                        save_test_img=save_test_img
                        )


    #dropout con entrenamiento
    # Models parameters
    save_train_info = True
    plot_epochs = False

    uses_redimension_vertical = True
    uses_redimension_horizontal = True

    for dropout_value in utilsParameters.DROPOUT_VALUES:
        for dataset_name in utilsParameters.DATASETS:
            training.TFMTrainModel(dataset_name=dataset_name,
                        dropout_value=dropout_value,
                        uses_redimension_vertical=uses_redimension_vertical,
                        uses_redimension_horizontal=uses_redimension_horizontal,
                        save_train_info=save_train_info,
                        plot_epochs=plot_epochs
                        )
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

    for dropout_value in utilsParameters.DROPOUT_VALUES:
        for dataset_name in utilsParameters.DATASETS:
            training.TFMValidation(dataset_name=dataset_name,
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

    #dropout en test
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
            val_dropout=0.3
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_SEILS,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            val_dropout=0.2
            ),
        utilsParameters.ForwardParameters(
            utilsParameters.DATASET_FMT_C,
            uses_redimension_horizontal=uses_redimension_horizontal,
            uses_redimension_vertical=uses_redimension_vertical,
            train_dropout=0.2,
            val_dropout=0.2
            )
    ]

    votes_threshold = 0.5

    for TEST_PARAMETER in MODELS_TO_TEST:
        for type_combination in utilsParameters.PREDICTIONS_COMBIATION_TYPES:
            for times_pass_model in utilsParameters.PREDICTIONS_COMBINATION_QUANTITIES:
                training.TFMValidation(dataset_name=TEST_PARAMETER.dataset_name,
                                dropout_value=TEST_PARAMETER.train_dropout,
                                val_dropout=TEST_PARAMETER.train_dropout,
                                times_pass_model=times_pass_model,
                                type_combination=type_combination,
                                votes_threshold=votes_threshold,
                                uses_redimension_vertical=TEST_PARAMETER.uses_redimension_vertical,
                                uses_redimension_horizontal=TEST_PARAMETER.uses_redimension_horizontal,
                                save_val_info=save_val_info,
                                save_val_imgs=save_val_imgs
                            )
                

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

    votes_threshold = 0.5


    for TEST_PARAMETER in MODELS_TO_TEST:
        for val_dropout in utilsParameters.DROPOUT_VALUES:
            if val_dropout != TEST_PARAMETER.train_dropout:
                training.TFMValidation(dataset_name=TEST_PARAMETER.dataset_name,
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