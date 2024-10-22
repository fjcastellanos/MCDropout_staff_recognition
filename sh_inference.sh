#!/bin/bash

db_train="Capitan"
db_test="Capitan"

output_file="logs/train_${db_train}/test_${db_test}.txt"


for db_train in "Capitan" "SEILS" "FMT_C"; do #"Capitan" "SEILS" "FMT_C"
    for db_test in "Capitan" "SEILS" "FMT_C"; do #"Capitan" "SEILS" "FMT_C"
        python -u maintest.py \
                -db_train ${db_train} \
                -db_test ${db_test} \
                &> ${output_file}

    done
done
