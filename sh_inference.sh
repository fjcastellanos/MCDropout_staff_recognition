#!/bin/bash

db_train="Capitan"
db_test=["Capitan","SEILS","FMT_C"]


mkdir "logs"

for db_train in "SEILS"; do #"Capitan" "SEILS" "FMT_C"
        mkdir "logs/train_${db_train}"		
        output_file="logs/train_${db_train}/test.txt"

        python -u maintest.py \
                -db_train ${db_train} \
                -db_test Capitan SEILS FMT_C \
                &> ${output_file}

    
done
