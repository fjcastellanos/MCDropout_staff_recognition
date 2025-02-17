#!/bin/bash

db_train="Capitan"
db_test=["Capitan","SEILS","FMT_C"]


mkdir "logs"

for db_train in "Capitan"; do #"Capitan" "SEILS" "FMT_C"
	for seed in 22 51 75 99 147 178 200 325 540 603 664 668 789 927 991 1003 1142 3346 3367 3388 5463 5668 8854 9120 9668; do #
        
		mkdir "logs/test_${db_train}"		
        output_file="logs/test_${db_train}/test.txt"

        python -u maintest.py \
                -db_train ${db_train} \
                -db_test ${db_train} \
				-seed ${seed}
                &> ${output_file}

    
done
done
