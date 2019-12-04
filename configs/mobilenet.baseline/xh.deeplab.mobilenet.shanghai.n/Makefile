train:
	python3 train.py

watch:
	python3 ../../../scripts/show_table.py

test:
	python3 ../../../scripts/base_test.py ${MODEL} ${BN} ${SAVE}

push_result:
	python3 ../../../scripts/get_IoU_csv.py ${EXP} ${NAME} ${BASELINE}
 
test_name:
	echo ${EXP}
	echo ${NAME}

BN = True
SAVE = 0
EXP = $$(basename "${PWD}")
NAME = test.json
BASELINE = False
