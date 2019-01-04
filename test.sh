#!/usr/bin/bash

image_dir=/tmp/YTS_DATA/unassisted/

mom=0
freeze=249
for lr in {00001,00003,00010,00030,00100,00300,001000,003000,010000,030000,10000,30000}; do
	run_name=run_${freeze}_${mom}_${lr}
	python finetune.py ${image_dir} -pc --valpart .4 --learning_rate 0.$lr --momentum 0.$mom --n_freeze $freeze --run_name ${run_name} --epochs 200 | tee results/${run_name}.txt
done

mom=0
freeze=311
for lr in {00001,00003,00010,00030,00100,00300,001000,003000,010000,030000,10000,30000}; do
	run_name=run_${freeze}_${mom}_${lr}
	python finetune.py ${image_dir} -pc --valpart .4 --learning_rate 0.$lr --momentum 0.$mom --n_freeze $freeze --run_name ${run_name} --epochs 200 | tee results/${run_name}.txt
done

mom=0
freeze=0
for lr in {00001,00003,00010,00030,00100,00300,001000,003000,010000,030000,10000,30000}; do
	run_name=run_${freeze}_${mom}_${lr}
	python finetune.py ${image_dir} -pc --valpart .4 --learning_rate 0.$lr --momentum 0.$mom --n_freeze $freeze --run_name ${run_name} --epochs 200 | tee results/${run_name}.txt
done

mom=9
freeze=249
for lr in {00001,00003,00010,00030,00100,00300,001000,003000,010000,030000,10000,30000}; do
	run_name=run_${freeze}_${mom}_${lr}
	python finetune.py ${image_dir} -pc --valpart .4 --learning_rate 0.$lr --momentum 0.$mom --n_freeze $freeze --run_name ${run_name} --epochs 200 | tee results/${run_name}.txt
done

mom=9
freeze=311
for lr in {00001,00003,00010,00030,00100,00300,001000,003000,010000,030000,10000,30000}; do
	run_name=run_${freeze}_${mom}_${lr}
	python finetune.py ${image_dir} -pc --valpart .4 --learning_rate 0.$lr --momentum 0.$mom --n_freeze $freeze --run_name ${run_name} --epochs 200 | tee results/${run_name}.txt
done

mom=9
freeze=0
for lr in {00001,00003,00010,00030,00100,00300,001000,003000,010000,030000,10000,30000}; do
	run_name=run_${freeze}_${mom}_${lr}
	python finetune.py ${image_dir} -pc --valpart .4 --learning_rate 0.$lr --momentum 0.$mom --n_freeze $freeze --run_name ${run_name} --epochs 200 | tee results/${run_name}.txt
done

cd results
for x in run_*; do echo "$x $(cat $x|grep acc\ 0.)" >> summary.txt; done
cd ..
