#!/bin/bash
# 0 setup directories: input: train, test and val directories and output directory and val.
echo "Caution: this will delete all input folder, but don't worry, will also generate a new one from the .rar data: Enter n for not repeat, other characters for repeat!"
read confirm 
compare='n'

if [ $confirm != $compare ]; then
	echo 'removing input data now...'
	rm -r ../input
	mkdir ../input ../output
	mv ../orig_data/*.rar ../input
	cd ../input
	unrar e *.rar
	rm uncomfortable_0.wav
	python3 ../src/clean.py

	mv *.rar ../orig_data
	mkdir train test
	mv test_* test

	mv *.wav train
	cd train
	mkdir hug hungry uncomfortable sleepy diaper awake
	mv awake_* awake
	mv hug_* hug
	mv hungry_* hungry
	mv uncomfortable_* uncomfortable
	mv sleepy_* sleepy
	mv diaper_* diaper

	cd ..
	#mkdir val
	#mkdir val/hug val/hungry val/uncomfortable val/sleepy val/diaper val/awake
	# 1600 means "1600Hz sampling rate"
	mkdir eda
	mkdir eda/1600
	mkdir eda/1600/hug eda/1600/hungry eda/1600/uncomfortable eda/1600/sleepy eda/1600/diaper eda/1600/awake
	cd ../src

	### Dealing with sampling rate problem
	# MOVE 1600 Hz sampling rate .wav files to corresponding folder

	# convert all .wav file from 44100 Hz -> 16000 Hz,

	python3 move_1600.py
	
	cd ../input
	crepe test/*
	crepe train/awake/*
	crepe train/diaper/*
	crepe train/hug/*
	crepe train/hungry/*
	crepe train/sleepy/*
	crepe train/uncomfortable/*


	# seperate 10 (.wav files, randomly chosen) * 6 (classes) into folder input/val for validation

	#python3 move_val.py

fi




