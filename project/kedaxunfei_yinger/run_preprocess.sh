#! /bin/bash

mkdir input
mv *.rar input
cd input
unrar e *.rar
mv *.rar ../
mkdir train test
mv test_* test
mv *.wav train
cd train
mkdir hug hungry uncomfortable sleepy diaper awake
mv hug_* hug
mv hungry_* hungry
mv uncomfortable_* uncomfortable
mv sleepy_* sleepy
mv diaper_* diaper
mv awake_* awake






