#!/bin/bash


mkdir -p /home/wb/processed_data

list=($(ls ../train_curated))
thres=49
count=0
max_count=49
max_wav=50
for f in "${list[@]}";do
    if [ True ]; then
	    n_f=$(ls ../train_curated/$f | wc -l)
	    if [ $n_f -gt $thres ]; then
		    echo '---'
		    echo $n_f
		    echo $f
		    n_dir_new=$(ls /home/wb/processed | wc -l)
		    max_folders=50
		    if [ $n_dir_new -lt $max_folders ]; then
			    mkdir -p /home/wb/processed/$f
			    wavs=($(ls ../train_curated/$f))
			    echo $wavs
			    for wav in "${wavs[@]}";do
				    n_wav_copied=$(ls /home/wb/processed/$f/ | wc -l)
				    if [ $n_wav_copied -lt $max_wav ]; then
					    cp ../train_curated/$f/$wav /home/wb/processed/$f/
				    fi
					    #echo $n_wav_copied
				    #ls $wav
			    done

		    fi


		    moved=0
		    total_move=5
		    while [ $moved -lt $total_move ]; do
			    for wav in $f; do
				    #ls -lh $wav
				    ((moved++))
			    done
		    done

		    if [ $count -eq $max_count ]; then
			    break
		    fi
		    ((count++))

	    fi

	    
    fi

done
echo $count
