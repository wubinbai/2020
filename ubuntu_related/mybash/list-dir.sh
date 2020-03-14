#!/bin/bash

list=($(ls))

for f in "${list[@]}";do
    if [ -d $f ]; then
	    echo $f
            ls $f | wc -l
    fi

done
