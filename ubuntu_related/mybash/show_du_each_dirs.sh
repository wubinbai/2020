#!/bin/bash

list=($(ls))

for f in "${list[@]}";do
    if [ -d $f ]; then
	    fname=${f}"/"
	    disk_usage=$(du -sh $fname)
	    echo $disk_usage
    fi

done
