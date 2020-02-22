#! /bin/bash

read DATA
if [ -f $DATA ]
then
	echo 'this is a file.'
elif [ -d $DATA ]
then
	echo 'this is a dir'
else
	echo 'this is else'
fi

