#!/bin/bash
# Author: Khaled Nakhleh. 2018.
# To give permission, type in the terminal: chmod 755 start.sh

function Cleaning {
echo "Running cleaning.py...\n"
python main.py
}

function Modeling {
echo "No model exists. Running model.py to train a model..."
sleep 2
echo "Running model.py..."
python model.py
}

function Run {

echo "Running check.py..."
sleep 2
python check.py
}

echo "Running:" $0

if
test -f clean.csv
	then
	echo "cleaned data version exits. Checking if model exits in directory..."
	sleep 1
    	if test -f model.h5
    		then
    		echo "Model exits. Running the program..."
			Run
	    	else
            echo "Model doesn't exit. Training a model and running the program..."
	    	Modeling
            Run
		fi
else
echo "No cleaned data version found. Creating cleaned version and training a model..."
Cleaning
Modeling
Run
fi

