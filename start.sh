#!/bin/bash
# Author: Khaled Nakhleh. 2018.
# To give permission, type in the terminal: chmod 755 start.sh

function Cleaning {
echo "Running cleaning.py...\n"
python main.py
}

function Modeling {
echo "No model exits. Running model.py to train a model..."
sleep 2
echo "Running model.py..."
python model.py
}

function Run {
read -p "Enter project name: " name

read -p "Country where project was started: " country

read -p "Currency used for funding? i.e. 'USD', 'CAD': " currency

read -p "Funding goal: " goal

read -p "Adjusted funding goal (default = goal): " adj_goal
adj_goal=${adj_goal:-$goal}
echo $adj_goal

read -p "projected raised amount: " raised

read -p "Adjusted raised amount (default = raised): " adj_raised
adj_raised=${adj_raised:-$raised}
echo $adj_raised

read -p "projected number of backers: " backers

echo "Running check.py..."
sleep 2
python check.py $name $country $currency $goal $adj_goal $raised $adj_raised $backers
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
	    	Modeling
		fi
else
echo "No cleaned data version found. Creating cleaned version and training a model..."
Cleaning
Modeling
Run
fi

