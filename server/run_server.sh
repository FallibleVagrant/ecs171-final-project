#!/bin/bash
if ! [ -f "random_forest.pkl" ]; then
	echo "Could not find model; generating random_forest.pkl file by running 'python3 gen_pkl_file.py'."
	python3 gen_pkl_file.py
	echo "Done! But if you're seeing error messages, you should CTRL-C and fix those."
	for i in $(seq 1 5);
	do
		echo "."
		sleep 1
	done
fi

echo "Starting server..."
flask --app server run

#If you encountered an error while running this, you probably need to
#pip install flask
