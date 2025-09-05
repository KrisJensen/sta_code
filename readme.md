
### Code for implementing, training, and analyzing spacetime attractors

To get started:

conda create -n pysta python=3.12 pip

conda activate pysta

pip install -r requirements.txt

pip install -e .


Then change the 'basedir' directory name in pysta/utils.py so it matches the path to the 'attractor_planner' folder locally.

python ./pysta/train_rnn.py

