
### Code for implementing, training, and analyzing spacetime attractors

To get started:

conda create -n pysta python=3.12 pip

conda activate pysta

pip install -r requirements.txt

pip install -e .

Change the 'basedir' directory name in pysta/utils.py so it matches the path to the 'sta_code' folder locally.

Initialise all the directories you need to save results\n
python ./scripts/initialise_directories.py

Run tests to check that everything works:\n
python ./tests/run_all_tests.py

To train an example RNN:\n
python ./pysta/train_rnn.py

To reproduce all results from the paper:\n
python ./scripts/run_handcrafted_analyses.py\n
python ./scripts/train_all_rnns.py\n
python ./scripts/run_rnn_analyses.py [run only after all RNN training has finished]\n
python ./figure_code/plot_all_figures.py

