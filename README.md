# BDQN-MxNet-Gluon

This repo is forked from the original author's repo on BDQN implementation using MxNet and Gluon packages.
The BDQN network implementation follows the tutorial in the BDQN.ipynb file.
The images and models folder are forked from the original repo so they have nothing to do with our experiments.
We have implemented a DDQN network in ddqn.py and utils.py. We have also modified the original BDQN networks and used the following codes
for experiments: bdqn.py, bdqn_new.py, bdqn_new2.py, bdqn_new3.py. These BDQN networks differ only by a layer or two.
Our main results are shown in the Results folder.
The tutorial BDQN.ipynb is contributed by the original author which we followed closely.

To run our experiments for all networks, we can simply run python3 file_name game_name
where game_name can be pong, assault, alien, or centipede, and file_name can be ddqn.py, bdqn.py, bdqn_new.py, or bdqn_new2.py.
