
# Super Mario
Refernce for this solution is MadMario
PyTorch [official tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) to build an AI-powered Mario.

Changed the CNN architecture with MaxPool layer.
Update the reward function to consider score.
Moved the Experience Replay memory from GPU to RAM since its asking for memory of 26GB. This is leading to overflow of GPU memory. 
Done some optimization for faster learning with more Batch Size. 
