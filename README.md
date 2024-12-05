# tum-adlr-10
# 11. Nov
    train baseline network with random exploration
# TODO：
- Do you split your data in Test and Training? (If you use a fixed test set, you can compare the data later a bit better)
    Use one trajectory for testing during training.
    For later experiments I think it make sense to set a seed for random policy.
- Did you also plot the test and training loss? -> It is not overfitting.
    seems to be nice
- Did you add noise to the training data? (If not, it is not necessary to do so yet)
    look worse. I guess that's for real world data with noise.
- Your input is force, last position, and last velocity, and your output is the current position, velocity? You simulate with discreet time steps?
    Yes!
    Should we check if the satisfy derivative relationship?
- How much data did you generate?
    traj long 500 * 100 traj

# 5. Dec
- try to run whole active learning algorithm with RandomSampling method
- first trial:
    using K=2000
    discard MPC, directly sample length of horizon
    choose B best sequences instead of only 1