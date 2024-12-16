# tum-adlr-10
## TODO
- Create presentation
- Create plots for presentation
- Try to find reason that causes random exploration to perform better than random sampling shooting
  - use random exploration to sample data points for the one-step predicitive accuracy evaluation
    instead of just randomly sampling states from the observation space and actions from the action space
    (temporarily hard coded fix in `one_step_pred_accuracy.py` specifically for spring-mass-damper system
    and short horizon)
  - implement reacher environment from paper, visualize the state space exploration and compare results with
    those from the paper
  - try RS without MPC
    - it's better? At least got similar performance.
    -> RS seems to favor exploring on velocity instead of position dimension (based on trajectory plot)
  - try use the older version of RS+MPC (without parellelizing batch of trajectories)
    - I checked the code and it should be correct
  - try print out variance value and total differential entropy
  - try reset input limitation
    - based on the trajectory plot, the datapoints seems to be too dispersed, which means input are too large
    - I tried set it to [-3,3], seems to be better, unlike RE, RS is trying to "reach out" to further regions
  - visualize predictive var on the whole state space
- Set up server to run random sampling shooting + MPC

## Presentation Outline
- Introduction to our topic/Introduction to the problem we are trying to solve (Yufei)
  - hard to explore, use as few as possible samples to learn physical dynamic
- Presentation of active learning and random sampling shooting via flow chart (Ben)
  - Active Learning: Algorithm 1
  - RS: Algorithm 4
- Experiments we have conducted so far including plots (Yufei)
  - Environment we use
  - Model performances BNN
    - Learning curves including train and test error
  - Active Learning evaluation
    - RS implementation
    - explain the plot
    - explain the results
  - Plot for exploration efficiency
    - visualization methods
- What are the next milestones? Are there any changes to the research hypothesis or problem statement from the pro-posal? (Ben)

- Save weights for le every activearning iteration to plot bayesian prediction variance (Yufei)
- Create training plots for every ative learning iteration (Ben)