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
- Set up server to run random sampling shooting + MPC

## Presentation Outline
- Introduction to our topic/Introduction to the problem we are trying to solve
- Presentation of active learning and random sampling shooting via flow chart
- Experiments we have conducted so far including plots
  - Learning curves including train and test error
  - Active Learning evaluation
  - Plot for exploration efficiency