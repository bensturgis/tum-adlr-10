import sys
import os
sys.path.append(os.path.abspath("."))

from environments.mass_spring_damper_system import TrueMassSpringDamperEnv
from environments.reacher import TrueReacherEnv
from utils.visualization_utils import *
from models.laplace_bnn import LaplaceBNN
from models.mc_dropout_bnn import MCDropoutBNN
import numpy as np

# '''
# for MSD
# '''
# # Use same environment and horizon as for the active learning
# env = TrueMassSpringDamperEnv(noise_var=0.0)
# HORIZON = 100

# state_bounds = env.get_state_bounds(horizon=HORIZON)
# actions_bounds = env.get_action_bounds()
# # Hyperparameters for neural network
# STATE_DIM = 2
# ACTION_DIM = 1
# HIDDEN_SIZE = 72          # Hidden units in the neural network
# DROP_PROB = 0.1           # Dropout probability for the bayesian neural network
# DEVICE = "cuda"
# # Initialize trained dynamics model and load saved weights
# dynamics_model = LaplaceBNN(
#     state_dim=STATE_DIM,
#     action_dim=ACTION_DIM,
#     input_expansion=env.input_expansion,
#     state_bounds=state_bounds,
#     action_bounds=actions_bounds,
#     hidden_size=HIDDEN_SIZE,
#     device=DEVICE,
# )
# exp_idx = 44
# rep = 0
# sampling_methods = ["Random Exploration", "Random Sampling Shooting", "Soft Actor Critic"]
# for rep in range(2,3):
#     for sp_mtd in sampling_methods:
#         for num_iter in range(1,21,1):
#             plot_msd_uncertainty(
#                 experiment=exp_idx,
#                 sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
#                 num_al_iterations=num_iter,
#                 true_env=env,
#                 horizon=HORIZON,
#                 repetition=rep,
#                 show_plot=False,
#                 title_size=-1,
#                 show_colorbar=True if sp_mtd=="Soft Actor Critic" else False,
#                 model=dynamics_model
#             )
#     #     generate_GIF(
#     #         experiment=exp_idx, 
#     #         sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
#     #         repetition=rep
#     #         )
#     # compare_GIF(exp_idx, rep)
#     # compare_png(exp_idx, rep, 3)
#     # compare_png(exp_idx, rep, 10)
#     # compare_png(exp_idx, rep, 20)

# '''
# for Reacher
# '''
# # # Use same environment and horizon as for the active learning
# env = TrueReacherEnv(noise_var=0.0)
# HORIZON = 50

# state_bounds = env.get_state_bounds(horizon=HORIZON, bound_shrink_factor=0.06)
# actions_bounds = env.get_action_bounds()
# # Hyperparameters for neural network
# STATE_DIM = 6
# ACTION_DIM = 2
# HIDDEN_SIZE = 72          # Hidden units in the neural network
# DROP_PROB = 0.1           # Dropout probability for the bayesian neural network
# DEVICE = "cuda"
# # Initialize trained dynamics model and load saved weights
# # bnn_model = MCDropoutBNN(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE, drop_prob=DROP_PROB, device=DEVICE)
# dynamics_model = LaplaceBNN(
#     state_dim=STATE_DIM,
#     action_dim=ACTION_DIM,
#     input_expansion=env.input_expansion,
#     state_bounds=state_bounds,
#     action_bounds=actions_bounds,
#     hidden_size=HIDDEN_SIZE,
#     device=DEVICE,
# )

# exp_idx = 42
# rep = 0
# sampling_methods = ["Random Exploration", "Random Sampling Shooting", "Soft Actor Critic"]
# for rep in range(2,3):
#     for sp_mtd in sampling_methods:
#         for num_iter in range(1,26,1):
#             plot_reacher_uncertainty(
#                 experiment=exp_idx,
#                 sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
#                 num_al_iterations=num_iter,
#                 true_env=env,
#                 horizon=HORIZON,
#                 repetition=rep,
#                 show_plot=False,
#                 title_size=-1,
#                 show_colorbar=True if sp_mtd=="Soft Actor Critic" else False,
#                 model=dynamics_model
#             )
#     #     generate_GIF(
#     #         experiment=exp_idx, 
#     #         sampling_method=sp_mtd, # "Random Exploration" "Random Sampling Shooting" "Soft Actor Critic"
#     #         repetition=rep
#     #         )
#     # compare_GIF(exp_idx, rep)
#     # compare_png(exp_idx, rep, 5)
#     # compare_png(exp_idx, rep, 15)
#     # compare_png(exp_idx, rep, 25)

'''
plot image array
'''
from PIL import Image, ImageDraw, ImageFont
exp_idx = 44
rep = 2
iter_list = [5,10,20]
sampling_methods = ["Random Exploration", "Random Sampling Shooting", "Soft Actor Critic"]
base_path = Path(__file__).parent / "active_learning_evaluations"
image_path = base_path / f"experiment_{exp_idx}"
imgs = np.empty((len(iter_list), len(sampling_methods)), dtype=object)
size = np.empty((len(iter_list), len(sampling_methods)), dtype=object)

for row_idx, num_iter in enumerate(iter_list):
    for column_idx, sp_mtd in enumerate(sampling_methods):
        imgs[row_idx][column_idx] = Image.open(image_path / f"{sp_mtd}_var_rep{rep}_iter{num_iter}.png")
        size[row_idx][column_idx] = imgs[row_idx][column_idx].size
width, height = size[0][0]
width_last = size[0][2][0]
mosaic = Image.new("RGB", (width * 2 + width_last, height * 3))
for i in range(3):
    for j in range(3):
        mosaic.paste(imgs[i][j], (width * j, height * i))
        print(width * i, height * j)

title_width = 150
font = ImageFont.truetype("arial.ttf", 100)
mosaic_with_title = Image.new("RGB", (mosaic.size[0] + title_width, mosaic.size[1] + title_width), "white")
mosaic_with_title.paste(mosaic, (title_width,title_width))
draw = ImageDraw.Draw(mosaic_with_title)
title = "Random Exploration        Random Sampling Shooting             Soft Actor Critic"
title = "  Random Exploration         Random Sampling Shooting               Soft Actor Critic"
draw.text((430, 30), title, font=font, fill="black")
text_img = Image.new("RGB", (mosaic_with_title.size[1], title_width), "white")  # 预留 100 像素宽度
text_draw = ImageDraw.Draw(text_img)
title = f"Iteration {iter_list[2]}                            Iteration {iter_list[1]}                             Iteration {iter_list[0]}"
title = f"Iteration {iter_list[2]}                               Iteration {iter_list[1]}                              Iteration {iter_list[0]}"
text_draw.text((480, 30), title, font=font, fill="black")
text_img = text_img.rotate(90, expand=True)
mosaic_with_title.paste(text_img, (0,0))
output_path = image_path / f"mosaic_plot_rep{rep}.png"
mosaic_with_title.save(output_path, format="PNG")
