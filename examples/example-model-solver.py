# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: ssb-model-solver
#     language: python
#     name: ssb-model-solver
# ---

# %% [markdown]
# # Example of use of **ModelSolver**
# #### Magnus Kvåle Helliesen, mkh@ssb.no
#
# This Notebook shows basic useage of **ModelSolver**.
# The algorithm analyzes and solves an input-output model used in the Norwegian monthly national accounts with more than 15,500 equations in under a minute for more than 30 months on a laptop computer.
#
# To import **ModelSolver** you need the following packages installed:
# * NumPy
# * NetworkX
# * Pandas
# * SymEngine
# * Numba
# * collections
# * functools
# * Matplotlib

# %% [markdown]
# ## Load class and initialize class
# We first load all packages and the model class itself:

# %%
import os

# Change directory until find project root
notebook_path = os.getcwd()
for _folder_level in range(50):
    if "pyproject.toml" in os.listdir():
        break
    os.chdir("../")

# %%
import pandas as pd
import src.model_solver as ms

# %matplotlib inline

# %% [markdown]
# We define lists of equations and endogenous variables.
# These are what together *define* a particular **ModelSolver** class instance:

# %%
equations = [
    "x1 = a1",
    "x2 = a2",
    "0.2*x1+0.7*x2 = 0.1*ca+0.8*cb+0.3*i1",
    "0.8*x1+0.3*x2 = 0.9*ca+0.2*cb+0.1*i2",
    "k1 = k1(-1)+i1",
    "k2 = k2(-1)+i2",
]
endogenous = ["x1", "x2", "ca", "cb", "k1", "k2"]

# %% [markdown]
# We initiate an instance of **ModelSolver** class called *model*:

# %%
model = ms.ModelSolver(equations, endogenous)

# %% [markdown]
# We let **ModelSolver** describe the model to us:

# %%
model.describe()

# %% [markdown]
# We can also inspect all the blocks:

# %%
model.show_blocks()

# %% [markdown]
# ## Solve the model subject to data
#
# We make a Pandas DataFrame:

# %%
input_data = pd.DataFrame(
    {
        "x1": [2, 4, 1, 2],
        "x2": [2, 1, 2, 3],
        "ca": [1, 3, 4, 1],
        "cb": [1, 2, 1, 4],
        "k1": [1, 3, 4, 1],
        "k2": [1, 2, 1, 4],
        "a1": [1, 2, 4, 4],
        "a2": [3, 2, 3, 4],
        "i1": [1, 2, 4, 4],
        "i2": [3, 2, 3, 4],
    },
    index=["2019Q1", "2019Q2", "2020Q3", "2020Q4"],
)
input_data.head()

# %% [markdown]
# The data contain exogenous variables and initial values for endogenous variables.

# %% [markdown]
# We call the method that solves the model, subject to the data in the DataFrame:

# %%
solution = model.solve_model(input_data)
solution.head()

# %% [markdown]
# We make a graph of some variable showing what block it's in, and what blocks are it's ancestors and decendants.

# %%
model.draw_blockwise_graph("ca")

# %% [markdown]
# ## Switch endogenous variables and analyze new model
#
# We ask **ModelSolver** to make $x1$ exogenous and $a1$ endogenous:

# %%
model.switch_endo_vars(["x2"], ["a2"])

# %% [markdown]
# We inspect the blocks of this new model:

# %%
model.show_blocks()

# %% [markdown]
# We make graph plot of the same variable as above showing what block it's in, and what blocks are it's ancestors and decendants.
# We observe that the graph has changed after swithcing endogenous variables.

# %%
model.draw_blockwise_graph("ca")

# %%
model.show_block_vals(5, 1)

# %%
model.show_block(2)
