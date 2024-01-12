# ModelSolver

`ModelSolver` is a class that defines, block analyses and solves dynamic and algebraic models numerically.
See [documentation](https://github.com/statisticsnorway/model-solver/blob/main/model-solver.pdf) for detailed information about theory and implementation of the class.

Opprettet av:
Magnus Kvåle Helliesen <mkh@ssb.no>

---

## Features

ModelSolver is a Python class. It defines, analyses and solves dynamic algebraic model with lots of equations.

The package is imported using

```python
import model_solver as ms
```

Usage is

```python
model = ms.ModelSolver(equations, endogenous)
```

where `equations` are equations and `endogenous` are endogenous variables, both stored as strings in lists.

## Built with
ModelSolver uses the following packages
* [NumPy](https://numpy.org/)
* [NetworkX](https://networkx.org/)
* [Pandas](https://pandas.pydata.org/)
* [SymEngine](https://pypi.org/project/symengine/)
* [Numba](https://numba.pydata.org/)
* [collections](https://docs.python.org/3/library/collections.html)
* [functools](https://docs.python.org/3/library/functools.html)
* [Matplotlib](https://matplotlib.org/)

## Example of use
Let `equations = ['x+y = 1', 'x-y = 2']` and `endogenous = ['x', 'y']`, then the model class is initialized by

```python
model = ms.ModelSolver(equations, endogenous)
```

When initialized, the class reads in the equations, analyzes them for any lags, before it block analyzes it to find the smalles model blocks that must be solved simultaneously.
Note that ModelSolver is not case sensitive, such that 'x' and 'X' are the same, both in equations, lists and dataframe (below).

When the class is finished initializing, the user can call the following methods:
* ```solution = model.solve(dataframe)``` where `dataframe` is a **Pandas** dataframe containing initial values for the endogenous variables and values for the exogenous variables. `solution` is a dataframe with same dimensions as `dataframe` containing the solutions for the endogenous variables.
* ```model.switch_endo_vars(old_endo_var, new_endo_var)``` switches the endogenous variables `old_endo_var` for `new_endo_var`.
* ```model.describe()``` writes out information about the model: the number of blocks, the size of the blocks etc.
* ```model.find_endo_var('var')``` returns the block number in which `var` is solved for.
* ```model.show_block(block_number)``` returns information about the block: endogenous variables, predetermined variables and equations.
* ```model.show_blocks()``` returns information about all blocks.
* ```model.trace_to_exog_vars(block_nunber)``` traces back to the exogenous variables that may affect the block.
* ```model.trace_to_exog_vals(block_nunber, period_index)``` traces back to the exogenous variable values for the period.
* ```model.draw_blockwise_graph(variable, maximum_ancestor_generations, maximum_decendants_generations)``` where `variable` is a variable of interest, and `maximum_ancestor_generations` and `maximum_decendants_generations`are non-negative integers that governs the number of generations before and after the variable to be graphed. The output is a HTML-file with a relational graph.
* ```model.sensitivity(block_nunber, period_index[, method='std', exog_subset=None])``` analyses the sensitivity of the endogenous variable in the block with respect to the exogenous variabels that determine the solution for the period.
