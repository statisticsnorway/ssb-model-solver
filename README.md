# ModelSolver

`ModelSolver` is a class that lets the user define, block analyze and solves dynamic and algebraic models numerically. See [documentation](https://github.com/statisticsnorway/model-solver/blob/main/model-solver.pdf) for detailed information about theory and implementation of the class.

Opprettet av:
Magnus Helliesen <mkh@ssb.no>

---

ModelSolver is a Python class. It defines, analyzes and solves dynamic algebraic model with lots of equations.

Usage is

```python
model = ModelSolver(equations, endogenous)
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
model = ModelSolver(equations, endogenous)
```

When initialized, the class reads in the equations, analyzes them for any lags, before it block analyzes it to find the smallest model blocks that must be solved simultaneously. Note that ModelSolver is not case sensitive, so 'x' and 'X' are the same, both in equations, lists and dataframe (below).

When the class is finished initializing, the user can call the following methods:
* `solution = model.solve(dataframe)` where `dataframe` is a **Pandas** dataframe containing initial values for the endogenous variables and values for the exogenous variables. `solution` is a dataframe with same dimensions as `dataframe` containing the solutions for the endogenous variables. Note that *all* variabels of the model must be present in `dataframe`.
* `model.swithc_endo_vars(old_endo_vars, new_endo_vars)` where `old_endo_vars` is a list of endogenous variables the user wants to switch with `new_endo_vars`.
* `model.describe()` writes out information about the model: The number of blocks, size of blokcs etc.
* `model.find_endo_var('<var>')` finds the block number in which \<var\> is solved for.
* `model.draw_blockwise_graph(variable, maximum_ancestor_generations, maximum_decendants_generations)` where `variable` is a variable of interest, and `maximum_ancestor_generations` and `maximum_decendants_generations`are non-negative integers that governs the number of generations before and after the variable to be graphed. The output is a HTML-file with a relational graph.
* More TBA
