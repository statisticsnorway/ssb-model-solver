# model-solver

Class to define, block analyze and solve dynamic and algebraic models numerically

Opprettet av:
Magnus Helliesen <mkh@ssb.no>

---

ModelSolver is a model class. It defines, analyzes and solves dynamic algebraic model with lots of equations.

Usage is

```python
model = ModelSolver(eqns, endo_vars)
```

where `eqns` are equations and `endo_vars` are endogenous variables, both stored as strings in lists.

## Built with
MNAModel uses the following packages
* [os](https://docs.python.org/3/library/os.html)
* [NumPy](https://numpy.org/)
* [NetworkX](https://networkx.org/)
* [Pyvis](https://pyvis.readthedocs.io/en/latest/)
* [Scipy](https://scipy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Symengine](https://pypi.org/project/symengine/)
* [Numba](https://numba.pydata.org/)
* [collections](https://docs.python.org/3/library/collections.html)

## Example of use
Let `eqns = ['x+y = 1', 'x-y = 2']` and `endo_vars = ['x', 'y']`, then the model class is initialized by

```python
model = MNAModel(eqns, endo_vars)
```

When initialized, the class reads in the equations, analyzes them for any lags, before it block analyzes it to find the smalles model blocks that must be solved simultaneously. Note that MNAModel is not case sensitive, so 'x' is the same as 'X', both in equations, lists and dataframe (below).

When the class is finished initializing, the user can call the following methods:
* `solution = model.solve(dataframe)` where `dataframe` is a **Pandas** dataframe containing initial values for the endogenous variables and values for the exogenous variables. `solution` is a dataframe with same dimensions as `dataframe` containing the solutions for the endogenous variables.
* `model.draw_blockwise_graph(variable, maximum_ancestor_generations, maximum_decendants_generations)` where `variable` is a variable of interest, and `maximum_ancestor_generations` and `maximum_decendants_generations`are non-negative integers that governs the number of generations before and after the variable to be graphed. The output is a HTML-file with a relational graph.
