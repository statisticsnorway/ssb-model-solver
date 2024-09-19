#########################################
# ModelSolver                           #
# By: Magnus Kvåle Helliesen            #
# mkh@ssb.no/magnus.helliesen@gmail.com #
#########################################

from collections import Counter
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from functools import cache
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray
from symengine import Add
from symengine import Lambdify
from symengine import Matrix
from symengine import Symbol
from symengine import var


class ModelSolver:
    """ModelSolver is designed to handle and solve mathematical models represented by a system of equations.

    It supports various mathematical functions such as min, max, log, and exp.
    This class allows you to initialize a model with a list of equations and endogenous variables.
    It subsequently solves the model using input data stored in a Pandas DataFrame.

    Usage Example:

    Let `equations` and `endogenous` be lists containing equations and endogenous variables, respectively, stored as strings, e.g.,

    .. code-block:: python

        equations = [
            'x + y = A',
            'x / y = B'
        ]
        endogenous = [
            'x',
            'y'
        ]

    where 'A' and 'B' are exogenous variables.

    To initialize a ModelSolver instance, use:

       model = ModelSolver(equations, endogenous)

    This reads in the equations and endogenous variables, performs block analysis and ordering, and generates simulation code.

    To solve the model using input data in a Pandas DataFrame, let's assume you have a DataFrame named "input_df" containing data on 'A' and 'B' as well as initial values for 'x' and 'y'.
    You can solve the model by invoking:

        solution_df = model.solve_model(input_df)

    Now, `solution_df` is a Pandas DataFrame with the same dimensions as `input_df`, but with the endogenous variables replaced by the solutions to the model.
    The last solution is also stored in `model.last_solution`.
    """

    _NO_SOLUTION_TEXT = "No solution exist"

    # Reads in equations and endogenous variables and does a number of operations, e.g. analyzing block structure using graph theory.
    def __init__(self, eqns: list[str], endo_vars: list[str]) -> None:
        """Reads in equations and endogenous variables and does a number of operations, e.g. analyzing block structure using graph theory.

        Args:
            eqns: A list of equations in string format
            endo_vars: A list of endogenous variables

        Example:
            >>> equations = ["x1 = a1", "x2 = a2", "0.2*x1+0.7*x2 = 0.1*ca+0.8*cb+0.3*i1", "0.8*x1+0.3*x2 = 0.9*ca+0.2*cb+0.1*i2", "k1 = k1(-1)+i1", "k2 = k2(-1)+i2"]
            >>> endogenous = ["x1", "x2", "ca", "cb", "k1", "k2"]
            >>> model = ModelSolver(equations, endogenous)
            ----------------------------------------------------------------------------------------------------
            Initializing model
            * Importing equations
            * Importing endogenous variables
            * Analyzing model
                    * Analyzing equation strings
                    * Generating bipartite graph (BiGraph) connecting equations and endogenous variables
                    * Finding maximum bipartite match (MBM) (i.e. associating every equation with exactly one endogenus variable)
                    * Generating directed graph (DiGraph) connecting endogenous variables using bipartite graph and MBM
                    * Finding condensation of DiGraph (i.e. determining minimal blocks of systems of simulataneous equations)
                    * Generating simulation code (i.e. block-wise symbolic objective function, symbolic Jacobian matrix and lists of endogenous and exogenous variables)
            Finished
            ----------------------------------------------------------------------------------------------------
        """
        self._lag_notation = "__LAG"
        self._max_lag = 0
        self._root_tolerance = 1e-7
        self._max_iter = 10

        print("-" * 100)
        print("Initializing model")

        # Model equations and endogenous variables are checked and stored as immutable tuples (as opposed to mutable lists)
        self._eqns, self._endo_vars = self._init_model(eqns, endo_vars)

        print("* Analyzing model")

        # Analyzing equation strings to determine variables, lags and coefficients
        self._eqns_analyzed, self._var_mapping, self._lag_mapping = self._analyze_eqns()

        # Perform block analysis and ordering of equations
        (
            self._eqns_endo_vars_match,
            self._condenced_model_digraph,
            self._augmented_condenced_model_digraph,
            self._node_varlist_mapping,
        ) = self._block_analyze_model()

        # Generating everything needed to simulate model
        self._sim_code, self._blocks = self._gen_sim_code_and_blocks()

        print("Finished")
        print("-" * 100)

    @property
    def eqns(self) -> tuple[str, ...]:
        """Return the equations in the model."""
        return self._eqns

    @property
    def endo_vars(self) -> tuple[str, ...]:
        """Return the endogenous variables in the model."""
        return self._endo_vars

    @property
    def exog_vars(self) -> tuple[str, ...]:
        """Return the exogenous variables in the model."""
        vars_: set[str] = set()
        for _, _, _, lag_mapping in self._eqns_analyzed:
            for _, val in lag_mapping.items():
                vars_.update((val[0],))
        return tuple(vars_.difference(self.endo_vars))

    @property
    def max_lag(self) -> int:
        """Return max_lag in the model."""
        return self._max_lag

    @property
    def last_solution(self) -> pd.DataFrame:
        """Returns the last found solution in the model.

        Returns:
            The last found solution as a dataframe.

        Raises:
            AttributeError: If no solution is found.
        """
        try:
            return self._last_solution.iloc[self.max_lag :, :]
        except AttributeError as exc:
            raise AttributeError(self._NO_SOLUTION_TEXT) from exc

    @property
    def root_tolerance(self) -> float:
        """Return root tolerance."""
        return self._root_tolerance

    @root_tolerance.setter
    def root_tolerance(self, val: float) -> None:
        if isinstance(val, float) is False:
            raise ValueError("tolerance for termination must be of type float")
        if val <= 0:
            raise ValueError("tolerance for termination must be positive")
        self._root_tolerance = val

    @property
    def max_iter(self) -> int:
        """Return maximum number of iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, val: int) -> None:
        if isinstance(val, int) is False:
            raise ValueError("maximum number of iterations must be an integer")
        if val < 0:
            raise ValueError("maximum number of iterations cannot be negative")
        self._max_iter = val

    # Imports lists containing equations and endogenous variables stored as strings
    # Checks that there are no blank lines, sets everything to lowercase and returns as tuples
    def _init_model(
        self, eqns: list[str], endo_vars: list[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        print("* Importing equations")
        if any(x.strip() == "" for x in eqns):
            raise ValueError("there are empty elements in equation list")

        print("* Importing endogenous variables")
        if any(x.strip() == "" for x in endo_vars):
            raise ValueError("there are empty elements in endogenous variable list")

        if len(eqns) != len(endo_vars):
            raise ValueError(
                "there is a different number of equations and endogenous variables"
            )

        return tuple(x.lower() for x in eqns), tuple(x.lower() for x in endo_vars)

    # Analyzes the equations of the model
    def _analyze_eqns(
        self,
    ) -> tuple[
        list[tuple[str, list[str], dict[str, str], dict[str, tuple[str, int]]]],
        dict[str, str],
        dict[str, tuple[str, int]],
    ]:
        print("\t* Analyzing equation strings")

        eqns_analyzed: list[
            tuple[str, list[str], dict[str, str], dict[str, tuple[str, int]]]
        ] = []
        var_mapping: dict[str, str] = {}
        lag_mapping: dict[str, tuple[str, int]] = {}

        for eqn in self._eqns:
            eqn_analyzed = (eqn, *self._analyze_eqn(eqn))
            eqns_analyzed += (eqn_analyzed,)
            var_mapping = {**var_mapping, **eqn_analyzed[2]}
            lag_mapping = {**lag_mapping, **eqn_analyzed[3]}

        return eqns_analyzed, var_mapping, lag_mapping

    # Takes an equation string and parses it into coefficients (special care is taken to deal with scientific notation), variables, lags and operators/brackets
    # I've written my own parser in stead of using some existing because it needs to take care of then (-)-notation for lags
    def _analyze_eqn(
        self, eqn: str
    ) -> tuple[list[str], dict[str, str], dict[str, tuple[str, int]]]:
        parsed_eqn_with_lag_notation: list[str] = []
        var_mapping: dict[str, str] = {}
        lag_mapping: dict[str, tuple[str, int]] = {}

        component, lag = "", ""
        is_num, is_var, is_lag, is_sci = False, False, False, False

        for chr_ in "".join([eqn, " "]):
            is_num = (chr_.isnumeric() and not is_var) or is_num
            is_var = (chr_.isalpha() and not is_num) or is_var
            is_lag = (is_var and chr_ == "(") or is_lag
            is_sci = (is_num and chr_ == "e") or is_sci

            if is_var and chr_ == "(" and component in ["max", "min", "log", "exp"]:
                parsed_eqn_with_lag_notation += ("".join([component, chr_]),)
                is_var, is_lag = False, False
                component, lag = "", ""
                continue

            # Check if character is something other than a numeric, variable or lag and write numeric or variable to parsed equation
            if chr_ in ["=", "+", "-", "*", "/", "(", ")", ",", " "] and not (
                is_lag or is_sci
            ):
                if is_num:
                    parsed_eqn_with_lag_notation += (str(component),)
                if is_var:
                    # Replace (-)-notation by LAG_NOTATION for lags and appends _ to the end to mark the end
                    pfx = (
                        ""
                        if lag == ""
                        else "".join([self._lag_notation, str(-int(lag[1:-1])), "_"])
                    )
                    parsed_eqn_with_lag_notation += ("".join([component, pfx]),)
                    var_mapping["".join([component, lag])] = "".join([component, pfx])
                    var_mapping["".join([component, pfx])] = "".join([component, lag])
                    lag_mapping["".join([component, pfx])] = (
                        component,
                        0 if lag == "" else -int(lag[1:-1]),
                    )
                    if lag != "":
                        self._max_lag = max(
                            self._max_lag, -int(lag.replace("(", "").replace(")", ""))
                        )
                if chr_ != " ":
                    parsed_eqn_with_lag_notation += (chr_,)
                component, lag = "", ""
                is_num, is_var, is_lag = False, False, False
                continue

            if is_sci and chr_.isnumeric():
                is_sci = False

            if is_num:
                component = "".join([component, chr_])
                continue

            if is_var and not is_lag:
                component = "".join([component, chr_])
                continue

            if is_var and is_lag:
                lag = "".join([lag, chr_])
                if chr_ == ")":
                    is_lag = False
        return parsed_eqn_with_lag_notation, var_mapping, lag_mapping

    # Performs block analysis of equations subject to endogenous variables
    # Analysis is a sequence of operations using graph theory
    def _block_analyze_model(
        self,
    ) -> tuple[
        dict[str | int, int | str],
        nx.DiGraph,
        nx.DiGraph,
        dict[str | int, tuple[str, ...]],
    ]:
        # Using graph theory to analyze equations using existing algorithms to establish minimum simultaneous blocks
        eqns_endo_vars_bigraph = self._gen_eqns_endo_vars_bigraph()
        eqns_endo_vars_match = self._find_max_bipartite_match(eqns_endo_vars_bigraph)
        model_digraph = self._gen_model_digraph(
            eqns_endo_vars_bigraph, eqns_endo_vars_match
        )
        condenced_model_digraph, condenced_model_node_varlist_mapping = (
            self._gen_condenced_model_digraph(model_digraph)
        )
        (
            augmented_condenced_model_digraph,
            augmented_condenced_model_node_varlist_mapping,
        ) = self._gen_augmented_condenced_model_digraph(
            condenced_model_digraph, eqns_endo_vars_match
        )

        node_varlist_mapping = {  # type: ignore
            **condenced_model_node_varlist_mapping,
            **augmented_condenced_model_node_varlist_mapping,
        }
        return (
            eqns_endo_vars_match,
            condenced_model_digraph,
            augmented_condenced_model_digraph,
            node_varlist_mapping,
        )

    # Generates bipartite graph (bigraph) connetcting equations (nodes in U) with endogenous variables (nodes in V)
    # See https://en.wikipedia.org/wiki/Bipartite_graph for a discussion of bigraphs
    def _gen_eqns_endo_vars_bigraph(self) -> nx.Graph:
        print(
            "\t* Generating bipartite graph (BiGraph) connecting equations and endogenous variables"
        )

        # Make nodes in bipartite graph with equations U (0) and endogenous variables in V (1)
        eqns_endo_vars_bigraph = nx.Graph()
        eqns_endo_vars_bigraph.add_nodes_from(
            [i for i, _ in enumerate(self.eqns)], bipartite=0
        )
        eqns_endo_vars_bigraph.add_nodes_from(self._endo_vars, bipartite=1)

        # Make edges between equations and endogenous variables
        for i, eqns in enumerate(self._eqns_analyzed):
            for endo_var in [x for x in eqns[2].keys() if x in self.endo_vars]:
                eqns_endo_vars_bigraph.add_edge(i, endo_var)

        return eqns_endo_vars_bigraph

    # Finds a maximum bipartite match (MBM) of bigraph connetcting equations (nodes in U) with endogenous variables (nodes in V)
    # See https://www.geeksforgeeks.org/maximum-bipartite-matching/ for more on MBM
    # Returns dict with matches (maps both ways, i.e. U-->V and U-->U)
    def _find_max_bipartite_match(
        self, eqns_endo_vars_bigraph: nx.Graph
    ) -> dict[str | int, int | str]:
        print(
            "\t* Finding maximum bipartite match (MBM) (i.e. associating every equation with exactly one endogenus variable)"
        )

        # Use maximum bipartite matching to make a one to one mapping between equations and endogenous variables
        try:
            maximum_bipartite_match: dict[str | int, int | str] = (
                nx.bipartite.maximum_matching(
                    eqns_endo_vars_bigraph, [i for i, _ in enumerate(self._eqns)]
                )
            )
            if len(maximum_bipartite_match) / 2 < len(self.eqns):
                raise RuntimeError("model is over or under spesified")
        except nx.AmbiguousSolution as exc:
            raise RuntimeError("unable to analyze model") from exc

        return maximum_bipartite_match

    # Makes a directed graph (digraph) showing how endogenous variables affect every other endogenous variable
    # See https://en.wikipedia.org/wiki/Directed_graph for more about directed graphs
    def _gen_model_digraph(
        self,
        eqns_endo_vars_bigraph: nx.Graph,
        eqns_endo_vars_match: dict[str | int, int | str],
    ) -> nx.DiGraph:
        print(
            "\t* Generating directed graph (DiGraph) connecting endogenous variables using bipartite graph and MBM"
        )

        # Make nodes in directed graph of endogenous variables
        model_digraph = nx.DiGraph()
        model_digraph.add_nodes_from(self.endo_vars)

        # Make directed edges showing how endogenous variables affect every other endogenous variables using bipartite graph and MBM
        for edge in eqns_endo_vars_bigraph.edges():
            if edge[0] != eqns_endo_vars_match[edge[1]]:
                model_digraph.add_edge(edge[1], eqns_endo_vars_match[edge[0]])

        return model_digraph

    # Makes a condencation of digraph of endogenous variables
    # Each node of condencation contains strongly connected components; this corresponds to the simulataneous model blocks
    # See https://en.wikipedia.org/wiki/Strongly_connected_component for more about strongly connected components
    def _gen_condenced_model_digraph(
        self, model_digraph: nx.DiGraph
    ) -> tuple[nx.DiGraph, dict[int, tuple[str, ...]]]:
        print(
            "\t* Finding condensation of DiGraph (i.e. determining minimal blocks of systems of simulataneous equations)"
        )

        # Generate condensation graph of equation graph such that every node is a strong component of the equation graph
        condenced_model_digraph = nx.condensation(model_digraph)

        # Make a dictionary that associate every node of condensation with a list of variables
        node_vars_mapping = {}
        for node in tuple(condenced_model_digraph.nodes()):
            node_vars_mapping[node] = tuple(
                condenced_model_digraph.nodes[node]["members"]
            )
        return condenced_model_digraph, node_vars_mapping

    # Augments condenced digraph with nodes and edges for exogenous variables in order to show what exogenous variables affect what strong components
    def _gen_augmented_condenced_model_digraph(
        self,
        condenced_model_digraph: nx.DiGraph,
        eqns_endo_vars_match: dict[str | int, int | str],
    ) -> tuple[nx.DiGraph, dict[str, tuple[str, ...]]]:
        augmented_condenced_model_digraph = condenced_model_digraph.copy()

        # Make edges between exogenous variables and strong components it is a part of
        node_vars_mapping = {}
        for node in condenced_model_digraph.nodes():
            for member in condenced_model_digraph.nodes[node]["members"]:
                index = eqns_endo_vars_match[member]
                if not isinstance(index, int):
                    raise TypeError("Index not of type int")

                for exog_var_adjacent_to_node in [
                    val
                    for key, val in self._eqns_analyzed[index][2].items()
                    if self._lag_notation not in val and key not in self.endo_vars
                ]:
                    augmented_condenced_model_digraph.add_edge(
                        exog_var_adjacent_to_node, node
                    )
                    node_vars_mapping[exog_var_adjacent_to_node] = (
                        exog_var_adjacent_to_node,
                    )
        return augmented_condenced_model_digraph, node_vars_mapping  # type: ignore

    # Generates simulation code and blocks
    # Simulation code contains a tuple of tuples for each strong component
    # The tuple for each strong component contains objective function, and Jacobian matrix, and lists of the variables in the strong component
    def _gen_sim_code_and_blocks(
        self,
    ) -> tuple[
        dict[
            int,
            tuple[
                Callable[..., NDArray[Any]] | None,
                Callable[[Iterable[Any], Any], NDArray[Any]] | None,
                Callable[[Iterable[Any], Any], NDArray[Any]] | None,
                tuple[str, ...],
                tuple[str, ...],
                tuple[list[str], ...],
            ],
        ],
        dict[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool]],
    ]:
        print(
            "\t* Generating simulation code (i.e. block-wise symbolic objective function,",
            "symbolic Jacobian matrix and lists of endogenous and exogenous variables)",
        )

        sim_code, blocks = {}, {}
        for i, node in enumerate(
            reversed(tuple(self._condenced_model_digraph.nodes()))
        ):
            block_endo_vars: tuple[str, ...] = ()
            block_eqns_orig: tuple[str, ...] = ()
            block_eqns_lags: tuple[list[str], ...] = ()
            block_pred_vars_set: set[str] = set()
            for member in self._condenced_model_digraph.nodes[node]["members"]:
                j = self._eqns_endo_vars_match[member]
                if not isinstance(j, int):
                    raise TypeError("Index j is not of type int")
                eqns_analyzed = self._eqns_analyzed[j]
                block_endo_vars += (member,)
                block_eqns_orig += (eqns_analyzed[0],)
                block_eqns_lags += (eqns_analyzed[1],)
                block_pred_vars_set.update(
                    [
                        val
                        for key, val in eqns_analyzed[2].items()
                        if self._lag_notation not in key
                    ]
                )

            block_pred_vars_set.difference_update(set(block_endo_vars))
            block_pred_vars = tuple(block_pred_vars_set)

            (def_fun, obj_fun, jac) = self._gen_def_or_obj_fun_and_jac(
                block_eqns_lags, block_endo_vars, block_pred_vars
            )
            sim_code[i + 1] = (
                def_fun,
                obj_fun,
                jac,
                block_endo_vars,
                block_pred_vars,
                block_eqns_lags,
            )
            blocks[i + 1] = (
                block_endo_vars,
                block_pred_vars,
                block_eqns_orig,
                True if def_fun else False,
            )

        return sim_code, blocks

    # Generates symbolic objective functon and Jacobian matrix for a given strong component
    @staticmethod
    def _gen_def_or_obj_fun_and_jac(
        eqns: tuple[list[str], ...],
        endo_vars: tuple[str, ...],
        pred_vars: tuple[str, ...],
    ) -> tuple[
        Callable[..., NDArray[Any]] | None,
        Callable[[Iterable[Any], Any], NDArray[Any]] | None,
        Callable[[Iterable[Any], Any], NDArray[Any]] | None,
    ]:
        endo_sym: list[Symbol] = []
        pred_sym: list[Symbol] = []
        obj_fun: list[Add] = []
        for endo_var in endo_vars:
            var(endo_var)
            endo_sym += (eval(endo_var),)
        for exog_var in pred_vars:
            var(exog_var)
            pred_sym += (eval(exog_var),)
        for eqn in eqns:
            i = eqn.index("=")
            lhs, rhs = eqn[:i], eqn[i + 1 :]
            rhs_str = "".join(rhs).strip().strip("+")
            if len(eqns) == 1 and endo_var == "".join(lhs) and endo_var not in rhs:
                if len(pred_vars) == 0:
                    return (
                        lambda _, rhs_str_=rhs_str: np.array([eval(rhs_str_)]),
                        None,
                        None,
                    )
                def_fun = eval(rhs_str)
                def_fun_lam = Lambdify([pred_sym], def_fun)

                def def_fun_out(
                    args: list[Symbol],
                    def_fun_lam_: Callable[[list[Symbol]], Any] = def_fun_lam,
                ) -> NDArray[np.float64]:
                    return np.array([def_fun_lam_(args)], dtype=np.float64)

                return def_fun_out, None, None

            if ("min(" in lhs) or ("min(" in rhs):
                raise RuntimeError(
                    "min-function is in block that is not a simple definition"
                )
            if ("max(" in lhs) or ("max(" in rhs):
                raise RuntimeError(
                    "max-function is in block that is not a simple definition"
                )

            obj_fun_row = eval(
                "-".join(
                    [
                        "".join(["(", "".join(lhs).strip().strip("+"), ")"]),
                        "".join(["(", "".join(rhs).strip().strip("+"), ")"]),
                    ]
                )
            )
            obj_fun += (obj_fun_row,)

        jac = Matrix(obj_fun).jacobian(Matrix(endo_sym)).tolist()

        obj_fun_lam = Lambdify([*endo_sym, *pred_sym], obj_fun, cse=True)
        jac_lam = Lambdify([*endo_sym, *pred_sym], jac, cse=True)

        def obj_fun_out(val_list, *args):  # type: ignore # noqa
            return obj_fun_lam(*val_list, *args)

        def jac_out(val_list, *args):  # type: ignore # noqa
            return jac_lam(*val_list, *args)

        return None, obj_fun_out, jac_out

    def switch_endo_vars(
        self, old_endo_vars: list[str], new_endo_vars: list[str]
    ) -> None:
        """Sets old_endo_vars as exogenous and new_endo_vars as endogenous and performs block analysis.

        Note:
            This function switches the endogenous and exogenous status of variables and
            performs block analysis on the model.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.switch_endo_vars(['var1', 'var2'], ['var3', 'var4'])

        Args:
            old_endo_vars: List of old endogenous variables to be switched to exogenous.
            new_endo_vars: List of new endogenous variables to be switched from exogenous.

        Raises:
            ValueError: If any variable in `old_endo_vars` is not in the current list
                of endogenous variables or if any variable in `new_endo_vars` is already
                in the list of endogenous variables.
        """
        if all(x in self.endo_vars for x in old_endo_vars) is False:
            raise ValueError("all variables in old_endo_vars are not endogenous")
        if any(x in self.endo_vars for x in new_endo_vars):
            raise ValueError("some variables in new_endo_vars are endogenous")

        print("Analyzing model")
        self._endo_vars = (
            *[x for x in self._endo_vars if x not in old_endo_vars],
            *new_endo_vars,
        )

        (
            self._eqns_endo_vars_match,
            self._condenced_model_digraph,
            self._augmented_condenced_model_digraph,
            self._node_varlist_mapping,
        ) = self._block_analyze_model()

        self._sim_code, self._blocks = self._gen_sim_code_and_blocks()

        print("Finished")

    def find_endo_var(self, endo_var: str, noisy: bool = False) -> int | None:
        """Find the block that solves the specified endogenous variable.

        Note:
            This function searches for the specified endogenous variable in the model's
            blocks and returns the block number of the block that solves it. If the endogenous
            variable is not found in any block, it returns None.

        Args:
            endo_var: The endogenous variable to be found.
            noisy: Whether output should be printed or returned.

        Returns:
            The block number of the block that solves the specified endogenous variable.
            Returns `None` if the endogenous variable is not found in any block.

        Raises:
            IndexError: If `endo_var` is not endogenous in model.
        """
        block = [key for key, val in self._blocks.items() if endo_var.lower() in val[0]]
        if block:
            if noisy:
                print(block[0])
                return None
            else:
                return block[0]
        else:
            raise IndexError(f"{endo_var} is not endogenous in model")

    def describe(self) -> None:
        """Display a summary of the model's characteristics.

        Prints information about the model, including the number of equations, blocks,
        simple definition blocks, and the distribution of equation counts in the blocks.
        """
        print("-" * 100)
        print(
            f"Model consists of {len(self.eqns)} equations in {len(self._blocks)} blocks"
        )
        print(
            f"{len([val[3] for _, val in self._blocks.items() if val[3]])} of the blocks consist of simple definitions\n"
        )
        for key, val in Counter(
            sorted([len(val[2]) for _, val in self._blocks.items()])
        ).items():
            print(f"{val} blocks have {key} equations")
        print("-" * 100)

    def show_blocks(self) -> None:
        """Prints endogenous and exogenous variables and equations for every block in the model.

        Iterates through all blocks in the model and calls the `show_block` function to display their details.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.show_blocks()

            The output is like this::

                --------------------------------------------------
                Block 1
                --------------------------------------------------
                Endogenous Variables:
                - var1
                - var2

                Exogenous Variables:
                - exog_var1
                - exog_var2

                Equations:
                - eqn1: var1 = exog_var1 + exog_var2
                - eqn2: var2 = var1 + exog_var2

                --------------------------------------------------
                Block 2
                --------------------------------------------------
                Endogenous Variables:
                - var3
                - var4

                Exogenous Variables:
                - exog_var3
                - exog_var4

                Equations:
                - eqn3: var3 = exog_var3 + exog_var4
                - eqn4: var4 = var3 + exog_var4

                ...

                --------------------------------------------------
                Block n
                --------------------------------------------------
                Endogenous Variables:
                - var_n1
                - var_n2

                Exogenous Variables:
                - exog_var_n1
                - exog_var_n2

                Equations:
                - eqn_n1: var_n1 = exog_var_n1 + exog_var_n2
                - eqn_n2: var_n2 = var_n1 + exog_var_n2

        """
        for key, _ in self._blocks.items():
            print(" ".join(["-" * 50, "Block", str(key), "-" * 50]))
            self.show_block(key)

    def show_block(self, i: int) -> None:
        """Prints endogenous and exogenous variables and equations for a given block.

        Args:
            i: The index of the block to display.

        Raises:
            IndexError: If block `i` is not in model.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.show_block(1)

            A block consists of an equation or a system of equations::

                5 endogenous variables:
                - var1
                - var2
                - var3
                - var4
                - var5

                3 predetermined variables:
                - pred_var1
                - pred_var2
                - pred_var3

                4 equations:
                - eqn1: var1 = pred_var1 + pred_var2
                - eqn2: var2 = var1 + pred_var2
                - eqn3: var3 = pred_var2 + pred_var3
                - eqn4: var4 = var3 + pred_var1
        """
        block = self._blocks.get(i)

        if block:
            print(
                " ".join(
                    [
                        "Block consists of",
                        (
                            "a definition"
                            if block[3]
                            else "an equation or a system of equations"
                        ),
                    ]
                )
            )
            print(f"\n{len(block[0])} endogenous variables:")
            print("\n".join([" ".join(x) for x in list(self._chunks(block[0], 25))]))
            print(f"\n{len(block[1])} predetermined variables:")
            print(
                "\n".join(
                    [
                        " ".join(x)
                        for x in list(
                            self._chunks([self._var_mapping[x] for x in block[1]], 25)
                        )
                    ]
                )
            )
            print(f"\n{len(block[2])} equations:")
            print("\n".join(block[2]))
        else:
            raise IndexError(f"block {i} is not in model")

    def solve_model(self, input_df: pd.DataFrame, jit: bool = True) -> pd.DataFrame:
        """Solves the model subject to a given DataFrame.

        Args:
            input_df: A DataFrame containing input data for the model.
            jit: Flag indicating whether to use just-in-time (JIT) compilation for solving equations.

        Returns:
            A DataFrame containing the model's output data.

        Raises:
            TypeError: If any column in `input_df` is not of numeric data type.

        Example:
            >>> equations = ["x1 = a1", "x2 = a2", "0.2*x1+0.7*x2 = 0.1*ca+0.8*cb+0.3*i1", "0.8*x1+0.3*x2 = 0.9*ca+0.2*cb+0.1*i2", "k1 = k1(-1)+i1", "k2 = k2(-1)+i2"]
            >>> endogenous = ["x1", "x2", "ca", "cb", "k1", "k2"]
            >>> model = ModelSolver(equations, endogenous)
            ----------------------------------------------------------------------------------------------------
            Initializing model
            * Importing equations
            * Importing endogenous variables
            * Analyzing model
                    * Analyzing equation strings
                    * Generating bipartite graph (BiGraph) connecting equations and endogenous variables
                    * Finding maximum bipartite match (MBM) (i.e. associating every equation with exactly one endogenus variable)
                    * Generating directed graph (DiGraph) connecting endogenous variables using bipartite graph and MBM
                    * Finding condensation of DiGraph (i.e. determining minimal blocks of systems of simulataneous equations)
                    * Generating simulation code (i.e. block-wise symbolic objective function, symbolic Jacobian matrix and lists of endogenous and exogenous variables)
            Finished
            ----------------------------------------------------------------------------------------------------
            >>> input_data = pd.DataFrame({"x1": [2, 4, 1, 2], "x2": [2, 1, 2, 3], "ca": [1, 3, 4, 1], "cb": [1, 2, 1, 4], "k1": [1, 3, 4, 1], "k2": [1, 2, 1, 4], "a1": [1, 2, 4, 4], "a2": [3, 2, 3, 4], "i1": [1, 2, 4, 4], "i2": [3, 2, 3, 4]})
            >>> output_data = model.solve_model(input_data)
            ----------------------------------------------------------------------------------------------------
            Solving model
                    First period: 1, last period: 3
                    Solving
                    |   |
                     ...
            Finished
            ----------------------------------------------------------------------------------------------------
        """
        if (
            all(np.issubdtype(input_df[x].dtype, np.number) for x in input_df.columns)  # type: ignore
            is False
        ):
            raise TypeError("all columns in input_df must be numeric")

        print("-" * 100)
        print("Solving model")

        output_df = input_df.copy(deep=True).astype(float)
        output_array = output_df.to_numpy(dtype=np.float64)
        var_col_index = {
            var: i for i, var in enumerate(output_df.columns.str.lower().to_list())
        }

        get_var_info = cache(self.gen_get_var_info(var_col_index))

        first_period, last_period = self._max_lag, output_array.shape[0] - 1
        periods = range(first_period, last_period + 1)
        print(
            f"\tFirst period: {output_df.index[first_period]}, last period: {output_df.index[last_period]}"
        )
        print("\tSolving")
        print("".join(["\t|", " " * (last_period - first_period + 1), "|"]))
        print("\t ", end="")

        warnings: list[str] = []

        for period in periods:
            print(".", end="")
            for key, val in self._sim_code.items():
                def_fun, obj_fun, jac, endo_vars, pred_vars, _ = val
                solution = self._solve_block(  # type: ignore
                    def_fun,
                    obj_fun,
                    jac,
                    get_var_info(endo_vars),
                    get_var_info(pred_vars),
                    output_array,
                    period,
                    jit=jit,
                )

                output_array[period, [var_col_index.get(x) for x in endo_vars]] = (
                    solution.get("x")
                )

                if solution.get("status") == 1:
                    warnings += (
                        f"Maximum number of iterations reached for block {key} in {input_df.index[period]}",
                    )
                if solution.get("status") == 2:
                    break
            else:
                continue
            break

        if len(warnings) > 0:
            print("\n")
            print("\n".join(warnings))
            print(f"Consider increasing max_iter from {self._max_iter}")

        self._last_solution = output_df

        if solution.get("status") == 2:
            print(f"\nFailed to solve block {key} in {input_df.index[period]}")
            self.show_block_vals(key, period)
            self.trace_to_exog_vals(key, period)
        else:
            print("\nFinished")

        print("-" * 100)

        return output_df.iloc[self.max_lag :, :]

    # Solves one block of the model for a given time period
    def _solve_block(  # type: ignore
        self,
        def_fun,  # noqa: ANN001
        obj_fun,  # noqa: ANN001
        jac,  # noqa: ANN001
        endo_vars_info,  # noqa: ANN001
        pred_vars_info,  # noqa: ANN001
        output_array,  # noqa: ANN001
        period,  # noqa: ANN001
        jit,  # noqa: ANN001
    ):
        (
            _,
            endo_vars_lags,
            endo_vars_cols,
        ) = endo_vars_info
        (
            _,
            pred_vars_lags,
            pred_vars_cols,
        ) = pred_vars_info

        # If block contains a definition this is calculated
        # Othwewise the objective function is sent to Newton-Raphson
        if def_fun:
            solution = {}
            try:
                solution["x"] = def_fun(
                    tuple(
                        self._get_vals(
                            output_array, pred_vars_cols, pred_vars_lags, period, jit
                        )
                    )
                )
                solution["status"] = 0
            except ZeroDivisionError:
                solution["x"] = np.nan
                solution["status"] = 2
        else:
            solution = self._newton_raphson(
                obj_fun,
                self._get_vals(
                    output_array, endo_vars_cols, endo_vars_lags, period, jit
                ),
                args=tuple(
                    self._get_vals(
                        output_array, pred_vars_cols, pred_vars_lags, period, jit
                    )
                ),
                jac=jac,
                tol=self._root_tolerance,
                maxiter=self.max_iter,
            )
            if all(np.isfinite(solution.get("x"))) is False:  # type: ignore
                solution["status"] = 2

        return solution

    # Gets values from DataFrame via array view for speed
    # If shape of request > 0 then the request is sent to njit'ed method for speed
    def _get_vals(
        self,
        array: NDArray[np.float64],
        cols: NDArray[np.int64],
        lags: NDArray[np.int64],
        period: int,
        jit: bool,
    ) -> NDArray[np.float64]:
        if cols.shape[0] == 0:
            return np.array([], np.float64)

        if any([period - x < 0 for x in lags]):
            raise IndexError("period is out of range")
        else:
            if jit:
                return self._get_vals_jit(array, cols, lags, period)  # type: ignore
            else:
                return self._get_vals_nojit(array, cols, lags, period)

    # Gets values from DataFrame via array view
    # Some weird stuff had to be implemented for njit to stop complaining
    # Not sure if njit increases efficiency
    @staticmethod
    @njit  # type: ignore
    def _get_vals_jit(
        array: NDArray[np.float64],
        cols: NDArray[np.int64],
        lags: NDArray[np.int64],
        period: int,
    ) -> NDArray[np.float64]:
        vals = np.array([0.0], dtype=np.float64)
        for col, lag in zip(cols, lags):  # noqa: B905
            vals = np.append(vals, array[period - lag, col])
        return vals[1:]

    # Gets values from DataFrame via array view
    # Runs if user sets jit to False
    @staticmethod
    def _get_vals_nojit(
        array: NDArray[np.float64],
        cols: NDArray[np.int64],
        lags: NDArray[np.int64],
        period: int,
    ) -> NDArray[np.float64]:
        vals = np.array([], dtype=np.float64)
        for col, lag in zip(cols, lags):  # noqa: B905
            vals = np.append(vals, array[period - lag, col])
        return vals

    # Solves root finding problem using simple Newton-Raphson method
    @staticmethod
    def _newton_raphson(
        f: Callable[[Iterable[Any], Any], NDArray[Any]],
        init: NDArray[np.float64],
        args: tuple[NDArray[np.float64]],
        jac: Callable[[Iterable[Any], Any], NDArray[Any]],
        tol: float,
        maxiter: int,
    ) -> dict[str, NDArray[np.float64] | bool | int]:
        success = True
        status = 0
        x_i = init
        f_i = np.array(f(init.tolist(), *args))
        i = 0
        while np.max(np.abs(f_i)) > 0:
            if i == maxiter:
                success = False
                status = 1
                break
            if all(np.isfinite(np.array(args))) is False:
                success = False
                status = 2
                break
            try:
                x_i_new = x_i - np.matmul(
                    np.linalg.inv(np.array(jac(x_i.tolist(), *args))), f_i
                )
                if np.max(np.abs(x_i_new - x_i)) <= tol:
                    break
                x_i = x_i_new
            except np.linalg.LinAlgError:
                success = False
                status = 2
                break
            f_i = np.array(f(x_i, *args))
            i += 1

        return {"x": x_i, "fun": f_i, "success": success, "status": status}

    def draw_blockwise_graph(
        self,
        variable: str,
        max_ancs_gens: int = 5,
        max_desc_gens: int = 5,
        max_nodes: int = 50,
        figsize: tuple[float, float] = (7.5, 7.5),
    ) -> None:
        """Draws a directed graph of a block containing the given variable with a limited number of ancestors and descendants.

        Args:
            variable: The variable for which the blockwise graph will be drawn.
            max_ancs_gens: Maximum number of generations of ancestors to include in the graph.
            max_desc_gens: Maximum number of generations of descendants to include in the graph.
            max_nodes: Maximum number of nodes to include in the graph. If the graph has more nodes, it won't be plotted.
            figsize: A tuple specifying the width and height of the figure for the graph.


        Example:
            Draws a directed graph of the block containing 'var1' with up to 3 generations of ancestors and 2 generations of descendants.

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.draw_blockwise_graph('var1', max_ancs_gens=3, max_desc_gens=2, max_nodes=30, figsize=(10, 10))
        """
        var_node = self._find_var_node(variable.lower())

        ancs_nodes = nx.ancestors(self._augmented_condenced_model_digraph, var_node)
        desc_nodes = nx.descendants(self._augmented_condenced_model_digraph, var_node)

        max_ancr_nodes = {
            x
            for x in ancs_nodes
            if nx.shortest_path_length(
                self._augmented_condenced_model_digraph, x, var_node
            )
            <= max_ancs_gens
        }
        max_desc_nodes = {
            x
            for x in desc_nodes
            if nx.shortest_path_length(
                self._augmented_condenced_model_digraph, var_node, x
            )
            <= max_desc_gens
        }

        subgraph = self._augmented_condenced_model_digraph.subgraph(
            {var_node}.union(max_ancr_nodes).union(max_desc_nodes)
        )
        graph_to_plot = nx.DiGraph()

        print(
            " ".join(
                [
                    f"Graph of block containing {variable} with <={max_ancs_gens} generations of ancestors and <={max_desc_gens} generations of decendants:",
                    str(subgraph),
                ]
            )
        )

        # Loop over all nodes in subgraph (chosen variable, it's ancestors and decendants) and make nodes and edges in pyvis subgraph
        mapping = {}
        for node in subgraph.nodes():
            if node in self._condenced_model_digraph:
                node_label = (
                    "\n".join(self._node_varlist_mapping[node])
                    if len(self._node_varlist_mapping[node]) < 10
                    else "***\nHUGE BLOCK\n***"
                )
                node_title = "<br>".join(self._node_varlist_mapping[node])
                if node == var_node:
                    node_size = 200
                    node_color = "#c4351c"
                if node in max_ancr_nodes:
                    node_size = 100
                    node_color = "#00824d"
                if node in max_desc_nodes:
                    node_size = 100
                    node_color = "#006cb6"
            else:
                node_label = None
                node_title = None
                node_size = 100
                node_color = "#6f9090"

            graph_to_plot.add_node(
                node,
                label=node_label,
                title=node_title,
                shape="circle",
                size=node_size,
                color=node_color,
            )

            if isinstance(node, int):
                mapping[node] = ":\n".join(
                    [
                        " ".join(["Block", str(len(self._blocks) - node)]),
                        (
                            "\n".join(self._node_varlist_mapping[node])
                            if len(self._node_varlist_mapping[node]) < 5
                            else "..."
                        ),
                    ]
                )
            else:
                mapping[node] = node

        if graph_to_plot.number_of_nodes() > max_nodes:
            print("Graph is too big to meaningfully plot")
            return

        graph_to_plot.add_edges_from(subgraph.edges())

        plt.figure(figsize=figsize)
        colors = [node[1]["color"] for node in graph_to_plot.nodes(data=True)]
        layout = nx.shell_layout(graph_to_plot)
        nx.draw(
            graph_to_plot,
            with_labels=True,
            labels=mapping,
            pos=layout,
            node_size=5000,
            node_color=colors,
            font_size=9,
            font_color="white",
            arrowsize=25,
            edge_color="#274247",
            width=2.0,
        )
        plt.plot()

    # Returns the node in  which var is endogenous
    def _find_var_node(self, var_: str) -> int:
        if any(
            [
                var_ in self._condenced_model_digraph.nodes[x]["members"]
                for x in self._condenced_model_digraph.nodes()
            ]
        ):
            var_node = [
                var_ in self._condenced_model_digraph.nodes[x]["members"]
                for x in self._condenced_model_digraph.nodes()
            ].index(True)
        # TODO: Check the old code commented out below. It must be wrong, returning
        # an integer below and a string above. Nodes are integers and comparing
        # with a string does not make sense.
        # elif var_ in self._augmented_condenced_model_digraph.nodes():
        #     var_node = var_
        elif any(
            [
                var_ in self._augmented_condenced_model_digraph.nodes[x]["members"]
                for x in self._augmented_condenced_model_digraph.nodes()
            ]
        ):
            var_node = [
                var_ in self._augmented_condenced_model_digraph.nodes[x]["members"]
                for x in self._augmented_condenced_model_digraph.nodes()
            ].index(True)
        else:
            raise NameError("variable is not in model")
        return var_node

    def trace_to_exog_vars(self, i: int, noisy: bool = True) -> list[str] | None:
        """Prints all exogenous variables that are ancestors to the given block.

        Args:
            i: The index of the block for which variable values will be displayed.
            noisy: Whether output should be printed or returned.

        Returns:
            A list of exogenous variables that are ancestors to the given block, or `None`.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.trace_to_exog_vars(1)
        """
        if noisy:
            print(
                "\n".join(
                    [
                        " ".join(x)
                        for x in list(self._chunks(self._trace_to_exog_vars(i), 10))
                    ]
                )
            )
            return None
        else:
            return self._trace_to_exog_vars(i)

    ## Finds all exogenous variables that are ancestors to block
    def _trace_to_exog_vars(self, i: int) -> list[str]:
        if i < 1:
            raise IndexError("block must be >=1")

        var_node = len(self._blocks) - i
        if var_node < 0:
            raise IndexError(f"block {i} is not in model")

        ancs_nodes = nx.ancestors(self._augmented_condenced_model_digraph, var_node)

        ancs_exog_vars: tuple[str, ...] = ()
        for node in ancs_nodes:
            if len(
                nx.ancestors(self._augmented_condenced_model_digraph, node)
            ) == 0 and self._node_varlist_mapping.get(node):
                ancs_exog_vars += self._node_varlist_mapping[node]

        return [x for x in ancs_exog_vars if x not in self.endo_vars]

    def trace_to_exog_vals(
        self, i: int, period_index: int, noisy: bool = True
    ) -> pd.Series | None:
        """Traces the given block back to exogenous values and prints those values.

        Args:
            i: The index of the block for which variable values will be displayed.
            period_index: The index of the period for which exogenous values will be traced.
            noisy: Whether output should be printed or returned.

        Returns:
            A `pd.Series` of exogenous values, or `None`.

        Raises:
            RuntimeError: If no solution is found.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.trace_to_exog_vals(1, 3)

            With output::

                Block 1 traces back to the following exogenous variable values in 2023-01-04:
                exog_var1=12.5
                exog_var2=8.2
                exog_var3=10.0
        """
        try:
            output_array = self._last_solution.to_numpy(dtype=np.float64, copy=True)
            var_col_index = {
                var: i
                for i, var in enumerate(
                    self._last_solution.columns.str.lower().to_list()
                )
            }

            get_var_info = self.gen_get_var_info(var_col_index)

            ancs_exog_vars = self._trace_to_exog_vars(i)
            if ancs_exog_vars:
                (
                    _,
                    ancs_exog_lags,
                    ancs_exog_cols,
                ) = get_var_info(self._var_mapping[x] for x in ancs_exog_vars)
                ancs_exog_vals = self._get_vals(
                    output_array, ancs_exog_cols, ancs_exog_lags, period_index, False
                )
                if noisy:
                    print(
                        f"\nBlock {i} traces back to the following exogenous variable values in {self._last_solution.index[period_index]}:"
                    )
                    print(
                        *[
                            "=".join([x, str(y)])
                            for x, y in zip(ancs_exog_vars, ancs_exog_vals, strict=True)
                        ],
                        sep="\n",
                    )
                    return None
                else:
                    return pd.Series(ancs_exog_vals, index=ancs_exog_vars)

            return None

        except AttributeError as exc:
            raise RuntimeError(self._NO_SOLUTION_TEXT) from exc

    def show_block_vals(
        self, i: int, period_index: int, noisy: bool = True
    ) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
        """Prints the values of endogenous and predetermined variables in a given block for a specific period.

        Args:
            i: The index of the block for which variable values will be displayed.
            period_index: The index of the period for which variable values will be shown.
            noisy: Whether output should be printed or returned.

        Returns:
            Two pd.Series of endogenous and predetermined values, or `None`, `None`.

        Raises:
            RuntimeError: If no solution is found.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                model.show_block_vals(1, 3)

            With output::

                Block 1 has endogenous variables in 2023-01-04 that evaluate to:
                var1=10.5
                var2=15.2
                ...
                Block 1 has predetermined variables in 2023-01-04 that evaluate to:
                pred_var1=8.1
                pred_var2=9.7
        """
        try:
            output_array = self._last_solution.to_numpy(dtype=np.float64, copy=True)
            var_col_index = {
                var: i
                for i, var in enumerate(
                    self._last_solution.columns.str.lower().to_list()
                )
            }

            get_var_info = self.gen_get_var_info(var_col_index)

            block = self._blocks[i]

            _, block_endo_lags, block_endo_cols = get_var_info(block[0])
            block_endo_vals = self._get_vals(
                output_array, block_endo_cols, block_endo_lags, period_index, False
            )

            _, block_pred_lags, block_pred_cols = get_var_info(block[1])
            block_pred_vals = self._get_vals(
                output_array, block_pred_cols, block_pred_lags, period_index, False
            )
            if noisy:
                print(
                    f"\nBlock {i} has endogenous variables in {self._last_solution.index[period_index]} that evaluate to:"
                )
                print(
                    *[
                        "=".join([x, str(y)])
                        for x, y in zip(
                            [self._var_mapping[x] for x in block[0]],
                            block_endo_vals,
                            strict=True,
                        )
                    ],
                    sep="\n",
                )
                print(
                    f"\nBlock {i} has predetermined variables in {self._last_solution.index[period_index]} that evaluate to:"
                )
                print(
                    *[
                        "=".join([x, str(y)])
                        for x, y in zip(
                            [self._var_mapping[x] for x in block[1]],
                            block_pred_vals,
                            strict=True,
                        )
                    ],
                    sep="\n",
                )
                return None, None
            else:
                return (
                    pd.Series(
                        block_endo_vals,
                        index=[self._var_mapping.get(x) for x in block[0]],
                    ),
                    pd.Series(
                        block_pred_vals,
                        index=[self._var_mapping.get(x) for x in block[1]],
                    ),
                )

        except AttributeError as exc:
            raise RuntimeError(self._NO_SOLUTION_TEXT) from exc

    def gen_get_var_info(
        self, var_col_index: dict[str, int]
    ) -> Callable[
        [Iterable[str]], tuple[list[str], NDArray[np.int64], NDArray[np.int64]]
    ]:
        """Function that returns function that returns names, columns and lags for variables."""

        def get_var_info(
            vars_: Iterable[str],
        ) -> tuple[list[str], NDArray[np.int64], NDArray[np.int64]]:
            if not vars_:
                return [], np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            # Stole zip-solution from: https://stackoverflow.com/questions/21444338/transpose-nested-list-in-python
            names, lags = tuple(
                map(list, zip(*[self._lag_mapping.get(x) for x in vars_], strict=True))
            )
            cols = tuple(var_col_index.get(x) for x in names)
            if any(x is None for x in cols):
                missing = [x for x, y in zip(names, cols, strict=True) if y is None]
                raise KeyError(f'{",".join(missing)} is not in DataFrame')
            return names, np.array(lags, dtype=np.int64), np.array(cols, dtype=np.int64)

        return get_var_info

    def sensitivity(
        self,
        i: int,
        period_index: int,
        method: str = "std",
        exog_subset: list[str] | None = None,
    ) -> pd.DataFrame:
        """Analyses sensitivity of endogenous variables to exogenous variables for a specific period.

        Args:
            i: The index of the block for which variable values will be displayed.
            period_index: The index of the period for which variable values will be shown.
            exog_subset: List of exogenous variables to be analysed.
                If `None`, all relevant exogenous variables will be analysed.
            method: Method for sensitivity analysis. Default is 'std'.

                - 'std': Adjusts variables by adding their standard deviation.
                - 'pct': Adjusts variables by adding 1% of their value.
                - 'one': Adjusts variables by adding 1 to their value.

        Returns:
            DataFrame showing the sensitivity of endogenous variables to exogenous variables.

        Raises:
            RuntimeError: If no solution is found.
            ValueError: If `method` is not `std`, `pct` or `one`.

        Example:

            .. code-block:: python

                model = ModelSolver(equations, endogenous)
                sensitivity_df = model.sensitivity(1, 3, method='pct', exog_subset=['exog_var1', 'exog_var2'])
                print(sensitivity_df)

            With output::

                           | endog_var1 | endog_var2 |
                --------------------------------------
                exog_var1  |    0.23    |    0.12    |
                exog_var2  |    0.45    |    0.56    |
        """
        try:
            var_col_index = {
                var: i
                for i, var in enumerate(
                    self._last_solution.columns.str.lower().to_list()
                )
            }
        except AttributeError as exc:
            raise RuntimeError(self._NO_SOLUTION_TEXT) from exc

        get_var_info = cache(self.gen_get_var_info(var_col_index))

        exog_vars = self._trace_to_exog_vars(i)

        if exog_subset:
            exog_vars = [x for x in exog_vars if x in exog_subset]

        n_exog_vars = len(exog_vars)
        div = min(10, n_exog_vars)

        result = {}

        div = min(10, n_exog_vars)
        print(
            f"Analysing sensitivity for block {i} in period {self._last_solution.index[period_index]}"
        )
        print(f"Number exogeneous variables to analyse: {len(exog_vars)}")
        print(
            "".join(
                [
                    "\t|",
                    " "
                    * sum(
                        [j % int(n_exog_vars / div) == 0 for j in range(len(exog_vars))]
                    ),
                    "|",
                ]
            )
        )
        print("\t ", end="")
        for j, exog_var in enumerate(exog_vars):
            if j % int(n_exog_vars / div) == 0:
                print(".", end="")

            var, lag = self._lag_mapping[self._var_mapping[exog_var]]
            solution_diff = self._last_solution.copy()

            if method == "std":
                solution_diff[var].iloc[period_index - lag] += solution_diff[var].std()
            elif method == "pct":
                solution_diff[var].iloc[period_index - lag] += (
                    solution_diff[var].iloc[period_index - lag] * 0.01
                )
            elif method == "one":
                solution_diff[var].iloc[period_index - lag] += 1
            else:
                raise ValueError("method must be std, pct or one")

            output_array = solution_diff.to_numpy(dtype=np.float64)

            for key, val in self._sim_code.items():
                def_fun, obj_fun, jac, endo_vars, pred_vars, _ = val

                solution = self._solve_block(  # type: ignore
                    def_fun,
                    obj_fun,
                    jac,
                    get_var_info(endo_vars),
                    get_var_info(pred_vars),
                    output_array,
                    period_index,
                    jit=False,
                )

                output_array[
                    period_index, [var_col_index.get(x) for x in endo_vars]
                ] = solution.get("x")

                if key == i:
                    result[exog_var] = solution.get("x")
                    break

        return (
            pd.DataFrame(result, index=endo_vars).T
            - self._last_solution[list(endo_vars)].iloc[period_index]
        )

    # Stole solution from https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    @staticmethod
    def _chunks(xs: Sequence[str], n: int) -> Generator[Sequence[str], None, None]:
        n = max(1, n)
        return (xs[i : i + n] for i in range(0, len(xs), n))
