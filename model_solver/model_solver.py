##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

import numpy as np
import networkx as nx
import pandas as pd
from symengine import var, Matrix, Lambdify, Max, Min, log, exp
import matplotlib.pyplot as plt
from collections import Counter
from functools import cache
from numba import njit


class ModelSolver:
    """
    EXAMPLE OF USE USE:

    Let "equations" and "endogenous" be lists containing equations and endogenous variables, respectively, stored as strings, e.g.

        equations = [
            'x+y=A',
            'x/y=B'
        ]
        endogenous = [
            'x',
            'y'
        ]

    where 'A' and 'B' are exogenous variables
    The solver supports the mathematical functions min, max, log and exp

    A class instance called "model" is initialized by

        model = ModelSolver(equations, endogenous)

    This reads in the equations and endogenous variables and perform block analysis and ordering and generates simulation code
    Upon completion, the model is ready to be solved subject to data (exogenous and initial values of endogenous variables) in a Pandas DataFrame
    Let "input_df" be a dataframe containing data on A and B and initial values for x and y. Then the model can be solved by invoking

        solution_df = model.solve_model(input_df)

    Now "solution_df" is a Pandas DataFrame with exactly the same dimensions as "input_df", but where the endogenous variables are replaced by the solutions to the model
    The last solution is also stored in "model.last_solution"

    ModelSolver also has a number of methods for analysis (TBA)
    """

    # Reads in equations and endogenous variables and does a number of operations, e.g. analyzing block structure using graph theory.
    def __init__(self, eqns: list[str], endo_vars: list[str]):
        self._some_error = False
        self._lag_notation = '__LAG'
        self._max_lag = 0
        self._root_tolerance = 1e-7
        self._max_iter = 10

        print('-'*100)
        print('Initializing model...')

        # Model equations and endogenous variables are checked and stored as immutable tuples (as opposed to mutable lists)
        self._eqns, self._endo_vars = self._init_model(eqns, endo_vars)

        print('* Analyzing model...')

        # Analyzing equation strings to determine variables, lags and coefficients
        self._eqns_analyzed, self._var_mapping, self._lag_mapping = self._analyze_eqns()

        # Perform block analysis and ordering of equations
        self._eqns_endo_vars_match, self._condenced_model_digraph, self._augmented_condenced_model_digraph, self._node_varlist_mapping = self._block_analyze_model()

        # Generating everything needed to simulate model
        self._sim_code, self._blocks = self._gen_sim_code_and_blocks()

        print('Finished')
        print('-'*100)


    @property
    def eqns(self):
        return self._eqns

    @property
    def endo_vars(self):
        return self._endo_vars

    @property
    def exog_vars(self):
        vars = set()
        for _, _, _, lag_mapping in self._eqns_analyzed:
            for _, val in lag_mapping.items():
                vars.update((val[0],))
        return tuple(vars.difference(self.endo_vars))

    @property
    def max_lag(self):
        return self._max_lag

    @property
    def root_tolerance(self):
        return self._root_tolerance

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def last_solution(self):
        try:
            return self._last_solution.iloc[self.max_lag:, :]
        except AttributeError:
            print('No solution exists')


    @root_tolerance.setter
    def root_tolerance(self, val: float):
        if isinstance(val, float) is False:
            raise ValueError('Tolerance for termination must be of type float')
        if val <= 0:
            raise ValueError('Tolerance for termination must be positive')
        self._root_tolerance = val

    @max_iter.setter
    def max_iter(self, val: int):
        if isinstance(val, int) is False:
            raise ValueError('Maximum number of iterations must be an integer')
        if val < 0:
            raise ValueError('Maximum number of iterations cannot be negative')
        self._max_iter = val


    # Imports lists containing equations and endogenous variables stored as strings
    # Checks that there are no blank lines, sets everything to lowercase and returns as tuples
    def _init_model(self, eqns: list, endo_vars: list):
        print('* Importing equations')
        for i, eqn in enumerate(eqns):
            if eqn.strip() == '':
                self._some_error = True
                raise ValueError('There are empty elements in equation list')
            eqns[i] = eqns[i].lower()

        print('* Importing endogenous variables')
        for endo_var in endo_vars:
            if endo_var.strip() == '':
                self._some_error = True
                raise ValueError('There are empty elements in endogenous variable list')
            endo_vars[i] = endo_vars[i].lower()

        return tuple(eqns), tuple(endo_vars)


    # Analyzes the equations of the model
    def _analyze_eqns(self):
        if self._some_error:
            return None, None

        print('\t* Analyzing equation strings')

        eqns_analyzed = []

        var_mapping, lag_mapping = {}, {}
        for eqn in self._eqns:
            eqn_analyzed = (eqn, *self._analyze_eqn(eqn))
            eqns_analyzed += eqn_analyzed,
            var_mapping = {**var_mapping, **eqn_analyzed[2]}
            lag_mapping = {**lag_mapping, **eqn_analyzed[3]}

        return eqns_analyzed, var_mapping, lag_mapping


    # Takes an equation string and parses it into coefficients (special care is taken to deal with scientific notation), variables, lags and operators/brackets
    # I've written my own parser in stead of using some existing because it needs to take care of then (-)-notation for lags
    def _analyze_eqn(self, eqn: str):
        if self._some_error:
            return

        parsed_eqn_with_lag_notation, var_mapping, lag_mapping = [], {}, {}
        component, lag = '', ''
        is_num, is_var, is_lag, is_sci = False, False, False, False

        for chr in ''.join([eqn, ' ']):
            is_num = (chr.isnumeric() and not is_var) or is_num
            is_var = (chr.isalpha()  and not is_num) or is_var
            is_lag = (is_var and chr == '(') or is_lag
            is_sci = (is_num and chr == 'e') or is_sci

            if (is_var and chr == '(' and component in ['max', 'min', 'log', 'exp']):
                parsed_eqn_with_lag_notation += ''.join([component, chr])
                is_var, is_lag = False, False
                component, lag = '', ''
                continue

            # Check if character is something other than a numeric, variable or lag and write numeric or variable to parsed equation
            if chr in ['=','+','-','*','/','(',')',',',' '] and not (is_lag or is_sci):
                if is_num:
                    parsed_eqn_with_lag_notation += str(component),
                if is_var:
                    # Replace (-)-notation by LAG_NOTATION for lags and appends _ to the end to mark the end
                    pfx = '' if lag == '' else ''.join([self._lag_notation, str(-int(lag[1:-1])), '_'])
                    parsed_eqn_with_lag_notation += ''.join([component, pfx]),
                    var_mapping[''.join([component, lag])] = ''.join([component, pfx])
                    var_mapping[''.join([component, pfx])] = ''.join([component, lag])
                    lag_mapping[''.join([component, pfx])] = (component, 0 if lag == '' else -int(lag[1:-1]))
                    if lag != '':
                        self._max_lag = max(self._max_lag, -int(lag.replace('(', '').replace(')', '')))
                if chr != ' ':
                    parsed_eqn_with_lag_notation += chr,
                component, lag = '', ''
                is_num, is_var, is_lag = False, False, False
                continue

            if is_sci and chr.isnumeric():
                is_sci = False

            if is_num:
                component = ''.join([component, chr])
                continue

            if is_var and not is_lag:
                component = ''.join([component, chr])
                continue

            if is_var and is_lag:
                lag = ''.join([lag, chr])
                if chr == ')':
                    is_lag = False

        return parsed_eqn_with_lag_notation, var_mapping, lag_mapping


    # Performs block analysis of equations subject to endogenous variables
    # Analysis is a sequence of operations using graph theory
    def _block_analyze_model(self):
        # Using graph theory to analyze equations using existing algorithms to establish minimum simultaneous blocks
        eqns_endo_vars_bigraph = self._gen_eqns_endo_vars_bigraph()
        eqns_endo_vars_match = self._find_max_bipartite_match(eqns_endo_vars_bigraph)
        model_digraph = self._gen_model_digraph(eqns_endo_vars_bigraph, eqns_endo_vars_match)
        condenced_model_digraph, condenced_model_node_varlist_mapping = self._gen_condenced_model_digraph(model_digraph)
        augmented_condenced_model_digraph, augmented_condenced_model_node_varlist_mapping = self._gen_augmented_condenced_model_digraph(condenced_model_digraph, eqns_endo_vars_match)
        
        node_varlist_mapping = {**condenced_model_node_varlist_mapping, **augmented_condenced_model_node_varlist_mapping}

        return eqns_endo_vars_match, condenced_model_digraph, augmented_condenced_model_digraph, node_varlist_mapping


    # Generates bipartite graph (bigraph) connetcting equations (nodes in U) with endogenous variables (nodes in V)
    # See https://en.wikipedia.org/wiki/Bipartite_graph for a discussion of bigraphs
    def _gen_eqns_endo_vars_bigraph(self):
        if self._some_error:
            return

        print('\t* Generating bipartite graph connecting equations and endogenous variables')

        # Make nodes in bipartite graph with equations U (0) and endogenous variables in V (1)
        eqns_endo_vars_bigraph = nx.Graph()
        eqns_endo_vars_bigraph.add_nodes_from([i for i, _ in enumerate(self.eqns)], bipartite=0)
        eqns_endo_vars_bigraph.add_nodes_from(self._endo_vars, bipartite=1)

        # Make edges between equations and endogenous variables
        for i, eqns in enumerate(self._eqns_analyzed):
            for endo_var in [x for x in eqns[2].keys() if x in self.endo_vars]:
                eqns_endo_vars_bigraph.add_edge(i, endo_var)

        return eqns_endo_vars_bigraph


    # Finds a maximum bipartite match (MBM) of bigraph connetcting equations (nodes in U) with endogenous variables (nodes in V)
    # See https://www.geeksforgeeks.org/maximum-bipartite-matching/ for more on MBM
    # Returns dict with matches (maps both ways, i.e. U-->V and U-->U)
    def _find_max_bipartite_match(self, eqns_endo_vars_bigraph):
        if self._some_error:
            return

        print('\t* Finding maximum bipartite match (MBM) (i.e. associating every equation with exactly one endogenus variable)')

        # Use maximum bipartite matching to make a one to one mapping between equations and endogenous variables
        try:
            maximum_bipartite_match = nx.bipartite.maximum_matching(eqns_endo_vars_bigraph, [i for i, _ in enumerate(self._eqns)])
            if len(maximum_bipartite_match)/2 < len(self.eqns):
                self._some_error = True
                print('ERROR: Model is over or under spesified')
                return
        except nx.AmbiguousSolution:
            self._some_error = True
            print('ERROR: Unable to analyze model')
            return

        return maximum_bipartite_match


    # Makes a directed graph (digraph) showing how endogenous variables affect every other endogenous variable
    # See https://en.wikipedia.org/wiki/Directed_graph for more about directed graphs
    def _gen_model_digraph(self, eqns_endo_vars_bigraph, eqns_endo_vars_match):
        if self._some_error:
            return

        print('\t* Generating directed graph (DiGraph) connecting endogenous variables using bipartite graph and MBM')

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
    def _gen_condenced_model_digraph(self, model_digraph):
        if self._some_error:
            return

        print('\t* Finding condensation of DiGraph (i.e. finding minimum simulataneous equation blocks)')

        # Generate condensation graph of equation graph such that every node is a strong component of the equation graph
        condenced_model_digraph = nx.condensation(model_digraph)

        # Make a dictionary that associate every node of condensation with a list of variables
        node_vars_mapping = {}
        for node in tuple(condenced_model_digraph.nodes()):
            node_vars_mapping[node] = tuple(condenced_model_digraph.nodes[node]['members'])

        return condenced_model_digraph, node_vars_mapping


    # Augments condenced digraph with nodes and edges for exogenous variables in order to show what exogenous variables affect what strong components
    def _gen_augmented_condenced_model_digraph(self, condenced_model_digraph, eqns_endo_vars_match):
        if self._some_error:
            return

        augmented_condenced_model_digraph = condenced_model_digraph.copy()

        # Make edges between exogenous variables and strong components it is a part of
        node_vars_mapping = {}
        for node in condenced_model_digraph.nodes():
            for member in condenced_model_digraph.nodes[node]['members']:
                for exog_var_adjacent_to_node in [val for key, val in self._eqns_analyzed[eqns_endo_vars_match[member]][2].items()
                                                  if self._lag_notation not in val and key not in self.endo_vars]:
                    augmented_condenced_model_digraph.add_edge(exog_var_adjacent_to_node, node)
                    node_vars_mapping[exog_var_adjacent_to_node] = exog_var_adjacent_to_node,

        return augmented_condenced_model_digraph, node_vars_mapping


    # Generates simulation code and blocks
    # Simulation code contains a tuple of tuples for each strong component
    # The tuple for each strong component contains objective function, and Jacobian matrix, and lists of the variables in the strong component
    def _gen_sim_code_and_blocks(self):
        if self._some_error:
            return

        print('\t* Generating simulation code (i.e. block-wise symbolic objective function, symbolic Jacobian matrix and lists of endogenous and exogenous variables)')

        sim_code, blocks = {}, {}
        for i, node in enumerate(reversed(tuple(self._condenced_model_digraph.nodes()))):
            block_endo_vars, block_eqns_orig, block_eqns_lags, block_exog_vars = tuple(), tuple(), tuple(), set()
            for member in self._condenced_model_digraph.nodes[node]['members']:
                j = self._eqns_endo_vars_match[member]
                eqns_analyzed = self._eqns_analyzed[j]
                block_endo_vars += member,
                block_eqns_orig += eqns_analyzed[0],
                block_eqns_lags += eqns_analyzed[1],
                block_exog_vars.update([val for key, val in eqns_analyzed[2].items() if self._lag_notation not in key])

            block_exog_vars.difference_update(set(block_endo_vars))
            block_exog_vars = tuple(block_exog_vars)

            (def_fun, obj_fun, jac) = self._gen_def_or_obj_fun_and_jac(block_eqns_lags, block_endo_vars, block_exog_vars)
            sim_code[i+1] = (
                def_fun,
                obj_fun,
                jac,
                block_endo_vars,
                block_exog_vars,
                block_eqns_lags
                )
            blocks[i+1] = (
                block_endo_vars,
                block_exog_vars,
                block_eqns_orig,
                True if def_fun else False
                )

        return sim_code, blocks


    # Generates symbolic objective functon and Jacobian matrix for a given strong component
    @staticmethod
    def _gen_def_or_obj_fun_and_jac(eqns: tuple[str],
                                    endo_vars: tuple[str],
                                    exog_vars: tuple[str]
                                    ):
        max, min = Max, Max
        endo_sym, exog_sym, obj_fun = [], [], []
        for endo_var in endo_vars:
            var(endo_var)
            endo_sym += eval(endo_var),
        for exog_var in exog_vars:
            var(exog_var)
            exog_sym += eval(exog_var),
        for eqn in eqns:
            i = eqn.index('=')
            lhs, rhs = eqn[:i], eqn[i+1:]
            if len(eqns) == 1 and endo_var == ''.join(lhs) and endo_var not in rhs:
                if len(exog_vars) == 0:
                    return lambda _: np.array([eval(''.join(rhs).strip().strip('+'))]), None, None
                def_fun = eval(''.join(rhs).strip().strip('+'))
                def_fun_lam = Lambdify([exog_sym], def_fun)
                def_fun_out = lambda args: np.array([def_fun_lam(args)], dtype=np.float64)
                return def_fun_out, None, None

            obj_fun_row = eval('-'.join([''.join(['(', ''.join(lhs).strip().strip('+'), ')']), ''.join(['(', ''.join(rhs).strip().strip('+'), ')'])]))
            obj_fun += obj_fun_row,

        jac = Matrix(obj_fun).jacobian(Matrix(endo_sym)).tolist()

        obj_fun_lam = Lambdify([*endo_sym, *exog_sym], obj_fun, cse=True)
        jac_lam = Lambdify([*endo_sym, *exog_sym], jac, cse=True)

        obj_fun_out = lambda val_list, *args: obj_fun_lam(*val_list, *args)
        jac_out = lambda val_list, *args: jac_lam(*val_list, *args)

        return None, obj_fun_out, jac_out


    def switch_endo_var(self, old_endo_vars: list[str], new_endo_vars: list[str]):
        """
        Sets old_endo_vars as exogenous and new_endo_vars as endogenous and performs block analysis
        """

        if all([x in self.endo_vars for x in old_endo_vars]) is False:
            print('All variables in old_endo_vars are not endogenous')
            return
        if any([x in self.endo_vars for x in new_endo_vars]):
            print('Some variables in new_endo_vars are endogenous')
            return

        print('Analyzing model...')
        self.endo_vars = (*[x for x in self._endo_vars if x not in old_endo_vars], *new_endo_vars)

        self._eqns_endo_vars_match, self._condenced_model_digraph, self._augmented_condenced_model_digraph, self._node_varlist_mapping = self._block_analyze_model()

        self._sim_code, self._blocks = self._gen_sim_code_and_blocks()

        print('Finished')


    def find_endo_var(self, endo_var: str):
        """
        Finds what block solves the given engoenous variable
        """

        block = [key for key, val in self._blocks.items() if endo_var.lower() in val[0]]
        if block:
            return block[0]
        else:
            print('{} is not endogenous in model'.format(endo_var))


    def describe(self):
        """
        Describes model, that is number of equations, number of simultaneous blocks and how many equations are in each block
        """

        print('-'*100)
        print('Model consists of {} equations in {} blocks'
              .format(len(self.eqns), len(self._blocks)))
        print('{} of the blocks consist of simple definitions\n'
              .format(len([val[3] for _, val in self._blocks.items() if val[3]])))
        for key, val in Counter(sorted([len(val[2]) for _, val in self._blocks.items()])).items():
            print('{} blocks have {} equations'.format(val, key))
        print('-'*100)


    def show_blocks(self):
        """
        Prints endogenous and exogenous variables and equations for every block in the model
        """

        for key, _ in self._blocks.items():
            print(' '.join(['-'*50, 'Block', str(key), '-'*50]))
            self.show_block(key)


    def show_block(self, i: int):
        """
        Prints endogenous and exogenous variables and equations for a given block
        """

        block = self._blocks.get(i)
        if block:
            print(' '.join(['Block consists of', 'a definition' if block[3] else 'an equation or a system of equations']))
            print('\n{} endogenous variables:'.format(len(block[0])))
            print('\n'.join([' '.join(x) for x in list(self._chunks(block[0], 25))]))
            print('\n{} predetermined variables:'.format(len(block[1])))
            print('\n'.join([' '.join(x) for x in list(self._chunks([self._var_mapping.get(x) for x in block[1]], 25))]))
            print('\n{} equations:'.format(len(block[2])))
            print('\n'.join(block[2]))
        else:
            print('Block {} is not in model'.format(block))


    def solve_model(self, input_df: pd.DataFrame, jit=True) -> pd.DataFrame:
        """
        Solves the model for a given DataFrame
        """

        if self._some_error:
            return

        print('-'*100)
        print('Solving model...')

        output_df = input_df.astype(float).copy(deep=True)
        output_array = output_df.to_numpy(dtype=np.float64)
        var_col_index = {var: i for i, var in enumerate(output_df.columns.str.lower().to_list())}

        get_var_info = cache(self.gen_get_var_info(var_col_index))

        first_period, last_period = self._max_lag, output_array.shape[0]-1
        periods = range(first_period, last_period+1)
        print('\tFirst period: {}, last period: {}'.format(output_df.index[first_period], output_df.index[last_period]))
        print('\tSolving')
        print(''.join(['\t|', ' '*(last_period-first_period+1), '|']))
        print('\t ', end='')

        for period in periods:
            print('.', end='')
            for key, val in self._sim_code.items():
                (def_fun, obj_fun, jac, endo_vars, pred_vars, _) = val
                solution = self._solve_block(
                    def_fun,
                    obj_fun,
                    jac,
                    get_var_info(endo_vars),
                    get_var_info(pred_vars),
                    output_array,
                    period,
                    jit=jit
                    )

                output_array[period, [var_col_index.get(x) for x in endo_vars]] = solution.get('x')

                if solution.get('status') == 2:
                    print('\nBlock {} consists of the following equations:'.format(key))
                    print(*[x for x in self._blocks.get(key)[2]], sep='\n')
                    ancs_exog_vars = self._trace_to_exog_vars(key)
                    if ancs_exog_vars:
                        _, ancs_exog_lags, ancs_exog_cols, = get_var_info((self._var_mapping.get(x) for x in ancs_exog_vars))
                        ancs_exog_vals = self._get_vals(output_array, ancs_exog_cols, ancs_exog_lags, period, jit)
                        print('\nBlock {} traces back to the following exogenous variable values in {}:'.format(key, input_df.index[period]))
                        print(*['='.join([x, str(y)]) for x, y in zip(ancs_exog_vars, ancs_exog_vals)], sep='\n')
                    raise ValueError('Block {} did not converge'.format(key))
                if solution.get('status') == 1:
                    print('Maximum number of iterations reached for block {} in {}'.format(key, input_df.index[period]))

        print('\nFinished')
        print('-'*100)

        self._last_solution = output_df

        return output_df.iloc[self.max_lag:, :]


    # Solves one block of the model for a given time period
    def _solve_block(self, def_fun, obj_fun, jac, endo_vars_info: tuple, pred_vars_info: tuple, output_array: np.array, period: int, jit: bool):
        endo_vars_names, endo_vars_lags, endo_vars_cols, = endo_vars_info
        pred_vars_names, pred_vars_lags, pred_vars_cols, = pred_vars_info

        # If block contains a definition this is calculated
        # Othwewise the objective function is sent to Newton-Raphson
        if def_fun:
            solution = {}
            try:
                solution['x'] = def_fun(tuple(self._get_vals(output_array, pred_vars_cols, pred_vars_lags, period, jit)))
                solution['status'] = 0
            except ZeroDivisionError:
                solution['x'] = np.nan
                solution['status'] = 2
        else:
            solution = self._newton_raphson(
                obj_fun,
                self._get_vals(output_array, endo_vars_cols, endo_vars_lags, period, jit),
                args = tuple(self._get_vals(output_array, pred_vars_cols, pred_vars_lags, period, jit)),
                jac = jac,
                tol = self._root_tolerance,
                maxiter=self.max_iter
                )
            if all(np.isfinite(solution.get('x'))) is False:
                solution['status'] = 2

        if solution.get('status') == 2:
            endo_vars_vals = self._get_vals(output_array, endo_vars_cols, endo_vars_lags, period, jit)
            exog_vars_vals = self._get_vals(output_array, pred_vars_cols, pred_vars_lags, period, jit)
            print()
            print('\nEndogenous variables in block upon failure:')
            print(*['='.join([x, str(y)]) for x, y in zip(endo_vars_names, endo_vars_vals)], sep='\n')
            print('\nPredetermined variables in block upon failure:')
            print(*['='.join([x, str(y)]) for x, y in zip(pred_vars_names, exog_vars_vals)], sep='\n')

        return solution


    # Gets values from DataFrame via array view for speed
    # If shape of request > 0 then the request is sent to njit'ed method for speed
    def _get_vals(self, array: np.array, cols: np.array, lags: np.array, period: int, jit: bool):
        if cols.shape[0] == 0:
            return np.array([], np.float64)
        else:
            if any([period-x < 0 for x in lags]):
                raise IndexError('Period is out of range')
            else:
                if jit:
                    return self._get_vals_jit(array, cols, lags, period)
                else:
                    return self._get_vals_nojit(array, cols, lags, period)


    # Gets values from DataFrame via array view
    # Some weird stuff had to be implemented for njit to stop complaining
    # Not sure if njit increases efficiency
    @staticmethod
    @njit
    def _get_vals_jit(array: np.array, cols: np.array, lags: np.array, period: int):
        vals = np.array([0.0], dtype=np.float64)
        for col, lag in zip(cols, lags):
            vals = np.append(vals, array[period-lag, col])
        return vals[1:]


    # Gets values from DataFrame via array view
    # Runs if user sets jit to False
    @staticmethod
    def _get_vals_nojit(array: np.array, cols: np.array, lags: np.array, period: int):
        vals = np.array([], dtype=np.float64)
        for col, lag in zip(cols, lags):
            vals = np.append(vals, array[period-lag, col])
        return vals


    # Solves root finding problem using simple Newton-Raphson method
    @staticmethod
    def _newton_raphson(f, init, args=None, jac=None, tol=None, maxiter=None):
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
            try:
                x_i_new = x_i-np.matmul(np.linalg.inv(np.array(jac(x_i.tolist(), *args))), f_i)
                if np.max(np.abs(x_i_new-x_i)) <= tol:
                    break
                x_i = x_i_new
            except np.linalg.LinAlgError:
                success = False
                status = 2
                break
            f_i = np.array(f(x_i, *args))
            i += 1

        return {'x': x_i, 'fun': f_i, 'success': success, 'status': status}


    def draw_blockwise_graph(
            self, var: str,
            max_ancs_gens: int=5,
            max_desc_gens: int=5,
            max_nodes: int=50,
            figsize=(7.5, 7.5)
            ):
        """
        Draws a directed graph of block in which variable is along with max number of ancestors and descendants.
        """

        if self._some_error:
            return

        var_node = self._find_var_node(var.lower())

        ancs_nodes = nx.ancestors(self._augmented_condenced_model_digraph, var_node)
        desc_nodes = nx.descendants(self._augmented_condenced_model_digraph, var_node)

        max_ancr_nodes = {x for x in ancs_nodes if nx.shortest_path_length(self._augmented_condenced_model_digraph, x, var_node) <= max_ancs_gens}
        max_desc_nodes = {x for x in desc_nodes if nx.shortest_path_length(self._augmented_condenced_model_digraph, var_node, x) <= max_desc_gens}

        subgraph = self._augmented_condenced_model_digraph.subgraph({var_node}.union(max_ancr_nodes).union(max_desc_nodes))
        graph_to_plot = nx.DiGraph()

        print(' '.join(['Graph of block containing {} with <={} generations of ancestors and <={} generations of decendants:'
                       .format(var, max_ancs_gens, max_desc_gens), str(subgraph)]))

        # Loop over all nodes in subgraph (chosen variable, it's ancestors and decendants) and make nodes and edges in pyvis subgraph
        mapping = {}
        for node in subgraph.nodes():
            if node in self._condenced_model_digraph:
                node_label = '\n'.join(self._node_varlist_mapping[node])\
                    if len(self._node_varlist_mapping[node]) < 10 else '***\nHUGE BLOCK\n***'
                node_title = '<br>'.join(self._node_varlist_mapping[node])
                if node == var_node:
                    node_size = 200
                    node_color = '#c4351c'
                if node in max_ancr_nodes:
                    node_size = 100
                    node_color = '#00824d'
                if node in max_desc_nodes:
                    node_size = 100
                    node_color = '#006cb6'
            else:
                node_label = None
                node_title = None
                node_size = 100
                node_color = '#6f9090'

            graph_to_plot.add_node(node, label=node_label, title=node_title, shape='circle', size=node_size, color=node_color)

            if isinstance(node, int):
                mapping[node] =  ':\n'.join([' '.join(['Block', str(len(self._blocks)-node)]),
                                            '\n'.join(self._node_varlist_mapping[node]) if len(self._node_varlist_mapping[node]) < 5 else '...'])
            else:
                mapping[node] = node

        if graph_to_plot.number_of_nodes() > max_nodes:
            print('Graph is too big to plot')
            return

        graph_to_plot.add_edges_from(subgraph.edges())

        plt.figure(figsize=figsize)
        colors = [node[1]['color'] for node in graph_to_plot.nodes(data=True)]
        layout = nx.shell_layout(graph_to_plot)
        nx.draw(graph_to_plot,
                with_labels=True,
                labels=mapping,
                pos=layout,
                node_size=5000,
                node_color=colors,
                font_size=9,
                font_color='white',
                arrowsize=25,
                edge_color='#274247',
                width=2.0
                )
        plt.plot()


    # Returns the node in  which var is endogenous
    def _find_var_node(self, var):
        if any([var in self._condenced_model_digraph.nodes[x]['members'] for x in self._condenced_model_digraph.nodes()]):
            var_node = [var in self._condenced_model_digraph.nodes[x]['members'] for x in self._condenced_model_digraph.nodes()].index(True)
        elif var in self._augmented_condenced_model_digraph.nodes(): 
            var_node = var
        else:
            raise NameError('Variable is not in model')
        return var_node


    def trace_to_exog_vars(self, block: str):
        """
        Prints all exogenous variables that are ancestors to block
        """

        if self._some_error:
            return

        print('\n'.join([' '.join(x) for x in list(self._chunks(self._trace_to_exog_vars(block), 25))]))


    ## Finds all exogenous variables that are ancestors to block
    def _trace_to_exog_vars(self, block):
        var_node = len(self._blocks)-block
        ancs_nodes = nx.ancestors(self._augmented_condenced_model_digraph, var_node)

        ancs_exog_vars = tuple()
        for node in ancs_nodes:
            if len(nx.ancestors(self._augmented_condenced_model_digraph, node)) == 0:
                ancs_exog_vars += self._node_varlist_mapping.get(node)

        return ancs_exog_vars


    def trace_to_exog_vals(self, block: int, period_index: int):
        """
        Traces block back to exogenous values
        """
        try:
            output_array = self.last_solution.to_numpy(dtype=np.float64, copy=True)
            var_col_index = {var: i for i, var in enumerate(self.last_solution.columns.str.lower().to_list())}

            get_var_info = self.gen_get_var_info(var_col_index)

            ancs_exog_vars = self._trace_to_exog_vars(block)
            if ancs_exog_vars:
                _, ancs_exog_lags, ancs_exog_cols, = get_var_info((self._var_mapping.get(x) for x in ancs_exog_vars))
                ancs_exog_vals = self._get_vals(output_array, ancs_exog_cols, ancs_exog_lags, period_index, False)
                print('\nBlock {} traces back to the following exogenous variable values in {}:'.format(block, self.last_solution.index[period_index]))
                print(*['='.join([x, str(y)]) for x, y in zip(ancs_exog_vars, ancs_exog_vals)], sep='\n')

        except AttributeError:
            print('No solution exists')


    def show_block_vals(self, i: int, period_index: int):
        """
        TBA
        """
        try:
            output_array = self.last_solution.to_numpy(dtype=np.float64, copy=True)
            var_col_index = {var: i for i, var in enumerate(self.last_solution.columns.str.lower().to_list())}

            get_var_info = self.gen_get_var_info(var_col_index)

            block = self._blocks.get(i)

            _, block_endo_lags, block_endo_cols = get_var_info(block[0])
            block_endo_vals = self._get_vals(output_array, block_endo_cols, block_endo_lags, period_index, False)
            print('\nBlock {} has endogenous variables in {} that evaluate to:'.format(i, self._last_solution.index[period_index]))
            print(*['='.join([x, str(y)]) for x, y in zip([self._var_mapping.get(x) for x in block[0]], block_endo_vals)], sep='\n')

            _, block_pred_lags, block_pred_cols = get_var_info(block[1])
            block_pred_vals = self._get_vals(output_array, block_pred_cols, block_pred_lags, period_index, False)
            print('\nBlock {} has predetermined variables in {} that evaluate to:'.format(i, self.last_solution.index[period_index]))
            print(*['='.join([x, str(y)]) for x, y in zip([self._var_mapping.get(x) for x in block[1]], block_pred_vals)], sep='\n')

        except AttributeError:
            print('No solution exists')


    # Function that returns function that returns names, columns and lags for variables
    def gen_get_var_info(self, var_col_index):
        def get_var_info(vars):
            if not vars:
                return (tuple([]), np.array([], dtype=int), np.array([], dtype=int))
            # Stole zip-solution from: https://stackoverflow.com/questions/21444338/transpose-nested-list-in-python
            names, lags = tuple(map(list, zip(*[self._lag_mapping.get(x) for x in vars])))
            cols = tuple(var_col_index.get(x) for x in names)
            if any(x is None for x in cols):
                missing = [x for x, y in zip(names, cols) if y is None]
                raise KeyError(f'{",".join(missing)} is not in DataFrame')
            return (names, np.array(lags, dtype=int), np.array(cols, dtype=int))
        return get_var_info


    # Stole solution from https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    @staticmethod
    def _chunks(xs, n):
        n = max(1, n)
        return (xs[i:i+n] for i in range(0, len(xs), n))