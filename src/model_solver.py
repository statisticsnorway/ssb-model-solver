##################################
# Author: Magnus Kvåle Helliesen #
##################################

import os
import numpy as np
import networkx as nx
from pyvis.network import Network
import pandas as pd
from symengine import var, Matrix, Lambdify
import matplotlib.pyplot as plt
from collections import Counter
from functools import cache
from numba import njit


class ModelSolver:
    """
    EXAMPLE OF USE USE:

    Let "equations" and "endogenous" be lists containing equations and endogenous variables, respectively, stored as strings. E.g.

        equations = [
            'x+y=A',
            'x/y=B'
            ]

        endogenous = [
            'x',
            'y'
            ]
    
    where 'A' and 'B' are exogenous variables
    Note that the solver does not support mathematical functions, only addition, subtraction, multiplication and division. Support for this may be added
    
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
    def __init__(self, eqns: list, endo_vars: list):
        self._some_error = False
        self._lag_notation = '___LAG'
        self._max_lag = 0
        self._root_tolerance = 1e-7
        self._max_iter = 10

        print('Initializing model...')

        # Model equations and endogenous variables are checked and stored as immutable tuples (as opposed to mutable lists)
        self._eqns, self._endo_vars = self._init_model(eqns, endo_vars)

        print('* Analyzing model...')

        # Analyzing equation strings to determine variables, lags and coefficients
        self._eqns_analyzed, self._var_mapping, self._lag_mapping = self._analyze_eqns()

        # Using graph theory to analyze equations using existing algorithms to establish minimum simultaneous blocks
        self._eqns_endo_vars_bigraph = self._gen_eqns_endo_vars_bigraph()
        self._eqns_endo_vars_match = self._find_max_bipartite_match()
        self._model_digraph = self._gen_model_digraph()
        self._condenced_model_digraph, self.condenced_model_node_varlist_mapping = self._gen_condenced_model_digraph()
        self._augmented_condenced_model_digraph = self._gen_augmented_condenced_model_digraph()

        # Generating everything needed to simulate model
        self._sim_code, self._blocks = self._gen_sim_code_and_blocks()

        print('Finished')


    @property
    def eqns(self):
        return self._eqns

    @property
    def endo_vars(self):
        return self._endo_vars

    @property
    def max_lag(self):
        return self._max_lag

    @property
    def blocks(self):
        return tuple(tuple([endo_vars, tuple(self._var_mapping.get(x)[2] for x in exog_vars), eqns]) for endo_vars, exog_vars, eqns in self._blocks)

    @property
    def root_tolerance(self):
        return self._root_tolerance

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def last_solution(self):
        try:
            return self._last_solution
        except AttributeError:
            print('No solution exists')


    @root_tolerance.setter
    def root_tolerance(self, val):
        if isinstance(val, float) is False:
            raise ValueError('Tolerance for termination must be of type float')
        if val <= 0:
            raise ValueError('Tolerance for termination must be positive')
        self._root_tolerance = val

    @max_iter.setter
    def max_iter(self, val):
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


    # Analyzes equations of the model
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
        num, var, lag = '', '', ''
        is_num, is_var, is_lag, is_sci = False, False, False, False

        for chr in ''.join([eqn, ' ']):
            is_num = (chr.isnumeric() and not is_var) or is_num
            is_var = (chr.isalpha()  and not is_num) or is_var
            is_lag = (is_var and chr == '(') or is_lag
            is_sci = (is_num and chr == 'e') or is_sci

            # Check if character is something other than a numeric, variable or lag and write numeric or variable to parsed equation
            if chr in ['=','+','-','*','/','(',')',' '] and not (is_lag or is_sci):
                if is_num:
                    parsed_eqn_with_lag_notation += str(num),
                if is_var:
                    # Replace (-)-notation by LAG_NOTATION for lags and appends _ to the end to mark the end
                    pfx = '' if lag == '' else ''.join([self._lag_notation, str(-int(lag[1:-1])), '_'])
                    parsed_eqn_with_lag_notation += ''.join([var, pfx]),
                    var_mapping[''.join([var, lag])] = ''.join([var, pfx])
                    var_mapping[''.join([var, pfx])] = ''.join([var, lag])
                    lag_mapping[''.join([var, pfx])] = (var, 0 if lag == '' else -int(lag[1:-1]))
                    if lag != '':
                        self._max_lag = max(self._max_lag, -int(lag.replace('(', '').replace(')', '')))
                if chr != ' ':
                    parsed_eqn_with_lag_notation += chr,
                num, var, lag = '', '', ''
                is_num, is_var, is_lag = False, False, False
                continue

            if is_sci and chr.isnumeric():
                is_sci = False

            if is_num:
                num = ''.join([num, chr])
                continue

            if is_var and not is_lag:
                var = ''.join([var, chr])
                continue

            if is_var and is_lag:
                lag = ''.join([lag, chr])
                if chr == ')':
                    is_lag = False

        eqn_with_lag_notation=''.join(parsed_eqn_with_lag_notation)

        return eqn_with_lag_notation, var_mapping, lag_mapping


    # Generates bipartite graph (bigraph) connetcting equations (nodes in U) with endogenous variables (nodes in V)
    # See https://en.wikipedia.org/wiki/Bipartite_graph for a discussion of bigraphs
    def _gen_eqns_endo_vars_bigraph(self):
        if self._some_error:
            return

        print('\t* Generating bipartite graph connecting equations and endogenous variables')

        # Make nodes in bipartite graph with equations U (0) and endogenous variables in V (1)
        eqns_endo_vars_bigraph = nx.Graph()
        eqns_endo_vars_bigraph.add_nodes_from([i for i, _ in enumerate(self._eqns)], bipartite=0)
        eqns_endo_vars_bigraph.add_nodes_from(self._endo_vars, bipartite=1)

        # Make edges between equations and endogenous variables
        for i, eqns in enumerate(self._eqns_analyzed):
            for endo_var in [x for x in eqns[2].keys() if x in self._endo_vars]:
                eqns_endo_vars_bigraph.add_edge(i, endo_var)

        return eqns_endo_vars_bigraph


    # Finds a maximum bipartite match (MBM) of bigraph connetcting equations (nodes in U) with endogenous variables (nodes in V)
    # See https://www.geeksforgeeks.org/maximum-bipartite-matching/ for more on MBM
    # Returns dict with matches (maps both ways, i.e. U-->V and U-->U)
    def _find_max_bipartite_match(self):
        if self._some_error:
            return

        print('\t* Finding maximum bipartite match (MBM) (i.e. associating every equation with exactly one endogenus variable)')

        # Use maximum bipartite matching to make a one to one mapping between equations and endogenous variables
        try:
            maximum_bipartite_match = nx.bipartite.maximum_matching(self._eqns_endo_vars_bigraph, [i for i, _ in enumerate(self._eqns)])
            if len(maximum_bipartite_match)/2 < len(self._eqns):
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
    def _gen_model_digraph(self):
        if self._some_error:
            return

        print('\t* Generating directed graph (DiGraph) connecting endogenous variables using bipartite graph and MBM')

        # Make nodes in directed graph of endogenous variables
        model_digraph = nx.DiGraph()
        model_digraph.add_nodes_from(self._endo_vars)

        # Make directed edges showing how endogenous variables affect every other endogenous variables using bipartite graph and MBM
        for edge in self._eqns_endo_vars_bigraph.edges():
            if edge[0] != self._eqns_endo_vars_match[edge[1]]:
                model_digraph.add_edge(edge[1], self._eqns_endo_vars_match[edge[0]])

        return model_digraph


    # Makes a condencation of digraph of endogenous variables
    # Each node of condencation contains strongly connected components; this corresponds to the simulataneous model blocks
    # See https://en.wikipedia.org/wiki/Strongly_connected_component for more about strongly connected components
    def _gen_condenced_model_digraph(self):
        if self._some_error:
            return

        print('\t* Finding condensation of DiGraph (i.e. finding minimum simulataneous equation blocks)')

        # Generate condensation graph of equation graph such that every node is a strong component of the equation graph
        condenced_model_digraph = nx.condensation(self._model_digraph)

        # Make a dictionary that associate every node of condensation with a list of variables
        node_vars_mapping = {}
        for node in tuple(condenced_model_digraph.nodes()):
            node_vars_mapping[node] = tuple(condenced_model_digraph.nodes[node]['members'])

        return condenced_model_digraph, node_vars_mapping


    # Augments condenced digraph with nodes and edges for exogenous variables in order to show what exogenous variables affect what strong components
    def _gen_augmented_condenced_model_digraph(self):
        if self._some_error:
            return

        augmented_condenced_model_digraph = self._condenced_model_digraph.copy()

        # Make edges between exogenous variables and strong components it is a part of
        for node in self._condenced_model_digraph.nodes():
            for member in self._condenced_model_digraph.nodes[node]['members']:
                for exog_var_adjacent_to_node in [val for key, val in self._eqns_analyzed[self._eqns_endo_vars_match[member]][2].items()
                                                  if self._lag_notation not in val and key not in self._endo_vars]:
                    augmented_condenced_model_digraph.add_edge(exog_var_adjacent_to_node, node)

        return augmented_condenced_model_digraph


    # Generates simulation code and blocks
    # Simulation code contains a tuple of tuples for each strong component
    # The tuple for each strong component contains objective function, and Jacobian matrix, and lists of the variables in the strong component
    def _gen_sim_code_and_blocks(self):
        if self._some_error:
            return

        print('\t* Generating simulation code (i.e. block-wise symbolic objective function, symbolic Jacobian matrix and lists of endogenous and exogenous variables)')

        sim_code, blocks = {}, {}
        for i, node in enumerate(reversed(tuple(self._condenced_model_digraph.nodes()))):
            block_endo_vars, block_eqns_orig, block_eqns_lags, block_exog_vars = [], [], [], set()
            for member in self._condenced_model_digraph.nodes[node]['members']:
                j = self._eqns_endo_vars_match[member]
                eqns_analyzed = self._eqns_analyzed[j]
                block_endo_vars += member,
                block_eqns_orig += eqns_analyzed[0],
                block_eqns_lags += eqns_analyzed[1],
                block_exog_vars.update([val for key, val in eqns_analyzed[2].items() if self._lag_notation not in key])

            block_exog_vars.difference_update(set(block_endo_vars))
            sim_code[i+1] = (*self._gen_obj_fun_and_jac(tuple(block_eqns_lags), tuple(block_endo_vars), tuple(block_exog_vars)),
                           tuple(block_endo_vars), tuple(block_exog_vars), tuple(block_eqns_lags))
            blocks[i+1] = (tuple(block_endo_vars), tuple(block_exog_vars), tuple(block_eqns_orig))

        return sim_code, blocks


    # Generates symbolic objective functon and Jacobian matrix for a given strong component
    @staticmethod
    def _gen_obj_fun_and_jac(eqns: tuple, endo_vars: tuple, exog_vars: tuple):
        endo_sym, exog_sym, obj_fun = [], [], []
        for endo_var in endo_vars:
            var(endo_var)
            endo_sym += eval(endo_var),
        for exog_var in exog_vars:
            var(exog_var)
            exog_sym += eval(exog_var),
        for eqn in eqns:
            lhs, rhs = eqn.split('=')
            obj_fun_row = eval('-'.join([''.join(['(', lhs.strip().strip('+'), ')']), ''.join(['(', rhs.strip().strip('+'), ')'])]))
            obj_fun += obj_fun_row,

        jac = Matrix(obj_fun).jacobian(Matrix(endo_sym)).tolist()

        obj_fun_lambdify = Lambdify([*endo_sym, *exog_sym], obj_fun, cse=True)
        jac_lambdify = Lambdify([*endo_sym, *exog_sym], jac, cse=True)

        output_obj_fun = lambda val_list, *args: obj_fun_lambdify(*val_list, *args)
        output_jac = lambda val_list, *args: jac_lambdify(*val_list, *args)

        return output_obj_fun, output_jac


    def switch_endo_var(self, old_endo, new_endo):
        """
        This method will allow the user to switch endogenos and exogenous variables
        """
        pass


    def find_endo_var(self, endo_var):
        """
        Finds what block solves the given engoenous variable
        """

        try:
            return [endo_var in x[0] for x in self._blocks].index(True)
        except ValueError:
            return


    def show_model_info(self):
        """
        Shows model info, that is number of equations, number of simultaneous blocks and how many equations are in each block
        """

        print('*'*100)
        print('Model consists of {} equations in {} blocks\n'.format(len(self._eqns), len(self._blocks)))
        for key, val in self._blocks.items():
            print('{} blocks have {} equations'.format(val, key))
        print('*'*100)


    def show_blocks(self):
        """
        Prints endogenous and exogenous variables and equations for every block in the model
        """

        for key, _ in self._blocks.items():
            print(' '.join(['*'*50, 'Block', str(key), '*'*50, '\n']))
            self.show_block(key)


    def show_block(self, i):
        """
        Prints endogenous and exogenous variables and equations for a given block
        """

        block = self._blocks.get(i)
        print('Endogenous ({} variables):'.format(len(block[0])))
        print('\n'.join([' '.join(x) for x in list(self._chunks(block[0], 25))]))
        print('\nExogenous ({} variables):'.format(len(block[1])))
        print('\n'.join([' '.join(x) for x in list(self._chunks([self._var_mapping.get(x) for x in block[1]], 25))]))
        print('\nEquations ({} equations):'.format(len(block[2])))
        print('\n'.join(block[2]))


    def solve_model(self, input_data: pd.DataFrame):
        """
        Solves the model for a given DataFrame
        """

        if self._some_error:
            return

        print('Solving model...')

        output_array = input_data.to_numpy(dtype=np.float64, copy=True)
        var_col_index = {var: i for i, var in enumerate(input_data.columns.str.lower().to_list())}

        # Function that gets name, column index and lag for variables
        # Function uses cache since it's calle repeatedly
        @cache
        def get_var_info(vars):
            if not vars:
                return (tuple([]), np.array([], dtype=int), np.array([], dtype=int))
            # Stole zip-solution from: https://stackoverflow.com/questions/21444338/transpose-nested-list-in-python
            names, lags = tuple(map(list, zip(*[self._lag_mapping.get(x) for x in vars])))
            cols = tuple(var_col_index.get(x) for x in names)
            return (names, np.array(lags, dtype=int), np.array(cols, dtype=int))

        first_period, last_period = self._max_lag, output_array.shape[0]-1
        periods = range(first_period, last_period+1)
        print('\tFirst period: {}, last period: {}'.format(input_data.index[first_period], input_data.index[last_period]))
        print('\tSolving')
        print(''.join(['\t|', ' '*(last_period-first_period+1), '|']))
        print('\t ', end='')

        for period in periods:
            print('.', end='')
            for key, val in self._sim_code.items():
                (obj_fun, jac, endo_vars, exog_vars, _) = val
                solution = self._solve_block(
                    obj_fun,
                    jac,
                    get_var_info(endo_vars),
                    get_var_info(exog_vars),
                    output_array,
                    period
                    )

                if solution.get('status') == 2:
                    raise ValueError('Block {} did not converge'.format(key))
                if solution.get('status') == 1:
                    print('Maximum number of iterations reached for block {} in {}'.format(key, input_data.index[period]))

                output_array[period, [var_col_index.get(x) for x in endo_vars]] = solution['x']

        print('\nFinished')

        self._last_solution = pd.DataFrame(output_array, columns=input_data.columns, index=input_data.index)

        return self._last_solution


    # Solves one block of the model for a given time period
    def _solve_block(self, obj_fun, jac, endo_vars_info: tuple, exog_vars_info: tuple, output_array: np.array, period: int):
        endo_vars_names, endo_vars_lags, endo_vars_cols, = endo_vars_info
        exog_vars_names, exog_vars_lags, exog_vars_cols, = exog_vars_info

        solution = self._newton_raphson(
            obj_fun,
            self._get_vals(output_array, endo_vars_cols, endo_vars_lags, period),
            args = tuple(self._get_vals(output_array, exog_vars_cols, exog_vars_lags, period)),
            jac = jac,
            tol = self._root_tolerance,
            maxiter=self._max_iter
            )
        
        if solution.get('status') == 2:
            print(*endo_vars_names, sep=' ')
            print(*self._get_vals(output_array, endo_vars_cols, endo_vars_lags, period), sep=' ')
            print(*exog_vars_names, sep=' ')
            print(*self._get_vals(output_array, exog_vars_cols, exog_vars_lags, period), sep=' ')

        return solution


    # Gets values from DataFrame via array view for speed
    # If shape of request > 0 then the request is sent to njit'ed method for speed
    def _get_vals(self, array: np.array, cols: np.array, lags: np.array, period: int):
        if cols.shape[0] == 0:
            return np.array([], np.float64)
        else:
            return self._get_vals_njit(array, cols, lags, period) 


    # Gets values from DataFrame via array view
    # Some weird stuff had to be implemented for njit to stop complaining
    @staticmethod
    @njit
    def _get_vals_njit(array: np.array, cols: np.array, lags: np.array, period: int):
        vals = np.array([0.0], dtype=np.float64)
        for col, lag in zip(cols, lags):
            vals = np.append(vals, array[period-lag, col])
        return vals[1:]


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


    def draw_blockwise_graph(self, variable: str, max_ancestor_generations: int, max_descentant_generations: int, max_nodes= int, in_notebook=False, html=False):
        """
        Draws a directed graph of block in which variable is along with max number of ancestors and descendants.
        """

        if self._some_error:
            return

        if any([variable in self._condenced_model_digraph.nodes[x]['members'] for x in self._condenced_model_digraph.nodes()]):
            variable_node = [variable in self._condenced_model_digraph.nodes[x]['members'] for x in self._condenced_model_digraph.nodes()].index(True)
        elif variable in self._augmented_condenced_model_digraph.nodes(): 
            variable_node = variable
        else:
            raise NameError('Variable is not in model')

        ancr_nodes = nx.ancestors(self._augmented_condenced_model_digraph, variable_node)
        desc_nodes = nx.descendants(self._augmented_condenced_model_digraph, variable_node)

        max_ancr_nodes = {x for x in ancr_nodes if\
            nx.shortest_path_length(self._augmented_condenced_model_digraph, x, variable_node) <= max_ancestor_generations}
        max_desc_nodes = {x for x in desc_nodes if\
            nx.shortest_path_length(self._augmented_condenced_model_digraph, variable_node, x) <= max_descentant_generations}

        subgraph = self._augmented_condenced_model_digraph.subgraph({variable_node}.union(max_ancr_nodes).union(max_desc_nodes))
        graph_to_plot = nx.DiGraph()

        print(' '.join(['Graph of block containing {} with <={} generations of ancestors and <={} generations of decendants:'
                       .format(variable, max_ancestor_generations, max_descentant_generations), str(subgraph)]))

        # Loop over all nodes in subgraph (chosen variable, it's ancestors and decendants) and make nodes and edges in pyvis subgraph
        mapping = {}
        for node in subgraph.nodes():
            if node in self._condenced_model_digraph:
                node_label = '\n'.join(self.condenced_model_node_varlist_mapping[node])\
                    if len(self.condenced_model_node_varlist_mapping[node]) < 10 else '***\nHUGE BLOCK\n***'
                node_title = '<br>'.join(self.condenced_model_node_varlist_mapping[node])
                if node == variable_node:
                    node_size = 200
                    node_color = 'red'
                if node in max_ancr_nodes:
                    node_size = 100
                    node_color = 'green'
                if node in max_desc_nodes:
                    node_size = 100
                    node_color = 'blue'
            else:
                node_label = None
                node_title = None
                node_size = 100
                node_color = 'silver'
            graph_to_plot.add_node(node, label=node_label, title=node_title, shape='circle', size=node_size, color=node_color)
            if node in self.condenced_model_node_varlist_mapping:
                mapping[node] =  ':\n'.join([' '.join(['Block', str(len(self._blocks)-node)]),
                                             '\n'.join(self.condenced_model_node_varlist_mapping[node]) if len(self.condenced_model_node_varlist_mapping[node]) < 5 else '...'])
            else:
                mapping[node] = str(node)

        if graph_to_plot.number_of_nodes() > max_nodes:
            print('Graph is too big to plot')
            return

        graph_to_plot.add_edges_from(subgraph.edges())
        if html:
            net = Network('2000px', '2000px', directed=True, notebook=in_notebook)
            net.from_nx(graph_to_plot)
            net.repulsion(node_distance=50, central_gravity=0.01, spring_length=100, spring_strength=0.02, damping=0.5)
            net.show('graph.html')
        else:
            plt.figure(figsize=(5, 5))
            colors = [node[1]['color'] for node in graph_to_plot.nodes(data=True)]
            layout = nx.shell_layout(graph_to_plot)
            nx.draw(graph_to_plot, with_labels=True, labels=mapping, pos=layout, node_size=3000, node_color=colors, font_size=7, font_color='white')
            plt.plot()


    # Stole solution from https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    @staticmethod
    def _chunks(xs, n):
        n = max(1, n)
        return (xs[i:i+n] for i in range(0, len(xs), n))
