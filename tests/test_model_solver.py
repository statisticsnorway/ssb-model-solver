import sys
import os
import pytest
import pandas as pd
import numpy as np

# Insert the path to your project root directly in the test file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the model_solver module correctly
import model_solver as ms

# Fixtures for test data
@pytest.fixture
def equations():
    return [
        'x1 = a1',
        'x2 = a2',
        '0.2*x1+0.7*x2 = 0.1*ca+0.8*cb+0.3*i1',
        '0.8*x1+0.3*x2 = 0.9*ca+0.2*cb+0.1*i2',
        'k1 = k1(-1)+i1',
        'k2 = k2(-1)+i2'
    ]

@pytest.fixture
def endogenous():
    return ['x1', 'x2', 'ca', 'cb', 'k1', 'k2']

@pytest.fixture
def input_data():
    return pd.DataFrame({
        'x1': [2, 4, 1, 2],
        'x2': [2, 1, 2, 3],
        'ca': [1, 3, 4, 1],
        'cb': [1, 2, 1, 4],
        'k1': [1, 3, 4, 1],
        'k2': [1, 2, 1, 4],
        'a1': [1, 2, 4, 4],
        'a2': [3, 2, 3, 4],
        'i1': [1, 2, 4, 4],
        'i2': [3, 2, 3, 4]
    }, index=['2019Q1', '2019Q2', '2020Q3', '2020Q4'])

# Test for model initialization
def test_model_initialization(equations, endogenous):
    model = ms.ModelSolver(equations, endogenous)
    assert model.eqns == tuple([eq.lower() for eq in equations])
    assert model.endo_vars == tuple([var.lower() for var in endogenous])

# Test for block identification
def test_show_blocks(equations, endogenous):
    model = ms.ModelSolver(equations, endogenous)
    model.show_blocks()
    # Assuming the blocks are stored in an internal structure
    assert len(model._blocks) == 5  # Check that 5 blocks are identified

# Test solving the model
def test_solve_model(equations, endogenous, input_data):
    model = ms.ModelSolver(equations, endogenous)
    model.describe()
    model.show_blocks()
    solution = model.solve_model(input_data)

    # Expecting 3 periods in the solution after dropping the first period
    expected_output_shape = (3, 10)
    assert solution.shape == expected_output_shape

    # Validating expected data in the solution for the last 3 periods (2019Q2, 2020Q3, 2020Q4)
    expected_values = {
        'x1': [2.0, 4.0, 4.0],
        'x2': [2.0, 3.0, 4.0],
        'ca': [1.942857, 3.857143, 3.885714],
        'cb': [1.257143, 1.642857, 2.514286],
        'k1': [3.0, 7.0, 11.0],
        'k2': [3.0, 6.0, 10.0],
        'a1': [2.0, 4.0, 4.0],
        'a2': [2.0, 3.0, 4.0],
        'i1': [2.0, 4.0, 4.0],
        'i2': [2.0, 3.0, 4.0],
    }

    tolerance = 1e-6  # Small tolerance for floating-point comparisons
    for col, values in expected_values.items():
        assert solution[col].tolist() == pytest.approx(values, rel=tolerance)

# Test switching endogenous variables
def test_switch_endo_vars(equations, endogenous):
    model = ms.ModelSolver(equations, endogenous)
    model.switch_endo_vars(['x2'], ['a2'])
    assert 'x2' not in model.endo_vars
    assert 'a2' in model.endo_vars
    model.show_blocks()  # Ensure the model is reconfigured after switching variables

# Test tracing to exogenous variables
def test_trace_to_exog_vars(equations, endogenous):
    model = ms.ModelSolver(equations, endogenous)
    exog_vars = model.trace_to_exog_vars(5, noisy=False)
    expected_exog_vars = ['i1', 'i2', 'a1', 'a2']
    assert sorted(exog_vars) == sorted(expected_exog_vars)

# Test tracing to exogenous values
def test_trace_to_exog_vals(equations, endogenous, input_data):
    model = ms.ModelSolver(equations, endogenous)
    solution = model.solve_model(input_data)
    exog_vals = model.trace_to_exog_vals(5, period_index=1, noisy=False)
    expected_exog_vals = pd.Series({
        'i1': 2.0,
        'i2': 2.0,
        'a1': 2.0,
        'a2': 2.0
    }).sort_index()
    exog_vals = exog_vals.sort_index()  # Sort before comparison
    
    pd.testing.assert_series_equal(exog_vals, expected_exog_vals, rtol=1e-6)

# Test showing block values
def test_show_block_vals(equations, endogenous, input_data):
    model = ms.ModelSolver(equations, endogenous)
    model.solve_model(input_data)
    endo_vals, pred_vals = model.show_block_vals(5, 1, noisy=False)
    
    # Ensure that the endo_vals and pred_vals match expected results from the notebook
    expected_endo_vals = pd.Series({'ca': 1.942857, 'cb': 1.257143})
    expected_pred_vals = pd.Series({'i1': 2.0, 'i2': 2.0, 'x1': 2.0, 'x2': 2.0})
    
    endo_vals = endo_vals.sort_index()  # Sort before comparison
    pred_vals = pred_vals.sort_index()  # Sort before comparison
    
    pd.testing.assert_series_equal(endo_vals, expected_endo_vals, rtol=1e-6)
    pd.testing.assert_series_equal(pred_vals, expected_pred_vals, rtol=1e-6)

