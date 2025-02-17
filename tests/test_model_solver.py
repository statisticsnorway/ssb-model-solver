import numpy as np
import pandas as pd
import pytest

import model_solver as ms

TOLERANCE = 1e-6  # Small tolerance for floating-point comparisons


# Fixtures for test data
@pytest.fixture
def equations() -> list[str]:
    return [
        "x1 = a1",
        "x2 = a2",
        "0.2*x1+0.7*x2 = 0.1*ca+0.8*cb+0.3*i1",
        "0.8*x1+0.3*x2 = 0.9*ca+0.2*cb+0.1*i2",
        "k1 = k1(-1)+i1",
        "k2 = k2(-1)+i2",
    ]


@pytest.fixture
def endogenous() -> list[str]:
    return ["x1", "x2", "ca", "cb", "k1", "k2"]


@pytest.fixture
def sensitivity_params_std() -> tuple[int, int, str]:
    return 1, 1, "std"


@pytest.fixture
def sensitivity_params_pct() -> tuple[int, int, str]:
    return 1, 1, "pct"


@pytest.fixture
def sensitivity_params_one() -> tuple[int, int, str]:
    return 1, 1, "one"


@pytest.fixture
def input_data() -> pd.DataFrame:
    return pd.DataFrame(
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


# Test for model initialization
def test_model_initialization(equations: list[str], endogenous: list[str]) -> None:
    model = ms.ModelSolver(equations, endogenous)
    assert model.eqns == tuple([eq.lower() for eq in equations])
    assert model.endo_vars == tuple([var.lower() for var in endogenous])


# Test for block identification
def test_show_blocks(equations: list[str], endogenous: list[str]) -> None:
    model = ms.ModelSolver(equations, endogenous)
    model.show_blocks()
    # Assuming the blocks are stored in an internal structure
    assert len(model._blocks) == 5  # Check that 5 blocks are identified


# Test solving the model
def test_solve_model(
    equations: list[str], endogenous: list[str], input_data: pd.DataFrame
) -> None:
    model = ms.ModelSolver(equations, endogenous)
    model.describe()
    model.show_blocks()
    solution = model.solve_model(input_data)

    # Expecting 3 periods in the solution after dropping the first period
    expected_output_shape = (3, 10)
    assert solution.shape == expected_output_shape

    # Validating expected data in the solution for the last 3 periods (2019Q2, 2020Q3, 2020Q4)
    expected_values = {
        "x1": [2.0, 4.0, 4.0],
        "x2": [2.0, 3.0, 4.0],
        "ca": [1.942857, 3.857143, 3.885714],
        "cb": [1.257143, 1.642857, 2.514286],
        "k1": [3.0, 7.0, 11.0],
        "k2": [3.0, 6.0, 10.0],
        "a1": [2.0, 4.0, 4.0],
        "a2": [2.0, 3.0, 4.0],
        "i1": [2.0, 4.0, 4.0],
        "i2": [2.0, 3.0, 4.0],
    }

    for col, values in expected_values.items():
        assert solution[col].tolist() == pytest.approx(values, rel=TOLERANCE)


# Test switching endogenous variables
def test_switch_endo_vars(equations: list[str], endogenous: list[str]) -> None:
    model = ms.ModelSolver(equations, endogenous)
    model.switch_endo_vars(["x2"], ["a2"])
    assert "x2" not in model.endo_vars
    assert "a2" in model.endo_vars
    model.show_blocks()  # Ensure the model is reconfigured after switching variables


# Test tracing to exogenous variables
def test_trace_to_exog_vars(equations: list[str], endogenous: list[str]) -> None:
    model = ms.ModelSolver(equations, endogenous)
    exog_vars = model.trace_to_exog_vars(5, noisy=False)
    expected_exog_vars = ["i1", "i2", "a1", "a2"]
    assert exog_vars is not None
    assert sorted(exog_vars) == sorted(expected_exog_vars)


# Test tracing to exogenous values
def test_trace_to_exog_vals(
    equations: list[str], endogenous: list[str], input_data: pd.DataFrame
) -> None:
    model = ms.ModelSolver(equations, endogenous)
    model.solve_model(input_data)
    exog_vals = model.trace_to_exog_vals(5, period_index=1, noisy=False)
    expected_exog_vals = pd.Series(
        {"i1": 2.0, "i2": 2.0, "a1": 2.0, "a2": 2.0}
    ).sort_index()
    assert exog_vals is not None
    exog_vals = exog_vals.sort_index()  # Sort before comparison

    pd.testing.assert_series_equal(exog_vals, expected_exog_vals, rtol=TOLERANCE)


# Test showing block values
def test_show_block_vals(
    equations: list[str], endogenous: list[str], input_data: pd.DataFrame
) -> None:
    model = ms.ModelSolver(equations, endogenous)
    model.solve_model(input_data)
    endo_vals, pred_vals = model.show_block_vals(5, 1, noisy=False)
    assert endo_vals is not None
    assert pred_vals is not None

    # Ensure that the endo_vals and pred_vals match expected results from the notebook
    expected_endo_vals = pd.Series({"ca": 1.942857, "cb": 1.257143})
    expected_pred_vals = pd.Series({"i1": 2.0, "i2": 2.0, "x1": 2.0, "x2": 2.0})

    endo_vals = endo_vals.sort_index()  # Sort before comparison
    pred_vals = pred_vals.sort_index()  # Sort before comparison

    pd.testing.assert_series_equal(endo_vals, expected_endo_vals, rtol=TOLERANCE)
    pd.testing.assert_series_equal(pred_vals, expected_pred_vals, rtol=TOLERANCE)


def test_sensitivity(
    equations: list[str],
    endogenous: list[str],
    input_data: pd.DataFrame,
    sensitivity_params_std: tuple[int, int, str],
    sensitivity_params_pct: tuple[int, int, str],
    sensitivity_params_one: tuple[int, int, str],
) -> None:
    model = ms.ModelSolver(equations, endogenous)
    model.solve_model(input_data)

    # Unpacks fixtures for std, pct and ones, then does analysis
    sens_std = model.sensitivity(*sensitivity_params_std)
    sens_pct = model.sensitivity(*sensitivity_params_pct)
    sens_one = model.sensitivity(*sensitivity_params_one)

    # Checks for results
    assert sens_std.empty is False
    assert sens_pct.empty is False
    assert sens_one.empty is False

    # k2 will be in the block analysed with current equation set
    assert "k2" in sens_std.columns
    assert np.allclose(4.732277, sens_std["k2"].sum(), atol=0.00001)


def test_validate_unique_column_names_passes() -> None:
    """Test that validation passes with unique column names."""
    df = pd.DataFrame(columns=["A", "B", "C"])
    ms.ModelSolver._validate_unique_column_names(df)  # Should pass without raising


def test_validate_unique_column_names_fails_single_duplicate() -> None:
    """Test that validation fails with a single duplicated column."""
    df = pd.DataFrame(columns=["A", "B", "A"])

    with pytest.raises(ValueError) as exc_info:
        ms.ModelSolver._validate_unique_column_names(df)

    assert "Found duplicate column names in DataFrame" in str(exc_info.value)
    assert "{'A': 2}" in str(exc_info.value)


def test_validate_unique_column_names_fails_multiple_duplicates() -> None:
    """Test that validation fails with multiple duplicated columns."""
    df = pd.DataFrame(columns=["A", "B", "A", "B", "C"])

    with pytest.raises(ValueError) as exc_info:
        ms.ModelSolver._validate_unique_column_names(df)

    assert "Found duplicate column names in DataFrame" in str(exc_info.value)
    # Check both duplicates are reported
    assert "'A': 2" in str(exc_info.value)
    assert "'B': 2" in str(exc_info.value)


def test_validate_unique_column_names_empty_df() -> None:
    """Test that validation passes with an empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError) as exc_info:
        ms.ModelSolver._validate_unique_column_names(df)
    assert "DataFrame has no columns" in str(exc_info)
