"""Test: forecast CLI must warn loudly when requested horizon has no tuned params."""

from __future__ import annotations

import io
from contextlib import redirect_stdout


def test_missing_horizon_raises_or_warns_loudly(tmp_path, monkeypatch):
    """If user requests --horizon 12 but only h4/h8 were tuned, user must be notified."""
    import econometrics.config.settings as settings
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    from econometrics.tuning.bvar_tuner import save_tuned_params

    save_tuned_params(
        "test_metro",
        {
            4: {"lambda": 0.3, "cross_shrink": 0.5, "ic": "aic", "exog_cols": [], "rmse": 0.02, "n_trials": 10, "h_quarters": 4},
            8: {"lambda": 0.3, "cross_shrink": 0.5, "ic": "aic", "exog_cols": [], "rmse": 0.03, "n_trials": 10, "h_quarters": 8},
        },
    )

    from econometrics.tuning.bvar_tuner import load_tuned_params_or_warn

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = load_tuned_params_or_warn("test_metro", 12)

    assert result is None
    output = buf.getvalue()
    assert "WARNING" in output.upper() or "NOT TUNED" in output.upper()
    assert "12" in output


def test_present_horizon_returns_params_without_warning(tmp_path, monkeypatch):
    """When the requested horizon IS tuned, the function must return params silently."""
    import econometrics.config.settings as settings
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    from econometrics.tuning.bvar_tuner import save_tuned_params, load_tuned_params_or_warn

    save_tuned_params(
        "test_metro",
        {
            4: {"lambda": 0.3, "cross_shrink": 0.5, "ic": "aic", "exog_cols": [], "rmse": 0.02, "n_trials": 10, "h_quarters": 4},
        },
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = load_tuned_params_or_warn("test_metro", 4)

    assert result is not None
    assert result["lambda"] == 0.3
    assert "WARNING" not in buf.getvalue().upper()
