def run_batch_backtests(args_list, q_s, fitting_window):
    import pandas as pd
    from parser import parse_command_line_arguments
    from quantile_estimator import quantileSeries
    from backtesting import back_test

    results = []
    for args in args_list:
        user_input = parse_command_line_arguments(args, split=False)
        log_returns = user_input.time_series
        q_series = quantileSeries(q=q_s, time_series=log_returns, fitting_window=fitting_window)
        fit = q_series.estimate()
        test = back_test(fit, q_s, log_returns.rv_name)
        results.append(test)
    out = pd.concat(results, ignore_index=True)
    return out



def _format_p(p, decimals=2):
    """Format p-value nicely for table; show '<0.001' when appropriate."""
    if p is None:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.{decimals}f}"

def _latex_escape(text: str) -> str:
    """Escape a few LaTeX-special characters we might see in asset names."""
    if text is None:
        return ""
    # escape underscore and percent and ampersand minimally
    return text.replace('\\', r'\textbackslash{}') \
               .replace('_', r'\_') \
               .replace('%', r'\%') \
               .replace('&', r'\&')

def generate_backtest_latex(results_df,
                            q_order=None,
                            assets_order=None,
                            filename="backtest_table.tex",
                            caption="Backtest of exceedances",
                            label="tab:backtest",
                            decimals_p=2,
                            include_table_env=True):
    """
    Build a LaTeX table file from the results DataFrame produced by run_batch_backtests.
    results_df must have columns: name, len, q, Expected, Conditional, p_Conditional,
    Unconditional, p_Unconditional.
    """
    required = {"name", "len", "q", "Expected", "Conditional", "p_Conditional", "Unconditional", "p_Unconditional"}
    if not required.issubset(set(results_df.columns)):
        raise ValueError(f"results_df missing columns (need at least): {required}")

    # asset order
    if assets_order is None:
        assets_order = list(results_df['name'].drop_duplicates())

    # lengths (take first occurrence per asset)
    lengths = results_df.groupby('name')['len'].first().to_dict()

    if q_order is None:
        q_order = sorted(results_df['q'].unique())

    # Build header columns using concatenation (avoid complex f-string nesting)
    header_cols = []
    for asset in assets_order:
        n = lengths.get(asset, "")
        asset_esc = _latex_escape(str(asset))
        if n == "" or n is None:
            # no length available
            shortstack = "\\shortstack{" + asset_esc + "}"
        else:
            shortstack = "\\shortstack{" + asset_esc + "\\\\" + str(int(n)) + "}"
        # wrap with multicolumn
        header = "\\multicolumn{1}{c}{" + shortstack + "}"
        header_cols.append(header)

    lines = []
    if include_table_env:
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + _latex_escape(caption) + "}")
        lines.append(r"\label{" + _latex_escape(label) + "}")

    # column spec: left column + one centered per asset
    colspec = "l" + "c" * len(header_cols)
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    # header row (ends with \\)
    lines.append("Length of test & " + " & ".join(header_cols) + " \\\\")
    lines.append(r"\midrule")

    # For each quantile block
    for q in q_order:
        # quantile header row (italic) - must end with \\
        lines.append(r"\multicolumn{1}{l}{\emph{" + f"{q} Quantile" + "}} \\\\")
        # Expected row
        row_vals = []
        for asset in assets_order:
            sel = results_df[(results_df['name'] == asset) & (results_df['q'] == q)]
            if len(sel) == 0:
                row_vals.append("")
            else:
                expected = int(sel['Expected'].iat[0])
                row_vals.append(str(expected))
        lines.append("Expected & " + " & ".join(row_vals) + " \\\\")
        # Conditional row: show count (pvalue)
        row_vals = []
        for asset in assets_order:
            sel = results_df[(results_df['name'] == asset) & (results_df['q'] == q)]
            if len(sel) == 0:
                row_vals.append("")
            else:
                count = int(sel['Conditional'].iat[0])
                p = float(sel['p_Conditional'].iat[0])
                row_vals.append(f"{count} ({_format_p(p, decimals=decimals_p)})")
        lines.append("Conditional & " + " & ".join(row_vals) + " \\\\")
        # Unconditional row
        row_vals = []
        for asset in assets_order:
            sel = results_df[(results_df['name'] == asset) & (results_df['q'] == q)]
            if len(sel) == 0:
                row_vals.append("")
            else:
                count = int(sel['Unconditional'].iat[0])
                p = float(sel['p_Unconditional'].iat[0])
                row_vals.append(f"{count} ({_format_p(p, decimals=decimals_p)})")
        lines.append("Unconditional & " + " & ".join(row_vals) + " \\\\")
        # a little vertical space between quantile blocks (booktabs)
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if include_table_env:
        lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"Wrote LaTeX table to {filename}")
    return filename
