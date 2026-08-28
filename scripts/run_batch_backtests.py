import time
from src.backtesting.batch_backtesting import run_batch_backtests
from src.reporting.latex import generate_backtest_latex


def main():
    t0 = time.perf_counter()
    from_date = "1995-01-01"
    to_date = "2026-08-01"
    args_list = [
        ["-s", "^GSPC,^GSPC", "-fd", from_date, "-td", to_date, "-t", "log_diff", "-l", "0"],
        ["-s", "^GDAXI,^GDAXI", "-fd", from_date, "-td", to_date, "-t", "log_diff", "-l", "0"],
        ["-s", "BMW.DE,BMW.DE", "-fd", from_date, "-td", to_date, "-t", "log_diff", "-l", "0"],
        ["-s", "GBP=X,GBP=X", "-fd", from_date, "-td", to_date, "-t", "log_diff", "-l", "0"],
        ["-s", "GC=F,GC=F", "-fd", from_date, "-td", to_date, "-t", "log_diff", "-l", "0"],
    ]

    q_s = [0.95, 0.99, 0.995]
    out = run_batch_backtests(args_list, q_s, fitting_window=1000)

    generate_backtest_latex(
        out,
        q_order=q_s,
        filename="backtest_table.tex",
        caption=(
            "Backtest of exceedances from "
            f"{from_date} to {to_date} (counts and p-values)"
        ),
        label="tab:backtest",
        decimals_p=2,
    )

    t1 = time.perf_counter()
    print(f"Estimation took {t1 - t0:.3f} seconds")
    print(out)
    total_tests = len(out)
    conditional_higher_p = (out["p_Conditional"] > out["p_Unconditional"]).sum()

    print(
        "Conditional model has a higher calibration p-value: "
        f"{conditional_higher_p} out of {total_tests}"
    )

    conditional_not_rejected = (out["p_Conditional"] >= 0.05).sum()
    print(f"Conditional not rejected: {conditional_not_rejected} out of {total_tests}")
    unconditional_not_rejected = (out["p_Unconditional"] >= 0.05).sum()
    print(f"Unconditional not rejected: {unconditional_not_rejected} out of {total_tests}")


if __name__ == "__main__":
    main()
