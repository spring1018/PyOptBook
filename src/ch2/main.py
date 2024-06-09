import pandas as pd
import pulp

gain_df = pd.DataFrame(
    {
        "p": ["p1", "p2", "p3", "p4"],
        "gain": [3, 4, 4, 5],
    }
)

require_df = pd.DataFrame(
    {
        "p": ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3", "p3", "p4", "p4", "p4"],
        "m": ["m1", "m2", "m3", "m1", "m2", "m3", "m1", "m2", "m3", "m1", "m2", "m3"],
        "require": [2, 0, 1, 3, 2, 0, 0, 2, 2, 2, 2, 2],
    }
)

stock_df = pd.DataFrame(
    {
        "m": ["m1", "m2", "m3"],
        "stock": [35, 22, 27],
    }
)


# problem を拡張しやすい設計にすることで、テストしやすくなる
def get_base_problem(
    gain_df,
    require_df,
    stock_df,
    cat="Continuous",
) -> tuple[pulp.LpProblem, dict, list]:
    P = gain_df["p"].tolist()
    M = stock_df["m"].tolist()

    stock = {row.m: row.stock for row in stock_df.itertuples()}
    gain = {row.p: row.gain for row in gain_df.itertuples()}
    require = {(row.p, row.m): row.require for row in require_df.itertuples()}

    problem = pulp.LpProblem("maximize_gain", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", P, cat=cat)

    for p in P:
        problem += x[p] >= 0

    for m in M:
        problem += pulp.lpSum(require[p, m] * x[p] for p in P) <= stock[m]

    problem += pulp.lpSum(gain[p] * x[p] for p in P)

    return problem, x, P


def solve_problem(problem, x, P):
    status = problem.solve()
    print("Status", pulp.LpStatus[status])

    for p in P:
        print(f"{p}: {x[p].value()}")

    print(f"Total gain: {problem.objective.value()}")


if __name__ == "__main__":
    problem, x, P = get_base_problem(
        gain_df,
        require_df,
        stock_df,
        # cat="Integer",
    )
    # problem += x["p2"] == 0
    # problem += x["p3"] == 0
    solve_problem(problem, x, P)
