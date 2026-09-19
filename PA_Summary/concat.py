import pandas as pd

# * Used to generate Portfolio Summary

def stats(criterion):
    df = pd.read_csv(f"{criterion}.csv")
    stats = pd.DataFrame(
        {
            "Mean": df.mean(),
            "Std": df.std(),
            "Min": df.min(),
            "Max": df.max(),
        }
    ).T
    print(stats)
    return stats


def modify_csv(criterion):
    df = stats(criterion)
    first_col = df.index
    df.insert(0, "Factors", first_col)
    # df = df.rename(columns={first_col: "Factors"})
    df.insert(0, "Criteria", criterion)

    return df


def concate():
    dt = pd.DataFrame()
    for criterion in ["CR", "AR", "SHR", "SOR", "MDD"]:
        df = modify_csv(criterion)
        dt = pd.concat([dt, df], ignore_index=True)
    dt.to_csv("../Result_tables/PA_Summary.csv", index=False)


if __name__ == "__main__":
    concate()
