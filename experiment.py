import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def regressions(independent, dependent):
    vaccine_data = pd.read_csv("place_holder.csv")
    misinfo_data = pd.read_csv("place_holder.csv")

    # finds appropriate dataset for dependent variable key
    if dependent in misinfo_data.keys():
        dep_data = misinfo_data[["FIPS", dependent]]
    else:
        dep_data = vaccine_data[["FIPS", dependent]]
    # finds appropriate dataset for independent variable key
    if independent in misinfo_data.keys():
        ind_data = misinfo_data[["FIPS", independent]]
    else:
        ind_data = vaccine_data[["FIPS", independent]]

    total_df = pd.merge(ind_data, dep_data, on="FIPS")
    data = total_df[[dependent, independent]].dropna()

    lin = LinearRegression()
    lin.fit(np.array(data[independent]).reshape(-1, 1), data[dependent])

    plt.plot(data[independent], data[dependent], '.')
    plt.plot(data[independent], lin.predict(np.array(data[independent]).reshape(-1, 1)))
    plt.xlabel(independent)
    plt.ylabel(dependent)
    print("R^2 VALUE:")
    print(lin.score(np.array(data[independent]).reshape(-1, 1), data[dependent]))
    plt.show()


if __name__ == "__main__":
    import sys
    regressions(sys.argv[1], sys.argv[2])
