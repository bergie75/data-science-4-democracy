import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
import warnings


def general_multi_regression(infile, attribute_list, response, outfile="place_holder.csv",
                             train_pct=0.9, alph=0.1, show_plot=True, covax=True):
    # this code produces some warnings from Pandas about assignment. In the future I would like to rectify
    # these, but for now we repress them along with divide-by-zero warnings when taking the logit
    # since later processing removes the undefined logits from the dataset
    warnings.simplefilter(action='ignore', category=Warning)

    # reads in the data
    path_to = "place_holder"
    df = pd.read_csv(path_to + infile)

    # reads in intended independent variables, separates out interaction terms and solo terms from specified variables
    solo_attribute_list = []
    mult_attribute_list = []
    for variable in attribute_list:
        if "*" in variable:
            mult_factors = variable.split("*")  # split command-line syntax of interaction term into list
            mult_attribute_list.append(mult_factors)
        else:
            solo_attribute_list.append(variable)
    sub_df = df[solo_attribute_list]  # extracts solo attributes
    # creates multiplied data columns
    for var_set in mult_attribute_list:
        col_mult = np.array(df.loc[:, var_set[0]])
        for i in range(1, len(var_set)):
            new_column = np.array(df.loc[:, var_set[i]])
            col_mult = np.multiply(col_mult, new_column)
            col_title = "*".join(var_set)
            sub_df[col_title] = col_mult

    target = np.array(df[response])
    name_vec = list(sub_df.columns)

    # checks to determine data is in [0,1]
    max_target = max(target)
    if max_target > 1.0:
        # uncomment the code below to produce runtime warnings for non-normalization
        # warnings.warn("Regression targets were not normalized, dividing by 100", RuntimeWarning)
        # the fix below is specific to my dataset, may need to be adjusted
        target = target/100.0

    logit_target = np.log(target)-np.log(1.0-target)
    sub_df["logit_target"] = logit_target

    # this section removes rows with undefined values, including those from the logit procedure
    nans_in_x = np.array(sub_df.isna().any(axis=1))
    infinite_logits = np.isinf(logit_target)  # comes from taking log(0)
    keep_rows = np.logical_not(np.logical_or(nans_in_x, infinite_logits))

    if covax:
        covax_target = "1_accounts_1_tweets_Mean_%_low-credibility"
        covax_col = np.array(df[covax_target])
        covax_present = covax_col > 0.0
        keep_rows = np.logical_and(covax_present, keep_rows)

    rows_remaining = sum(keep_rows)
    print("Rows remaining: " + str(rows_remaining))
    print("States of removed rows:")
    print(set(df['States'].loc[np.logical_not(keep_rows)]))

    sub_df = sub_df.loc[keep_rows, :]
    logit_target = np.array(sub_df.loc[:, sub_df.columns == "logit_target"])
    total_data_unscaled = np.array(sub_df.loc[:, sub_df.columns != "logit_target"])

    # rescales data using scikit-learn's standard scaler object
    scaler = StandardScaler()
    total_data = scaler.fit_transform(total_data_unscaled)

    if train_pct < 1.0:
        X_train, X_test, y_train, y_test = train_test_split(total_data, logit_target, train_size=train_pct)
    else:
        X_train = total_data
        y_train = logit_target

    # if there is only one variable, perform vanilla regression
    if alph > 0.0:
        regression = Lasso(alpha=alph).fit(X_train, y_train)
    else:
        regression = LinearRegression().fit(X_train, y_train)

    # outputs coefficient information for easy examination
    output_file = path_to + outfile
    coef_vec = np.reshape(regression.coef_, (len(name_vec), ))
    header_coef = ["variable", "coefficient"]

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_coef)
        for i in range(0, len(name_vec)):
            row = [name_vec[i], coef_vec[i]]
            writer.writerow(row)
    f.close()

    # outputs accuracy of regression if there is a train-test split
    if train_pct < 1.0:
        y_pred = np.reshape(regression.predict(X_test), np.shape(y_test))
        exp_var = explained_variance_score(y_test, y_pred)
        print("Explained variance: " + str(exp_var))
        if show_plot:
            residuals = y_test - y_pred
            scale = np.abs(residuals).max()
            plt.scatter(y_pred, residuals)
            plt.title("Predicted Logits vs. Residual Error")
            plt.ylim([-scale, scale])
            plt.ylabel("Residual")
            plt.xlabel("Predicted Logits")
            plt.show()

    return regression, exp_var


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "-ui":
        file = input('Enter the name of the input file here: ')
        resp = input('Type response variable here: ')
        tpct = float(input('Type train-test split here (float in (0,1)): '))
        attr_list_string = input('Type regression variables here, separated by spaces: ')
        attr_list = attr_list_string.split()
    elif sys.argv[1] == "-lv":
        file = sys.argv[2]
        a = float(sys.argv[3])
        attr_list = sys.argv[4:]
        resp = "CVR_2022_03_03"
        tpct = 0.9
    else:
        file = sys.argv[1]
        resp = sys.argv[2]
        tpct = float(sys.argv[3])
        attr_list = sys.argv[4:]

    general_multi_regression(infile=file, attribute_list=attr_list, response=resp, train_pct=tpct, alph=a)
