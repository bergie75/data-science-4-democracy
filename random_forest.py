import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import warnings


def random_forest(infile, attribute_list, response, max_samp=0.3, outfile="place_holder.csv",
                  train_pct=0.9, show_plot=True, covax=True):
    # this code produces some warnings from Pandas about assignment. In the future I would like to rectify
    # these, but for now we repress them along with divide-by-zero warnings when taking the logit
    # since later processing removes the undefined logits from the dataset
    warnings.simplefilter(action='ignore', category=Warning)

    path_to = "place_holder"
    df = pd.read_csv(path_to + infile)

    # reads in intended independent variables, separates out interaction terms and solo terms
    solo_attribute_list = []
    mult_attribute_list = []
    for variable in attribute_list:
        if "*" in variable:
            mult_factors = variable.split("*")
            mult_attribute_list.append(mult_factors)
        else:
            solo_attribute_list.append(variable)
    sub_df = df[solo_attribute_list]
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

    forest = RandomForestRegressor(max_samples=max_samp).fit(X_train, y_train)

    # outputs coefficient information for easy examination
    output_file = path_to + outfile
    importances = np.reshape(forest.feature_importances_, (len(name_vec),))
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)



    header_imp = ["variable", "importance"]
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_imp)
        for i in range(0, len(name_vec)):
            row = [name_vec[i], importances[i]]
            writer.writerow(row)
        f.close()

    # outputs accuracy of regression if there is a train-test split
    if train_pct < 1.0:
        y_pred = forest.predict(X_test)
        exp_var = explained_variance_score(y_test, y_pred)
        print("Explained variance: " + str(exp_var))
        if show_plot:
            #residuals = np.transpose(y_test) - y_pred
            #plt.scatter(y_pred, residuals)
            #scale = np.abs(residuals).max()
            #plt.title("Predicted Logits vs. Residual Error")
            #plt.ylim([-scale, scale])
            #plt.ylabel("Residual")
            #plt.xlabel("Predicted Logits")
            #plt.show()

            forest_importances = pd.Series(importances, index=name_vec)
            forest_importances.plot.bar(yerr=std)
            plt.title("Feature Importances in Random Forest")
            plt.ylabel("Mean decrease in impurity")
            plt.gcf().subplots_adjust(bottom=0.45)
            plt.show()

    return forest, exp_var


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "-ui":
        file = input('Enter the name of the input file here: ')
        resp = input('Type response variable here: ')
        tpct = float(input('Type train-test split here (float in (0,1)): '))
        ms = float(input('Type max-sample percentage here (float in (0,1)): '))
        attr_list_string = input('Type regression variables here, separated by spaces: ')
        attr_list = attr_list_string.split()
    elif sys.argv[1] == "-lv":
        file = sys.argv[2]
        ms = float(sys.argv[3])
        attr_list = sys.argv[4:]
        resp = "CVR_2022_03_03"
        tpct = 0.9
    else:
        file = sys.argv[1]
        resp = sys.argv[2]
        tpct = float(sys.argv[3])
        ms = float(sys.argv[4])
        attr_list = sys.argv[5:]

    random_forest(infile=file, attribute_list=attr_list, response=resp, train_pct=tpct)
