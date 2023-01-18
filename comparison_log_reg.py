if __name__ == "__main__":
    import sys
    from general_multi_regression import general_multi_regression
    import matplotlib.pyplot as plt

    file = sys.argv[1]
    a = float(sys.argv[2])
    attr_list = sys.argv[3:]
    resp = "CVR_2022_03_03"
    tpct = 0.9

    full_variances = []
    just_gop_variances = []

    for i in range(0, 500):
        full_vexp = general_multi_regression(infile=file, attribute_list=attr_list, response=resp, train_pct=tpct,
                                  show_plot=False, alph=a)[1]
        full_variances.append(full_vexp)

        gop_vexp = general_multi_regression(infile=file, attribute_list=["prop_gop_vote"], response=resp, train_pct=tpct,
                                 show_plot=False, alph=a)[1]
        just_gop_variances.append(gop_vexp)

    data = [full_variances, just_gop_variances]

    # plotting results
    plt.figure(figsize=(9, 9))
    plt.title("Box Plot of Explained Variance in 500 Logistic Regression Models")

    # Creating plot
    plt.boxplot(data)

    # show plot
    plt.show()
