import numpy as np
from numpy.core.numeric import correlate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import pickle
import os
from .csv_data import *
from config.logger import *

logger = logging.getLogger('EDA')
setup_logger(logger,'logs/EDA.logs')

df = get_data('AReM')

pf = ProfileReport(df)


pf.to_file("/logistic_regression/main/EDA_Report/profile_report/AReM_Profile_Report.html")
logger.info("profile report has been saved before deletion of nulls at main/EDA_Report/profile_report/")

# After looking into the prifle report we found that there are 480 values are nulls. We want to delete these
# records becasue 6 ouf of 7 columns are nulls there so there is no use to fill these data with mean, mode or median.

def del_nan(df):
    col_num = len(df.columns)
    df = df.dropna(thresh = col_num-2)
    logger.info("all rows has been deleted where N-2 columns are blank")
    return df

df = del_nan(df)

pf = ProfileReport(df)

pf.to_file("/logistic_regression/main/EDA_Report/profile_report/without_null_AReM_Profile_Report.html")
logger.info("profile report has been saved after deletion of nulls at main/EDA_Report/profile_report/")

# After looking into the prifle report we found columns are skewed like below: -
# avr_rss12 = right skewed
# var_rss12 = left skewed
# var_rss13 = left skewed
# var_rss23 = left skewed
# So we can say that data is not not normalized, so this land us to check the outliers available in datasets

def boxplt(df,figname):
    try:
        ax  = plt.subplots(figsize = (20,20))
        sns.boxplot(data = df)
        plt.savefig("/logistic_regression/main/EDA_Report/box_plot/"+figname+".png")
        logger.info("boxplot has been saved without scaling the features")
    except Exception as e:
        logger.error("problem with boxplot in saving the image",e)
    return "box plot saved"

boxplt(df,"box_plot_before_scaling")

# After reviewing the data in box plot, we can say that it needs transformation/scaling.
# We will check box plot again after scaling the data.
def ftre_trgt_data(df=pd.DataFrame()):
    feature_data = df.drop(labels = 'Target', axis = 1)
    logger.info("segregated data for feature columns in feature_data")
    target_data = df['Target']
    logger.info("segregated data for target column in target_data")
    return feature_data,pd.DataFrame(target_data)

def scaling(df=pd.DataFrame()):
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    feature_scale = scaler.fit_transform(ftre_trgt_data(df)[0])
    logger.info("scaling feature data and keeping in feature scale")
    target_scale = label_encoder.fit_transform(ftre_trgt_data(df)[1])
    logger.info("scaling target data and keeping in target scale")
    return feature_scale, target_scale

# Now we can see that there are so many outliers available in data set, which we need to check and optimise
# to get a dataset where outliers is at minimum
boxplt(pd.DataFrame(scaling(df)[0]),"box_plot_after_scaling")

# calculating VIF score to check the multi-correlanity 
def vif_score():
    vif = pd.DataFrame([[ftre_trgt_data(df)[0].columns[i], variance_inflation_factor(scaling(df)[0],i)] for i in range(scaling(df)[0].shape[1])], columns = ['Features','VIF_Score'])
    logger.info("calculating VIF to check multicollinearity")
    return vif

logger.info((vif_score()))
# As all vif score is between 1 and 2 so its moderately correlated and we do not need to perform any activity here.


q = df['avg_rss13'].quantile(0.99)
df_new = df[df['avg_rss13']<q]

boxplt(pd.DataFrame(scaling(df_new)[0]),"box_plot_after_removing_elements_from_avg_rss13")

def removing_outliers(df_n,column=str,pct=int):
    q = df_n[column].quantile(pct)
    logger.info("defining column " + column + " and percentage " + str(pct) + " for quantile")
    df_new = df_n[df_n[column]<q]
    logger.info("limitating the dataset with given percentage so we can remove outliers")
    return df_new

boxplt(pd.DataFrame(scaling(removing_outliers(df_new,'var_rss13',0.99))[0]),"box_plot_after_removing_elements_from_var_rss13")

boxplt(pd.DataFrame(scaling(removing_outliers(df_new,'var_rss12',0.99))[0]),"box_plot_after_removing_elements_from_var_rss12")

boxplt(pd.DataFrame(scaling(removing_outliers(df_new,'var_rss23',0.99))[0]),"box_plot_after_removing_elements_from_var_rss23")

def new_data():
    return df_new