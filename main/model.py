from logging import Logger
from sklearn.linear_model import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from .data_analysis import new_data,scaling
from config.logger import *
import mlflow

logger = logging.getLogger('model')
setup_logger(logger,'logs/model.logs')

df = new_data()
print(df.head()) 

mlflow.sklearn.autolog()
with mlflow.start_run():

    def trn_tst_split(x,y,tst_size_pct,random_state_num):
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = tst_size_pct, random_state = random_state_num)
            logger.info("Splitting the data into train test")
        except Exception as e:
            logger.error("Error in splitting the data into train test",e)
        return x_train, x_test, y_train, y_test
    

    def log_reg(solver_name,multi_clss_nme):
        try:
            trn_tst = trn_tst_split(scaling(df)[0],scaling(df)[1],0.20,144)
            logr_saga = LogisticRegression(solver = solver_name, multi_class= multi_clss_nme)
            logger.info("Used logistic regression with solver {} and multi class name is {}".format(solver_name,multi_clss_nme))
            logr_saga.fit(trn_tst[0], trn_tst[2])
            logger.info("Training the model with solver {} and multi class name is {}".format(solver_name,multi_clss_nme))
            y_pred = logr_saga.predict(trn_tst[1])
            mlflow.log_param("Prdicting the values with solver ",solver_name)
            return logr_saga, y_pred
        except Exception as e:
            # print(e)
            logger.error("Error in logistic regression",e)
        
        

    # print(trn_tst_split(scaling(df)[0],scaling(df)[1],0.20,144)[1][0])

    # print("log_reg",log_reg('saga','ovr'))

    logger.info("confusion_matrix",confusion_matrix(trn_tst_split(scaling(df)[0],scaling(df)[1],0.20,144)[3], log_reg('saga','ovr')[1]))
    # logger.info(("confusion_matrix",confusion_matrix(trn_tst_split(scaling(df)[0],scaling(df)[1],0.20,144)[3], log_reg('saga','ovr')[1])))
    
    def crs_vldtn(crs_vldtn_num):
        try:
            scores = cross_val_score(log_reg('saga','ovr')[0], scaling(df)[0], scaling(df)[1], cv = crs_vldtn_num, scoring = 'accuracy')
            mlflow.log_metric(("cross_validation_score",scores))
            return scores.mean()
        except Exception as e:
            logger.error("Error in cross validation",e)
        

    crs_vldtn(10)

    logger.info("classification_report",classification_report(trn_tst_split(scaling(df)[0],scaling(df)[1],0.20,144)[3], log_reg('saga','ovr')[1]))
