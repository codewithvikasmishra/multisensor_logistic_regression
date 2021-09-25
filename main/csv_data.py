import os
import pandas as pd
from config.logger import *

logger = logging.getLogger('csv_data_merge')
setup_logger(logger,'logs/csv_data_merge.logs')

file = []

def get_files(folder_name=str):
    for (root, dir, files) in os.walk('/logistic_regression/'+folder_name):
        try:
            for trgt in dir:
                logger.info("list of directories : {}".format(str(trgt)))
                for (root, dir, files) in os.walk('/logistic_regression/'+folder_name + '/' +trgt):
                    for i in files:
                        logger.info("list of files : {}".format(str(i)))
                        if os.path.splitext(i)[1]=='.csv':
                            logger.info("csv files : {}".format(str(i)))
                            file.append([trgt,root+'/'+i])
            return file
        except Exception as e:
            logger.error("given folder is not available",e)
    

def get_data(tgt_folder=str):
    data_frame = pd.DataFrame()
    for file in get_files(tgt_folder):
        logger.info("csv file to get data : {}{}".format(str(file[0]),str(file[1])))
        try:
            data = pd.read_csv(file[1],skiprows=4,error_bad_lines=False,warn_bad_lines=True)
            data['Target']=file[0]
            data_frame = data_frame.append(data,ignore_index=True)       
            logger.info("csv data appended in DataFrame")
        except Exception as e:
            logger.error("error in reading csv file",e)
    return data_frame