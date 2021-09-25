from typing import Collection
from flask import Flask, request, jsonify,abort, after_this_request
import os
import subprocess
import json
from flask.ctx import after_this_request
from main.model import *
from config.logger import *

logger = logging.getLogger('api')
setup_logger(logger,'logs/api.logs')

app = Flask(__name__)

@app.route('/multisensor_data_fusion/predict', methods = ['POST'])
def logstc_reg_pred():
    pred_list = request.get_json(['ftr_lst'])
    if request.get_json().keys()!={'ftr_lst'}:
        return abort(400, "Please pass the key as ftr_lst.")
    elif request.get_json('ftr_lst')['ftr_lst']=='':
        return abort(400, "Please pass key and value.")
    # trn_tst = trn_tst_split(scaling(df)[0],scaling(df)[1],0.20,144)
    y_pred = log_reg('saga','ovr')[0].predict(pred_list['ftr_lst'])
    logger.info("Prdicting the values {}".format(y_pred))
    return str(y_pred)

# print(logstc_reg_pred([[-1.15499754, 1.18953025, -0.47488601 ,0.71432177, -0.92055492, -0.63724411, -0.70785032]]))

if __name__ == '__main__':
    app.run(debug= True)