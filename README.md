File description:

model: I have two model structure in total (XGB and LSTM)
       /model/whole_model_best_model__.h5 are the whole LSTM models
       /model/best_model__.h5 are the weight for the LSTM models
       /model/xgb_model_fold__.model are the model for XGBoost models

script to train the model:
  --> 1-nlp_feature_extraction.py,
      2-non-nlp_feature_extraction.py,
      2.1-magicFeature.py and
      2.2-glove_embedding.py are the four pre-request file need to run BEFORE training the model

  --> 3-modelLSTM.py and 3-modelXGB.py are the script to TRAIN the model
  --> 4-postpreocess.py is for data rescaling

data needed:
      in order to run the code, need to download the following files, unzipped them, and put under /data director
      1. train.csv.zip: https://www.kaggle.com/c/quora-question-pairs/data (ONLY train.csv)
      2. pre-trained word vector, glove.840B.300d, from https://nlp.stanford.edu/projects/glove/

input sample: /data/test.csv
              this file contains the first 17 question pairs in the provided test set.
              you can change the question inside it, but need to maintain the number of question pairs
              otherwise, you need to re-run all the file in the sequence I mentioned above

test script: 0-test.py

Instructions to run the model and make predictions:
      1. download 3 files mentioned above
      2. cd to this main directory
      3. type 'python3 0-test.py' in terminal (this file load the model and predict the result) (takes around 3 minutes to compile)
      4. type 'python3 4-postprocess.py' in terminal (this file rescale training prediction to test prediction)
      5. the '/predictions/submission.csv' is the final outputs
      6. 'is_duplicate' in /predictions/submission.csv is the prediction result. if number close to 0, means the question pair is   
          more likely not duplicate. If the number is close to 1, means it is more probable to be duplicate
          (the test sample is the first 17 question pairs from the provided dataset. The best result (log_loss=0.12890) I submitted for that 17 pairs are:
              is_duplicate	test_id
              0.0025486459	         0
              0.048980465	           1
              0.0716757667	         2
              1.67843909793338E-06	 3
              0.1147136606	         4
              0.0016034311	         5
              0.9995872145	         6
              0.4532509187	         7
              0.0949054497	         8
              0.001004457	           9
              0.0092897234	         10
              2.08631441023057E-06	 11
              1.17789102260214E-07	 12
              0.0414555062	         13
              0.0127388094	         14
              0.0042287693	         15
              3.30835855200168E-05	 16)
            you can compare the result from submission file to these data
