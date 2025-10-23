import pandas as pd
import joblib
import os
import numpy as np 

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

MODEL_FILE="model.pkl"
Pipeline_FILE="pipeline.pkl"

def build_pipeline(num_attri, cat_attri):
    num_pipeline=Pipeline([
        ("scaler" , StandardScaler())
    ])

    cat_pipeline=Pipeline([
        ("onehot" , OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline=ColumnTransformer([
        ("num" , num_pipeline , num_attri),
        ("cat" , cat_pipeline , cat_attri)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    insurance=pd.read_csv("insurance.csv")
    insurance["bmi_cat"]=pd.cut(insurance["bmi"] , bins=[15 , 25 , 35 , 45 , 55 , np.inf] , labels=[1 ,2 ,3 ,4, 5])

    split=StratifiedShuffleSplit(n_splits=1 , test_size=0.2 , random_state=42)
    for train_index , test_index in split.split(insurance , insurance["bmi_cat"]):
        train_set=insurance.iloc[train_index].drop("bmi_cat" , axis=1)
        test_set=insurance.iloc[test_index].drop("bmi_cat" , axis=1).to_csv("testset.csv")

    insurance_labels=train_set["charges"].copy()
    insurance_features=train_set.drop("charges" , axis=1)

    num_attri=["age" , "bmi" , "children"]
    cat_attri=["sex","smoker","region"]

    pipeline=build_pipeline(num_attri ,cat_attri)
    final_insurance=pipeline.fit_transform(insurance_features)

    model=DecisionTreeRegressor(random_state=42)
    model.fit(final_insurance , insurance_labels)

    joblib.dump(model ,MODEL_FILE)
    joblib.dump(pipeline ,Pipeline_FILE)
    print("congrats the model is trained")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(Pipeline_FILE)
    input_data=pd.read_csv("testset.csv")
    transformes_input=pipeline.transform(input_data)
    predictions=model.predict(transformes_input)
    input_data["charges"]=predictions

    input_data.to_csv("output.csv")
    print("done")


