import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use - run with all features for now
    num_features =  [
        'total_area_sqm', 'surface_land_sqm', 
        'nbr_frontages', 'nbr_bedrooms', 'terrace_sqm', 'garden_sqm',
        'primary_energy_consumption_sqm', 'cadastral_income'
        ]
    fl_features = [
        'fl_furnished',
        'fl_open_fire',
        'fl_terrace',
        'fl_garden',
        'fl_swimming_pool',
        'fl_floodzone',
        'fl_double_glazing'
        ]
    cat_features = [
        'property_type',
        'subproperty_type',
        'region',
        'province',
        'locality',
        'equipped_kitchen',
        'state_building',
        'epc',
        'heating_type'
    ]


    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # WIP
    #-------- Use DataFrameMapper to select which columns to standardize ------ 
    imputer = SimpleImputer(strategy="mean")
    enc = OneHotEncoder()
    scaler = StandardScaler()
    mapper = DataFrameMapper([
    (num_features, [imputer, scaler]),
    (cat_features, enc)
    ], df_out=True)
    
    # Test this first
    #result = np.round(mapper.fit_transform(X_train.copy()), 2)
    Z_train = mapper.fit_transform(X_train)
    Z_test = mapper.transform(X_test)

    # scaler = StandardScaler()
    # normalized_x_train = scaler.fit_transform(X_train)
    #-------------------------------------------------#

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model
    model = LinearRegression()
    model.fit(Z_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(Z_train))
    test_score = r2_score(y_test, model.predict(Z_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
        "scaler": scaler
    }
    joblib.dump(artifacts, "models/artifacts-standardscaler.joblib")


if __name__ == "__main__":
    train()