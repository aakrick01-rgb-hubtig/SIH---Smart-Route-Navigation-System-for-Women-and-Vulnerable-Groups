import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATASET_PATH = "data/processed/road_safety_dataset.csv"
MODEL_PATH = "data/processed/safety_model.joblib"


def main():

    print("================================")
    print("SAFEPATH AI - ML MODEL TRAINING")
    print("================================")

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded!")
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    # -----------------------------
    # Target
    # -----------------------------
    target = "safety_score"

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset."
        )

    # -----------------------------
    # Features
    # -----------------------------
    features = [
        "length",
        "highway",
        "road_connectivity",
        "road_type_score",
        "nearest_crime_m",
        "nearest_amenity_m",
        "average_lux_lighting",
        "cctv_coverage_pct",
        "crowd_density_index",
        "ambient_distance_m"
    ]

    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    X = df[features].copy()
    y = df[target].astype(float)

    # -----------------------------
    # Identify feature types
    # -----------------------------
    categorical_features = [
        "highway",
        "crowd_density_index"
    ]

    numerical_features = [
        "length",
        "road_connectivity",
        "road_type_score",
        "nearest_crime_m",
        "nearest_amenity_m",
        "average_lux_lighting",
        "cctv_coverage_pct",
        "ambient_distance_m"
    ]

    # -----------------------------
    # Preprocessing
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # -----------------------------
    # Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # -----------------------------
    # Train
    # -----------------------------
    print("\nTraining Random Forest...")

    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n================================")
    print("MODEL PERFORMANCE")
    print("================================")

    print("MAE:", round(mae, 2))
    print("R² Score:", round(r2, 2))

    # -----------------------------
    # Feature importance
    # -----------------------------
    preprocessor_fitted = pipeline.named_steps[
        "preprocessor"
    ]

    rf_model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor_fitted
        .get_feature_names_out()
    )

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(
        "importance",
        ascending=False
    )

    print("\n================================")
    print("FEATURE IMPORTANCE")
    print("================================")

    print(
        importance_df.to_string(index=False)
    )

    # -----------------------------
    # Save model
    # -----------------------------
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("\n================================")
    print("MODEL SAVED")
    print("================================")

    print(
        "Location:",
        MODEL_PATH
    )

    print("\nRandom Forest integration ready!")


if __name__ == "__main__":
    main()