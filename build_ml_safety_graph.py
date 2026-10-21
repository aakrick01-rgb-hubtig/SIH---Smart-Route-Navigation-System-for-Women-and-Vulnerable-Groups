import os
import joblib
import pandas as pd
import networkx as nx


DATASET_PATH = "data/processed/road_safety_dataset.csv"
MODEL_PATH = "data/processed/safety_model.joblib"

INPUT_GRAPH = "data/processed/safepath_safety_graph.graphml"
OUTPUT_GRAPH = "data/processed/safepath_ml_safety_graph.graphml"

PREDICTION_OUTPUT = (
    "data/processed/road_safety_ml_predictions.csv"
)


FEATURES = [
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


def main():

    print("================================")
    print("SAFEPATH AI - ML GRAPH BUILDER")
    print("================================")

    # ==================================================
    # 1. CHECK REQUIRED FILES
    # ==================================================

    required_files = [
        DATASET_PATH,
        MODEL_PATH,
        INPUT_GRAPH
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nMissing required file:\n{path}"
            )

    # ==================================================
    # 2. LOAD SAFETY DATASET
    # ==================================================

    df = pd.read_csv(DATASET_PATH)

    print("\nSafety dataset loaded!")
    print("Rows:", len(df))

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"\nMissing model features:\n"
            f"{missing_features}"
        )

    # ==================================================
    # 3. LOAD RANDOM FOREST MODEL
    # ==================================================

    model = joblib.load(MODEL_PATH)

    print("Random Forest model loaded!")

    # ==================================================
    # 4. GENERATE ML PREDICTIONS
    # ==================================================

    X = df[FEATURES].copy()

    predictions = model.predict(X)

    # Keep predictions within valid safety-score range
    predictions = predictions.clip(0, 100)

    df["predicted_safety_score"] = predictions

    # Preserve the original engineered score
    df["baseline_safety_score"] = (
        df["safety_score"]
    )

    # ML prediction becomes the active score
    df["safety_score"] = (
        df["predicted_safety_score"]
    )

    # Lower risk = safer
    df["safety_risk"] = (
        100 - df["predicted_safety_score"]
    )

    # ==================================================
    # 5. SAVE PREDICTIONS
    # ==================================================

    df.to_csv(
        PREDICTION_OUTPUT,
        index=False
    )

    print("\nML predictions generated!")

    print(
        "Prediction range:",
        round(
            df["predicted_safety_score"].min(),
            2
        ),
        "-",
        round(
            df["predicted_safety_score"].max(),
            2
        )
    )

    print(
        "Predictions saved:",
        PREDICTION_OUTPUT
    )

    # ==================================================
    # 6. LOAD GRAPH USING NETWORKX
    # ==================================================
    #
    # IMPORTANT:
    # We deliberately use NetworkX instead of
    # ox.load_graphml().
    #
    # OSMnx has an attribute conversion issue with
    # the existing GraphML file.
    #
    # NetworkX successfully loads the same graph:
    # 34 nodes / 61 edges.
    # ==================================================

    G = nx.read_graphml(INPUT_GRAPH)

    print("\nSafety graph loaded!")
    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))

    # ==================================================
    # 7. VERIFY GRAPH/DATASET ALIGNMENT
    # ==================================================

    if len(G.edges) != len(df):

        raise RuntimeError(
            "\nGRAPH/DATASET MISMATCH\n"
            f"Graph edges: {len(G.edges)}\n"
            f"Dataset rows: {len(df)}\n"
            "\nPredictions cannot safely be attached."
        )

    # ==================================================
    # 8. ATTACH ML PREDICTIONS TO GRAPH EDGES
    # ==================================================

    print("\nAttaching ML predictions to graph...")

    graph_edges = list(
        G.edges(
            keys=True,
            data=True
        )
    )

    for index, (u, v, k, data) in enumerate(
        graph_edges
    ):

        row = df.iloc[index]

        # Original engineered score
        data["baseline_safety_score"] = float(
            row["baseline_safety_score"]
        )

        # Random Forest prediction
        data["predicted_safety_score"] = float(
            row["predicted_safety_score"]
        )

        # Active score used by routing
        data["safety_score"] = float(
            row["predicted_safety_score"]
        )

        # Lower = safer
        data["safety_risk"] = float(
            row["safety_risk"]
        )

    # ==================================================
    # 9. VALIDATE ATTACHMENT
    # ==================================================

    print("\nValidating ML graph...")

    validation_edges = list(
        G.edges(
            keys=True,
            data=True
        )
    )

    missing_predictions = 0

    for u, v, k, data in validation_edges:

        if "predicted_safety_score" not in data:

            missing_predictions += 1

    if missing_predictions > 0:

        raise RuntimeError(
            f"\n{missing_predictions} edges "
            "are missing ML predictions."
        )

    # ==================================================
    # 10. SAVE ML GRAPH
    # ==================================================

    nx.write_graphml(
        G,
        OUTPUT_GRAPH
    )

    # ==================================================
    # 11. FINAL REPORT
    # ==================================================

    print("\n================================")
    print("ML SAFETY GRAPH CREATED")
    print("================================")

    print(
        "Output:",
        OUTPUT_GRAPH
    )

    print(
        "Nodes:",
        len(G.nodes)
    )

    print(
        "Edges:",
        len(G.edges)
    )

    print(
        "\nML safety score range:",
        round(
            df["predicted_safety_score"].min(),
            2
        ),
        "-",
        round(
            df["predicted_safety_score"].max(),
            2
        )
    )

    print("\nSample ML edge:")

    sample_u, sample_v, sample_k, sample_data = (
        validation_edges[0]
    )

    print(
        "From:",
        sample_u
    )

    print(
        "To:",
        sample_v
    )

    print(
        "Baseline score:",
        round(
            float(
                sample_data["baseline_safety_score"]
            ),
            2
        )
    )

    print(
        "ML predicted score:",
        round(
            float(
                sample_data["predicted_safety_score"]
            ),
            2
        )
    )

    print(
        "Active safety score:",
        round(
            float(
                sample_data["safety_score"]
            ),
            2
        )
    )

    print(
        "Safety risk:",
        round(
            float(
                sample_data["safety_risk"]
            ),
            2
        )
    )

    print("\n================================")
    print("ML GRAPH INTEGRATION COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()
    