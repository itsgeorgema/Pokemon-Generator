#!/usr/bin/env python3
"""
Precompute and cache Pokemon stat models at Docker build time.

This script mirrors the logic in app.load_pokemon_data, but runs offline during
image build so that your Fly.io machines don't have to do this work on boot.

Inspired by the \"do heavy work at build time\" pattern described here:
https://fly.io/phoenix-files/speed-up-your-boot-times-with-this-one-dockerfile-trick/
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def precompute_models(
    data_path: str = "data/Pokemon_stats.csv",
    cache_path: str = "models/stat_models.pkl",
) -> None:
    """Train RandomForest models for each stat and cache them to disk."""
    print(f"[BUILD] Precomputing stat models from {data_path}")
    metadata = pd.read_csv(data_path)

    type1_col = "Type 1" if "Type 1" in metadata.columns else "Type1"
    type2_col = "Type 2" if "Type 2" in metadata.columns else "Type2"

    types = sorted(
        list(
            set(
                metadata[type1_col].dropna().tolist()
                + metadata[type2_col].dropna().tolist()
            )
        )
    )

    type_encoder = LabelEncoder()
    metadata["Type1_encoded"] = type_encoder.fit_transform(metadata[type1_col])

    metadata["Type2_filled"] = metadata[type2_col].fillna("None")
    all_types = list(type_encoder.classes_) + ["None"]
    type_encoder.classes_ = np.array(all_types)
    metadata["Type2_encoded"] = type_encoder.transform(metadata["Type2_filled"])

    stats_data = pd.read_csv(data_path)

    models = {}
    for stat in ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]:
        # Create a mapping from Type 1/Type 2 to the stats
        type_to_stat = {}
        for _, row in stats_data.iterrows():
            type1 = row["Type 1"]
            type2 = row["Type 2"] if pd.notna(row["Type 2"]) else "None"
            key = (type1, type2)
            if key not in type_to_stat:
                type_to_stat[key] = []
            type_to_stat[key].append(row[stat])

        # For each type combination, take the average stat value
        for key, values in type_to_stat.items():
            type_to_stat[key] = sum(values) / len(values)

        # Train a model on the encoded types
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Create training data from the type_to_stat mapping
        X_train = []
        y_train = []
        for (type1, type2), stat_value in type_to_stat.items():
            type1_encoded = 0
            type2_encoded = 0

            if type1 in type_encoder.classes_:
                type1_idx = np.where(type_encoder.classes_ == type1)[0]
                if len(type1_idx) > 0:
                    type1_encoded = int(type1_idx[0])

            if type2 in type_encoder.classes_:
                type2_idx = np.where(type_encoder.classes_ == type2)[0]
                if len(type2_idx) > 0:
                    type2_encoded = int(type2_idx[0])

            X_train.append([type1_encoded, type2_encoded])
            y_train.append(stat_value)

        model.fit(X_train, y_train)
        models[stat] = model

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "metadata": metadata,
                "types": types,
                "type_encoder": type_encoder,
                "models": models,
            },
            f,
        )

    print(f"[BUILD] Cached stat models to {cache_path}")


if __name__ == "__main__":
    precompute_models()


