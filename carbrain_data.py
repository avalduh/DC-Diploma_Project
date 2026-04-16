from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "df_final.csv"


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = [col.strip() for col in df.columns]

    for col in ["MAKETXT", "MODELTXT", "COMPDESC", "CDESCR"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    numeric_cols = ["INJURED", "DEATHS", "MILES", "OCCURENCES", "YEARTXT", "AGE"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FIRE" in df.columns:
        df["FIRE"] = df["FIRE"].map({"Y": 1, "N": 0})
    if "CRASH" in df.columns:
        df["CRASH"] = df["CRASH"].map({"Y": 1, "N": 0})

    df = df[df["YEARTXT"].notna()].copy()
    df = df[df["MAKETXT"].ne("NAN") & df["MODELTXT"].ne("NAN")].copy()

    df = df[(df["INJURED"].fillna(0) >= 0) & (df["INJURED"].fillna(0) <= 10)].copy()
    df = df[(df["DEATHS"].fillna(0) >= 0) & (df["DEATHS"].fillna(0) <= 10)].copy()

    df["compdesc_clean"] = df["COMPDESC"].astype(str).str.upper().str.strip()
    bad_values = {"UNKNOWN", "NONE", "NAN", "OTHER/UNKNOWN", "OTHER/I AM NOT SURE", ""}
    df = df[~df["compdesc_clean"].isin(bad_values)].copy()

    df["compdesc_lvl1"] = df["compdesc_clean"].str.split(":").str[0]

    df.loc[df["MILES"] < 0, "MILES"] = np.nan
    df.loc[df["OCCURENCES"] < 0, "OCCURENCES"] = np.nan
    df.loc[df["AGE"] < 0, "AGE"] = np.nan

    for col in ["MILES", "OCCURENCES"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

    df["OCCURENCES"] = df["OCCURENCES"].fillna(1)
    df["component_family"] = df["compdesc_lvl1"].apply(_map_component_family)

    return df.reset_index(drop=True)


def build_vehicle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    vehicle_df = (
        df.groupby(["MAKETXT", "MODELTXT", "YEARTXT"], dropna=False)
        .agg(
            total_complaints=("MODELTXT", "size"),
            avg_age=("AGE", "mean"),
            median_miles=("MILES", "median"),
            avg_occurrences=("OCCURENCES", "mean"),
            crash_rate=("CRASH", "mean"),
            fire_rate=("FIRE", "mean"),
            injured_rate=("INJURED", lambda s: (s.fillna(0) > 0).mean()),
            total_injured=("INJURED", "sum"),
            total_deaths=("DEATHS", "sum"),
            top_issue=("compdesc_lvl1", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
    )

    family_dist = (
        pd.crosstab(
            index=[df["MAKETXT"], df["MODELTXT"], df["YEARTXT"]],
            columns=df["component_family"],
            normalize="index",
        )
        .reset_index()
    )

    vehicle_df = vehicle_df.merge(
        family_dist,
        on=["MAKETXT", "MODELTXT", "YEARTXT"],
        how="left",
    )

    expected_families = [
        "seguridad_pasiva",
        "seguridad_activa_control",
        "propulsion_electrico_termico",
        "asistencia_visibilidad",
        "otros",
    ]
    for col in expected_families:
        if col not in vehicle_df.columns:
            vehicle_df[col] = 0.0

    fill_cols = [
        "avg_age",
        "median_miles",
        "avg_occurrences",
        "crash_rate",
        "fire_rate",
        "injured_rate",
        "total_injured",
        "total_deaths",
    ]
    for col in fill_cols:
        vehicle_df[col] = vehicle_df[col].fillna(vehicle_df[col].median())

    vehicle_df["log_total_complaints"] = np.log1p(vehicle_df["total_complaints"])
    vehicle_df["log_median_miles"] = np.log1p(vehicle_df["median_miles"])
    vehicle_df["log_avg_occurrences"] = np.log1p(vehicle_df["avg_occurrences"])

    vehicle_df["severity_score"] = (
        0.50 * vehicle_df["crash_rate"]
        + 0.20 * vehicle_df["fire_rate"]
        + 0.30 * vehicle_df["injured_rate"]
    )

    vehicle_df["severity_binary"] = (
        vehicle_df["crash_rate"] + vehicle_df["injured_rate"]
    )

    vehicle_df["component_risk_score"] = (
        0.45 * vehicle_df["seguridad_pasiva"]
        + 0.30 * vehicle_df["seguridad_activa_control"]
        + 0.15 * vehicle_df["propulsion_electrico_termico"]
        + 0.10 * vehicle_df["asistencia_visibilidad"]
    )

    volume_score = (
        vehicle_df["log_total_complaints"] / vehicle_df["log_total_complaints"].max()
    ).fillna(0.0)

    vehicle_df["risk_score"] = (
        0.70 * vehicle_df["severity_score"]
        + 0.15 * vehicle_df["component_risk_score"]
        + 0.15 * volume_score
    )
    vehicle_df["risk_score"] = vehicle_df["risk_score"].clip(0, 1)

    vehicle_df["risk_percentile"] = vehicle_df["risk_score"].rank(pct=True)
    vehicle_df["risk_label"] = vehicle_df["risk_score"].apply(_risk_label)
    vehicle_df["decision"] = vehicle_df["risk_score"].apply(_recommendation_from_score)

    vehicle_df = _assign_clusters(vehicle_df)
    vehicle_df = _assign_subclusters(vehicle_df)

    return vehicle_df.sort_values(
        ["risk_score", "total_complaints"], ascending=[False, False]
    ).reset_index(drop=True)


def build_brand_ranking(vehicle_df: pd.DataFrame, min_complaints: int = 30) -> pd.DataFrame:
    brand_df = vehicle_df[vehicle_df["total_complaints"] >= min_complaints].copy()
    if brand_df.empty:
        brand_df = vehicle_df.copy()

    ranking = (
        brand_df.groupby("MAKETXT")
        .agg(
            models=("MODELTXT", "nunique"),
            total_complaints=("total_complaints", "sum"),
            avg_risk_score=("risk_score", "mean"),
            avg_risk_percentile=("risk_percentile", "mean"),
            avg_crash_rate=("crash_rate", "mean"),
            avg_fire_rate=("fire_rate", "mean"),
            avg_injured_rate=("injured_rate", "mean"),
        )
        .reset_index()
        .sort_values(["avg_risk_score", "total_complaints"], ascending=[True, False])
    )

    if ranking["avg_risk_score"].nunique() >= 4:
        ranking["brand_label"] = pd.qcut(
            ranking["avg_risk_score"],
            q=4,
            labels=["Excelente", "Buena", "Precaución", "Riesgosa"],
            duplicates="drop",
        )
    else:
        ranking["brand_label"] = "Sin suficiente dispersión"

    return ranking.reset_index(drop=True)


def get_vehicle_record(
    vehicle_df: pd.DataFrame, make: str, model: str, year: int | float
) -> pd.Series | None:
    result = vehicle_df[
        (vehicle_df["MAKETXT"].str.upper() == str(make).upper())
        & (vehicle_df["MODELTXT"].str.upper() == str(model).upper())
        & (vehicle_df["YEARTXT"] == year)
    ]
    if result.empty:
        return None
    return result.iloc[0]


def build_chat_context(
    record: pd.Series | None,
    brand_ranking: pd.DataFrame,
    vehicle_df: pd.DataFrame,
) -> str:
    lines = [
        "Sistema CarBrain para apoyar decisiones de compra de autos seminuevos.",
        f"Total de vehículos analizados en la base: {len(vehicle_df):,}.",
    ]

    best_brands = brand_ranking.head(5)
    if not best_brands.empty:
        lines.append("Marcas que en general muestran mejor comportamiento en la base:")
        for _, row in best_brands.iterrows():
            lines.append(
                f"- {row['MAKETXT']}: comportamiento general {row.get('brand_label', 'No disponible')}"
            )

    if record is None:
        lines.append("")
        lines.append(
            "No hay suficiente información del vehículo seleccionado para dar una recomendación específica."
        )
        return "\n".join(lines)

    brand_peers = vehicle_df[vehicle_df["MAKETXT"] == record["MAKETXT"]].copy()
    brand_avg = brand_peers["risk_score"].mean() if not brand_peers.empty else np.nan

    same_brand_pool = vehicle_df[
        (vehicle_df["MAKETXT"] == record["MAKETXT"])
        & (vehicle_df["total_complaints"] >= 20)
    ].copy()

    dataset_percentile = float(record.get("risk_percentile", np.nan))
    brand_better_ratio = (
        (same_brand_pool["risk_score"] < record["risk_score"]).mean()
        if not same_brand_pool.empty
        else np.nan
    )

    if pd.notna(dataset_percentile):
        if dataset_percentile <= 0.33:
            similar_comparison = "Tiene menos señales de alerta que la mayoría de autos similares."
        elif dataset_percentile <= 0.66:
            similar_comparison = "Está en un punto intermedio frente a autos similares."
        else:
            similar_comparison = "Presenta más señales de alerta que la mayoría de autos similares."
    else:
        similar_comparison = "No hay suficiente información para compararlo con autos similares."

    if pd.notna(brand_avg):
        if record["risk_score"] > brand_avg:
            brand_comparison = "Está por encima del promedio de alertas de su marca."
        else:
            brand_comparison = "Está igual o mejor que el promedio de su marca."
    else:
        brand_comparison = "No hay suficiente información para compararlo contra su marca."

    subcluster_label = str(record.get("subcluster_label", "")).strip().lower()
    if "eléctrico" in subcluster_label or "combustible" in subcluster_label:
        issue_pattern = "Las fallas suelen concentrarse en sistema eléctrico o combustible."
    elif "seguridad" in subcluster_label:
        issue_pattern = "Las alertas se concentran más en seguridad y control del vehículo."
    elif "desgaste" in subcluster_label or "antigüedad" in subcluster_label:
        issue_pattern = "El patrón apunta más a desgaste por uso y antigüedad."
    elif "base confiable" in subcluster_label:
        issue_pattern = "No destaca un patrón grave dominante frente a otros autos de la base."
    elif "alto riesgo crítico" in subcluster_label:
        issue_pattern = "El patrón general muestra alertas importantes que requieren mucha cautela."
    else:
        issue_pattern = "Hay un patrón de fallas reportadas, pero no es concluyente."

    recommendation_rule = _recommendation_from_score(float(record["risk_score"]))

    lines.extend(
        [
            "",
            "Vehículo seleccionado:",
            f"- Marca: {record['MAKETXT']}",
            f"- Modelo: {record['MODELTXT']}",
            f"- Año: {int(record['YEARTXT'])}",
            f"- Veredicto sugerido: {record.get('decision', recommendation_rule)}",
            f"- Nivel general de atención: {record['risk_label']}",
            f"- Total de reportes: {int(record['total_complaints'])}",
            f"- Falla más reportada: {record['top_issue']}",
            f"- Choques reportados: {record['crash_rate']:.1%}",
            f"- Incendios reportados: {record['fire_rate']:.1%}",
            f"- Casos con lesionados: {record['injured_rate']:.1%}",
            f"- Comparación frente a autos similares: {similar_comparison}",
            f"- Comparación frente a su marca: {brand_comparison}",
            f"- Patrón principal detectado: {issue_pattern}",
        ]
    )

    if pd.notna(brand_better_ratio):
        lines.append(
            f"- Dentro de su marca, aproximadamente {brand_better_ratio:.1%} de los modelos comparables salen mejor parados."
        )

    lines.extend(
        [
            "",
            "Guía para responder al usuario:",
            "- Explica si parece buena compra o no.",
            "- Señala las fallas o alertas más importantes.",
            "- Recomienda qué revisar antes de comprar.",
            "- Usa lenguaje sencillo y práctico.",
            "- No hables de cluster, subcluster, percentil, silhouette o score estadístico.",
        ]
    )

    return "\n".join(lines)


def summarize_cluster_profiles(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = (
        vehicle_df.groupby(["cluster_id", "cluster_label"])
        .agg(
            vehicles=("cluster_id", "size"),
            avg_risk_score=("risk_score", "mean"),
            avg_percentile=("risk_percentile", "mean"),
            avg_complaints=("total_complaints", "mean"),
            avg_crash_rate=("crash_rate", "mean"),
            avg_fire_rate=("fire_rate", "mean"),
            avg_injured_rate=("injured_rate", "mean"),
            avg_age=("avg_age", "mean"),
            top_issue=("top_issue", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
        .sort_values(["avg_risk_score", "vehicles"], ascending=[False, False])
    )

    cluster_df["cluster_description"] = cluster_df.apply(_cluster_description, axis=1)
    return cluster_df


def summarize_subcluster_profiles(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = vehicle_df[vehicle_df["subcluster"] >= 0].copy()
    if sub_df.empty:
        return pd.DataFrame()

    profile = (
        sub_df.groupby(["subcluster", "subcluster_label"])
        .agg(
            vehicles=("subcluster", "size"),
            avg_risk_score=("risk_score", "mean"),
            avg_complaints=("total_complaints", "mean"),
            avg_crash_rate=("crash_rate", "mean"),
            avg_fire_rate=("fire_rate", "mean"),
            avg_injured_rate=("injured_rate", "mean"),
            avg_age=("avg_age", "mean"),
            top_issue=("top_issue", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
        .sort_values(["avg_risk_score", "vehicles"], ascending=[False, False])
    )
    return profile


def get_clustering_diagnostics(vehicle_df: pd.DataFrame) -> dict:
    cluster_features = [
        "severity_score",
        "component_risk_score",
        "log_total_complaints",
        "avg_age",
        "log_median_miles",
        "log_avg_occurrences",
        "crash_rate",
        "fire_rate",
        "injured_rate",
    ]

    cluster_input = vehicle_df[cluster_features].copy()
    cluster_input = cluster_input.replace([np.inf, -np.inf], np.nan)
    cluster_input = cluster_input.fillna(cluster_input.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(cluster_input)

    max_k = min(8, len(vehicle_df) - 1)
    diagnostics = []

    if max_k < 2:
        return {
            "best_k": 2,
            "best_silhouette": np.nan,
            "diagnostics": pd.DataFrame(columns=["k", "inertia", "silhouette_score"]),
        }

    best_k = 2
    best_score = -1.0

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        diagnostics.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(sil),
            }
        )
        if sil > best_score:
            best_score = sil
            best_k = k

    return {
        "best_k": best_k,
        "best_silhouette": best_score,
        "diagnostics": pd.DataFrame(diagnostics),
    }


def _map_component_family(value: str) -> str:
    if value in ["AIR BAGS", "STRUCTURE"]:
        return "seguridad_pasiva"
    if value in [
        "SERVICE BRAKES",
        "SERVICE BRAKES, HYDRAULIC",
        "STEERING",
        "VEHICLE SPEED CONTROL",
        "ELECTRONIC STABILITY CONTROL (ESC)",
        "SUSPENSION",
        "TIRES",
    ]:
        return "seguridad_activa_control"
    if value in [
        "ELECTRICAL SYSTEM",
        "FUEL/PROPULSION SYSTEM",
        "FUEL SYSTEM, GASOLINE",
        "ENGINE",
        "ENGINE AND ENGINE COOLING",
        "POWER TRAIN",
    ]:
        return "propulsion_electrico_termico"
    if value in [
        "EXTERIOR LIGHTING",
        "VISIBILITY",
        "VISIBILITY/WIPER",
        "FORWARD COLLISION AVOIDANCE",
    ]:
        return "asistencia_visibilidad"
    return "otros"


def _risk_label(score: float) -> str:
    if score <= 0.08:
        return "Muy confiable"
    if score <= 0.15:
        return "Confiable"
    if score <= 0.25:
        return "Con precaución"
    if score <= 0.40:
        return "Riesgoso"
    return "Muy riesgoso"


def _recommendation_from_score(score: float) -> str:
    if score < 0.15:
        return "Recomendado"
    if score < 0.25:
        return "Con precaución"
    return "No recomendado"


def _assign_clusters(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    cluster_features = [
        "severity_score",
        "component_risk_score",
        "log_total_complaints",
        "avg_age",
        "log_median_miles",
        "log_avg_occurrences",
        "crash_rate",
        "fire_rate",
        "injured_rate",
    ]

    cluster_input = vehicle_df[cluster_features].copy()
    cluster_input = cluster_input.replace([np.inf, -np.inf], np.nan)
    cluster_input = cluster_input.fillna(cluster_input.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(cluster_input)

    max_k = min(8, len(vehicle_df) - 1)
    best_k = 2
    best_score = -1.0

    if max_k >= 2:
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(scaled)
            sil = silhouette_score(scaled, labels)
            if sil > best_score:
                best_score = sil
                best_k = k

    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    best_labels = model.fit_predict(scaled)

    vehicle_df = vehicle_df.copy()
    vehicle_df["cluster_id"] = best_labels

    cluster_profile = (
        vehicle_df.groupby("cluster_id")
        .agg(
            avg_risk_score=("risk_score", "mean"),
            avg_crash_rate=("crash_rate", "mean"),
            avg_fire_rate=("fire_rate", "mean"),
            avg_injured_rate=("injured_rate", "mean"),
            avg_age=("avg_age", "mean"),
            avg_complaints=("total_complaints", "mean"),
        )
        .reset_index()
        .sort_values(["avg_risk_score", "avg_crash_rate"], ascending=[True, True])
    )

    label_map = _build_cluster_label_map(cluster_profile)
    desc_map = _build_cluster_description_map(cluster_profile)

    vehicle_df["cluster_label"] = vehicle_df["cluster_id"].map(label_map)
    vehicle_df["cluster_description"] = vehicle_df["cluster_id"].map(desc_map)

    return vehicle_df


def _assign_subclusters(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    vehicle_df = vehicle_df.copy()
    vehicle_df["subcluster"] = -1
    vehicle_df["subcluster_label"] = "Alto nivel de alerta"

    cluster_risk_map = (
        vehicle_df.groupby("cluster_id")["risk_score"]
        .mean()
        .sort_values()
    )

    if cluster_risk_map.empty:
        return vehicle_df

    low_risk_cluster = int(cluster_risk_map.index[0])
    low_risk_df = vehicle_df[vehicle_df["cluster_id"] == low_risk_cluster].copy()

    if len(low_risk_df) < 10:
        vehicle_df.loc[
            vehicle_df["cluster_id"] == low_risk_cluster, "subcluster_label"
        ] = "Base confiable"
        return vehicle_df

    subcluster_features = [
        "severity_score",
        "total_complaints",
        "avg_age",
        "median_miles",
        "avg_occurrences",
        "crash_rate",
        "fire_rate",
        "injured_rate",
    ]

    X_sub = low_risk_df[subcluster_features].copy()
    X_sub = X_sub.replace([np.inf, -np.inf], np.nan)
    X_sub = X_sub.fillna(X_sub.median(numeric_only=True))

    X_sub_scaled = StandardScaler().fit_transform(X_sub)

    k_sub = min(3, max(2, len(low_risk_df) // 150))
    kmeans_sub = KMeans(n_clusters=k_sub, random_state=42, n_init=20)
    low_risk_df["subcluster"] = kmeans_sub.fit_predict(X_sub_scaled)

    subcluster_profile = (
        low_risk_df.groupby("subcluster")
        .agg(
            size=("subcluster", "size"),
            avg_risk=("risk_score", "mean"),
            avg_crash=("crash_rate", "mean"),
            avg_fire=("fire_rate", "mean"),
            avg_injured=("injured_rate", "mean"),
            avg_age=("avg_age", "mean"),
            avg_complaints=("total_complaints", "mean"),
            avg_miles=("median_miles", "mean"),
        )
        .sort_values("avg_risk", ascending=False)
    )

    subcluster_profile["subcluster_label"] = subcluster_profile.apply(
        lambda row: _label_subcluster(row, subcluster_profile), axis=1
    )

    subcluster_label_map = subcluster_profile["subcluster_label"].to_dict()

    vehicle_df.loc[low_risk_df.index, "subcluster"] = low_risk_df["subcluster"]
    vehicle_df.loc[low_risk_df.index, "subcluster_label"] = low_risk_df["subcluster"].map(
        subcluster_label_map
    )

    high_risk_mask = vehicle_df["cluster_id"] != low_risk_cluster
    vehicle_df.loc[high_risk_mask, "subcluster"] = -1
    vehicle_df.loc[high_risk_mask, "subcluster_label"] = "Alto nivel de alerta"

    return vehicle_df


def _build_cluster_label_map(cluster_profile: pd.DataFrame) -> dict[int, str]:
    cluster_profile = cluster_profile.sort_values("avg_risk_score", ascending=True).reset_index(drop=True)
    ordered_ids = cluster_profile["cluster_id"].tolist()

    if len(ordered_ids) == 2:
        return {
            int(ordered_ids[0]): "Menor nivel de alerta",
            int(ordered_ids[1]): "Mayor nivel de alerta",
        }

    base_labels = [
        "Menor nivel de alerta",
        "Comportamiento estable",
        "Desgaste visible",
        "Alertas de seguridad",
        "Alertas eléctricas o térmicas",
        "Riesgo elevado",
        "Riesgo alto",
        "Riesgo muy alto",
    ]
    label_map: dict[int, str] = {}
    for idx, cluster_id in enumerate(ordered_ids):
        label_map[int(cluster_id)] = base_labels[min(idx, len(base_labels) - 1)]
    return label_map


def _build_cluster_description_map(cluster_profile: pd.DataFrame) -> dict[int, str]:
    return {
        int(row["cluster_id"]): _cluster_description(row)
        for _, row in cluster_profile.iterrows()
    }


def _cluster_description(row: pd.Series) -> str:
    fire = float(row.get("avg_fire_rate", 0.0))
    crash = float(row.get("avg_crash_rate", 0.0))
    injured = float(row.get("avg_injured_rate", 0.0))
    risk = float(row.get("avg_risk_score", 0.0))
    age = float(row.get("avg_age", 0.0))
    complaints = float(row.get("avg_complaints", 0.0))

    if risk >= 0.25 or crash >= 0.25 or injured >= 0.18:
        return "Historial con señales de alerta importantes."
    if fire >= max(crash, injured) and fire >= 0.05:
        return "Predominan reportes relacionados con sistema eléctrico o temperatura."
    if crash >= 0.10 or injured >= 0.08:
        return "Predominan reportes relacionados con seguridad, golpes o lesiones."
    if age >= 6 and complaints >= 100:
        return "Predomina desgaste acumulado por uso y antigüedad."
    return "Comportamiento relativamente estable frente al resto de la base."


def _label_subcluster(row: pd.Series, profile: pd.DataFrame) -> str:
    complaints_median = float(profile["avg_complaints"].median())
    age_median = float(profile["avg_age"].median())

    if float(row["avg_crash"]) > 0.15 or float(row["avg_injured"]) > 0.10:
        return "Alertas de seguridad"
    if float(row["avg_fire"]) > 0.05:
        return "Alertas eléctricas o de combustible"
    if float(row["avg_complaints"]) > complaints_median:
        return "Muchos reportes acumulados"
    if float(row["avg_age"]) > age_median:
        return "Mayor desgaste por antigüedad"
    return "Base confiable"