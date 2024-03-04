import pandas as pd
from datetime import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def load_data():
    path = "../data/"

    data_file = "ChallengeXHEC.xlsx"

    df_JAN24 = pd.read_excel(path + data_file, sheet_name=0)
    df_clients = pd.read_excel(path + data_file, sheet_name=1)
    df_intervenants = pd.read_excel(path + data_file, sheet_name=2)
    df_dispo = pd.read_excel(path + data_file, sheet_name=3)

    return {
        "JAN24": df_JAN24,
        "clients": df_clients,
        "intervenants": df_intervenants,
        "dispo": df_dispo,
    }


def transform_data(data_):

    data = data_.copy()

    # JAN24 DataFrame

    data["JAN24"]["Heure de début"] = data["JAN24"][["Date", "Heure de début"]].apply(
        lambda x: datetime.combine(x["Date"], x["Heure de début"]), axis=1
    )
    data["JAN24"]["Heure de fin"] = data["JAN24"][["Date", "Heure de fin"]].apply(
        lambda x: datetime.combine(x["Date"], x["Heure de fin"]), axis=1
    )
    data["JAN24"]["ID Client"] = data["JAN24"]["ID Client"].astype("str")
    data["JAN24"]["ID Intervenant"] = data["JAN24"]["ID Intervenant"].astype("str")
    data["JAN24"]["Prestation"] = data["JAN24"]["Prestation"].astype("category")

    # Client DataFrame

    data["clients"]["ID Client"] = data["clients"]["ID Client"].astype("str")

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(data["clients"]["Longitude"], data["clients"]["Latitude"])
    ]
    data["clients"] = gpd.GeoDataFrame(
        data["clients"], geometry=geometry, crs="EPSG:4326"
    )

    # data["clients"] = data["clients"].drop(columns=["Latitude", "Longitude"])

    # Intervenant DataFrame

    data["intervenants"]["ID Intervenant"] = data["intervenants"][
        "ID Intervenant"
    ].astype("str")

    data["intervenants"]["Permis"] = (
        data["intervenants"]["Permis"].replace({"Oui": 1, "Non": 0}).fillna(1)
    )

    data["intervenants"]["Véhicule personnel"] = (
        data["intervenants"]["Véhicule personnel"]
        .replace({"Oui": 1, "Non": 0})
        .fillna(1)
    )

    competency_list = np.unique(
        sum(data["intervenants"]["Compétences"].str.split(", ").to_numpy(), [])
    )

    df_competency = (
        data["intervenants"]["Compétences"]
        .str.split(", ")
        .apply(lambda x: np.isin(competency_list, x))
        .apply(pd.Series)
        .rename(
            {k: v for k, v in zip(np.arange(len(competency_list)), competency_list)},
            axis="columns",
        )
    )

    data["intervenants"] = pd.concat(
        [data["intervenants"], df_competency], axis=1
    ).drop(columns="Compétences")

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(
            data["intervenants"]["Longitude"], data["intervenants"]["Latitude"]
        )
    ]
    data["intervenants"] = gpd.GeoDataFrame(
        data["intervenants"], geometry=geometry, crs="EPSG:4326"
    )

    # data["intervenants"] = data["intervenants"].drop(columns=["Latitude", "Longitude"])

    # Dispo DataFrame

    data["dispo"]["ID Intervenant"] = data["dispo"]["ID Intervenant"].astype("str")

    return data


def merge_data(data_):

    data = data_.copy()

    df_merge = (
        pd.merge(
            data["JAN24"],
            data["clients"].drop(columns=["Longitude", "Latitude"]),
            on="ID Client",
        )
        .merge(
            data["intervenants"].drop(columns=["Longitude", "Latitude"]),
            on="ID Intervenant",
            suffixes=("_client", "_intervenant"),
        )
        .merge(data["dispo"], on="ID Intervenant")
    ).drop(columns=("Dispo / Indispo"))

    return df_merge


def get_mean_time_presta_client(data_):

    df_merge = data_.copy()

    df_merge["mean_time_per_client_prestation"] = (
        df_merge.assign(
            time_per_client_prestation=lambda x: (
                x["Heure de fin"] - x["Heure de début"]
            )
            .dt.total_seconds()
            .mul(1 / 60)
        )
        .groupby(["ID Client", "Prestation"])
        .time_per_client_prestation.transform("mean")
    )

    return df_merge
