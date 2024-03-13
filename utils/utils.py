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


def compute_travel_time(data_, dist_mtx_c2c_, dist_mtx_i2c_):
    _data = data_.copy()
    dist = []
    time = []
    for _, row in _data.iterrows():
        dist.append(
            dist_mtx_i2c_[
                dist_mtx_i2c_["ID Intervenant"] == row["ID Intervenant"],
                row["ID Client"][0],
            ]
        )
        for i_client in range(len(row["ID Intervenant"]) - 1):
            try:
                dist.append(
                    dist_mtx_i2c_[
                        row["ID Client"][i_client], row["ID Client"][i_client + 1]
                    ]
                )
            except:
                dist.append(
                    dist_mtx_i2c_[
                        row["ID Client"][i_client + 1], row["ID Client"][i_client]
                    ]
                )

        dist.append(dist_mtx_i2c_[row["ID Client"][-1], row["ID Intervenant"]])
    
def parse_duration(duration_str):
    total_minutes = 0
    
    if 'hour' in duration_str:
        hours_part = duration_str.split('hour')[0].strip()
        total_minutes += int(hours_part) * 60  # Convertir les heures en minutes
        duration_str = duration_str.split('hour')[1].strip()  # Prendre la partie restante après 'hour'
    
    if 'min' in duration_str:
        mins_part = duration_str.split('min')[0].strip()
        total_minutes += int(mins_part)
    
    return total_minutes


def determine_time_window(prestation):
    if prestation == 'PDJ':
        return (7, 9)
    elif prestation == 'DEJ':
        return (12, 14)
    elif prestation == 'DIN':
        return (19, 21)
    elif prestation == 'TOILETTE_MAT':
        return (7, 10)
    elif prestation == 'TOILETTE_SOIR':
        return (18, 20)
    else:
        return (7,22)
    
def prestation_duration(h1,h2):
    datetime1 = datetime.combine(datetime.today(), h1)
    datetime2 = datetime.combine(datetime.today(), h2)

    time_diff = datetime2-datetime1

    return(time_diff.total_seconds()//60)

def convert_duration(nb_secondes):
    return nb_secondes/60

def define_client_time_matrix(
    df, vehicle_type, client_id_to_idx, out_value=4000
):
    list_clients_1 = df["ID Client 1"].unique()
    list_clients_2 = df["ID Client 2"].unique()
    full_list = list(set([*list_clients_1, *list_clients_2]))

    matrix = [[0 for i in range(len(full_list))] for k in range(len(full_list))]
    col_type = "duration_" + vehicle_type
    for client_1_id in full_list:
        client_1_idx = client_id_to_idx[client_1_id]
        for client_2_id in full_list:
            client_2_idx = client_id_to_idx[client_2_id]
            if client_1_id != client_2_id:
                try:
                    travel_time = convert_duration(
                        df.loc[
                            (df["ID Client 1"] == client_1_id)
                            & (df["ID Client 2"] == client_2_id),
                            col_type,
                        ].values[0]
                    )
                    matrix[client_1_idx][client_2_idx] = int(travel_time)
                except IndexError:
                    try:
                        travel_time = convert_duration(
                            df.loc[
                                (df["ID Client 1"] == client_2_id)
                                & (df["ID Client 2"] == client_1_id),
                                col_type,
                            ].values[0]
                        )
                        matrix[client_1_idx][client_2_idx] = int(travel_time)
                    except IndexError:
                        matrix[client_1_idx][client_2_idx] = out_value

    return np.array(matrix)

def define_inter_time_matrix(
    df,
    full_client_list,
    inter_id_to_idx,
    client_id_to_idx,
    vehicle_type="Car",
    out_value=4000,
):
    
    list_inter_columns = df['ID Intervenant']
    full_inter_list = list_inter_columns.unique()

    matrix = [
        [0 for i in range(len(full_client_list))] for k in range(len(full_inter_list))
    ]

    for inter_id in full_inter_list:
        inter_idx = inter_id_to_idx[inter_id]

        for client_id in full_client_list:
            client_idx = client_id_to_idx[client_id]
            try:
                travel_time = convert_duration(
                    df.loc[(df["ID Client"] == client_id) & (df["ID Intervenant"] == inter_id), 'duration'].values[0]
                )

                matrix[inter_idx][client_idx] = int(travel_time)
            except IndexError:
                matrix[inter_idx][client_idx] = out_value

    return np.array(matrix)

def create_full_time_matrix(client_time_matrix, inter_time_matrix, out_value=4000):
    nb_client = client_time_matrix.shape[0]
    nb_inter = inter_time_matrix.shape[0]

    matrix = [
        [out_value for i in range(nb_client + nb_inter)]
        for k in range(nb_client + nb_inter)
    ]

    for i in range(nb_client):
        for j in range(nb_client):
            matrix[i][j] = client_time_matrix[i][j]

    for i in range(nb_inter):
        for j in range(nb_client):
            matrix[i + nb_client][j] = inter_time_matrix[i][j]
            matrix[j][i + nb_client] = matrix[i + nb_client][j]

    return np.array(matrix)