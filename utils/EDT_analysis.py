import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import altair as alt
import datetime

parent_directory = os.path.dirname(os.getcwd())
sys.path.append(parent_directory)

from utils.utils import (
    load_data,
    transform_data,
    merge_data,
    get_mean_time_presta_client,
)


def load_dist_matrix():
    path = "../data/"

    dist_mtx_c2c = pd.read_csv(path + "/distance_matrix_client_to_client.csv")
    dist_mtx_c2c_ = dist_mtx_c2c.copy()
    dist_mtx_c2c_["ID Client 1"], dist_mtx_c2c_["ID Client 2"] = (
        dist_mtx_c2c_["ID Client 2"],
        dist_mtx_c2c_["ID Client 1"],
    )
    dist_mtx_c2c = pd.concat([dist_mtx_c2c, dist_mtx_c2c_], axis=0)
    dist_mtx_c2c["ID Client 1"] = dist_mtx_c2c["ID Client 1"].astype(str)
    dist_mtx_c2c["ID Client 2"] = dist_mtx_c2c["ID Client 2"].astype(str)

    dist_mtx_i2c = pd.read_csv(path + "/distance_matrix_inter_to_client.csv")
    dist_mtx_i2c["ID Intervenant"] = dist_mtx_i2c["ID Intervenant"].astype(str)
    dist_mtx_i2c["ID Client"] = dist_mtx_i2c["ID Client"].astype(str)

    return dist_mtx_c2c, dist_mtx_i2c


def get_nbr_worked_day(df_merge):

    drop_prestations = [
        "ADMINISTRATION",
        "FORMATION",
        "COORDINATION",
        "HOMMES TOUTES MAINS",
        "VISITE MEDICALE",
    ]

    columns_to_keep = [
        col for col in df_merge.columns if type(col) == datetime.datetime
    ]

    nbr_day_worked = (
        df_merge.drop_duplicates(subset="ID Intervenant")[columns_to_keep].sum().sum()
    )

    return nbr_day_worked


def compute_travel_time(df_merge, dist_mtx_c2c_, dist_mtx_i2c_):

    df_merge = df_merge.copy()

    df_client_list = (
        df_merge.sort_values(by="Heure de début")
        .groupby(by=["ID Intervenant", "Date"])["ID Client"]
        .apply(list)
        .reset_index()
    )

    for _, row in tqdm(df_client_list.iterrows(), total=len(df_client_list)):

        id_intervenant = row["ID Intervenant"]

        client_list = row["ID Client"]

        time = []

        have_licence = df_merge[
            df_merge["ID Intervenant"] == id_intervenant
        ].Permis.to_numpy()[0]

        if have_licence == 1:
            duration_c2c = "duration_car"
        else:
            duration_c2c = "duration_bike"

        time.append(
            dist_mtx_i2c_[
                (dist_mtx_i2c_["ID Intervenant"] == id_intervenant)
                & (dist_mtx_i2c_["ID Client"] == client_list[0])
            ]["duration"].to_numpy()
        )

        for i_client in range(len(client_list) - 1):

            time.append(
                dist_mtx_c2c_[
                    (dist_mtx_c2c_["ID Client 1"] == client_list[i_client])
                    & (dist_mtx_c2c_["ID Client 2"] == client_list[i_client + 1])
                ][duration_c2c].to_numpy()
            )

        time.append(
            dist_mtx_i2c_[
                (dist_mtx_i2c_["ID Intervenant"] == id_intervenant)
                & (dist_mtx_i2c_["ID Client"] == client_list[-1])
            ]["duration"].to_numpy()
        )
        df_client_list.loc[_, "duration_tot"] = np.sum(np.concatenate(time))

    return df_client_list


def compute_travel_distance(df_merge, dist_mtx_c2c_, dist_mtx_i2c_):
    _data = df_merge.copy()

    df_client_list = (
        _data.sort_values(by="Heure de début")
        .groupby(by=["ID Intervenant", "Date"])["ID Client"]
        .apply(list)
        .reset_index()
    )

    for _, row in tqdm(df_client_list.iterrows(), total=len(df_client_list)):

        id_intervenant = row["ID Intervenant"]

        client_list = row["ID Client"]

        distance = []

        have_licence = _data[
            _data["ID Intervenant"] == id_intervenant
        ].Permis.to_numpy()[0]

        if have_licence == 1:
            distance_c2c = "distance_car"
        else:
            distance_c2c = "distance_bike"

        distance.append(
            dist_mtx_i2c_[
                (dist_mtx_i2c_["ID Intervenant"] == id_intervenant)
                & (dist_mtx_i2c_["ID Client"] == client_list[0])
            ]["distance"].to_numpy()
        )

        for i_client in range(len(client_list) - 1):

            distance.append(
                dist_mtx_c2c_[
                    (dist_mtx_c2c_["ID Client 1"] == client_list[i_client])
                    & (dist_mtx_c2c_["ID Client 2"] == client_list[i_client + 1])
                ][distance_c2c].to_numpy()
            )

        distance.append(
            dist_mtx_i2c_[
                (dist_mtx_i2c_["ID Intervenant"] == id_intervenant)
                & (dist_mtx_i2c_["ID Client"] == client_list[-1])
            ]["distance"].to_numpy()
        )
        df_client_list.loc[_, "distance_tot"] = np.sum(np.concatenate(distance))

    return df_client_list


def get_dist_time_df(df_merge, dist_mtx_c2c_, dist_mtx_i2c_):
    df_test = compute_travel_time(df_merge, dist_mtx_c2c_, dist_mtx_i2c_)
    df_test["distance_tot"] = compute_travel_distance(
        df_merge, dist_mtx_c2c_, dist_mtx_i2c_
    )["distance_tot"]
    return df_test


def dist_bike_car(df_merge, dist_mtx_c2c_, dist_mtx_i2c_):

    df_test = get_dist_time_df(df_merge, dist_mtx_c2c_, dist_mtx_i2c_)

    source = (
        df_test.assign(
            distance_tot=lambda x: x["distance_tot"] / 100,
            duration_tot=lambda x: x["duration_tot"] / 60,
        )
        .merge(
            df_merge[["ID Intervenant", "Permis"]].drop_duplicates(),
            on="ID Intervenant",
        )
        .groupby("Permis")
        .distance_tot.sum()
        .to_frame()
        .reset_index()
        .replace({0: "Bike", 1: "Car"})
    )

    source["percentage"] = (
        100 * source.distance_tot / source.distance_tot.sum()
    ).round(2)

    donut = (
        alt.Chart(source)
        .mark_arc(innerRadius=100)
        .encode(
            theta=alt.Theta(field="distance_tot", type="quantitative"),
            color=alt.Color(field="Permis", type="nominal"),
            text=alt.Text(field="percentage", format=".2f"),
        )
    )

    donut

    return (source, donut)


def nbr_inter_for_TOILETTE(df_merge):
    fig, ax = plt.subplots(figsize=(7, 4))

    data = (
        df_merge.query('Prestation == "TOILETTE"')
        .groupby("ID Client")["ID Intervenant"]
        .nunique()
    )

    data_count = data.value_counts()

    # Plotting the bars
    bars = ax.bar(
        data_count.sort_index().index,
        data_count.sort_index(),
        color="orange",
        width=0.8,
        edgecolor="black",
    )

    # Centering xticks on the bars
    xticks_positions = np.arange(1, len(data_count) + 1)
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(data_count.sort_index().index, ha="center")

    # Adding labels and title
    ax.set_xlabel("Number of differents homecare agents per client")
    ax.set_ylabel("Frequency")
    ax.set_title("Number of differents homecare agents per client for TOILETTE task")

    plt.show()

    return (
        df_merge.query('Prestation == "TOILETTE"')
        .groupby("ID Client")["ID Intervenant"]
        .nunique()
        .agg(["mean", "std"])
    )


def time_between_client(row, distance_matrix_):
    if row["Permis"] == 1:
        duration = "duration_car"
    else:
        duration = "duration_bike"

    time = distance_matrix_[
        (distance_matrix_["ID Client 1"] == row["ID Client"])
        & (distance_matrix_["ID Client 2"] == row["Next_client_ID"])
    ][duration].to_numpy()

    if len(time) == 1:
        time = round(time[0] / 60, 1)
    else:
        time = 0

    return time


def downtime_study(df_merge, distance_matrix_, drop_prestations):

    nbr_day_worked = get_nbr_worked_day(df_merge)

    df_downtime = (
        df_merge.sort_values("Heure de début")
        .set_index(["ID Intervenant", "Date"])
        .sort_index(level=1)[
            ["Permis", "ID Client", "Heure de début", "Heure de fin", "Prestation"]
        ]
    )

    df_downtime["Next_hour_start"] = df_downtime.groupby(
        level=["ID Intervenant", "Date"]
    )["Heure de début"].shift(-1)

    df_downtime["Next_client_ID"] = df_downtime.groupby(
        level=["ID Intervenant", "Date"]
    )["ID Client"].shift(-1)

    df_downtime["Downtime"] = (
        df_downtime["Next_hour_start"] - df_downtime["Heure de fin"]
    ).dt.total_seconds() / 60

    df_downtime["nbr_missions_day"] = df_downtime.groupby(
        level=["ID Intervenant", "Date"]
    )["Heure de début"].count()

    df_downtime["travel_time"] = df_downtime.apply(
        time_between_client, distance_matrix_=distance_matrix_, axis=1
    )

    df_downtime = df_downtime[~df_downtime.Prestation.isin(drop_prestations)]

    df_downtime["real_downtime"] = df_downtime.Downtime - df_downtime.travel_time

    print(
        f"Percentage of tasks followed by a downtime : {df_downtime.real_downtime.gt(30).mean() : .2%}"
    )

    print(
        f'Total time (in hours) lost in downtime on January : {df_downtime.real_downtime.to_frame().query("real_downtime > 30").sum().values[0]/60 :.2f}'
    )

    print(
        f'Total time (in hours) lost in downtime on January per day worked: {df_downtime.real_downtime.to_frame().query("real_downtime > 30").sum().values[0]/60/nbr_day_worked :.2f}'
    )

    return df_downtime[df_downtime.real_downtime.gt(30)].real_downtime.plot.hist(
        rwidth=0.9, bins=50
    )


import matplotlib.pyplot as plt


def scarcity_of_a_skill(df_merge, drop_prestations):
    data = load_data()
    data = transform_data(data)

    df_time = (
        df_merge[["Prestation", "Heure de début", "Heure de fin"]]
        .assign(
            time_delta=lambda x: (x["Heure de fin"] - x["Heure de début"])
            .dt.total_seconds()
            .div(60 * 60)
        )
        .groupby("Prestation")
        .time_delta.sum()
    )

    cleaned_column_names = [
        x.strip().replace("\n", "") for x in df_merge.Prestation.unique()
    ]

    # Filter the column names in data['JAN24'] using the cleaned column names
    filtered_columns = [
        col for col in cleaned_column_names if col in data["intervenants"].columns
    ]

    # Select the columns from data['JAN24']
    selected_columns = data["intervenants"][filtered_columns]
    df_nbr_inter = selected_columns.sum(axis=0)

    # Remove corresponding indices from the DataFrame
    cleaned_df = (
        (df_time / df_nbr_inter)
        .drop(drop_prestations, errors="ignore")
        .dropna()
        .sort_values()
    )

    # Plotting the cleaned DataFrame
    ax = cleaned_df.plot.bar()

    # Add title and other customizations
    ax.set_title("Scarcity of a Skill")
    ax.set_xlabel("Prestation")
    ax.set_ylabel(
        "Number of needed hours per intervenant"
    )  # Update with appropriate unit
    ax.legend(["Total Time"])  # Update legend if necessary

    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    plt.show()


def intensity_metrics(df_merge, hours_per_day=8):

    df_time = df_merge[["Prestation", "Heure de début", "Heure de fin"]].assign(
        time_delta=lambda x: (
            x["Heure de fin"] - x["Heure de début"]
        ).dt.total_seconds()
    )
    tot_worked_time = df_time.time_delta.sum() / 60 / 60 / hours_per_day

    nbr_work_day = get_nbr_worked_day(df_merge)

    return tot_worked_time / nbr_work_day


def get_travel_time(df_merge, distance_matrix_, drop_prestations):

    nbr_day_worked = get_nbr_worked_day(df_merge)

    df_downtime = (
        df_merge.sort_values("Heure de début")
        .set_index(["ID Intervenant", "Date"])
        .sort_index(level=1)[
            ["Permis", "ID Client", "Heure de début", "Heure de fin", "Prestation"]
        ]
    )

    df_downtime["Next_hour_start"] = df_downtime.groupby(
        level=["ID Intervenant", "Date"]
    )["Heure de début"].shift(-1)

    df_downtime["Next_client_ID"] = df_downtime.groupby(
        level=["ID Intervenant", "Date"]
    )["ID Client"].shift(-1)

    df_downtime["Downtime"] = (
        df_downtime["Next_hour_start"] - df_downtime["Heure de fin"]
    ).dt.total_seconds() / 60

    df_downtime["nbr_missions_day"] = df_downtime.groupby(
        level=["ID Intervenant", "Date"]
    )["Heure de début"].count()

    df_downtime["travel_time"] = df_downtime.apply(
        time_between_client, distance_matrix_=distance_matrix_, axis=1
    )

    df_downtime = df_downtime[~df_downtime.Prestation.isin(drop_prestations)]

    print(
        f"Time (in hours) spend in travel per workday: {df_downtime.travel_time.sum() / nbr_day_worked / 60:.2f}"
    )

    print(
        f"Time (in hours) spend in travel on January: {df_downtime.travel_time.sum() / 60:.2f}"
    )

    return
