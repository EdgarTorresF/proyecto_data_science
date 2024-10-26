
from datetime import date, datetime, timedelta
import os
import math

import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn

from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("Modelo de Deteccion de Fraudes con Redes Neuronales en Streamlit")

transactions_df = pd.read_feather("combined_file.feather")

st.header("Muestra aleatoria de algunos elementos de la base de datos")
st.write(transactions_df.sample(10, random_state=0))

# st.subheader("Tipo de datos en nuestra base de datos ")
# st.text(transactions_df.info())

# Descripcion de la informacion de nuestra base de datos
st.subheader("Informacion basica de la base de datos")
st.write(transactions_df.describe())

# Revisamos valores nulos
st.subheader("Revision de valores nulos")
st.write(transactions_df.isna().sum())

# Revisamos valores duplicados
st.subheader("Revision de valores duplicados")
st.text(transactions_df.duplicated().sum())

st.divider()

# Revisamos cantidad de transacciones fraudulentas vs no fraudulentas y totales
st.subheader("Revision de transacciones fraudulentas, no fraudulentas y totales")
not_fraud_count, fraud_count = np.bincount(transactions_df["TX_FRAUD"])

total_count = not_fraud_count + fraud_count
st.text(
    (
        f"Data:\n"
        f"    Total: {total_count}\n"
        f"    No Fraudulentas: {not_fraud_count} ({100 * not_fraud_count / total_count:.2f}% of total)\n"
        f"    Fraudulentas: {fraud_count} ({100 * fraud_count / total_count:.2f}% del total)\n"
    )
)


df = pd.concat(
    [
        transactions_df[transactions_df["TX_FRAUD"] == 0].sample(1000, random_state=0),
        transactions_df[transactions_df["TX_FRAUD"] == 1].sample(1000, random_state=0),
    ]
)

st.divider()

# Mostramos un grafico del numero de transacciones por monto
st.subheader("Numero de transacciones por Monto")
fig = px.histogram(
    df,
    x="TX_AMOUNT",
    color="TX_FRAUD",
    marginal="box",
)
fig.update_traces(opacity=0.75)
fig.update_layout(barmode="overlay")
fig

# En esta seccion se crean algunas variables adicionales para mejorar el performance del modelo
cleaned_df = pd.DataFrame()

cleaned_df["amount"] = transactions_df["TX_AMOUNT"]
cleaned_df["is_fraud"] = transactions_df["TX_FRAUD"]
cleaned_df["is_weekend"] = transactions_df["TX_DATETIME"].dt.weekday >= 5
cleaned_df["is_night"] = transactions_df["TX_DATETIME"].dt.hour <= 6

# Numero de transacciones por usuario en los ultimos 1/7/30 dias
cleaned_df["customer_num_transactions_1_day"] = transactions_df.groupby(
    "CUSTOMER_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("1d", on="TX_DATETIME").count()
, include_groups=False)["TX_AMOUNT"]

cleaned_df["customer_num_transactions_7_day"] = transactions_df.groupby(
    "CUSTOMER_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("7d", on="TX_DATETIME").count()
, include_groups=False)["TX_AMOUNT"]

cleaned_df["customer_num_transactions_30_day"] = transactions_df.groupby(
    "CUSTOMER_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("30d", on="TX_DATETIME").count()
, include_groups=False)["TX_AMOUNT"]

# El valor promedio de las transacciones en los ultimos 1/7/30 dias
cleaned_df["customer_avg_amount_1_day"] = transactions_df.groupby(
    "CUSTOMER_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("1d", on="TX_DATETIME").mean()
, include_groups=False)["TX_AMOUNT"]

cleaned_df["customer_avg_amount_7_day"] = transactions_df.groupby(
    "CUSTOMER_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("7d", on="TX_DATETIME").mean()
, include_groups=False)["TX_AMOUNT"]

cleaned_df["customer_avg_amount_30_day"] = transactions_df.groupby(
    "CUSTOMER_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("30d", on="TX_DATETIME").mean()
, include_groups=False)["TX_AMOUNT"]

# Se asume que despues de 7 dias podemos confirmar si las transacciones fueron fraudulentas o no
# por lo que se crea una funcion que calcula las transacciones fraudulentas en un periodo N-7
DAY_DELAY = 7


@st.cache_data
def get_count_risk_rolling_window(
    terminal_transactions, window_size, delay_period=DAY_DELAY
):
    terminal_transactions = terminal_transactions.sort_values("TX_DATETIME")

    frauds_in_delay = terminal_transactions.rolling(
        str(delay_period) + "d", on="TX_DATETIME"
    )["TX_FRAUD"].sum()
    transactions_in_delay = terminal_transactions.rolling(
        str(delay_period) + "d", on="TX_DATETIME"
    )["TX_FRAUD"].count()

    frauds_until_window = terminal_transactions.rolling(
        str(delay_period + window_size) + "d", on="TX_DATETIME"
    )["TX_FRAUD"].sum()
    transactions_until_window = terminal_transactions.rolling(
        str(delay_period + window_size) + "d", on="TX_DATETIME"
    )["TX_FRAUD"].count()

    frauds_in_window = frauds_until_window - frauds_in_delay
    transactions_in_window = transactions_until_window - transactions_in_delay

    terminal_transactions["fraud_risk"] = (
        frauds_in_window / transactions_in_window
    ).fillna(0)

    return terminal_transactions


# Se crean variables que cuentan el numero de transacciones en la terminal en los ultimos 1/7/30 dias
cleaned_df["terminal_num_transactions_1_day"] = transactions_df.groupby(
    "TERMINAL_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("1d", on="TX_DATETIME").count()
, include_groups=False)["TX_AMOUNT"]

cleaned_df["terminal_num_transactions_7_day"] = transactions_df.groupby(
    "TERMINAL_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("7d", on="TX_DATETIME").count()
, include_groups=False)["TX_AMOUNT"]

cleaned_df["terminal_num_transactions_30_day"] = transactions_df.groupby(
    "TERMINAL_ID", group_keys=False
).apply(
    lambda x: x.sort_values("TX_DATETIME")[["TX_DATETIME", "TX_AMOUNT"]]
    .rolling("30d", on="TX_DATETIME").count()
, include_groups=False)["TX_AMOUNT"]

# Se crean variables que calculan el riesgo de fraude en la terminal en los ultimos 1/7/30 dias
cleaned_df["terminal_fraud_risk_1_day"] = transactions_df.groupby("TERMINAL_ID", group_keys=False).apply(
    lambda x: get_count_risk_rolling_window(x.sort_values("TX_DATETIME"), 1, 7)
, include_groups=False)["fraud_risk"]

cleaned_df["terminal_fraud_risk_7_day"] = transactions_df.groupby("TERMINAL_ID", group_keys=False).apply(
    lambda x: get_count_risk_rolling_window(x.sort_values("TX_DATETIME"), 7, 7)
, include_groups=False)["fraud_risk"]

cleaned_df["terminal_fraud_risk_30_day"] = transactions_df.groupby("TERMINAL_ID", group_keys=False).apply(
    lambda x: get_count_risk_rolling_window(x.sort_values("TX_DATETIME"), 30, 7)
, include_groups=False)["fraud_risk"]

# Estas variables nos sirven para la division de los datos
cleaned_df["day"] = transactions_df["TX_TIME_DAYS"]
cleaned_df["datetime"] = transactions_df["TX_DATETIME"]
cleaned_df["customer_id"] = transactions_df["CUSTOMER_ID"]
cleaned_df["id"] = transactions_df["TRANSACTION_ID"]

st.divider()

# Se muestran algunos ejemplos de transacciones fraudulentas y no fraudulentas
st.subheader("Muestreo de algunas transacciones fraudulentas y no fraudulentas")
sample_df = pd.concat(
    [
        cleaned_df[cleaned_df["is_fraud"] == 1].sample(5, random_state=0),
        cleaned_df[cleaned_df["is_fraud"] == 0].sample(5, random_state=0),
    ]
).sample(10, random_state=0)

st.table(sample_df)

# Se divide los datos para entrenamiento, validacion y prueba
@st.cache_data
def get_train_test_set(
    df,
    start_date_training,
    delta_train=7,
    delta_delay=DAY_DELAY,
    delta_test=7,
    random_state=0,
):

    # Se define la data para entrenamiento
    train_df = df[
        (df["datetime"] >= start_date_training)
        & (df["datetime"] < start_date_training + timedelta(days=delta_train))
    ]

    # Se define la data para prueba
    test_df = []

    # Nota: Se eliminan de la prueba las tarjetas que se sabe que estan comprometidas, es decir,
    # por cada dia de prueba, todas las transacciones fraudulentas el tiempo de gracia son removidos
   
    # Se obtienen los usuarios fraudulentos del set de entrenamiento
    known_defrauded_customers = set(train_df[train_df["is_fraud"] == 1]["customer_id"])

    # Se obtiene el dia de inicio relativo del set de entrenamiento
    start_tx_time_days_training = train_df["day"].min()

    # Por cada dia del set de prueba
    for day in range(delta_test):

        # Se obtiene la data de prueba para ese dia 
        test_df_day = df[
            df["day"] == start_tx_time_days_training + delta_train + delta_delay + day
        ]

        # Las tarjetas fraudulentas de ese dia de prueba, menos el periodo de gracia, se agregan all pool de usuarios defraudados
        test_df_day_delay_period = df[
            df["day"] == start_tx_time_days_training + delta_train + day - 1
        ]

        new_defrauded_customers = set(
            test_df_day_delay_period[test_df_day_delay_period["is_fraud"] == 1][
                "customer_id"
            ]
        )
        known_defrauded_customers = known_defrauded_customers.union(
            new_defrauded_customers
        )

        test_df_day = test_df_day[
            ~test_df_day["customer_id"].isin(known_defrauded_customers)
        ]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # Se ordenan los datasets en orden ascendente de acuerdo al Transaction ID
    train_df = train_df.sort_values("id")
    test_df = test_df.sort_values("id")

    return (train_df, test_df)


train_df, test_df = get_train_test_set(
    cleaned_df, datetime(2018, 7, 25), delta_train=21
)
train_df, val_df = get_train_test_set(train_df, datetime(2018, 7, 25))

# Para cada uno de los sets, se generan arrays de variables (las variables que queremos usar para el entrenamientos)
# y etiquetas (lo que se quiere predecir)

label_columns = ["is_fraud"]
feature_columns = [
    "amount",
    "is_weekend",
    "is_night",
    "customer_num_transactions_1_day",
    "customer_num_transactions_7_day",
    "customer_num_transactions_30_day",
    "customer_avg_amount_1_day",
    "customer_avg_amount_7_day",
    "customer_avg_amount_30_day",
    "terminal_num_transactions_1_day",
    "terminal_num_transactions_7_day",
    "terminal_num_transactions_30_day",
    "terminal_fraud_risk_1_day",
    "terminal_fraud_risk_7_day",
    "terminal_fraud_risk_30_day",
]

train_labels = np.array(train_df[label_columns])
val_labels = np.array(val_df[label_columns])
test_labels = np.array(test_df[label_columns])

train_features = np.array(train_df[feature_columns])
val_features = np.array(val_df[feature_columns])
test_features = np.array(test_df[feature_columns])

# Nos aseguramos de que todos los valores tengan una escala similar

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

st.divider()
st.subheader("Tamaño de los Datos de Prueba y Entrenamiento")
st.text(
    (
        f"Training labels shape: {train_labels.shape}\n"
        f"Validation labels shape: {val_labels.shape}\n"
        f"Test labels shape: {test_labels.shape}\n"

        f"Training features shape: {train_features.shape}\n"
        f"Validation features shape: {val_features.shape}\n"
        f"Test features shape: {test_features.shape}\n"
    )
)

st.divider()

# Debido a que la mayor parte de nuestros datos son transacciones no fraudulentas, tenemos que hacer un
# balanceo de los datos para que el modelo le de una mayor importancia a las transacciones fraudulentas
st.subheader("Balanceo de Datos")
weight_for_not_fraud = (1.0 / not_fraud_count) * total_count / 2.0
weight_for_fraud = (1.0 / fraud_count) * total_count / 2.0

class_weight = {"No Fraudulentas": weight_for_not_fraud, "Fraudulentas": weight_for_fraud}

class_weight

st.divider()

# Se genera un modelo con 2 capas ocultas usando 500 nodos cada uno
# Se usa una capa dropout para prevenir el overfit
# La funcion de perdida es: "binary cross entropy" la cual es estandar para problemas de clasificacion binaria

output_bias = tf.keras.initializers.Constant(np.log([fraud_count / not_fraud_count]))

model = keras.Sequential(
    [
        keras.layers.Dense(
            500, activation="relu", input_shape=(train_features.shape[-1],)
        ),
        keras.layers.Dense(
            500, activation="relu", input_shape=(train_features.shape[-1],)
        ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="prc", curve="PR"),
    ],
)
model.summary()

# Se agrega un "early stop" para prevenir el overfitting
BATCH_SIZE = 64

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_prc", verbose=1, patience=10, mode="max", restore_best_weights=True
)

training_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=40,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels),
    class_weight=class_weight,
)

# Se generan graficos de las metricas que se van a evaluar
res = []

metrics_to_plot = [
    ("loss", "Loss"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("auc", "Area under ROC curve"),
    ("prc", "Area under PR curve"),
]
fig = make_subplots(rows=len(metrics_to_plot), cols=1)


for metric, name in metrics_to_plot:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=training_history.epoch,
                y=training_history.history[metric],
                mode="lines",
                name="Training",
            ),
            go.Scatter(
                x=training_history.epoch,
                y=training_history.history["val_" + metric],
                mode="lines",
                line={"dash": "dash"},
                name="Validation",
            ),
        ]
    )
    fig.update_yaxes(title=name)
    fig.update_xaxes(title="Epoch")

    if (metric, name) == metrics_to_plot[0]:
        fig.update_layout(
            height=250, title="Historial de Entrenamiento", margin={"b": 0, "t": 50}
        )
    else:
        fig.update_layout(height=200, margin={"b": 0, "t": 0})
    fig


# El modelo nos da un valor entre 0 y 1 que se puede leer como la probabilidad de fraude

train_predictions = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions = model.predict(test_features, batch_size=BATCH_SIZE)

predictions_df = pd.DataFrame(
    {"Prediction": train_predictions.ravel(), "Label": train_labels.ravel()}
)
predictions_df = pd.concat(
    [
        predictions_df[predictions_df["Label"] == 0].sample(5000, random_state=0),
        predictions_df[predictions_df["Label"] == 1].sample(500, random_state=0),
    ]
)
fig = px.histogram(
    predictions_df,
    x="Prediction",
    title="Predicción",
    color="Label",
    marginal="box",
    labels={"0": "Legitimate", "1": "Fraudulent"},
)
fig.update_traces(opacity=0.75)
fig.update_layout(barmode="overlay")
fig

# Generar una curva ROC nos ayuda a decidir un balance entre posibles Falsos Positivos y Falsos Negativos
@st.cache_data
def make_roc_df(name, predictions, labels):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    return pd.DataFrame({"fp": fp * 100, "tp": tp * 100, "Dataset": name})


roc_df = pd.concat(
    [
        make_roc_df("Training", train_predictions, train_labels),
        make_roc_df("Test", test_predictions, test_labels),
    ]
)

fig = px.line(
    roc_df,
    title="ROC Curve",
    x="fp",
    y="tp",
    color="Dataset",
    labels={"fp": "False Positives (%)", "tp": "True Positives (%)"},
)
fig.update_yaxes(range=[60, 100])
fig.update_traces(line={"dash": "dash"}, selector={"name": "test"})
fig