import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit_deckgl as st_deckgl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime, warnings, scipy
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

# Read airports data from CSV file
preprocessed_flights_df = pd.read_csv("/Users/avnekrasov/Library/CloudStorage/OneDrive-Личная/(2) Личное/Личное развитие/202306 Data Alalyst в Сбере/Дипломная работа/Graduate_work/preprocessed_flights.csv")
preprocessed_flights_df.dropna(inplace = True)
airports_df = pd.read_csv("/Users/avnekrasov/Library/CloudStorage/OneDrive-Личная/(2) Личное/Личное развитие/202306 Data Alalyst в Сбере/Дипломная работа/Graduate_work/airports.csv")
airports_df = airports_df.dropna(subset=['LATITUDE', 'LONGITUDE'])

st.title("Дипломная работа ")
st.write ("Некрасов Андрей. Школа DA поток Февраль 2023")
st.title("Predicting flight delays")

# Настройка боковой панели
st.sidebar.title("Выбор моделей и параметров рассчета")
st.sidebar.info("Выберите модели для рассчета ")

# Create a checkbox for model selection
linear_regression_selected = st.sidebar.checkbox('Linear Regression', value=True)
random_forest_selected = st.sidebar.checkbox('Random Forest')
gradient_boosting_selected = st.sidebar.checkbox('Gradient Boosting')
default_t_size = 20
st.sidebar.info("Выберите пропорции разбивики Train и Test ")
t_size = st.sidebar.slider("Select test size (%)", min_value=0, max_value=100, value=default_t_size)
st.sidebar.write('test_size =' + str(t_size))
test_size = t_size / 100

# Create a PyDeck scatterplot layer for the airports
layer = pdk.Layer(
    "ScatterplotLayer",
    data=airports_df,
    get_position='[LONGITUDE, LATITUDE]',
    get_radius=1000,
    get_fill_color=[0, 0, 255],
    pickable=True,
)

# Set the initial view state for the map
view_state = pdk.ViewState(
    latitude = 40.65236,
    longitude = -75.4404,
    zoom=8,
    pitch=0,
)


# Create the PyDeck deck
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{AIRPORT} - {IATA_CODE}"},
)

# Display the map using Streamlit
st.markdown('Карта аэропортов')
st.pydeck_chart(deck)


# Function to handle airport selection
def on_airport_selected(selected_airport):
    origin_airport = selected_airport['IATA_CODE']
    df_origin = preprocessed_flights_df[preprocessed_flights_df['ORIGIN_AIRPORT'] == origin_airport]

    # Display the origin airport information
    st.write(f"Total Outgoing Flights: {len(df_origin)}")
    st.write(f"Unique Arrival Airports: {df_origin['DESTINATION_AIRPORT'].nunique()}")
    st.write(f"Airlines Flying from this Airport: {df_origin['AIRLINE'].nunique()}")


# Create a dropdown list of airports
options = airports_df['AIRPORT']
default_airport = options[165]
selected_airport = st.selectbox('Выбор аэропорта вылета', options)

# Check if an airport is selected
if selected_airport:
    # Retrieve the selected airport information
    selected_airport_info = airports_df[airports_df['AIRPORT'] == selected_airport].iloc[0]
    st.write(f"IATA Code: {selected_airport_info['IATA_CODE']}")
    st.write(f"City: {selected_airport_info['CITY']}")
    st.write(f"Latitude: {selected_airport_info['LATITUDE']}")
    st.write(f"Longitude: {selected_airport_info['LONGITUDE']}")

    # Handle airport selection event
    on_airport_selected(selected_airport_info)


# Create a "Confirm Selection" button
if st.button('Обучить модели для выбранного аэропорта'):
    # Set the origin_airport variable to the selected airport's IATA_CODE
    origin_airport = selected_airport_info['IATA_CODE']
    # Generate the df_origin dataframe
    df_origin = preprocessed_flights_df[preprocessed_flights_df['ORIGIN_AIRPORT'] == origin_airport]
    st.write (f"Аэропорт вылета : {selected_airport_info['IATA_CODE']}")

    df_encoded = df_origin[['DESTINATION_AIRPORT', 'AIRLINE']].copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['DESTINATION_AIRPORT', 'AIRLINE' ])
    df_encoded = pd.concat([df_origin, df_encoded], axis=1)
    features = df_encoded.columns.values.tolist()
    features_train_test = df_encoded.columns.values.tolist()
    features_to_remove = ['AIRLINE', 'ORIGIN_AIRPORT', 'ARRIVAL_DELAY']
    features_to_remove_1 = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_DELAY']
    for feature_id in features_to_remove:
    # Проходим по списку и удаляем.
        features.remove(feature_id)
    for feature_id in features_to_remove_1:
    # Проходим по списку и удаляем.
        features_train_test.remove(feature_id)
    # Делим на тестовую и обучающую части 
    X = df_encoded[features]
    y = df_encoded['ARRIVAL_DELAY']
    X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = X_train_all[features_train_test]
    X_test = X_test_all[features_train_test]

    # Perform calculations based on selected models
    if linear_regression_selected:
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)
        rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
        st.write("Linear Regression RMSE:", rmse_linear)
        arrival_data_linear = pd.DataFrame({'DESTINATION_AIRPORT': X_test_all['DESTINATION_AIRPORT'], 'ARRIVAL_DELAY_TEST': y_test, 'ARRIVAL_DELAY_PRED_LINEAR':y_pred_linear})
        grouped_data_linear = arrival_data_linear.groupby('DESTINATION_AIRPORT')
        rmse_list_linear = []
        # Iterate over the groups and calculate RMSE for each group
        for group_name, group_data in grouped_data_linear:
            group_indices = group_data.index
            group_y_test = group_data['ARRIVAL_DELAY_TEST']
            group_y_pred_linear = group_data['ARRIVAL_DELAY_PRED_LINEAR']
            rmse_linear = mean_squared_error(group_y_test, group_y_pred_linear, squared=False)
            rmse_list_linear.append((group_name, rmse_linear))
        # Sort the RMSE values in ascending order
        rmse_list_linear.sort(key=lambda x: x[1])
        # Select the top 3 arrival airports with the minimal RMSE
        top_3_airports_linear = [airport for airport, _ in rmse_list_linear[:3]]
        st.write(f'Топ 3 аэропортпа по прогнозу линейной регрессии: {top_3_airports_linear}')
        # Calculate residuals
        residuals_linear = y_test - y_pred_linear
        residuals_linear = pd.Series(residuals_linear)

    if random_forest_selected:
    # Обучаем модель случайных деревьев
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
        st.write("Random Forest RMSE:", rmse_rf)
        arrival_data_rf = pd.DataFrame({'DESTINATION_AIRPORT': X_test_all['DESTINATION_AIRPORT'], 'ARRIVAL_DELAY_TEST': y_test, 'ARRIVAL_DELAY_PRED_RF': y_pred_rf })
        grouped_data_rf = arrival_data_rf.groupby('DESTINATION_AIRPORT')
        rmse_list_rf = []
        # Iterate over the groups and calculate RMSE for each group
        for group_name, group_data in grouped_data_rf:
            group_indices = group_data.index
            group_y_test = group_data['ARRIVAL_DELAY_TEST']
            group_y_pred_rf = group_data['ARRIVAL_DELAY_PRED_RF']
            rmse_rf= mean_squared_error(group_y_test, group_y_pred_rf, squared=False)
            rmse_list_rf.append((group_name, rmse_rf))
        # Sort the RMSE values in ascending order
        rmse_list_rf.sort(key=lambda x: x[1])
        # Select the top 3 arrival airports with the minimal RMSE
        top_3_airports_rf = [airport for airport, _ in rmse_list_rf[:3]]
        st.write(f'Топ 3 аэропортпа по прогнозу модели Random Forest: {top_3_airports_rf}')
        # Calculate residuals
        residuals_rf = y_test - y_pred_rf
        residuals_rf = pd.Series(residuals_rf)


    if gradient_boosting_selected:
    # Обучаем гдадиент бустинг модель 
        gb_model = GradientBoostingRegressor()
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
        st.write("Gradient Boosting RMSE:", rmse_gb)
        arrival_data_gb = pd.DataFrame({'DESTINATION_AIRPORT': X_test_all['DESTINATION_AIRPORT'], 'ARRIVAL_DELAY_TEST': y_test, 'ARRIVAL_DELAY_PRED_GB': y_pred_gb })
        grouped_data_gb = arrival_data_gb.groupby('DESTINATION_AIRPORT')
        rmse_list_gb= []
        # Iterate over the groups and calculate RMSE for each group
        for group_name, group_data in grouped_data_gb:
            group_indices = group_data.index
            group_y_test = group_data['ARRIVAL_DELAY_TEST']
            group_y_pred_gb = group_data['ARRIVAL_DELAY_PRED_GB']
            rmse_gb = mean_squared_error(group_y_test, group_y_pred_gb, squared=False)
            rmse_list_gb.append((group_name, rmse_gb))
        # Sort the RMSE values in ascending order
        rmse_list_gb.sort(key=lambda x: x[1])
        # Select the top 3 arrival airports with the minimal RMSE
        top_3_airports_gb = [airport for airport, _ in rmse_list_gb[:3]]
        st.write(f'Топ 3 аэропортпа по прогнозу модели Gradient Boosting: {top_3_airports_gb}')
        # Calculate residuals
        residuals_gb = y_test - y_pred_gb
        residuals_gb = np.residuals_gb


    # Create a "Confirm Selection" button
        if st.button('Построить графики'):
            # Group data together
            hist_data = [residuals_linear, residuals_rf, residuals_gb]
            group_labels = ['Linear', 'RF', 'DF']

        # Create distplot with custom bin_size
            fig = ff.create_distplot(
            hist_data, group_labels)

        # Plot!
            st.plotly_chart(fig, use_container_width=True)



