#_________________________________
#Импорт библиотек
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit_deckgl as st_deckgl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
import matplotlib as mpl
import datetime, warnings, scipy
import plotly.figure_factory as ff
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
warnings.filterwarnings("ignore")
#_____________________________________________________


#______________________________________________________
# Загрузка данных
preprocessed_flights_df = pd.read_csv("/Users/avnekrasov/Library/CloudStorage/OneDrive-Личная/(2) Личное/Личное развитие/202306 Data Alalyst в Сбере/Дипломная работа/Graduate_work/preprocessed_flights.csv")
preprocessed_flights_df.dropna(inplace = True)
airports_df = pd.read_csv("/Users/avnekrasov/Library/CloudStorage/OneDrive-Личная/(2) Личное/Личное развитие/202306 Data Alalyst в Сбере/Дипломная работа/Graduate_work/airports.csv")
airports_df = airports_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
#______________________________________________________

#______________________________________________________
#Оформление боковой панели и шапки страницы
st.title("Дипломная работа ")
st.write ("Некрасов Андрей. Школа DA поток Февраль 2023")
st.title("Predicting flight delays")

# Настройка боковой панели
st.sidebar.write("Некрасов Андрей")
st.sidebar.write("Школа DA поток Февраль 2023")
st.sidebar.title('')
st.sidebar.title("Выбор моделей и параметров рассчета")
st.sidebar.info("Выберите модели для рассчета ")

# Создаем блок с чекбоксами для выбора моделей
linear_regression_selected = st.sidebar.checkbox('Linear Regression', value=True)
random_forest_selected = st.sidebar.checkbox('Random Forest')
gradient_boosting_selected = st.sidebar.checkbox('Gradient Boosting')
default_t_size = 20
st.sidebar.info("Выберите пропорции разбивики Train и Test ")
t_size = st.sidebar.slider("Select test size (%)", min_value=0, max_value=100, value=default_t_size)
st.sidebar.write('test_size =' + str(t_size))
test_size = t_size / 100
#________________________________________________________________

#________________________________________________________________
# Создание карты
m = folium.Map()

# Создание кластера маркеров
marker_cluster = MarkerCluster().add_to(m)

# Добавление точек аэропортов на карту и определение границ
for _, row in airports_df.iterrows():
    folium.Marker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        popup=row['AIRPORT'],
        tooltip=row['IATA_CODE']
    ).add_to(marker_cluster)

# Определение границ
bounds = marker_cluster.get_bounds()

# Подстройка масштаба карты
m.fit_bounds(bounds)

# Вывод карты
st.markdown("### Карта аэропортов")
folium_static(m)
#__________________________________________________________

#__________________________________________________________
# Функция для обработки выбора аэропорта и данных по нему
def on_airport_selected(selected_airport):
    origin_airport = selected_airport['IATA_CODE']
    df_origin = preprocessed_flights_df[preprocessed_flights_df['ORIGIN_AIRPORT'] == origin_airport]

    # Отображаем данные по аэропорту вылета 
    st.write(f"Total Outgoing Flights: {len(df_origin)}")
    st.write(f"Unique Arrival Airports: {df_origin['DESTINATION_AIRPORT'].nunique()}")
    st.write(f"Airlines Flying from this Airport: {df_origin['AIRLINE'].nunique()}")


# Создаем выпадающий список аэропортов
options = airports_df['AIRPORT']
default_airport = options[165]
selected_airport = st.selectbox('Выбор аэропорта вылета', options)

# Проверяем выбран ли аэропорт
if selected_airport:
    # Retrieve the selected airport information
    selected_airport_info = airports_df[airports_df['AIRPORT'] == selected_airport].iloc[0]
    st.write(f"IATA Code: {selected_airport_info['IATA_CODE']}")
    st.write(f"City: {selected_airport_info['CITY']}")
    st.write(f"Latitude: {selected_airport_info['LATITUDE']}")
    st.write(f"Longitude: {selected_airport_info['LONGITUDE']}")

    # Обработка события выбора аэропорта
    on_airport_selected(selected_airport_info)
#_________________________________________________________________________


#__________________________________________________________________________
# Создаем кнопку для подтверждения выбора и подготовки данных для обучения моделей
if st.button('Обучить модели для выбранного аэропорта'):
    # Задаем для переменной origin_airport значение IATA_CODE выбранного аэропорта
    origin_airport = selected_airport_info['IATA_CODE']
    # Создаем df_origin dataframe
    df_origin = preprocessed_flights_df[preprocessed_flights_df['ORIGIN_AIRPORT'] == origin_airport]
    st.markdown (f"Аэропорт вылета : {selected_airport_info['IATA_CODE']}")
    # Создаем df_encoded для подггтовки строковых данных к обучению
    df_encoded = df_origin[['DESTINATION_AIRPORT', 'AIRLINE']].copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['DESTINATION_AIRPORT', 'AIRLINE' ])
    df_encoded = pd.concat([df_origin, df_encoded], axis=1)
    #формируем перечень фичей 2-х видов с фичей для группировки 
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
    #_______________________________________________________________


    #_______________________________________________________________
    # Выполнение расчетов для линейной модели
    if linear_regression_selected:
        st.markdown("### Модель - Linear Regression")
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)
        rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
        st.write("Linear Regression RMSE:", rmse_linear)
        arrival_data_linear = pd.DataFrame({'DESTINATION_AIRPORT': X_test_all['DESTINATION_AIRPORT'], 'ARRIVAL_DELAY_TEST': y_test, 'ARRIVAL_DELAY_PRED_LINEAR':y_pred_linear})
        grouped_data_linear = arrival_data_linear.groupby('DESTINATION_AIRPORT')
        rmse_list_linear = []
        # Итерация по группам и вычисление RMSE для каждой группы
        for group_name, group_data in grouped_data_linear:
            group_indices = group_data.index
            group_y_test = group_data['ARRIVAL_DELAY_TEST']
            group_y_pred_linear = group_data['ARRIVAL_DELAY_PRED_LINEAR']
            rmse_linear = mean_squared_error(group_y_test, group_y_pred_linear, squared=False)
            rmse_list_linear.append((group_name, rmse_linear))

        # сортируем RMSE 
        rmse_list_linear.sort(key=lambda x: x[1])
        # выбраем 3 лучших аэропорта прибытия с минимальным RMSE
        top_3_airports_linear = [airport for airport, _ in rmse_list_linear[:3]]
        st.write(f'Топ 3 аэропортпа по прогнозу линейной регрессии: {top_3_airports_linear}')

        rmse_list_linear_df = pd.DataFrame(rmse_list_linear).sort_values(by = 1)
        #выводим датафрейм с рассчетом RMSE и кодами аэропортов
        rmse_list_linear_df = pd.DataFrame({'DESTINATION_AIRPORT': rmse_list_linear_df[0], 'RMSE_LINEAR': rmse_list_linear_df[1]})

        # Определяем функцию, которая будет применяться к каждой строке датафрейма
        def color_rows(row):
            color = 'SlateBlue' if row.name < 3 else ''  # Выделяем первые 3 строки
            return ['background-color: {}'.format(color)] * len(row)

        rmse_list_linear_df = rmse_list_linear_df.merge(airports_df[['AIRPORT', 'IATA_CODE']], how = 'left', left_on='DESTINATION_AIRPORT', right_on='IATA_CODE')

        # Применяем функцию к датафрейму и отображаем первые 5 строк с раскрашенными данными
        rmse_list_linear_df = rmse_list_linear_df.head(5).style.apply(color_rows, axis=1)
        st.dataframe(rmse_list_linear_df)

        #Строим пузырьковый график для предсказаний и реальных зачений
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_linear, color='SlateBlue', alpha=0.4, marker="o" )
        # Добавление заголовка графика
        ax.set_title('Linear Regression')
        # Подписи осей
        ax.set_xlabel('y_test')
        ax.set_ylabel('y_pred_linear')
        st.pyplot(fig)

        # рассчитываем разницу между реальными данными и предсказханиями 
        residuals_gb = y_test - y_pred_linear
        fig, ax = plt.subplots()
        ax.hist(residuals_gb, color='SlateBlue',alpha = 0.6, bins=50)
        # Добавление заголовка графика
        ax.set_title('Linear Regression - Histogram of Residuals')
        # Подписи осей
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        #______________________________________________________


    #____________________________________________________________
    # Выполнение расчетов для модели случайных деревьев
    if random_forest_selected:
        st.markdown("### Модель - Random Forest")
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
        st.write("Random Forest RMSE:", rmse_rf)
        arrival_data_rf = pd.DataFrame({'DESTINATION_AIRPORT': X_test_all['DESTINATION_AIRPORT'], 'ARRIVAL_DELAY_TEST': y_test, 'ARRIVAL_DELAY_PRED_RF': y_pred_rf })
        grouped_data_rf = arrival_data_rf.groupby('DESTINATION_AIRPORT')
        rmse_list_rf = []
        # Итерация по группам и вычисление RMSE для каждой группы
        for group_name, group_data in grouped_data_rf:
            group_indices = group_data.index
            group_y_test = group_data['ARRIVAL_DELAY_TEST']
            group_y_pred_rf = group_data['ARRIVAL_DELAY_PRED_RF']
            rmse_rf= mean_squared_error(group_y_test, group_y_pred_rf, squared=False)
            rmse_list_rf.append((group_name, rmse_rf))
        # Сортируем RMSE 
        rmse_list_rf.sort(key=lambda x: x[1])
        # выбраем 3 лучших аэропорта прибытия с минимальным RMSE
        top_3_airports_rf = [airport for airport, _ in rmse_list_rf[:3]]
        st.write(f'Топ 3 аэропортпа по прогнозу модели Random Forest: {top_3_airports_rf}')

        rmse_list_rf_df = pd.DataFrame(rmse_list_rf).sort_values(by = 1)
        #выводим датафрейм с рассчетом RMSE и кодами аэропортов
        rmse_list_rf_df = pd.DataFrame({'DESTINATION_AIRPORT': rmse_list_rf_df[0], 'RMSE_LINEAR': rmse_list_rf_df[1]})

        # Определяем функцию, которая будет применяться к каждой строке датафрейма
        def color_rows(row):
            color = 'LightGreen' if row.name < 3 else ''  # Выделяем первые 3 строки
            return ['background-color: {}'.format(color)] * len(row)

        rmse_list_rf_df = rmse_list_rf_df.merge(airports_df[['AIRPORT', 'IATA_CODE']], how = 'left', left_on='DESTINATION_AIRPORT', right_on='IATA_CODE')

        # Применяем функцию к датафрейму и отображаем первые 5 строк с раскрашенными данными
        rmse_list_rf_df = rmse_list_rf_df.head(5).style.apply(color_rows, axis=1)
        st.dataframe(rmse_list_rf_df)


        #Строим пузырьковый график для предсказаний и реальных зачений
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_rf, color='LightGreen', alpha=0.4, marker="v" )
        # Добавление заголовка графика
        ax.set_title('Random Forest')
        # Подписи осей
        ax.set_xlabel('y_test')
        ax.set_ylabel('y_pred_rf')
        st.pyplot(fig)

        # рассчитываем разницу между реальными данными и предсказханиями 
        residuals_gb = y_test - y_pred_rf

        fig, ax = plt.subplots()
        ax.hist(residuals_gb, color='LightGreen', alpha=0.6,  bins=50)
        # Добавление заголовка графика
        ax.set_title('Random Forest - Histogram of Residuals')
        # Подписи осей
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        #________________________________________________________________
    


    #________________________________________________________________
    # Обучаем гдадиент бустинг модель 
    if gradient_boosting_selected:
        st.markdown("### Модель - Gradient Boosting")
        gb_model = GradientBoostingRegressor()
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
        st.write("Gradient Boosting RMSE:", rmse_gb)
        arrival_data_gb = pd.DataFrame({'DESTINATION_AIRPORT': X_test_all['DESTINATION_AIRPORT'], 'ARRIVAL_DELAY_TEST': y_test, 'ARRIVAL_DELAY_PRED_GB': y_pred_gb })
        grouped_data_gb = arrival_data_gb.groupby('DESTINATION_AIRPORT')
        rmse_list_gb= []
        # Итерация по группам и вычисление RMSE для каждой группы
        for group_name, group_data in grouped_data_gb:
            group_indices = group_data.index
            group_y_test = group_data['ARRIVAL_DELAY_TEST']
            group_y_pred_gb = group_data['ARRIVAL_DELAY_PRED_GB']
            rmse_gb = mean_squared_error(group_y_test, group_y_pred_gb, squared=False)
            rmse_list_gb.append((group_name, rmse_gb))
        # Сортируем RMSE 
        rmse_list_gb.sort(key=lambda x: x[1])
        # выбраем 3 лучших аэропорта прибытия с минимальным RMSE
        top_3_airports_gb = [airport for airport, _ in rmse_list_gb[:3]]
        st.write(f'Топ 3 аэропортпа по прогнозу модели Gradient Boosting: {top_3_airports_gb}')

        rmse_list_gb_df = pd.DataFrame(rmse_list_gb).sort_values(by = 1)
        #выводим датафрейм с рассчетом RMSE и кодами аэропортов
        rmse_list_gb_df = pd.DataFrame({'DESTINATION_AIRPORT': rmse_list_gb_df[0], 'RMSE_LINEAR': rmse_list_gb_df[1]})

        # Определяем функцию, которая будет применяться к каждой строке датафрейма
        def color_rows(row):
            color = 'IndianRed' if row.name < 3 else ''  # Выделяем первые 3 строки
            return ['background-color: {}'.format(color)] * len(row)

        rmse_list_gb_df = rmse_list_gb_df.merge(airports_df[['AIRPORT', 'IATA_CODE']], how = 'left', left_on='DESTINATION_AIRPORT', right_on='IATA_CODE')

        # Применяем функцию к датафрейму и отображаем первые 5 строк с раскрашенными данными
        rmse_list_gb_df = rmse_list_gb_df.head(5).style.apply(color_rows, axis=1)
        st.dataframe(rmse_list_gb_df)

        #Строим пузырьковый график для предсказаний и реальных зачений
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_gb, color='IndianRed', alpha=0.4, marker="s" )
        # Добавление заголовка графика
        ax.set_title('Gradient Boosting')
        # Подписи осей
        ax.set_xlabel('y_test')
        ax.set_ylabel('y_pred_gb')
        st.pyplot(fig)

        # рассчитываем разницу между реальными данными и предсказханиями 
        residuals_gb = y_test - y_pred_gb

        fig, ax = plt.subplots()
        ax.hist(residuals_gb, color='IndianRed',alpha = 0.6, bins=50)
        # Добавление заголовка графика
        ax.set_title('Gradient Boosting - Histogram of Residuals')
        # Подписи осей
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)




