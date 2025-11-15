import cdsapi
import zipfile
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

API_URL = "https://ads.atmosphere.copernicus.eu/api"
API_KEY = " " 

def api(url, key):
    rc_file = os.path.expanduser('~/.cdsapirc')
    with open(rc_file, 'w') as f:
        f.write(f"url: {url}\n")
        f.write(f"key: {key}\n")
    print(f" Configurare API completa. ")

def download():
    output_zip = "download.zip"
    extract_folder = "date_input"
    
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    client = cdsapi.Client()

    print(" Descarc de la Copernicus ")
    client.retrieve(
        'cams-global-reanalysis-eac4',
        {
            'date': '2024-12-31/2024-12-31',
            'time': '12:00',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 
                '2m_temperature', 'particulate_matter_2.5um', 'surface_pressure'
            ],
            'area': [48.5, 20, 43, 30], # Romania
            'format': 'netcdf_zip',
        },
        output_zip
    )
    
    print("Dezarhivare")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        nc_file = [f for f in zip_ref.namelist() if f.endswith('.nc')][0]
    return os.path.join(extract_folder, nc_file)

def ruleazaanaliza():
    api(API_URL, API_KEY)
    try:
        nc_path = download()
    except Exception as e:
        print(f"Eroare la download: {e}")
        return
    #Procesare
    print(f" Procesare fisier: {nc_path} ")
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    df = ds.to_dataframe().reset_index().dropna()
    
    rename_map = {
        'u10': 'u_wind', 'v10': 'v_wind', 't2m': 'temperature', 'sp': 'pressure', 'pm2p5': 'actual_pm25' 
    }
    for col in df.columns:
        if col in rename_map:
            df.rename(columns={col: rename_map[col]}, inplace=True)

    df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
    df['temp_celsius'] = df['temperature'] - 273.15
    df['pressure_hpa'] = df['pressure'] / 100.0
    df['actual_pm25_ug'] = df['actual_pm25'] * 1e9

    X = df[['wind_speed', 'temp_celsius', 'pressure_hpa', 'latitude', 'longitude']]
    y = df['actual_pm25_ug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Model1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    #Model2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results = pd.DataFrame({
        'Real_CAMS': y_test,
        'Pred_LinearReg': y_pred_lr,
        'Pred_RandomForest': y_pred_rf
    })
    
    results_sorted = results.sort_values(by='Real_CAMS').reset_index(drop=True)
    plt.figure(figsize=(14, 7))
    plt.plot(results_sorted['Real_CAMS'], label='Date Reale (CAMS)', color='black', linewidth=2, linestyle='-')
    plt.plot(results_sorted['Pred_LinearReg'], label='Linear Regression', color='red', marker='.', linestyle='', markersize=8)
    plt.plot(results_sorted['Pred_RandomForest'], label='Random Forest', color='green', linestyle=':', linewidth=3)
    plt.title('Predictie Poluare PM2.5 - Romania (Date Reale Copernicus)', fontsize=16)
    plt.xlabel('Puncte geografice (sortate dupa nivelul poluarii)', fontsize=12)
    plt.ylabel('Concentratie PM2.5 [micrograme/m^3]', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    nume_fisier_grafic = "grafic_predictie_poluare.png"
    plt.savefig(nume_fisier_grafic, dpi=300)
    print(f"Graficul a fost salvat ca: {nume_fisier_grafic}")
    plt.show()

if __name__ == "__main__":
    ruleazaanaliza()