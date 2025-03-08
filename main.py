import os
import time
import threading
import requests
import pandas as pd
import numpy as np
from flask import Flask
from datetime import datetime

# Variables de entorno
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TARGET_THREAD_ID = int(os.getenv("TARGET_THREAD_ID", 2740))  # ID del tema "Ecosystem Signals"

# Configuración de CoinGecko
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
COINS = {
    "bitcoin": "BTCUSDT"
    # Puedes agregar más monedas aquí, usando el id de CoinGecko y el símbolo que prefieras.
}

def get_ohlc(coin_id="bitcoin", vs_currency="usd", days=14):
    """
    Obtiene datos OHLC de CoinGecko.
    Para days=14 se obtienen velas con intervalos de 4h (según la API de CoinGecko).
    Retorna un DataFrame con columnas: time, open, high, low, close.
    """
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": vs_currency,
        "days": days
    }
    response = requests.get(url, params=params)
    data = response.json()
    # data es una lista de listas: [timestamp, open, high, low, close]
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    # Convertir columnas numéricas
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])
    return df

def calculate_indicators(df):
    """
    Calcula los indicadores técnicos: EMAs, RSI y Bandas de Bollinger utilizando solo pandas y numpy.
    """
    # EMAs: ejemplo de 12 y 26 periodos
    df["EMA_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=26, adjust=False).mean()

    # RSI (14 periodos)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bandas de Bollinger (20 periodos, desviación 2)
    rolling_mean = df["close"].rolling(window=20).mean()
    rolling_std = df["close"].rolling(window=20).std()
    df["BBU"] = rolling_mean + (rolling_std * 2)
    df["BBL"] = rolling_mean - (rolling_std * 2)

    return df

def check_golden_cross(df):
    """
    Detecta Golden Cross: cuando la EMA rápida cruza al alza la EMA lenta.
    """
    if df["EMA_fast"].iloc[-2] < df["EMA_slow"].iloc[-2] and df["EMA_fast"].iloc[-1] > df["EMA_slow"].iloc[-1]:
        return True, "Golden Cross detectado (alcista)"
    return False, ""

def check_death_cross(df):
    """
    Detecta Death Cross: cuando la EMA rápida cruza a la baja la EMA lenta.
    """
    if df["EMA_fast"].iloc[-2] > df["EMA_slow"].iloc[-2] and df["EMA_fast"].iloc[-1] < df["EMA_slow"].iloc[-1]:
        return True, "Death Cross detectado (bajista)"
    return False, ""

def check_bollinger_signals(df):
    """
    Detecta señales con Bandas de Bollinger:
      - Cruce al alza del precio sobre la banda superior.
      - Convergencia de las bandas (reducción de la volatilidad).
    """
    latest = df.iloc[-1]
    signal = ""
    # Si el cierre previo estaba por debajo de la banda superior y el cierre actual la supera:
    if df["close"].iloc[-2] < df["BBU"].iloc[-2] and latest["close"] > latest["BBU"]:
        signal = "Precio cruza hacia arriba la Banda Superior de Bollinger"
    # Comprobamos si las bandas se están acercando (por ejemplo, diferencia menor al 1% del precio)
    band_width = latest["BBU"] - latest["BBL"]
    if band_width < (latest["close"] * 0.01):
        if signal:
            signal += " | "
        signal += "Convergencia de Bandas (baja volatilidad)"
    if signal:
        return True, signal
    return False, ""

def send_telegram_message(message):
    """
    Envía un mensaje a Telegram usando la API del bot.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=data)
    return response.json()

def scan_markets():
    """
    Función principal que analiza el mercado para cada moneda y envía señales.
    """
    for coin_id, symbol in COINS.items():
        try:
            df = get_ohlc(coin_id=coin_id)
            df = calculate_indicators(df)

            messages = []

            golden, golden_msg = check_golden_cross(df)
            if golden:
                messages.append(
                    f"*ESS SCAN* [Thread {TARGET_THREAD_ID}]:\nAgente ha detectado un movimiento alcista en {symbol}.\n"
                    f"{golden_msg}.\nEMAs: {df['EMA_fast'].iloc[-1]:.2f} / {df['EMA_slow'].iloc[-1]:.2f}.\n"
                    "Se recomienda discreción en la toma de decisiones. ¡Suerte, agente!"
                )

            death, death_msg = check_death_cross(df)
            if death:
                messages.append(
                    f"*ESS SCAN* [Thread {TARGET_THREAD_ID}]:\nAgente ha detectado un movimiento bajista en {symbol}.\n"
                    f"{death_msg}.\nEMAs: {df['EMA_fast'].iloc[-1]:.2f} / {df['EMA_slow'].iloc[-1]:.2f}.\n"
                    "Se recomienda discreción en la toma de decisiones. ¡Suerte, agente!"
                )

            bb_signal, bb_msg = check_bollinger_signals(df)
            if bb_signal:
                messages.append(
                    f"*ESS SCAN* [Thread {TARGET_THREAD_ID}]:\nAnálisis de Bandas de Bollinger en {symbol}:\n{bb_msg}.\n"
                    "Se recomienda discreción. ¡Suerte, agente!"
                )

            # Enviar mensajes si se han detectado señales
            for msg in messages:
                print(f"{datetime.now()} - Enviando mensaje para {symbol}:")
                print(msg)
                send_telegram_message(msg)
        except Exception as e:
            print(f"Error al procesar {coin_id}: {e}")

def background_scan():
    """
    Ejecuta el escaneo del mercado cada 5 segundos en un bucle indefinido.
    """
    while True:
        scan_markets()
        time.sleep(5)

# Configuración de la aplicación Flask para Railway
app = Flask(__name__)

@app.route("/")
def index():
    return "ESS SCAN Bot en ejecución 24/7"

@app.route("/scan", methods=["GET"])
def scan_route():
    scan_markets()
    return "Escaneo completado y mensajes enviados."

if __name__ == "__main__":
    # Inicia el thread de escaneo en segundo plano
    scan_thread = threading.Thread(target=background_scan, daemon=True)
    scan_thread.start()

    # Inicia la aplicación Flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
