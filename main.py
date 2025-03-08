import os
import time
import threading
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import mplfinance as mpf
import openai
import xgboost as xgb
from flask import Flask
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from langdetect import detect

# ─────────────────────────────────────────────
# Configuración y variables de entorno
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TU_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID")
TARGET_THREAD_ID = int(os.getenv("TARGET_THREAD_ID", 2740))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "TU_OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Si tienes una API key para CoinGecko (plan Enterprise, por ejemplo)
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

COINS = {
    "bitcoin": "BTCUSDT"
}

# Timeframe y configuración de datos OHLC
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
OHLC_DAYS = 14  # Para obtener velas de 4h se recomienda 14 días

# Intervalo de escaneo en segundos (60 para respetar rate limit)
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 60))

# Configuración para el modelo ML
feature_columns = ["open", "high", "low", "close", "EMA_fast", "EMA_slow", "RSI", "BBU", "BBL"]
MODEL = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
MODEL_FITTED = False

# Variable global para controlar log de no detección (una vez por minuto)
last_no_detection_log_time = 0

# ─────────────────────────────────────────────
# Funciones del Telegram Handler
# ─────────────────────────────────────────────
def send_telegram_message(message, chat_id=TELEGRAM_CHAT_ID, message_thread_id=TARGET_THREAD_ID):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "message_thread_id": message_thread_id
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Error al enviar mensaje a Telegram: {response.text}")
    except Exception as e:
        print(f"Error en la conexión con Telegram: {e}")

def analyze_signal_with_chatgpt(message):
    system_prompt = (
        "Eres Higgs X, el agente de inteligencia encargado de vigilar el ecosistema. "
        "Responde de forma concisa y con un toque misterioso."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error en análisis AI: {e}"

def get_updates():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("result", [])
        elif response.status_code == 404:
            print(f"getUpdates devolvió 404 (posiblemente por uso de webhook): {response.text}")
            return []
        else:
            print(f"Error al obtener actualizaciones: {response.text}")
            return []
    except Exception as e:
        print(f"Error en la conexión con Telegram: {e}")
        return []

def telegram_bot_loop():
    # Se reduce la frecuencia a 60 segundos
    last_update_id = None
    while True:
        try:
            updates = get_updates()
            if updates:
                for update in updates:
                    update_id = update.get("update_id")
                    if last_update_id is None or update_id > last_update_id:
                        last_update_id = update_id
            time.sleep(60)
        except Exception as e:
            print(f"Error en el bucle del bot: {e}")
            time.sleep(10)

# ─────────────────────────────────────────────
# Funciones para obtener datos de mercado
# ─────────────────────────────────────────────
def get_ohlc(coin_id="bitcoin", vs_currency="usd", days=OHLC_DAYS):
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if isinstance(data, dict) and "status" in data:
            print(f"Error al obtener OHLC: {data}")
            return pd.DataFrame()
        if not isinstance(data, list):
            print("Error en la estructura de la respuesta OHLC:", data)
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception as e:
        print(f"Error al obtener OHLC para {coin_id}: {e}")
        return pd.DataFrame()

def fetch_data_coingecko(coin_id="bitcoin", vs_currency="usd", days=OHLC_DAYS):
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if "prices" not in data:
            print("Error al obtener datos de CoinGecko: 'prices' no encontrado en la respuesta.")
            print("Respuesta completa:", data)
            return pd.DataFrame()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["price"]
        df["high"] = df["price"]
        df["low"] = df["price"]
        df["close"] = df["price"]
        df["volume"] = 1
        return df
    except Exception as e:
        print(f"Error al obtener datos de CoinGecko: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# Funciones de Indicadores Técnicos
# ─────────────────────────────────────────────
def calculate_indicators(df):
    df = df.copy()
    df["EMA_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    rolling_mean = df["close"].rolling(window=20).mean()
    rolling_std = df["close"].rolling(window=20).std()
    df["BBU"] = rolling_mean + (rolling_std * 2)
    df["BBL"] = rolling_mean - (rolling_std * 2)
    df["sma_short"] = SMAIndicator(df["close"], window=10).sma_indicator()
    df["sma_long"] = SMAIndicator(df["close"], window=25).sma_indicator()
    return df

def check_golden_cross(df):
    if len(df) < 2:
        return False, ""
    if df["EMA_fast"].iloc[-2] < df["EMA_slow"].iloc[-2] and df["EMA_fast"].iloc[-1] > df["EMA_slow"].iloc[-1]:
        return True, "Golden Cross detectado (alcista)"
    return False, ""

def check_death_cross(df):
    if len(df) < 2:
        return False, ""
    if df["EMA_fast"].iloc[-2] > df["EMA_slow"].iloc[-2] and df["EMA_fast"].iloc[-1] < df["EMA_slow"].iloc[-1]:
        return True, "Death Cross detectado (bajista)"
    return False, ""

def check_bollinger_signals(df):
    if len(df) < 2:
        return False, ""
    latest = df.iloc[-1]
    signal = ""
    if df["close"].iloc[-2] < df["BBU"].iloc[-2] and latest["close"] > latest["BBU"]:
        signal = "Precio cruza hacia arriba la Banda Superior de Bollinger"
    band_width = latest["BBU"] - latest["BBL"]
    if band_width < (latest["close"] * 0.01):
        signal += " | Convergencia de Bandas (baja volatilidad)"
    return (True, signal) if signal else (False, "")

# ─────────────────────────────────────────────
# Funciones para el Modelo ML (XGBoost)
# ─────────────────────────────────────────────
def add_extra_features(data):
    data = data.copy()
    data["EMA_fast"] = data["close"].ewm(span=12, adjust=False).mean()
    data["EMA_slow"] = data["close"].ewm(span=26, adjust=False).mean()
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    rolling_mean = data["close"].rolling(window=20).mean()
    rolling_std = data["close"].rolling(window=20).std()
    data["BBU"] = rolling_mean + (rolling_std * 2)
    data["BBL"] = rolling_mean - (rolling_std * 2)
    return data

def train_ml_model(data):
    global MODEL_FITTED
    data = add_extra_features(data)
    features = data[feature_columns].pct_change().dropna()
    if features.empty:
        print("No hay datos suficientes para entrenar el modelo ML.")
        return
    target = (features["close"] > 0).astype(int)
    try:
        MODEL.fit(features, target)
        MODEL_FITTED = True
    except Exception as e:
        print(f"Error al entrenar el modelo ML: {e}")

def predict_cross_strength(data):
    if not MODEL_FITTED:
        print("Modelo ML no está entrenado. Se requiere llamar a train_ml_model antes.")
        return None
    data = add_extra_features(data)
    features = data[feature_columns].pct_change().dropna().iloc[-1:][feature_columns]
    try:
        prob = MODEL.predict_proba(features)[0]
        return prob[1]
    except Exception as e:
        print(f"Error en la predicción ML: {e}")
        return None

# ─────────────────────────────────────────────
# Función para generar y enviar gráficos de análisis
# ─────────────────────────────────────────────
def send_scan_graph(chat_id=TELEGRAM_CHAT_ID, timeframe="1h", cross_type="golden"):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="H")
    data = pd.DataFrame({
        "open": np.random.uniform(300, 400, size=100),
        "high": np.random.uniform(400, 500, size=100),
        "low": np.random.uniform(200, 300, size=100),
        "close": np.random.uniform(300, 400, size=100),
        "volume": np.random.uniform(100, 1000, size=100)
    }, index=dates)
    
    sma_short = data["close"].rolling(window=10).mean()
    sma_long = data["close"].rolling(window=25).mean()
    support = data["close"].min()
    resistance = data["close"].max()
    
    buf = io.BytesIO()
    caption = f"{COINS.get('bitcoin', 'BTCUSDT')} - {timeframe} - {cross_type.capitalize()} Cross"
    
    mc = mpf.make_marketcolors(up="green", down="red", inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="--", facecolor="black", edgecolor="white", gridcolor="white")
    ap0 = mpf.make_addplot(sma_short, color="cyan")
    ap1 = mpf.make_addplot(sma_long, color="magenta")
    sr_support = [support] * len(data)
    sr_resistance = [resistance] * len(data)
    ap2 = mpf.make_addplot(sr_support, color="green", linestyle="--", width=0.8)
    ap3 = mpf.make_addplot(sr_resistance, color="red", linestyle="--", width=0.8)
    
    fig, _ = mpf.plot(data, type="candle", style=style, title=caption,
                      volume=False, addplot=[ap0, ap1, ap2, ap3], returnfig=True)
    fig.savefig(buf, dpi=100, format="png")
    plt.close(fig)
    buf.seek(0)
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": buf}
    payload = {"chat_id": chat_id, "caption": caption, "message_thread_id": TARGET_THREAD_ID}
    try:
        response = requests.post(url, data=payload, files=files)
        if response.status_code != 200:
            print(f"Error al enviar el gráfico: {response.text}")
    except Exception as e:
        print(f"Error al enviar el gráfico a Telegram: {e}")

# ─────────────────────────────────────────────
# Función principal de escaneo de mercados
# ─────────────────────────────────────────────
def scan_markets():
    global last_no_detection_log_time
    for coin_id, symbol in COINS.items():
        try:
            df = get_ohlc(coin_id=coin_id, days=OHLC_DAYS)
            if df.empty or len(df) < 2:
                print(f"{datetime.now()} - No hay suficientes datos para analizar {symbol}")
                continue
            df = calculate_indicators(df)
            messages = []
            
            golden, golden_msg = check_golden_cross(df)
            death, death_msg = check_death_cross(df)
            bb_signal, bb_msg = check_bollinger_signals(df)
            
            rsi_value = df["RSI"].iloc[-1]
            strength = predict_cross_strength(df)
            
            if golden and rsi_value < 40:
                msg = (f"*ESS SCAN* [Thread {TARGET_THREAD_ID}]:\nAgente ha detectado un movimiento alcista en {symbol}.\n"
                       f"{golden_msg}.\nEMAs: {df['EMA_fast'].iloc[-1]:.2f} / {df['EMA_slow'].iloc[-1]:.2f}.\n"
                       f"RSI: {rsi_value:.2f} | Fuerza de señal (ML): {strength if strength is not None else 'N/A'}.\n"
                       "Se recomienda discreción en la toma de decisiones. ¡Suerte, agente!")
                messages.append(msg)
                send_scan_graph(cross_type="golden")
            if death and rsi_value > 60:
                msg = (f"*ESS SCAN* [Thread {TARGET_THREAD_ID}]:\nAgente ha detectado un movimiento bajista en {symbol}.\n"
                       f"{death_msg}.\nEMAs: {df['EMA_fast'].iloc[-1]:.2f} / {df['EMA_slow'].iloc[-1]:.2f}.\n"
                       f"RSI: {rsi_value:.2f} | Fuerza de señal (ML): {strength if strength is not None else 'N/A'}.\n"
                       "Se recomienda discreción en la toma de decisiones. ¡Suerte, agente!")
                messages.append(msg)
                send_scan_graph(cross_type="death")
            if bb_signal:
                msg = (f"*ESS SCAN* [Thread {TARGET_THREAD_ID}]:\nAnálisis de Bandas de Bollinger en {symbol}:\n"
                       f"{bb_msg}.\nRSI: {rsi_value:.2f}.\nSe recomienda discreción. ¡Suerte, agente!")
                messages.append(msg)
            
            if messages:
                for msg in messages:
                    print(f"{datetime.now()} - Enviando mensaje para {symbol}:")
                    print(msg)
                    send_telegram_message(msg)
                    analysis = analyze_signal_with_chatgpt(msg)
                    print(f"Análisis AI: {analysis}")
                    send_telegram_message(f"Análisis AI: {analysis}")
            else:
                # Solo loguear una vez por minuto si no se detecta nada
                now = time.time()
                if now - last_no_detection_log_time >= 60:
                    print(f"{datetime.now()} - No se detectó ninguna señal en {symbol}.")
                    last_no_detection_log_time = now
        except Exception as e:
            print(f"Error al procesar {coin_id}: {e}")

def log_indicators_status():
    """
    Cada minuto, obtiene los datos OHLC, calcula indicadores y registra
    en los logs el precio, SMA corta, SMA larga y RSI.
    """
    while True:
        for coin_id, symbol in COINS.items():
            df = get_ohlc(coin_id=coin_id, days=OHLC_DAYS)
            if df.empty or len(df) < 2:
                print(f"{datetime.now()} - [Status] No hay datos para {symbol}")
            else:
                df = calculate_indicators(df)
                latest = df.iloc[-1]
                log_msg = (f"{datetime.now()} - [Status] {symbol}: Precio: {latest['close']:.2f}, "
                           f"SMA_corta: {latest['sma_short']:.2f}, SMA_larga: {latest['sma_long']:.2f}, RSI: {latest['RSI']:.2f}")
                print(log_msg)
        time.sleep(60)

def background_scan():
    while True:
        scan_markets()
        time.sleep(SCAN_INTERVAL)

# ─────────────────────────────────────────────
# Configuración de la aplicación Flask
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return "ESS SCAN Bot en ejecución 24/7"

@app.route("/scan", methods=["GET"])
def scan_route():
    scan_markets()
    return "Escaneo completado y mensajes enviados."

# ─────────────────────────────────────────────
# Bloque principal
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Enviar mensaje inicial de activación
    send_telegram_message("Ess Sacan Activated, Cyb3er Conexion established...")
    # Entrenar el modelo ML con datos históricos
    historical_data = fetch_data_coingecko(coin_id="bitcoin", days=OHLC_DAYS)
    if not historical_data.empty:
        train_ml_model(historical_data)
    else:
        print("No se pudieron obtener datos históricos para entrenar el modelo ML.")
    # Iniciar hilo para escaneo de mercado
    scan_thread = threading.Thread(target=background_scan, daemon=True)
    scan_thread.start()
    # Iniciar hilo para registrar el estado de los indicadores cada minuto
    status_thread = threading.Thread(target=log_indicators_status, daemon=True)
    status_thread.start()
    # Iniciar hilo del bot de Telegram (opcional; se usa polling cada 60 segundos)
    telegram_thread = threading.Thread(target=telegram_bot_loop, daemon=True)
    telegram_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
