import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from translations import bilanco_translations, gelir_tablosu_translations, nakit_akisi_translations


st.set_page_config(
    page_title="Hisse Analiz Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #2c3e50;
    }
    
    .info-card h3 {
        color: #2c3e50;
        margin-top: 0;
        font-size: 1.3rem;
    }
    
    .positive {
        color: #27ae60;
    }
    
    .negative {
        color: #e74c3c;
    }
    
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

###########################
# Yardımcı Fonksiyonlar
###########################
def calculate_rsi(data, periods=14):
    delta = data['Kapanış'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Kapanış'].rolling(window=window).mean()
    std = data['Kapanış'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Kapanış'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Kapanış'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def translate_index(df, translation_dict):
    df = df.copy()
    df.index = [translation_dict.get(i, i) for i in df.index]
    return df

def calculate_ichimoku(data):
    tenkan_sen = (data['Yüksek'].rolling(9).max() + data['Düşük'].rolling(9).min()) / 2
    kijun_sen = (data['Yüksek'].rolling(26).max() + data['Düşük'].rolling(26).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (data['Yüksek'].rolling(52).max() + data['Düşük'].rolling(52).min()) / 2
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

def get_total_returns(ticker):
    end_date = datetime.today()
    start_date = end_date - pd.DateOffset(years=1)

    assets = {
        'BIST100': 'XU100.IS',
        'Dolar': 'TRY=X',
        'Euro': 'EURTRY=X',
        'Altın': 'GC=F',
        'Gümüş': 'SI=F',
        'Bitcoin': 'BTC-USD',
        'Hisse': ticker
    }

    returns = {}
    for name, symbol in assets.items():
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                returns[name] = float(((end_price - start_price) / start_price) * 100)
        except Exception as e:
            print(f"{name} hata: {e}")
            continue

    return pd.DataFrame.from_dict(returns, orient='index', columns=['Getiri (%)'])



###########################
# Sidebar Parametreleri
###########################
with st.sidebar:
    st.header("Analiz Parametreleri")
    ticker = st.text_input("Hisse Senedi Sembolü", "KCHOL.IS")
    start_date = st.date_input("Başlangıç Tarihi", datetime(2020, 1, 1))
    end_date = st.date_input("Bitiş Tarihi", datetime.today())

    if st.button("Analizi Başlat", type="primary", use_container_width=True):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error("Veri bulunamadı. Lütfen geçerli bir sembol girin.")
            else:
                data = data.reset_index()
                data_new = data[['Date', 'Close', 'Volume', 'High', 'Low']].copy()
                data_new.columns = ['Tarih', 'Kapanış', 'Hacim', 'Yüksek', 'Düşük']
                data_new['Tarih'] = pd.to_datetime(data_new['Tarih'])
                data_new['MA20'] = data_new['Kapanış'].rolling(20).mean()
                data_new['MA50'] = data_new['Kapanış'].rolling(50).mean()
                data_new['MA200'] = data_new['Kapanış'].rolling(200).mean()
                data_new['RSI'] = calculate_rsi(data_new)
                data_new['BB_SMA'], data_new['BB_Upper'], data_new['BB_Lower'] = calculate_bollinger_bands(data_new)
                data_new['MACD'], data_new['MACD_Signal'], data_new['MACD_Hist'] = calculate_macd(data_new)
                data_new['Hacim_Fark'] = data_new['Hacim'].diff()
                data_new['Tenkan_Sen'], data_new['Kijun_Sen'], data_new['Senkou_Span_A'], data_new['Senkou_Span_B'] = calculate_ichimoku(data_new)

                st.session_state.data = data_new
        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")

###########################
# Ana Başlık
###########################
st.title("📈 Hisse Senedi Analiz Dashboard")

if 'bist100_value' not in st.session_state or 'bankacilik_value' not in st.session_state or 'btc_value' not in st.session_state:
    try:
        bist100_data = yf.download("XU100.IS", period="1d")['Close'].iloc[-1]
        bankacilik_data = yf.download("XBANK.IS", period="1d")['Close'].iloc[-1]
        btc = yf.download("BTC-USD", period="1d")['Close'].iloc[-1]
        st.session_state.bist100_value = float(bist100_data)
        st.session_state.bankacilik_value = float(bankacilik_data)
        st.session_state.btc_value = float(btc)
    except Exception:
        st.session_state.bist100_value = None
        st.session_state.bankacilik_value = None
        st.session_state.btc_value = None

if st.session_state.get('bist100_value') is not None and st.session_state.get('bankacilik_value') is not None and st.session_state.get('btc_value') is not None:
    st.markdown(
        f"**BIST 100 Endeksi:** {st.session_state.bist100_value:,.2f} | **Bankacılık Endeksi:** {st.session_state.bankacilik_value:,.2f} | **Bitcoin:** {st.session_state.btc_value:,.2f} USD",
        unsafe_allow_html=True
    )
else:
    st.markdown("**BIST 100 Endeksi:** Veri alınamadı | **Bankacılık Endeksi:** Veri alınamadı | **Bitcoin: Veri alınamadı ", unsafe_allow_html=True)


st.markdown("---")

###########################
# Teknik Analiz Dashboard
###########################
if 'data' in st.session_state:
    data_new = st.session_state.data
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        df_prophet = data_new[['Tarih', 'Kapanış']].rename(columns={'Tarih': 'ds', 'Kapanış': 'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        merged = pd.merge(data_new[['Tarih', 'Kapanış']], forecast[['ds', 'yhat']], left_on='Tarih', right_on='ds', how='inner')
        mae = mean_absolute_error(merged['Kapanış'], merged['yhat'])
        mse = mean_squared_error(merged['Kapanış'], merged['yhat'])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((merged['Kapanış'] - merged['yhat']) / merged['Kapanış'])) * 100

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Kapanış'], name='Kapanış', line=dict(color='#3498db')))
        fig_price.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['MA20'], name='20 Günlük MA', line=dict(color='#e74c3c', dash='dot')))
        fig_price.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['MA50'], name='50 Günlük MA', line=dict(color='#2ecc71', dash='dot')))
        fig_price.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['MA200'], name='200 Günlük MA', line=dict(color='#9b59b6', dash='dot')))

        fig_price.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Tahmin (Prophet)',
            line=dict(color='#8e44ad', dash='dash')
        ))

        golden_crosses = []
        death_crosses = []

        for i in range(1, len(data_new)):
            ma50_now = data_new['MA50'].iloc[i]
            ma200_now = data_new['MA200'].iloc[i]
            ma50_prev = data_new['MA50'].iloc[i - 1]
            ma200_prev = data_new['MA200'].iloc[i - 1]

            if pd.notna(ma50_now) and pd.notna(ma200_now) and pd.notna(ma50_prev) and pd.notna(ma200_prev):
                if ma50_now > ma200_now and ma50_prev <= ma200_prev:
                    golden_crosses.append((data_new['Tarih'].iloc[i], data_new['Kapanış'].iloc[i]))
                elif ma50_now < ma200_now and ma50_prev >= ma200_prev:
                    death_crosses.append((data_new['Tarih'].iloc[i], data_new['Kapanış'].iloc[i]))

        for date, price in golden_crosses:
            fig_price.add_trace(go.Scatter(
                x=[date],
                y=[price],
                mode='markers+text',
                name='Golden Cross',
                marker=dict(color='gold', size=10, symbol='triangle-up'),
                text=["Golden Cross"],
                textposition="top center"
            ))

        for date, price in death_crosses:
            fig_price.add_trace(go.Scatter(
                x=[date],
                y=[price],
                mode='markers+text',
                name='Death Cross',
                marker=dict(color='black', size=10, symbol='triangle-down'),
                text=["Death Cross"],
                textposition="bottom center"
            ))

        support_level = data_new['Kapanış'].rolling(window=50, min_periods=1).min().iloc[-1]
        resistance_level = data_new['Kapanış'].rolling(window=50, min_periods=1).max().iloc[-1]

        fig_price.add_hline(
            y=support_level,
            line=dict(color='#0000FF', dash='dash', width=1.5),  
            annotation_text=f"Destek: {support_level:.2f}",
            annotation_position="bottom right",
            annotation=dict(font=dict(color='#0000FF'))
        )
        fig_price.add_hline(
            y=resistance_level,
            line=dict(color='#ff0000', dash='dash', width=1.5), 
            annotation_text=f"Direnç: {resistance_level:.2f}",
            annotation_position="top right",
            annotation=dict(font=dict(color='#ff0000'))
        )


        fig_price.add_annotation(
            text=f"📊 Prophet Başarı:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%",
            xref="paper", yref="paper",
            x=0.01, y=0.99, showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#2c3e50",
            borderwidth=1
        )

        fig_price.update_layout(
            title='Kapanış Fiyatı ve Hareketli Ortalamalar + Prophet Tahmini',
            xaxis_title='Tarih',
            yaxis_title='Fiyat (TL)',
            template='plotly_white'
        )

        st.markdown("""
        <div style="display: flex; gap: 10px; margin-top: 10px; margin-bottom: -20px;">
            <div style="flex: 1; background-color: #eafbea; padding: 10px; border-radius: 8px; border-left: 5px solid #27ae60;">
                <p style="margin: 0; font-size: 13px;">✨ <strong>Son Yaşanmış Golden Cross:</strong><br>""" + 
                (golden_crosses[-1][0].strftime('%Y-%m-%d') if golden_crosses else "Yok") + """
                </p>
            </div>
            <div style="flex: 1; background-color: #fdecea; padding: 10px; border-radius: 8px; border-left: 5px solid #c0392b;">
                <p style="margin: 0; font-size: 13px;">⚠️ <strong>Son Yaşanmış Death Cross:</strong><br>""" + 
                (death_crosses[-1][0].strftime('%Y-%m-%d') if death_crosses else "Yok") + """
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_price, use_container_width=True)

        latest_price = data_new['Kapanış'].iloc[-1]
        ma20 = data_new['MA20'].iloc[-1]
        ma50 = data_new['MA50'].iloc[-1]
        ma200 = data_new['MA200'].iloc[-1]
        trend = "Yükseliş" if ma20 > ma50 else "Düşüş"
        trend_color = "#27ae60" if ma20 > ma50 else "#e74c3c"

        st.markdown(f"""
        <div class="info-card">
            <h3>📈 Son Fiyat Bilgileri</h3>
            <p style="font-size: 24px; color: {trend_color};">{latest_price:.2f} TL</p>
            <p>20 Günlük MA: {ma20:.2f}</p>
            <p>50 Günlük MA: {ma50:.2f}</p>
            <p>200 Günlük MA: {ma200:.2f}</p>
            <p>Trend: <span style="color: {trend_color};">{trend}</span></p>
            <p style="font-style: italic;">(Kapanış fiyatı, ortalamalar ve Prophet tahmini dahil)</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['RSI'], name='RSI', line=dict(color='#9b59b6')))
        fig_rsi.update_layout(title='Göreceli Güç Endeksi (RSI)', yaxis_range=[0, 100], xaxis_title='Tarih', yaxis_title='RSI', template='plotly_white')
        fig_rsi.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1)
        fig_rsi.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1)
        st.plotly_chart(fig_rsi, use_container_width=True)

        latest_rsi = data_new['RSI'].iloc[-1]
        rsi_status = "Aşırı Alım" if latest_rsi > 70 else ("Aşırı Satım" if latest_rsi < 30 else "Normal")
        rsi_color = "#e74c3c" if latest_rsi > 70 else ("#27ae60" if latest_rsi < 30 else "#2c3e50")
        st.markdown(f"""
        <div class="info-card">
            <h3>💹 RSI Durumu</h3>
            <p style="font-size: 24px; color: {rsi_color};">{latest_rsi:.1f}</p>
            <p>Durum: {rsi_status}</p>
            <p style="font-style: italic;">(RSI grafiği ile aşırı alım/satım durumunun belirlenmesi)</p>
        </div>
        """, unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['BB_Upper'], name='Üst Bant', line=dict(color='#95a5a6')))
        fig_bb.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['BB_Lower'], name='Alt Bant', line=dict(color='#95a5a6'), fill='tonexty'))
        fig_bb.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Kapanış'], name='Kapanış', line=dict(color='#3498db')))
        fig_bb.update_layout(title='Bollinger Bantları', xaxis_title='Tarih', yaxis_title='Fiyat (TL)', template='plotly_white')
        st.plotly_chart(fig_bb, use_container_width=True)

        latest_bb = data_new['Kapanış'].iloc[-1]
        bb_position = "Üst Bandın Üzerinde" if latest_bb > data_new['BB_Upper'].iloc[-1] else ("Alt Bandın Altında" if latest_bb < data_new['BB_Lower'].iloc[-1] else "Bantlar Arasında")
        st.markdown(f"""
        <div class="info-card">
            <h3>📉 Bollinger Bantları</h3>
            <p>Son Fiyat: {latest_bb:.2f} TL</p>
            <p>Pozisyon: {bb_position}</p>
            <p style="font-style: italic;">(Bollinger Bantları grafiği ile fiyat volatilitesi ve konum analizi)</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(x=data_new['Tarih'], y=data_new['MACD_Hist'], name='Histogram', marker=dict(color=np.where(data_new['MACD_Hist'] < 0, '#e74c3c', '#2ecc71'))))
        fig_macd.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['MACD'], name='MACD', line=dict(color='#3498db')))
        fig_macd.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['MACD_Signal'], name='Sinyal', line=dict(color='#e67e22')))
        fig_macd.update_layout(title='MACD Göstergesi', xaxis_title='Tarih', yaxis_title='Değer', template='plotly_white')
        st.plotly_chart(fig_macd, use_container_width=True)

        latest_macd = data_new['MACD'].iloc[-1]
        latest_macd_signal = data_new['MACD_Signal'].iloc[-1]
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 MACD Bilgisi</h3>
            <p>MACD: {latest_macd:.2f}</p>
            <p>Sinyal: {latest_macd_signal:.2f}</p>
            <p style="font-style: italic;">(MACD grafiği ile trend dönüşü ve momentum analizi)</p>
        </div>
        """, unsafe_allow_html=True)

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=data_new['Tarih'], y=data_new['Hacim'], name='Hacim'))
        fig_vol.update_traces(marker_color='#0000FF', marker_line_width=0, marker_opacity=1)
        fig_vol.update_layout(title='Hacim Zaman Serisi', xaxis_title='Tarih', yaxis_title='Hacim', template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)

        latest_volume = data_new['Hacim'].iloc[-1]
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 Hacim Bilgisi</h3>
            <p>Son Hacim: {latest_volume:,.0f}</p>
            <p style="font-style: italic;">(Hacim grafiği ile işlem yoğunluğu analizi)</p>
        </div>
        """, unsafe_allow_html=True)

    with row3_col2:
        fig_vol_diff = go.Figure()
        fig_vol_diff.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Hacim_Fark'], mode='lines', name='Hacim Farkı', line=dict(color='#e67e22')))
        fig_vol_diff.update_layout(title='Günlük Hacim Farkı', xaxis_title='Tarih', yaxis_title='Hacim Farkı', template='plotly_white')
        st.plotly_chart(fig_vol_diff, use_container_width=True)

        latest_vol_diff = data_new['Hacim_Fark'].iloc[-1]
        vol_diff_color = "#27ae60" if latest_vol_diff >= 0 else "#e74c3c"
        st.markdown(f"""
        <div class="info-card">
            <h3>🔄 Günlük Hacim Değişimi</h3>
            <p style="color:{vol_diff_color};">Son Değişim: {latest_vol_diff:,.0f}</p>
            <p style="font-style: italic;">(Hacim farkı grafiği ile günlük hacim değişimleri takibi)</p>
        </div>
        """, unsafe_allow_html=True)

    ######################################
    # Fibonacci Retracement Analizi
    ######################################
    st.markdown("---")
    st.subheader("Fibonacci Retracement Analizi")

    low_price = data_new['Kapanış'].min()
    high_price = data_new['Kapanış'].max()
    diff = high_price - low_price

    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    retracement_levels = [high_price - level * diff for level in levels]

    fig_fib = go.Figure()
    fig_fib.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Kapanış'], name='Kapanış', line=dict(color='#3498db')))
    for level, retracement in zip(levels, retracement_levels):
        fig_fib.add_hline(y=retracement, line=dict(dash='dot'), annotation_text=f'{level*100:.1f}%', annotation_position="right")
    fig_fib.update_layout(title='Fibonacci Retracement Analizi', xaxis_title='Tarih', yaxis_title='Fiyat (TL)', template='plotly_white')
    st.plotly_chart(fig_fib, use_container_width=True)

    current_price = data_new['Kapanış'].iloc[-1]
    fibonacci_zone = None
    for i in range(len(retracement_levels)-1):
        if current_price <= retracement_levels[i] and current_price > retracement_levels[i+1]:
            fibonacci_zone = f"{levels[i]*100:.1f}% - {levels[i+1]*100:.1f}%"
            break
    if fibonacci_zone is None:
        fibonacci_zone = "Üstünde (%0 seviyesi)" if current_price > retracement_levels[0] else "Altında (%100 seviyesi)"

    st.markdown(f"""
    <div class="info-card">
        <h3>🚩 Fibonacci Detayları</h3>
        <ul>
            <li><strong>%0 (Direnç Seviyesi):</strong> {retracement_levels[0]:.2f} TL - En yüksek fiyat; direnç bölgesi.</li>
            <li><strong>%23.6 (Hafif Düzeltme):</strong> {retracement_levels[1]:.2f} TL - Kısa vadeli hafif geri çekilme sinyali.</li>
            <li><strong>%38.2 (Önemli Destek/Direnç):</strong> {retracement_levels[2]:.2f} TL - İlk önemli destek/direnç noktası.</li>
            <li><strong>%50 (Kritik Seviye):</strong> {retracement_levels[3]:.2f} TL - Güçlü geri çekilme ve denge bölgesi.</li>
            <li><strong>%61.8 (Güçlü Destek):</strong> {retracement_levels[4]:.2f} TL - Fiyat toparlanması için kritik destek.</li>
            <li><strong>%78.6 (Derin Düzeltme):</strong> {retracement_levels[5]:.2f} TL - Derin geri çekilme, önemli destek alanı.</li>
            <li><strong>%100 (Destek Seviyesi):</strong> {retracement_levels[6]:.2f} TL - En düşük fiyat; kritik destek noktası.</li>
        </ul>
        <p>Mevcut fiyat: {current_price:.2f} TL, Fibonacci aralığında: {fibonacci_zone}</p>
        <p style="font-style: italic;">(Grafikteki Fibonacci seviyeleri, ilgili fiyat noktaları ve açıklamaları)</p>
    </div>
    """, unsafe_allow_html=True)

    ######################################
    # Ichimoku Cloud Analizi
    ######################################
    st.markdown("---")
    st.subheader("Ichimoku Cloud Analizi")

    fig_ichimoku = go.Figure()
    fig_ichimoku.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Kapanış'], name='Kapanış', line=dict(color='#3498db')))
    fig_ichimoku.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Tenkan_Sen'], name='Tenkan-Sen', line=dict(color='#e74c3c')))
    fig_ichimoku.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Kijun_Sen'], name='Kijun-Sen', line=dict(color='#2ecc71')))
    fig_ichimoku.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Senkou_Span_A'], name='Senkou Span A', line=dict(color='#9b59b6')))
    fig_ichimoku.add_trace(go.Scatter(x=data_new['Tarih'], y=data_new['Senkou_Span_B'], name='Senkou Span B', line=dict(color='#e67e22'), fill='tonexty', fillcolor='rgba(155, 89, 182, 0.2)'))
    fig_ichimoku.update_layout(title='Ichimoku Cloud Analizi', xaxis_title='Tarih', yaxis_title='Fiyat (TL)', template='plotly_white')
    st.plotly_chart(fig_ichimoku, use_container_width=True)

    latest_price = data_new['Kapanış'].iloc[-1]
    latest_tenkan = data_new['Tenkan_Sen'].iloc[-1]
    latest_kijun = data_new['Kijun_Sen'].iloc[-1]
    latest_span_a = data_new['Senkou_Span_A'].iloc[-1]
    latest_span_b = data_new['Senkou_Span_B'].iloc[-1]

    trend_status = "Yükseliş" if latest_price > latest_span_a and latest_price > latest_span_b else ("Düşüş" if latest_price < latest_span_a and latest_price < latest_span_b else "Nötr")
    trend_color = "#27ae60" if trend_status == "Yükseliş" else ("#e74c3c" if trend_status == "Düşüş" else "#2c3e50")

    st.markdown(f"""
    <div class="info-card">
        <h3>☁️ Ichimoku Detayları</h3>
        <ul>
            <li><strong>Tenkan-Sen (Dönüş Çizgisi):</strong> {latest_tenkan:.2f} TL - 9 günlük kısa vadeli trend göstergesi.</li>
            <li><strong>Kijun-Sen (Temel Çizgi):</strong> {latest_kijun:.2f} TL - 26 günlük orta vadeli trend ve destek/direnç.</li>
            <li><strong>Senkou Span A (Bulut Önü A):</strong> {latest_span_a:.2f} TL - Bulutun ilk sınırı, destek/direnç seviyesi.</li>
            <li><strong>Senkou Span B (Bulut Önü B):</strong> {latest_span_b:.2f} TL - Bulutun ikinci sınırı, uzun vadeli denge.</li>
        </ul>
        <p>Mevcut Fiyat: {latest_price:.2f} TL</p>
        <p>Trend Durumu: <span style="color: {trend_color};">{trend_status}</span></p>
        <p style="font-style: italic;">(Ichimoku grafiği ile trend yönü, momentum ve destek/direnç analizi)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Getiri Karşılaştırması (Son 1 Yıl)")

    returns_df = get_total_returns(ticker)

    if not returns_df.empty:
        returns_df["Varlık"] = returns_df.index
        returns_df = returns_df.sort_values("Getiri (%)", ascending=True).reset_index(drop=True)

        fig_returns = go.Figure()
        for i, row in returns_df.iterrows():
            color = '#f39c12' if row['Varlık'] == 'Hisse' else ('#e74c3c' if row['Getiri (%)'] < 0 else '#2ecc71')
            text_value = f"{row['Getiri (%)']:.2f}%"
            
            fig_returns.add_trace(go.Bar(
                x=[row['Varlık']],
                y=[row['Getiri (%)']],
                marker_color=color,
                text=text_value,
                textposition='auto',
                width=[0.5],
                hovertemplate='%{x}<br>Getiri: %{y:.2f}%<extra></extra>',
                textfont=dict(
                    family="Arial Black",
                    size=14,
                    color="#2c3e50"
                )
            ))

        fig_returns.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=30),
            yaxis=dict(title="Getiri (%)", zeroline=True),
            xaxis=dict(title="Varlıklar"),
            title={
                'text': "Seçili Hisse ile Diğer Varlıkların 1 Yıllık Getiri Karşılaştırması",
                'x': 0.5,
                'xanchor': 'center'
            },
            template="plotly_white",
            showlegend=False,
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    else:
        st.warning("Getiri verileri alınamadı.")



    ######################################
    # Şirket Bilgileri ve Finansal Tablolar
    ######################################
    st.markdown("---")
    st.subheader("🏢 Şirket Bilgileri ve Finansal Tablolar")

    ticker_object = yf.Ticker(ticker)
    try:
        info = ticker_object.info
        balance_sheet = ticker_object.balance_sheet
        income_statement = ticker_object.financials
        cash_flow = ticker_object.cashflow

        st.markdown(f"""
        <div class="info-card">
            <h3>🏢 {info.get("longName", "Bilgi Yok")}</h3>
            <p><strong>Sektör:</strong> {info.get("sector", "Sektör Bilgisi Yok")}</p>
            <p><strong>Endüstri:</strong> {info.get("industry", "Endüstri Bilgisi Yok")}</p>
            <p><strong>Özet Bilgi:</strong> {info.get("longBusinessSummary", "Özet Bulunamadı")}</p>
        </div>
        """, unsafe_allow_html=True)

        def format_numbers(df):
            return df.applymap(lambda x: "{:,.0f}".format(x) if pd.notnull(x) else "N/A")

        def format_dates(df):
            df.columns = [pd.to_datetime(col).strftime('%Y-%m-%d') if pd.notnull(col) else col for col in df.columns]
            return df

        st.markdown("### 📄 Bilanço")
        if balance_sheet is not None and not balance_sheet.empty:
            balance_sheet = format_dates(balance_sheet)
            balance_sheet = translate_index(balance_sheet, bilanco_translations)  
            formatted_balance_sheet = format_numbers(balance_sheet)
            st.dataframe(formatted_balance_sheet)

        st.markdown(f"""
        <div class="info-card">
            <h3>Ne İşe Yarar?</h3>
            <p style="font-style: italic;">Bilanço, şirketin belirli bir tarihteki varlık, borç ve özkaynak durumunu gösterir.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📑 Gelir Tablosu")
        if income_statement is not None and not income_statement.empty:
            income_statement = format_dates(income_statement)
            income_statement = translate_index(income_statement, gelir_tablosu_translations)  
            formatted_income_statement = format_numbers(income_statement)
            st.dataframe(formatted_income_statement)


        st.markdown(f"""
        <div class="info-card">
            <h3>Ne İşe Yarar?</h3>
            <p style="font-style: italic;">Gelir tablosu, şirketin belirli bir dönemdeki gelir, gider ve kâr/zarar durumunu yansıtır.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💰 Nakit Akışı")
        if cash_flow is not None and not cash_flow.empty:
            cash_flow = format_dates(cash_flow)
            cash_flow = translate_index(cash_flow, nakit_akisi_translations)  
            formatted_cash_flow = format_numbers(cash_flow)
            st.dataframe(formatted_cash_flow)

        st.markdown(f"""
        <div class="info-card">
            <h3>Ne İşe Yarar?</h3>
            <p style="font-style: italic;">Nakit akışı, şirketin belirli bir dönemdeki nakit giriş-çıkışlarını ve likidite durumunu gösterir.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Şirket bilgileri alınamadı. Hata: {str(e)}")

###########################
# Gösterge Açıklamaları
###########################
with st.expander("📌 Gösterge Açıklamaları"):
    st.markdown("""
    **1. Kapanış Fiyatı ve Hareketli Ortalamalar**
    - 20, 50 ve 200 günlük hareketli ortalamalar kısa, orta ve uzun vadeli trendleri gösterir.
    - MA20 > MA50 yükseliş; MA20 < MA50 düşüş sinyali verir.
    - MA50 ve MA200 kesişimlerinden **Golden Cross** (yükseliş) ve **Death Cross** (düşüş) oluşabilir.
    
    **2. Prophet Tahmini**
    - Facebook tarafından geliştirilen Prophet modeli, zaman serisi tahmini yapar.
    - MAE, RMSE ve MAPE gibi hata ölçümleri ile tahmin başarısı değerlendirilir.
    - Kapanış fiyatı ile birlikte tahmin çizgisi gösterilir.

    **3. RSI (Göreceli Güç Endeksi)**
    - 70 üzeri aşırı alım, 30 altı aşırı satım sinyali verir.
    - Trend dönüş noktalarının belirlenmesinde kullanılır.

    **4. Bollinger Bantları**
    - Fiyatın volatiliteye göre yukarı/aşağı hareket aralığını gösterir.
    - Fiyatın üst banda yaklaşması aşırı alım; alt banda yaklaşması aşırı satım sinyali olabilir.

    **5. MACD**
    - Kısa ve uzun vadeli hareketli ortalamalar arasındaki farkı izler.
    - MACD ve sinyal çizgisi kesişimlerinden trend dönüşü anlaşılır.
    - Histogram, momentum gücünü yansıtır.

    **6. Hacim ve Hacim Farkı**
    - Hacim, işlem yoğunluğunu gösterir.
    - Hacim farkı, günlük değişimleri ve piyasadaki ani hareketleri belirlemek için kullanılır.

    **7. Fibonacci Retracement Analizi**
    - Belirli bir yükseliş veya düşüş sonrası olası destek ve direnç seviyelerini belirler.
    - %0'dan %100'e kadar düzeltme seviyeleri grafik üzerinde gösterilir.
    - Mevcut fiyatın hangi Fibonacci aralığında yer aldığı belirtilir.

    **8. Ichimoku Cloud (Bulut)**
    - Tenkan-Sen (Kısa Vadeli Trend) ve Kijun-Sen (Orta Vadeli Trend) çizgileri yön ve momentum gösterir.
    - Senkou Span A ve B ile oluşan "bulut", destek ve direnç bölgelerini yansıtır.
    - Fiyat bulutun üstündeyse yükseliş, altındaysa düşüş sinyali alınır.

    **9. Finansal Tablolar**
    - **Bilanço**: Şirketin belirli bir tarihteki varlık, borç ve özkaynak durumunu gösterir.
    - **Gelir Tablosu**: Belirli bir dönemdeki gelir, gider ve kâr/zarar bilgilerini sunar.
    - **Nakit Akışı**: Şirketin faaliyet, yatırım ve finansman kaynaklı nakit giriş/çıkışlarını gösterir.

    **10. Getiri Karşılaştırması (Piyasa Karşılaştırması)**
    - Seçilen hissenin son 1 yıldaki performansı, diğer yatırım araçlarıyla (BIST 100, Dolar, Euro, Altın, Gümüş, Bitcoin) karşılaştırılır.
    - Her varlık için yıllık yüzdelik getiri hesaplanır ve görselleştirilir.
    - Renkli bar grafik sayesinde kullanıcı, alternatif yatırım araçları arasında karşılaştırmalı değerlendirme yapabilir.
    - **Yeşil**: Pozitif getiri, **Kırmızı**: Negatif getiri, **Sarı**: Analiz edilen hisseyi temsil eder.
    """)
st.markdown("---")
st.caption("© 2025 Hisse Analiz Paneli - Tüm hakları saklıdır.")
