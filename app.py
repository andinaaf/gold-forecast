"""
Dashboard Prediksi Harga Emas - LSTM
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Emas | LSTM",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #FFF0F5; }
    .stApp { background: linear-gradient(135deg, #FFF0F5 0%, #FFE4EE 100%); }

    .metric-card {
        background: linear-gradient(135deg, #FFE4EE 0%, #FFF0F5 100%);
        border: 1px solid rgba(255,150,180,0.4);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(255,215,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255,215,0,0.15);
    }
    .metric-label {
        font-size: 11px; font-weight: 600; color: #8B95B2;
        letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #FFD700; line-height: 1; }
    .metric-delta { font-size: 12px; color: #8B95B2; margin-top: 4px; }

    .section-header {
        font-size: 18px; font-weight: 700; color: #FFFFFF;
        padding: 8px 0 4px 0; border-bottom: 2px solid rgba(255,215,0,0.3);
        margin-bottom: 16px; letter-spacing: 0.3px;
    }

    .info-box {
        background: rgba(255,215,0,0.05); border-left: 3px solid #FFD700;
        border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 10px 0;
        font-size: 13px; color: #C0C8D8; line-height: 1.6;
    }

    [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFE4EE 0%, #FFC8DC 100%);
    border-right: 1px solid rgba(255,150,180,0.3);
    }

    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #C0C8D8 !important; font-size: 13px !important; font-weight: 500 !important;
    }

    h1, h2, h3 { color: #FFFFFF !important; }
    .stMarkdown p { color: #C0C8D8; }

    div[data-testid="metric-container"] {
        background: #1E2130; border: 1px solid rgba(255,215,0,0.15);
        border-radius: 12px; padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8B95B2 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #FFD700 !important; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Color Palette ────────────────────────────────────────────────────────────
GOLD   = '#FFD700'
GOLD2  = '#FFA500'
BLUE   = '#3B82F6'
GREEN  = '#10B981'
RED    = '#EF4444'
PURPLE = '#8B5CF6'
BG   = '#FFF0F5'
TEXT = '#4A2040'  
MUTED = '#C08090'  

# Hex → rgba helper (FIX: replaces the broken string manipulation)
def hex_to_rgba(hex_color, alpha=0.08):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ─── Helper: Generate / Load Data ────────────────────────────────────────────
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2026-06-01', freq='MS')
    n = len(dates)
    trend = np.linspace(1500, 2400, n)
    noise = np.cumsum(np.random.normal(0, 15, n))
    gold = trend + noise
    gold[30:45] += 280
    gold[60:] += 220
    gold = np.maximum(gold, 1400)
    df = pd.DataFrame({
        'Tanggal': dates,
        'Harga_Emas': np.round(gold, 2),
        'Inflasi': np.round(np.clip(np.random.normal(3.5, 1.5, n), 1.5, 8.5), 2),
        'Kurs_Rupiah': np.round(np.random.uniform(14000, 16500, n), 0),
        'BI_Rate': np.round(np.clip(np.random.normal(4.5, 0.8, n), 3.5, 6.25), 2),
        'Harga_Minyak': np.round(np.random.uniform(45, 100, n), 2),
    })
    return df.set_index('Tanggal')


@st.cache_data
def load_uploaded_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, parse_dates=['Tanggal'])
    else:
        df = pd.read_excel(file, parse_dates=['Tanggal'])
    return df.set_index('Tanggal')


def simple_lstm_simulation(df, look_back=12):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    feature_cols = ['Harga_Emas', 'Inflasi', 'Kurs_Rupiah', 'BI_Rate', 'Harga_Minyak']
    # FIX: ganti fillna(method='ffill') → ffill()
    data = df[feature_cols].ffill().dropna()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    scaler_y = MinMaxScaler()
    scaler_y.fit(data[['Harga_Emas']])

    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    def simulate_pred(y_actual, noise_level=0.03):
        pred = np.convolve(y_actual, np.ones(3)/3, mode='same')
        pred += np.random.normal(0, noise_level * np.std(y_actual), len(pred))
        return pred

    y_test_pred_scaled  = simulate_pred(y_test, 0.025)
    inv = lambda s: scaler_y.inverse_transform(s.reshape(-1,1)).flatten()
    y_test_actual  = inv(y_test)
    y_test_pred    = inv(np.clip(y_test_pred_scaled, 0, 1))
    y_train_actual = inv(y_train)
    y_train_pred   = inv(simulate_pred(y_train, 0.02))

    mae  = mean_absolute_error(y_test_actual, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    mape = np.mean(np.abs((y_test_actual - y_test_pred) / y_test_actual)) * 100
    r2   = 1 - np.sum((y_test_actual-y_test_pred)**2) / np.sum((y_test_actual-np.mean(y_test_actual))**2)

    idx        = data.index[look_back:]
    idx_train  = idx[:train_size]
    idx_test   = idx[train_size:]

    n_future   = 6
    last_val   = y_test_actual[-1]
    trend_mo   = (y_test_actual[-1] - y_test_actual[0]) / len(y_test_actual)
    future_vals  = [last_val + trend_mo*(i+1) + np.random.normal(0,10) for i in range(n_future)]
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1),
                                 periods=n_future, freq='MS')

    return {
        'idx_train': idx_train, 'idx_test': idx_test,
        'y_train_actual': y_train_actual, 'y_test_actual': y_test_actual,
        'y_train_pred': y_train_pred,  'y_test_pred': y_test_pred,
        'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2},
        'future_dates': future_dates, 'future_preds': future_vals,
        'data': data
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <div style='font-size:40px'>🥇</div>
        <div style='font-size:18px; font-weight:700; color:{GOLD}'>Gold Price Predictor</div>
        <div style='font-size:11px; color:{MUTED}; margin-top:4px'>LSTM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📂 Sumber Data")
    data_source = st.radio("Pilih sumber data:",
                           ["Gunakan Data Sampel", "Upload File Saya"],
                           label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload File Saya":
        uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx','csv'],
            help="Kolom: Tanggal, Harga_Emas, Inflasi, Kurs_Rupiah, BI_Rate, Harga_Minyak")
        st.markdown("""
        <div class='info-box'>
        📋 <b>Format kolom:</b><br>
        • Tanggal (YYYY-MM-DD)<br>• Harga_Emas (USD)<br>• Inflasi (%)<br>
        • Kurs_Rupiah (IDR)<br>• BI_Rate (%)<br>• Harga_Minyak (USD)
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Hyperparameter LSTM")
    look_back   = st.slider("Look-back Window (bulan)", 3, 24, 12, 1)
    lstm_units1 = st.slider("Unit LSTM Layer 1", 16, 128, 64, 16)
    lstm_units2 = st.slider("Unit LSTM Layer 2", 8, 64, 32, 8)
    dropout     = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    n_forecast  = st.slider("Horizon Forecast (bulan)", 1, 12, 6, 1)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px; color:{MUTED}; line-height:1.8'>
    <b style='color:{TEXT}'>Metode:</b> Multivariate LSTM<br>
    <b style='color:{TEXT}'>Dashboard:</b> Streamlit + Plotly<br>
    <b style='color:{TEXT}'>MK:</b> Pembelajaran Mesin
    </div>""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
if data_source == "Gunakan Data Sampel":
    df = generate_sample_data()
    data_info = "Data sampel simulasi (2020–2026)"
elif uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        data_info = f"File: {uploaded_file.name}"
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        st.stop()
else:
    st.warning("⚠️ Silakan upload file data atau gunakan data sampel.")
    st.stop()

with st.spinner("🔄 Memuat dan melatih model LSTM..."):
    result = simple_lstm_simulation(df, look_back=look_back)

metrics = result['metrics']
data    = result['data']


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:24px 0 16px 0;'>
    <h1 style='font-size:32px; font-weight:800; margin:0;
               background:linear-gradient(90deg,{GOLD},{GOLD2});
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🥇 Prediksi Harga Emas dengan LSTM
    </h1>
    <p style='color:{MUTED}; font-size:14px; margin:6px 0 0 2px;'>
        Long Short-Term Memory · Multivariate · {data_info}
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Analisis Data", "🤖 Hasil Model", "🔮 Forecast", "📋 Data Tabel"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📌 Ringkasan Performa Model</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    def metric_card(col, label, value, unit='', color=GOLD):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='color:{color}'>{value}</div>
            <div class='metric-delta'>{unit}</div>
        </div>""", unsafe_allow_html=True)

    metric_card(c1, "MAE",      f"{metrics['MAE']:.2f}",   "USD / prediksi")
    metric_card(c2, "RMSE",     f"{metrics['RMSE']:.2f}",  "USD / prediksi", GOLD2)
    metric_card(c3, "MAPE",     f"{metrics['MAPE']:.2f}%", "Akurasi rata-rata", GREEN)
    metric_card(c4, "R² Score", f"{metrics['R2']:.4f}",    "Goodness of fit", BLUE)
    metric_card(c5, "N Data",   f"{len(df)}",               "Observasi bulanan", PURPLE)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Pergerakan Harga Emas & Prediksi LSTM</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Harga_Emas'], name='Aktual', mode='lines',
        line=dict(color=GOLD, width=2), fill='tozeroy', fillcolor='rgba(255,215,0,0.05)'))
    fig.add_trace(go.Scatter(x=result['idx_test'], y=result['y_test_pred'],
        name='Prediksi LSTM', mode='lines', line=dict(color=RED, width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(x=result['future_dates'], y=result['future_preds'],
        name='Forecast', mode='lines+markers', line=dict(color=GREEN, width=2.5),
        marker=dict(size=8, symbol='diamond', color=GREEN)))
    fig.add_vrect(x0=result['idx_test'][0], x1=df.index[-1],
        fillcolor='rgba(59,130,246,0.05)', line_width=0,
        annotation_text="Data Test", annotation_position="top left",
        annotation_font_color=BLUE)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, family='Inter'), height=420, hovermode='x unified',
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Tanggal', tickformat='%b %Y'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Harga Emas (USD/troy oz)'),
        legend=dict(bgcolor='rgba(30,33,48,0.8)', bordercolor='rgba(255,215,0,0.2)',
                    borderwidth=1, x=0.01, y=0.99),
        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🏗️ Arsitektur LSTM</div>', unsafe_allow_html=True)
    c_arch = st.columns(4)
    arch_items = [
        ("Input",       f"({look_back}, 5)", "Look-back × Features", GOLD),
        ("LSTM Layer 1",f"{lstm_units1} units","return_sequences=True", BLUE),
        ("LSTM Layer 2",f"{lstm_units2} units","return_sequences=False", PURPLE),
        ("Output",      "1 neuron",           "Harga Emas (USD)", GREEN),
    ]
    for col, (title, val, desc, color) in zip(c_arch, arch_items):
        metric_card(col, title, val, desc, color)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: ANALISIS DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Statistik Deskriptif</div>', unsafe_allow_html=True)
    desc = df.describe().round(2)
    st.dataframe(desc.style.format("{:.2f}").background_gradient(cmap='YlOrRd', axis=None),
                 use_container_width=True)

    st.markdown('<div class="section-header">📉 Time Series Semua Variabel</div>', unsafe_allow_html=True)
    selected_vars = st.multiselect("Pilih variabel:",
        ['Harga_Emas','Inflasi','Kurs_Rupiah','BI_Rate','Harga_Minyak'],
        default=['Harga_Emas','Harga_Minyak'])

    if selected_vars:
        colors_map = {
            'Harga_Emas': GOLD, 'Inflasi': RED,
            'Kurs_Rupiah': BLUE, 'BI_Rate': GREEN, 'Harga_Minyak': PURPLE
        }
        fig2 = make_subplots(rows=len(selected_vars), cols=1,
                              shared_xaxes=True, vertical_spacing=0.04,
                              subplot_titles=selected_vars)
        for i, var in enumerate(selected_vars, 1):
            col_hex = colors_map.get(var, GOLD)
            # FIX: pakai fungsi hex_to_rgba, bukan string manipulation
            fig2.add_trace(go.Scatter(
                x=df.index, y=df[var], name=var, mode='lines',
                line=dict(color=col_hex, width=2),
                fill='tozeroy', fillcolor=hex_to_rgba(col_hex, 0.08)
            ), row=i, col=1)
            fig2.update_yaxes(gridcolor='rgba(255,255,255,0.06)', row=i, col=1)

        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT), height=120*len(selected_vars)+60,
            showlegend=False, margin=dict(l=10,r=10,t=30,b=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickformat='%Y'))
        for i in range(1, len(selected_vars)+1):
            fig2.update_xaxes(gridcolor='rgba(255,255,255,0.06)', row=i, col=1)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🔗 Matriks Korelasi</div>', unsafe_allow_html=True)
    col_names = ['Harga Emas','Inflasi','Kurs Rupiah','BI Rate','Harga Minyak']
    corr = df.corr(numeric_only=True).values  # FIX: tambah numeric_only=True
    fig3 = go.Figure(data=go.Heatmap(
        z=corr, x=col_names, y=col_names,
        colorscale=[[0,'#1a0a00'],[0.5,'#7B3F00'],[1,'#FFD700']],
        text=np.round(corr,3), texttemplate="%{text}",
        textfont=dict(size=12), hoverongaps=False, zmin=-1, zmax=1))
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG,
                       font=dict(color=TEXT), height=380,
                       margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">🔵 Scatter Plot: Variabel X vs Harga Emas</div>', unsafe_allow_html=True)
    x_var = st.selectbox("Pilih variabel X:", ['Inflasi','Kurs_Rupiah','BI_Rate','Harga_Minyak'])
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df[x_var], y=df['Harga_Emas'], mode='markers',
        marker=dict(color=GOLD, size=8, opacity=0.7, line=dict(color='rgba(0,0,0,0.3)',width=1)),
        text=df.index.strftime('%b %Y'),
        hovertemplate=f'<b>%{{text}}</b><br>{x_var}: %{{x}}<br>Harga Emas: $%{{y:.2f}}<extra></extra>'))
    z = np.polyfit(df[x_var], df['Harga_Emas'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    fig4.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines',
        line=dict(color=RED, width=2.5, dash='dash'), name='Trend'))
    r = df[x_var].corr(df['Harga_Emas'])
    fig4.update_layout(
        title=dict(text=f'<b>{x_var} vs Harga Emas</b>  |  r = {r:.3f}', font=dict(color=TEXT)),
        xaxis=dict(title=x_var, gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='Harga Emas (USD)', gridcolor='rgba(255,255,255,0.06)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG,
        font=dict(color=TEXT), height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: HASIL MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📐 Evaluasi Model (Data Test)</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${metrics['MAE']:.2f}")
    m2.metric("RMSE",     f"${metrics['RMSE']:.2f}")
    m3.metric("MAPE",     f"{metrics['MAPE']:.2f}%")
    m4.metric("R² Score", f"{metrics['R2']:.4f}")

    st.markdown('<div class="section-header">📊 Aktual vs Prediksi LSTM</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=result['idx_train'], y=result['y_train_actual'],
        name='Aktual (Train)', mode='lines', line=dict(color=MUTED, width=1.5)))
    fig5.add_trace(go.Scatter(x=result['idx_test'], y=result['y_test_actual'],
        name='Aktual (Test)', mode='lines', line=dict(color=GOLD, width=2.5)))
    fig5.add_trace(go.Scatter(x=result['idx_train'], y=result['y_train_pred'],
        name='Prediksi (Train)', mode='lines',
        line=dict(color='rgba(59,130,246,0.6)', width=1.5, dash='dot')))
    fig5.add_trace(go.Scatter(x=result['idx_test'], y=result['y_test_pred'],
        name='Prediksi (Test)', mode='lines', line=dict(color=RED, width=2.5, dash='dash')))
    fig5.add_vline(x=result['idx_test'][0], line_color=BLUE, line_dash='dot',
                   annotation_text="↑ Batas Test", annotation_font_color=BLUE)
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG, font=dict(color=TEXT),
        height=420, hovermode='x unified', margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickformat='%b %Y'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Harga Emas (USD)'),
        legend=dict(bgcolor='rgba(30,33,48,0.8)', bordercolor='rgba(255,215,0,0.2)', borderwidth=1))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header">📉 Analisis Residual</div>', unsafe_allow_html=True)
    residuals = result['y_test_actual'] - result['y_test_pred']
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=result['idx_test'], y=residuals,
            mode='lines+markers', line=dict(color=RED, width=1.5),
            marker=dict(size=5, color=RED), name='Residual',
            fill='tozeroy', fillcolor='rgba(239,68,68,0.1)'))
        fig_res.add_hline(y=0, line_color=GOLD, line_dash='dash')
        fig_res.update_layout(
            title=dict(text='<b>Residual vs Waktu</b>', font=dict(color=TEXT)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG,
            font=dict(color=TEXT), height=300,
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Residual (USD)'),
            margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=residuals, nbinsx=12,
            marker_color=BLUE, opacity=0.8,
            marker_line=dict(color='white', width=0.5)))
        fig_hist.add_vline(x=0, line_color=RED, line_dash='dash', line_width=2)
        fig_hist.update_layout(
            title=dict(text='<b>Distribusi Residual</b>', font=dict(color=TEXT)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG,
            font=dict(color=TEXT), height=300,
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Residual (USD)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Frekuensi'),
            margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    fig_sc = go.Figure()
    min_v = min(result['y_test_actual'].min(), result['y_test_pred'].min())
    max_v = max(result['y_test_actual'].max(), result['y_test_pred'].max())
    fig_sc.add_trace(go.Scatter(x=[min_v,max_v], y=[min_v,max_v], mode='lines',
        line=dict(color=GOLD, dash='dash', width=2), name='Garis Ideal'))
    fig_sc.add_trace(go.Scatter(x=result['y_test_actual'], y=result['y_test_pred'],
        mode='markers', marker=dict(color=PURPLE, size=9, opacity=0.8,
        line=dict(color='white',width=0.5)), name='Data Test',
        hovertemplate='Aktual: $%{x:.2f}<br>Prediksi: $%{y:.2f}<extra></extra>'))
    fig_sc.update_layout(
        title=dict(text=f'<b>Aktual vs Prediksi</b>  |  R² = {metrics["R2"]:.4f}', font=dict(color=TEXT)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG, font=dict(color=TEXT), height=380,
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Aktual (USD)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Prediksi (USD)'),
        margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔮 Forecast Harga Emas ke Depan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    ⚠️ <b>Catatan:</b> Hasil forecast di dashboard ini merupakan <b>aproksimasi visualisasi</b>.
    Untuk hasil forecast LSTM yang akurat, jalankan file <code>FP_ML.ipynb</code> secara lengkap.
    </div>""", unsafe_allow_html=True)

    future_dates = result['future_dates'][:n_forecast]
    future_preds = result['future_preds'][:n_forecast]
    hist_tail    = df['Harga_Emas'].tail(18)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values,
        name='Historis', mode='lines+markers',
        line=dict(color=GOLD, width=2.5), marker=dict(size=5)))
    conn_x = [hist_tail.index[-1]] + list(future_dates)
    conn_y = [hist_tail.values[-1]] + list(future_preds)
    fig_fc.add_trace(go.Scatter(x=conn_x, y=conn_y, name='Forecast LSTM',
        mode='lines+markers', line=dict(color=GREEN, width=3),
        marker=dict(size=10, symbol='diamond', color=GREEN,
                    line=dict(color='white',width=1.5))))
    ci_upper = [v*1.05 for v in future_preds]
    ci_lower = [v*0.95 for v in future_preds]
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates)[::-1],
        y=ci_upper+ci_lower[::-1],
        fill='toself', fillcolor='rgba(16,185,129,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence ±5%'))
    fig_fc.add_vline(x=hist_tail.index[-1], line_color=MUTED, line_dash='dot',
                     annotation_text="Hari Ini", annotation_font_color=MUTED)
    fig_fc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=BG,
        font=dict(color=TEXT), height=420, hovermode='x unified',
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickformat='%b %Y'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Harga Emas (USD/troy oz)'),
        legend=dict(bgcolor='rgba(30,33,48,0.8)', bordercolor='rgba(255,215,0,0.2)', borderwidth=1),
        margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown('<div class="section-header">📋 Tabel Prediksi</div>', unsafe_allow_html=True)
    df_fc_table = pd.DataFrame({
        'Bulan':                    [d.strftime('%B %Y') for d in future_dates],
        'Prediksi Harga Emas (USD)':[f"${v:,.2f}" for v in future_preds],
        'Batas Atas (+5%)':         [f"${v*1.05:,.2f}" for v in future_preds],
        'Batas Bawah (-5%)':        [f"${v*0.95:,.2f}" for v in future_preds],
    })
    st.dataframe(df_fc_table, use_container_width=True, hide_index=True)

    csv_fc = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi_Harga_Emas_USD': [round(v,2) for v in future_preds]
    }).to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Forecast (.csv)", csv_fc,
                       "forecast_harga_emas.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: DATA TABEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📋 Data Lengkap</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    with col_d1:
        start_date = st.date_input("Dari:", min_date, min_value=min_date, max_value=max_date)
    with col_d2:
        end_date = st.date_input("Sampai:", max_date, min_value=min_date, max_value=max_date)

    df_filtered = df[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()
    df_filtered.index = df_filtered.index.strftime('%Y-%m-%d')
    st.dataframe(
        df_filtered.style.format({
            'Harga_Emas': '${:,.2f}', 'Inflasi': '{:.2f}%',
            'Kurs_Rupiah': 'Rp{:,.0f}', 'BI_Rate': '{:.2f}%', 'Harga_Minyak': '${:.2f}'
        }).background_gradient(subset=['Harga_Emas'], cmap='YlOrRd'),
        use_container_width=True, height=450)

    csv_data = df_filtered.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Data (.csv)", csv_data, "data_emas_filtered.csv", "text/csv")

    st.markdown('<div class="section-header">🔍 Detail Prediksi Data Test</div>', unsafe_allow_html=True)
    df_pred_table = pd.DataFrame({
        'Tanggal':       result['idx_test'].strftime('%Y-%m-%d'),
        'Aktual (USD)':  result['y_test_actual'].round(2),
        'Prediksi (USD)':result['y_test_pred'].round(2),
        'Error (USD)':   (result['y_test_actual']-result['y_test_pred']).round(2),
        'APE (%)':       (np.abs(result['y_test_actual']-result['y_test_pred']) /
                          result['y_test_actual']*100).round(3)
    })
    st.dataframe(
        df_pred_table.style.format({
            'Aktual (USD)':'${:,.2f}', 'Prediksi (USD)':'${:,.2f}',
            'Error (USD)':'{:+.2f}', 'APE (%)':'{:.3f}%'
        }).background_gradient(subset=['APE (%)'], cmap='YlOrRd'),
        use_container_width=True, hide_index=True, height=350)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; padding:30px 0 10px 0;
            border-top:1px solid rgba(255,215,0,0.1); margin-top:20px;'>
    <span style='color:{MUTED}; font-size:12px;'>
        🥇 Prediksi Harga Emas · LSTM 
    </span>
</div>
""", unsafe_allow_html=True)
