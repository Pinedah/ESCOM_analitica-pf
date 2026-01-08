import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from datetime import datetime
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="Factores que Influyen en Precios de Autos", layout="wide")

# Título principal con contexto
st.title("Análisis: Factores que Influyen en el Precio de Venta de Vehículos")
st.markdown("### *Evolución temporal y variables determinantes del precio*")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/vehicle_sales_sample.csv')
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)
    df = df.dropna(subset=['saledate'])
    df['saledate'] = df['saledate'].dt.tz_localize(None)
    df['year_sale'] = df['saledate'].dt.year
    df['month_sale'] = df['saledate'].dt.month
    df['vehicle_age'] = df['year_sale'] - df['year']
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar el dataset: {e}")
    st.stop()

# ====== SIDEBAR: FILTROS GENERALES ======
st.sidebar.header("Filtros Generales")

year_range = st.sidebar.slider(
    "Año del vehículo",
    int(df['year'].min()),
    int(df['year'].max()),
    (int(df['year'].min()), int(df['year'].max()))
)

makes = ['Todos'] + sorted(df['make'].unique().tolist())
selected_make = st.sidebar.selectbox("Marca", makes)

bodies = ['Todos'] + sorted(df['body'].unique().tolist())
selected_body = st.sidebar.selectbox("Tipo de carrocería", bodies)

transmissions = ['Todos'] + sorted(df['transmission'].unique().tolist())
selected_transmission = st.sidebar.selectbox("Transmisión", transmissions)

states = ['Todos'] + sorted(df['state'].unique().tolist())
selected_state = st.sidebar.selectbox("Estado", states)

price_range = st.sidebar.slider(
    "Rango de precio ($)",
    float(df['sellingprice'].min()),
    float(df['sellingprice'].max()),
    (float(df['sellingprice'].min()), float(df['sellingprice'].max()))
)

odometer_range = st.sidebar.slider(
    "Kilometraje",
    float(df['odometer'].min()),
    float(df['odometer'].max()),
    (float(df['odometer'].min()), float(df['odometer'].max()))
)

# Aplicar filtros
filtered_df = df[
    (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
    (df['sellingprice'] >= price_range[0]) & (df['sellingprice'] <= price_range[1]) &
    (df['odometer'] >= odometer_range[0]) & (df['odometer'] <= odometer_range[1])
]

if selected_make != 'Todos':
    filtered_df = filtered_df[filtered_df['make'] == selected_make]
if selected_body != 'Todos':
    filtered_df = filtered_df[filtered_df['body'] == selected_body]
if selected_transmission != 'Todos':
    filtered_df = filtered_df[filtered_df['transmission'] == selected_transmission]
if selected_state != 'Todos':
    filtered_df = filtered_df[filtered_df['state'] == selected_state]

filtered_df_clean = filtered_df.dropna().copy()

st.sidebar.info(f"Registros filtrados: {len(filtered_df):,}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Pregunta Principal:**")
st.sidebar.markdown("*¿Qué factores influyen en el precio de venta y cómo evolucionan en el tiempo?*")

# ====== MÉTRICAS CLAVE ======
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Precio Promedio", f"${filtered_df['sellingprice'].mean():,.0f}")
with col2:
    st.metric("Total Ventas", f"{len(filtered_df):,}")
with col3:
    st.metric("Edad Promedio", f"{filtered_df['vehicle_age'].mean():.1f} años")
with col4:
    st.metric("Kilometraje Prom.", f"{filtered_df['odometer'].mean():,.0f}")
with col5:
    correlacion = filtered_df[['sellingprice', 'odometer']].corr().iloc[0, 1]
    st.metric("Corr. Precio-Km", f"{correlacion:.3f}")

st.markdown("---")

# ====== LAYOUT EN 3 COLUMNAS ======
col1, col2, col3 = st.columns(3)

# ========== COLUMNA 1: FACTORES INTRÍNSECOS DEL VEHÍCULO ==========
with col1:
    st.markdown("### **Factores Intrínsecos**")
    st.caption("*Variables propias del vehículo que determinan su valor*")
    
    # Gráfico 1: Precio vs Kilometraje
    st.markdown("#### 1. Impacto del Kilometraje en Precio")
    st.caption("**¿Por qué?** El kilometraje es el indicador más directo del desgaste. A mayor uso, menor precio de venta.")
    
    # Crear bins para mejor visualización
    filtered_df['odometer_bin'] = pd.cut(filtered_df['odometer'], bins=10)
    price_by_km = filtered_df.groupby('odometer_bin')['sellingprice'].mean().reset_index()
    price_by_km['odometer_mid'] = price_by_km['odometer_bin'].apply(lambda x: x.mid)
    
    fig1 = px.scatter(filtered_df.sample(min(500, len(filtered_df))), 
                      x='odometer', y='sellingprice',
                      opacity=0.4, color='sellingprice',
                      color_continuous_scale='RdYlGn',
                      labels={'odometer': 'Kilometraje', 'sellingprice': 'Precio ($)'})
    fig1.add_scatter(x=price_by_km['odometer_mid'], y=price_by_km['sellingprice'],
                     mode='lines', name='Precio Promedio', line=dict(color='red', width=3))
    fig1.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Depreciación por Edad
    st.markdown("#### 2. Depreciación por Edad")
    st.caption("**¿Por qué?** Muestra la caída del precio promedio conforme aumenta la edad del vehículo (depreciación).")
    
    age_price = filtered_df.groupby('vehicle_age').agg({
        'sellingprice': ['mean', 'count']
    }).reset_index()
    age_price.columns = ['vehicle_age', 'avg_price', 'count']
    age_price = age_price[age_price['count'] >= 10]  # Filtrar grupos pequeños
    
    fig2 = px.line(age_price, x='vehicle_age', y='avg_price',
                   labels={'vehicle_age': 'Edad (años)', 'avg_price': 'Precio Promedio ($)'},
                   markers=True)
    fig2.update_traces(line_color='#1f77b4', line_width=3, marker=dict(size=8))
    fig2.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Precio según Condición
    st.markdown("#### 3. Precio según Condición")
    st.caption("**¿Por qué?** La condición refleja el estado físico. A mejor condición, mayor precio de venta alcanzado.")
    
    # Crear bins de condición
    filtered_df['condition_category'] = pd.cut(filtered_df['condition'], 
                                               bins=[0, 20, 30, 40, 50],
                                               labels=['Malo', 'Regular', 'Bueno', 'Excelente'])
    condition_price = filtered_df.groupby('condition_category')['sellingprice'].mean().reset_index()
    
    fig3 = px.bar(condition_price, x='condition_category', y='sellingprice',
                  labels={'condition_category': 'Condición', 'sellingprice': 'Precio Promedio ($)'},
                  color='sellingprice', color_continuous_scale='Greens',
                  text='sellingprice')
    fig3.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    fig3.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20),
                      showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ========== COLUMNA 2: FACTORES DE MERCADO Y MARCA ==========
with col2:
    st.markdown("### **Factores de Mercado**")
    st.caption("*Variables externas: marca, tipo y ubicación que afectan el precio*")
    
    # Gráfico 4: Top Marcas por Precio
    st.markdown("#### 4. Precio Promedio por Marca")
    st.caption("**¿Por qué?** La marca es clave en el precio. Marcas premium obtienen precios más altos.")
    
    top_makes = filtered_df['make'].value_counts().head(10).index
    make_price = filtered_df[filtered_df['make'].isin(top_makes)].groupby('make').agg({
        'sellingprice': 'mean'
    }).reset_index()
    make_price.columns = ['make', 'avg_price']
    make_price = make_price.sort_values('avg_price', ascending=True)
    
    fig4 = px.bar(make_price, y='make', x='avg_price', orientation='h',
                  labels={'make': 'Marca', 'avg_price': 'Precio Promedio ($)'},
                  color='avg_price', color_continuous_scale='Viridis')
    fig4.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Gráfico 5: Precio por Tipo de Carrocería
    st.markdown("#### 5. Precio por Tipo de Vehículo")
    st.caption("**¿Por qué?** El tipo de carrocería determina el segmento y afecta directamente el precio de venta.")
    
    body_price = filtered_df.groupby('body').agg({
        'sellingprice': ['mean', 'count']
    }).reset_index()
    body_price.columns = ['body', 'avg_price', 'count']
    body_price = body_price[body_price['count'] >= 20]
    body_price = body_price.sort_values('avg_price', ascending=False)
    
    fig5 = px.bar(body_price, x='body', y='avg_price',
                  labels={'body': 'Tipo', 'avg_price': 'Precio Promedio ($)'},
                  color='avg_price', color_continuous_scale='Plasma')
    fig5.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Gráfico 6: Precio por Estado
    st.markdown("#### 6. Precio por Ubicación Geográfica")
    st.caption("**¿Por qué?** Los precios varían según el estado por diferencias en demanda, clima e ingresos locales.")
    
    state_price = filtered_df.groupby('state').agg({
        'sellingprice': ['mean', 'count']
    }).reset_index()
    state_price.columns = ['state', 'avg_price', 'count']
    state_price = state_price[state_price['count'] >= 50]
    state_price = state_price.sort_values('avg_price', ascending=False).head(15)
    
    fig6 = px.bar(state_price, x='state', y='avg_price',
                  labels={'state': 'Estado', 'avg_price': 'Precio Promedio ($)'},
                  color='avg_price', color_continuous_scale='Reds')
    fig6.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      showlegend=False, xaxis={'tickangle': -45})
    st.plotly_chart(fig6, use_container_width=True)

# ========== COLUMNA 3: EVOLUCIÓN TEMPORAL ==========
with col3:
    st.markdown("### **Evolución Temporal**")
    st.caption("*Cómo los precios y factores cambian a lo largo del tiempo*")
    
    # Gráfico 7: Evolución Temporal del Precio
    st.markdown("#### 7. Evolución del Precio en el Tiempo")
    st.caption("**¿Por qué?** Muestra cómo el precio promedio ha cambiado mes a mes, revelando tendencias y ciclos.")
    
    monthly_avg = filtered_df_clean.groupby(filtered_df_clean['saledate'].dt.to_period('M')).agg({
        'sellingprice': 'mean',
        'odometer': 'mean'
    }).reset_index()
    monthly_avg['saledate'] = monthly_avg['saledate'].dt.to_timestamp()
    
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=monthly_avg['saledate'], y=monthly_avg['sellingprice'],
                              mode='lines+markers', name='Precio',
                              line=dict(color='blue', width=2),
                              fill='tozeroy', fillcolor='rgba(0,100,255,0.2)'))
    
    # Media móvil
    monthly_avg['rolling_avg'] = monthly_avg['sellingprice'].rolling(window=3).mean()
    fig7.add_trace(go.Scatter(x=monthly_avg['saledate'], y=monthly_avg['rolling_avg'],
                              mode='lines', name='Tendencia (3 meses)',
                              line=dict(color='red', width=2, dash='dash')))
    
    fig7.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title='Fecha', yaxis_title='Precio ($)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig7, use_container_width=True)
    
    # Gráfico 8: Precio por Transmisión
    st.markdown("#### 8. Precio por Transmisión")
    st.caption("**¿Por qué?** El tipo de transmisión (automática vs manual) influye en el precio por preferencias del mercado y tecnología.")
    
    trans_price = filtered_df.groupby('transmission').agg({
        'sellingprice': ['mean', 'count']
    }).reset_index()
    trans_price.columns = ['transmission', 'avg_price', 'count']
    trans_price = trans_price[trans_price['count'] >= 20]
    
    fig8 = px.bar(trans_price, x='transmission', y='avg_price',
                  labels={'transmission': 'Transmisión', 'avg_price': 'Precio Promedio ($)'},
                  color='avg_price', color_continuous_scale='Teal')
    fig8.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)
    
    # Gráfico 9: Precio Real vs Valor de Referencia
    st.markdown("#### 9. Precio Real vs Valor MMR")
    st.caption("**¿Por qué?** Compara el precio de venta real contra el valor de mercado (MMR). Identifica si se vende arriba o abajo del valor esperado.")
    
    # Calcular diferencia porcentual
    filtered_df['price_diff'] = ((filtered_df['sellingprice'] - filtered_df['mmr']) / filtered_df['mmr'] * 100)
    
    # Agrupar por mes
    monthly_comparison = filtered_df_clean.groupby(filtered_df_clean['saledate'].dt.to_period('M')).agg({
        'sellingprice': 'mean',
        'mmr': 'mean'
    }).reset_index()
    monthly_comparison['saledate'] = monthly_comparison['saledate'].dt.to_timestamp()
    
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=monthly_comparison['saledate'], 
                              y=monthly_comparison['sellingprice'],
                              mode='lines', name='Precio Real',
                              line=dict(color='blue', width=2)))
    fig9.add_trace(go.Scatter(x=monthly_comparison['saledate'], 
                              y=monthly_comparison['mmr'],
                              mode='lines', name='Valor MMR',
                              line=dict(color='orange', width=2, dash='dot')))
    
    fig9.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title='Fecha', yaxis_title='Precio ($)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig9, use_container_width=True)

# ====== SECCIÓN FINAL: MATRIZ DE CORRELACIÓN ======
st.markdown("---")
st.markdown("### **Matriz de Correlación: Relaciones entre Factores de Precio**")
st.caption("**¿Por qué?** Esta matriz identifica qué variables están más fuertemente relacionadas con el precio, confirmando los factores más influyentes.")

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    # Matriz de correlación
    numeric_cols = ['year', 'condition', 'odometer', 'mmr', 'sellingprice', 'vehicle_age']
    corr_data = filtered_df_clean[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=['Año', 'Condición', 'Kilometraje', 'MMR', 'Precio', 'Edad'],
        y=['Año', 'Condición', 'Kilometraje', 'MMR', 'Precio', 'Edad'],
        colorscale='RdBu',
        zmid=0,
        text=corr_data.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 11},
        colorbar=dict(title="Correlación")
    ))
    
    fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

with col2:
    st.markdown("#### **Hallazgos Clave:**")
    
    # Calcular correlaciones con precio
    price_corr = corr_data['sellingprice'].drop('sellingprice').abs().sort_values(ascending=False)
    
    st.markdown("**Variables más influyentes:**")
    for var, corr_val in price_corr.items():
        symbol = "[ALTA]" if corr_val > 0.5 else "[MOD]" if corr_val > 0.3 else "[BAJA]"
        var_name = {
            'mmr': 'Valor MMR',
            'year': 'Año',
            'vehicle_age': 'Edad',
            'odometer': 'Kilometraje',
            'condition': 'Condición'
        }.get(var, var)
        st.markdown(f"{symbol} **{var_name}**: {corr_val:.3f}")
    
    st.markdown("---")
    st.markdown("**Interpretación:**")
    st.markdown("- [ALTA] Alta correlación (>0.5)")
    st.markdown("- [MOD] Moderada (0.3-0.5)")
    st.markdown("- [BAJA] Baja (<0.3)")

with col3:
    # Gráfico de importancia de factores
    st.markdown("#### **Importancia de Factores**")
    
    importance_data = pd.DataFrame({
        'Factor': ['Valor MMR', 'Año', 'Edad', 'Kilometraje', 'Condición'],
        'Correlación': [abs(corr_data.loc['mmr', 'sellingprice']),
                       abs(corr_data.loc['year', 'sellingprice']),
                       abs(corr_data.loc['vehicle_age', 'sellingprice']),
                       abs(corr_data.loc['odometer', 'sellingprice']),
                       abs(corr_data.loc['condition', 'sellingprice'])]
    }).sort_values('Correlación', ascending=True)
    
    fig_importance = px.bar(importance_data, y='Factor', x='Correlación',
                           orientation='h',
                           color='Correlación',
                           color_continuous_scale='RdYlGn',
                           labels={'Correlación': 'Correlación Absoluta'})
    fig_importance.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_importance, use_container_width=True)

# ====== SECCIÓN DE ANÁLISIS AVANZADOS ======
st.markdown("---")
st.markdown("### **Análisis Avanzados: Técnicas Multivariadas**")
st.caption("**Objetivo:** Aplicar técnicas estadísticas avanzadas para entender mejor las relaciones entre variables y el precio.")

col1, col2, col3 = st.columns(3)

# ========== COLUMNA 1: PCA ==========
with col1:
    st.markdown("#### **Análisis de Componentes Principales (PCA)**")
    st.caption("**¿Qué es?** PCA reduce la dimensionalidad identificando las direcciones de máxima varianza en los datos.")
    st.caption("**¿Por qué?** Permite identificar qué combinaciones de variables explican mejor la variabilidad del precio.")
    
    # Preparar datos para PCA
    numeric_features = ['year', 'condition', 'odometer', 'mmr', 'sellingprice', 'vehicle_age']
    pca_data = filtered_df_clean[numeric_features].dropna()
    
    # Normalizar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(pca_data)
    
    # Aplicar PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    # Gráfico 1: Varianza explicada
    st.markdown("**1. Varianza Explicada por Componente**")
    var_exp = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Varianza (%)': pca.explained_variance_ratio_ * 100
    })
    
    fig_pca1 = px.bar(var_exp, x='Componente', y='Varianza (%)',
                      labels={'Varianza (%)': 'Varianza Explicada (%)'},
                      color='Varianza (%)', color_continuous_scale='Blues')
    fig_pca1.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig_pca1, use_container_width=True)
    
    st.caption(f"Los primeros 2 componentes explican **{(pca.explained_variance_ratio_[:2].sum()*100):.1f}%** de la varianza total.")
    
    # Gráfico 2: Loadings (Contribuciones)
    st.markdown("**2. Contribución de Variables**")
    st.caption("Muestra qué variables influyen más en cada componente principal.")
    
    loadings_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=['Año', 'Condición', 'Kilometraje', 'MMR', 'Precio', 'Edad']
    )
    
    fig_pca2 = go.Figure(data=go.Heatmap(
        z=loadings_df.values,
        x=loadings_df.columns,
        y=loadings_df.index,
        colorscale='RdBu',
        zmid=0,
        text=loadings_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Loading")
    ))
    fig_pca2.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_pca2, use_container_width=True)
    
    st.caption("**Interpretación:** Valores cercanos a ±1 indican fuerte influencia en el componente.")
    
    # Gráfico 3: Proyección 2D
    st.markdown("**3. Proyección en 2D (PC1 vs PC2)**")
    st.caption("Visualiza cómo se distribuyen los vehículos según los componentes principales, coloreado por precio.")
    
    sample_size = min(500, len(pca_result))
    indices = np.random.choice(len(pca_result), sample_size, replace=False)
    
    pca_plot_df = pd.DataFrame({
        'PC1': pca_result[indices, 0],
        'PC2': pca_result[indices, 1],
        'Precio': pca_data.iloc[indices]['sellingprice'].values
    })
    
    fig_pca3 = px.scatter(pca_plot_df, x='PC1', y='PC2', color='Precio',
                         color_continuous_scale='Viridis',
                         labels={'Precio': 'Precio ($)'})
    fig_pca3.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_pca3, use_container_width=True)

# ========== COLUMNA 2: MDS ==========
with col2:
    st.markdown("#### **Escalamiento Multidimensional (MDS)**")
    st.caption("**¿Qué es?** MDS proyecta datos en menor dimensión preservando las distancias entre observaciones.")
    st.caption("**¿Por qué?** Revela patrones de similitud entre vehículos basados en precio y características.")
    
    # Preparar datos para MDS
    mds_features = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
    sample_size_mds = min(500, len(filtered_df_clean))
    mds_data = filtered_df_clean[mds_features].sample(n=sample_size_mds, random_state=42)
    
    # Normalizar
    scaler_mds = StandardScaler()
    data_scaled_mds = scaler_mds.fit_transform(mds_data)
    
    # Aplicar MDS
    with st.spinner('Calculando MDS...'):
        mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
        mds_result = mds.fit_transform(data_scaled_mds)
    
    # Gráfico 1: MDS coloreado por precio
    st.markdown("**1. Proyección MDS por Precio**")
    st.caption("Vehículos similares en precio y características aparecen cercanos.")
    
    mds_plot_df = pd.DataFrame({
        'MDS1': mds_result[:, 0],
        'MDS2': mds_result[:, 1],
        'Precio': mds_data['sellingprice'].values,
        'Kilometraje': mds_data['odometer'].values,
        'Año': mds_data['year'].values
    })
    
    fig_mds1 = px.scatter(mds_plot_df, x='MDS1', y='MDS2', color='Precio',
                         color_continuous_scale='Plasma',
                         labels={'Precio': 'Precio ($)'})
    fig_mds1.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_mds1, use_container_width=True)
    
    # Gráfico 2: MDS coloreado por kilometraje
    st.markdown("**2. Proyección MDS por Kilometraje**")
    st.caption("Identifica si el kilometraje genera clusters distintos de vehículos.")
    
    fig_mds2 = px.scatter(mds_plot_df, x='MDS1', y='MDS2', color='Kilometraje',
                         color_continuous_scale='YlOrRd',
                         labels={'Kilometraje': 'Kilometraje'})
    fig_mds2.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_mds2, use_container_width=True)
    
    # Gráfico 3: MDS coloreado por año
    st.markdown("**3. Proyección MDS por Año**")
    st.caption("Muestra cómo vehículos de diferentes años se agrupan en el espacio MDS.")
    
    fig_mds3 = px.scatter(mds_plot_df, x='MDS1', y='MDS2', color='Año',
                         color_continuous_scale='Teal',
                         labels={'Año': 'Año'})
    fig_mds3.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_mds3, use_container_width=True)
    
    st.caption(f"**Stress MDS:** {mds.stress_:.3f} (valores <0.1 indican excelente ajuste)")

# ========== COLUMNA 3: CORRELACIÓN PARCIAL DE PEARSON ==========
with col3:
    st.markdown("#### **Correlación Parcial de Pearson**")
    st.caption("**¿Qué es?** Mide la correlación entre dos variables controlando el efecto de otras variables.")
    st.caption("**¿Por qué?** Identifica relaciones directas con el precio, eliminando efectos confusores.")
    
    st.markdown("**1. Correlación Parcial con Precio**")
    st.caption("Correlación de cada variable con el precio, controlando por las demás variables.")
    
    # Calcular correlaciones parciales
    def partial_corr(df, x, y, control_vars):
        """Calcula la correlación parcial entre x e y controlando por control_vars"""
        # Crear matriz de variables
        vars_list = [x, y] + control_vars
        data = df[vars_list].dropna()
        
        # Calcular matriz de correlación
        corr_matrix = data.corr().values
        
        # Calcular correlación parcial usando la inversa de la matriz
        precision_matrix = np.linalg.inv(corr_matrix)
        
        # La correlación parcial se calcula de la matriz de precisión
        partial_corr_value = -precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])
        
        return partial_corr_value
    
    # Variables a analizar
    vars_to_analyze = ['year', 'condition', 'odometer', 'mmr', 'vehicle_age']
    
    partial_corr_results = []
    for var in vars_to_analyze:
        control_vars = [v for v in vars_to_analyze if v != var]
        try:
            p_corr = partial_corr(filtered_df_clean, var, 'sellingprice', control_vars)
            partial_corr_results.append({
                'Variable': var,
                'Correlación Parcial': p_corr
            })
        except:
            pass
    
    partial_corr_df = pd.DataFrame(partial_corr_results)
    partial_corr_df['Variable'] = partial_corr_df['Variable'].map({
        'year': 'Año',
        'condition': 'Condición',
        'odometer': 'Kilometraje',
        'mmr': 'Valor MMR',
        'vehicle_age': 'Edad'
    })
    partial_corr_df = partial_corr_df.sort_values('Correlación Parcial', key=abs, ascending=False)
    
    fig_partial1 = px.bar(partial_corr_df, y='Variable', x='Correlación Parcial',
                         orientation='h',
                         color='Correlación Parcial',
                         color_continuous_scale='RdBu',
                         range_color=[-1, 1],
                         labels={'Correlación Parcial': 'Correlación Parcial'})
    fig_partial1.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_partial1, use_container_width=True)
    
    st.caption("**Interpretación:** Valores cercanos a ±1 indican fuerte relación directa con el precio.")
    
    # Gráfico 2: Comparación correlación simple vs parcial
    st.markdown("**2. Comparación: Simple vs Parcial**")
    st.caption("Diferencia entre correlación simple y parcial revela efectos indirectos.")
    
    simple_corr = []
    for var in vars_to_analyze:
        corr_val = filtered_df_clean[[var, 'sellingprice']].corr().iloc[0, 1]
        simple_corr.append(corr_val)
    
    comparison_df = pd.DataFrame({
        'Variable': partial_corr_df['Variable'].values,
        'Simple': simple_corr,
        'Parcial': partial_corr_df['Correlación Parcial'].values
    })
    
    fig_partial2 = go.Figure()
    fig_partial2.add_trace(go.Bar(
        y=comparison_df['Variable'],
        x=comparison_df['Simple'],
        name='Correlación Simple',
        orientation='h',
        marker=dict(color='lightblue')
    ))
    fig_partial2.add_trace(go.Bar(
        y=comparison_df['Variable'],
        x=comparison_df['Parcial'],
        name='Correlación Parcial',
        orientation='h',
        marker=dict(color='darkblue')
    ))
    fig_partial2.update_layout(
        barmode='group',
        height=220,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title='Correlación',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_partial2, use_container_width=True)
    
    # Tabla resumen
    st.markdown("**3. Tabla de Resultados**")
    st.caption("Resumen de correlaciones simples y parciales con el precio.")
    
    comparison_styled = comparison_df.copy()
    comparison_styled['Simple'] = comparison_styled['Simple'].apply(lambda x: f"{x:.3f}")
    comparison_styled['Parcial'] = comparison_styled['Parcial'].apply(lambda x: f"{x:.3f}")
    comparison_styled['Diferencia'] = (comparison_df['Simple'] - comparison_df['Parcial']).apply(lambda x: f"{x:.3f}")
    
    st.dataframe(comparison_styled, use_container_width=True, hide_index=True)
    
    st.caption("**Nota:** Gran diferencia entre simple y parcial indica efectos indirectos (mediados por otras variables).")

# Footer con conclusiones
st.markdown("---")
st.markdown("""
### **Conclusiones sobre Factores que Influyen en el Precio:**

1. **Factor más determinante:** El valor MMR (Manheim Market Report) muestra la correlación más alta, validando que el mercado sigue referencias establecidas.

2. **Depreciación:** La edad del vehículo y el kilometraje son factores negativos claros - a mayor edad/uso, menor precio.

3. **Marca y tipo:** Ciertos fabricantes y tipos de carrocería (SUV, Sedan) mantienen mejor su valor.

4. **Evolución temporal:** Los precios muestran tendencias y estacionalidad, indicando momentos óptimos de compra/venta.

5. **Condición física:** El estado del vehículo impacta significativamente, siendo un factor controlable por el vendedor.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Optimizado")
st.sidebar.markdown("*Análisis completo en una página*")
st.sidebar.markdown("**Creado con Streamlit**")
