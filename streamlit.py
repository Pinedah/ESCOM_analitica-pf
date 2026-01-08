import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Dashboard de Precios de Autos", layout="wide")

# Título principal
st.title("Dashboard de Análisis de Precios de Autos")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('car_prices_clean.csv')
    # Convertir saledate a datetime con manejo de errores y UTC
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)
    
    # Eliminar filas donde saledate no pudo ser convertido
    df = df.dropna(subset=['saledate'])
    
    # Convertir a timezone naive para evitar problemas
    df['saledate'] = df['saledate'].dt.tz_localize(None)
    
    # Extraer componentes temporales
    df['year_sale'] = df['saledate'].dt.year
    df['month_sale'] = df['saledate'].dt.month
    df['day_of_week'] = df['saledate'].dt.day_name()
    
    return df

# Cargar datos
try:
    df = load_data()
    st.sidebar.success(f"Dataset cargado: {len(df):,} registros")
except Exception as e:
    st.error(f"Error al cargar el dataset: {e}")
    st.stop()

# ====== FILTROS GENERALES EN SIDEBAR ======
st.sidebar.header("Filtros Generales")

# Filtro por año del vehículo
year_range = st.sidebar.slider(
    "Año del vehículo",
    int(df['year'].min()),
    int(df['year'].max()),
    (int(df['year'].min()), int(df['year'].max()))
)

# Filtro por marca
makes = ['Todos'] + sorted(df['make'].unique().tolist())
selected_make = st.sidebar.selectbox("Marca", makes)

# Filtro por tipo de carrocería
bodies = ['Todos'] + sorted(df['body'].unique().tolist())
selected_body = st.sidebar.selectbox("Tipo de carrocería", bodies)

# Filtro por transmisión
transmissions = ['Todos'] + sorted(df['transmission'].unique().tolist())
selected_transmission = st.sidebar.selectbox("Transmisión", transmissions)

# Filtro por estado
states = ['Todos'] + sorted(df['state'].unique().tolist())
selected_state = st.sidebar.selectbox("Estado", states)

# Filtro por rango de precio
price_range = st.sidebar.slider(
    "Rango de precio ($)",
    float(df['sellingprice'].min()),
    float(df['sellingprice'].max()),
    (float(df['sellingprice'].min()), float(df['sellingprice'].max()))
)

# Filtro por kilometraje
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

filtered_df_clean = filtered_df.dropna().copy()
filtered_df_clean = filtered_df_clean.sort_values('saledate')


if selected_make != 'Todos':
    filtered_df = filtered_df[filtered_df['make'] == selected_make]
if selected_body != 'Todos':
    filtered_df = filtered_df[filtered_df['body'] == selected_body]
if selected_transmission != 'Todos':
    filtered_df = filtered_df[filtered_df['transmission'] == selected_transmission]
if selected_state != 'Todos':
    filtered_df = filtered_df[filtered_df['state'] == selected_state]

st.sidebar.info(f"Registros filtrados: {len(filtered_df):,}")

# ====== PESTAÑAS ======
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen del Dataset", 
    "Series de Tiempo", 
    "Matriz de Correlación",
    "MDS",
    "PCA",
    "Resumen Temporal"
])

# ====== PESTAÑA 1: RESUMEN DEL DATASET ======
with tab1:
    st.header("Resumen del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de registros", f"{len(filtered_df):,}")
    with col2:
        st.metric("Precio promedio", f"${filtered_df['sellingprice'].mean():,.2f}")
    with col3:
        st.metric("Kilometraje promedio", f"{filtered_df['odometer'].mean():,.0f}")
    with col4:
        st.metric("Año promedio", f"{filtered_df['year'].mean():.0f}")
    
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(filtered_df.describe(), use_container_width=True)
    
    st.subheader("Distribución de Variables Numéricas")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(filtered_df, x='sellingprice', nbins=50, 
                          title='Distribución de Precios de Venta')
        fig.update_layout(xaxis_title="Precio de Venta ($)", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.histogram(filtered_df, x='year', nbins=30, 
                          title='Distribución por Año del Vehículo')
        fig.update_layout(xaxis_title="Año", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(filtered_df, x='odometer', nbins=50, 
                          title='Distribución de Kilometraje')
        fig.update_layout(xaxis_title="Kilometraje", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.histogram(filtered_df, x='condition', nbins=20, 
                          title='Distribución de Condición')
        fig.update_layout(xaxis_title="Condición", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Marcas por Cantidad")
    top_makes = filtered_df['make'].value_counts().head(10)
    fig = px.bar(x=top_makes.values, y=top_makes.index, orientation='h',
                 title='Top 10 Marcas', labels={'x': 'Cantidad', 'y': 'Marca'})
    st.plotly_chart(fig, use_container_width=True)

# ====== PESTAÑA 2: SERIES DE TIEMPO ======
with tab2:
    st.header("Series de Tiempo")
    
    # Precio promedio por mes
    st.subheader("Evolución del Precio Promedio en el Tiempo")
    monthly_avg = filtered_df_clean.groupby(filtered_df_clean['saledate'].dt.to_period('M'))['sellingprice'].mean()
    monthly_avg.index = monthly_avg.index.to_timestamp()
    fig.update_xaxes(range=[monthly_avg.index.min(), monthly_avg.index.max()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, 
                            mode='lines+markers', name='Precio Promedio'))
    fig.update_layout(
        title='Precio Promedio de Venta por Mes',
        xaxis_title='Fecha',
        yaxis_title='Precio Promedio ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volumen de ventas por mes
    st.subheader("Volumen de Ventas en el Tiempo")
    monthly_count = filtered_df.groupby(filtered_df['saledate'].dt.to_period('M')).size()
    monthly_count.index = monthly_count.index.to_timestamp()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_count.index, y=monthly_count.values, 
                        name='Volumen de Ventas'))
    fig.update_layout(
        title='Volumen de Ventas por Mes',
        xaxis_title='Fecha',
        yaxis_title='Número de Ventas',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Series de tiempo por marca (top 5)
    st.subheader("Precio Promedio por Marca (Top 5)")
    top5_makes = filtered_df['make'].value_counts().head(5).index
    
    fig = go.Figure()
    for make in top5_makes:
        make_data = filtered_df[filtered_df['make'] == make]
        monthly_make = make_data.groupby(make_data['saledate'].dt.to_period('M'))['sellingprice'].mean()
        monthly_make.index = monthly_make.index.to_timestamp()
        fig.add_trace(go.Scatter(x=monthly_make.index, y=monthly_make.values, 
                                mode='lines+markers', name=make))
    
    fig.update_layout(
        title='Evolución del Precio por Marca',
        xaxis_title='Fecha',
        yaxis_title='Precio Promedio ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ====== PESTAÑA 3: MATRIZ DE CORRELACIÓN ======
with tab3:
    st.header("Matriz de Correlación")
    
    # Seleccionar solo columnas numéricas
    numeric_cols = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
    corr_data = filtered_df_clean[numeric_cols].corr()
    
    # Crear heatmap con plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title='Matriz de Correlación de Variables Numéricas',
        width=800,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Análisis de Correlaciones Significativas")
    
    # Mostrar las correlaciones más fuertes
    corr_pairs = []
    for i in range(len(corr_data.columns)):
        for j in range(i+1, len(corr_data.columns)):
            corr_pairs.append({
                'Variable 1': corr_data.columns[i],
                'Variable 2': corr_data.columns[j],
                'Correlación': corr_data.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlación', key=abs, ascending=False)
    st.dataframe(corr_df, use_container_width=True)
    
    # Scatterplots de las correlaciones más fuertes
    st.subheader("Visualización de Correlaciones")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_df, x='odometer', y='sellingprice', 
                        title='Precio vs Kilometraje',
                        trendline="ols", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(filtered_df, x='year', y='sellingprice', 
                        title='Precio vs Año',
                        trendline="ols", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

# ====== PESTAÑA 4: MDS ======
with tab4:
    st.header("Análisis MDS (Multidimensional Scaling)")
    
    st.info("MDS reduce la dimensionalidad preservando las distancias entre observaciones")
    
    # Preparar datos para MDS
    numeric_features = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
    
    # Tomar una muestra si el dataset es muy grande
    sample_size = min(1000, len(filtered_df))
    mds_data = filtered_df_clean[numeric_features].sample(n=min(1000, len(filtered_df_clean)), random_state=42)
    
    # Normalizar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(mds_data)
    
    # Aplicar MDS
    with st.spinner('Calculando MDS...'):
        mds = MDS(n_components=2, random_state=42)
        mds_result = mds.fit_transform(data_scaled)
    
    # Crear DataFrame con resultados
    mds_df = pd.DataFrame({
        'MDS1': mds_result[:, 0],
        'MDS2': mds_result[:, 1],
        'precio': mds_data['sellingprice'].values,
        'año': mds_data['year'].values,
        'kilometraje': mds_data['odometer'].values
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(mds_df, x='MDS1', y='MDS2', color='precio',
                        title='MDS - Coloreado por Precio',
                        color_continuous_scale='Viridis',
                        labels={'precio': 'Precio ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(mds_df, x='MDS1', y='MDS2', color='año',
                        title='MDS - Coloreado por Año',
                        color_continuous_scale='Plasma',
                        labels={'año': 'Año'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Stress del MDS", f"{mds.stress_:.2f}")
    st.caption("Un stress menor indica mejor preservación de las distancias originales")

# ====== PESTAÑA 5: PCA ======
with tab5:
    st.header("Análisis de Componentes Principales (PCA)")
    
    st.info("PCA identifica las direcciones de máxima varianza en los datos")
    
    # Preparar datos para PCA
    numeric_features = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
    pca_data = filtered_df_clean[numeric_features]
    
    # Normalizar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(pca_data)
    
    # Aplicar PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    # Varianza explicada
    st.subheader("Varianza Explicada por Componente")
    var_exp = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Varianza Explicada (%)': pca.explained_variance_ratio_ * 100,
        'Varianza Acumulada (%)': np.cumsum(pca.explained_variance_ratio_) * 100
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(var_exp, x='Componente', y='Varianza Explicada (%)',
                    title='Varianza Explicada por Componente')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(var_exp, x='Componente', y='Varianza Acumulada (%)',
                     title='Varianza Acumulada', markers=True)
        fig.add_hline(y=95, line_dash="dash", line_color="red", 
                     annotation_text="95%")
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(var_exp, use_container_width=True)
    
    # Biplot de los dos primeros componentes
    st.subheader("Biplot - Primeros Dos Componentes")
    
    # Tomar muestra para visualización
    sample_size = min(1000, len(pca_result))
    indices = np.random.choice(len(pca_result), sample_size, replace=False)
    
    pca_df = pd.DataFrame({
        'PC1': pca_result[indices, 0],
        'PC2': pca_result[indices, 1],
        'precio': pca_data.iloc[indices]['sellingprice'].values,
        'año': pca_data.iloc[indices]['year'].values
    })
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='precio',
                    title='Proyección en los dos primeros componentes',
                    color_continuous_scale='Viridis',
                    labels={'precio': 'Precio ($)'})
    
    # Agregar vectores de las variables
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(numeric_features):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0] * 3,
            y=loadings[i, 1] * 3,
            showarrow=True,
            arrowsize=1,
            arrowhead=2,
            arrowcolor="red",
            text=feature
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Contribución de variables a los componentes
    st.subheader("Contribución de Variables a los Componentes")
    loadings_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=numeric_features
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings_df.values,
        x=loadings_df.columns,
        y=loadings_df.index,
        colorscale='RdBu',
        zmid=0,
        text=loadings_df.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    fig.update_layout(title='Loadings de las Variables en los Componentes Principales')
    st.plotly_chart(fig, use_container_width=True)

# ====== PESTAÑA 6: RESUMEN TEMPORAL ======
with tab6:
    st.header("Resumen Temporal")
    
    # Métricas temporales
    st.subheader("Estadísticas por Período")
    
    # Por año de venta
    yearly_stats = filtered_df_clean.groupby('year_sale').agg({
        'sellingprice': ['mean', 'median', 'count'],
        'odometer': 'mean',
        'condition': 'mean'
    }).round(2)
    
    yearly_stats.columns = ['Precio Promedio', 'Precio Mediano', 'Num. Ventas', 
                            'Kilometraje Promedio', 'Condición Promedio']
    
    st.subheader("Estadísticas por Año de Venta")
    st.dataframe(yearly_stats, use_container_width=True)
    
    # Ventas por día de la semana
    st.subheader("Ventas por Día de la Semana")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = filtered_df.groupby('day_of_week').agg({
        'sellingprice': ['mean', 'count']
    }).round(2)
    day_stats.columns = ['Precio Promedio', 'Cantidad de Ventas']
    day_stats = day_stats.reindex(day_order)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(day_stats, y='Cantidad de Ventas',
                    title='Volumen de Ventas por Día de la Semana')
        fig.update_xaxes(title='Día de la Semana')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(day_stats, y='Precio Promedio',
                    title='Precio Promedio por Día de la Semana')
        fig.update_xaxes(title='Día de la Semana')
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis mensual
    st.subheader("Análisis Mensual")
    monthly_stats = filtered_df.groupby('month_sale').agg({
        'sellingprice': ['mean', 'count'],
        'odometer': 'mean'
    }).round(2)
    monthly_stats.columns = ['Precio Promedio', 'Num. Ventas', 'Kilometraje Promedio']
    
    month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    monthly_stats.index = [month_names[i-1] for i in monthly_stats.index]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_stats.index, y=monthly_stats['Precio Promedio'],
                            mode='lines+markers', name='Precio Promedio',
                            yaxis='y1'))
    fig.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Num. Ventas'],
                        name='Volumen de Ventas', yaxis='y2', opacity=0.6))
    
    fig.update_layout(
        title='Precio Promedio y Volumen de Ventas por Mes',
        xaxis=dict(title='Mes'),
        yaxis=dict(title='Precio Promedio ($)', side='left'),
        yaxis2=dict(title='Volumen de Ventas', side='right', overlaying='y'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tendencias temporales
    st.subheader("Tendencias Temporales")
    
    # Crear series temporales diarias
    daily_stats = filtered_df.groupby(filtered_df['saledate'].dt.date).agg({
        'sellingprice': 'mean',
        'odometer': 'mean'
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats['sellingprice'],
                            mode='lines', name='Precio Promedio',
                            line=dict(width=1)))
    
    # Agregar media móvil de 30 días
    rolling_mean = daily_stats['sellingprice'].rolling(window=30).mean()
    fig.add_trace(go.Scatter(x=daily_stats.index, y=rolling_mean,
                            mode='lines', name='Media Móvil (30 días)',
                            line=dict(width=2, color='red')))
    
    fig.update_layout(
        title='Evolución del Precio Promedio con Media Móvil',
        xaxis_title='Fecha',
        yaxis_title='Precio Promedio ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard creado con Streamlit")
st.sidebar.markdown("Análisis de precios de vehículos")