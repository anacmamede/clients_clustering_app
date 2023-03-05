import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from clusterclients.ClusterClients import ClusterClients

st.set_page_config(
    page_title="Classificação de Clientes",
    page_icon='👑' ,
    layout="wide",
)

def load_data(file_path):
    return pd.read_csv(file_path)

def cluster_rename(df):
    cluster_map = {7:'Principe', 5:'Duque', 1:'Marques', 2:'Conde', 4:'Visconde', 3:'Barao', 6:'Burgues', 0:'Plebeu'}
    df['cluster_name'] = df['cluster'].map(cluster_map)
    return df

def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def graf_pie_gross(df):
    st.markdown('### Faturamento Total por cluster')
    fig = px.pie(df, values='gross_revenue', names='cluster_name', color = 'cluster_name')
    fig.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font_size=15))
    # fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    return None

def graf_qtde_products(df):        
    st.markdown('### Quantidade de produtos comprados por cluster')
    aux = df[['qtde_products','cluster_name']].groupby('cluster_name').sum().reset_index().sort_values('qtde_products', ascending=True)
    fig = px.bar(aux, x='qtde_products', y='cluster_name')#, color = 'cluster_name')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    return None

def graf_median_gross(df):
    st.markdown('### Valor mediano de faturamento por cluster')
    aux = df[['gross_revenue','cluster_name']].groupby('cluster_name').median().reset_index().sort_values('gross_revenue', ascending=False)
    fig = px.bar(aux, y='gross_revenue', x='cluster_name')#, color = 'cluster_name')
    # fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    return None

def graf_returns(df):
    st.markdown('### Quantidade de devoluções por cluster')
    aux = df[['qtde_returns','cluster_name']].groupby('cluster_name').sum().reset_index().sort_values('qtde_returns', ascending=False)
    fig = px.bar(aux, y='qtde_returns', x='cluster_name')#, color = 'cluster_name')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    return None

def graf_dist(df_cluster):
    st.markdown('### Distribuição de clientes por cluster')
    fig = px.pie(df_cluster, values='perc_customer', names='cluster_name', color = 'cluster_name')
    fig.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font_size=15))
    st.plotly_chart(fig, use_container_width=True)
    return None

# dashboard title
st.title("🏬 Classificação e Fidelização de Clientes - A Realeza 👑")

selected = option_menu( menu_title=None, 
                       options=['Visualização', 'Clusterização de novos clientes'],
                       icons = ['bar-chart-fill', 'diagram-3-fill'],
                       orientation='horizontal')

pipeline = ClusterClients()

if selected=='Visualização':
    # st.markdown('## Acompanhamento Programa Príncipes')
    # file_path = 'data/df_cluster.csv'
    file_path = r"C:\Users\User\repos\pa_005\cluster_app\data\df_cluster.csv"
    df9 = load_data(file_path)
    df9 = cluster_rename(df9)
    df_cluster = pipeline.cluster_profile(df9)
    df_cluster = cluster_rename(df_cluster)
    m1, m2, m3, m4 = st.columns(spec=[1,1,1,1], gap='medium')
    m1.metric('Quantidade de clientes Príncipe:', value=len(df9[df9['cluster_name']=='Principe']))
    m2.metric('Quantidade de clientes Duque:', value=len(df9[df9['cluster_name']=='Duque']))
    m3.metric('Quantidade de clientes Marquês:', value=len(df9[df9['cluster_name']=='Marques']))
    m4.metric('Quantidade de clientes Conde:', value=len(df9[df9['cluster_name']=='Conde']))

    # c1, c, c2 = st.columns(spec=[1,0.25,1], gap='medium')
    with st.expander('', expanded=True):
        c1, c, c2 = st.columns(spec=[1,0.25,1], gap='medium')
        with c1:
            graf_pie_gross(df9)
        with c2:
            graf_qtde_products(df9)
    
    with st.expander('', expanded=True):
        c1, c, c2 = st.columns(spec=[1,0.25,1], gap='medium')
        with c1:        
            graf_median_gross(df9)
        with c2:
            graf_returns(df9)

    with st.expander('', expanded=True):
        c1, c, c2 = st.columns(spec=[1,0.25,1], gap='medium')
        with c1:  
            graf_dist(df_cluster)
        with c2:
            st.markdown('### Cluster Profile')
            st.dataframe(df_cluster[['cluster_name', 'perc_customer', 'gross_revenue',
            'recency_days', 'qtde_products', 'frequency', 'qtde_returns']], use_container_width=True)# width=800)
       



if selected =='Clusterização de novos clientes':
    # with st.expander('Dados dos clientes'):
    st.markdown('##### Upload dos dados dos clientes: ')
    uploaded_file = st.file_uploader('Insira um arquivo csv',type='csv')
    if uploaded_file is not None:
        dataframe_raw = pd.read_csv(uploaded_file)

        st.markdown('### Dataframe loaded')
        st.markdown('Quantidade de linhas: {}'.format(len(dataframe_raw )))
        st.markdown('Quantidade de columns: {}'.format(len(dataframe_raw .columns)))
        st.dataframe(dataframe_raw.head())

            # pipeline = ClusterClients()
        st.markdown('### Clusterização ...')
        df1 = pipeline.data_cleaning(dataframe_raw)
        st.text('Cleaning done')
        df2 = pipeline.data_filtering(df1)
        st.text('Filtering done')
        df4 = pipeline.feature_engineering(df2)
        st.text('Feature engineering done')
        df4_3 = pipeline.data_preparation(df4)
        st.text('Data preparation done')
        X, df_tree = pipeline.data_embedding(df4_3)
        st.text('Embedding done')
        df9_2 = pipeline.get_cluster(X, df_tree, df4)
        st.text('Clustering done')
        cluster_prof = pipeline.cluster_profile(df9_2)
        st.text('Profile done')
        st.text('Done!')

        # Name clusters
        df9_2 = cluster_rename(df9_2)
        cluster_prof = cluster_rename(cluster_prof)

        st.markdown('## Resumo da clusterização:')
        with st.expander('', expanded=True): 
            c1,c2,c3 = st.columns(spec=[1,1,1], gap='large')
            with c1:
                graf_dist(cluster_prof)
            with c2:          
                st.markdown('### Cluster Profile')
                st.dataframe(cluster_prof[['cluster_name', 'perc_customer', 'gross_revenue',
                'recency_days', 'qtde_products', 'frequency', 'qtde_returns']], use_container_width=True)# width=800)     
            with c3:  
                graf_pie_gross(df9_2) 

        with st.expander('', expanded=True): 
            c1,c2,c3 = st.columns(spec=[1,1,1], gap='large')
            with c1: 
                graf_qtde_products(df9_2)
            with c2:
                graf_median_gross(df9_2)        
            with c3:
                graf_returns(df9_2)

        st.download_button('Download csv clusterização de clientes', data=convert_df(df9_2), file_name='ClusterizaçaoClientes.csv', mime='text/csv')
        st.download_button('Download csv Cluster Profile', data=convert_df(cluster_prof), file_name='Cluster Profile.csv', mime='text/csv')