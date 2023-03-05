import pickle
import inflection
import pandas as pd
import numpy as np

class ClusterClients(object):
    def __init__(self):
        self.home_path = 'C:/Users/User/repos/pa_005/'        
        self.gross_revenue_scaler = pickle.load(open(self.home_path + 'src/features/gross_revenue_scaler.pkl', 'rb'))
        self.recency_days_scaler  = pickle.load(open(self.home_path + 'src/features/recency_days_scaler.pkl', 'rb'))
        self.qtde_products_scaler = pickle.load(open(self.home_path + 'src/features/qtde_products_scaler.pkl', 'rb'))
        self.frequency_scaler     = pickle.load(open(self.home_path + 'src/features/frequency_scaler.pkl', 'rb'))
        self.qtde_returns_scaler  = pickle.load(open(self.home_path + 'src/features/qtde_returns_scaler.pkl', 'rb'))
        self.rf_model = pickle.load(open(self.home_path +'src/models/rf_model.pkl', 'rb' ) )
        self.reducer  = pickle.load(open(self.home_path +'src/features/umap_reducer.pkl', 'rb' ) )
        self.kmeans   = pickle.load(open(self.home_path +'src/models/kmeans_model.pkl', 'rb' ) )

    def data_cleaning(self, df_raw):
        df_raw = df_raw.drop('Unnamed: 8', axis =1)

        # Passo 1.0 - Descrição dos dados
        df1 = df_raw.copy()

        ## Rename columns
        cols_old = df1.columns
        snakecase = lambda x: inflection.underscore( x )
        cols_new = list( map( snakecase, cols_old ) )
        df1.columns = cols_new

        df_missing = df1.loc[df1['customer_id'].isna(),:]
        df_not_missing = df1.loc[~df1['customer_id'].isna(),:]

        # create reference
        df_backup = pd.DataFrame( df_missing['invoice_no'].drop_duplicates() )
        df_backup['customer_id'] = np.arange( 19000, 19000+len( df_backup ), 1)

        # merge original with reference dataframe
        df1 = pd.merge( df1, df_backup, on='invoice_no', how='left' )

        # coalesce 
        df1['customer_id'] = df1['customer_id_x'].combine_first( df1['customer_id_y'] )

        # drop extra columns
        df1 = df1.drop( columns=['customer_id_x', 'customer_id_y'], axis=1 )

        ## Change dtypes

        # invoice_date
        df1['invoice_date'] = pd.to_datetime(df1['invoice_date'], format='%d-%b-%y')

        # customer_id
        df1['customer_id'] = df1['customer_id'].astype(int)

        return df1

    def data_filtering(self, df1):
        # Passo 3.0 - Filtragem de variáveis

        df2=df1.copy()

        # ====== Numerical Attributes ======

        # === Numerical attributes ====
        df2 = df2.loc[df2['unit_price'] >= 0.04, :]

        # === Categorical attributes ====
        df2 = df2[~df2['stock_code'].isin( ['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY', 'DCGSSGIRL', 'PADS', 'B', 'CRUK'] ) ]

        # description
        df2 = df2.drop( columns='description', axis=1 )

        # map -  
        df2 = df2[~df2['country'].isin( ['European Community', 'Unspecified' ] ) ]

        # bad users
        df2 = df2[~df2['customer_id'].isin( [16446] )]

        # quantity
        df2_returns = df2.loc[df2['quantity'] < 0, :]
        df2_purchases = df2.loc[df2['quantity'] >= 0, :]

        return df2

    def feature_engineering(self, df2):
        # quantity
        df2_returns = df2.loc[df2['quantity'] < 0, :]
        df2_purchases = df2.loc[df2['quantity'] >= 0, :]

        # Passo 2.0 - Feature Engineering
        df3 = df2.copy()

        ## Feature Creation

        # data reference
        df_ref = df3.drop(columns=['invoice_no', 'stock_code', 'quantity', 'invoice_date',
               'unit_price', 'country'], axis=1).drop_duplicates(ignore_index=True)

        ### Gross Revenue
        # Gross Revenue (Faturamento = quantity x price)
        df2_purchases.loc[:, 'gross_revenue'] = df2_purchases.loc[:, 'quantity'] * df2_purchases.loc[:, 'unit_price']

        # Monetary
        df_monetary = df2_purchases.loc[:, ['customer_id', 'gross_revenue']].groupby( 'customer_id' ).sum().reset_index()
        df_ref = pd.merge( df_ref, df_monetary, on='customer_id', how='left' )

        ### Recency -  Day from last purchase
        # Recency -  Day from last purchase
        df_recency = df2_purchases.loc[:, ['customer_id', 'invoice_date']].groupby( 'customer_id' ).max().reset_index()
        df_recency['recency_days'] = ( df2['invoice_date'].max() - df_recency['invoice_date'] ).dt.days
        df_recency = df_recency[['customer_id', 'recency_days']].copy()
        df_ref = pd.merge( df_ref, df_recency, on='customer_id', how='left' )

        ### Quantity of products purchased
        # Frequency
        df_prd = (df2_purchases.loc[:, ['customer_id', 'stock_code']].groupby( 'customer_id' ).count()
                                                                   .reset_index()
                                                                   .rename( columns={'stock_code': 'qtde_products'} ) )
        df_ref = pd.merge( df_ref, df_prd, on='customer_id', how='left' )

        ### Frequency Purchase
        df_aux = ( df2_purchases[['customer_id', 'invoice_no', 'invoice_date']].drop_duplicates()
                                                                     .groupby( 'customer_id')
                                                                     .agg( max_ = ( 'invoice_date', 'max' ), 
                                                                           min_ = ( 'invoice_date', 'min' ),
                                                                           days_= ( 'invoice_date', lambda x: ( ( x.max() - x.min() ).days ) + 1 ),
                                                                           buy_ = ( 'invoice_no', 'count' ) ) ).reset_index()
        # Frequency
        df_aux['frequency'] = df_aux[['buy_', 'days_']].apply( lambda x: x['buy_'] / x['days_'] if  x['days_'] != 0 else 0, axis=1 )
        # Merge
        df_ref = pd.merge( df_ref, df_aux[['customer_id', 'frequency']], on='customer_id', how='left' )

        ### Number of returns
        # Number of Returns
        df_returns = df2_returns[['customer_id', 'quantity']].groupby( 'customer_id' ).sum().reset_index().rename( columns={'quantity':'qtde_returns'} )
        df_returns['qtde_returns'] = df_returns['qtde_returns'] * -1
        df_ref = pd.merge( df_ref, df_returns, how='left', on='customer_id' )
        df_ref.loc[df_ref['qtde_returns'].isna(), 'qtde_returns'] = 0

        df4 = df_ref.dropna().copy()

        return df4

    def data_preparation(self, df4):
        ## Estudo de Espaço
        # Selected dataset
        cols_selected = ['customer_id', 'gross_revenue', 'recency_days', 'qtde_products', 'frequency', 'qtde_returns']
        df4_3 = df4[ cols_selected ].drop( columns='customer_id', axis=1 )
        df4_3['gross_revenue'] = self.gross_revenue_scaler.transform( df4_3[['gross_revenue']] ) 
        df4_3['recency_days'] = self.recency_days_scaler.transform( df4_3[['recency_days']] )
        df4_3['qtde_products'] = self.qtde_products_scaler .transform( df4_3[['qtde_products']] )
        df4_3['frequency'] = self.frequency_scaler.transform( df4_3[['frequency']] )
        df4_3['qtde_returns'] = self.qtde_returns_scaler.transform( df4_3[['qtde_returns']] )

        return df4_3


    def data_embedding(self, df4_3):
        # training dataset
        X = df4_3.drop( columns=['gross_revenue'], axis=1 )
        y = df4_3['gross_revenue']

        # Leaf 
        df_leaf = pd.DataFrame( self.rf_model.apply( X ) )

        # Reduzer dimensionality
        embedding = self.reducer.transform( df_leaf )

        # embedding
        df_tree = pd.DataFrame()
        df_tree['embedding_x'] = embedding[:, 0]
        df_tree['embedding_y'] = embedding[:, 1]
        return X, df_tree

    def get_cluster(self, X, df_tree, df4):
        # Passo 5.0 - Data preparation

        # Tree-Based Embedding
        df5 = df_tree.copy()

        # clustering
        labels = self.kmeans.labels_

        # Passo 9.0 - Cluster Analysis

        df9 = X.copy()
        df9['cluster'] = labels

        ## Cluster Profile
        cols_selected = ['customer_id', 'gross_revenue', 'recency_days', 'qtde_products', 'frequency', 'qtde_returns']
        df9_2 = df4[ cols_selected ].copy()
        df9_2['cluster'] = labels
        return df9_2

    def cluster_profile(self, df9_2):
        # Number of customers
        df_cluster = df9_2[['customer_id', 'cluster']].groupby('cluster').count().reset_index()
        df_cluster['perc_customer'] = 100*(df_cluster['customer_id']/df_cluster['customer_id'].sum())

        # Avg 'gross_revenue'
        df_avg_gross_revenue = df9_2[['gross_revenue', 'cluster']].groupby('cluster').mean().reset_index()
        df_cluster = pd.merge(df_cluster, df_avg_gross_revenue, on='cluster', how='inner')

        # Avg 'recency_days'
        df_avg_recency_days = df9_2[['recency_days', 'cluster']].groupby('cluster').mean().reset_index()
        df_cluster = pd.merge(df_cluster, df_avg_recency_days, on='cluster', how='inner')

        # Avg 'qtde_products'
        df_avg_invoice_no = df9_2[['qtde_products', 'cluster']].groupby('cluster').mean().reset_index()
        df_cluster = pd.merge(df_cluster, df_avg_invoice_no, on='cluster', how='inner')

        # Avg 'frequency'
        df_avg_ticket = df9_2[['frequency', 'cluster']].groupby('cluster').mean().reset_index()
        df_cluster = pd.merge(df_cluster, df_avg_ticket, on='cluster', how='inner')

        # Avg 'qtde_returns'
        df_avg_ticket = df9_2[['qtde_returns', 'cluster']].groupby('cluster').mean().reset_index()
        df_cluster = pd.merge(df_cluster, df_avg_ticket, on='cluster', how='inner')

        return df_cluster.sort_values('gross_revenue', ascending=False)