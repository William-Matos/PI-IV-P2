import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Configurações da App
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide")

# --- Título Principal ---
st.title("Análise de Cluster de Escolas (IDEB e SAESB)")
st.markdown("Comparativo de desempenho entre Escolas Militares e Demais Escolas.")

# --- 1. Carregamento de Dados (com Cache) ---
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Erro: Arquivo não encontrado em '{path}'.")
        st.info("Certifique-se de que o 'dataset_final.csv' está na pasta 'dados/'.")
        return None

# --- Definições Globais ---
features = ['ideb', 'nota_saeb_matematica', 'nota_saeb_lingua_portuguesa', 'taxa_aprovacao']
coluna_analise = 'vinculo_seguranca_publica'

# Carregar os dados
df_master = load_data(r'dados/dataset_final.csv')

# --- 2. Barra Lateral de Navegação ---
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Escolha a análise:",
    ("Apresentação do Projeto", "Ensino Fundamental", "Ensino Médio", "Conclusão")
)

if df_master is not None:
    # ===================================================================
# --- PÁGINA 0: APRESENTAÇÃO DO PROJETO ---
# ===================================================================
    if pagina == "Apresentação do Projeto":
        st.title("Análise de Cluster de Desempenho Escolar no Brasil")
        st.markdown("---")

        st.header('1. Problema de Pesquisa e Contextualização')
        st.markdown("""
        A qualidade da educação básica é um pilar para o desenvolvimento social, mas as escolas no Brasil apresentam um desempenho muito heterogêneo. Este projeto investiga essa heterogeneidade, focando numa comparação específica entre tipos de administração escolar para responder à seguinte questão central:
        """)
        st.info("#### Qual é o perfil de desempenho das escolas militares em comparação com as demais escolas, usando dados públicos como IDEB e SAESB?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Medindo o Desempenho")
            st.markdown("""
            O desempenho escolar não é medido por um único indicador. Para capturar um perfil mais completo, utilizamos uma abordagem de **Clusterização (Agrupamento)**.
            
            - **Métricas Utilizadas:** A análise agrupa as escolas com base em quatro indicadores-chave de performance: `IDEB`, `Nota SAEB de Matemática`, `Nota SAEB de Língua Portuguesa` e `Taxa de Aprovação`.
            - **K-Means:** Usamos o algoritmo K-Means para identificar "perfis" de escolas, segmentando-as em grupos naturais de **Alto**, **Médio** e **Baixo** desempenho.
            """)

        with col2:
            st.subheader("Foco da Comparação")
            st.markdown("""
            O foco principal do estudo é entender como diferentes tipos de gestão se posicionam dentro desses clusters de desempenho.
            
            - **Variável de Análise:** A comparação é feita usando a coluna `vinculo_seguranca_publica`, que identifica escolas com gestão militar ou de corpos de segurança.
            - **Desequilíbrio de Classes:** Como o número de escolas militares é muito menor que o de "demais escolas", uma simples contagem seria enganosa.
            - **Análise Percentual:** Por isso, a análise principal foca na **distribuição percentual**, respondendo: "Do total de escolas militares, quantos porcento estão no cluster de Alto Desempenho?"
            """)
        
        st.markdown("---")
        st.header("2. Objetivo, Hipóteses e Variáveis")
        st.markdown("🎯 **Objetivo:** Identificar perfis de desempenho (clusters) nas escolas brasileiras e comparar, percentualmente, a distribuição de escolas com vínculo militar e demais escolas dentro desses perfis.")
        
        st.markdown("""
        Para guiar nossa análise, partimos das seguintes hipóteses:
        - **Hipótese 1:** As escolas brasileiras não são homogêneas e podem ser agrupadas de forma significativa em clusters (grupos) de desempenho distintos (ex: Alto, Médio, Baixo).
        - **Hipótese 2:** As escolas com `vinculo_seguranca_publica` (militares) estão desproporcionalmente concentradas nos clusters de maior desempenho quando comparadas às demais escolas.
        """)

        st.subheader("Variáveis Utilizadas na Modelagem")
        st.markdown("""
        Para construir os clusters e realizar a análise, o dataset foi dividido em dois conjuntos de variáveis:
        """)

        # Detalhando as variáveis de Clusterização (Features)
        st.markdown("📊 **Variáveis de Clusterização (Features 'X'):**")
        st.markdown("Estas são as 4 métricas usadas pelo K-Means para decidir como agrupar as escolas:")
        with st.expander("Clique para ver as 4 features de desempenho"):
            st.markdown("""
            - **`ideb`**: Índice de Desenvolvimento da Educação Básica, a métrica mais conhecida.
            - **`nota_saeb_matematica`**: Desempenho padronizado em matemática.
            - **`nota_saeb_lingua_portuguesa`**: Desempenho padronizado em português.
            - **`taxa_aprovacao`**: Percentual de alunos aprovados na série.
            """)
        
        # Detalhando a variável de Comparação
        st.markdown("🔍 **Variável de Comparação (Categoria):**")
        st.markdown("Após a criação dos clusters, esta variável é usada para comparar os grupos:")
        st.markdown("- **`vinculo_seguranca_publica`**: Variável binária (0 ou 1) que indica se a escola possui vínculo com órgãos de segurança pública (ex: polícias militares, bombeiros).")

        st.markdown("---")
        st.header('3. Metodologia Analítica')
        st.write("A abordagem metodológica foi dividida em duas etapas principais, executadas de forma independente para o Ensino Fundamental e o Ensino Médio.")
        
        col_met_1, col_met_2 = st.columns(2)
        with col_met_1:
            st.subheader('A. Clusterização K-Means')
            st.write(r"""
                **Objetivo:** Segmentar as escolas em K grupos.
                
                1.  **Padronização:** As 4 features de desempenho são padronizadas (StandardScaler) para que tenham a mesma escala.
                2.  **Método do Cotovelo:** O "Número K" de clusters ideal (K=3) é determinado visualmente através do "Método do Cotovelo" (Elbow Method).
                3.  **Agrupamento:** O K-Means é executado com K=3, atribuindo a cada escola um rótulo de cluster.
                4.  **Nomeação:** Os clusters (ex: 0, 1, 2) são nomeados ("Alto", "Médio", "Baixo") com base na sua média de IDEB.
                """)
        with col_met_2:
            st.subheader("B. Análise Percentual Comparativa")
            st.write("""
                **Objetivo:** Comparar os grupos de forma justa.
                
                1.  **Contagem:** Contamos quantas "Escolas Militares" e "Demais Escolas" existem em cada um dos 3 clusters.
                2.  **Cálculo Percentual:** Calculamos a proporção *dentro de cada tipo*. Por exemplo, (Nº de Militares em 'Alto') / (Nº *Total* de Militares).
                3.  **Visualização:** Os resultados são apresentados em tabelas e gráficos de barras agrupadas (via Plotly) para permitir uma comparação visual direta das distribuições percentuais.
                """)
                
        st.markdown("---")
        st.info("Navegue pelas páginas 'Ensino Fundamental' e 'Ensino Médio' no menu lateral para ver os resultados da análise.")
    # ===================================================================
    # --- PÁGINA 1: ANÁLISE FUNDAMENTAL ---
    # ===================================================================
    if pagina == "Ensino Fundamental":
        st.header("Análise: Ensino Fundamental")

        # --- Preparação dos Dados ---
        df_fund = df_master[df_master['ensino'] == 'fundamental'].copy()
        df_fund.dropna(subset=features + [coluna_analise], inplace=True)
        
        x_fund = df_fund[features].values
        scaler_fund = StandardScaler()
        x_fund_scaled = scaler_fund.fit_transform(x_fund)

        # --- 1. Método do Cotovelo (com Plotly) ---
        st.subheader("1. Método do Cotovelo (Elbow Method)")
        with st.spinner("Calculando WCSS para o Ensino Fundamental..."):
            wcss_fund = []
            K_range = range(1, 11)
            for i in K_range:
                kmeans_fund = KMeans(n_clusters=i, n_init=10, random_state=42)
                kmeans_fund.fit(x_fund_scaled)
                wcss_fund.append(kmeans_fund.inertia_)
            
            df_elbow_fund = pd.DataFrame({'Clusters': list(K_range), 'WCSS': wcss_fund})
            
            fig_elbow_fund = px.line(
                df_elbow_fund, 
                x='Clusters', 
                y='WCSS', 
                title='Método do Cotovelo - Ensino Fundamental',
                markers=True, 
                labels={'Clusters': 'Número de Clusters'}
            )
            fig_elbow_fund.update_traces(line_color='blue', marker_symbol='x')
            
            # --- CORREÇÃO APLICADA AQUI ---
            fig_elbow_fund.update_xaxes(tickvals=list(K_range))
            
            st.plotly_chart(fig_elbow_fund, use_container_width=True)
        
        st.info("O gráfico acima mostra um 'cotovelo' claro em K=3. Vamos usar K=3 para a análise.")
        k_fundamental = 3

        # --- 2. Análise de Cluster (K=3) ---
        st.subheader(f"2. Análise de Cluster com K={k_fundamental}")
        
        kmeans_fund_final = KMeans(n_clusters=k_fundamental, n_init=10, random_state=42)
        df_fund['cluster_num'] = kmeans_fund_final.fit_predict(x_fund_scaled)
        
        perfil_fund = df_fund.groupby('cluster_num')[features].mean().sort_values(by='ideb', ascending=False)
        map_fund = {
            perfil_fund.index[0]: 'Alto Desempenho',
            perfil_fund.index[1]: 'Médio Desempenho',
            perfil_fund.index[2]: 'Baixo Desempenho'
        }
        df_fund['cluster'] = df_fund['cluster_num'].map(map_fund)
        
        st.markdown("Perfil dos Clusters (Baseado no IDEB médio):")
        st.dataframe(perfil_fund)

        # --- 3. Análise Percentual (com Plotly) ---
        st.subheader("3. Comparação Percentual (Militares vs. Demais)")
        df_fund['Tipo_Escola'] = df_fund[coluna_analise].map({0.0: 'Demais Escolas', 1.0: 'Escolas Militares'})
        
        comparacao_fund = df_fund.groupby(['cluster', 'Tipo_Escola']).size().unstack(fill_value=0)
        perc_fund = comparacao_fund.apply(lambda x: (x / x.sum()) * 100).round(2)
        perc_fund_sorted = perc_fund.reindex(['Alto Desempenho', 'Médio Desempenho', 'Baixo Desempenho'])
        
        st.markdown("Distribuição Percentual DENTRO de cada tipo de escola:")
        st.dataframe(perc_fund_sorted)
        
        df_plot_fund = perc_fund_sorted.reset_index().melt(
            id_vars='cluster', 
            var_name='Tipo_Escola', 
            value_name='Percentual (%)'
        )
        
        fig_perc_fund = px.bar(
            df_plot_fund,
            x='cluster',
            y='Percentual (%)',
            color='Tipo_Escola',
            barmode='group',
            title='Distribuição Percentual por Cluster de Desempenho (Fundamental)',
            labels={'cluster': 'Cluster de Desempenho'}
        )
        
        st.plotly_chart(fig_perc_fund, use_container_width=True)

    # ===================================================================
    # --- PÁGINA 2: ANÁLISE MÉDIO ---
    # ===================================================================
    elif pagina == "Ensino Médio":
        st.header("Análise: Ensino Médio")

        # --- Preparação dos Dados ---
        df_med = df_master[df_master['ensino'] == 'medio'].copy()
        df_med.dropna(subset=features + [coluna_analise], inplace=True)
        
        X_med = df_med[features].values
        scaler_med = StandardScaler()
        X_med_scaled = scaler_med.fit_transform(X_med)

        # --- 1. Método do Cotovelo (com Plotly) ---
        st.subheader("1. Método do Cotovelo (Elbow Method)")
        with st.spinner("Calculando WCSS para o Ensino Médio..."):
            wcss_med = []
            K_range = range(1, 11)
            for i in K_range:
                kmeans_med = KMeans(n_clusters=i, n_init=10, random_state=42)
                kmeans_med.fit(X_med_scaled)
                wcss_med.append(kmeans_med.inertia_)
            
            df_elbow_med = pd.DataFrame({'Clusters': list(K_range), 'WCSS': wcss_med})
            
            fig_elbow_med = px.line(
                df_elbow_med, 
                x='Clusters', 
                y='WCSS', 
                title='Método do Cotovelo - Ensino Médio',
                markers=True,
                labels={'Clusters': 'Número de Clusters'}
            )
            fig_elbow_med.update_traces(line_color='red', marker_symbol='x')
            
            # --- CORREÇÃO APLICADA AQUI ---
            fig_elbow_med.update_xaxes(tickvals=list(K_range))
            
            st.plotly_chart(fig_elbow_med, use_container_width=True)
        
        st.info("O gráfico acima também sugere K=3. Vamos usar K=3 para a análise.")
        k_medio = 3

        # --- 2. Análise de Cluster (K=3) ---
        st.subheader(f"2. Análise de Cluster com K={k_medio}")
        
        kmeans_med_final = KMeans(n_clusters=k_medio, n_init=10, random_state=42)
        df_med['cluster_num'] = kmeans_med_final.fit_predict(X_med_scaled)
        
        perfil_med = df_med.groupby('cluster_num')[features].mean().sort_values(by='ideb', ascending=False)
        map_med = {
            perfil_med.index[0]: 'Alto Desempenho',
            perfil_med.index[1]: 'Médio Desempenho',
            perfil_med.index[2]: 'Baixo Desempenho'
        }
        df_med['cluster'] = df_med['cluster_num'].map(map_med)
        
        st.markdown("Perfil dos Clusters (Baseado no IDEB médio):")
        st.dataframe(perfil_med)

        # --- 3. Análise Percentual (com Plotly) ---
        st.subheader("3. Comparação Percentual (Militares vs. Demais)")
        df_med['Tipo_Escola'] = df_med[coluna_analise].map({0.0: 'Demais Escolas', 1.0: 'Escolas Militares'})
        
        comparacao_med = df_med.groupby(['cluster', 'Tipo_Escola']).size().unstack(fill_value=0)
        perc_med = comparacao_med.apply(lambda x: (x / x.sum()) * 100).round(2)
        perc_med_sorted = perc_med.reindex(['Alto Desempenho', 'Médio Desempenho', 'Baixo Desempenho'])
        
        st.markdown("Distribuição Percentual DENTRO de cada tipo de escola:")
        st.dataframe(perc_med_sorted)
        
        df_plot_med = perc_med_sorted.reset_index().melt(
            id_vars='cluster', 
            var_name='Tipo_Escola', 
            value_name='Percentual (%)'
        )
        
        fig_perc_med = px.bar(
            df_plot_med,
            x='cluster',
            y='Percentual (%)',
            color='Tipo_Escola',
            barmode='group',
            title='Distribuição Percentual por Cluster de Desempenho (Médio)',
            labels={'cluster': 'Cluster de Desempenho'}
        )
        
        st.plotly_chart(fig_perc_med, use_container_width=True)
    
    # ===================================================================
# --- PÁGINA 3: CONCLUSÃO ---
# ===================================================================
    elif pagina == "Conclusão":
        
        st.title("Conclusão da Análise")
        st.markdown("---")

        st.header("Pergunta Central: Escolas militares têm desempenho melhor no IDEB?")
        
        st.success("""
        **Sim. A análise dos dados indica inequivocamente que as escolas com vínculo à segurança pública (militares) apresentam um desempenho superior.**
        
        Mais do que isso, a nossa análise de clusterização revela que este grupo não é apenas "um pouco melhor", mas representa um perfil de performance distinto, concentrando-se de forma desproporcional no estrato mais alto de desempenho educacional.
        """)
        
        st.header("As Evidências Principais")
        st.markdown("""
        Para uma comparação justa, que levasse em conta o número muito menor de escolas militares, a análise final focou na **distribuição percentual**. 
        
        A pergunta foi: "Do total de 100% de escolas militares, quantas estão no cluster de Alto Desempenho, em comparação com as demais?"
        """)
        
        # --- Nota de Performance ---
        # Numa app ideal, estes cálculos seriam feitos uma vez e guardados (cache).
        # Para manter a estrutura da tua app, recalculamos os dados necessários aqui.
        
        with st.spinner("A gerar gráficos de conclusão..."):
            # --- Início do Recálculo (Fundamental) ---
            k_fundamental = 3
            df_fund = df_master[df_master['ensino'] == 'fundamental'].copy()
            df_fund.dropna(subset=features + [coluna_analise], inplace=True)
            x_fund = df_fund[features].values
            scaler_fund = StandardScaler()
            x_fund_scaled = scaler_fund.fit_transform(x_fund)
            kmeans_fund_final = KMeans(n_clusters=k_fundamental, n_init=10, random_state=42)
            df_fund['cluster_num'] = kmeans_fund_final.fit_predict(x_fund_scaled)
            perfil_fund = df_fund.groupby('cluster_num')[features].mean().sort_values(by='ideb', ascending=False)
            map_fund = { perfil_fund.index[0]: 'Alto Desempenho', perfil_fund.index[1]: 'Médio Desempenho', perfil_fund.index[2]: 'Baixo Desempenho' }
            df_fund['cluster'] = df_fund['cluster_num'].map(map_fund)
            df_fund['Tipo_Escola'] = df_fund[coluna_analise].map({0.0: 'Demais Escolas', 1.0: 'Escolas Militares'})
            comparacao_fund = df_fund.groupby(['cluster', 'Tipo_Escola']).size().unstack(fill_value=0)
            perc_fund = comparacao_fund.apply(lambda x: (x / x.sum()) * 100).round(2)
            perc_fund_sorted = perc_fund.reindex(['Alto Desempenho', 'Médio Desempenho', 'Baixo Desempenho'])
            df_plot_fund = perc_fund_sorted.reset_index().melt(id_vars='cluster', var_name='Tipo_Escola', value_name='Percentual (%)')
            fig_perc_fund = px.bar(
                df_plot_fund, x='cluster', y='Percentual (%)', color='Tipo_Escola', barmode='group',
                title='Distribuição Percentual (Ensino Fundamental)',
                labels={'cluster': 'Cluster de Desempenho'}
            )
            # --- Fim do Recálculo (Fundamental) ---

            # --- Início do Recálculo (Médio) ---
            k_medio = 3
            df_med = df_master[df_master['ensino'] == 'medio'].copy()
            df_med.dropna(subset=features + [coluna_analise], inplace=True)
            X_med = df_med[features].values
            scaler_med = StandardScaler()
            X_med_scaled = scaler_med.fit_transform(X_med)
            kmeans_med_final = KMeans(n_clusters=k_medio, n_init=10, random_state=42)
            df_med['cluster_num'] = kmeans_med_final.fit_predict(X_med_scaled)
            perfil_med = df_med.groupby('cluster_num')[features].mean().sort_values(by='ideb', ascending=False)
            map_med = { perfil_med.index[0]: 'Alto Desempenho', perfil_med.index[1]: 'Médio Desempenho', perfil_med.index[2]: 'Baixo Desempenho' }
            df_med['cluster'] = df_med['cluster_num'].map(map_med)
            df_med['Tipo_Escola'] = df_med[coluna_analise].map({0.0: 'Demais Escolas', 1.0: 'Escolas Militares'})
            comparacao_med = df_med.groupby(['cluster', 'Tipo_Escola']).size().unstack(fill_value=0)
            perc_med = comparacao_med.apply(lambda x: (x / x.sum()) * 100).round(2)
            perc_med_sorted = perc_med.reindex(['Alto Desempenho', 'Médio Desempenho', 'Baixo Desempenho'])
            df_plot_med = perc_med_sorted.reset_index().melt(id_vars='cluster', var_name='Tipo_Escola', value_name='Percentual (%)')
            fig_perc_med = px.bar(
                df_plot_med, x='cluster', y='Percentual (%)', color='Tipo_Escola', barmode='group',
                title='Distribuição Percentual (Ensino Médio)',
                labels={'cluster': 'Cluster de Desempenho'}
            )
            # --- Fim do Recálculo (Médio) ---

        st.subheader("📈 Ensino Fundamental")
        st.markdown(f"""
        Nos dados do Ensino Fundamental, a disparidade é clara:
        - **92%** das Escolas Militares foram classificadas no cluster de **Alto Desempenho**.
        - Nas Demais Escolas, 58.8% ficaram neste mesmo cluster.
        """)
        st.plotly_chart(fig_perc_fund, use_container_width=True)

        st.subheader("📊 Ensino Médio")
        st.markdown(f"""
        No Ensino Médio, a tendência repete-se:
        - **81.5%** das Escolas Militares foram classificadas no cluster de **Alto Desempenho**.
        - Nas Demais Escolas, apenas 32.3% alcançaram este perfil, com a maioria (53.2%) a ficar no cluster de Desempenho Médio.
        """)
        st.plotly_chart(fig_perc_med, use_container_width=True)

        st.markdown("---")
        st.header("Limitações e Próximos Passos")
        
        st.warning("""
        **Importante: Correlação não é Causalidade.**
        
        Esta análise é **descritiva** e confirma *o quê* (as escolas militares performam melhor), mas não explica *o porquê*. 
        Os dados mostram uma forte correlação, mas não isolam as causas.
        """)
        
        st.markdown("Possíveis fatores que contribuem para este resultado e que não foram isolados nesta análise:")
        
        with st.expander("Clique para ver os Fatores Contribuintes e Próximos Passos"):
            st.markdown("""
            * **Processo Seletivo:** Muitas destas escolas aplicam exames de admissão, selecionando alunos que já possuem um desempenho académico superior.
            * **Perfil Socioeconómico:** O perfil (ex: `media_inse`, que está no dataset) dos alunos que procuram e ingressam nessas escolas pode ser, em média, mais alto que o das demais escolas.
            * **Investimento e Recursos:** Diferenças no financiamento por aluno, infraestrutura e corpo docente.
            * **Modelo de Gestão:** A filosofia de disciplina e gestão pedagógica.
            
            #### Próximos Passos
            
            Como **próximos passos**, sugere-se uma análise de regressão ou um estudo pareado (matching) que tente isolar estas variáveis, comparando escolas militares apenas com escolas "civis" que possuam perfis de `media_inse` e investimento semelhantes.
            """)

        st.markdown("---")
        st.header("Veredito Final")
        st.info("""
        A pergunta central foi respondida. Os dados não apenas confirmam a hipótese de que as escolas militares têm um desempenho melhor, mas demonstram que elas operam num patamar de performance (Alto Desempenho) que é a exceção, e não a regra, no sistema educacional brasileiro analisado.
        """)
else:
    st.warning("O carregamento dos dados falhou. A aplicação não pode continuar.")