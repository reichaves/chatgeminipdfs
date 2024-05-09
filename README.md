Chatbot para Entrevistar PDFs Jornalisticante com Gemini-1.0-pro, Embedding-001 e Streamlit

Introdução

Este script apresenta um chatbot que utiliza modelos de linguagem avançados para entrevistar arquivos PDF com fins jornalísticos. O projeto foi desenvolvido durante a Imersão IA 2024 da Alura e do Google.

Funcionalidades

    Entrevista PDFs: O chatbot pode ser usado para analisar e resumir documentos PDF, como processos judiciais, contratos públicos, pedidos da LAI e outros.
    Contexto Jornalístico: O chatbot é treinado com um conjunto de instruções e informações específicas para o contexto jornalístico, garantindo respostas relevantes e precisas.
    Respostas Abrangentes: O chatbot fornece resumos abrangentes dos documentos, incluindo informações sobre o tipo de documento, partes envolvidas, principais argumentos e resultados.
    Segurança e Neutralidade: O chatbot é configurado com medidas de segurança para evitar a geração de conteúdo prejudicial e garantir a neutralidade nas respostas.

Como Funciona

    Carregar PDFs: O usuário carrega os arquivos PDF que deseja analisar.
    Processamento de Texto: O script extrai o texto dos PDFs e o divide em partes menores para processamento.
    Criação de Vetor: O texto é convertido em vetores numéricos usando um modelo de linguagem avançado.
    Análise e Resumo: O chatbot utiliza um modelo de linguagem conversacional para analisar os vetores e gerar resumos informativos e relevantes.
    Interação: O usuário pode interagir com o chatbot fazendo perguntas sobre os documentos e recebendo respostas detalhadas.

Benefícios

    Auxílio na Pesquisa Jornalística: O chatbot pode auxiliar jornalistas na pesquisa e análise de documentos complexos, economizando tempo e esforço.
    Acesso à Informação: O chatbot facilita o acesso à informação contida em documentos PDF, tornando-a mais acessível ao público.
    Transparência e Accountability: O uso do chatbot na análise de documentos públicos pode contribuir para a transparência e accountability do governo.

Tecnologias Utilizadas

    Gemini-1.0-pro: Modelo de linguagem conversacional do Google AI
    Embedding-001: Modelo de incorporação de texto do Google AI
    Streamlit: Biblioteca Python para criar interfaces web interativas
    PyPDF2: Biblioteca Python para trabalhar com arquivos PDF
    FAISS: Biblioteca para indexação e busca de vetores
    Langchain: Biblioteca Python para construir cadeias de processamento de linguagem natural

Observações

    O script ainda está em desenvolvimento e pode ser aprimorado com novas funcionalidades e modelos de linguagem.
    É importante lembrar que as respostas do chatbot devem ser sempre verificadas com as fontes originais de informação.
