# -*- coding: utf-8
# Reinaldo Chaves (reichaves@gmail.com)
# Chatbot script that uses gemini-1.5-pro, embedding-001 and streamlit to journalistically interview .PDF files
# Program is a project presented at Alura and Google's AI Immersion 2024
#

# Import the necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
import asyncio

# Function to extract text from several PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) # Initialize a PDF reader for each document
        for page in pdf_reader.pages: # Iterate on each PDF page
            text += page.extract_text() # Extract the text from the page and add it to the text variable
    return text # Returns the concatenated text of all PDFs

# Function to split the text into parts that are easier to manage and process
def get_text_chunks(text):
    # Configure the text splitter to split the text into chunks, each up to 10,000 characters long, with an overlap of 1,000 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text) # Split the text into parts (chunks)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)  # Load the embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create a FAISS vector store from the text blocks
    st.session_state['vector_store'] = vector_store  # Store the vector store in the session state instead of saving it locally.
    # Local option
    # vector_store.save_local("faiss_index") # Save the vector store locally for later use

# Function to create a chain of conversational responses using a template
def get_conversational_chain(api_key):
    # Detailed instructions on chatbot operation and response format - in Brazilian Portuguese and for types of documents of public interest in the country
    instructions = """
    Sempre termine as respostas com "Todas as informações precisam ser checadas com as fontes das informações".
    Você é um assistente para analisar documentos .PDF com um contexto jornalístico. Por exemplo: 
    documentos da Lei nº 12.527/2011 (Lei de Acesso à Informação), contratos públicos, processos judiciais etc.
    Explique os passos de forma simples. Mantenha as respostas concisas e inclua links para ferramentas, pesquisas e páginas da Web das quais você cita informações.
    Quando o usuário pedir recursos, certifique-se de que cita as ligações para a investigação ou exemplos.
    Se eu lhe pedir para resumir uma passagem, escreva-a ao nível universitário
    Quando for relevante, divida os tópicos em partes mais pequenas e fáceis de entender. Quando estiver a editar um texto para mim, 
    faça uma lista de pontos com todas as alterações no final.
    Antes de começar uma tarefa, respire fundo e execute-a passo a passo.
    Seja claro, breve e ordenado nas respostas. Seja direto e claro.
    Evite opiniões e tente ser neutro.
    Se baseie nas classes processuais do Direito no Brasil que estão neste site -  https://www.cnj.jus.br/sgt/consulta_publica_classes.php
    Se não souber a resposta diga que não sabe
    
    Quando analisar documentos de processos judiciais procure priorizar nos resumos:
    - Verifique se é uma petição inicial, decisão ou sentença
    - Faça uma apresentação da ação e de suas partes: breve síntese do processo e de seus pólos ativo e passivo, indicando o tipo de processo, advogados e magistrados
    - Motivos que levaram o autor a ajuizar a ação: explicação sucinta do porquê de o autor ter proposto a ação em face do réu
    - O que o autor requereu com a ação: citação de todos os requerimentos e pedidos que o autor realizou através do processo, tanto liminarmente quanto no mérito
    - Resultado das decisões: exposição do que foi decidido nas decisões interlocutórias (liminares) e na sentença
    - Status: Ao final do resumo é importante que se indique o status do processo

    Quando analisar documentos de licitações ou contratos públicos, saiba isto:
    As licitações públicas no Brasil são um processo administrativo formal e transparente, utilizado pela administração pública para contratar bens, serviços ou obras. O objetivo principal é garantir a aplicação dos princípios da administração pública, como impessoalidade, publicidade, economicidade, eficiência, moralidade e igualdade.
Etapas do Processo Licitatório:
    Fase Preparatória: A administração pública define a necessidade de contratar um bem, serviço ou obra e elabora o edital da licitação, que contém todas as regras e procedimentos do processo.
    Divulgação do Edital: O edital é publicado em diário oficial e em outros meios de comunicação, para que empresas e pessoas interessadas possam tomar conhecimento e participar da licitação.
    Apresentação de Propostas: As empresas e pessoas interessadas apresentam suas propostas, que devem atender às exigências do edital.
    Julgamento das Propostas: Uma comissão de licitação analisa as propostas e seleciona a mais vantajosa para a administração pública, levando em consideração critérios como preço, qualidade, prazo e experiência do licitante.
    Adjudicação e Contratação: A administração pública adjudica o contrato à empresa vencedora da licitação e formaliza o contrato.
    Execução do Contrato: A empresa vencedora executa o contrato, fornecendo o bem, serviço ou obra contratado.
    Fiscalização e Recebimento: A administração pública fiscaliza a execução do contrato e recebe o bem, serviço ou obra, após verificar se está de acordo com o contratado.
Modalidades de Licitação:
A Lei de Licitações e Contratos (Lei nº 14.133/2021) prevê diversas modalidades de licitação, cada uma com suas características e procedimentos específicos. As modalidades mais comuns são:
    Pregão: Modalidade mais rápida e simples, utilizada para compras de bens e serviços de valor pequeno ou médio.
    Concorrência: Modalidade utilizada para compras de bens e serviços de valor elevado, obras públicas e serviços de engenharia.
    Tomada de Preços: Modalidade utilizada para contratação de obras públicas e serviços de engenharia de valor pequeno ou médio.
    Concurso: Modalidade utilizada para a seleção de projetos técnicos, científicos ou artísticos.
    Leilão: Modalidade utilizada para a venda de bens públicos, como imóveis e veículos.
Importância das Licitações Públicas:
As licitações públicas são importantes para garantir:
    Transparência: O processo licitatório é público e transparente, o que permite que qualquer cidadão possa acompanhar as etapas do processo e fiscalizar a utilização dos recursos públicos.
    Competitividade: As empresas e pessoas interessadas competem entre si para apresentar a proposta mais vantajosa para a administração pública, o que garante a obtenção de melhores preços e serviços.
    Eficiência: A administração pública contrata o bem, serviço ou obra que melhor atende às suas necessidades, com o melhor custo-benefício.
    Moralidade: O processo licitatório contribui para prevenir a corrupção e o favorecimento de empresas ou pessoas específicas.
Onde Obter Mais Informações:
Para mais informações sobre licitações públicas no Brasil, você pode consultar os seguintes sites:
    Portal da Transparência: https://portaldatransparencia.gov.br/
    Ministério da Economia: https://www.gov.br/economia/pt-br
    Tribunal de Contas da União: https://www.TCU.gov.br/

    No Brasil, o termo mais comum para se referir a licitações sem concorrentes é inexigibilidade ou dispensa de licitação.
Este termo está previsto na Lei de Licitações e Contratos (Lei nº 14.133/2021), que define as hipóteses em que a administração pública pode contratar bens, serviços ou obras sem a necessidade de realizar licitação.
Outras expressões que podem ser utilizadas para se referir a licitações sem concorrentes:
    Licitação deserta: Essa expressão é utilizada para indicar que a licitação não teve nenhum participante, ou seja, nenhuma empresa apresentou proposta.
    Licitação única: Essa expressão é utilizada para indicar que apenas uma empresa apresentou proposta, o que significa que a licitação não foi competitiva.
    Contratação direta: Essa expressão é utilizada para se referir à modalidade de contratação que a administração pública pode utilizar em casos de inexigibilidade de licitação.
É importante ressaltar que a inexigibilidade de licitação não é sinônimo de falta de transparência ou de controle. A Lei de Licitações e Contratos estabelece diversas regras e procedimentos que a administração pública deve seguir para garantir a lisura e a economicidade na contratação de bens, serviços ou obras, mesmo em casos de inexigibilidade de licitação.
Alguns exemplos de situações em que a inexigibilidade de licitação pode ser aplicada:
    Aquisição de bens ou serviços com fornecedor único: Quando existe apenas um único fornecedor para o bem ou serviço que a administração pública precisa adquirir, a licitação torna-se inviável.
    Contratação em caso de emergência: Em situações de urgência ou calamidade pública, a administração pública pode contratar bens, serviços ou obras sem licitação, para garantir o atendimento imediato das necessidades da população.
    Contratação de serviços artísticos ou culturais: A Lei de Licitações e Contratos permite a contratação direta de artistas ou profissionais de cultura, sem a necessidade de licitação, para a realização de obras de arte, espetáculos ou outros eventos culturais.

    Os documentos que trazem respostas de um pedido de acesso à informação pela Lei nº 12.527/2011 (LAI - Lei de Acesso à Informação) normalmente possuem:
- Nome do órgão público
- Nomes dos setores do órgão público responsáveis pelas informações
- Assunto
- Resumo da demanda
- Informações complementares
- Nomes das pessoas responsáveis pela resposta do pedido da LAI
- Data da resposta
É importante que a análise dos documentos que citam a LAI feita por este chatbot tragam informações:
- Data
- Protocolo NUP
- Nome do órgão público
- Nomes das pessoas responsáveis pela resposta do pedido da LAI
- Data da resposta
- E demais informações de resumo que demonstrem se o pedido da LAI foi totalmente atendido, parcialmente ou foi negado

    """

    prompt_template = f"""
    {instructions}
    Contexto:\n{{context}}\n
    Questão: \n{{question}}\n

    Resposta:
    """
    
    # Load the conversational AI model with the specified security settings
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", 
                                   temperature=0,
                                   candidate_count=1,
                                   safety_settings = {
                                       HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                       HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                       HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                       HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                       HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                                      }, 
                                  api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) # Configure the prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) # Load the Q&A string with the template and prompt
    return chain

# Function to process user input and generate responses
def user_input(user_question, api_key):
    if 'history' not in st.session_state:  # Initialize session history, if not already present
        st.session_state.history = []

    # Check if vector store not found
    if 'vector_store' not in st.session_state:
        st.error("O armazenamento de vetores não foi encontrado. Faça upload e processe os documentos PDF primeiro.")
        return

    vector_store = st.session_state['vector_store']
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)  # Load embeddings
    docs = vector_store.similarity_search(user_question)  # Perform similarity search with user question
    chain = get_conversational_chain(api_key)  # Get the conversation chain

    # Get a response from the chatbot
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.history.append({"question": user_question, "answer": response["output_text"]})  # Attach the interaction to the history
    
    for interaction in st.session_state.history:
        st.write(f":bust_in_silhouette: {interaction['question']}")  # Show the question
        st.write(f"🤖{interaction['answer']}")  # Show the answer

# Main function for configuring the Streamlit application
def main():   
    st.set_page_config(page_title="Chatbot com vários PDFs", page_icon=":books:")
    st.header("Chatbot com vários PDFs :books:")

    with st.sidebar:
        st.title("Menu:")
        st.markdown("""
        - Se encontrar erros de processamento, reinicie com F5. Utilize arquivos .PDF com textos não digitalizados como imagens.
        - Para recomeçar uma nova sessão pressione F5.
        """)
        
        st.warning(
            """
        Atenção: Os documentos que você compartilhar com o modelo de IA generativa podem ser usados pelo Gemini para treinar o sistema. Portanto, evite compartilhar documentos PDF que contenham:
        1. Dados bancários e financeiros
        2. Dados de sua própria empresa
        3. Informações pessoais
        4. Informações de propriedade intelectual
        5. Conteúdos autorais
        
        E não use IA para escrever um texto inteiro! O auxílio é melhor para gerar resumos, filtrar informações ou auxiliar a entender contextos - que depois devem ser checados. Inteligência Artificial comete erros (alucinações, viés, baixa
qualidade, problemas éticos)!
        
        Este projeto não se responsabiliza pelos conteúdos criados a partir deste site.
        """
            )
        st.sidebar.title("Sobre este app")
        st.sidebar.info(
            "Este aplicativo foi desenvolvido por Reinaldo Chaves. "
            "Para mais informações, contribuições e feedback, visite o repositório do projeto: "
            "[GitHub](https://github.com/reichaves/chatgeminipdfs)."
        )

    
    # Create a new loop if there isn't an existing one
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Field to enter the Gemini API Key
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []

    if not st.session_state.api_key:
        st.write("Digite sua API Key do Gemini")
        api_key = st.text_input("API Key do Gemini", type="password")
        st.markdown(
            f'<p style="font-size:18px;">Veja como obter uma API Key neste <a href="https://ai.google.dev/gemini-api/docs/api-key?hl=pt-br">site</a>!</p>',
            unsafe_allow_html=True)
        if api_key:
            st.session_state.api_key = api_key

    # Upload PDF documents
    if not st.session_state.uploaded_pdfs:
        st.write("Por favor, faça o upload e processe os documentos PDF para ativar o chat")
        pdf_docs = st.file_uploader("Carregar PDFs", type=["pdf"], accept_multiple_files=True)
        if pdf_docs:
            st.session_state.uploaded_pdfs = pdf_docs

    if st.session_state.api_key and st.session_state.uploaded_pdfs:
        genai.configure(api_key=st.session_state.api_key)
        #st.write(f"Chave API fornecida: {api_key}")  # Adding a debug log
        
        with st.sidebar:           
            if st.session_state.uploaded_pdfs:
                with st.spinner("Processando..."):
                    raw_text = get_pdf_text(st.session_state.uploaded_pdfs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, st.session_state.api_key)
                    st.success("Done")
                    st.session_state['docs_processed'] = True
            else:
                st.error("Por favor, faça o upload de pelo menos um arquivo PDF antes de processar.")
    
        if st.session_state['docs_processed']:
            user_question = st.text_input("Faça perguntas para 'entrevistar' o PDF (por exemplo, processos judicias, contratos públicos, respostas da LAI etc). Se citar siglas nas perguntas coloque - a sigla e o seu significado. Atenção: Todas as respostas precisam ser checadas!", key="user_question_input")
            if user_question:
                user_input(user_question, st.session_state.api_key)
    
if __name__ == "__main__":
    main()
