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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    instructions = """
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
    Se não souber a resposta diga que não sabe
    Sempre ressalte que todas as informações precisam ser checadas com as fontes

    Quando analisar documentos de processos judiciais procure priorizar:
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

    """

    prompt_template = f"""
    {instructions}
    Context:\n{{context}}\n
    Question: \n{{question}}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", 
                                   temperature=0.3,
                                   safety_settings = {
                                       HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                                       HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                                       HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                       HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                                       HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                                      })
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if 'history' not in st.session_state:
        st.session_state.history = []

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.history.append({"question": user_question, "answer": response["output_text"]})
    
    for interaction in st.session_state.history:
        st.write(f":robot: {interaction['question']}")
        st.write(f":bust_in_silhouette: {interaction['answer']}")

def main():
    st.set_page_config(page_title="Chatbot com vários PDFs", page_icon=":books:")
    st.header("Chatbot com vários PDFs :books:")
    user_question = st.text_input("Faça perguntas para 'entrevistar' o PDF (por exemplo, processos judicias, contratos públicos, respostas da LAI etc.)")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Faça o upload de seus arquivos PDF e Clique no botão Processar", accept_multiple_files=True)
        if st.button("Processar"):
            with st.spinner("Processando..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
