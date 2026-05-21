import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ⚡ Ускорение скачивания моделей
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ⚙️ Настройка страницы
st.set_page_config(page_title="ФНП СРД – консультант", page_icon="⚖️")
st.title("📘 Консультант по ФНП СРД")
st.caption("Задайте вопрос по оборудованию под давлением. Ответ – строго по тексту документа.")

# 🔐 Читаем API-ключ из секретов Streamlit Cloud
api_key = st.secrets["DEEPSEEK_API_KEY"]

# 📎 Функция инициализации RAG-конвейера
@st.cache_resource
def load_rag_chain():
    # 1️⃣ Загружаем документ
    file_path = "ФНП СРД.docx"
    loader = Docx2txtLoader(file_path)
    documents = loader.load()

    # 2️⃣ Разбиваем на фрагменты
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = text_splitter.split_documents(documents)

    # 3️⃣ Эмбеддинги (локальная модель, бесплатно)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4️⃣ Векторная база
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    # 5️⃣ Языковая модель DeepSeek
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0,
    )

    # 6️⃣ Системный промпт
    system_prompt = (
      "Ты — эксперт по промышленной безопасности, отвечающий строго по загруженному документу «ФНП СРД.docx».\n"
    "Правила:\n"
    "1. Используй ТОЛЬКО предоставленный контекст (фрагменты документа).\n"
    "2. Если в контексте нет информации для ответа, напиши: «В документе не указано».\n"
    "3. В ответе ОБЯЗАТЕЛЬНО указывай номера пунктов (например, п. 223) и приводи краткую цитату из документа.\n"
    "4. Отвечай понятным языком, без излишней технической сложности, но точно.\n"
    "5. Не придумывай ничего от себя.\n"
    "6. Если вопрос не относится к оборудованию под давлением, вежливо сообщи, что ты консультируешь только по ФНП СРД.\n"
    "Контекст:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # 7️⃣ Функция форматирования документов в строку
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 8️⃣ Цепочка LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 🚀 Загружаем цепочку
with st.spinner("Загружаю документ и подготавливаю базу знаний..."):
    qa_chain = load_rag_chain()

# 💬 История сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🔄 Обработка вопроса
if prompt := st.chat_input("Введите ваш вопрос по ФНП СРД"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищу в документе..."):
            answer = qa_chain.invoke(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
