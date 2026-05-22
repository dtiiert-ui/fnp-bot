import streamlit as st
import uuid
from supabase import create_client, Client
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ⚙️ Настройка страницы
st.set_page_config(page_title="ФНП СРД – консультант", page_icon="⚖️")
st.title("📘 Консультант по ФНП СРД")
st.caption("Задайте вопрос по оборудованию под давлением. Ответ – строго по тексту документа.")

# 🔐 Читаем секреты
api_key = st.secrets["DEEPSEEK_API_KEY"]
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# 🆔 Управление идентификатором пользователя (сохраняется в URL)
query_params = st.query_params
if "user_id" not in query_params:
    user_id = str(uuid.uuid4())
    st.query_params.clear()
    st.query_params.update({"user_id": user_id})
    st.stop()
else:
    user_id = query_params["user_id"]

# 📎 Функция инициализации RAG-конвейера
@st.cache_resource
def load_rag_chain():
    file_path = "ФНП СРД.docx"
    loader = Docx2txtLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0,
    )

    system_prompt = (
        "Ты — эксперт по промышленной безопасности, отвечающий строго по загруженному документу «ФНП СРД.docx».\n"
        "Если в запросе присутствует «История диалога», используй её только для понимания уточняющих вопросов.\n"
        "Ответ ВСЕГДА формируй на основе предоставленных ниже фрагментов документа.\n"
        "Правила:\n"
        "1. Используй ТОЛЬКО предоставленный контекст.\n"
        "2. Сначала дай краткий ответ своими словами, объясняя суть простым языком.\n"
        "3. После этого перечисли конкретные пункты (например, «п. 223»), которые относятся к вопросу, и приведи краткую цитату из документа для каждого.\n"
        "4. Если в контексте нет информации, напиши: «В документе не указано».\n"
        "5. Не придумывай ничего от себя.\n"
        "6. Строго соблюдай перечни должностных лиц, оборудования и других сущностей, указанные в пунктах.\n"
        "Контекст:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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

# 📥 Загружаем историю из Supabase
if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        res = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at").execute()
        if res.data:
            for row in res.data:
                st.session_state.messages.append({"role": row["role"], "content": row["content"]})
    except Exception:
        pass

# 💬 Отображаем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🧠 Чекбокс для включения контекста диалога
with st.sidebar:
    use_context = st.checkbox("Учитывать контекст диалога", value=False,
                              help="Если включено, бот будет понимать уточняющие вопросы, но строго по документу")

# 🔄 Обработка вопроса
if prompt := st.chat_input("Введите ваш вопрос по ФНП СРД"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Сохраняем вопрос в Supabase
    try:
        supabase.table("messages").insert({"user_id": user_id, "role": "user", "content": prompt}).execute()
    except Exception:
        pass

    with st.chat_message("assistant"):
        with st.spinner("Ищу в документе..."):
            history_context = ""
            if use_context:
                recent_msgs = st.session_state.messages[:-1]
                if len(recent_msgs) > 0:
                    for msg in recent_msgs[-3:]:
                        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                        history_context += f"{role}: {msg['content']}\n"
                    history_context = f"История диалога:\n{history_context}\n"

            full_query = history_context + "Текущий вопрос: " + prompt if history_context else prompt
            answer = qa_chain.invoke(full_query)
            st.markdown(answer)

    # Сохраняем ответ в Supabase
    st.session_state.messages.append({"role": "assistant", "content": answer})
    try:
        supabase.table("messages").insert({"user_id": user_id, "role": "assistant", "content": answer}).execute()
    except Exception:
        pass
