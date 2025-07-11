import streamlit as st
from backend import ChatBackend
from main import classify_question_domain, load_vector_db, clear_gpu_memory
import os
from datetime import datetime
import logging
import warnings
import sys
import time
import streamlit.watcher.local_sources_watcher

# Suppress torch.classes warning
streamlit.watcher.local_sources_watcher.get_module_paths = lambda x: []

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'backend' not in st.session_state:
    st.session_state.backend = ChatBackend()
if 'current_model_config' not in st.session_state:
    st.session_state.current_model_config = {
        'embedding_model': "HuggingFace Embeddings",
        'llm_model': "llama"
    }

with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSr2-hIfIOEB2-bok5hY83nxSQhmqOr0ANvTw&s",
            width=150
        )
    
    st.title("Model Configuration")
    
    embedding_model = st.selectbox(
        "Choose Embedding Model",
        ["HuggingFace Embeddings"],
        index=0
    )
    
    llm_model = st.selectbox(
        "Choose LLM Model",
        ["llama"],
        index=0,
        help="llama uses local model from @models folder"
    )
    
    if st.button("Apply Configuration"):
        with st.spinner("Loading model... This may take a few moments."):
            try:
                model_path = "DATN_ChatbotHUST/Fine_tuning/models"
                required_files = [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "tokenizer.json",
                    "special_tokens_map.json",
                    "tokenizer_config.json"
                ]
                
                for file in required_files:
                    if not os.path.exists(os.path.join(model_path, file)):
                        raise ValueError(f"Không tìm thấy file {file} trong thư mục model")

                st.session_state.backend.setup_qa_chain(
                    embedding_model_choice=embedding_model,
                    llm_model_choice=llm_model
                )
                st.session_state.current_model_config = {
                    'embedding_model': embedding_model,
                    'llm_model': llm_model
                }
                st.success(f"""
                Configuration applied successfully:
                - Embedding Model: {embedding_model}
                - Language Model: {llm_model}
                """)
            except Exception as e:
                st.error(f"Error applying configuration: {str(e)}")
                logger.error(f"Configuration error: {str(e)}")

    st.markdown("---")
    st.subheader("Conversations")
    
    if st.button("🆕 New Chat"):
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.conversations[new_id] = {
            "title": "New Chat",
            "messages": [],
            "model_config": st.session_state.current_model_config.copy()
        }
        st.session_state.current_conversation_id = new_id
        st.session_state.chat_history = []
        
        st.session_state.backend = ChatBackend()
        
        try:
            with st.spinner("Setting up model..."):
                st.session_state.backend.setup_qa_chain(
                    llm_model_choice=st.session_state.current_model_config['llm_model'],
                    embedding_model_choice=st.session_state.current_model_config['embedding_model']
                )
        except Exception as e:
            st.error(f"Error setting up model: {str(e)}")
            logger.error(f"Model setup error: {str(e)}")
        
        st.rerun()

    st.markdown("""
        <style>
            [data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
                max-height: 300px;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("Chat History", expanded=True):
        for conv_id, conv_data in st.session_state.conversations.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                display_title = conv_data.get('title', 'New Chat')
                if st.button(f"📝 {display_title}", key=f"conv_{conv_id}"):
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.chat_history = conv_data['messages']
                    try:
                        st.session_state.backend.setup_qa_chain(
                            llm_model_choice=st.session_state.current_model_config['llm_model'],
                            embedding_model_choice=st.session_state.current_model_config['embedding_model']
                        )
                    except Exception as e:
                        st.error(f"Error setting up model: {str(e)}")
                        logger.error(f"Model setup error: {str(e)}")
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    del st.session_state.conversations[conv_id]
                    if st.session_state.current_conversation_id == conv_id:
                        st.session_state.current_conversation_id = None
                        st.session_state.chat_history = []
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 16px; font-style: italic;'>One love, one future.</p>", 
        unsafe_allow_html=True
    )

st.title("💬 HUST Admissions Consulting Assistant")

if st.session_state.current_conversation_id:
    current_chat = st.session_state.conversations[st.session_state.current_conversation_id]
    st.subheader(f"Current Chat: {current_chat['title']}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.markdown("""
    ### 🧠 Gợi ý câu hỏi thường gặp:
    """)

    faq_questions = [
        "Điểm chuẩn ngành Khoa học máy tính năm 2024 là bao nhiêu?",
        "Thi đánh giá tư duy gồm những môn gì?",
        "Làm sao để đăng ký xét tuyển tài năng?",
        "Chứng chỉ IELTS có được quy đổi điểm không?",
        "Ngành Kỹ thuật Điều khiển - Tự động hóa học gì?"
    ]

    faq_cols = st.columns(len(faq_questions))

    for i, question in enumerate(faq_questions):
        if faq_cols[i].button(question, key=f"faq_{i}"):
            st.session_state.faq_question_clicked = question

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            st.markdown("**Nguồn tham khảo:**")
            
            for source in message["sources"]:
                if isinstance(source, dict):
                    st.markdown(f"**{source['name']}**")
                    
                    for i, vector in enumerate(source['vectors'], 1):
                        st.markdown(f"{vector['content']}")
                        st.markdown("---")
                else:
                    st.markdown(f"**{source}**")

user_input = st.chat_input("Hãy đặt câu hỏi về tuyển sinh...")

if 'faq_question_clicked' in st.session_state:
        user_input = st.session_state.faq_question_clicked
        del st.session_state['faq_question_clicked']

if user_input and st.session_state.current_conversation_id:
    current_chat = st.session_state.conversations[st.session_state.current_conversation_id]
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    if len(current_chat["messages"]) == 0:
        title = user_input if len(user_input) <= 50 else user_input[:47] + "..."
        current_chat["title"] = title
        current_chat["messages"] = st.session_state.chat_history.copy()
        st.session_state.conversations[st.session_state.current_conversation_id] = current_chat
        st.rerun()
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            logger.debug(f"Processing user input: {user_input}")
            status_placeholder.info("🤔 Đang phân tích câu hỏi của bạn...")
            time.sleep(1)
            
            if ('model_config' not in current_chat or 
                current_chat['model_config'] != st.session_state.current_model_config):
                status_placeholder.info("⚙️ Đang cấu hình lại model...")
                time.sleep(1)
                st.session_state.backend = ChatBackend()
                st.session_state.backend.setup_qa_chain(
                    llm_model_choice=st.session_state.current_model_config['llm_model'],
                    embedding_model_choice=st.session_state.current_model_config['embedding_model']
                )
                current_chat['model_config'] = st.session_state.current_model_config.copy()
            
            status_placeholder.info("🔍 Đang tìm kiếm thông tin liên quan...")
            time.sleep(1)
            
            result = st.session_state.backend.get_chat_response(
                user_input,
                model_choice=current_chat['model_config']['llm_model']
            )
            
            status_placeholder.info("📝 Đang tổng hợp câu trả lời...")
            
            status_placeholder.empty()
            
            message_placeholder.markdown(result['answer'])

            avg_score = None
            num_sources = len(result.get('sources', []))
            try:
                score_values = [float(s.split("Độ tương đồng:")[-1].strip())
                                for s in result.get('sources', []) if "Độ tương đồng:" in s]
                if score_values:
                    avg_score = sum(score_values) / len(score_values)
            except:
                pass

            confidence = "Không rõ"
            if avg_score and num_sources:
                if avg_score >= 0.85 and num_sources >= 3:
                    confidence = "Cao"
                elif avg_score >= 0.7:
                    confidence = "Trung bình"
                else:
                    confidence = "Thấp"

            st.markdown(f"**🔍 Độ tin cậy của câu trả lời: `{confidence}`**")
            
            if 'sources' in result and result['sources']:
                st.markdown("**Nguồn tham khảo:**")
                for source in result['sources']:
                    if isinstance(source, dict):
                        st.markdown(f"**{source.get('name', 'Unknown')}**")
                        for i, vector in enumerate(source.get('vectors', []), 1):
                            st.markdown(f"{vector.get('content', '')}")
                            st.markdown("---")
                    else:
                        st.markdown(f"**{source}**")
            
            assistant_response = {
                "role": "assistant",
                "content": result['answer'],
                "sources": result.get('sources', [])
            }
            st.session_state.chat_history.append(assistant_response)
            current_chat["messages"] = st.session_state.chat_history.copy()
            st.session_state.conversations[st.session_state.current_conversation_id] = current_chat
            logger.debug("Response displayed and saved to chat history")

            with st.expander("❗ Báo lỗi nếu câu trả lời chưa đúng hoặc không rõ ràng"):
                report_text = st.text_area("Mô tả vấn đề bạn gặp phải")
                if st.button("Gửi báo lỗi"):
                    try:
                        log_dir = os.path.join(os.path.dirname(__file__), "logs")
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "report_log.txt")

                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write("\n" + "="*50 + "\n")
                            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Câu hỏi: {user_input}\n")
                            f.write(f"Câu trả lời: {result['answer']}\n")
                            f.write(f"Vấn đề được báo cáo: {report_text}\n")

                        st.success("✅ Cảm ơn bạn đã phản hồi. Chúng tôi sẽ cải thiện chatbot tốt hơn!")
                    except Exception as e:
                        st.error(f"Không thể ghi log báo lỗi: {str(e)}")

        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ Có lỗi xảy ra: {str(e)}")
            logger.error(f"Error details: {str(e)}")

def on_shutdown():
    if 'backend' in st.session_state:
        st.session_state.backend.cleanup()

if __name__ == "__main__":
    try:
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*torch.classes.*')
        on_shutdown()
        print(sys.path)
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")