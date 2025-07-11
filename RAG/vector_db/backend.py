from main import (
    load_config, 
    load_model, 
    load_vector_db, 
    classify_question_domain,
    clear_gpu_memory
)
from model import (
    create_prompt,
    create_qa_chain,
    get_relevant_chunks,
    load_config,
    CustomRetriever
)
from search_web import test_search
from chatbot import generate_response, create_prompt_llama, get_data_with_context
import logging
import os
import torch
import gc

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChatBackend:
    def __init__(self):
        self.qa_chain = None
        self.current_model = None
        self.config = load_config()
        self.chat_history = []
        
    @staticmethod
    def clean_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Cleaned GPU memory")

    def setup_qa_chain(self, embedding_model_choice="HuggingFace", llm_model_choice="llama"):
        try:
            logger.debug(f"Setting up QA chain with embedding: {embedding_model_choice}, LLM: {llm_model_choice}")
            logger.debug("QA chain setup complete")
            return None 
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise

    def get_file_name(self, file_path):
        if not file_path or file_path == "Unknown":
            return "Unknown"
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        return name_without_ext.replace('_', ' ').title()

    def get_domain_name(self, domain):
        domain_names = {
            'gioi_thieu': 'gioi_thieu',
            'de_an': 'de_an',
            'tai_nang': 'tai_nang',
            'diem_chuan': 'diem_chuan',
            'danh_gia_tu_duy': 'danh_gia_tu_duy',
            'ngoai_ngu': 'ngoai_ngu',
            'huong_nghiep': 'huong_nghiep',
            'qa': 'qa'
        }
        return domain_names.get(domain, domain)

    def get_chat_response(self, query, model_choice="llama"):
        try:
            self.clean_gpu_memory()
            domain = classify_question_domain(query)
            logger.info(f"Classified domain: {domain}")
            logger.debug(f"[DEBUG] Calling load_vector_db with domain={domain}")
            context, web_url, sources = get_data_with_context(query)
            logger.debug(f"[DEBUG] domain_db loaded: {domain}")
            logger.debug(f"Context length: {len(context) if context else 0}, Web URL: {web_url}, Sources: {sources}")
            
            if not context:
                logger.warning(f"No context found for query: {query}")
                fallback_message = (
                    f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                )
                return {
                    "answer": fallback_message,
                    "sources": [],
                    "domain": domain
                }
            
            prompt = create_prompt_llama(query, context)
            logger.debug("Prompt created for Llama")
            response = generate_response(prompt)
            logger.debug("Response generated")
            
            answer = response.strip()
            
            if web_url:
                answer += f"\n\nBạn có thể tham khảo thêm thông tin mới nhất tại: {web_url}"
            
            self.clean_gpu_memory()
            return {
                "answer": answer,
                "sources": list(sources),
                "domain": self.get_domain_name(domain)
            }

        except Exception as e:
            logger.error(f"Error in get_chat_response for Llama: {str(e)}")
            return {
                "answer": f"Đã xảy ra lỗi: {str(e)}",
                "sources": [],
                "domain": domain
            }
        
    def cleanup(self):
        try:
            clear_gpu_memory()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 