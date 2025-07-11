import os
import torch
import logging
import sys
from typing import List, Tuple, Set
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import (
    classify_question_domain,
    load_vector_db,
    get_relevant_chunks,
    load_config
)
from search_web import test_search,google_search,calculate_relevance_score

class ModelManager:
    _instance = None
    _pipe = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance
    
    def __init__(self):
        if self._pipe is not None:
            return
        self._pipe = self.load_fine_tuned_model()

    @staticmethod
    def clean_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Cleaned GPU memory")

    def load_fine_tuned_model(self):
        try:
            if torch.cuda.is_available():
                self.clean_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "models")
            logger.debug(f"Loading PEFT config from {adapter_path}")
            
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            logger.debug(f"Base model name: {base_model_name}")
            
            quant_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                device="cuda"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.debug("Base model loaded")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Tokenizer loaded")
            
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.debug("Adapter weights loaded")
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False,
                truncation=True,
                max_length=1500
            )
            logger.debug("Pipeline created")
            
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise

def get_data_with_context(question: str) -> Tuple[str, str, Set[str]]:
    try:
        logger.debug(f"Processing question: {question}")
        config = load_config()
        domain = classify_question_domain(question)
        web_results = google_search(question, config['GOOGLE_API_KEY'], config['SEARCH_ENGINE_ID'], num_results=3)
        
        best_web_context=""
        best_web_url=None
        if web_results:
            best_result=max(web_results,key=lambda x:calculate_relevance_score(question,x))
            best_web_context=best_result.get("extracted_text","").strip()
            best_web_url=best_result.get("link","")
        
        domain_db, qa_db = load_vector_db(domain)
        logger.debug(f"Loaded vector DB for domain: {domain}")
        
        documents = get_relevant_chunks(question, domain_db, qa_db, k=10)
        logger.debug(f"Retrieved {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents retrieved, returning fallback")
            return None, best_web_url, None

        documents.sort(key=lambda x: float(x.metadata.get('score', 0)), reverse=True)
        logger.debug("Documents sorted by relevance score")
        
        expanded_context_parts = []
        sources = set()
        
        if best_web_context:
            expanded_context_parts.append(f"Data from web: {best_web_context[:1000]}")

        for i, doc in enumerate(documents):
            current_metadata = doc.metadata
            source = current_metadata.get('source')
            page = current_metadata.get('page', 'Unknown')
            score = current_metadata.get('score', 'Unknown')
            title = current_metadata.get('title', os.path.basename(source) if source else 'Unknown')
            content=doc.page_content[:1000]
            logger.debug(f"Document {i+1}: Source={source}, Page={page}, Score={score}, Content={content[:200]}...")

            if source and source != 'Unknown':
                # Clean title: remove underscores, capitalize, and remove extension
                clean_title = title.replace('-', ' ').replace('_', ' ').replace('.pdf', '').replace('.docx', '').title()
                source_info = f"{clean_title} (trang {page})" if page != 'Unknown' else clean_title
                sources.add(source_info)
            
            expanded_context_parts.append(f"Đoạn {i+1}: {content}")

        context = "\n\n".join(expanded_context_parts)
        logger.debug(f"Context created, length: {len(context)}")
        
        return context, best_web_url, sources

    except Exception as e:
        logger.error(f"Error in get_data_with_context: {str(e)}")
        return None, None, None

def generate_response(prompt: str) -> str:
    try:
        logger.debug("Generating response")
        model_manager = ModelManager()
        model_manager.clean_gpu_memory()
        
        pipe = model_manager._pipe
        response = pipe(prompt)[0]['generated_text']
        logger.debug("Response generated")
        
        model_manager.clean_gpu_memory()
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def create_prompt_llama(question: str, context: str) -> str:
    prompt = f"""Bạn là một giáo viên trong phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
Bạn cần trả lời câu hỏi của học sinh bằng ngôn ngữ thân thiện, rõ ràng và chính xác.
Nếu có nhiều phương án, hãy trình bày ưu nhược điểm rõ ràng để người dùng dễ lựa chọn.
Nếu không đủ thông tin, hãy khuyến khích người dùng tra thêm trên website: https://ts.hust.edu.vn/.

### Câu hỏi:
{question}

### Ngữ cảnh:
{context}

### Trả lời:
"""
    logger.debug("Prompt created")
    return prompt

def main():
    try:
        while True:
            question = input("\nNhập câu hỏi (hoặc 'q' để thoát): ")
            if question.lower() == 'q':
                break
            
            context, web_url, sources = get_data_with_context(question)
            
            if context is None:
                logger.warning(f"No context found for question: {question}")
                print(f"\nKhông tìm thấy thông tin phù hợp. Bạn có thể tham khảo thêm tại: {web_url}")
                continue
            
            prompt = create_prompt_llama(question, context)
            response = generate_response(prompt)
            
            print("\nCâu trả lời:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            if sources:
                print("\nNguồn tham khảo:")
                for source in sources:
                    print(f"- {source}")
            
            if web_url:
                print(f"\nTham khảo thêm tại: {web_url}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()