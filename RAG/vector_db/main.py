print(f"✅ main.py đang chạy từ: {__file__}")
import os
import json
import logging
import warnings
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from langchain.schema import Document, BaseRetriever
from pydantic import BaseModel, Field
from typing import List, Any
import gc
from search_web import test_search

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r') as config_file:
            print(f"\n✅ Đang dùng config.json tại: {config_path}\n")
            config = json.load(config_file)
            logger.debug(f"Loaded config from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_model(model_type="llama"):
    try:
        logger.debug(f"Loading model type: {model_type}")
        if model_type == "llama":
            if torch.cuda.is_available():
                clear_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "..", "Fine_tuning", "models")
            
            logger.debug(f"Loading PEFT config from {adapter_path}")
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            logger.debug(f"Base model name: {base_model_name}")
            
            quant_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                device="cuda"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
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
                truncation=True
            )
            logger.debug("Pipeline created")
            
            class LlamaWrapper:
                def __init__(self, pipe):
                    self._pipe = pipe
                
                def clean_gpu_memory(self):
                    clear_gpu_memory()
                
                def __call__(self, prompt):
                    return self._pipe(prompt)[0]['generated_text']
            
            return LlamaWrapper(pipe)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_vector_db(collection_name):
    try:
        original_collection_name = collection_name 
        config=load_config()
        raw_path=config.get("chroma_db_path")
        if raw_path:
            if os.path.isabs(raw_path):
                persist_directory=raw_path
            else:
                base_dir=os.path.dirname(os.path.abspath(__file__))
                persist_directory=os.path.abspath(os.path.join(base_dir,raw_path))
        else:
            persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
        logger.debug(f"Loading vector DB from {persist_directory}")
        
        print(f"\n🐞 ĐANG GỌI load_vector_db() từ: {__file__}")

        if collection_name == "qa":
            collection_name = "qa_data"
        
        if not os.path.exists(persist_directory):
            logger.error(f"Persist directory does not exist: {persist_directory}")
            raise FileNotFoundError(f"Persist directory does not exist: {persist_directory}")
            
        collection_path = os.path.join(persist_directory, collection_name)
        qa_path = os.path.join(persist_directory, "qa_data") 

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        logger.debug("Embedding model initialized: sentence-transformers/all-MiniLM-L6-v2")

        domain_db = Chroma(
            collection_name=collection_name,
            persist_directory=collection_path,
            embedding_function=embedding_model
        )

        qa_db = None
        if os.path.exists(qa_path):
            qa_db = Chroma(
                collection_name="qa_data",
                persist_directory=qa_path,
                embedding_function=embedding_model
            )
            logger.debug("QA DB loaded")

        domain_collection = domain_db.get()
        logger.debug(f"Domain DB ({collection_name}) contains {len(domain_collection['ids'])} documents")

        if qa_db:
            qa_collection = qa_db.get()
            logger.debug(f"Q_A DB contains {len(qa_collection['ids'])} documents")
            if len(qa_collection['ids']) == 0:
                logger.warning("No documents found in Q_A DB")
        else:
            logger.warning("QA DB is not available or not found.")

        if len(domain_collection['ids']) == 0:
            logger.warning(f"No documents found in domain DB: {collection_name}")

        return domain_db, qa_db

    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise

def classify_question_domain(question):
    question = question.lower()

    qa_indicators = ["là gì", "làm sao", "như thế nào", "bao nhiêu", "khi nào", "ở đâu", "tại sao", "mục đích"]
    if any(ind in question for ind in qa_indicators):
        logger.debug("Câu hỏi mang tính hỏi đáp, phân vào domain 'qa'")
        return "qa"
    
    domain_keywords = {
        "gioi_thieu": ["giới thiệu", "tổng quan", "bách khoa", "trường"],
        "de_an": ["đề án", "phương thức", "chỉ tiêu", "phương án"],
        "tai_nang": ["xét tuyển tài năng", "năng khiếu", "portfolio", "giải thưởng"],
        "diem_chuan": ["điểm chuẩn", "điểm trúng tuyển", "điểm xét tuyển", "năm 2024"],
        "danh_gia_tu_duy": ["đánh giá tư duy", "ĐGTD", "kỳ thi tư duy", "thi tư duy"],
        "ngoai_ngu": ["ielts", "toefl", "chứng chỉ", "tiếng anh", "xác thực"],
        "huong_nghiep": ["ngành", "hướng nghiệp", "định hướng", "chọn ngành"]
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in question for keyword in keywords):
            logger.debug(f"Question classified to domain: {domain}")
            return domain
    logger.debug("Question classified to default domain: qa")
    return "qa"

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logger.debug("GPU memory cleared")