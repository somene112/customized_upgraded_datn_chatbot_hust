from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import os
import json
import logging
import warnings
from search_web import test_search
from main import load_model, load_vector_db, classify_question_domain
from langchain.chains import RetrievalQA

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            logger.debug(f"Loaded config from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def create_prompt():
    template = """Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn, truyền tải đầy đủ thông tin bạn nhận được.
    Hãy nói với vai trò là một thầy cô giáo trong trường, sử dụng ngôn ngữ thân thiện và dễ hiểu.
    
    Nếu không tìm thấy thông tin trong context hoặc thông tin không đầy đủ, hãy trả lời:
    "Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question", "web_url"])
    logger.debug("Prompt template created")
    return prompt

def get_relevant_chunks(question, domain_db, qa_db=None, k=10, score_threshold=0.3):
    try:
        logger.debug(f"Searching for relevant chunks for question: {question}")
        
        domain_collection = domain_db.get()
        logger.debug(f"Domain DB contains {len(domain_collection['ids'])} documents")
        
        if qa_db:
            qa_collection = qa_db.get()
            logger.debug(f"Q_A DB contains {len(qa_collection['ids'])} documents")

        # Search domain DB
        domain_results = []
        try:
            domain_results = domain_db.similarity_search_with_relevance_scores(question, k=20)
            logger.debug(f"Found {len(domain_results)} results in domain DB")
        except Exception as exc:
            logger.warning(f"Error searching domain DB: {str(exc)}")

        # Search QA DB (if available)
        qa_results = []
        if qa_db:
            try:
                qa_results = qa_db.similarity_search_with_relevance_scores(question, k=5)
                logger.debug(f"Found {len(qa_results)} results in Q_A DB")
            except Exception as exc:
                logger.warning(f"Error searching Q_A DB: {str(exc)}")

        # Combine results
        combined_results = domain_results + qa_results

        # Filter by score
        filtered_results = []
        for doc, score in combined_results:
            if score >= score_threshold:
                metadata = doc.metadata.copy()
                metadata.update({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'score': f"{score:.4f}",
                    'doc_id': doc.metadata.get('doc_id', None)
                })
                filtered_results.append(Document(
                    page_content=doc.page_content,
                    metadata=metadata
                ))

        # Thêm các đoạn lân cận từ domain_db
        doc_ids = domain_collection.get("ids", [])
        selected_ids = {doc.metadata.get("doc_id") for doc in filtered_results if doc.metadata.get("doc_id")}

        for doc_id in list(selected_ids):
            try:
                index = doc_ids.index(doc_id)
                if index > 0:
                    prev_doc = domain_db.similarity_search_by_vector(doc_ids[index - 1], k=1)[0]
                    filtered_results.append(prev_doc)
                if index < len(doc_ids) - 1:
                    next_doc = domain_db.similarity_search_by_vector(doc_ids[index + 1], k=1)[0]
                    filtered_results.append(next_doc)
            except Exception as e:
                logger.warning(f"Error retrieving neighbor doc for doc_id {doc_id}: {str(e)}")

        # Sắp xếp kết quả và giới hạn
        filtered_results.sort(key=lambda x: float(x.metadata['score']), reverse=True)
        return filtered_results[:k]

    except Exception as e:
        logger.error(f"Error getting relevant chunks: {str(e)}")
        return []
    
def process_chunks(docs, all_docs, selected_chunks):
    try:
        doc_ids = all_docs.get('ids', [])
        logger.debug(f"Processing {len(docs)} documents, total IDs: {len(doc_ids)}")
        
        for doc in docs:
            doc_id = doc.metadata.get('doc_id')
            if not doc_id:
                logger.warning("Document missing doc_id")
                continue
                
            try:
                current_index = doc_ids.index(doc_id)
                selected_chunks.add(current_index)
                
                if current_index > 0:
                    selected_chunks.add(current_index - 1)
                    
                if current_index < len(doc_ids) - 1:
                    selected_chunks.add(current_index + 1)
                logger.debug(f"Added chunk index {current_index} and neighbors")
            except ValueError:
                logger.warning(f"Doc_id {doc_id} not found in collection")
                continue
    except Exception as e:
        logger.warning(f"Error processing chunks: {str(e)}")

class CustomRetriever(BaseRetriever, BaseModel):
    domain_db: Any = Field(default=None)
    qa_db: Any = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, domain_db, qa_db):
        super().__init__()
        self.domain_db = domain_db
        self.qa_db = qa_db

    def get_relevant_documents(self, query: str) -> List[Document]:
        return get_relevant_chunks(query, self.domain_db, self.qa_db)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")

def create_qa_chain(domain_db,qa_db, model, prompt, web_url):
    try:
        logger.debug("Creating QA chain")
        retriever = CustomRetriever(domain_db,qa_db)
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt.partial(web_url=web_url)
            }
        )
        logger.debug("QA chain created")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise

def get_answer(question, model_type="llama"):
    try:
        logger.debug(f"Processing question: {question}")
        config = load_config()
        
        domain = classify_question_domain(question)
        logger.info(f"Domain classified: {domain}")
        
        model = load_model(model_type)
        domain_db, qa_db = load_vector_db(domain)
        web_url = test_search(question)
        logger.debug(f"Web search URL: {web_url}")
        
        documents = get_relevant_chunks(question, domain_db, qa_db, k=10)
        logger.debug(f"Retrieved {len(documents)} chunks")
        
        print("\nCác chunks được tìm thấy:")
        for i, doc in enumerate(documents, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Nội dung: {doc.page_content}")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
                
        if not documents:
            logger.warning(f"No chunks found for question: {question}")
            fallback_message = (
                f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
            )
            return {
                "answer": fallback_message,
                "source_documents": [],
                "domain": domain
            }
            
        prompt = create_prompt()
        qa_chain = create_qa_chain(domain_db, model, prompt, web_url)
        logger.debug("QA chain initialized")
        
        result = qa_chain({"query": question})
        answer = result["result"]
        
        print("\nCâu trả lời:")
        print(answer)
        print("\nNguồn tham khảo:")
        for doc in result.get("source_documents", []):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('score', 'Unknown')
            print(f"- Nguồn: {source}")
            print(f"  Trang: {page}")
            print(f"  Độ tương đồng: {score}")
            print(f"  Nội dung: {doc.page_content[:200]}...")
        
        return {
            "answer": answer,
            "source_documents": result.get("source_documents", []),
            "domain": domain
        }
        
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise

def get_db_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chroma_db")
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        logger.info(f"Created DB directory: {db_path}")
    return os.path.normpath(db_path)

if __name__ == "__main__":
    while True:
        question = input("\nCâu hỏi: ")
        if question.lower() in ['quit', 'q', 'exit']:
            break
            
        try:
            result = get_answer(question, "llama")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Error in main loop: {str(e)}")