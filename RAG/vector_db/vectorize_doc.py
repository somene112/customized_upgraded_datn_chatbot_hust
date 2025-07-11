import os
import json
import logging
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import csv
from langchain.schema import Document
import torch
import glob

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            logger.debug(f"Loaded config from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_txt_docs(folder_path):
    txt_files=glob.glob(os.path.join(folder_path,"**/*.txt"),recursive=True)
    documents=[]
    for txt in txt_files:
        loader=TextLoader(txt,autodetect_encoding=True)
        try:
            documents.extend(loader.load())
        except Exception as e:
            logger.warning(f"Couldn't load file {txt}: {e}")
    logger.debug(f"Loaded {len(documents)} txt documents from {folder_path}")
    return documents

def load_documents_from_folder(folder_path, domain):
    try:
        logger.debug(f"Loading documents from {folder_path} for domain {domain}")
        
        # Load PDF files
        pdf_loader = DirectoryLoader(
            folder_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            use_multithreading=True,
            recursive=True
        )
        
        # Load DOCX files
        docx_loader = DirectoryLoader(
            folder_path,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            use_multithreading=True,
            recursive=True
        )
        
        pdf_docs = pdf_loader.load()
        docx_docs = docx_loader.load()
        txt_docs=load_txt_docs(os.path.join(folder_path,domain))
        documents = pdf_docs + docx_docs + txt_docs
        
        # Assign domain to metadata
        for doc in documents:
            doc.metadata['domain'] = domain
        
        logger.debug(f"Loaded {len(documents)} documents from {folder_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from {folder_path}: {str(e)}")
        return []

def load_qa_from_csv(csv_path):
    try:
        logger.debug(f"Loading Q&A from {csv_path}")
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return []
        
        documents = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Check for 'context', 'question', 'answer' columns
            if 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
                logger.error(f"CSV file {csv_path} missing 'question' or 'answer' columns")
                return []
            
            for row in reader:
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip()
                context = row.get('context', '').strip()
                if question and answer:
                    # Include context in content if available
                    content = f"Bối cảnh: {context}\nCâu hỏi: {question}\nTrả lời: {answer}" if context else f"Câu hỏi: {question}\nTrả lời: {answer}"
                    documents.append(Document(
                        page_content=content,
                        metadata={'source': csv_path, 'domain': 'thong_tin'}
                    ))
        
        logger.debug(f"Loaded {len(documents)} Q&A pairs")
        return documents
    except Exception as e:
        logger.error(f"Error loading Q&A CSV: {str(e)}")
        return []

def vectorize_documents():
    try:
        config = load_config()
        data_path = config.get('data_path', 'data')
        csv_path = config.get('csv_path', 'data/Q&A&C.csv')
        persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            logger.info(f"Created directory: {persist_directory}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        logger.debug("Embedding model initialized: sentence-transformers/all-MiniLM-L6-v2")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        # Define domain mappings based on subdirectory names
        domain_mappings = {
            '1.thong_tin_chung': 'gioi_thieu',
            '2.de_an_tuyen_sinh': 'de_an',
            '3.xet_tuyen_tai_nang': 'tai_nang',
            '4.diem_chuan_tuyen_sinh': 'diem_chuan',
            '5.ky_thi_danh_gia_tu_duy': 'danh_gia_tu_duy',
            '6.xac_thuc_chung_chi_ngoai_ngu': 'ngoai_ngu',
            '7.huong_nghiep': 'huong_nghiep',
        }
        
        # Log all subdirectories in data_path for debugging
        logger.debug(f"Checking subdirectories in {data_path}")
        if os.path.exists(data_path):
            subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            logger.debug(f"Found subdirectories: {subdirs}")
        else:
            logger.error(f"Data directory {data_path} does not exist")
            return
        
        # Initialize domain_docs with an empty list for thong_tin
        domain_docs = {domain: [] for domain in set(domain_mappings.values())}
        domain_docs['qa']=[]
        
        # Load documents from each domain subdirectory and accumulate them
        for subdir, domain in domain_mappings.items():
            subdir_path = os.path.join(data_path, subdir)
            if os.path.exists(subdir_path):
                docs = load_documents_from_folder(subdir_path, domain)
                if docs:
                    domain_docs[domain].extend(docs)  # Extend instead of assign
                    logger.debug(f"Added {len(docs)} documents from {subdir_path} to domain {domain}")
                else:
                    logger.warning(f"No documents loaded from {subdir_path}")
            else:
                logger.warning(f"Subdirectory {subdir_path} does not exist")
        
        # Load Q&A CSV (skipped as per requirement)
        qa_documents = load_qa_from_csv(csv_path)
        if qa_documents:
            logger.debug("Q&A documents loaded")
            domain_docs['qa'].extend(qa_documents)
            pass
        
        txt_documents=load_txt_docs(data_path)
        if txt_documents:
            logger.debug("Other .txt files loaded successfully")
            domain_docs['qa'].extend(txt_documents)
            pass

        # Vectorize documents for each domain
        for domain, docs in domain_docs.items():
            logger.debug(f"Processing domain: {domain}, {len(docs)} documents")
            chunks = []
            for doc in docs:
                split_docs = text_splitter.split_documents([doc])
                for i, split_doc in enumerate(split_docs):
                    split_doc.metadata['doc_id'] = f"{doc.metadata['source']}_{i}"
                    split_doc.metadata['title'] = os.path.basename(doc.metadata['source'])
                    if split_doc.metadata.get('page', 'Unknown') in ['0', '1', 'Unknown']:
                        split_doc.metadata['priority'] = 'high'
                    else:
                        split_doc.metadata['priority'] = 'low'
                    chunks.append(split_doc)
            
            if chunks:
                collection_path = os.path.join(persist_directory, domain)
                collection_name = domain if domain != "qa" else "qa_data"
                if not os.path.exists(collection_path):
                    os.makedirs(collection_path)
                    logger.info(f"Created collection directory: {collection_path}")
                
                logger.debug(f"Vectorizing {len(chunks)} chunks for domain: {domain}")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    collection_name=collection_name,
                    persist_directory=collection_path
                )
                logger.debug(f"Vectorized and saved {len(chunks)} chunks for domain: {domain}")
            else:
                logger.warning(f"No chunks to vectorize for domain: {domain}")
        
        os.chmod(persist_directory, 0o777)
        logger.info("Vectorization completed")
        
    except Exception as e:
        logger.error(f"Error in vectorize_documents: {str(e)}")
        raise
    
if __name__ == "__main__":
    vectorize_documents()