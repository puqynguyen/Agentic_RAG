import os
import sys
import json
import yaml
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
from autogen import AssistantAgent, UserProxyAgent
import deeplake
import logging
import uuid
import threading
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions as google_exceptions

# Khóa để ngăn xử lý đồng thời
process_lock = threading.Lock()

# Hàm tạo request ID
def generate_request_id():
    return str(uuid.uuid4())

# Thiết lập logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Tải biến môi trường
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    logging.error("Thiếu GOOGLE_API_KEY trong file .env")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Tải cấu hình agent từ YAML
def load_agents_config(yaml_path="agents.yaml"):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for agent_config in config.values():
            if 'llm_config' in agent_config and 'config_list' in agent_config['llm_config']:
                for llm in agent_config['llm_config']['config_list']:
                    if llm.get('api_key') == 'env:GOOGLE_API_KEY':
                        llm['api_key'] = GOOGLE_API_KEY
        return config
    except Exception as e:
        logging.error(f"Lỗi khi tải cấu hình YAML: {e}")
        sys.exit(1)

# Tải vector store
def load_vector_store(dataset_path="./deeplake/legal_vectors"):
    try:
        if not os.path.exists(dataset_path):
            logging.error(f"Vector store không tồn tại tại {dataset_path}")
            sys.exit(1)
        ds = deeplake.load(dataset_path)
        return ds
    except Exception as e:
        logging.error(f"Lỗi khi tải vector store: {e}")
        sys.exit(1)

# Hàm nhúng với retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(google_exceptions.InternalServerError)
)
def embed_single_text(text, request_id):
    return genai.embed_content(model="models/text-embedding-004", content=text)

# Tìm kiếm trong DeepLake
def search_vector_store(dataset_path, query, k=3, request_id=None):
    try:
        ds = load_vector_store(dataset_path)
        required_tensors = ['text', 'embedding', 'metadata']
        missing_tensors = [t for t in required_tensors if t not in ds.tensors]
        if missing_tensors:
            logging.error(f"[Request {request_id}] Vector store thiếu tensor: {missing_tensors}")
            return []
        
        response = embed_single_text(query, request_id)
        query_embedding = np.array(response['embedding'], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        all_embeddings = np.array([
            sample.embedding.data()['value'] / np.linalg.norm(sample.embedding.data()['value'])
            for sample in ds
        ])
        all_texts = [sample.text.data()['value'] for sample in ds]
        all_metadata = [sample.metadata.data()['value'] for sample in ds]
        
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        results_list = []
        for idx in top_k_indices:
            if similarities[idx] < 0.65:
                continue
            meta = all_metadata[idx]
            dieu_str = ", ".join(map(str, meta.get('dieu_list', []))) if meta.get('dieu_list') else "Không xác định"
            source = f"Luật {meta['ten_luat']} {meta['doc_id']}, Chương {meta['so_chuong']} - {meta['ten_chuong']}, Điều {dieu_str}"
            if 'khoan_list' in meta and meta['khoan_list']:
                source += f", Khoản {', '.join(map(str, meta['khoan_list']))}"
            if 'diem_list' in meta and meta['diem_list']:
                source += f", Điểm {', '.join(map(str, meta['diem_list']))}"
            relevance_score = float(similarities[idx])
            results_list.append({
                'text': f"Độ liên quan: {relevance_score:.4f}\nNguồn: {source}\nNội dung: {all_texts[idx]}",
                'score': relevance_score,
                'content': all_texts[idx],
                'metadata': meta
            })
        
        return results_list
    except Exception as e:
        logging.error(f"[Request {request_id}] Lỗi khi tìm kiếm: {e}")
        return []

def clean_json_string(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    return text

# Xử lý truy vấn
def process_query(query, dataset_path, agents_config):
    request_id = generate_request_id()
    logging.info(f"[Request {request_id}] Query người dùng: {query}")
    
    with process_lock:
        # Tạo agents từ cấu hình YAML
        query_interpreter = AssistantAgent(**agents_config['QueryInterpreter'])
        user_proxy = UserProxyAgent(**agents_config['UserProxy'])

        # Giai đoạn 1: Phân tích câu hỏi
        query_interpreter_response = user_proxy.initiate_chat(
            recipient=query_interpreter,
            message=f"Phân tích câu hỏi: {query}",
            clear_history=True,
            silent=True,
            max_turns=1
        )
        
        response_text = query_interpreter_response.chat_history[-1]["content"]
        try:
            cleaned_response = clean_json_string(response_text)
            response_json = json.loads(cleaned_response)
            if "sub_queries" in response_json:
                sub_queries = response_json["sub_queries"]
                logging.info(f"[Request {request_id}] Expanded queries: {sub_queries}")
            else:
                logging.info(f"[Request {request_id}] Không cần truy xuất: {response_text}")
                return response_text
        except json.JSONDecodeError as e:
            logging.error(f"[Request {request_id}] Lỗi khi phân tích JSON từ QueryInterpreter: {e}, Chuỗi: {response_text}")
            return "Lỗi hệ thống: Phản hồi không hợp lệ từ QueryInterpreter."
        
        # Giai đoạn 2: Truy xuất thông tin
        results = []
        for sub_query in sub_queries:
            sub_results = search_vector_store(dataset_path, sub_query, k=3, request_id=request_id)
            logging.info(f"[Request {request_id}] Top 3 kết quả cho sub-query '{sub_query}':")
            for i, item in enumerate(sub_results, 1):
                logging.info(f"Kết quả {i}: {item['text']}")
            results.append({"sub_query": sub_query, "items": sub_results})
            input(f"\n\n\n\n\nNhấn Enter để tiếp tục với sub-query: {sub_query} (hoặc gõ 'exit' để thoát): ")
        
        # Giai đoạn 3: Định dạng phản hồi
        response_parts = [f"Phản hồi cho câu hỏi: {query}"]
        for result in results:
            sub_query = result["sub_query"]
            response_parts.append(f"\nKết quả từ sub-query: '{sub_query}'")
            if result["items"]:
                for item in result["items"]:
                    response_parts.append(item["text"])
            else:
                response_parts.append("Không tìm thấy kết quả phù hợp.")
            response_parts.append("-" * 50)
        
        return "\n".join(response_parts)

# Hàm chính
# def main():
dataset_path = "./deeplake/legal_vectors"
agents_config = load_agents_config(r"src\agents.yaml")
print("Đang tải vector store...")
print("Hệ thống Agentic RAG Luật đã sẵn sàng. Gõ 'exit' để thoát.")
while True:
    query = input("Nhập câu hỏi: ")
    if query.lower() == "exit":
        logging.info("Người dùng thoát chương trình.")
        break
    response = process_query(query, dataset_path, agents_config)
    print("\n\n\n\n\n")
    print(f"Phản hồi:\n{response}\n{'-'*50}")
            
# if "__name__" == "__main__":
#     main()