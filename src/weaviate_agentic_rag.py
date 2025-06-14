import os
import sys
import json
import yaml
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional
from autogen import AssistantAgent, UserProxyAgent, register_function
import weaviate
import logging
import uuid
import threading
from embed.weaviate.searcher import search_weaviate

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

# Kết nối Weaviate
def connect_weaviate():
    try:
        client = weaviate.connect_to_local()
        logging.info("Kết nối thành công với Weaviate")
        return client
    except Exception as e:
        logging.error(f"Lỗi kết nối Weaviate: {e}")
        sys.exit(1)

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
def process_query(query, client, agents_config):
    request_id = generate_request_id()
    logging.info(f"[Request {request_id}] Query người dùng: {query}")
    
    with process_lock:
        # Tạo agents từ cấu hình YAML
        query_interpreter = AssistantAgent(**agents_config['QueryInterpreter'])
        info_retriever = AssistantAgent(**agents_config['InfoRetriever'])
        user_proxy = UserProxyAgent(**agents_config['UserProxy'])
        def search_weaviate_tool(query: str, k: int = 3, request_id: Optional[str] = None, alpha: float = 0.5):
            return search_weaviate(client, query, k, request_id, alpha)
        # Đăng ký tool search_weaviate cho InfoRetriever
        register_function(
            f=search_weaviate_tool,
            caller=info_retriever,
            executor=user_proxy,
            name="search_weaviate",
            description="Tìm kiếm thông tin luật trong Weaviate bằng hybrid search."
        )

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
        
        # Giai đoạn 2: Truy xuất thông tin và định dạng phản hồi
        response_parts = [f"Phản hồi cho câu hỏi: {query}"]
        for sub_query in sub_queries:
            retriever_response = user_proxy.initiate_chat(
                recipient=info_retriever,
                message=f"Thực hiện tìm kiếm với sub-query: {sub_query}",
                clear_history=True,
                silent=True,
                max_turns=1
            )
            
            retriever_text = retriever_response.chat_history[-1]["content"]
            try:
                cleaned_retriever_text = clean_json_string(retriever_text)
                sub_results = json.loads(cleaned_retriever_text)
                logging.info(f"[Request {request_id}] Top {len(sub_results)} kết quả cho sub-query '{sub_query}':")
                response_parts.append(f"\nKết quả từ sub-query: '{sub_query}'")
                if sub_results:
                    for item in sub_results:
                        logging.info(f"Kết quả: {item['text']}")
                        response_parts.append(item["text"])
                else:
                    response_parts.append("Không tìm thấy kết quả phù hợp.")
                response_parts.append("-" * 50)
            except json.JSONDecodeError as e:
                logging.error(f"[Request {request_id}] Lỗi khi phân tích JSON từ InfoRetriever: {e}, Chuỗi: {retriever_text}")
                response_parts.append(f"\nKết quả từ sub-query: '{sub_query}'")
                response_parts.append("Không tìm thấy kết quả phù hợp.")
                response_parts.append("-" * 50)
        
        return "\n".join(response_parts)

# Hàm chính
def main():
    agents_config = load_agents_config(r"src\agents.yaml")
    client = connect_weaviate()
    print("Đang kết nối Weaviate...")
    print("Hệ thống Agentic RAG Luật đã sẵn sàng. Gõ 'exit' để thoát.")
    try:
        while True:
            query = input("Nhập câu hỏi: ")
            if query.lower() == "exit":
                logging.info("Người dùng thoát chương trình.")
                break
            response = process_query(query, client, agents_config)
            print("\n\n\n\n\n")
            print(f"Phản hồi:\n{response}\n{'-'*50}")
    finally:
        client.close()

if __name__ == "__main__":
    main()