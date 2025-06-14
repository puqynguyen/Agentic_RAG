import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import weaviate
import subprocess
import time
from core_functions import (
    generate_request_id,
    get_legal_documents,
    create_weaviate_store,
    check_weaviate_data,
    view_weaviate_data,
    search_weaviate
)

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"weaviate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def load_environment():
    load_dotenv()
    env_vars = {
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'MONGO_URI': os.getenv('MONGO_URI'),
        'MONGO_DB': os.getenv('MONGO_DB'),
        'MONGO_COLLECTION': os.getenv('MONGO_COLLECTION')
    }
    if not all(env_vars.values()):
        logging.error("Thiếu thông tin cấu hình trong file .env")
        sys.exit(1)
    return env_vars

def connect_mongodb(mongo_uri, mongo_db, mongo_collection):
    try:
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        collection = db[mongo_collection]
        logging.info("Kết nối thành công với MongoDB")
        return client, collection
    except Exception as e:
        logging.error(f"Lỗi kết nối MongoDB: {e}")
        sys.exit(1)

def start_weaviate():
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=weaviate"],
            capture_output=True, text=True
        )
        if result.stdout:
            logging.info("Weaviate container đã chạy")
            return
        logging.info("Khởi động Weaviate container...")
        subprocess.run(
            [
                "docker", "run", "-d",
                "-p", "8080:8080",
                "-p", "50051:50051",
                "--name", "weaviate",
                "cr.weaviate.io/semitechnologies/weaviate:1.26.1"
            ],
            check=True
        )
        time.sleep(10)  # Tăng thời gian chờ để đảm bảo khởi động
        logging.info("Weaviate container đã khởi động")
    except subprocess.CalledProcessError as e:
        logging.error(f"Lỗi khi khởi động Weaviate: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Docker không được cài đặt hoặc không tìm thấy. Vui lòng cài đặt Docker.")
        sys.exit(1)

def check_weaviate_version():
    try:
        client = weaviate.connect_to_local(host="localhost", port=8080, skip_init_checks=True)
        # Kiểm tra phiên bản server
        meta = client.get_meta()
        server_version = meta.get("version", "Unknown")
        logging.info(f"Weaviate server version: {server_version}")
        # Kiểm tra phiên bản client
        client_version = weaviate.__version__
        logging.info(f"Weaviate Python client version: {client_version}")
        return server_version, client_version
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra phiên bản Weaviate: {e}")
        return None, None
    finally:
        client.close()
        
def main():
    log_file = setup_logging()
    env_vars = load_environment()
    server_version, client_version = check_weaviate_version()
    if server_version != "1.26.1":
        logging.warning(f"Phiên bản Weaviate server ({server_version}) không phải 1.26.1. Đề xuất dùng 1.26.1.")
    if client_version != "4.9.0":  # Đề xuất client tương thích
        logging.warning(f"Phiên bản Weaviate client ({client_version}) không phải 4.9.0. Đề xuất dùng 4.9.0.")
    # Khởi động Weaviate
    start_weaviate()

    import google.generativeai as genai
    genai.configure(api_key=env_vars['GOOGLE_API_KEY'])

    mongo_client, collection = connect_mongodb(
        env_vars['MONGO_URI'], env_vars['MONGO_DB'], env_vars['MONGO_COLLECTION']
    )

    try:
        texts, chunks_for_json = get_legal_documents(mongo_client, collection)
        if not texts:
            logging.error("Không có dữ liệu để xử lý.")
            sys.exit(1)

        create_weaviate_store(texts, chunks_for_json)

        if not check_weaviate_data():
            logging.error("Không có dữ liệu trong Weaviate")
            sys.exit(1)

        view_weaviate_data()

        weaviate_client = weaviate.connect_to_local(host="localhost", port=8080, skip_init_checks=True)
        query = "Báo cáo nghiên cứu tiền khả thi đầu tư xây dựng là gì?"
        results = search_weaviate(weaviate_client, query, request_id=generate_request_id())
        if results:
            print("\nKết quả tìm kiếm:")
            for result in results:
                print(f"\nScore: {result['score']:.4f}")
                print(f"Text: {result['text'][:150]}...")
                print(f"Metadata: {result['metadata']}")
        else:
            print("\nKhông tìm thấy kết quả.")
        weaviate_client.close()

        print(f"\nĐã hoàn tất. Dữ liệu được lưu trong Weaviate.")
        print(f"Log chi tiết tại: {log_file}")
        print(f"Chunks được lưu tại: chunks/chunks_*.json")
    finally:
        mongo_client.close()
        logging.info("Đã đóng kết nối MongoDB")

if __name__ == "__main__":
    main()