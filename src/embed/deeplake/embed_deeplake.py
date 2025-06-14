import os
import sys
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
import pandas as pd
import google.generativeai as genai
import deeplake
import logging
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Thiết lập logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB = os.getenv('MONGO_DB')
MONGO_COLLECTION = os.getenv('MONGO_COLLECTION')

if not all([GOOGLE_API_KEY, MONGO_URI, MONGO_DB, MONGO_COLLECTION]):
    logging.error("Thiếu thông tin cấu hình trong file .env")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Kết nối MongoDB
def connect_mongodb():
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        logging.info("Kết nối thành công với MongoDB")
        return collection
    except Exception as e:
        logging.error(f"Lỗi kết nối MongoDB: {e}")
        sys.exit(1)

# Hàm đếm số từ
def count_words(text):
    return len(text.split())

def get_legal_documents(collection):
    try:
        documents = list(collection.find())
        texts = []
        metadata_list = []
        chunks_for_json = []
        MIN_WORDS = 200
        MAX_WORDS = 400

        for doc in documents:
            doc_id = str(doc.get('_id', ''))
            ten_luat = doc.get('ten_luat', '')
            for chuong in doc.get('chuong', []):
                so_chuong = chuong.get('so_chuong', '')
                ten_chuong = chuong.get('ten_chuong', '')
                chuong_header = f"CHƯƠNG {so_chuong}\n{ten_chuong}"
                chuong_words = count_words(chuong_header)
                current_chunk = chuong_header
                current_word_count = chuong_words
                current_dieu_list = []
                current_khoan_list = []
                current_diem_list = []
                current_dieu = None
                current_khoan = None
                current_ten_dieu = None

                for dieu in chuong.get('noi_dung', []):
                    if dieu.get('loai') != 'dieu':
                        continue
                    so_dieu = dieu.get('so_dieu', '')
                    ten_dieu = dieu.get('ten_dieu', '')
                    noi_dung_chinh = dieu.get('noi_dung_chinh', '')
                    dieu_header = f"\nĐiều {so_dieu}. {ten_dieu}\n{noi_dung_chinh}"
                    dieu_words = count_words(dieu_header)

                    # Nếu thêm điều này vượt MAX_WORDS
                    if current_word_count + dieu_words > MAX_WORDS:
                        if current_word_count >= MIN_WORDS:
                            chunk_id = str(uuid.uuid4())
                            texts.append(current_chunk)
                            metadata = {
                                'chunk_id': chunk_id,
                                'doc_id': doc_id,
                                'ten_luat': ten_luat,
                                'so_chuong': so_chuong,
                                'ten_chuong': ten_chuong,
                                'dieu_list': current_dieu_list,
                                'khoan_list': current_khoan_list,
                                'diem_list': current_diem_list,
                                'source_type': 'chuong' if len(current_dieu_list) > 1 else 'dieu',
                                'word_count': current_word_count
                            }
                            metadata_list.append(metadata)
                            chunks_for_json.append({
                                'chunk_id': chunk_id,
                                'text': current_chunk,
                                'metadata': metadata
                            })
                            current_chunk = chuong_header
                            current_word_count = chuong_words
                            current_dieu_list = [so_dieu] if current_dieu else []
                            current_khoan_list = []
                            current_diem_list = []
                        # Chia nhỏ điều nếu quá dài
                        if dieu_words > MAX_WORDS:
                            words = noi_dung_chinh.split()
                            current_dieu_chunk = []
                            current_dieu_word_count = count_words(f"\nĐiều {so_dieu}. {ten_dieu}\n")
                            for word in words:
                                if current_dieu_word_count + 1 > MAX_WORDS:
                                    chunk_text = f"{chuong_header}\nĐiều {so_dieu}. {ten_dieu}\n{' '.join(current_dieu_chunk)}"
                                    chunk_id = str(uuid.uuid4())
                                    texts.append(chunk_text)
                                    metadata = {
                                        'chunk_id': chunk_id,
                                        'doc_id': doc_id,
                                        'ten_luat': ten_luat,
                                        'so_chuong': so_chuong,
                                        'ten_chuong': ten_chuong,
                                        'dieu_list': [so_dieu],
                                        'khoan_list': [],
                                        'diem_list': [],
                                        'source_type': 'dieu',
                                        'word_count': current_dieu_word_count
                                    }
                                    metadata_list.append(metadata)
                                    chunks_for_json.append({
                                        'chunk_id': chunk_id,
                                        'text': chunk_text,
                                        'metadata': metadata
                                    })
                                    current_dieu_chunk = []
                                    current_dieu_word_count = count_words(f"\nĐiều {so_dieu}. {ten_dieu}\n")
                                current_dieu_chunk.append(word)
                                current_dieu_word_count += 1
                            if current_dieu_chunk:
                                chunk_text = f"{chuong_header}\nĐiều {so_dieu}. {ten_dieu}\n{' '.join(current_dieu_chunk)}"
                                chunk_id = str(uuid.uuid4())
                                texts.append(chunk_text)
                                metadata = {
                                    'chunk_id': chunk_id,
                                    'doc_id': doc_id,
                                    'ten_luat': ten_luat,
                                    'so_chuong': so_chuong,
                                    'ten_chuong': ten_chuong,
                                    'dieu_list': [so_dieu],
                                    'khoan_list': [],
                                    'diem_list': [],
                                    'source_type': 'dieu',
                                    'word_count': current_dieu_word_count
                                }
                                metadata_list.append(metadata)
                                chunks_for_json.append({
                                    'chunk_id': chunk_id,
                                    'text': chunk_text,
                                    'metadata': metadata
                                })
                        else:
                            current_chunk += dieu_header
                            current_word_count += dieu_words
                            current_dieu_list.append(so_dieu)
                    else:
                        current_chunk += dieu_header
                        current_word_count += dieu_words
                        current_dieu_list.append(so_dieu)
                    current_dieu = so_dieu
                    current_ten_dieu = ten_dieu

                    # Thêm khoản và điểm
                    for khoan in dieu.get('khoan', []):
                        so_khoan = khoan.get('so_khoan', '')
                        noi_dung_khoan = khoan.get('noi_dung', '')
                        diem_list = khoan.get('diem', [])
                        has_diem = len(diem_list) > 0
                        if noi_dung_khoan:
                            khoan_text = f"\n{so_khoan}. {noi_dung_khoan}"
                            khoan_words = count_words(khoan_text)
                            if current_word_count + khoan_words > MAX_WORDS:
                                if current_word_count >= MIN_WORDS:
                                    chunk_id = str(uuid.uuid4())
                                    texts.append(current_chunk)
                                    metadata = {
                                        'chunk_id': chunk_id,
                                        'doc_id': doc_id,
                                        'ten_luat': ten_luat,
                                        'so_chuong': so_chuong,
                                        'ten_chuong': ten_chuong,
                                        'dieu_list': current_dieu_list,
                                        'khoan_list': current_khoan_list,
                                        'diem_list': current_diem_list,
                                        'source_type': 'chuong' if len(current_dieu_list) > 1 else 'dieu',
                                        'word_count': current_word_count
                                    }
                                    metadata_list.append(metadata)
                                    chunks_for_json.append({
                                        'chunk_id': chunk_id,
                                        'text': current_chunk,
                                        'metadata': metadata
                                    })
                                    current_chunk = f"{chuong_header}\nĐiều {so_dieu}. {ten_dieu}"
                                    current_word_count = count_words(current_chunk)
                                    current_khoan_list = []
                                    current_diem_list = []
                                current_chunk += khoan_text
                                current_word_count += khoan_words
                                current_khoan_list.append(so_khoan)
                            else:
                                current_chunk += khoan_text
                                current_word_count += khoan_words
                                current_khoan_list.append(so_khoan)
                            current_khoan = so_khoan

                        for diem in diem_list:
                            ky_hieu = diem.get('ky_hieu', '')
                            noi_dung_diem = diem.get('noi_dung', '')
                            if noi_dung_diem:
                                diem_text = f"\n{ky_hieu}. {noi_dung_diem}"
                                diem_words = count_words(diem_text)
                                if current_word_count + diem_words > MAX_WORDS:
                                    if current_word_count >= MIN_WORDS:
                                        chunk_id = str(uuid.uuid4())
                                        texts.append(current_chunk)
                                        metadata = {
                                            'chunk_id': chunk_id,
                                            'doc_id': doc_id,
                                            'ten_luat': ten_luat,
                                            'so_chuong': so_chuong,
                                            'ten_chuong': ten_chuong,
                                            'dieu_list': current_dieu_list,
                                            'khoan_list': current_khoan_list,
                                            'diem_list': current_diem_list,
                                            'source_type': 'chuong' if len(current_dieu_list) > 1 else 'dieu',
                                            'word_count': current_word_count
                                        }
                                        metadata_list.append(metadata)
                                        chunks_for_json.append({
                                            'chunk_id': chunk_id,
                                            'text': current_chunk,
                                            'metadata': metadata
                                        })
                                        current_chunk = f"{chuong_header}\nĐiều {so_dieu}. {ten_dieu}\n{so_khoan}."
                                        current_word_count = count_words(current_chunk)
                                        current_diem_list = []
                                    current_chunk += diem_text
                                    current_word_count += diem_words
                                    current_diem_list.append(ky_hieu)
                                else:
                                    current_chunk += diem_text
                                    current_word_count += diem_words
                                    current_diem_list.append(ky_hieu)

                        # Kiểm tra nếu chunk ≥ MIN_WORDS và kết thúc khoản, chỉ áp dụng cho khoản có điểm
                        if has_diem and current_word_count >= MIN_WORDS:
                            chunk_id = str(uuid.uuid4())
                            texts.append(current_chunk)
                            metadata = {
                                'chunk_id': chunk_id,
                                'doc_id': doc_id,
                                'ten_luat': ten_luat,
                                'so_chuong': so_chuong,
                                'ten_chuong': ten_chuong,
                                'dieu_list': current_dieu_list,
                                'khoan_list': current_khoan_list,
                                'diem_list': current_diem_list,
                                'source_type': 'chuong' if len(current_dieu_list) > 1 else 'dieu',
                                'word_count': current_word_count
                            }
                            metadata_list.append(metadata)
                            chunks_for_json.append({
                                'chunk_id': chunk_id,
                                'text': current_chunk,
                                'metadata': metadata
                            })
                            current_chunk = chuong_header
                            current_word_count = chuong_words
                            current_dieu_list = [so_dieu] if current_dieu else []
                            current_khoan_list = []
                            current_diem_list = []

                    # Kiểm tra nếu chunk ≥ MIN_WORDS và kết thúc điều
                    if current_word_count >= MIN_WORDS:
                        chunk_id = str(uuid.uuid4())
                        texts.append(current_chunk)
                        metadata = {
                            'chunk_id': chunk_id,
                            'doc_id': doc_id,
                            'ten_luat': ten_luat,
                            'so_chuong': so_chuong,
                            'ten_chuong': ten_chuong,
                            'dieu_list': current_dieu_list,
                            'khoan_list': current_khoan_list,
                            'diem_list': current_diem_list,
                            'source_type': 'chuong' if len(current_dieu_list) > 1 else 'dieu',
                            'word_count': current_word_count
                        }
                        metadata_list.append(metadata)
                        chunks_for_json.append({
                            'chunk_id': chunk_id,
                            'text': current_chunk,
                            'metadata': metadata
                        })
                        current_chunk = chuong_header
                        current_word_count = chuong_words
                        current_dieu_list = []
                        current_khoan_list = []
                        current_diem_list = []

                # Lưu chunk cuối nếu đủ MIN_WORDS
                if current_word_count >= MIN_WORDS:
                    chunk_id = str(uuid.uuid4())
                    texts.append(current_chunk)
                    metadata = {
                        'chunk_id': chunk_id,
                        'doc_id': doc_id,
                        'ten_luat': ten_luat,
                        'so_chuong': so_chuong,
                        'ten_chuong': ten_chuong,
                        'dieu_list': current_dieu_list,
                        'khoan_list': current_khoan_list,
                        'diem_list': current_diem_list,
                        'source_type': 'chuong' if len(current_dieu_list) > 1 else 'dieu',
                        'word_count': current_word_count
                    }
                    metadata_list.append(metadata)
                    chunks_for_json.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk,
                        'metadata': metadata
                    })

        # Lưu chunks vào file JSON
        json_dir = "chunks"
        os.makedirs(json_dir, exist_ok=True)
        json_file = os.path.join(json_dir, f"chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_for_json, f, ensure_ascii=False, indent=2)
            logging.info(f"Đã lưu {len(chunks_for_json)} chunk vào file JSON: {json_file}")
        except Exception as e:
            logging.error(f"Lỗi khi lưu file JSON: {e}")
            sys.exit(1)

        # Validate chunks
        for i, chunk in enumerate(texts):
            word_count = count_words(chunk)
            if not (MIN_WORDS <= word_count <= MAX_WORDS):
                logging.warning(f"Chunk {i+1} có số từ không hợp lệ: {word_count}")

        logging.info(f"Đã xử lý {len(texts)} chunk từ {len(documents)} tài liệu")
        logging.info(f"Mẫu chunk đầu tiên: {texts[:3]}")
        return texts, metadata_list
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu: {e}")
        return [], []

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions as google_exceptions

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(google_exceptions.InternalServerError)
)
def embed_single_text(text):
    return genai.embed_content(model="models/text-embedding-004", content=text)

def batch_embed_texts(texts, batch_size=10):
    embeddings = []
    try:
        sample_embedding = embed_single_text("test")['embedding']
        embedding_dim = len(sample_embedding)
        logging.info(f"Kích thước embedding: {embedding_dim}")
    except Exception as e:
        logging.error(f"Lỗi khi lấy kích thước embedding: {e}")
        embedding_dim = 768
        logging.warning(f"Sử dụng kích thước embedding mặc định: {embedding_dim}")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_embeddings = []
            for text in batch_texts:
                try:
                    response = embed_single_text(text)
                    batch_embeddings.append(np.array(response['embedding'], dtype=np.float32))
                except Exception as e:
                    logging.error(f"Lỗi khi nhúng văn bản: {text[:50]}...: {e}")
                    batch_embeddings.append(np.zeros(embedding_dim))
            embeddings.extend(batch_embeddings)
            logging.info(f"Đã nhúng batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")
        except Exception as e:
            logging.error(f"Lỗi khi nhúng batch {i // batch_size + 1}: {e}")
            embeddings.extend([np.zeros(embedding_dim)] * len(batch_texts))
    return embeddings

# Kiểm tra độ tương đồng cosine
def check_embedding_similarity(embeddings, texts, sample_size=5):
    sample_indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
    sample_embeddings = np.array([embeddings[i] for i in sample_indices])
    sample_texts = [texts[i] for i in sample_indices]
    
    similarity_matrix = cosine_similarity(sample_embeddings)
    
    print("\nĐộ tương đồng cosine giữa các văn bản mẫu:")
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            print(f"- Văn bản {i+1} vs {j+1}: {similarity_matrix[i][j]:.4f}")
            print(f"  Văn bản {i+1}: {sample_texts[i][:100]}...")
            print(f"  Văn bản {j+1}: {sample_texts[j][:100]}...")

# Trực quan hóa embeddings
def visualize_embeddings(embeddings, texts, metadata_list):
    try:
        perplexity = min(30, max(5, len(embeddings)-1))  # Điều chỉnh perplexity động
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(np.array(embeddings))
        
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'text': [t[:100] + "..." for t in texts],
            'so_chuong': [m.get('so_chuong', 'Unknown') for m in metadata_list],
            'ten_chuong': [m.get('ten_chuong', 'Unknown') for m in metadata_list],
            'dieu_list': [m.get('dieu_list', []) for m in metadata_list]
        })
        
        df.to_csv('embeddings_2d.csv', index=False)
        logging.info("Đã lưu thông tin điểm vào embeddings_2d.csv")
        
        plt.figure(figsize=(12, 8))
        for chuong in df['so_chuong'].unique():
            subset = df[df['so_chuong'] == chuong]
            plt.scatter(subset['x'], subset['y'], label=f"Chương {chuong} - {subset['ten_chuong'].iloc[0]}", alpha=0.6)
        
        plt.title("Trực quan hóa Embeddings theo Chương (t-SNE)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("embeddings_visualization.png", bbox_inches='tight')
        plt.close()
        
        logging.info("Đã lưu hình ảnh trực quan hóa tại embeddings_visualization.png")
    except Exception as e:
        logging.error(f"Lỗi khi trực quan hóa embeddings: {e}")

# Tạo vector store
def create_vector_store(texts, metadata_list):
    try:
        dataset_path = "./deeplake/legal_vectors"
        ds = deeplake.empty(dataset_path, overwrite=True)
        
        with ds:
            ds.create_tensor('text', htype='text')
            ds.create_tensor('embedding', htype='embedding', dtype=np.float32)
            ds.create_tensor('metadata', htype='json')
            
            embeddings = batch_embed_texts(texts)
            
            for i, (emb, text) in enumerate(zip(embeddings, texts)):
                if np.isnan(emb).any() or np.all(emb == 0):
                    logging.warning(f"Embedding không hợp lệ cho văn bản: {text[:50]}...")
            
            for text, emb, meta in zip(texts, embeddings, metadata_list):
                ds.append({
                    'text': text,
                    'embedding': emb,
                    'metadata': meta
                })
        
        logging.info(f"Đã tạo vector store tại {dataset_path}")
        check_embedding_similarity(embeddings, texts)
        visualize_embeddings(embeddings, texts, metadata_list)
        return dataset_path
    except Exception as e:
        logging.error(f"Lỗi khi tạo vector store: {e}")
        sys.exit(1)

# Xem embeddings
def view_embeddings(dataset_path):
    try:
        ds = deeplake.load(dataset_path)
        print(f"Dataset tại {dataset_path}:")
        print(f"Tổng số mẫu: {len(ds)}")
        for i in range(min(5, len(ds))):
            print(f"\nMẫu {i+1}:")
            print(f"Text: {ds.text[i].data()['value'][:150]}...")
            print(f"Embedding: {ds.embedding[i].data()['value'][:5]}... (kích thước: {ds.embedding[i].data()['value'].shape})")
            print(f"Metadata: {ds.metadata[i].data()['value']}")
    except Exception as e:
        logging.error(f"Lỗi khi xem embeddings: {e}")

# Hàm chính
def main():
    collection = connect_mongodb()
    texts, metadata_list = get_legal_documents(collection)
    if not texts:
        logging.error("Không có dữ liệu để xử lý.")
        sys.exit(1)
    
    dataset_path = create_vector_store(texts, metadata_list)
    view_embeddings(dataset_path)
    print(f"\nĐã hoàn tất. Vector store được lưu tại: {dataset_path}")
    print("Xem hình ảnh trực quan hóa tại: embeddings_visualization.png")
    print("Xem thông tin điểm tại: embeddings_2d.csv")
    print(f"Log chi tiết tại: {log_file}")
    print(f"Chunks được lưu tại: chunks/chunks_*.json")

if __name__ == "__main__":
    main()