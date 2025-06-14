import json
import sys
import uuid
import logging
import numpy as np
import weaviate
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions as google_exceptions
import os

def generate_request_id():
    """Tạo UUID duy nhất làm request ID."""
    return str(uuid.uuid4())

def count_words(text: str) -> int:
    """Đếm số từ trong chuỗi văn bản."""
    return len(text.split())

def process_dieu(dieu, current_chunk, current_word_count, max_words, min_words, doc_id, ten_luat, so_chuong, ten_chuong, so_dieu, ten_dieu, current_dieu_list, current_khoan_list, current_diem_list, texts, chunks_for_json):
    noi_dung_chinh = dieu.get('noi_dung_chinh', '').strip()
    if noi_dung_chinh:
        dieu_content = noi_dung_chinh
        dieu_words = count_words(dieu_content)
        if current_word_count + dieu_words > max_words:
            if current_word_count >= min_words:
                texts, chunks_for_json = save_chunk(
                    current_chunk, current_word_count, min_words, doc_id, ten_luat,
                    so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                    current_diem_list, texts, chunks_for_json
                )
                current_chunk = ""
                current_word_count = 0
                current_dieu_list = [str(so_dieu)]
                current_khoan_list = []
                current_diem_list = []
            current_chunk += dieu_content + " "
            current_word_count += dieu_words
        else:
            current_chunk += dieu_content + " "
            current_word_count += dieu_words
        if str(so_dieu) not in current_dieu_list:
            current_dieu_list.append(str(so_dieu))
    for khoan in dieu.get('khoan', []):
        so_khoan = str(khoan.get('so_khoan', ''))
        noi_dung_khoan = khoan.get('noi_dung', '').strip()
        if noi_dung_khoan:
            khoan_text = noi_dung_khoan
            khoan_words = count_words(khoan_text)
            if current_word_count + khoan_words > max_words:
                if current_word_count >= min_words:
                    texts, chunks_for_json = save_chunk(
                        current_chunk, current_word_count, min_words, doc_id, ten_luat,
                        so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                        current_diem_list, texts, chunks_for_json
                    )
                    current_chunk = ""
                    current_word_count = 0
                    current_khoan_list = [so_khoan]
                    current_diem_list = []
                current_chunk += khoan_text + " "
                current_word_count += khoan_words
            else:
                current_chunk += khoan_text + " "
                current_word_count += khoan_words
            if so_khoan and so_khoan not in current_khoan_list:
                current_khoan_list.append(so_khoan)
        for diem in khoan.get('diem', []):
            ky_hieu = str(diem.get('ky_hieu', ''))
            noi_dung_diem = diem.get('noi_dung', '').strip()
            if noi_dung_diem:
                diem_text = noi_dung_diem
                diem_words = count_words(diem_text)
                if current_word_count + diem_words > max_words:
                    if current_word_count >= min_words:
                        texts, chunks_for_json = save_chunk(
                            current_chunk, current_word_count, min_words, doc_id, ten_luat,
                            so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                            current_diem_list, texts, chunks_for_json
                        )
                        current_chunk = ""
                        current_word_count = 0
                        current_diem_list = [ky_hieu]
                    current_chunk += diem_text + " "
                    current_word_count += diem_words
                else:
                    current_chunk += diem_text + " "
                    current_word_count += diem_words
                if ky_hieu and ky_hieu not in current_diem_list:
                    current_diem_list.append(ky_hieu)
    return current_chunk, current_word_count, current_dieu_list, current_khoan_list, current_diem_list, texts, chunks_for_json

def save_chunk(current_chunk, current_word_count, min_words, doc_id, ten_luat, so_chuong, ten_chuong, current_dieu_list, current_khoan_list, current_diem_list, texts, chunks_for_json):
    """Lưu chuỗi văn bản (chunk) vào danh sách."""
    if current_word_count >= min_words:
        chunk_text = current_chunk.strip()
        texts.append(chunk_text)
        chunk_id = str(uuid.uuid4())
        metadata = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "ten_luat": ten_luat,
            "so_chuong": so_chuong,
            "ten_chuong": ten_chuong,
            "dieu_list": [str(x) for x in current_dieu_list],
            "khoan_list": [str(x) for x in current_khoan_list],
            "diem_list": [str(x) for x in current_diem_list],
            "source_type": "legal_document",
            "word_count": current_word_count,
            "keywords": []
        }
        for field, values in [("dieu_list", metadata["dieu_list"]), ("khoan_list", metadata["khoan_list"]), ("diem_list", metadata["diem_list"])]:
            for value in values:
                if not isinstance(value, str):
                    logging.error(f"Lỗi kiểu dữ liệu trong {field}: Giá trị {value} (type {type(value)}) không phải string")
                    sys.exit(1)
        chunks_for_json.append({"text": chunk_text, "metadata": metadata})
    return texts, chunks_for_json

def get_legal_documents(client, collection, min_words=100, max_words=200):
    """Lấy tài liệu pháp lý từ MongoDB và chia thành các chunk."""
    try:
        texts = []
        chunks_for_json = []
        documents = list(collection.find())
        for doc in documents:
            doc_id = str(doc.get('_id', ''))
            ten_luat = doc.get('ten_luat', '')
            for chuong in doc.get('chuong', []):
                so_chuong = str(chuong.get('so_chuong', ''))
                ten_chuong = chuong.get('ten_chuong', '')
                current_chunk = ""
                current_word_count = 0
                current_dieu_list = []
                current_khoan_list = []
                current_diem_list = []
                for dieu in chuong.get('noi_dung', []):
                    if dieu.get('loai') != 'dieu':
                        continue
                    so_dieu = str(dieu.get('so_dieu', ''))
                    ten_dieu = dieu.get('ten_dieu', '')
                    noi_dung_chinh = dieu.get('noi_dung_chinh', '')
                    dieu_content = noi_dung_chinh
                    dieu_words = count_words(dieu_content)
                    if current_word_count + dieu_words > max_words:
                        if current_word_count >= min_words:
                            texts, chunks_for_json = save_chunk(
                                current_chunk, current_word_count, min_words, doc_id, ten_luat,
                                so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                                current_diem_list, texts, chunks_for_json
                            )
                            current_chunk = ""
                            current_word_count = 0
                            current_dieu_list = [so_dieu]
                            current_khoan_list = []
                            current_diem_list = []
                        if dieu_words > max_words:
                            current_chunk, current_word_count, current_dieu_list, current_khoan_list, current_diem_list, texts, chunks_for_json = process_dieu(
                                dieu, current_chunk, current_word_count, max_words, min_words, doc_id, ten_luat,
                                so_chuong, ten_chuong, so_dieu, ten_dieu, current_dieu_list, current_khoan_list,
                                current_diem_list, texts, chunks_for_json
                            )
                        else:
                            current_chunk += dieu_content + " "
                            current_word_count += dieu_words
                            current_dieu_list.append(so_dieu)
                    else:
                        current_chunk += dieu_content + " "
                        current_word_count += dieu_words
                        current_dieu_list.append(so_dieu)
                    for khoan in dieu.get('khoan', []):
                        so_khoan = str(khoan.get('so_khoan', ''))
                        noi_dung_khoan = khoan.get('noi_dung', '')
                        diem_list = khoan.get('diem', [])
                        if noi_dung_khoan:
                            khoan_text = noi_dung_khoan
                            khoan_words = count_words(khoan_text)
                            if current_word_count + khoan_words > max_words:
                                if current_word_count >= min_words:
                                    texts, chunks_for_json = save_chunk(
                                        current_chunk, current_word_count, min_words, doc_id, ten_luat,
                                        so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                                        current_diem_list, texts, chunks_for_json
                                    )
                                    current_chunk = ""
                                    current_word_count = 0
                                    current_khoan_list = [so_khoan]
                                    current_diem_list = []
                                current_chunk += khoan_text + " "
                                current_word_count += khoan_words
                                current_khoan_list.append(so_khoan)
                            else:
                                current_chunk += khoan_text + " "
                                current_word_count += khoan_words
                                current_khoan_list.append(so_khoan)
                        for diem in diem_list:
                            ky_hieu = str(diem.get('ky_hieu', ''))
                            noi_dung_diem = diem.get('noi_dung', '')
                            if noi_dung_diem:
                                diem_text = noi_dung_diem
                                diem_words = count_words(diem_text)
                                if current_word_count + diem_words > max_words:
                                    if current_word_count >= min_words:
                                        texts, chunks_for_json = save_chunk(
                                            current_chunk, current_word_count, min_words, doc_id, ten_luat,
                                            so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                                            current_diem_list, texts, chunks_for_json
                                        )
                                        current_chunk = ""
                                        current_word_count = 0
                                        current_diem_list = [ky_hieu]
                                    current_chunk += diem_text + " "
                                    current_word_count += diem_words
                                    current_diem_list.append(ky_hieu)
                                else:
                                    current_chunk += diem_text + " "
                                    current_word_count += diem_words
                                    current_diem_list.append(ky_hieu)
                    if current_word_count >= min_words:
                        texts, chunks_for_json = save_chunk(
                            current_chunk, current_word_count, min_words, doc_id, ten_luat,
                            so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                            current_diem_list, texts, chunks_for_json
                        )
                        current_chunk = ""
                        current_word_count = 0
                        current_dieu_list = []
                        current_khoan_list = []
                        current_diem_list = []
                if current_word_count >= min_words:
                    texts, chunks_for_json = save_chunk(
                        current_chunk, current_word_count, min_words, doc_id, ten_luat,
                        so_chuong, ten_chuong, current_dieu_list, current_khoan_list,
                        current_diem_list, texts, chunks_for_json
                    )
        json_dir = "chunks"
        os.makedirs(json_dir, exist_ok=True)
        json_file = os.path.join(json_dir, f"chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            for chunk in chunks_for_json:
                chunk["metadata"]["dieu_list"] = [str(x) for x in chunk["metadata"]["dieu_list"]]
                chunk["metadata"]["khoan_list"] = [str(x) for x in chunk["metadata"]["khoan_list"]]
                chunk["metadata"]["diem_list"] = [str(x) for x in chunk["metadata"]["diem_list"]]
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_for_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Lỗi khi lưu file JSON: {e}")
            sys.exit(1)
        return texts, chunks_for_json
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu: {e}")
        return [], []

def check_vector_compatibility():
    """Kiểm tra tính tương thích giữa vector trong collection và vector truy vấn."""
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get("LegalDocument")
        response = collection.query.fetch_objects(limit=1, include_vector=True)
        if not response.objects:
            logging.error("Không có đối tượng nào trong collection LegalDocument")
            return False
        obj = response.objects[0]
        stored_vector = obj.vector.get('default', []) if isinstance(obj.vector, dict) else obj.vector
        query = "test query"
        query_embedding = embed_single_text(query, generate_request_id())['embedding']
        if len(stored_vector) != len(query_embedding):
            logging.error(f"Vector không tương thích: stored_vector={len(stored_vector)}, query_embedding={len(query_embedding)}")
            return False
        stored_norm = np.linalg.norm(np.array(stored_vector))
        query_norm = np.linalg.norm(np.array(query_embedding))
        if not np.isclose(stored_norm, 1.0, rtol=1e-5) or not np.isclose(query_norm, 1.0, rtol=1e-5):
            logging.warning(f"Vector không chuẩn hóa: stored_norm={stored_norm}, query_norm={query_norm}")
            return False
        return True
    except Exception as e:
        logging.error(f"Lỗi kiểm tra vector: {e}")
        return False
    finally:
        client.close()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(google_exceptions.InternalServerError)
)
def embed_single_text(text: str, request_id: str) -> dict:
    """Nhúng văn bản thành vector sử dụng Gemini."""
    if not text.strip():
        logging.error(f"[Request {request_id}] Text rỗng, không thể nhúng")
        raise ValueError("Text rỗng")
    response = genai.embed_content(model="models/text-embedding-004", content=text)
    if not response.get('embedding'):
        logging.error(f"[Request {request_id}] Embedding rỗng cho text: {text[:50]}...")
        raise ValueError("Embedding rỗng")
    embedding = response['embedding']
    if len(embedding) != 768:
        logging.error(f"[Request {request_id}] Unexpected embedding length {len(embedding)} for text '{text[:50]}...'")
        raise ValueError(f"Embedding length {len(embedding)} != 768")
    embedding_np = np.array(embedding)
    norm = np.linalg.norm(embedding_np)
    if not np.isclose(norm, 1.0, rtol=1e-5):
        logging.warning(f"[Request {request_id}] Vector không chuẩn hóa, norm={norm}. Chuẩn hóa lại.")
        embedding_np = embedding_np / norm
        embedding = embedding_np.tolist()
    return {'embedding': embedding}

def batch_embed_texts(texts: list, batch_size: int = 10) -> list:
    """Nhúng hàng loạt văn bản thành vector."""
    embeddings = []
    texts_valid = []
    request_id = generate_request_id()
    try:
        sample_embedding = embed_single_text("test", request_id)['embedding']
        embedding_dim = len(sample_embedding)
    except Exception as e:
        logging.error(f"Lỗi khi lấy kích thước embedding: {e}")
        sys.exit(1)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = []
        for text in batch_texts:
            try:
                if not text.strip():
                    logging.warning(f"[Request {request_id}] Bỏ qua text rỗng: {text[:50]}...")
                    continue
                response = embed_single_text(text, request_id)
                embedding = response['embedding']
                if len(embedding) != embedding_dim:
                    logging.error(f"[Request {request_id}] Embedding không hợp lệ: {text[:50]}... Kích thước: {len(embedding)}")
                    continue
                batch_embeddings.append(embedding)
                texts_valid.append(text)
            except Exception as e:
                logging.error(f"[Request {request_id}] Lỗi khi nhúng: {text[:50]}...: {e}")
                continue
        embeddings.extend(batch_embeddings)
    return embeddings, texts_valid

def create_weaviate_store(texts: list, chunks_for_json: list):
    """Tạo và lưu trữ dữ liệu vào Weaviate."""
    client = None
    try:
        import weaviate.classes as wvc
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            skip_init_checks=True,
            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=30))
        )
        schema = {
            "class": "LegalDocument",
            "properties": [
                {"name": "chunk_id", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "text", "dataType": ["text"], "indexFilterable": True, "indexSearchable": True},
                {"name": "doc_id", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "ten_luat", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "so_chuong", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "ten_chuong", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "dieu_list", "dataType": ["string[]"], "indexFilterable": True, "indexSearchable": True},
                {"name": "khoan_list", "dataType": ["string[]"], "indexFilterable": True, "indexSearchable": True},
                {"name": "diem_list", "dataType": ["string[]"], "indexFilterable": True, "indexSearchable": True},
                {"name": "source_type", "dataType": ["string"], "indexFilterable": True, "indexSearchable": True},
                {"name": "word_count", "dataType": ["int"], "indexFilterable": True},
                {"name": "keywords", "dataType": ["string[]"], "indexFilterable": True, "indexSearchable": True}
            ],
            "vectorizer": "none",
            "vectorIndexConfig": {"distanceMetric": "cosine"},
            "moduleConfig": {
                "hybrid": {
                    "enabled": True
                }
            }
        }
        if client.collections.exists("LegalDocument"):
            client.collections.delete("LegalDocument")
        client.collections.create_from_dict(schema)
        embeddings, texts_valid = batch_embed_texts(texts)
        if len(embeddings) != len(texts_valid):
            logging.error("Số lượng embedding không khớp với số lượng text hợp lệ")
            sys.exit(1)
        chunks_for_json_valid = [
            chunk for chunk in chunks_for_json if chunk["text"] in texts_valid
        ]
        if len(embeddings) != len(chunks_for_json_valid):
            logging.error("Số lượng embedding không khớp với số lượng chunk hợp lệ")
            sys.exit(1)
        collection = client.collections.get("LegalDocument")
        failed_objects = []
        for i, chunk_data in enumerate(chunks_for_json_valid):
            metadata = chunk_data["metadata"]
            for field, values in [("dieu_list", metadata["dieu_list"]), ("khoan_list", metadata["khoan_list"]), ("diem_list", metadata["diem_list"])]:
                for value in values:
                    if not isinstance(value, str):
                        logging.error(f"Lỗi kiểu dữ liệu trong {field}: Giá trị {value} (type {type(value)}) không phải string")
                        sys.exit(1)
        with collection.batch.dynamic() as batch:
            for i, (text, chunk_data, embedding) in enumerate(zip(texts_valid, chunks_for_json_valid, embeddings)):
                try:
                    if not isinstance(embedding, list) or len(embedding) != 768:
                        logging.error(f"Embedding không hợp lệ tại index {i}: Length={len(embedding)}, Type={type(embedding)}")
                        failed_objects.append({"index": i, "text": text[:50], "error": "Invalid embedding"})
                        continue
                    metadata = chunk_data["metadata"]
                    obj = {
                        "chunk_id": str(metadata["chunk_id"]),
                        "text": text,
                        "doc_id": str(metadata["doc_id"]),
                        "ten_luat": str(metadata["ten_luat"]),
                        "so_chuong": str(metadata["so_chuong"]),
                        "ten_chuong": str(metadata["ten_chuong"]),
                        "dieu_list": [str(x) for x in metadata["dieu_list"]],
                        "khoan_list": [str(x) for x in metadata["khoan_list"]],
                        "diem_list": [str(x) for x in metadata["diem_list"]],
                        "source_type": str(metadata["source_type"]),
                        "word_count": int(metadata["word_count"]),
                        "keywords": [str(x) for x in metadata["keywords"]] if metadata["keywords"] else []
                    }
                    batch.add_object(properties=obj, vector=embedding, uuid=metadata["chunk_id"])
                except Exception as e:
                    logging.error(f"Lỗi khi lưu đối tượng {i + 1}: {e}")
                    failed_objects.append({"object": obj, "error": str(e)})
        if failed_objects:
            logging.error(f"Có {len(failed_objects)} đối tượng không lưu được")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Lỗi khi tạo Weaviate store: {e}")
        sys.exit(1)
    finally:
        if client:
            client.close()

def check_weaviate_data():
    """Kiểm tra dữ liệu trong Weaviate."""
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get("LegalDocument")
        response = collection.query.fetch_objects(limit=10, include_vector=True)
        return len(response.objects) > 0
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra dữ liệu Weaviate: {e}")
        return False
    finally:
        if client:
            client.close()

def view_weaviate_data():
    """Hiển thị dữ liệu mẫu từ Weaviate để kiểm tra."""
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get("LegalDocument")
        response = collection.query.fetch_objects(limit=5)
        print("\nDữ liệu mẫu trong Weaviate (top 5):")
        for obj in response.objects:
            print(f"\nChunk ID: {obj.properties['chunk_id']}")
            print(f"Text: {obj.properties['text'][:150]}...")
            print(f"Metadata: {obj.properties}")
    except Exception as e:
        logging.error(f"Lỗi khi xem dữ liệu Weaviate: {e}")
    finally:
        if client:
            client.close()