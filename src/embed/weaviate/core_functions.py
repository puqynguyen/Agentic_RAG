import json
import sys
import uuid
import logging
import weaviate
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions as google_exceptions
from numpy import dot
from numpy.linalg import norm
import os

def generate_request_id():
    """Tạo một UUID duy nhất làm request ID.
    
    Returns:
        str: UUID dưới dạng chuỗi.
    """
    return str(uuid.uuid4())

def count_words(text: str) -> int:
    """Đếm số từ trong một chuỗi văn bản.
    
    Args:
        text (str): Văn bản cần đếm từ.
    
    Returns:
        int: Số từ trong văn bản.
    """
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
    """Lưu chunk vào danh sách."""
    if current_word_count >= min_words:
        chunk_text = current_chunk.strip()
        texts.append(chunk_text)
        metadata = {
            "doc_id": doc_id,
            "ten_luat": ten_luat,
            "so_chuong": so_chuong,
            "ten_chuong": ten_chuong,
            "dieu_list": [str(x) for x in current_dieu_list],
            "khoan_list": [str(x) for x in current_khoan_list],
            "diem_list": [str(x) for x in current_diem_list]
        }
        chunks_for_json.append({"text": chunk_text, "metadata": metadata})
    return texts, chunks_for_json

def get_legal_documents(client, collection, min_words=100, max_words=200):
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
            # Đảm bảo tất cả giá trị trong metadata là string khi lưu JSON
            for chunk in chunks_for_json:
                chunk["metadata"]["dieu_list"] = [str(x) for x in chunk["metadata"]["dieu_list"]]
                chunk["metadata"]["khoan_list"] = [str(x) for x in chunk["metadata"]["khoan_list"]]
                chunk["metadata"]["diem_list"] = [str(x) for x in chunk["metadata"]["diem_list"]]
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_for_json, f, ensure_ascii=False, indent=2)
            logging.info(f"Đã lưu {len(chunks_for_json)} chunk vào file JSON: {json_file}")
        except Exception as e:
            logging.error(f"Lỗi khi lưu file JSON: {e}")
            sys.exit(1)
        for i, chunk in enumerate(texts):
            word_count = count_words(chunk)
            if not (min_words <= word_count <= max_words):
                logging.warning(f"Chunk {i+1} có số từ không hợp lệ: {word_count}")
            logging.info(f"Chunk {i+1} sample: {chunk[:100]}...")
        logging.info(f"Đã xử lý {len(texts)} chunk từ {len(documents)} tài liệu")
        return texts, chunks_for_json
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu: {e}")
        return [], []
    
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(google_exceptions.InternalServerError)
)
def embed_single_text(text: str, request_id: str) -> dict:
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
    return response

def batch_embed_texts(texts: list, batch_size: int = 10) -> list:
    """Nhúng hàng loạt văn bản thành vector.
    
    Args:
        texts (list): Danh sách văn bản cần nhúng.
        batch_size (int, optional): Kích thước batch. Mặc định là 10.
    
    Returns:
        list: Danh sách các vector embedding.
    """
    embeddings = []
    request_id = generate_request_id()
    try:
        sample_embedding = embed_single_text("test", request_id)['embedding']
        embedding_dim = len(sample_embedding)
        logging.info(f"Kích thước embedding: {embedding_dim}")
    except Exception as e:
        logging.error(f"Lỗi khi lấy kích thước embedding: {e}")
        sys.exit(1)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = []
        for text in batch_texts:
            try:
                if not text.strip():
                    logging.warning(f"[Request {request_id}] Text rỗng: {text[:50]}...")
                    batch_embeddings.append([0] * embedding_dim)
                    continue
                response = embed_single_text(text, request_id)
                embedding = response['embedding']
                if len(embedding) != embedding_dim:
                    logging.error(f"[Request {request_id}] Embedding không hợp lệ: {text[:50]}... Kích thước: {len(embedding)}")
                    batch_embeddings.append([0] * embedding_dim)
                else:
                    batch_embeddings.append(embedding)
                    logging.info(f"[Request {request_id}] Nhúng thành công: {text[:50]}... Length: {len(embedding)}")
            except Exception as e:
                logging.error(f"[Request {request_id}] Lỗi khi nhúng: {text[:50]}...: {e}")
                batch_embeddings.append([0] * embedding_dim)
        embeddings.extend(batch_embeddings)
        logging.info(f"Đã nhúng batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")
    return embeddings

def save_chunk(current_chunk, current_word_count, min_words, doc_id, ten_luat, so_chuong, ten_chuong, current_dieu_list, current_khoan_list, current_diem_list, texts, chunks_for_json):
    """Lưu chunk vào danh sách."""
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
        # Kiểm tra kiểu dữ liệu
        for field, values in [("dieu_list", metadata["dieu_list"]), ("khoan_list", metadata["khoan_list"]), ("diem_list", metadata["diem_list"])]:
            for value in values:
                if not isinstance(value, str):
                    logging.error(f"Lỗi kiểu dữ liệu trong {field}: Giá trị {value} (kiểu {type(value)}) không phải string")
                    logging.error(f"Chunk text: {chunk_text[:200]}...")
                    logging.error(f"Metadata: {metadata}")
                    sys.exit(1)
        chunks_for_json.append({"text": chunk_text, "metadata": metadata})
    return texts, chunks_for_json

def create_weaviate_store(texts: list, chunks_for_json: list):
    client = None
    try:
        import weaviate.classes as wvc
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            skip_init_checks=True,
            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=30))
        )
        logging.info("Kết nối thành công với Weaviate")
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
            "vectorIndexConfig": {"distanceMetric": "cosine"}
        }
        client.collections.delete("LegalDocument")
        client.collections.create_from_dict(schema)
        logging.info("Đã tạo schema LegalDocument")
        embeddings = batch_embed_texts(texts)
        if len(embeddings) != len(texts):
            logging.error("Số lượng embedding không khớp với số lượng text")
            sys.exit(1)
        collection = client.collections.get("LegalDocument")
        failed_objects = []
        for i, chunk_data in enumerate(chunks_for_json):
            metadata = chunk_data["metadata"]
            for field, values in [("dieu_list", metadata["dieu_list"]), ("khoan_list", metadata["khoan_list"]), ("diem_list", metadata["diem_list"])]:
                for value in values:
                    if not isinstance(value, str):
                        logging.error(f"Lỗi kiểu dữ liệu trước khi lưu vào Weaviate trong {field}: Giá trị {value} (kiểu {type(value)}) không phải string")
                        sys.exit(1)
        with collection.batch.dynamic() as batch:
            for i, (text, chunk_data, embedding) in enumerate(zip(texts, chunks_for_json, embeddings)):
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
                    if (i + 1) % 10 == 0:
                        logging.info(f"Đã lưu {i + 1} đối tượng vào Weaviate")
                except Exception as e:
                    logging.error(f"Lỗi khi lưu đối tượng {i + 1}: {e}")
                    failed_objects.append({"object": obj, "error": str(e)})
        if failed_objects:
            logging.error(f"Có {len(failed_objects)} đối tượng không lưu được:")
            for failed in failed_objects:
                logging.error(f"Đối tượng thất bại: {failed.get('object', failed.get('index'))}, Lỗi: {failed['error']}")
            sys.exit(1)
        else:
            logging.info(f"Đã lưu thành công {len(texts)} đối tượng vào Weaviate")
    except Exception as e:
        logging.error(f"Lỗi khi tạo Weaviate store: {e}")
        sys.exit(1)
    finally:
        if client:
            client.close()
            logging.info("Đã đóng kết nối Weaviate")


def check_weaviate_data():
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get("LegalDocument")
        response = collection.query.fetch_objects(limit=10, include_vector=True)
        logging.info(f"Tổng số đối tượng: {len(response.objects)}")
        for obj in response.objects:
            logging.info(f"Chunk ID: {obj.properties['chunk_id']}, Text: {obj.properties['text'][:200]}...")
            vector = obj.vector
            if isinstance(vector, dict):
                # Lấy vector từ dict, giả sử key là 'default' (kiểm tra log Weaviate để xác nhận key)
                vector = vector.get('default', [])
            logging.info(f"Vector type: {type(vector)}, Vector length: {len(vector)}")
            logging.info(f"Vector sample: {vector[:5]}")
            logging.info(f"Keywords: {obj.properties.get('keywords', [])}")
            logging.info(f"Dieu list: {obj.properties['dieu_list']}")
        return len(response.objects) > 0
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra dữ liệu Weaviate: {e}")
        return False
    finally:
        if client:
            client.close()
            logging.info("Đã đóng kết nối Weaviate")

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
            logging.info("Đã đóng kết nối Weaviate")

# def search_weaviate(client: weaviate.Client, query: str, k: int = 3, request_id: str = None, alpha: float = 0.75) -> list:
#     try:
#         collection = client.collections.get("LegalDocument")
#         response = embed_single_text(query, request_id)
#         query_embedding = response['embedding']
#         logging.info(f"[Request {request_id}] Query embedding length: {len(query_embedding)}")
#         result = collection.query.hybrid(
#             query=query,
#             vector=query_embedding,
#             alpha=alpha,
#             limit=k,
#             return_metadata=["distance"]
#         )
#         results_list = []
#         logging.info(f"[Request {request_id}] Query: {query}, Total results: {len(result.objects)}")
#         for obj in result.objects:
#             meta = obj.properties
#             distance = obj.metadata.distance
#             relevance_score = 1 - distance if distance is not None else 0.0
#             logging.info(f"[Request {request_id}] Chunk: {meta['chunk_id']}, Distance: {distance}, Score: {relevance_score:.4f}")
#             results_list.append({
#                 'text': meta['text'],
#                 'score': relevance_score,
#                 'metadata': meta
#             })
#         return results_list
#     except Exception as e:
#         logging.error(f"[Request {request_id}] Lỗi tìm kiếm: {e}")
#         return []
    
def search_weaviate(client, query, k=3, request_id=None, alpha=0.5):
    try:
        collection = client.collections.get("LegalDocument")
        results_list = []
        
        # Thử hybrid query
        response = embed_single_text(query, request_id)
        query_embedding = response['embedding']
        logging.info(f"[Request {request_id}] Query embedding length: {len(query_embedding)}")
        hybrid_result = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=k,
            return_metadata=["distance", "score", "certainty"]
        )
        logging.info(f"[Request {request_id}] Hybrid Query: {query}, Alpha: {alpha}, Total results: {len(hybrid_result.objects)}")
        for obj in hybrid_result.objects:
            meta = obj.properties
            distance = obj.metadata.distance
            score = obj.metadata.score
            certainty = obj.metadata.certainty
            relevance_score = 1 - distance if distance is not None else 0.0
            logging.info(f"[Request {request_id}] Hybrid Chunk: {meta['chunk_id']}, Distance: {distance}, Score: {score}, Certainty: {certainty}, Relevance: {relevance_score:.4f}")
            results_list.append({
                'text': meta['text'],
                'score': relevance_score,
                'metadata': meta,
                'type': 'hybrid'
            })
        
        # Nếu hybrid thất bại, dùng workaround
        if all(r['score'] == 0 for r in results_list):
            logging.info(f"[Request {request_id}] Hybrid query không trả về distance, chuyển sang kết hợp BM25 và near_vector")
            results_list = []
            bm25_result = collection.query.bm25(
                query=query,
                limit=k,
                return_metadata=["score"]
            )
            bm25_scores = {obj.properties['chunk_id']: obj.metadata.score or 0.0 for obj in bm25_result.objects}
            vector_result = collection.query.near_vector(
                near_vector=query_embedding,
                limit=k,
                return_metadata=["distance"]
            )
            vector_scores = {obj.properties['chunk_id']: 1 - (obj.metadata.distance or 0.0) for obj in vector_result.objects}
            all_chunk_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
            for chunk_id in all_chunk_ids:
                bm25_score = bm25_scores.get(chunk_id, 0.0)
                vector_score = vector_scores.get(chunk_id, 0.0)
                combined_score = alpha * vector_score + (1 - alpha) * bm25_score
                obj = next((o for o in bm25_result.objects + vector_result.objects if o.properties['chunk_id'] == chunk_id), None)
                if obj:
                    meta = obj.properties
                    logging.info(f"[Request {request_id}] Combined Chunk: {chunk_id}, BM25: {bm25_score:.4f}, Vector: {vector_score:.4f}, Combined: {combined_score:.4f}")
                    results_list.append({
                        'text': meta['text'],
                        'score': combined_score,
                        'metadata': meta,
                        'type': 'combined'
                    })
            results_list = sorted(results_list, key=lambda x: x['score'], reverse=True)[:k]
        
        return results_list
    except Exception as e:
        logging.error(f"[Request {request_id}] Lỗi tìm kiếm: {e}")
        return []