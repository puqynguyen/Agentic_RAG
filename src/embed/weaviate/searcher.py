import logging
import weaviate
import google.generativeai as genai
from weaviate.classes.query import MetadataQuery, HybridFusion
from .embedder import embed_single_text, generate_request_id

def search_weaviate(client, query, k=3, request_id=None, alpha=0.5):
    """Thực hiện hybrid search trên Weaviate.
    
    Args:
        client: Weaviate client instance.
        query (str): Chuỗi truy vấn.
        k (int): Số kết quả trả về (default: 3).
        request_id (str): ID yêu cầu (default: None, sẽ tự tạo).
        alpha (float): Trọng số giữa vector search và BM25 (0: chỉ BM25, 1: chỉ vector, default: 0.5).
    
    Returns:
        list: Danh sách các kết quả tìm kiếm với text, score, metadata, và type.
    """
    try:
        if request_id is None:
            request_id = generate_request_id()
        collection = client.collections.get("LegalDocument")
        response = embed_single_text(query, request_id)
        query_embedding = response['embedding']
        logging.info(f"[Request {request_id}] Query embedding length: {len(query_embedding)}")
        hybrid_result = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            limit=k,
            return_metadata=MetadataQuery(score=True, explain_score=True)
        )
        results_list = []
        logging.info(f"[Request {request_id}] Hybrid Query: {query}, Alpha: {alpha}, Total results: {len(hybrid_result.objects)}")
        for obj in hybrid_result.objects:
            meta = obj.properties
            score = obj.metadata.score
            explain_score = obj.metadata.explain_score
            logging.info(f"[Request {request_id}] Hybrid Chunk: {meta['chunk_id']}, Score: {score}, Explain: {explain_score}, Relevance: {score:.4f}")
            results_list.append({
                'text': meta['text'],
                'score': score,
                'metadata': meta,
                'type': 'hybrid'
            })
        return results_list
    except Exception as e:
        logging.error(f"[Request {request_id}] Lỗi tìm kiếm: {e}")
        return []