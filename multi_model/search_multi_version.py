import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

#v1_section_text2vec_base : shibing624--text2vec-base-chinese--by_section CHUNK_SIZE = 250
#v2_recursive_text2vec_base : shibing624--text2vec-base-chinese--recuraive CHUNK_SIZE = 250
#v3_recursive_m3e_base : moka-ai/m3e-base -- recursive CHUNK_SIZE = 250
#v4_recursive_bge-large-zh-v1.5 : BAAI/bge-large-zh-v1.5 --recursive CHUNK_SIZE = 250
#v5_recursive_bge-large-zh-v1.5 : BAAI/bge-large-zh-v1.5 --recursive CHUNK_SIZE = 128
#v6_recursive_bge-large-zh-v1.5 : BAAI/bge-large-zh-v1.5 --recursive CHUNK_SIZE = 512

EXPERIMENT_NAME = "v5_recursive_bge-large-zh-v1.5"
QUERY_TEXT = "头晕是什么原因引起的，应该如何治疗？"


def main():
    
    model_cache_path = '../models'
    
    index_directory = f'./faiss_index_{EXPERIMENT_NAME}'
    index_filepath = os.path.join(index_directory, 'faiss.index')
    metadata_filepath = os.path.join(index_directory, 'metadata.json')
    config_filepath = os.path.join(index_directory, 'config.json')

    if not all([os.path.exists(f) for f in [index_filepath, metadata_filepath, config_filepath]]):
        print(f"错误: 实验版本 '{EXPERIMENT_NAME}' 的索引文件、元数据或配置文件未找到。")
        print(f"请确认您已经使用 '{EXPERIMENT_NAME}' 作为 EXPERIMENT_NAME 成功运行了 'create_model.py'。")
        return

    print(f"正在加载实验版本 '{EXPERIMENT_NAME}' 的索引和元数据...")
    index = faiss.read_index(index_filepath)
    with open(metadata_filepath, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    print("加载成功。")

    model_name = config_data.get('model_name')
    if not model_name:
        print("错误：配置文件中未找到模型名称 (model_name)。")
        return
        
    print(f"\n正在从缓存加载模型: '{model_name}'")
    model = SentenceTransformer(model_name, cache_folder=model_cache_path, device='cpu')
    print("模型加载完成。")

    print("\n--- 语义搜索演示 ---")
    
    print(f"正在对查询进行向量化: '{QUERY_TEXT}'")
    query_vector = model.encode([QUERY_TEXT])
    
    k = 5
    distances, indices = index.search(query_vector.astype('float32'), k)

    print(f"\n与查询 '{QUERY_TEXT}' 语义最相关的 {k} 个结果:")
    retrieved_contexts = []
    for i in range(k):
        idx = indices[0][i]
        dist = distances[0][i]
        
        retrieved_chunk = metadata[idx]
        retrieved_contexts.append(retrieved_chunk)
        
        print(f"\n{i+1}. 结果 (来自: '{retrieved_chunk['name']}', 距离: {dist:.4f}):")
        print(f"   - 内容片段: {retrieved_chunk['chunk_text'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
