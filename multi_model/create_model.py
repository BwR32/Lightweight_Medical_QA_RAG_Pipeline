import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time

EXPERIMENT_NAME = "v6_recursive_bge-large-zh-v1.5"
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
CHUNK_STRATEGY = 'recursive' 
CHUNK_SIZE = 512
CHUNK_OVERLAP = 40

def load_cleaned_data(filepath):
    print(f"正在加载清洗后的数据从: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析文件 {filepath}: {e}")
        return None

def chunk_documents(data, strategy, chunk_size, chunk_overlap):
    print(f"\n正在使用 '{strategy}' 策略进行文本切分...")
    
    all_chunks = []
    all_metadata = []

    for entry in data:
        full_content_parts = [entry['name']]
        if isinstance(entry.get('sections'), dict):
            for title, content in entry['sections'].items():
                text = f"{title}: {' '.join(content) if isinstance(content, list) else content}"
                full_content_parts.append(text)
        full_document_text = "\n\n".join(full_content_parts)

        base_metadata = {'name': entry['name'], 'id': entry['id'], 'url': entry['url'], 'type': entry['type']}

        if strategy == 'by_section':
            if isinstance(entry.get('sections'), dict) and entry['sections']:
                for title, content in entry['sections'].items():
                    chunk_text = f"{entry['name']} - {title}: {' '.join(content) if isinstance(content, list) else content}"
                    all_chunks.append(chunk_text)
                    all_metadata.append(base_metadata)
            else:
                all_chunks.append(full_document_text)
                all_metadata.append(base_metadata)

        elif strategy == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", "。", "，", ""]
            )
            chunks = text_splitter.split_text(full_document_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append(base_metadata)

    print(f"切分完成，共生成 {len(all_chunks)} 个知识片段 (Chunks)。")
    return all_chunks, all_metadata

def main():
    
    # [*** 核心修正 ***]
    # 根据您新的文件结构，更新输入和输出文件的相对路径
    # '..' 代表上一级目录，所以 '../..' 代表上两级目录
    cleaned_file_path = '../cleaned_data/cleaned_medical_data_v2.json'
    model_cache_path = '../models' # 建议将models文件夹放在项目根目录
    
    # 输出文件夹将创建在当前脚本所在的 muti_model/ 目录下
    output_directory = f'./faiss_index_{EXPERIMENT_NAME}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    index_filepath = os.path.join(output_directory, 'faiss.index')
    metadata_filepath = os.path.join(output_directory, 'metadata.json')
    config_filepath = os.path.join(output_directory, 'config.json')

    medical_data = load_cleaned_data(cleaned_file_path)
    if not medical_data: return

    corpus_chunks, metadata = chunk_documents(medical_data, CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"\n正在加载/下载 Sentence Transformer 模型: '{MODEL_NAME}'")
    print(f"所有模型将被缓存到: {os.path.abspath(model_cache_path)}")
    model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_path, device='cpu')

    print("模型加载完成。开始将所有知识片段向量化...")
    start_time = time.time()
    embeddings = model.encode(corpus_chunks, show_progress_bar=True)
    end_time = time.time()
    print(f"向量化完成，耗时: {end_time - start_time:.2f} 秒。")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"\nFAISS索引已创建，共包含 {index.ntotal} 个向量。")

    print(f"正在将FAISS索引保存到: {index_filepath}")
    faiss.write_index(index, index_filepath)
    
    for i in range(len(metadata)):
        metadata[i]['chunk_text'] = corpus_chunks[i]

    print(f"正在将元数据保存到: {metadata_filepath}")
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    config_data = {'experiment_name': EXPERIMENT_NAME, 'model_name': MODEL_NAME}
    print(f"正在将配置信息保存到: {config_filepath}")
    with open(config_filepath, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)

    print(f"\n🎉 实验版本 '{EXPERIMENT_NAME}' 创建成功！")

if __name__ == "__main__":
    main()
