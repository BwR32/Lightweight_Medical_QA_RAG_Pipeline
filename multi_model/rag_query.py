import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
from openai import OpenAI
import argparse # 用于接收命令行参数

EXPERIMENT_NAME = "v6_recursive_bge-large-zh-v1.5"
RETRIEVAL_K = 25
SYSTEM_PROMPT = "你是一个医疗信息摘要AI。严格根据“背景知识”回答问题。核心规则：1. 绝对忠实于背景知识，禁止外部信息。2. 答案必须简洁至上，直击要点，省略客套话。3. 如果知识不足，只回答：“根据提供的资料，无法回答此问题。”"

# --- API 密钥 ---
#DEEPSEEK_API_KEY：sk-fc4cac1d2b1e41ebb5185ed612414325 MAC/Linux 可在终端输入：export DEEPSEEK_API_KEY="sk-fc4cac1d2b1e41ebb5185ed612414325"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 全局变量
retriever_model, faiss_index, metadata, deepseek_client = None, None, None, None

def load_knowledge_base():
    """加载知识库和句向量模型"""
    global retriever_model, faiss_index, metadata
    print("正在初始化知识库和模型...") 
    model_cache_path = '../models'
    index_directory = f'./faiss_index_{EXPERIMENT_NAME}'
    index_filepath = os.path.join(index_directory, 'faiss.index')
    metadata_filepath = os.path.join(index_directory, 'metadata.json')
    config_filepath = os.path.join(index_directory, 'config.json')
    if not all([os.path.exists(f) for f in [index_filepath, metadata_filepath, config_filepath]]):
        print(f"错误: 知识库文件缺失 ({EXPERIMENT_NAME})。")
        return False
    faiss_index = faiss.read_index(index_filepath)
    with open(metadata_filepath, 'r', encoding='utf-8') as f: metadata = json.load(f)
    with open(config_filepath, 'r', encoding='utf-8') as f: config = json.load(f)
    model_name = config.get('model_name')
    retriever_model = SentenceTransformer(model_name, cache_folder=model_cache_path, device='cpu')
    print("初始化完成。")
    return True

def setup_deepseek_client():
    """配置DeepSeek API客户端"""
    global deepseek_client
    if not DEEPSEEK_API_KEY:
        print("错误: 未找到DeepSeek API密钥。")
        return False
    deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    return True

def search(query, k):
    """执行检索步骤"""
    query_vector = retriever_model.encode([query])
    _, indices = faiss_index.search(query_vector.astype('float32'), k)
    return [metadata[idx] for idx in indices[0]]

def generate_answer(system_prompt, query, contexts):
    """执行生成步骤，调用DeepSeek LLM"""
    context_str = "\n\n---\n\n".join([f"来源: {ctx['name']}\n内容: {ctx['chunk_text']}" for ctx in contexts])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"背景知识:\n{context_str}\n\n我的问题是: {query}"}
    ]
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat", messages=messages, stream=False, max_tokens=2048, temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用DeepSeek API时出错: {e}"

# ---主流程 ---

def main(query_text):
    """主函数，接收查询，执行RAG流程，并只打印最终答案"""
    if not load_knowledge_base() or not setup_deepseek_client():
        return

    print(f"\n正在处理查询: {query_text}")
    
    # 1. 检索 (Retrieval) - 在后台进行，不打印
    contexts = search(query_text, k=RETRIEVAL_K)
    
    # 2. 生成 (Generation) - 在后台进行
    final_answer = generate_answer(SYSTEM_PROMPT.strip(), query_text, contexts)
    
    # 3. 输出最终答案
    print("\n模型回答:")
    print("-" * 30)
    print(final_answer)
    print("-" * 30)

if __name__ == "__main__":
    # --- 如何提供查询 ---
    # 方式一：直接修改下面的变量
    default_query = "头晕是什么原因引起的？" 
    
    # 方式二：通过命令行参数传入 (优先级更高)
    parser = argparse.ArgumentParser(description='简单的RAG查询脚本')
    parser.add_argument('-q', '--query', type=str, help='您想问的问题')
    args = parser.parse_args()

    query_to_run = args.query if args.query else default_query
    
    main(query_to_run)

### 如何使用
# python rag_query.py -q "我妈妈有高血压，饮食上应该注意些什么？"
        
# python rag_query.py --query "布洛芬有哪些副作用？"
