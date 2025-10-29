import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
from openai import OpenAI
import argparse
import sys
def load_config(config_path='config.yaml'):
    """加载YAML配置文件"""
    print(f"--- 正在加载配置文件: {config_path} ---")
    if not os.path.exists(config_path):
        print(f"错误: 配置文件未找到 {config_path}")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件加载成功。")
        return config
    except yaml.YAMLError as e:
        print(f"解析配置文件时出错: {e}")
        return None
    except Exception as e:
        print(f"加载配置文件时发生未知错误: {e}")
        return None

# --- 全局变量 (用于缓存加载的对象) ---
loaded_kb = {}
loaded_models = {}
llm_clients = {}

def load_knowledge_base(kb_version, base_path):
    """加载指定版本的知识库"""
    if kb_version in loaded_kb:
        return loaded_kb[kb_version]

    print(f"--- 正在加载知识库版本: {kb_version} ---")
    index_directory = os.path.join(base_path, f'faiss_index_{kb_version}') # 使用配置的基础路径
    index_filepath = os.path.join(index_directory, 'faiss.index')
    metadata_filepath = os.path.join(index_directory, 'metadata.json')
    config_filepath = os.path.join(index_directory, 'config.json') # 知识库自身的配置

    if not all([os.path.exists(f) for f in [index_filepath, metadata_filepath, config_filepath]]):
        print(f"错误: 知识库文件缺失 ({kb_version})。检查路径: {os.path.abspath(index_directory)}")
        return None, None, None

    try:
        faiss_index = faiss.read_index(index_filepath)
        with open(metadata_filepath, 'r', encoding='utf-8') as f: metadata = json.load(f)
        with open(config_filepath, 'r', encoding='utf-8') as f: kb_config = json.load(f)
        
        loaded_kb[kb_version] = (faiss_index, metadata, kb_config)
        print("知识库加载成功。")
        return faiss_index, metadata, kb_config
    except Exception as e:
        print(f"加载知识库时出错: {e}"); return None, None, None

def load_retriever_model(model_name, cache_path):
    """加载句向量模型"""
    if model_name in loaded_models:
        return loaded_models[model_name]

    print(f"--- 正在加载句向量模型: {model_name} ---")
    # 确保缓存路径存在
    os.makedirs(cache_path, exist_ok=True) 
    try:
        model = SentenceTransformer(model_name, cache_folder=cache_path, device='cpu')
        loaded_models[model_name] = model
        print("句向量模型加载成功。")
        return model
    except Exception as e:
        print(f"加载句向量模型时出错: {e}"); return None

def setup_llm_client(provider, config):
    """根据配置配置并返回LLM客户端"""
    if provider in llm_clients:
        return llm_clients[provider]

    print(f"--- 正在配置LLM客户端: {provider} ---")
    if provider == "deepseek":
        api_key_env = config.get('api_key_env_var', 'DEEPSEEK_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"错误: 未找到环境变量 {api_key_env}。")
            return None
        base_url = config.get('base_url', "https://api.deepseek.com/v1")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            llm_clients[provider] = client
            print("DeepSeek客户端配置成功。")
            return client
        except Exception as e:
            print(f"配置DeepSeek客户端时出错: {e}"); return None
    else:
        print(f"错误: 不支持的LLM提供商 '{provider}'。")
        return None

def search(query, k, faiss_index, metadata, retriever_model):
    """执行检索步骤"""
    print(f"\n--- 步骤 1: 正在执行向量检索 (k={k}) ---")
    start_time = time.time()
    try:
        query_vector = retriever_model.encode([query])
        distances, indices = faiss_index.search(query_vector.astype('float32'), k)
        
        retrieved_contexts = []
        print("检索到的相关上下文 (Top 3 预览):")
        for i in range(len(indices[0])):
            idx = indices[0][i]; dist = distances[0][i]; chunk = metadata[idx]
            context_item = {'source_name': chunk.get('name', 'N/A'), 'distance': float(dist), 'chunk_text': chunk.get('chunk_text', '')}
            retrieved_contexts.append(context_item)
            if i < 3: print(f"  {i+1}. [来源: {context_item['source_name']}] {context_item['chunk_text'][:80]}...")
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        print(f"检索完成，耗时: {retrieval_time:.4f} 秒。")
        return retrieved_contexts, retrieval_time
    except Exception as e:
        print(f"检索时出错: {e}"); return [], 0

def generate_answer(query, contexts, llm_client, generation_config, system_prompt):
    """执行生成步骤，调用指定的LLM"""
    provider = generation_config.get('provider')
    model_name = generation_config.get(provider, {}).get('model_name', 'default-model')
    print(f"\n--- 步骤 2: 正在调用 {provider} ({model_name}) 生成答案 ---")
    start_time = time.time()
    
    context_str = "\n\n---\n\n".join([f"来源: {ctx['source_name']}\n内容: {ctx['chunk_text']}" for ctx in contexts])
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"背景知识:\n{context_str}\n\n我的问题是: {query}"}]
    full_prompt = json.dumps(messages, ensure_ascii=False, indent=2) # 用于输出
    
    final_answer = ""
    print("\n模型回复:")
    print("-" * 30)
    try:
        if provider == "deepseek":
            stream = llm_client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                stream=True, 
                max_tokens=generation_config.get(provider, {}).get('max_tokens', 2048), 
                temperature=generation_config.get(provider, {}).get('temperature', 0.1)
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_part = chunk.choices[0].delta.content
                    print(content_part, end='', flush=True)
                    final_answer += content_part
            print("\n" + "-" * 30)
        else:
            final_answer = f"错误：不支持的生成提供商 '{provider}'。"
            print(final_answer)

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"生成完成，耗时: {generation_time:.4f} 秒。")
        return final_answer, generation_time, full_prompt

    except Exception as e:
        error_msg = f"调用LLM API时出错: {e}"
        print(error_msg); return error_msg, 0, full_prompt
    
def run_pipeline(query, config):
    """
    执行完整的、可配置的RAG Pipeline。
    """
    # --- 从配置中读取参数 ---
    kb_version = config.get('knowledge_base', {}).get('version', 'default_kb')
    kb_base_path = config.get('knowledge_base', {}).get('base_path', './')
    retrieval_k = config.get('retriever', {}).get('top_k', 5)
    model_cache_path = config.get('retriever', {}).get('model_cache_path', '../models')
    
    generator_config = config.get('generator', {})
    llm_provider = generator_config.get('provider', 'deepseek')
    
    prompts_config = config.get('prompts', {})
    active_prompt_key = prompts_config.get('active_template', 'A_BASELINE')
    system_prompt = prompts_config.get('templates', {}).get(active_prompt_key, "Default prompt").strip()
    
    output_metrics = config.get('output', {}).get('additional_metrics', [])

    # --- 1. 加载资源 ---
    faiss_index, metadata, kb_config = load_knowledge_base(kb_version, kb_base_path)
    if not faiss_index: return {"error": "无法加载知识库"}
    
    retriever_model_name = kb_config.get('model_name')
    if not retriever_model_name: return {"error": "知识库配置文件中缺少模型名称"}
    retriever_model = load_retriever_model(retriever_model_name, model_cache_path)
    if not retriever_model: return {"error": "无法加载句向量模型"}

    llm_client = setup_llm_client(llm_provider, generator_config.get(llm_provider, {}))
    if not llm_client: return {"error": "无法配置LLM客户端"}

    # --- 2. 执行流程 ---
    print(f"\n{'='*20} 开始执行RAG流程 {'='*20}")
    print(f"用户查询: {query}")
    print(f"配置: KB={kb_version}, K={retrieval_k}, LLM={llm_provider}, Prompt={active_prompt_key}")

    contexts, retrieval_time = search(query, retrieval_k, faiss_index, metadata, retriever_model)
    final_answer, generation_time, full_prompt = generate_answer(query, contexts, llm_client, generator_config, system_prompt)
    
    print(f"\n{'='*20} RAG流程执行完毕 {'='*20}")

    # --- 3. 组装并返回输出 ---
    output = {"final_answer": final_answer}
    if "retrieved_contexts" in output_metrics:
        output["retrieved_contexts"] = contexts
    if "retrieval_time" in output_metrics:
        output["retrieval_time"] = retrieval_time
    if "generation_time" in output_metrics:
        output["generation_time"] = generation_time
    if "full_prompt" in output_metrics:
        output["full_prompt"] = full_prompt
        
    return output


if __name__ == "__main__":
    # --- 加载配置 ---
    config = load_config('config.yaml')
    if not config:
        sys.exit(1) # 如果配置加载失败，则退出

    # --- 获取查询 ---
    default_query = "头晕是什么原因引起的？" 
    parser = argparse.ArgumentParser(description='可配置的RAG Pipeline脚本')
    parser.add_argument('-q', '--query', type=str, help='您想问的问题 (覆盖配置文件中的默认问题)')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='指定配置文件的路径')
    args = parser.parse_args()

    if args.config != 'config.yaml':
        config = load_config(args.config)
        if not config:
            sys.exit(1)
            
    query_to_run = args.query if args.query else default_query
    
    # --- 执行Pipeline ---
    result = run_pipeline(query_to_run, config)
