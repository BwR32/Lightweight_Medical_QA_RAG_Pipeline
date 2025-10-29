import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 消除tokenizer警告

import streamlit as st
import yaml
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time
import sys


# 使用 Streamlit 的缓存装饰器来加载配置文件，只在文件变化时重新加载
@st.cache_data
def load_config(config_path='config.yaml'):
    """加载YAML配置文件"""
    print(f"--- 正在加载配置文件: {config_path} ---") # 在后台打印日志
    if not os.path.exists(config_path):
        st.error(f"错误: 配置文件未找到 {config_path}")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件加载成功。")
        return config
    except Exception as e:
        st.error(f"加载配置文件时出错: {e}")
        return None

# 使用 Streamlit 的资源缓存装饰器来加载耗时资源，只在应用启动时加载一次
@st.cache_resource
def load_knowledge_base(kb_version, base_path):
    """加载指定版本的知识库 (FAISS索引, 元数据, 配置)"""
    print(f"--- 正在加载知识库版本: {kb_version} ---")
    index_directory = os.path.join(base_path, f'faiss_index_{kb_version}')
    index_filepath = os.path.join(index_directory, 'faiss.index')
    metadata_filepath = os.path.join(index_directory, 'metadata.json')
    config_filepath = os.path.join(index_directory, 'config.json')

    if not all([os.path.exists(f) for f in [index_filepath, metadata_filepath, config_filepath]]):
        st.error(f"错误: 知识库文件缺失 ({kb_version})。检查路径: {os.path.abspath(index_directory)}")
        return None, None, None

    try:
        faiss_index = faiss.read_index(index_filepath)
        with open(metadata_filepath, 'r', encoding='utf-8') as f: metadata = json.load(f)
        with open(config_filepath, 'r', encoding='utf-8') as f: kb_config = json.load(f)
        print("知识库加载成功。")
        return faiss_index, metadata, kb_config
    except Exception as e:
        st.error(f"加载知识库时出错: {e}"); return None, None, None

@st.cache_resource
def load_retriever_model(model_name, cache_path):
    """加载句向量模型"""
    print(f"--- 正在加载句向量模型: {model_name} ---")
    os.makedirs(cache_path, exist_ok=True)
    try:
        model = SentenceTransformer(model_name, cache_folder=cache_path, device='cpu')
        print("句向量模型加载成功。")
        return model
    except Exception as e:
        st.error(f"加载句向量模型时出错: {e}"); return None

@st.cache_resource
def setup_llm_client(provider, config):
    """配置并返回LLM客户端"""
    print(f"--- 正在配置LLM客户端: {provider} ---")
    if provider == "deepseek":
        api_key_env = config.get('api_key_env_var', 'DEEPSEEK_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            st.error(f"错误: 未找到环境变量 {api_key_env}。请在运行Streamlit前设置。")
            return None
        base_url = config.get('base_url', "https://api.deepseek.com/v1")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            print("DeepSeek客户端配置成功。")
            return client
        except Exception as e:
            st.error(f"配置DeepSeek客户端时出错: {e}"); return None
    else:
        st.error(f"错误: 不支持的LLM提供商 '{provider}'。")
        return None


def search(query, k, faiss_index, metadata, retriever_model):
    """执行检索步骤"""
    start_time = time.time()
    try:
        query_vector = retriever_model.encode([query])
        distances, indices = faiss_index.search(query_vector.astype('float32'), k)
        
        retrieved_contexts = []
        for i in range(len(indices[0])):
            idx = indices[0][i]; dist = distances[0][i]; chunk = metadata[idx]
            context_item = {'source_name': chunk.get('name', 'N/A'), 'distance': float(dist), 'chunk_text': chunk.get('chunk_text', '')}
            retrieved_contexts.append(context_item)
        
        retrieval_time = time.time() - start_time
        print(f"检索完成，耗时: {retrieval_time:.4f} 秒。") # 后台日志
        return retrieved_contexts, retrieval_time
    except Exception as e:
        st.error(f"检索时出错: {e}"); return [], 0

def generate_answer(query, contexts, llm_client, generation_config, system_prompt):
    """执行生成步骤，调用指定的LLM (非流式)"""
    provider = generation_config.get('provider')
    model_name = generation_config.get(provider, {}).get('model_name', 'default-model')
    print(f"--- 正在调用 {provider} ({model_name}) 生成答案 ---") # 后台日志
    start_time = time.time()
    
    context_str = "\n\n---\n\n".join([f"来源: {ctx['source_name']}\n内容: {ctx['chunk_text']}" for ctx in contexts])
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"背景知识:\n{context_str}\n\n我的问题是: {query}"}]
    
    final_answer = ""
    try:
        if provider == "deepseek":
            response = llm_client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                stream=False, # Streamlit更适合非流式，简化处理
                max_tokens=generation_config.get(provider, {}).get('max_tokens', 2048), 
                temperature=generation_config.get(provider, {}).get('temperature', 0.1)
            )
            final_answer = response.choices[0].message.content
        else:
            final_answer = f"错误：不支持的生成提供商 '{provider}'。"
        
        generation_time = time.time() - start_time
        print(f"生成完成，耗时: {generation_time:.4f} 秒。") # 后台日志
        return final_answer, generation_time

    except Exception as e:
        st.error(f"调用LLM API时出错: {e}"); return f"调用LLM API时出错: {e}", 0

# --- Streamlit 界面构建 ---

# --- 加载全局配置 ---
config = load_config()

if config:
    # --- 根据配置加载核心资源 ---
    kb_version = config.get('knowledge_base', {}).get('version', 'default_kb')
    kb_base_path = config.get('knowledge_base', {}).get('base_path', './')
    faiss_index, metadata, kb_config = load_knowledge_base(kb_version, kb_base_path)
    
    retriever_model_name = kb_config.get('model_name') if kb_config else None
    retriever_cache_path = config.get('retriever', {}).get('model_cache_path', '../models')
    retriever_model = load_retriever_model(retriever_model_name, retriever_cache_path) if retriever_model_name else None

    generator_config = config.get('generator', {})
    llm_provider = generator_config.get('provider', 'deepseek')
    llm_client = setup_llm_client(llm_provider, generator_config.get(llm_provider, {}))

    prompts_config = config.get('prompts', {})
    active_prompt_key = prompts_config.get('active_template', 'A_BASELINE')
    system_prompt = prompts_config.get('templates', {}).get(active_prompt_key, "Default prompt").strip()

    # --- 界面元素 ---
    st.set_page_config(page_title="医疗RAG问答系统", layout="wide")
    st.title("⚕️ 医疗健康RAG问答系统")
    st.caption(f"知识库版本: {kb_version} | Prompt: {active_prompt_key}")

    # 使用session state来存储聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "contexts" in message:
                 with st.expander("查看检索到的上下文 ({}条)".format(len(message["contexts"]))):
                    for i, ctx in enumerate(message["contexts"]):
                        st.info(f"**来源:** {ctx['source_name']} (距离: {ctx['distance']:.4f})\n\n**内容:** {ctx['chunk_text']}")
            st.markdown(message["content"])

    # 用户输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 将用户问题添加到聊天记录并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- 执行RAG流程 ---
        if faiss_index and metadata and retriever_model and llm_client:
            with st.chat_message("assistant"):
                # 显示思考状态
                with st.spinner("正在思考中... (检索+生成)"):
                    # 1. 检索
                    retrieval_k = config.get('retriever', {}).get('top_k', 5)
                    contexts, retrieval_time = search(prompt, retrieval_k, faiss_index, metadata, retriever_model)
                    
                    # 2. 生成
                    final_answer, generation_time = generate_answer(prompt, contexts, llm_client, generator_config, system_prompt)
                
                # 3. 显示结果和上下文
                with st.expander("查看检索到的上下文 ({}条)".format(len(contexts))):
                     for i, ctx in enumerate(contexts):
                        st.info(f"**来源:** {ctx['source_name']} (距离: {ctx['distance']:.4f})\n\n**内容:** {ctx['chunk_text']}")
                
                st.markdown(final_answer)

                # 将机器人的回答和上下文存入聊天记录
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_answer, 
                    "contexts": contexts
                })
        else:
            st.error("RAG Pipeline核心组件未能完全加载，无法处理查询。请检查后台日志。")

else:
    st.error("无法加载配置文件 'config.yaml'，应用无法启动。")
