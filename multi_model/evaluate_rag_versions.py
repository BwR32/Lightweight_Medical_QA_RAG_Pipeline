import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

EXPERIMENTS_TO_RUN = [
    "v1_section_text2vec_base",
    "v2_recursive_text2vec_base",
    "v3_recursive_m3e_base",
    "v4_recursive_bge-large-zh-v1.5",
    "v5_recursive_bge-large-zh-v1.5",
    "v6_recursive_bge-large-zh-v1.5",
    #v1_section_text2vec_base : shibing624--text2vec-base-chinese--by_section CHUNK_SIZE = 250
    #v2_recursive_text2vec_base : shibing624--text2vec-base-chinese--recuraive CHUNK_SIZE = 250
    #v3_recursive_m3e_base : moka-ai/m3e-base -- recursive CHUNK_SIZE = 250
    #v4_recursive_bge-large-zh-v1.5 : BAAI/bge-large-zh-v1.5 --recursive CHUNK_SIZE = 250
    #v5_recursive_bge-large-zh-v1.5 : BAAI/bge-large-zh-v1.5 --recursive CHUNK_SIZE = 128
    #v6_recursive_bge-large-zh-v1.5 : BAAI/bge-large-zh-v1.5 --recursive CHUNK_SIZE = 512
]

TEST_BENCHMARK = {
    # 症状查询
    "什么是偏头痛，它和普通头痛有什么区别？": ["偏头痛", "头痛"],
    "经常感觉疲劳乏力是怎么回事？": ["疲劳", "乏力"],
    "晚上睡觉时小腿抽筋是什么原因？": ["抽筋", "肌肉痉挛"],
    "皮肤上出现红疹还很痒，可能是哪些情况？": ["红疹", "皮疹", "瘙痒"],
    "除了感冒，还有什么病会引起喉咙痛？": ["喉咙痛", "咽痛"],
    "如何缓解胃酸反流（烧心）的症状？": ["胃酸反流", "烧心"],
    # 疾病查询
    "请详细介绍一下2型糖尿病。": ["2型糖尿病", "糖尿病"],
    "高血压的诊断标准是什么？": ["高血压"],
    "如何预防骨质疏松？": ["骨质疏松"],
    "胆结石必须要做手术吗？有哪些治疗方法？": ["胆结石"],
    "抑郁症的早期症状有哪些？": ["抑郁症"],
    "荨麻疹的病因是什么？能根治吗？": ["荨麻疹"],
    # 药品查询
    "布洛芬缓释胶囊是用来做什么的？": ["布洛芬"],
    "阿莫西林的用法用量和注意事项是什么？": ["阿莫西林"],
    "蒙脱石散有哪些副作用？": ["蒙脱石散"],
    "请问“川芎口服液”的成分是什么？": ["川芎口服液"],
    "哺乳期妇女可以服用对乙酰氨基酚吗？": ["对乙酰氨基酚"],
    "降压药需要终身服用吗？": ["降压药", "高血压"],
    # 检查与治疗查询
    "核磁共振（MRI）检查是用来做什么的？": ["核磁共振"],
    "什么是靶向治疗？": ["靶向治疗"],
    "做胃镜之前需要做哪些准备？": ["胃镜"],
    "“血常规”检查能查出什么问题？": ["血常规"],
    # 复合与场景化查询
    "我最近总是头晕，还恶心想吐，会是什么病？": ["头晕", "恶心", "呕吐"],
    "孩子发烧咳嗽，可以吃点头孢吗？": ["发烧", "咳嗽", "头孢"],
    "运动后膝盖疼，是应该冷敷还是热敷？": ["膝盖疼", "关节痛"],
    "我妈妈有高血压，饮食上应该注意些什么？": ["高血压"],
    "感冒和流感有什么不一样，怎么区分？": ["感冒", "流感"],
    "长期失眠应该怎么办，有什么推荐的治疗方法？": ["失眠"],
    "吃完海鲜后身上起了很多红点，非常痒，我该怎么办？": ["过敏", "荨麻疹", "红点"],
    "体检发现尿酸高，需要吃药吗？平时要注意什么？": ["尿酸高", "痛风"],
}

TOP_K = 3


class Evaluator:
    def __init__(self):
        self.models = {}
        self.indexes = {}

    def _load_model(self, model_cache_path, model_name):
        """加载或从缓存中获取模型"""
        if model_name in self.models:
            return self.models[model_name]
        
        print(f"\n首次加载模型: '{model_name}'...")
        
        model = SentenceTransformer(model_name, cache_folder=model_cache_path, device='cpu')
        
        self.models[model_name] = model
        print("模型加载完成。")
        return model

    def _load_index(self, experiment_name):
        """加载或从缓存中获取索引、配置和元数据"""
        if experiment_name in self.indexes:
            return self.indexes[experiment_name]

        print(f"\n首次加载实验版本 '{experiment_name}' 的索引...")
        index_directory = f'./faiss_index_{experiment_name}'
        index_filepath = os.path.join(index_directory, 'faiss.index')
        metadata_filepath = os.path.join(index_directory, 'metadata.json')
        config_filepath = os.path.join(index_directory, 'config.json')

        if not all([os.path.exists(f) for f in [index_filepath, metadata_filepath, config_filepath]]):
            print(f"错误: 无法加载 '{experiment_name}'，文件缺失。请检查路径: {index_directory}")
            return None, None, None

        index = faiss.read_index(index_filepath)
        with open(metadata_filepath, 'r', encoding='utf-8') as f: metadata = json.load(f)
        with open(config_filepath, 'r', encoding='utf-8') as f: config = json.load(f)
        
        self.indexes[experiment_name] = (index, config, metadata)
        print("索引加载成功。")
        return index, config, metadata

    def evaluate_question(self, model_cache_path, experiment_name, question, target_keywords):
        """对单个问题进行评测"""
        index, config, metadata = self._load_index(experiment_name)
        if index is None:
            return False, ["加载失败"]

        model_name = config.get('model_name')
        model = self._load_model(model_cache_path, model_name)

        query_vector = model.encode([question])
        _, indices = index.search(query_vector.astype('float32'), TOP_K)

        is_hit, retrieved_results = False, []
        for i in range(TOP_K):
            idx = indices[0][i]
            retrieved_chunk = metadata[idx]
            retrieved_title = retrieved_chunk.get('name', '')
            retrieved_results.append(retrieved_title)

            if any(keyword in retrieved_title for keyword in target_keywords):
                is_hit = True
        
        return is_hit, retrieved_results

def generate_report(results):
    """根据评测结果生成Markdown报告"""
    report_content = "# RAG知识库版本评测报告\n\n## 1. 总体性能概览 (Hit Rate @{})\n\n".format(TOP_K)
    report_content += "| 实验版本 (Experiment) | 命中率 (Hit Rate) | 命中数/总数 |\n| :--- | :--- | :--- |\n"
    sorted_results = sorted(results.items(), key=lambda item: item[1]['score'], reverse=True)
    for name, res in sorted_results:
        total, hits, score_percent = res['total_questions'], res['hits'], res['score'] * 100
        bar = "█" * int(score_percent / 5)
        report_content += f"| **{name}** | **{score_percent:.2f}%** `{bar}` | {hits} / {total} |\n"
    report_content += "\n## 2. 各问题详细评测结果\n\n"
    report_content += "| 问题 (Query) | " + " | ".join(results.keys()) + " |\n| :--- | " + " | ".join([":---:"] * len(results)) + " |\n"
    for i, (question, _) in enumerate(TEST_BENCHMARK.items()):
        row = f"| **Q{i+1}:** {question[:20]}... |"
        for name in results.keys():
            detail = results[name]['details'][i]
            symbol = "✅" if detail['hit'] else "❌"
            retrieved_str = ", ".join(detail['retrieved']).replace("\n", " ")
            row += f" {symbol}<br><small>*{retrieved_str}*</small> |"
        report_content += row + "\n"
    best_version = sorted_results[0][0]
    report_content += f"\n## 3. 结论\n\n根据本次基于 {len(TEST_BENCHMARK)} 个问题的自动化评测，**'{best_version}'** 版本在 Hit Rate @{TOP_K} 指标上表现最佳。\n"
    with open("evaluation_report.md", "w", encoding="utf-8") as f: f.write(report_content)
    print("\n🎉 评测报告已生成: evaluation_report.md")


def main():

    evaluator = Evaluator()
    model_cache_path = '../models' 
    
    results = {}

    for experiment_name in EXPERIMENTS_TO_RUN:
        print(f"\n{'='*20} 正在评测版本: {experiment_name} {'='*20}")
        hits, details = 0, []
        
        for i, (question, target_keywords) in enumerate(TEST_BENCHMARK.items()):
            print(f"  -> Q{i+1}/{len(TEST_BENCHMARK)}: {question}")
            is_hit, retrieved = evaluator.evaluate_question(model_cache_path, experiment_name, question, target_keywords)
            if is_hit: hits += 1
            details.append({'question': question, 'hit': is_hit, 'retrieved': retrieved})
            
        total = len(TEST_BENCHMARK)
        score = hits / total if total > 0 else 0
        results[experiment_name] = {'score': score, 'hits': hits, 'total_questions': total, 'details': details}
        print(f"\n版本 '{experiment_name}' 评测完成: 命中率 = {score*100:.2f}% ({hits}/{total})")

    generate_report(results)

if __name__ == "__main__":
    main()
