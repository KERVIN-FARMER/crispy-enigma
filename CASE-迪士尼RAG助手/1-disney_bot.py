# -*- coding: utf-8 -*-
"""
迪士尼客服 RAG 助手

它将演示如何处理多种格式的文档（Word），提取文本、表格和图片，
然后使用 Embedding 模型（text-embedding-v4, CLIP）和 FAISS 向量库构建一个
能够回答文本和图片相关问题的智能问答系统。

在运行前，请确保完成以下准备工作：

1. 安装所有必需的 Python 库:
   pip install openai "faiss-cpu" python-docx PyMuPDF Pillow pytesseract transformers torch requests

2. 安装 Google Tesseract OCR 引擎:
   - Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载并安装。
   - macOS: brew install tesseract
   - Linux (Ubuntu): sudo apt-get install tesseract-ocr
   请确保 tesseract 的可执行文件路径已添加到系统的 PATH 环境变量中。

3. 设置环境变量:
   - DASHSCOPE_API_KEY: 您从阿里云百炼平台获取的 API Key。
   - HF_TOKEN: (可选) 您的 Hugging Face Token，用于下载 CLIP 模型，避免手动确认。
"""
import os
import re
import numpy as np
import faiss
from openai import OpenAI
from docx import Document as DocxDocument
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch

# Step0. 全局配置与模型加载

# 检查环境变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

# 初始化百炼兼容的 OpenAI 客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 加载 CLIP 模型用于图像处理 (如果本地没有会自动下载)
print("正在加载 CLIP 模型...")
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP 模型加载成功。")
except Exception as e:
    print(f"加载 CLIP 模型失败，请检查网络连接或 Hugging Face Token。错误: {e}")
    exit()

# 定义全局变量
DOCS_DIR = "disney_knowledge_base"
IMG_DIR = os.path.join(DOCS_DIR, "images")
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
TEXT_EMBEDDING_DIM = 1024
IMAGE_EMBEDDING_DIM = 512 # CLIP 'vit-base-patch32' 模型的输出维度

# Step1. 文档解析与内容提取
def parse_docx(file_path):
    """解析 DOCX 文件，提取文本和表格（转为Markdown）。"""
    doc = DocxDocument(file_path)
    content_chunks = []
    
    for element in doc.element.body:
        if element.tag.endswith('p'):
            # 处理段落
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            
            if paragraph_text.strip():
                content_chunks.append({"type": "text", "content": paragraph_text.strip()})
                
        elif element.tag.endswith('tbl'):
            # 处理表格
            md_table = []
            table = [t for t in doc.tables if t._element is element][0]
            
            if table.rows:
                # 添加表头
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|"*len(header))
                
                # 添加数据行
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                
                table_content = "\n".join(md_table)
                if table_content.strip():
                    content_chunks.append({"type": "table", "content": table_content})
    
    return content_chunks

def image_to_text(image_path):
    """对图片进行OCR和CLIP描述。"""
    try:
        image = Image.open(image_path)
        # OCR
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
        return {"ocr": ocr_text}
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return {"ocr": ""}

# Step2. Embedding 与索引构建
def get_text_embedding(text):
    """获取文本的 Embedding。"""
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIM
    )
    return response.data[0].embedding

def get_image_embedding(image_path):
    """获取图片的 Embedding。"""
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].numpy()

def get_clip_text_embedding(text):
    """使用CLIP的文本编码器获取文本的Embedding。"""
    inputs = clip_processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features[0].numpy()

def build_knowledge_base(docs_dir, img_dir):
    """构建完整的知识库，包括解析、切片、Embedding和索引。"""
    print("\n--- 步骤 1 & 2: 正在解析、Embedding并索引知识库 ---")
    
    metadata_store = []
    text_vectors = []
    image_vectors = []
    
    doc_id_counter = 0

    # 处理Word文档
    for filename in os.listdir(docs_dir):
        if filename.startswith('.') or os.path.isdir(os.path.join(docs_dir, filename)):
            continue
            
        file_path = os.path.join(docs_dir, filename)
        if filename.endswith(".docx"):
            print(f"  - 正在处理: {filename}")
            chunks = parse_docx(file_path)
            
            for chunk in chunks:
                metadata = {
                    "id": doc_id_counter,
                    "source": filename,
                    "page": 1
                }
                
                if chunk["type"] == "text" or chunk["type"] == "table":
                    text = chunk["content"]
                    if not text.strip(): 
                        continue
                    
                    metadata["type"] = "text"
                    metadata["content"] = text
                    
                    vector = get_text_embedding(text)
                    text_vectors.append(vector)
                    metadata_store.append(metadata)
                    doc_id_counter += 1

    # 处理images目录中的独立图片文件
    print("  - 正在处理独立图片文件...")
    for img_filename in os.listdir(img_dir):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(img_dir, img_filename)
            print(f"    - 处理图片: {img_filename}")
            
            img_text_info = image_to_text(img_path)
            
            metadata = {
                "id": doc_id_counter,
                "source": f"独立图片: {img_filename}",
                "type": "image",
                "path": img_path,
                "ocr": img_text_info["ocr"],
                "page": 1
            }
            
            vector = get_image_embedding(img_path)
            image_vectors.append(vector)
            metadata_store.append(metadata)
            doc_id_counter += 1

    # 创建 FAISS 索引
    # 文本索引
    text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
    text_index_map = faiss.IndexIDMap(text_index)
    text_ids = [m["id"] for m in metadata_store if m["type"] == "text"]
    if text_vectors:  # 只有当有文本向量时才添加到索引
        text_index_map.add_with_ids(np.array(text_vectors).astype('float32'), np.array(text_ids))
    
    # 图像索引
    image_index = faiss.IndexFlatL2(IMAGE_EMBEDDING_DIM)
    image_index_map = faiss.IndexIDMap(image_index)
    image_ids = [m["id"] for m in metadata_store if m["type"] == "image"]
    if image_vectors:  # 只有当有图像向量时才添加到索引
        image_index_map.add_with_ids(np.array(image_vectors).astype('float32'), np.array(image_ids))
    
    print(f"索引构建完成。共索引 {len(text_vectors)} 个文本片段和 {len(image_vectors)} 张图片。")
    
    return metadata_store, text_index_map, image_index_map

# Step3. RAG 问答流程
def rag_ask(query, metadata_store, text_index, image_index, k=3):
    """
    执行完整的 RAG 流程：检索 -> 构建Prompt -> 生成答案
    """
    print(f"\n--- 收到用户提问: '{query}' ---")
    
    # 步骤 1: 检索
    print("  - 步骤 1: 向量化查询并进行检索...")
    retrieved_context = []
    
    # 文本检索
    query_text_vec = np.array([get_text_embedding(query)]).astype('float32')
    distances, text_ids = text_index.search(query_text_vec, k)
    for i, doc_id in enumerate(text_ids[0]):
        if doc_id != -1:
            # 通过ID在元数据中查找
            match = next((item for item in metadata_store if item["id"] == doc_id), None)
            if match:
                retrieved_context.append(match)
                print(f"    - 文本检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")

    # 图像检索 (使用CLIP文本编码器)
    # 简单判断是否需要检索图片
    if any(keyword in query.lower() for keyword in ["海报", "图片", "长什么样", "看看", "万圣节", "聚在一起"]):
        print("  - 检测到图像查询关键词，执行图像检索...")
        query_clip_vec = np.array([get_clip_text_embedding(query)]).astype('float32')
        distances, image_ids = image_index.search(query_clip_vec, 1) # 只找最相关的1张图
        for i, doc_id in enumerate(image_ids[0]):
            if doc_id != -1:
                match = next((item for item in metadata_store if item["id"] == doc_id), None)
                if match:
                    # 将OCR内容也加入上下文
                    context_text = f"找到一张相关图片，图片路径: {match['path']}。图片上的文字是: '{match['ocr']}'"
                    retrieved_context.append({"type": "image_context", "content": context_text, "metadata": match})
                    print(f"    - 图像检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")
    
    # 步骤 2: 构建 Prompt 并生成答案
    print("  - 步骤 2: 构建 Prompt...")
    context_str = ""
    for i, item in enumerate(retrieved_context):
        content = item.get('content', '')
        source = item.get('metadata', {}).get('source', item.get('source', '未知来源'))
        context_str += f"背景知识 {i+1} (来源: {source}):\n{content}\n\n"
        
    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识，用友好和专业的语气回答用户的问题。请只使用背景知识中的信息，不要自行发挥。

[背景知识]
{context_str}
[用户问题]
{query}
"""
    print("--- Prompt Start ---")
    print(prompt)
    print("--- Prompt End ---")
    
    print("\n  - 步骤 3: 调用 LLM 生成最终答案...")
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 使用一个强大的模型进行生成
            messages=[
                {"role": "system", "content": "你是一个迪士尼客服助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        final_answer = completion.choices[0].message.content
        
        # 答案后处理：如果上下文中包含图片，提示用户
        image_path_found = None
        for item in retrieved_context:
            if item.get("type") == "image_context":
                image_path_found = item.get("metadata", {}).get("path")
                break
        
        if image_path_found:
            final_answer += f"\n\n(同时，我为您找到了相关图片，路径为: {image_path_found})"

    except Exception as e:
        final_answer = f"调用LLM时出错: {e}"

    print("\n--- 最终答案 ---")
    print(final_answer)
    return final_answer

# --- 主函数 ---
if __name__ == "__main__":
    # 1. 构建知识库 (一次性离线过程)
    metadata_store, text_index, image_index = build_knowledge_base(DOCS_DIR, IMG_DIR)
    
    # 2. 开始问答
    print("\n=============================================")
    print("迪士尼客服RAG助手已准备就绪，开始模拟提问。")
    print("=============================================")
    
    # 案例1: 文本问答
    rag_ask(
        query="我想了解一下迪士尼门票的退款流程",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    
    # 案例2: 多模态问答
    rag_ask(
        query="最近万圣节的活动海报是什么",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )
    
    print("\n---------------------------------------------\n")
    
    # 案例3: 年卡相关问答
    rag_ask(
        query="迪士尼年卡有什么优惠",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )