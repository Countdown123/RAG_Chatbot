# graph.py

import pdfplumber
from langchain.docstore.document import Document
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langsmith.wrappers import wrap_openai
import openai
import re
from dotenv import load_dotenv
import re
from typing import TypedDict, List, Set, Literal, Dict, Any
import os
from pinecone import Pinecone, ServerlessSpec
from langsmith import traceable
from langgraph.graph import StateGraph, END
import logging
import json

import unicodedata
from difflib import SequenceMatcher

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOKENS = 110000

def find_most_similar_speaker(query, speaker_list, threshold=0.6):
    """
    주어진 쿼리와 가장 유사한 발언자를 찾습니다.

    Args:
        query (str): 검색할 발언자 이름.
        speaker_list (List[str]): 발언자 이름 목록.
        threshold (float): 유사도 임계값.

    Returns:
        str or None: 가장 유사한 발언자 이름 또는 None.
    """
    query = unicodedata.normalize("NFC", query)
    best_match = None
    highest_ratio = 0

    for speaker in speaker_list:
        normalized_speaker = unicodedata.normalize("NFC", speaker)
        ratio = SequenceMatcher(None, query, normalized_speaker).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = speaker

    if highest_ratio >= threshold:
        return best_match
    return None

def find_most_similar_item(query, item_list, threshold=0.6):
    """
    주어진 쿼리와 가장 유사한 아이템을 찾습니다.

    Args:
        query (str): 검색할 아이템 이름.
        item_list (List[str]): 아이템 이름 목록.
        threshold (float): 유사도 임계값.

    Returns:
        str or None: 가장 유사한 아이템 이름 또는 None.
    """
    best_match = None
    highest_ratio = 0

    for item in item_list:
        ratio = SequenceMatcher(None, query.lower(), item.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = item

    return best_match if highest_ratio >= threshold else None

class GraphState(TypedDict):
    question: str
    answer: str
    verification_result: bool
    page_numbers: List[int]
    db: str
    file_paths: List[str]
    pdfs_loaded: bool
    verification_count: int
    excluded_pages: Set[int]
    next_question: bool
    file_types: List[Literal["pdf"]]
    next_node: Literal[
        "pdf_processing_with_speaker",
        "pdf_processing_without_speaker",
        "query_processing_with_speaker",
        "query_processing_with_filter",
        "query_processing_without_filter",
    ]
    processed_data: str
    max_pages: int
    metadata: Dict[str, Any]
    completed: bool
    search_filters: List[Dict[str, Any]]

def data_input_node(state: GraphState) -> GraphState:
    """
    데이터 입력 노드입니다. 파일 경로를 확인하고 다음 노드를 결정합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    file_paths = state["file_paths"]

    logging.info(f"Received file paths: {file_paths}")

    if not file_paths:
        logging.error("서버에서 제공된 파일 경로가 없습니다.")
        raise ValueError("서버에서 제공된 파일 경로가 없습니다.")

    valid_paths = []
    has_speaker = False

    for file_path in file_paths:
        logging.info(f"Processing file: {file_path}")
        if not os.path.exists(file_path):
            logging.warning(f"유효하지 않은 파일 경로입니다: {file_path}")
            continue

        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == ".pdf":
            logging.info(f"Valid PDF file found: {file_path}")
            valid_paths.append(file_path)

            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and re.search(
                            r"^◯\s*(\S+\s\S+)(\s|$)", text, re.MULTILINE
                        ):
                            has_speaker = True
                            logging.info(
                                f"Speaker found in file {file_path} on page {page_num + 1}"
                            )
                            break  # 현재 파일에서 발언자를 찾았으므로 페이지 검색 중단
            except Exception as e:
                logging.error(f"PDF 파일 '{file_path}' 처리 중 오류 발생: {str(e)}")
        else:
            logging.warning(f"지원되지 않는 파일 형식입니다: {file_path}")

    if not valid_paths:
        logging.error("처리할 수 있는 유효한 PDF 파일이 없습니다.")
        raise ValueError("처리할 수 있는 유효한 PDF 파일이 없습니다.")

    logging.info(
        f"{len(valid_paths)}개의 유효한 PDF 파일이 처리를 위해 준비되었습니다."
    )
    for path in valid_paths:
        logging.info(f"- {path}")

    next_node = (
        "pdf_processing_with_speaker"
        if has_speaker
        else "pdf_processing_without_speaker"
    )
    logging.info(f"Next node: {next_node}")

    return GraphState(
        file_paths=valid_paths,
        file_types=["pdf"] * len(valid_paths),
        pdfs_loaded=False,
        next_node=next_node,
    )

def pdf_processing_with_speaker(state: GraphState) -> GraphState:
    """
    발언자가 있는 PDF를 처리합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    print("발언자가 있는 PDF입니다")
    all_documents = []
    metadata = {}

    counter = 0

    for pdf_path in state["file_paths"]:
        documents = []
        file_name = os.path.basename(pdf_path)
        file_metadata = {"max_pages": 0, "speakers": set()}

        with pdfplumber.open(pdf_path) as pdf:
            file_metadata["max_pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    content = ""
                    current_speaker = None
                    for line in lines:
                        speaker_match = re.match(r"^◯\s*(\S+\s\S+)(\s|$)", line)
                        if speaker_match:
                            if content:
                                documents.append(
                                    Document(
                                        page_content=content.strip(),
                                        metadata={
                                            "file_name": file_name,
                                            "page_number": i + 1,
                                            "speaker": current_speaker or "unknown",
                                            "type": "speaker_content",
                                            "order": counter,
                                        },
                                    )
                                )
                                content = ""
                                counter += 1

                            current_speaker = speaker_match.group(1).strip()
                            file_metadata["speakers"].add(current_speaker)
                            content = line[speaker_match.end() :].strip() + "\n"
                        else:
                            content += line + "\n"

                    if content:
                        documents.append(
                            Document(
                                page_content=content.strip(),
                                metadata={
                                    "file_name": file_name,
                                    "page_number": i + 1,
                                    "speaker": current_speaker or "unknown",
                                    "type": "speaker_content",
                                    "order": counter,
                                },
                            )
                        )
                        counter += 1

        all_documents.extend(documents)
        file_metadata["speakers"] = list(file_metadata["speakers"])
        metadata[file_name] = file_metadata

    return GraphState(processed_data=all_documents, pdfs_loaded=True, metadata=metadata)

def pdf_processing_without_speaker(state: GraphState) -> GraphState:
    """
    발언자가 없는 PDF를 처리합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    print("발언자가 없는 PDF입니다")

    all_documents = []
    metadata = {}

    counter = 0

    for pdf_path in state["file_paths"]:
        documents = []
        file_name = os.path.basename(pdf_path)
        file_metadata = {"max_pages": 0, "speakers": []}

        with pdfplumber.open(pdf_path) as pdf:
            file_metadata["max_pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "file_name": file_name,
                                "page_number": i + 1,
                                "type": "page_content",
                                "order": counter,
                            },
                        )
                    )
                    counter += 1

        all_documents.extend(documents)
        metadata[file_name] = file_metadata

    return GraphState(processed_data=all_documents, pdfs_loaded=True, metadata=metadata)

def vector_storage_node(state: GraphState) -> GraphState:
    """
    벡터 저장 노드입니다. 처리된 데이터를 벡터스토어에 저장합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    load_dotenv()
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    file_paths = state["file_paths"]
    processed_data = state["processed_data"]

    combined_filename = "-".join(
        [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
    )
    index_name = re.sub(r"[^a-zA-Z0-9-]", "-", combined_filename.lower())
    index_name = f"doc-{index_name[:40]}"
    index_name = re.sub(r"-+", "-", index_name).strip("-")

    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embeddings = OpenAIEmbeddings()

    documents = []
    for doc in processed_data:
        if isinstance(doc, Document):
            metadata = {
                "file_name": doc.metadata.get("file_name", "unknown"),
                "page_number": doc.metadata.get("page_number", "unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "order": doc.metadata.get("order", "unknown"),
            }
            if "speaker" in doc.metadata:
                metadata["speaker"] = doc.metadata["speaker"]

            documents.append(Document(page_content=doc.page_content, metadata=metadata))
        else:
            logger.warning(f"Unexpected document type: {type(doc)}")

    try:
        vectorstore = LangchainPinecone.from_documents(
            documents=documents, embedding=embeddings, index_name=index_name
        )

        return GraphState(
            db=index_name,
            file_types=["pdf"],
            max_pages=state.get("max_pages", {}),
            metadata=state.get("metadata", {}),
            completed=True,  # 처리 완료 플래그 설정
        )
    except StopIteration:
        logger.info("Vector storage process completed successfully.")
        return GraphState(**state, completed=True)
    except Exception as e:
        logger.error(f"Error in vector storage: {str(e)}")
        return GraphState(error=str(e), completed=True)

def chat_interface_node(state: GraphState) -> GraphState:
    """
    채팅 인터페이스 노드입니다. 질문을 분석하고 다음 노드를 결정합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    query = state["question"]
    new_metadata = state.get("metadata", {})

    # 새 메타데이터 구조를 이전 구조로 변환
    old_metadata = {
        "file_names": list(new_metadata.keys()),
        "speakers": list(
            set(
                speaker
                for file_info in new_metadata.values()
                for speaker in file_info["speakers"]
            )
        ),
        "max_pages": max(file_info["max_pages"] for file_info in new_metadata.values()),
    }

    # 질문 분석 및 필터 생성
    analysis_prompt = f"""
    다음 질문을 분석하여 검색에 필요한 정보를 추출하세요:
    질문: {query}

    1. 이 질문이 특정 파일에 관한 것인가요? 만약 그렇다면 해당 파일명을 추출하세요.
    2. 이 질문이 특정 페이지에 관한 것인가요? 만약 그렇다면 해당 페이지 번호를 추출하세요.
    3. 이 질문이 특정 발언자에 관한 것인가요? 만약 그렇다면 해당 발언자의 이름을 정확히 추출하세요.
    4. 아래 형식을 지켜서 반환해주세요

    분석 결과:
    파일명: [파일명 또는 '없음']
    페이지 번호: [번호 또는 '없음']
    발언자: [이름 또는 '없음']
    """

    client = wrap_openai(openai.Client())
    analysis_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    analysis_result = analysis_response.choices[0].message.content

    specific_files = []
    specific_pages = []
    specific_speakers = []
    speaker_list = old_metadata.get(
        "speakers", []
    )  # 여기서 speaker_list를 초기화합니다.

    for line in analysis_result.split("\n"):
        print(f"분석된 라인: {line}")
        if line.startswith("파일명:"):
            files = line.split(":")[1].strip()
            if files != "없음":
                file_list = old_metadata.get("file_names", [])
                most_similar_file = find_most_similar_item(files, file_list)
                if most_similar_file:
                    specific_files = [most_similar_file]
            print(f"메타데이터에서 매칭된 파일명: {specific_files}")

        elif line.startswith("페이지 번호:"):
            page = line.split(":")[1].strip()
            if page != "없음":
                specific_pages = [int(p) for p in re.findall(r"\d+", page)]
            print(f"추출된 페이지 번호: {specific_pages}")

        elif line.startswith("발언자:"):
            speakers = line.split(":")[1].strip()
            if speakers != "없음":
                speaker_list = old_metadata.get("speakers", [])
                most_similar_speaker = find_most_similar_speaker(speakers, speaker_list)
                if most_similar_speaker:
                    specific_speakers = [most_similar_speaker]
                logger.info(f"Query: {speakers}")
                logger.info(f"메타데이터에서 매칭된 발언자: {specific_speakers}")
    all_files = old_metadata.get("file_names", [])

    search_filters = []
    if specific_files:
        for file_name in specific_files:
            filter_dict = {"file_name": file_name}
            if specific_pages:
                filter_dict["page_number"] = {"$in": specific_pages}
            if specific_speakers:
                filter_dict["speaker"] = {"$in": specific_speakers}
            search_filters.append(filter_dict)
    elif specific_pages or specific_speakers:
        for file_name in all_files:  # 모든 파일에 대해 필터 생성
            filter_dict = {"file_name": file_name}
            if specific_pages:
                filter_dict["page_number"] = {"$in": specific_pages}
            if specific_speakers:
                filter_dict["speaker"] = {"$in": specific_speakers}
            search_filters.append(filter_dict)

    # 다음 노드 결정
    if specific_speakers:
        next_node = "query_processing_with_speaker"
        print(
            f"발언자가 검출되었습니다: {specific_speakers}. 다음 노드: query_processing_with_speaker"
        )
    elif specific_pages:
        next_node = "query_processing_with_filter"
        print(
            f"특정 페이지 번호가 검출되었습니다: {specific_pages}. 다음 노드: query_processing_with_filter"
        )
    elif specific_files:
        next_node = "query_processing_with_filter"
        print(
            f"특정 파일명이 검출되었습니다: {specific_files}. 다음 노드: query_processing_with_filter"
        )
    else:
        next_node = "query_processing_without_filter"
        print(
            "특정 필터가 검출되지 않았습니다. 다음 노드: query_processing_without_filter"
        )

    print(f"생성된 검색 필터: {search_filters}")

    return GraphState(
        question=query, search_filters=search_filters, next_node=next_node
    )

@traceable()
def query_processing_with_speaker(state: GraphState) -> GraphState:
    """
    발언자 필터가 적용된 질문을 처리합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    query = state['question']
    index_name = state['db']
    search_filters = state['search_filters']

    client = wrap_openai(openai.Client())

    try:
        print("Pinecone 인덱스 로드 시작")
        embeddings = OpenAIEmbeddings()
        vectorstore = LangchainPinecone.from_existing_index(index_name, embeddings)
        print("Pinecone 인덱스 로드 성공")
    except Exception as e:
        print(f"Pinecone 인덱스 로드 오류 발생: {str(e)}")
        return GraphState(
            answer="인덱스 로드 중 오류가 발생했습니다.",
            page_numbers={},
            next_question=True,
            completed=True
        )

    docs = []
    print("=== 검색 실행 시작 ===")
    try:
        for search_filter in search_filters:
            docs.extend(vectorstore.similarity_search(query, k=1000, filter=search_filter))
    except Exception as e:
        print(f"검색 실행 오류 발생: {str(e)}")
        return GraphState(
            answer="검색 실행 중 오류가 발생했습니다.",
            page_numbers={},
            next_question=True,
            completed=True
        )
    seen_contents = {}
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents[doc.page_content] = doc.metadata

    if not unique_docs:
        return GraphState(
            answer="관련 정보를 찾을 수 없습니다.",
            page_numbers={},
            next_question=True
        )

    unique_docs.sort(key=lambda x: x.metadata.get('order', 0))

    context = ""
    page_numbers = {}
    current_tokens = 0

    for doc in unique_docs:
        file_name = doc.metadata.get('file_name', 'Unknown')
        page_number = doc.metadata.get('page_number', 'N/A')
        speaker = doc.metadata.get('speaker', 'Unknown')
        content = f"File: {file_name}, Page {page_number}, Speaker {speaker}: {doc.page_content}\n\n"

        tokens_in_content = len(content)
        if current_tokens + tokens_in_content > MAX_TOKENS:
            break

        context += content
        current_tokens += tokens_in_content

        if file_name not in page_numbers:
            page_numbers[file_name] = []
        page_numbers[file_name].append(page_number)

    functions = [
    {
            "name": "structured_answer",
            "description": "질문에 대한 구조화된 답변을 제공합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "질문에 대한 답변"
                    },
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "확인된 페이지 번호들의 배열"
                    }
                },
                "required": ["answer", "pages"]
            }
        }
    ]

    answer_prompt = f"""
    다음은 PDF 파일에서 추출한 관련 정보입니다:

    {context}

    이 정보를 바탕으로 다음 질문에 답변해 주세요:
    {query}

    답변 시 아래의 사항을 준수해 주세요:
    1. 문서의 내용을 기반으로 답변하세요.
    2. 추측이나 외부 지식을 사용하지 말고, 제공된 정보만을 참고해 답변하세요.
    3. 질문에 대한 정보가 제공된 데이터에 없다면, 그 사실을 명확히 언급하세요.
    4. 여러 파일에서 정보가 나왔다면, 각 파일의 정보를 구분하여 답변하세요.
    """

    answer_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": answer_prompt}],
        functions=functions,
        function_call={"name": "structured_answer"}
    )

    function_response = json.loads(answer_response.choices[0].message.function_call.arguments)

    final_answer = {
        "answer": function_response["answer"],
        "page_numbers": function_response["pages"]
    }

    return GraphState(
        answer=final_answer,
        page_numbers=page_numbers,
        next_question=True,
        completed=True
    )
    
@traceable()
def query_processing_with_filter(state: GraphState) -> GraphState:
    query = state['question']
    index_name = state['db']
    search_filters = state['search_filters']

    client = wrap_openai(openai.Client())

    try:
        print("Pinecone 인덱스 로드 시작")
        embeddings = OpenAIEmbeddings()
        vectorstore = LangchainPinecone.from_existing_index(index_name, embeddings)
        print("Pinecone 인덱스 로드 성공")
    except Exception as e:
        print(f"Pinecone 인덱스 로드 오류 발생: {str(e)}")
        return GraphState(
            answer="인덱스 로드 중 오류가 발생했습니다.",
            page_numbers={},
            next_question=True,
            completed=True
        )

    docs = []
    print("=== 검색 실행 시작 ===")
    try:
        for search_filter in search_filters:
            docs.extend(vectorstore.similarity_search(query, k=1000, filter=search_filter))
    except Exception as e:
        print(f"검색 실행 오류 발생: {str(e)}")
        return GraphState(
            answer="검색 실행 중 오류가 발생했습니다.",
            page_numbers={},
            next_question=True,
            completed=True
        )
    seen_contents = {}
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents[doc.page_content] = doc.metadata

    if not unique_docs:
        return GraphState(
            answer="관련 정보를 찾을 수 없습니다.",
            page_numbers={},
            next_question=True
        )

    unique_docs.sort(key=lambda x: x.metadata.get('order', 0))

    context = ""
    page_numbers = {}
    current_tokens = 0

    for doc in unique_docs:
        file_name = doc.metadata.get('file_name', 'Unknown')
        page_number = doc.metadata.get('page_number', 'N/A')
        speaker = doc.metadata.get('speaker', 'Unknown')
        content = f"File: {file_name}, Page {page_number}, Speaker {speaker}: {doc.page_content}\n\n"

        tokens_in_content = len(content)
        if current_tokens + tokens_in_content > MAX_TOKENS:
            break

        context += content
        current_tokens += tokens_in_content

        if file_name not in page_numbers:
            page_numbers[file_name] = []
        page_numbers[file_name].append(page_number)

    functions = [
        {
            "name": "structured_answer",
            "description": "질문에 대한 구조화된 답변을 제공합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "질문에 대한 답변"
                    },
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "확인된 페이지 번호들의 배열"
                    }
                },
                "required": ["answer", "pages"]
            }
        }
    ]

    answer_prompt = f"""
    다음은 PDF 파일에서 추출한 관련 정보입니다:

    {context}

    이 정보를 바탕으로 다음 질문에 답변해 주세요:
    {query}

    답변 시 아래의 사항을 준수해 주세요:
    1. 질문에 직접적으로 관련된 정보만을 포함하여 간결하게 답변하세요.
    2. 문서의 내용만을 기반으로 답변하고, 추측이나 외부 지식을 사용하지 마세요.
    3. 관련된 파일명과 페이지 번호를 반드시 언급하세요. 특히 질문이 특정 파일이나 페이지에 관한 것이라면, 해당 정보를 답변의 시작 부분에 명시하세요.
    4. 여러 페이지나 파일에 걸쳐 정보가 분산되어 있다면, 각 페이지나 파일의 정보를 명확히 구분하여 설명하세요.
    5. 질문에 대한 정보가 제공된 데이터에 없다면, 그 사실을 간단히 언급하세요.
    6. 답변은 명확하고 직설적으로 작성하여, 사용자가 추가 설명 없이도 이해할 수 있도록 하세요.
    7. 불필요한 세부 사항이나 부가 설명은 생략하고, 질문의 핵심에 맞는 답변만 제공하세요.
    """
    answer_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": answer_prompt}],
        functions=functions,
        function_call={"name": "structured_answer"}
    )

    function_response = json.loads(answer_response.choices[0].message.function_call.arguments)

    final_answer = {
        "answer": function_response["answer"],
        "page_numbers": function_response["pages"]
    }

    return GraphState(
        answer=final_answer,
        page_numbers=page_numbers,
        next_question=True,
        completed=True
    )

@traceable()
def query_processing_without_filter(state: GraphState) -> GraphState:
    """
    필터 없이 질문을 처리합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        GraphState: 업데이트된 상태.
    """
    query = state['question']
    index_name = state['db']

    client = wrap_openai(openai.Client())
    

    try:
        print("Pinecone 인덱스 로드 시작")
        embeddings = OpenAIEmbeddings()
        vectorstore = LangchainPinecone.from_existing_index(index_name, embeddings)
        print("Pinecone 인덱스 로드 성공")
    except Exception as e:
        print(f"Pinecone 인덱스 로드 오류 발생: {str(e)}")
        return GraphState(
            answer="인덱스 로드 중 오류가 발생했습니다.",
            page_numbers={},
            next_question=True,
            completed=True
        )

    docs = []
    print("=== 검색 실행 시작 ===")
    try:
        docs = vectorstore.similarity_search(query, k=10)
    except Exception as e:
        print(f"검색 실행 오류 발생: {str(e)}")
        return GraphState(
            answer="검색 실행 중 오류가 발생했습니다.",
            page_numbers={},
            next_question=True,
            completed=True
        )
    seen_contents = {}
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents[doc.page_content] = doc.metadata

    if not unique_docs:
        return GraphState(
            answer="관련 정보를 찾을 수 없습니다.",
            page_numbers={},
            next_question=True
        )

    unique_docs.sort(key=lambda x: x.metadata.get('order', 0))

    context = ""
    page_numbers = {}
    current_tokens = 0

    for doc in unique_docs:
        file_name = doc.metadata.get('file_name', 'Unknown')
        page_number = doc.metadata.get('page_number', 'N/A')
        speaker = doc.metadata.get('speaker', 'Unknown')
        content = f"File: {file_name}, Page {page_number}, Speaker {speaker}: {doc.page_content}\n\n"

        tokens_in_content = len(content)
        if current_tokens + tokens_in_content > MAX_TOKENS:
            break

        context += content
        current_tokens += tokens_in_content

        if file_name not in page_numbers:
            page_numbers[file_name] = []
        page_numbers[file_name].append(page_number)

    functions = [
        {
            "name": "structured_answer",
            "description": "질문에 대한 구조화된 답변을 제공합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "질문에 대한 답변"
                    },
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "확인된 페이지 번호들의 배열"
                    }
                },
                "required": ["answer", "pages"]
            }
        }
    ]

    answer_prompt = f"""
    다음은 PDF 파일들에서 추출한 관련 정보입니다:

    {context}

    이 정보를 바탕으로 다음 질문에 종합적으로 답변해 주세요:
    {query}

    답변 시 아래의 사항을 준수해 주세요:
    1. 질문에 직접적으로 관련된 정보만을 사용하여 간결하고 명확하게 답변하세요.
    2. 제공된 정보만을 바탕으로 답변하고, 외부 지식이나 추측을 사용하지 마세요.
    3. 여러 문서나 페이지의 정보를 사용할 경우, 정보를 논리적으로 연결하여 일관성 있게 답변하세요.
    4. 제공된 정보들 사이에 상충되는 내용이 있다면 간단히 언급하세요.
    5. 질문에 대한 직접적인 답변이 제공된 데이터에 없다면, 그 사실을 명시하고 가장 관련성 높은 정보를 간단히 제시하세요.
    6. 불필요한 세부 사항이나 부가 설명은 생략하고, 질문의 핵심에 맞는 답변만 제공하세요.
    """
    answer_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": answer_prompt}],
        functions=functions,
        function_call={"name": "structured_answer"}
    )

    function_response = json.loads(answer_response.choices[0].message.function_call.arguments)

    final_answer = {
        "answer": function_response["answer"],
        "page_numbers": function_response["pages"]
    }

    return GraphState(
        answer=final_answer,
        page_numbers=page_numbers,
        next_question=True,
        completed=True
    )

def route_by_file_type(state: GraphState) -> str:
    file_types = state["file_types"]

    if not file_types:
        raise ValueError("파일 유형이 지정되지 않았습니다.")

    if "pdf" in file_types:
        return "pdf_processing"
    else:
        raise ValueError("지원되지 않는 파일 형식입니다.")


def route_by_query_type(state: GraphState) -> str:
    return state["next_node"]

def route_by_file_type(state: GraphState) -> str:
    """
    파일 유형에 따라 다음 노드를 결정합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        str: 다음 노드 이름.
    """
    return state["next_node"]

def should_continue(state: GraphState) -> bool:
    """
    처리를 계속할지 여부를 결정합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        bool: 계속할지 여부.
    """
    return state["question"] != "" and not state.get("completed", False)

def should_end(state: GraphState) -> bool:
    """
    처리를 종료할지 여부를 결정합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        bool: 종료할지 여부.
    """
    return (
        state.get("next_question", False)
        or state["question"] == ""
        or state.get("completed", False)
    )

def create_file_processing_workflow():
    """
    파일 처리 워크플로우를 생성합니다.

    Returns:
        Compiled workflow graph.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("data_input", data_input_node)
    workflow.add_node("pdf_processing_with_speaker", pdf_processing_with_speaker)
    workflow.add_node("pdf_processing_without_speaker", pdf_processing_without_speaker)
    workflow.add_node("vector_storage", vector_storage_node)

    workflow.add_conditional_edges(
        "data_input",
        route_by_file_type,
        {
            "pdf_processing_with_speaker": "pdf_processing_with_speaker",
            "pdf_processing_without_speaker": "pdf_processing_without_speaker",
        },
    )
    workflow.add_edge("pdf_processing_with_speaker", "vector_storage")
    workflow.add_edge("pdf_processing_without_speaker", "vector_storage")

    def check_vector_storage_completion(state):
        return state.get("completed", False) or state.get("error") is not None

    workflow.add_conditional_edges(
        "vector_storage",
        check_vector_storage_completion,
        {True: END, False: "vector_storage"},
    )

    workflow.set_entry_point("data_input")
    logger.debug("File processing workflow created")
    return workflow.compile()

def create_qa_workflow():
    """
    QA 워크플로우를 생성합니다.

    Returns:
        Compiled workflow graph.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("chat_interface", chat_interface_node)
    workflow.add_node("query_processing_with_speaker", query_processing_with_speaker)
    workflow.add_node("query_processing_with_filter", query_processing_with_filter)
    workflow.add_node(
        "query_processing_without_filter", query_processing_without_filter
    )

    workflow.add_conditional_edges(
        "chat_interface",
        route_by_query_type,
        {
            "query_processing_with_speaker": "query_processing_with_speaker",
            "query_processing_with_filter": "query_processing_with_filter",
            "query_processing_without_filter": "query_processing_without_filter",
        },
    )

    workflow.add_conditional_edges(
        "query_processing_with_speaker",
        should_end,
        {True: END, False: "chat_interface"},
    )
    workflow.add_conditional_edges(
        "query_processing_with_filter", should_end, {True: END, False: "chat_interface"}
    )
    workflow.add_conditional_edges(
        "query_processing_without_filter",
        should_end,
        {True: END, False: "chat_interface"},
    )

    workflow.set_entry_point("chat_interface")
    return workflow.compile()

def process_files(file_paths: List[str]) -> str:
    """
    파일들을 처리합니다.

    Args:
        file_paths (List[str]): 파일 경로 목록.

    Returns:
        str: 데이터베이스 인덱스 이름.

    Raises:
        ValueError: 처리 중 오류 발생 시.
    """
    workflow = create_file_processing_workflow()
    state = GraphState(
        file_paths=file_paths,
        processed_data=[],
        db="",
        completed=False,
        file_types=["pdf"],
    )

    for output in workflow.stream(state):
        if output.get("vector_storage", {}).get("completed", False):
            return output["vector_storage"]["db"]

    raise ValueError("파일 처리 중 오류가 발생했습니다.")

def process_query(state: GraphState) -> Dict:
    """
    사용자의 질문을 처리합니다.

    Args:
        state (GraphState): 현재 상태.

    Returns:
        Dict: 답변과 관련 정보.
    """
    workflow = create_qa_workflow()

    for output in workflow.stream(state):
        if not output:
            break
        current_node = next(iter(output))
        new_state = output[current_node]
        state.update(new_state)

        if current_node in [
            "query_processing_with_speaker",
            "query_processing_with_filter",
            "query_processing_without_filter",
        ]:
            if state.get("next_question", False):
                return state["answer"]

    return state.get("answer", {})
