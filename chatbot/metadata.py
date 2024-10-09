# 필요한 라이브러리 임포트
import pandas as pd
import sqlite3
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.prompts import (
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from deep_translator import GoogleTranslator
from collections import Counter
import re


class SQLChatbot:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = self._get_file_name()
        self.model = self._determine_model()
        self.df = self._load_data()
        self.exception_words = self._extract_exception_words()
        self.conn = self._create_database()
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.file_name}.db")
        self.llm = ChatOpenAI(model_name=self.model, temperature=0)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.prompt = self._create_prompt()
        self.agent_executor = self._initialize_agent_executor()

    def _get_file_name(self):
        # 파일 이름만 가져오기
        return self.file_path.split("\\")[-1].rsplit(".", 1)[0]

    def _determine_model(self):
        # 파일 형식에 따른 모델 지정
        if self.file_path.endswith(".csv"):
            return "gpt-3.5-turbo"
        elif self.file_path.endswith(".xls") or self.file_path.endswith(".xlsx"):
            return "gpt-4-turbo"
        else:
            raise ValueError(
                "지원되지 않는 파일 형식입니다. CSV, XLS, XLSX 파일만 지원됩니다."
            )

    def _load_data(self):
        # 파일 형식 확인 및 데이터 로드
        if self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".xls") or self.file_path.endswith(".xlsx"):
            df = pd.read_excel(self.file_path, header=0)
        else:
            raise ValueError(
                "지원되지 않는 파일 형식입니다. CSV, XLS, XLSX 파일만 지원됩니다."
            )
        return df

    def _create_database(self):
        # SQLite 데이터베이스 생성 및 데이터 삽입
        conn = sqlite3.connect(f"{self.file_name}.db")
        self.df.to_sql(self.file_name, conn, index=False, if_exists="replace")
        return conn

    def _create_prompt(self):
        # 커스텀 프롬프트 정의 및 컬럼 이름 포함
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.file_name})")
        columns = [info[1] for info in cursor.fetchall()]
        columns_str = ", ".join(columns)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    f"""
당신은 데이터베이스에 대한 전문가입니다. 사용자의 질문에 정확하고 상세한 답변을 제공해주세요. SQL 쿼리를 사용하여 데이터베이스에서 필요한 정보를 가져올 수 있습니다.
가능하면 이전 대화 내용을 참고하여 사용자의 의도를 파악하고, 지시사항을 따르기 위해 필요한 도구를 사용해주세요.
"""
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        return prompt

    def _initialize_agent_executor(self):
        # 에이전트 실행기 초기화
        agent_executor = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            llm=self.llm,
            tools=self.toolkit.get_tools(),
            verbose=True,
            memory=self.memory,
            agent_kwargs={
                "prompt": self.prompt,
            },
        )
        return agent_executor

    def _extract_exception_words(self):
        # 모든 컬럼에서 예외 단어 추출
        exception_words_set = set()

        # 각 컬럼에서 고유 값 추출하여 집합에 추가
        for column in self.df.columns:
            # NaN 값 제거 후 문자열로 변환하여 고유 값 추출
            unique_values = self.df[column].dropna().astype(str).unique()
            exception_words_set.update(unique_values)

        # 집합을 리스트로 변환하여 반환
        exception_words = list(exception_words_set)
        return exception_words

    def process_response(self, response):
        exception_words = self.exception_words

        # 예외 단어가 없을 경우 일반 번역 수행
        if not exception_words:
            translated_response = GoogleTranslator(
                source="auto", target="ko"
            ).translate(response)
            return translated_response

        # 예외 단어를 플레이스홀더로 대체
        placeholder_mapping = {}
        for idx, word in enumerate(exception_words):
            placeholder = f"__EXCEPTION_{idx}__"
            pattern = r"\b{}\b".format(re.escape(word))
            response = re.sub(pattern, placeholder, response, flags=re.IGNORECASE)
            placeholder_mapping[placeholder] = word

        # 번역 수행
        translated_response = GoogleTranslator(source="auto", target="ko").translate(
            response
        )

        # 플레이스홀더를 원래 단어로 복원
        for placeholder, word in placeholder_mapping.items():
            translated_response = translated_response.replace(placeholder, word)

        return translated_response

    def ask_question(self, question, num_trials=3):
        responses = []
        total_tokens_used = 0
        for i in range(num_trials):
            with get_openai_callback() as cb:
                response = self.agent_executor.run(input=question)
                # 응답 후처리
                response = self.process_response(response)
                responses.append(response)
                total_tokens_used += cb.total_tokens
        # 응답 빈도 계산
        response_count = Counter(responses)
        most_common_response = response_count.most_common(1)[0][0]
        print(f"가장 빈번한 응답: {most_common_response}")
        # 모든 응답 출력
        print(f"User: {question}")
        for idx, resp in enumerate(responses):
            print(f"Assistant (Trial {idx+1}): {resp}")
        # 응답 일관성 확인
        if all(resp == responses[0] for resp in responses):
            print("모든 응답이 동일합니다.")
        else:
            print("응답이 서로 다릅니다.")
        print(f"Total Tokens Used: {total_tokens_used}\n")
        return most_common_response


# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    file_path = "/content/sports.xlsx"  # 실제 파일 경로로 변경하세요

    # SQLChatbot 인스턴스 생성
    chatbot = SQLChatbot(file_path)

    # # csv 예시 상호 작용
    # chatbot.ask_question("가장 높은 공격력을 가진 포켓몬의 이름은 무엇인가요?")
    # chatbot.ask_question("그 포켓몬의 타입은 무엇인가요?")
    # chatbot.ask_question("방금 말한 포켓몬의 방어력은 얼마인가요?")
    # chatbot.ask_question("그래서 지금 이야기하는 포켓몬 이름이 뭐라고?")

    # # xlsx 예시 상호 작용
    # chatbot.ask_question("무슨 컬럼이 있어?")
    # chatbot.ask_question("가장 좋은 답변 결과를 가진 스포츠는?")