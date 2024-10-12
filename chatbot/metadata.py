import os
import re
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


class SQLChatbot:

    def __init__(self, file_path, sanitized_filename=None):
        self.file_path = file_path
        self.file_name = sanitized_filename or self._get_file_name()
        self.table_name = self._get_table_name()
        self.model = self._determine_model()
        self.df = self._load_data()
        self.conn = self._create_database()
        self.db = self._create_sql_database()
        self.llm = ChatOpenAI(model_name=self.model, temperature=0)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.prompt = self._create_prompt()
        self.agent_executor = self._initialize_agent_executor()
        self.exception_words = self._extract_exception_words()

    def _get_file_name(self):
        base_name = os.path.basename(self.file_path)
        return "".join(e for e in base_name if e.isalnum() or e in [".", "_"])

    def _get_table_name(self):
        table_name = re.sub(r"\W+", "_", self.file_name)
        table_name = re.sub(r"^_+|_+$", "", table_name)
        if table_name[0].isdigit():
            table_name = "T_" + table_name
        return table_name

    def _determine_model(self):
        if self.file_path.endswith(".csv"):
            return "gpt-4o"
        elif self.file_path.endswith(".xls") or self.file_path.endswith(".xlsx"):
            return "gpt-4o"
        else:
            raise ValueError(
                "지원되지 않는 파일 형식입니다. CSV, XLS, XLSX 파일만 지원됩니다."
            )

    def _load_data(self):
        try:
            if self.file_path.endswith(".csv"):
                encodings = ["utf-8", "cp949", "euc-kr", "iso-8859-1"]
                for encoding in encodings:
                    try:
                        df = pd.read_csv(self.file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(
                        f"Unable to read the CSV file with any of the attempted encodings: {encodings}"
                    )
            elif self.file_path.endswith(".xls") or self.file_path.endswith(".xlsx"):
                df = pd.read_excel(self.file_path, header=0)
            else:
                raise ValueError(
                    "지원되지 않는 파일 형식입니다. CSV, XLS, XLSX 파일만 지원됩니다."
                )

            if df.empty:
                raise ValueError("파일에 데이터가 없습니다.")

            return df
        except pd.errors.EmptyDataError:
            raise ValueError("파일이 비어있거나 파싱할 수 있는 열이 없습니다.")
        except Exception as e:
            raise ValueError(f"파일을 불러오는 중 오류가 발생했습니다: {str(e)}")

    def _create_database(self):
        conn = sqlite3.connect(f"{self.file_name}.db")
        self.df.to_sql(self.table_name, conn, index=False, if_exists="replace")

        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'"
        )
        if cursor.fetchone() is None:
            raise ValueError(f"Failed to create table: {self.table_name}")

        return conn

    def _create_sql_database(self):
        db_path = f"sqlite:///{self.file_name}.db"
        try:
            db = SQLDatabase.from_uri(db_path)
            db.run(f"SELECT * FROM {self.table_name} LIMIT 1")
            return db
        except Exception as e:
            raise ValueError(f"Failed to create SQLDatabase: {str(e)}")

    def _create_prompt(self):
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        columns_str = ", ".join(columns)
        print(f"Table name: {self.table_name}")
        print(f"Columns: {columns_str}")

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    f"""
You are a database expert. Provide accurate and detailed answers to user questions. You can use SQL queries to retrieve necessary information from the database.
If possible, refer to previous conversation content to understand the user's intent and use the necessary tools to follow instructions.
The table name is '{self.table_name}' and it has the following columns: {columns_str}
"""
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        return prompt

    def _initialize_agent_executor(self):
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
        exception_words = set()
        # 컬럼명 추가
        exception_words.update(self.df.columns)
        # 각 컬럼의 고유한 값들 추가
        for column in self.df.columns:
            exception_words.update(self.df[column].astype(str).unique())
        return list(exception_words)

    def process_response(self, response):
        translator = GoogleTranslator(source="auto", target="ko")

        # 예외 단어들을 위한 임시 플레이스홀더 생성
        placeholders = {}
        for idx, word in enumerate(self.exception_words):
            placeholder = f"__EXCEPTION_{idx}__"
            placeholders[word] = placeholder
            response = re.sub(
                r"\b" + re.escape(word) + r"\b",
                placeholder,
                response,
                flags=re.IGNORECASE,
            )

        # 번역 수행
        translated = translator.translate(response)

        # 플레이스홀더를 원래 단어로 대체
        for word, placeholder in placeholders.items():
            translated = translated.replace(placeholder, word)

        return translated

    def ask_question(self, question, num_trials=3):
        responses = []
        total_tokens_used = 0
        for i in range(num_trials):
            with get_openai_callback() as cb:
                try:
                    response = self.agent_executor.run(input=question)
                    response = self.process_response(response)
                    responses.append(response)
                    total_tokens_used += cb.total_tokens
                except Exception as e:
                    print(f"Error in trial {i+1}: {str(e)}")
                    responses.append(f"Error: {str(e)}")

        response_count = Counter(responses)
        most_common_response = response_count.most_common(1)[0][0]
        print(f"Most frequent response: {most_common_response}")

        print(f"User: {question}")
        for idx, resp in enumerate(responses):
            print(f"Assistant (Trial {idx+1}): {resp}")

        if all(resp == responses[0] for resp in responses):
            print("All responses are identical.")
        else:
            print("Responses differ.")

        print(f"Total Tokens Used: {total_tokens_used}\n")
        return most_common_response


# Usage example
if __name__ == "__main__":
    file_path = "/content/sports.xlsx"  # Change to your actual file path

    try:
        chatbot = SQLChatbot(file_path)
        chatbot.ask_question("What columns are in the table?")
        chatbot.ask_question(
            "What is the name of the sport with the best answer results?"
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")
