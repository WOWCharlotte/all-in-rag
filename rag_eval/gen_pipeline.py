import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import warnings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.cache import DiskCacheBackend
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
import traceback
import logging

warnings.filterwarnings("ignore")
if load_dotenv(".env"):
    print("Loaded environment variables from .env file")
    ENV_LOADED = True
else:
    print("No .env file found")
    ENV_LOADED = False

# 设置日志级别：INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GeneratePipeline():
    def __init__(self,input_dir:str,output_dir:str='evals/experiments'):
        if ENV_LOADED:
            cache = DiskCacheBackend()
            llm = ChatOpenAI(
                model=os.getenv("EVAL_LLM_MODEL"),
                base_url=os.getenv("EVAL_LLM_BINDING_HOST"),
                api_key=os.getenv("EVAL_LLM_BINDING_API_KEY"),
                temperature=1.0,
            )
            embeddings = OpenAIEmbeddings(
                model = os.getenv("EVAL_EMBEDDING_MODEL"),
                base_url=os.getenv("EVAL_EMBEDDING_BINDING_HOST"),
                api_key=os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
            )
            self.llm = LangchainLLMWrapper(llm, cache=cache)
            self.embeddings = LangchainEmbeddingsWrapper(embeddings, cache=cache)
            self.input_dir = input_dir
            if os.path.exists(output_dir):
                self.output_dir = output_dir
            else:
                os.makedirs(output_dir)
                self.output_dir = output_dir
            self._display_configuration()
        else:
            raise Exception("Please load environment variables from .env file")
        
    def _display_configuration(self):
        # 打印模型配置信息
        logger.info("Model Configuration:")
        logger.info(f"LLM: {self.llm}")
        logger.info(f"Embeddings: {self.embeddings}")
        # 打印文件配置信息
        logger.info("File Configuration:")
        logger.info(f"Input Directory: {self.input_dir}")
        logger.info(f"Output Directory: {self.output_dir}")

    async def _load_dataset(self,file_type:str='txt'):
        loader = DirectoryLoader(self.input_dir, glob=f"**/*.{file_type}") # 实际应用中数据格式多样，需要不同的数据加载策略
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents")
        return docs
    
    async def _gen_rag_testcase(
            self,
            docs,
            single_rate:float=0.5,
            multi_rate:float=0.5,
            language:str='chinese',
    ):
        # 安全判断
        if not os.path.exists(self.input_dir):
            raise Exception(f"Input directory {self.input_dir} does not exist")
        if single_rate < 0.0 or single_rate > 1.0 or multi_rate < 0.0 or multi_rate > 1.0:
            raise Exception("Rate must be between 0.0 and 1.0")
        if single_rate + multi_rate != 1.0:
            raise Exception("Single rate and multi rate must sum up to 1.0")
        if not docs:
            raise Exception("No documents loaded. Please run _load_dataset() before generating test cases")
        # 设定角色
        personas = [
            Persona(
                name="大学生",
                role_description="在校大学生，没有AI经验，想要入门RAG",
            ),
            Persona(
                name="AI应用工程师",
                role_description="AI应用工程师，有三年工作经验，想要了解RAG的技术原理",
            )
        ]
        # 单跳/多跳查询问题比例设置
        if single_rate == 1.0:
            distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=self.llm), 1.0),
            ]
        elif multi_rate == 1.0:
            distribution = [
                (MultiHopSpecificQuerySynthesizer(llm=self.llm), 1.0),
            ]
        else:
            distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=self.llm),single_rate),
                (MultiHopSpecificQuerySynthesizer(llm=self.llm), multi_rate),
            ]
        for query, _ in distribution:
            # 适配中文提示词，提升生成任务的中文场景贴合度
            prompts = await query.adapt_prompts(language, llm=self.llm)
            query.set_prompts(**prompts)

        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            persona_list=personas
        )
        try:
            dataset = generator.generate_with_langchain_docs(
                docs[:],
                testset_size=10, # 生成的测试用例数量
                query_distribution=distribution,
            )
            dataset = dataset.to_pandas()
            dataset.to_csv(
                f"{self.output_dir}/testcase.csv",
                index=False,
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Error generating test cases: {e}")
            logger.error(traceback.format_exc())



async def main_sync():
    pipeline = GeneratePipeline("evals/datasets/custom")
    docs = await pipeline._load_dataset("md")
    await pipeline._gen_rag_testcase(docs)

if __name__ == "__main__":
    asyncio.run(main_sync())