from ragas import evaluate
from typing import Any
import httpx
from typing import Dict
import pandas as pd
import logging
import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from ragas import DiskCacheBackend
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import asyncio
from datetime import datetime
from rag import default_rag_client
from datasets import Dataset
from ragas.metrics import ContextPrecision,ContextRecall,ContextEntityRecall,NoiseSensitivity,AnswerRelevancy,Faithfulness
import numpy as np

if load_dotenv():
    print("Loaded environment variables from .env file.")
    ENV_LOADED = True
else:
    print("No .env file found. Using default configuration.")
    ENV_LOADED = False

# 日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)

class RAGEvalPipeline:
    def __init__(self,  test_dataset_path: str = "evals/experiments/testcase.csv"):
        if ENV_LOADED:
            # 初始化模型
            cache = DiskCacheBackend()
            llm = ChatOpenAI( 
                model=os.getenv("EVAL_LLM_MODEL"),
                base_url=os.getenv("EVAL_LLM_BINDING_HOST"),
                api_key=os.getenv("EVAL_LLM_BINDING_API_KEY"),
                temperature=1.0,
            )
            self.chat_model = llm
            self.llm = LangchainLLMWrapper(llm, cache=cache)
            embeddings = OpenAIEmbeddings(
                model = os.getenv("EVAL_EMBEDDING_MODEL"),
                base_url=os.getenv("EVAL_EMBEDDING_BINDING_HOST"),
                api_key=os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
            )
            self.embeddings = LangchainEmbeddingsWrapper(embeddings, cache=cache)
            self.eval_max_retries = int(os.getenv("EVAL_MAX_RETRIES", "5"))
            self.eval_timeout = int(os.getenv("EVAL_TIMEOUT", "60"))
            self.test_dataset_path = Path(test_dataset_path)
            self.results_dir = Path(__file__).parent / "results"
            self.results_dir.mkdir(exist_ok=True)

            # Add FileHandler
            log_file = self.results_dir / "eval_pipeline.log"
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)

            self.test_cases = self._load_test_dataset()
            self._display_configuration()
        else:
            raise Exception("Please load environment variables from .env file")
        
    def _display_configuration(self):
        """Display all evaluation configuration settings"""
        logger.info("Evaluation Models:")
        logger.info("  • LLM Model:            %s", self.llm)
        logger.info("  • Embedding Model:      %s", self.embeddings)

        logger.info("Concurrency & Rate Limiting:")
        query_top_k = int(os.getenv("EVAL_QUERY_TOP_K", "10"))
        logger.info("  • Query Top-K:          %s Entities/Relations", query_top_k)
        logger.info("  • LLM Max Retries:      %s", self.eval_max_retries)
        logger.info("  • LLM Timeout:          %s seconds", self.eval_timeout)

        logger.info("Test Configuration:")
        logger.info("  • Total Test Cases:     %s", len(self.test_cases))
        logger.info("  • Test Dataset:         %s", self.test_dataset_path.name)
        logger.info("  • Results Directory:    %s", self.results_dir.name)

    def _load_test_dataset(self):
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.test_dataset_path}")
        df = pd.read_csv(self.test_dataset_path)
        # 校验字段
        must_columns = ['user_input','reference_contexts','reference']
        for col in must_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        return df
    
    async def gen_rag_response(self):
        rag_logs_dir = self.results_dir / "logs"
        rag_logs_dir.mkdir(exist_ok=True)
        rag_client = default_rag_client(llm_client=self.chat_model, logdir=str(rag_logs_dir))

        top_k = 5

        async def run_case(row):
            question = row["user_input"]
            last_error = None
            for attempt in range(self.eval_max_retries):
                try:
                    result = await asyncio.wait_for(
                        rag_client.query(question, top_k=top_k),
                        timeout=self.eval_timeout
                    )
                    answer = result.get("answer", "")
                    # 如果返回的是错误信息则重试
                    if isinstance(answer, str) and answer.startswith("Error"):
                        last_error = answer
                        continue
                    return {
                        "user_input": question,
                        "reference_contexts": row.get("reference_contexts", ""),
                        "reference": row.get("reference", ""),
                        "answer": answer,
                        "run_id": result.get("run_id", ""),
                    }
                except Exception as e:
                    last_error = str(e)
                    continue
            # 重试失败后返回错误
            return {
                "user_input": question,
                "reference_contexts": row.get("reference_contexts", ""),
                "reference": row.get("reference", ""),
                "answer": f"Error after retries: {last_error}",
                "run_id": "",
            }

        tasks = [run_case(row) for _, row in self.test_cases.iterrows()]
        results = await asyncio.gather(*tasks)
        df = pd.DataFrame(results)
        return df

    async def eval_rag_response(self,eval_dataset:pd.DataFrame):
        must_columns = ['user_input','reference_contexts','reference','answer']
        for col in must_columns:
            if col not in eval_dataset.columns:
                raise ValueError(f"Missing column: {col}")
        
        # 准备数据，转换为 Ragas Dataset 格式
        data = {
            "question": eval_dataset["user_input"].tolist(),
            "answer": eval_dataset["answer"].tolist(),
            "ground_truth": eval_dataset["reference"].tolist(),
            "retrieved_contexts": [],
        }
        
        for _, row in eval_dataset.iterrows():
            ctx = row["reference_contexts"]
            if isinstance(ctx, str):
                data["retrieved_contexts"].append([ctx])
            elif isinstance(ctx, list):
                data["retrieved_contexts"].append(ctx)
            else:
                data["retrieved_contexts"].append([str(ctx)] if ctx is not None else [])

        dataset = Dataset.from_dict(data)

        metrics = [
            ContextPrecision(), 
            ContextRecall(), 
            AnswerRelevancy(), 
            ContextEntityRecall(), 
            NoiseSensitivity(), 
            Faithfulness() 
        ]
        
        logger.info("Starting bulk evaluation with Ragas...")
        
        # 使用 asyncio.to_thread 避免阻塞主事件循环，因为 evaluate 可能是同步调用的
        result = await asyncio.to_thread(
            evaluate,
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False
        )
        
        out_df = result.to_pandas()
        
        outfile = self.results_dir / f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_df.to_csv(outfile, index=False, encoding="utf-8")
        logger.info("Saved RAG eval results to %s", outfile)
        return out_df

    def _display_results_table(self, results: pd.DataFrame):
        logger.info("")
        logger.info("%s", "=" * 115)
        logger.info("📊 EVALUATION RESULTS SUMMARY")
        logger.info("%s", "=" * 115)

        logger.info(
            "%-4s | %-50s | %6s | %7s | %6s | %7s | %6s | %6s | %6s",
            "#",
            "Question",
            "Faith",
            "AnswRel",
            "CtxRec",
            "CtxPrec",
            "EntRec",
            "Noise",
            "Status",
        )
        logger.info("%s", "-" * 115)

        def fmt(val, width):
            try:
                if val is None or (isinstance(val, float) and (pd.isna(val) or val != val)) or (hasattr(np, "isnan") and np.isnan(val)):
                    return f"{'N/A':>{width}}"
                return f"{float(val):>{width}.3f}"
            except Exception:
                return f"{str(val)[:width]:>{width}}"

        for idx, row in results.iterrows():
            q = row.get("question", row.get("user_input", ""))
            q_disp = (q[:47] + "...") if isinstance(q, str) and len(q) > 50 else q
            is_error = row.get("status", "") == "error"
            if is_error:
                err = row.get("error", "Unknown error")
                err_disp = (err[:20] + "...") if isinstance(err, str) and len(err) > 23 else err
                logger.info(
                    "%-4d | %-50s | %6s | %7s | %6s | %7s | %6s | %6s | ✗ %s",
                    idx + 1,
                    q_disp,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    err_disp,
                )
                continue

            faith = row.get("faithfulness", None)
            ans_rel = row.get("answer_relevancy", row.get("answer_relevance", None))
            ctx_rec = row.get("context_recall", None)
            ctx_prec = row.get("context_precision", None)
            ent_rec = row.get("context_entity_recall", None)
            noise = row.get("noise_sensitivity", None)
            logger.info(
                "%-4d | %-50s | %s | %s | %s | %s | %s | %s  | %s",
                idx + 1,
                q_disp,
                fmt(faith, 6),
                fmt(ans_rel, 7),
                fmt(ctx_rec, 6),
                fmt(ctx_prec, 7),
                fmt(ent_rec, 6),
                fmt(noise, 6),
                "✓",
            )

        logger.info("%s", "=" * 115)
        
def main():
    pipeline = RAGEvalPipeline()
    rag_df = asyncio.run(pipeline.gen_rag_response())
    resp_file = pipeline.results_dir / f"rag_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rag_df.to_csv(resp_file, index=False)
    logger.info("Saved RAG responses to %s", resp_file)
    eval_df = asyncio.run(pipeline.eval_rag_response(rag_df))
    pipeline._display_results_table(eval_df)
    print(f"Responses: {resp_file}")
    print(f"Eval rows: {len(eval_df)}")
    print("Done.")

if __name__ == "__main__":
    main()
