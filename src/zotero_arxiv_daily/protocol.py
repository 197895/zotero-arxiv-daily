from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    authors_cn: Optional[list[str]] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', '中文')
        prompt = (
            f"请用{lang}分析下面这篇论文，按以下格式输出：\n\n"
            "领域：[从视频生成/图像生成/世界模型/VLA/VLM/MLLM/LLM推理/推理系统/扩散模型/RL/Agent/具身智能/自动驾驶/语音/音频/3D/数据/安全/对齐/其他 中选择1-2个最匹配的]\n"
            "方法：[从无训练/缓存加速/少步生成加速/蒸馏加速/并行解码加速/投机解码加速/算子优化/编译优化/稀疏加速/量化加速/KV Cache优化/检索增强/模型架构改进/训练框架改进/数据增强/后训练优化 中选择1-3个最匹配的，没有就写'其他']\n"
            "方法是否需要训练：是/否\n"
            "测试指标：[列出论文中主要使用的评测指标，如VBench/LPIPS/SSIM/PSNR/TTFT/FPS/准确率/BLEU等，逗号分隔]\n"
            "创新点罗列：1. ...\n2. ...\n3. ...（列出核心创新点）\n"
            "简介：用一句通俗的话总结这篇论文做了什么、有什么用。\n\n"
            "注意：创新点要具体，不要泛泛而谈。简介要让非专业人士也能听懂。\n\n"
            "正确示例：\n"
            "领域：视频生成\n"
            "方法：少步生成加速、KV Cache优化\n"
            "方法是否需要训练：否\n"
            "测试指标：VBench, FVD, LPIPS, CLIP-SIM\n"
            "创新点罗列：1. 提出时序回溯搜索，允许在扩散过程中回退并重新生成出错片段\n2. 设计前缀共享机制，多个搜索分支复用已生成的正确前缀，避免重复计算\n3. 引入时序过程验证器，自动检测生成过程中的物理不一致并定位错误帧\n"
            "简介：这篇文章解决的是AI生成视频时动作不合理的问题——比如人走路突然飘起来。它让模型像下棋一样边生成边回头看，发现不对劲就撤回重来，最终生成的视频物理上更合理。\n\n"
        )
        if self.title:
            prompt += f"标题：\n{self.title}\n\n"

        if self.abstract:
            prompt += f"摘要：\n{self.abstract}\n\n"

        if self.full_text:
            prompt += f"正文片段：\n{self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "无法生成 TLDR：未提供摘要或正文。"

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"你是一位擅长分析学术论文的助手。请用{lang}回答，"
                        "严格按照用户指定的格式输出，不要添加额外说明。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]