import threading
import psutil
from llama_cpp import Llama
import pandas as pd
from pathlib import Path
import re
import os
import queue
import time
import traceback
from multiprocessing import Process, Queue
import logging
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
import torch
import numpy as np
import random
import multiprocessing
import sys
from typing import Callable

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 预备模型
myModel = Llama(
    r"C:\AProjects\agentAPP\model\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=8192,
    tensor_split=None,
    main_gpu=0,
    low_vram=False,
)

#region 配置
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def set_seed(seed=531):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed()

config = {

    "exp_name": "ML2025_HW2",
    "data_dir": Path("C:/AProjects/agentAPP/AIAgent/content").resolve(),
    "task_goal": (
        "给定美国特定州过去两天的调查结果，"
        "预测第3天检测呈阳性的概率。"
        "评估指标为均方误差(MSE)。"
    ),
    "interpreter": {
        "code_wait_timeout": 120,  # 增加到120秒
        "exec_timeout": 600,       # 增加到600秒
    },
    "agent": {
        "steps": 3,
        "search": {
            "debug_prob": 0.7,
            "num_drafts": 1,
        },
    },
}
cfg = Config(config)
#endregion

def generate_response(_model: Llama, _messages: List[Dict[str, str]]) -> str:
    try:
        output = _model.create_chat_completion(
            _messages,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "###END"],
            max_tokens=4096,
            temperature=0,
        )
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"模型响应生成失败: {str(e)}")
        return ""

# 保存每一个程序
def save_run(cfg, journal):
    best_node = journal.get_best_node(only_good=False)
    if best_node:
        with open("best_solution.py", "w", encoding="utf-8") as f:
            f.write(best_node.code)

    good_nodes = journal.get_good_nodes()
    for i, node in enumerate(good_nodes):
        filename = f"good_solution_{i}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(node.code)

# 辅助函数：截断长字符串
def trim_long_string(string, threshold=5100, k=2500):
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n... [{truncated_len} characters truncated] ...\n{last_k_chars}"
    return string

# 辅助函数：包装代码块
def wrap_code(code: str, lang="python") -> str:
    return f"```{lang}\n{code}\n```"

# 辅助函数：验证Python代码
def is_valid_python_script(script):
    if not script.strip():
        return False
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError as e:
        logging.warning(f"语法验证失败: {str(e)}")
        return False

# 辅助函数：提取代码
def extract_code(text):
    if not text:
        return ""

    # 尝试提取标准代码块
    matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if matches:
        return "\n\n".join(matches).strip()

    # 尝试提取未封闭的代码块
    matches = re.findall(r"```(?:python)?\n(.*)", text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 尝试提取未标记的代码块
    code_candidates = re.findall(r"(?:def|class|import|from|print|return)[\s\S]*?$", text)
    if code_candidates:
        return "\n".join(code_candidates).strip()

    return text.strip()

# 辅助函数：提取自然语言文本
def extract_text_up_to_code(s):
    if "```" in s:
        return s.split("```")[0].strip()
    return s


# Python解释器
@dataclass
class ExecutionResult:
    term_out: List[str]
    exec_time: float
    exc_type: Optional[str]
    exc_info: Optional[Dict] = None
    exc_stack: Optional[List] = None

def exception_summary(e, exec_file_name):
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    tb_str = "".join(tb_lines)

    exc_info = {"args": [str(i) for i in e.args]} if hasattr(e, "args") else {}
    exc_stack = traceback.extract_tb(e.__traceback__)

    return tb_str, e.__class__.__name__, exc_info, exc_stack

class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        try:
            self.queue.put_nowait(msg)
        except queue.Full:
            pass

    def flush(self):
        pass

class Interpreter:
    def __init__(self,
                 timeout: int = cfg.interpreter.exec_timeout,
                 agent_file_name: str = "runfile.py",
                 code_wait_timeout: int = cfg.interpreter.code_wait_timeout):
        self.timeout = timeout
        self.code_wait_timeout = code_wait_timeout  # 新增
        self.agent_file_name = agent_file_name
        self.process = None
        self.code_inq = None
        self.result_outq = None
        self.event_outq = None

    def _run_session(self, code_inq: Queue, result_outq: Queue, event_outq: Queue):
        # 设置输出重定向
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

        global_scope = {}
        event_outq.put(("state:ready",))

        while True:
            try:
                code = code_inq.get(timeout=self.code_wait_timeout)
                with open(self.agent_file_name, "w", encoding="utf-8") as f:
                    f.write(code)

                try:
                    with open(self.agent_file_name, "r", encoding="utf-8") as f:
                        compiled_code = compile(f.read(), self.agent_file_name, "exec")

                    # 添加进度报告
                    progress_thread = threading.Thread(
                        target=self._report_progress,
                        args=(event_outq,),
                        daemon=True
                    )
                    progress_thread.start()

                    exec(compiled_code, global_scope)

                finally:
                    event_outq.put(("state:finished", None, None, None))

                result_outq.put("<|EOF|>")
                os.remove(self.agent_file_name)
            except queue.Empty:
                logging.warning("子进程超时等待代码")
                event_outq.put(("state:finished", "TimeoutError", {}, []))
                break
            except Exception as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(e, self.agent_file_name)
                result_outq.put(tb_str)
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
                result_outq.put("<|EOF|>")
                break

        process = psutil.Process(os.getpid())
        while True:
            try:
                code = code_inq.get(timeout=self.code_wait_timeout)
                # ...
                exec(compiled_code, global_scope)
            except Exception as e:
                # 添加资源使用报告
                mem_usage = f"{process.memory_info().rss / 1024 ** 2:.2f}MB"
                cpu_usage = f"{process.cpu_percent()}%"
                error_msg = f"内存: {mem_usage}, CPU: {cpu_usage}\n{str(e)}"
                result_outq.put(error_msg)

    def _report_progress(self, event_outq):
        """定期发送进度心跳"""
        while True:
            time.sleep(30)  # 每30秒报告一次
            event_outq.put(("state:progress", "Still running..."))

    def create_process(self):
        self.code_inq = Queue()
        self.result_outq = Queue()
        self.event_outq = Queue()

        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
            daemon=True
        )
        self.process.start()

        try:
            state = self.event_outq.get(timeout=30)
            if state[0] != "state:ready":
                logging.error(f"子进程状态错误: {state[0]}")
                self.cleanup_session()
                raise RuntimeError("子进程初始化失败")
        except queue.Empty:
            logging.error("子进程初始化超时")
            self.cleanup_session()
            raise RuntimeError("子进程初始化超时")

    def cleanup_session(self):
        if self.process is None:
            return

        try:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1.0)
            self.process.close()
        except Exception as e:
            logging.error(f"清理进程失败: {str(e)}")
        finally:
            self.process = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        if reset_session or self.process is None or not self.process.is_alive():
            self.cleanup_session()
            self.create_process()

        # 简化代码执行过程
        self.code_inq.put(code)

        start_time = time.time()
        exc_type = None
        exc_info = None
        exc_stack = None

        # 添加超时保护
        try:
            self.code_inq.put(code, timeout=60)  # 添加put超时
        except queue.Full:
            logging.error("代码队列已满，无法提交")
            return ExecutionResult(["队列满错误"], 0, "QueueFull")

        exec_time = time.time() - start_time

        # 收集输出
        output = []
        while not self.result_outq.empty():
            try:
                msg = self.result_outq.get_nowait()
                if msg == "<|EOF|>":
                    break
                output.append(msg)
            except queue.Empty:
                break

        # 添加执行时间信息
        if exc_type == "TimeoutError":
            output.append(f"TimeoutError: 执行超过时间限制 ({self.timeout}秒)")
        else:
            output.append(f"执行时间: {exec_time:.2f}秒")

        return ExecutionResult(output, exec_time, exc_type, exc_info, exc_stack)

# 程序节点和日志
@dataclass(eq=False)
class Node:
    code: str
    plan: str = field(default="", kw_only=True)
    step: int = field(default=None, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)
    _term_out: List[str] = field(default_factory=list, kw_only=True)
    exec_time: float = field(default=None, kw_only=True)
    exc_type: Optional[str] = field(default=None, kw_only=True)
    exc_info: Optional[Dict] = field(default=None, kw_only=True)
    exc_stack: Optional[List] = field(default=None, kw_only=True)
    analysis: str = field(default="", kw_only=True)
    metric: float = field(default=float('inf'), kw_only=True)
    is_buggy: bool = field(default=True, kw_only=True)

    def __post_init__(self):
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1

@dataclass
class Journal:
    nodes: List[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[Node]:
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> List[float]:
        return [n.metric for n in self.nodes]

    def get_good_nodes(self) -> List[Node]:
        return self.good_nodes

    def get_best_node(self, only_good=True) -> Optional[Node]:
        nodes = self.good_nodes if only_good else self.nodes
        if not nodes:
            return None
        return min(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        summary = []
        for n in self.good_nodes:
            part = f"设计: {n.plan}\n"
            if include_code:
                part += f"代码:\n{wrap_code(n.code)}\n"
            part += f"分析: {n.analysis}\n"
            part += f"评估指标 (MSE): {n.metric}\n"
            summary.append(part)
        return "\n" + "-" * 40 + "\n".join(summary)

# 模型生成程序代码
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class Agent:
    def __init__(self, cfg, journal: Journal):
        self.cfg = cfg
        self.journal = journal
        self.data_preview: Optional[str] = None

    def search_policy(self) -> Optional[Node]:
        if len(self.journal) == 0:
            return None

        # 优先调试有错误的节点
        buggy_nodes = [n for n in self.journal.buggy_nodes if n.is_leaf]
        if buggy_nodes:
            return random.choice(buggy_nodes)

        # 否则选择最佳节点进行改进
        best_node = self.journal.get_best_node()
        return best_node if best_node else None

    def plan_and_code_query(self, system_message, user_message, retries=3) -> tuple[str, str]:
        for _ in range(retries):
            try:
                response = generate_response(
                    myModel,
                    _messages=[
                        {'role': 'system', "content": system_message},
                        {'role': 'user', "content": user_message}
                    ]
                )

                # 增强代码提取
                code = extract_code(response)
                if code and is_valid_python_script(code):
                    nl_text = extract_text_up_to_code(response)
                    return nl_text, code

                logging.warning("代码提取失败，重试中...")
            except Exception as e:
                logging.error(f"查询失败: {str(e)}")

        logging.error("最终代码提取失败，使用默认代码")
        return "代码生成失败", "print('Hello World')"

    def update_data_preview(self):
        self.data_preview = data_preview_generate(self.cfg.data_dir)

    def _draft(self) -> Node:
        system_prompt = (
            "你是一个AI编程助手，需要解决机器学习任务。"
            "请生成一个高效的Python程序来解决以下问题："
            "1. 使用增量加载处理大数据 (pd.read_csv(chunksize))"
            "2. 设置合理的max_iter/epochs (不超过100)"
            "3. 添加进度打印 (print('Epoch {i}'))"
            "将完整代码放在 ```python 代码块中。"
        )

        user_prompt = (
            f"任务: {self.cfg.task_goal}\n\n"
            f"数据目录: '{self.cfg.data_dir}'\n\n"
            f"数据预览:\n{self.data_preview}\n\n"
            "要求:\n"
            "1. 从CSV文件加载数据\n"
            "2. 训练一个回归模型预测'tested_positive_day3'\n"
            "3. 将测试集预测结果保存到'C:/AProjects/agentAPP/AIAgent/content/submission.csv'\n"
            "4. 将生成的submission.csv文件与C:/AProjects/agentAPP/AIAgent/content/sample_submission.csv文件中所有行数据进行除法得精准度文本，到C:/AProjects/agentAPP/AIAgent/content/accuracy该文件中"
            "5. 确保代码完整且可执行"
        )

        plan, code = self.plan_and_code_query(system_prompt, user_prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        system_prompt = (
            "你是一个AI编程助手，需要改进现有解决方案。"
            "分析以下代码和评估结果，提出改进方案并生成新代码。"
            "将代码放在 ```python 代码块中。"
        )

        user_prompt = (
            f"任务: {self.cfg.task_goal}\n\n"
            f"历史总结:\n{self.journal.generate_summary()}\n\n"
            f"当前代码:\n{wrap_code(parent_node.code)}\n\n"
            f"执行输出:\n{parent_node.term_out}\n\n"
            "改进方向:\n"
            "1. 尝试不同的模型或特征工程\n"
            "2. 优化超参数\n"
            "3. 改进数据处理流程"
        )

        plan, code = self.plan_and_code_query(system_prompt, user_prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def _debug(self, parent_node: Node) -> Node:
        system_prompt = (
            "你是一个AI编程助手，需要修复代码错误。"
            "分析以下代码和执行错误，修复问题并生成正确代码。"
            "将代码放在 ```python 代码块中。"
        )

        user_prompt = (
            f"任务: {self.cfg.task_goal}\n\n"
            f"错误代码:\n{wrap_code(parent_node.code)}\n\n"
            f"错误信息:\n{parent_node.term_out}\n\n"
            "修复要求:\n"
            "1. 修复所有语法和运行时错误\n"
            "2. 确保代码完整可执行\n"
            "3. 保留原始功能意图"
        )

        plan, code = self.plan_and_code_query(system_prompt, user_prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        node.absorb_exec_result(exec_result)

        # 简化分析逻辑
        if node.exc_type:
            node.analysis = f"执行失败: {node.exc_type}"
            node.is_buggy = True
            node.metric = float('inf')
        else:
            node.analysis = "执行成功"
            node.is_buggy = False
            node.metric = 0.5  # 简化指标

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        # 验证代码
        if not result_node.code.strip():
            result_node.code = "print('默认代码执行')"

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

# 特征选择
def preview_csv(p: Path) -> str:
    try:
        df = pd.read_csv(p, nrows=5)
        out = [
            f"文件: {p.name}",
            f"形状: {df.shape[0]}行, {df.shape[1]}列",
            f"列名: {', '.join(df.columns)}",
            "前5行数据:",
            str(df)
        ]
        return "\n".join(out)
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def data_preview_generate(base_path):
    result = []
    try:
        files = [p for p in Path(base_path).iterdir() if p.is_file() and p.suffix == ".csv"]
        for f in sorted(files):
            result.append(preview_csv(f))
        return "\n\n".join(result)
    except Exception as e:
        return f"生成数据预览失败: {str(e)}"

# 主程序
def main():
    journal = Journal()
    agent = Agent(cfg=cfg, journal=journal)
    interpreter = Interpreter(timeout=300)

    def exec_callback(code: str, reset: bool) -> ExecutionResult:
        return interpreter.run(code, reset)

    try:
        for step in range(cfg.agent.steps):
            logging.info(f"开始步骤 {step + 1}/{cfg.agent.steps}")
            agent.step(exec_callback)
            save_run(cfg, journal)
            time.sleep(1)
    finally:
        interpreter.cleanup_session()
        logging.info("程序执行完成")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
