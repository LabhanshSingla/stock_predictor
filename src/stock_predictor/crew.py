import os, yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from .tools import fetch_prices, feature_engineer, train_linear, predict_close

load_dotenv()

CFG_DIR = os.path.join(os.path.dirname(__file__), "config")

def load_yaml(name: str):
    with open(os.path.join(CFG_DIR, name), "r") as f:
        return yaml.safe_load(f)

def build_llm(llm_cfg: dict) -> LLM:
    return LLM(
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.2),
        max_tokens=llm_cfg.get("max_tokens", 800),
    )

def get_tools(names):
    lut = {
        "fetch_prices": fetch_prices,
        "feature_engineer": feature_engineer,
        "train_linear": train_linear,
        "predict_close": predict_close,
    }
    return [lut[n] for n in names]

def build_crew():
    cfg_agents = load_yaml("agents.yaml")
    cfg_tasks = load_yaml("tasks.yaml")
    llm = build_llm(cfg_agents["llm"])

    # Agents
    A = {}
    for key, a in cfg_agents["agents"].items():
        A[key] = Agent(
            role=a["role"],
            goal=a["goal"],
            backstory=a["backstory"],
            allow_delegation=a.get("allow_delegation", False),
            verbose=a.get("verbose", False),
            llm=llm,
            tools=get_tools(a.get("tools", []))
        )

    # Tasks (note: we chain tools inside API call rather than auto-run here)
    T_cfg = cfg_tasks["tasks"]
    T = {
        "collect_task": Task(
            description=T_cfg["collect_task"]["description"],
            expected_output=T_cfg["collect_task"]["expected_output"],
            agent=A["collector"],
        ),
        "analyze_task": Task(
            description=T_cfg["analyze_task"]["description"],
            expected_output=T_cfg["analyze_task"]["expected_output"],
            agent=A["analyst"],
        ),
        "predict_task": Task(
            description=T_cfg["predict_task"]["description"],
            expected_output=T_cfg["predict_task"]["expected_output"],
            agent=A["predictor"],
        ),
    }

    crew = Crew(agents=list(A.values()), tasks=list(T.values()))
    return crew, A, T
