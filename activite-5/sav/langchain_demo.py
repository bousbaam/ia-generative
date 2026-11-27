import ast
import os
import re
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, before_agent, dynamic_prompt, wrap_model_call, wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from sav.graphes import retriever_tool


def modele_local(nom=None):
    return ChatOllama(
        model=nom or os.getenv("SAV_MODEL", "llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0, num_predict=256, client_kwargs={"timeout": 45},
    )


@dataclass
class Contexte:
    niveau: str = "debutant"


@tool
def calculer_ttc(prix_ht: float) -> float:
    """Calcule un prix TTC avec une TVA fictive de 20 %."""
    if not 0 <= prix_ht <= 100_000:
        raise ValueError("Le prix doit être compris entre 0 et 100000.")
    return round(prix_ht * 1.20, 2)


@tool
def ouvrir_ticket_demo(sujet: str) -> str:
    """Simule l’ouverture d’un ticket SAV, avec approbation humaine."""
    return f"Ticket simulé approuvé pour « {sujet[:100]} ». Aucun ticket externe créé."


@tool
def chercher_duckduckgo(requete: str) -> str:
    """Recherche web publique DuckDuckGo. Utiliser des mots-clés génériques."""
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    outil = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(backend="duckduckgo"))
    return str(outil.invoke(requete[:200]))[:2200]


@tool
def chercher_tavily(requete: str) -> str:
    """Recherche web publique Tavily. Nécessite TAVILY_API_KEY."""
    from langchain_tavily import TavilySearch

    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("Clé Tavily absente.")
    return str(TavilySearch(max_results=3).invoke({"query": requete[:200]}))[:2200]


@tool
def python_repl_calcul(expression: str) -> str:
    """Évalue un calcul arithmétique court via PythonREPLTool."""
    from langchain_experimental.tools import PythonREPLTool

    if len(expression) > 80:
        raise ValueError("Expression trop longue.")
    arbre = ast.parse(expression.strip(), mode="eval")
    autorises = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, autorises):
            raise ValueError("Seuls les nombres et +, -, *, / sont autorisés.")
        if isinstance(noeud, ast.Constant) and (
            type(noeud.value) not in (int, float) or not abs(noeud.value) <= 1_000_000
        ):
            raise ValueError("Constante invalide.")
    return PythonREPLTool().invoke(f"print({ast.unparse(arbre.body)})").strip()


@dynamic_prompt
def consigne_dynamique(request):
    niveau = getattr(request.runtime.context, "niveau", "debutant")
    detail = "Donne des détails techniques." if niveau == "expert" else "Utilise des mots simples."
    return "Assistant SAV francophone. " + detail + " Ne donne pas d’information personnelle."


@wrap_tool_call
def gerer_erreur(request, handler):
    try:
        return handler(request)
    except Exception:
        return ToolMessage(
            content="L’outil a échoué. Vérifie les paramètres et la configuration.",
            tool_call_id=request.tool_call["id"], status="error",
        )


@before_agent(can_jump_to=["end"])
def garde_fou(state, runtime):
    dernier = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    texte = dernier if isinstance(dernier, str) else str(dernier)
    if len(texte) > 2000 or re.search(r"(?:sk-[\w-]{16,}|\b\d{13,19}\b)", texte):
        return {"messages": [AIMessage(content="Retire les données sensibles ou raccourcis la demande.")], "jump_to": "end"}
    return None


def agent_simple(modele=None):
    return create_agent(modele or modele_local(), system_prompt="Réponds brièvement en français.")


def creer_agent(modele=None, avance=None, web=False):
    base = modele or modele_local(os.getenv("SAV_MODEL_STANDARD", "qwen2.5:1.5b"))
    expert = avance or modele_local(os.getenv("SAV_MODEL_EXPERT", "llama3.2:3b"))

    @wrap_model_call
    def dynamic_model(request, handler):
        niveau = getattr(request.runtime.context, "niveau", "debutant")
        return handler(request.override(model=expert if niveau == "expert" else base))

    outils = [calculer_ttc, python_repl_calcul, ouvrir_ticket_demo]
    if web:
        outils.extend([chercher_duckduckgo, chercher_tavily])
    return create_agent(
        base, outils, context_schema=Contexte, checkpointer=InMemorySaver(),
        middleware=[
            garde_fou, dynamic_model, consigne_dynamique, gerer_erreur,
            HumanInTheLoopMiddleware(interrupt_on={"ouvrir_ticket_demo": {"allowed_decisions": ["approve", "reject"]}}),
        ],
    )


def chatbot_rag_langchain(modele=None):
    return create_agent(
        modele or modele_local(), tools=[retriever_tool],
        system_prompt="Pour les questions documentaires, utilise retriever_tool et cite les identifiants et pages.",
        checkpointer=InMemorySaver(),
    )
