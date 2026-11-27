import re
from typing import Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from sav.client_mcp import appeler_mcp
from sav.donnees_garantie import consulter_garantie


class DemandeSAV(TypedDict, total=False):
    question: str
    image_ref: str
    image_base64: str
    categorie: Literal["documentation", "garantie"]
    agent: str
    reponse: str
    sources: list[dict]


@tool("consulter_garantie")
def outil_garantie(numero_serie: str) -> dict:
    """Lire l’état de la garantie d’un numéro PX200 du registre fictif."""
    return consulter_garantie(numero_serie)


def orchestrateur(state):
    if state.get("reponse"):
        return {}
    texte = state["question"]
    garantie = re.search(r"\b(garantie|numéro de série|sous garantie)\b", texte, flags=re.I)
    return {"categorie": "garantie" if garantie else "documentation"}


async def agent_documentaire(state):
    arguments = {"question": state["question"]}
    arguments.update({k: state[k] for k in ("image_ref", "image_base64") if state.get(k)})
    try:
        resultat = await appeler_mcp("rag_multimodal", arguments)
    except Exception:
        return {
            "agent": "agent_documentaire", "sources": [],
            "reponse": "Le service documentaire est indisponible ou la demande n’a pas pu être traitée.",
        }
    return {"agent": "agent_documentaire", **resultat}


def agent_garantie(state):
    numero = re.search(r"\bPX200-\d{4}\b", state["question"], flags=re.I)
    if not numero:
        reponse = "Indiquez le numéro de série complet au format PX200-0042."
    else:
        resultat = outil_garantie.invoke({"numero_serie": numero.group()})
        if resultat["statut"] == "active":
            reponse = f"La garantie de {resultat['numero_serie']} est active jusqu’au {resultat['fin']} inclus."
        elif resultat["statut"] == "expiree":
            reponse = f"La garantie de {resultat['numero_serie']} a expiré le {resultat['fin']}."
        else:
            reponse = "Ce numéro n’est pas reconnu dans le registre de démonstration."
    return {"agent": "agent_garantie", "reponse": reponse, "sources": []}


def choix(state):
    if state.get("reponse"):
        return END
    return "agent_" + ("garantie" if state["categorie"] == "garantie" else "documentaire")


definition = StateGraph(DemandeSAV)
definition.add_node("orchestrateur", orchestrateur)
definition.add_node("agent_documentaire", agent_documentaire)
definition.add_node("agent_garantie", agent_garantie)
definition.add_edge(START, "orchestrateur")
definition.add_conditional_edges("orchestrateur", choix, [END, "agent_documentaire", "agent_garantie"])
definition.add_edge("agent_documentaire", "orchestrateur")
definition.add_edge("agent_garantie", "orchestrateur")
sav_graph = definition.compile()
