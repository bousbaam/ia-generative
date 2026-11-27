import os

from mcp.server.fastmcp import FastMCP

from sav.documentation import rechercher
from sav.graphes import rag_chatbot


serveur = FastMCP(
    "SAV PX-200", host="127.0.0.1", port=int(os.getenv("MCP_PORT", "8005")),
    stateless_http=True, json_response=True,
)


@serveur.tool()
def rechercher_manuel(question: str, image_ref: str | None = None) -> list[dict]:
    """Recherche les passages et les figures du manuel de démonstration PX-200."""
    return rechercher(question, image_ref=image_ref)


@serveur.tool()
def rag_multimodal(
    question: str, image_ref: str | None = None, image_base64: str | None = None,
) -> dict:
    """Répond à une question sur le PX-200, éventuellement avec une image, et fournit les sources."""
    entree = {"question": question, "image_ref": image_ref, "image_base64": image_base64}
    sortie = rag_chatbot.invoke(entree)
    return {"reponse": sortie["reponse"], "sources": sortie["sources"]}


if __name__ == "__main__":
    serveur.run(transport="streamable-http")
