import asyncio
import json
import os
import httpx
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def adresse_mcp():
    return os.getenv("MCP_URL", "http://127.0.0.1:8005/mcp")


async def appeler_mcp(outil, arguments):
    async with (
        httpx.AsyncClient(timeout=15) as client,
        streamable_http_client(adresse_mcp(), http_client=client) as (lecture, ecriture, _),
    ):
        async with ClientSession(lecture, ecriture) as session:
            await session.initialize()
            resultat = await session.call_tool(outil, arguments)
            if resultat.isError:
                raise ValueError("Le serveur MCP a refusé la requête documentaire.")
            if resultat.structuredContent is not None:
                contenu = resultat.structuredContent
                return contenu.get("result", contenu)
            return json.loads(resultat.content[0].text)


async def agent_mcp(modele):
    client = MultiServerMCPClient({"sav": {"transport": "streamable_http", "url": adresse_mcp()}})
    outils = await client.get_tools()
    return create_agent(modele, outils, system_prompt=(
        "Pour toute question sur le manuel PX-200, appelle rag_multimodal et cite ses sources. "
        "Passe la référence de l’image lorsqu’elle est présente."
    ))


async def main():
    from sav.langchain_demo import modele_local

    agent = await agent_mcp(modele_local())
    sortie = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Que signifie le voyant de la photo img_px200_01 ?"}]},
        config={"recursion_limit": 12},
    )
    print(sortie["messages"][-1].content)
    for message in sortie["messages"]:
        if message.type == "tool" and message.name == "rag_multimodal" and message.status == "success":
            document = json.loads(message.content_blocks[0]["text"])
            for source in document["sources"]:
                print(f'{source["id"]} · p. {source["page"]} · {source["type"]}')


if __name__ == "__main__":
    asyncio.run(main())
