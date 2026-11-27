import asyncio
import base64
import json
import os
import unittest
from datetime import date
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from sav.client_mcp import agent_mcp, appeler_mcp
from sav.documentation import illustration, rechercher
from sav.donnees_garantie import consulter_garantie
from sav.graphes import graphe_agentique, graphe_bases, rag_chatbot
from sav.langchain_demo import Contexte, agent_simple, creer_agent, python_repl_calcul
from sav.orchestrateur import sav_graph


class ModeleDeTest(GenericFakeChatModel):
    """Réponses fixées pour vérifier la mécanique des agents."""
    def bind_tools(self, tools, **kwargs):
        return self


def action(nom, args, identifiant="action-1"):
    return AIMessage(content="", tool_calls=[{"name": nom, "args": args, "id": identifiant}])


def message(question):
    return {"messages": [{"role": "user", "content": question}]}


class TestsLocaux(unittest.TestCase):
    def test_bases_et_recherche(self):
        self.assertIn("Bienvenue", graphe_bases().invoke({"entree": " Bonjour "})["reponse"])
        self.assertEqual(rag_chatbot.invoke({"question": "Bonjour"})["sources"], [])
        self.assertEqual(rechercher("Voyant imprimante QZ-990"), [])
        self.assertEqual(rechercher("Que signifie img_px200_99 ?"), [])
        self.assertEqual(rechercher("Voyant de la photo IMG_PX200_01 ?")[0]["id"], "PX200-F01")
        for index in ("01", "02", "03"):
            reference = f"img_px200_{index}"
            data = base64.b64encode(illustration(reference)).decode()
            sources = rechercher("Quel est ce voyant du PX-200 ?", image_base64=data)
            self.assertEqual(sources[0]["image_ref"], reference)
            self.assertEqual({s["type"] for s in sources}, {"texte", "figure"})

    def test_registre_et_repl(self):
        self.assertEqual(consulter_garantie("PX200-0042", date(2026, 11, 27))["statut"], "active")
        self.assertEqual(consulter_garantie("PX200-0099", date(2026, 11, 27))["statut"], "expiree")
        self.assertEqual(consulter_garantie("PX200-0000")["statut"], "introuvable")
        self.assertEqual(python_repl_calcul.invoke({"expression": "(7+2)*3"}), "27")
        with self.assertRaises(ValueError):
            python_repl_calcul.invoke({"expression": "print(1)"})

    def test_agent_simple_memoire_et_middleware(self):
        simple = agent_simple(ModeleDeTest(messages=iter(["Bonjour."])))
        self.assertEqual(simple.invoke(message("Bonjour"))["messages"][-1].content, "Bonjour.")
        base = ModeleDeTest(messages=iter([
            action("calculer_ttc", {"prix_ht": 100}), "Le TTC est 120.", "Deuxième tour.",
        ]))
        expert = ModeleDeTest(messages=iter(["Mode expert."]))
        agent = creer_agent(base, expert)
        cfg = {"configurable": {"thread_id": "memoire"}}
        resultat = agent.invoke(message("Prix TTC de 100 ?"), cfg, context=Contexte())
        self.assertTrue(any(m.type == "tool" and "120" in m.content for m in resultat["messages"]))
        agent.invoke(message("Deuxième question"), cfg, context=Contexte())
        etat = agent.get_state(cfg)
        self.assertEqual(sum(m.type == "human" for m in etat.values["messages"]), 2)
        rep = agent.invoke(message("Donne des détails."), cfg, context=Contexte("expert"))
        self.assertEqual(rep["messages"][-1].content, "Mode expert.")
        bloque = agent.invoke(message("x" * 2001), cfg, context=Contexte())
        self.assertIn("raccourcis", bloque["messages"][-1].content)

    def test_erreur_et_approbation(self):
        base = ModeleDeTest(messages=iter([
            action("calculer_ttc", {"prix_ht": -1}), "Prix invalide.",
            action("ouvrir_ticket_demo", {"sujet": "Voyant orange"}, "ticket-1"), "Ticket traité.",
        ]))
        agent = creer_agent(base, base)
        cfg = {"configurable": {"thread_id": "approbation"}}
        err = agent.invoke(message("Prix TTC de -1"), cfg, context=Contexte())
        self.assertTrue(any(m.type == "tool" and m.status == "error" for m in err["messages"]))
        pause = agent.invoke(message("Ouvre un ticket de démonstration."), cfg, context=Contexte())
        self.assertTrue(pause["__interrupt__"])
        repris = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg, context=Contexte())
        self.assertTrue(any(m.type == "tool" and "Ticket simulé approuvé" in m.content for m in repris["messages"]))

    def test_rag_agentique(self):
        modele = ModeleDeTest(messages=iter([
            action("retriever_tool", {"question": "voyant orange papier", "image_ref": "img_px200_01"}),
            "Vérifiez le bac papier [PX200-T01, p. 4].",
        ]))
        res = graphe_agentique(modele, memoire=False).invoke(message("Voyant orange ?"))
        self.assertTrue(any(m.type == "tool" and "PX200-T01" in m.content for m in res["messages"]))
        self.assertIn("p. 4", res["messages"][-1].content)

        sans_appel = ModeleDeTest(messages=iter(["Réponse initiale sans source.", "Vérifiez le bac papier."]))
        res = graphe_agentique(sans_appel, memoire=False).invoke(message("Voyant img_px200_01 ?"))
        self.assertTrue(any(m.type == "tool" and "PX200-T01" in m.content for m in res["messages"]))

        mauvais_format = ModeleDeTest(messages=iter([
            action("retriever_tool", {"question": "voyant", "image_ref": {"reference": "img_px200_01"}}),
            "Vérifiez le bac papier.",
        ]))
        res = graphe_agentique(mauvais_format, memoire=False).invoke(message("Voyant img_px200_01 ?"))
        self.assertIn("[PX200-T01, p. 4]", res["messages"][-1].content)

    def test_workflow_n8n(self):
        workflow = json.loads(Path(__file__).with_name("n8n_sav_mcp.json").read_text())
        mcp = next(n for n in workflow["nodes"] if n["name"] == "MCP Client")
        self.assertEqual(mcp["parameters"]["serverTransport"], "httpStreamable")
        self.assertEqual(workflow["connections"]["MCP Client"]["ai_tool"][0][0]["node"], "Agent SAV")
        self.assertFalse(workflow["active"])


@unittest.skipUnless(os.getenv("TEST_MCP") == "1", "Démarrer sav.serveur_mcp et définir TEST_MCP=1")
class TestsHTTP(unittest.IsolatedAsyncioTestCase):
    async def test_outils_et_agent_mcp(self):
        sources = await appeler_mcp("rechercher_manuel", {"question": "voyant orange papier"})
        self.assertTrue(any(s["id"] == "PX200-T01" for s in sources))
        modele = ModeleDeTest(messages=iter([
            action("rag_multimodal", {"question": "Voyant de la photo img_px200_01 ?"}),
            "Figure du bac papier [PX200-T01, p. 4].",
        ]))
        agent = await agent_mcp(modele)
        sortie = await agent.ainvoke(message("Que signifie img_px200_01 ?"))
        observation = next(m for m in sortie["messages"] if m.type == "tool")
        self.assertEqual(observation.name, "rag_multimodal")
        self.assertNotEqual(observation.status, "error")
        document = json.loads(observation.content_blocks[0]["text"])
        self.assertTrue(any(s["id"] == "PX200-T01" for s in document["sources"]))

    async def test_routage_sav(self):
        entree = {"question": "Que signifie le pictogramme sur la photo de mon imprimante PX-200 ?",
                  "image_ref": "img_px200_01"}
        etapes = [etape async for etape in sav_graph.astream(entree, stream_mode="updates")]
        self.assertEqual([next(iter(e)) for e in etapes], ["orchestrateur", "agent_documentaire", "orchestrateur"])
        doc = etapes[1]["agent_documentaire"]
        self.assertIn("page", doc["sources"][0])
        self.assertIn("[PX200-T01, p. 4]", doc["reponse"])
        garantie = await sav_graph.ainvoke({"question": "Garantie du numéro de série PX200-0042 ?"})
        self.assertEqual((garantie["categorie"], garantie["agent"]), ("garantie", "agent_garantie"))
        self.assertIn("2027-08-14", garantie["reponse"])

    async def test_streamlit(self):
        from streamlit.testing.v1 import AppTest

        def interface():
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
            app.chat_input[0].set_value("Garantie du numéro de série PX200-0042 ?").run()
            app.selectbox[0].select("Voyant orange et feuille").run()
            app.chat_input[0].set_value("Quel est le pictogramme sur mon PX-200 ?").run()
            return app

        app = await asyncio.to_thread(interface)
        self.assertFalse(app.exception)
        self.assertTrue(any("2027-08-14" in m.value for m in app.markdown))
        self.assertTrue(any("PX200-T01" in m.value for m in app.markdown))


if __name__ == "__main__":
    unittest.main(verbosity=2)
