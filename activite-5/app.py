import asyncio
import base64
import json
import uuid

import streamlit as st

from sav.documentation import illustration
from sav.graphes import graphe_agentique
from sav.langchain_demo import modele_local
from sav.orchestrateur import sav_graph


st.set_page_config(page_title="SAV PX-200", page_icon="🛠️")
st.title("Assistance PX-200")
st.caption("Démonstration pédagogique avec un manuel et un registre fictifs.")
mode = st.sidebar.radio("Parcours", ["SAV multimodal et garantie", "Chatbot agentique LangGraph"])
options = {"Aucune": None, "Voyant orange et feuille": "img_px200_01",
           "Voyant rouge et cartouche": "img_px200_02", "Voyant bleu et Wi-Fi": "img_px200_03"}
choix = st.sidebar.selectbox("Image de démonstration", list(options))
reference = options[choix]
fichier = None
if mode == "SAV multimodal et garantie":
    fichier = st.sidebar.file_uploader("Ou joindre une image", type=["png", "jpg", "jpeg"])
if reference and not fichier:
    st.sidebar.image(illustration(reference), caption=reference)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.graphe = None

if st.sidebar.button("Nouvelle conversation"):
    st.session_state.messages = []
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.graphe = None
    st.rerun()


def afficher(sortie):
    st.markdown(sortie["reponse"])
    if sortie.get("agent"):
        st.caption(f"{sortie['categorie']} · {sortie['agent']}")
    for source in sortie.get("sources", []):
        st.caption(f"{source['id']} · p. {source['page']} · {source['type']}")
        if source.get("image_ref"):
            st.image(illustration(source["image_ref"]), width=230)


for role, item in st.session_state.messages:
    with st.chat_message(role):
        if role == "user":
            st.write(item)
        else:
            afficher(item)

question = st.chat_input("Ex. Que signifie le voyant de mon imprimante ?")
if question:
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"), st.spinner("Analyse de la demande…"):
        try:
            if mode == "SAV multimodal et garantie":
                entree = {"question": question}
                if fichier:
                    if fichier.size > 2_000_000:
                        raise ValueError("Choisis une image de 2 Mo maximum.")
                    entree["image_base64"] = base64.b64encode(fichier.getvalue()).decode()
                elif reference:
                    entree["image_ref"] = reference
                resultat = asyncio.run(sav_graph.ainvoke(entree))
            else:
                if st.session_state.graphe is None:
                    st.session_state.graphe = graphe_agentique(modele_local())
                texte = question
                if reference and reference not in texte.lower():
                    texte += f" Référence image : {reference}."
                reponse = st.session_state.graphe.invoke(
                    {"messages": [{"role": "user", "content": texte}]},
                    {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 12},
                )
                messages = reponse["messages"]
                sources = {}
                for message in reversed(messages[:-1]):
                    if message.type == "human":
                        break
                    if message.type == "tool" and message.status == "success":
                        for source in json.loads(message.content):
                            sources[source["id"]] = source
                resultat = {
                    "reponse": messages[-1].content or "Le modèle n’a pas fourni de réponse.",
                    "sources": list(sources.values()),
                }
        except ValueError as erreur:
            resultat = {"reponse": str(erreur)}
        except Exception:
            resultat = {"reponse": "Le service est indisponible. Vérifie que le serveur MCP ou Ollama est démarré pour ce parcours."}
        afficher(resultat)
    st.session_state.messages.append(("assistant", resultat))
