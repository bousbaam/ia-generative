from sav.graphes import graphe_agentique, graphe_bases, rag_chatbot
from sav.langchain_demo import modele_local

bases = graphe_bases()
rag = rag_chatbot
agentique = graphe_agentique(modele_local(), memoire=False)
