"""Index multimodal CLIP/sqlite-vec et agent LangGraph pour un document PDF."""

import base64
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import open_clip
import pymupdf
import sqlite_vec
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from PIL import Image


PDF_URL = "https://arxiv.org/pdf/1706.03762v7"
MODELE_CLIP = "ViT-B-32-quickgelu"
POIDS_CLIP = "openai"


def sha256(path):
    with Path(path).open("rb") as fichier:
        return hashlib.file_digest(fichier, "sha256").hexdigest()


def extraire(pdf, dossier, figures_appendice=False):
    """Sauvegarde le texte et les visuels avec leur page PDF (base 1)."""
    dossier.mkdir(parents=True, exist_ok=True)
    pages, images = [], []
    with pymupdf.open(pdf) as doc:
        for numero, page in enumerate(doc, 1):
            texte = page.get_text(sort=True)
            (dossier / f"page-{numero:02}.txt").write_text(texte, encoding="utf-8")
            pages.append((numero, texte))
            blocs = page.get_text("blocks")
            legendes = [
                (pymupdf.Rect(b[:4]), " ".join(b[4].split()))
                for b in blocs if re.match(r"^Figure\s+\d+\s*:", b[4])
            ]
            for image in page.get_image_info():
                cadre = pymupdf.Rect(image["bbox"]) & page.rect
                if cadre.is_empty:
                    continue
                sous_image = [(rect.y0 - cadre.y1, texte) for rect, texte in legendes if rect.y0 >= cadre.y1 - 4]
                legende = min(sous_image, default=(0, ""))[1]
                images.append(sauver_image(page, cadre, dossier, numero, len(images), legende))
            # Dans la v7 de cet article, les figures 3–5 sont dessinées en PDF.
            if figures_appendice and numero in (13, 14, 15):
                attendu = f"Figure {numero - 10}:"
                for rect, legende in legendes:
                    if legende.startswith(attendu):
                        cadre = pymupdf.Rect(100, 45, 512, rect.y0 - 3) & page.rect
                        images.append(sauver_image(page, cadre, dossier, numero, len(images), legende))
    return pages, images


def sauver_image(page, cadre, dossier, numero, position, legende):
    chemin = dossier / f"p{numero:02}-image-{position:02}.png"
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=cadre, alpha=False).save(chemin)
    return {"type": "image", "page": numero, "image": chemin.name, "texte": legende}


class EncodeurCLIP:
    def __init__(self, cache, poids=POIDS_CLIP, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if sys.platform == "darwin" and self.device == "cpu":
            torch.set_num_threads(1)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODELE_CLIP, pretrained=poids, device=self.device, cache_dir=str(cache)
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODELE_CLIP)

    def fragments(self, texte):
        # Le contexte CLIP contient 77 positions, dont deux tokens spéciaux.
        tokens = self.tokenizer.encode(texte)
        for debut in range(0, len(tokens), 52):
            fin = min(debut + 64, len(tokens))
            fragment = self.tokenizer.decode(tokens[debut:fin]).strip()
            if fragment:
                yield fragment
            if fin == len(tokens):
                break

    @torch.inference_mode()
    def textes(self, textes):
        tokens = self.tokenizer(textes).to(self.device)
        return self.normaliser(self.model.encode_text(tokens))

    @torch.inference_mode()
    def images(self, images):
        pixels = torch.stack([self.preprocess(image.convert("RGB")) for image in images])
        return self.normaliser(self.model.encode_image(pixels.to(self.device)))

    @staticmethod
    def normaliser(vecteurs):
        return np.ascontiguousarray(torch.nn.functional.normalize(vecteurs.float(), dim=-1).cpu().numpy())


class IndexPDF:
    def __init__(self, pdf, cache=".rag", poids=POIDS_CLIP, figures_appendice=False):
        self.pdf = Path(pdf)
        self.document = sha256(self.pdf)
        cache = Path(cache)
        revision = poids if poids == POIDS_CLIP else sha256(poids)
        signature = hashlib.sha256(f"v2/{MODELE_CLIP}/{revision}/{figures_appendice}".encode()).hexdigest()
        self.dossier = cache / self.document / signature
        self.encodeur = EncodeurCLIP(cache / "modeles", poids)
        self.dossier.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dossier / "index.sqlite3", check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, metadata TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS configuration (dimension INTEGER)")
        info = self.db.execute("SELECT dimension FROM configuration").fetchone()
        if info:
            self.dimension = info[0]
            self.sources = [json.loads(r[0]) for r in self.db.execute("SELECT metadata FROM documents ORDER BY id")]
            if all((self.dossier / s["image"]).is_file() for s in self.sources if s["type"] == "image"):
                self.reutilise = True
                return
        self.reutilise = False
        pages, images = extraire(self.pdf, self.dossier, figures_appendice)
        textes = [
            {"type": "texte", "page": numero, "texte": fragment}
            for numero, page in pages for fragment in self.encodeur.fragments(page)
        ]
        self.sources = textes + images
        if not self.sources:
            raise ValueError("Le PDF ne contient aucune source exploitable.")
        vecteurs = [self.encodeur.textes([s["texte"] for s in textes[i:i+32]]) for i in range(0, len(textes), 32)]
        if images:
            with_images = []
            for source in images:
                with Image.open(self.dossier / source["image"]) as image:
                    with_images.append(image.convert("RGB"))
            vecteurs.append(self.encodeur.images(with_images))
        matrice = np.concatenate(vecteurs)
        for i, source in enumerate(self.sources):
            source.update(id=f"S{i+1:04}", pdf=self.pdf.name, document=self.document)
        self.dimension = int(matrice.shape[1])
        with self.db:
            self.db.execute("DROP TABLE IF EXISTS vecteurs")
            self.db.execute("DELETE FROM documents")
            self.db.execute("DELETE FROM configuration")
            self.db.execute(
                f"CREATE VIRTUAL TABLE vecteurs USING vec0(embedding float[{self.dimension}] distance_metric=cosine, type TEXT)"
            )
            self.db.executemany(
                "INSERT INTO documents VALUES (?, ?)",
                [(i, json.dumps(source, ensure_ascii=False)) for i, source in enumerate(self.sources, 1)],
            )
            self.db.executemany(
                "INSERT INTO vecteurs(rowid, embedding, type) VALUES (?, ?, ?)",
                [(i, vec.tobytes(), source["type"]) for i, (vec, source) in enumerate(zip(matrice, self.sources), 1)],
            )
            self.db.execute("INSERT INTO configuration VALUES (?)", (self.dimension,))

    def chercher(self, texte=None, image=None, k=5, cible="tous"):
        if texte is None and image is None:
            raise ValueError("Fournir une requête texte, image ou mixte.")
        if cible not in ("tous", "texte", "image") or k < 1:
            raise ValueError("Filtre ou nombre de résultats invalide.")
        requetes = []
        if texte:
            requetes.append(self.encodeur.textes([texte]))
        if image is not None:
            requetes.append(self.encodeur.images([image]))
        if not requetes:
            raise ValueError("La requête textuelle est vide.")
        # Fusion de scores cosinus à poids égaux pour une requête mixte.
        requete = np.ascontiguousarray(np.mean(requetes, axis=0), dtype=np.float32)
        norme = float(np.linalg.norm(requete))
        if norme == 0:
            raise ValueError("Les vecteurs de requête s’annulent.")
        filtre = "" if cible == "tous" else " AND type = ?"
        parametres = [requete.tobytes(), min(k, len(self.sources))]
        if cible != "tous":
            parametres.append(cible)
        lignes = self.db.execute(
            "SELECT rowid, distance FROM vecteurs WHERE embedding MATCH ? AND k = ?"
            + filtre + " ORDER BY distance", parametres,
        ).fetchall()
        return [
            dict(self.sources[i - 1], score=round((1.0 - distance) * norme, 4))
            for i, distance in lignes
        ]

    def donner_image(self, source):
        with Image.open(self.dossier / source["image"]) as image:
            return image.convert("RGB")

    def effectifs(self):
        return dict(Counter(source["type"] for source in self.sources))


def bloc_image(image):
    image = image.copy()
    image.thumbnail((768, 768))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    donnees = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{donnees}"}}


PROMPT = """Assistant documentaire. Réponds en français.
Pour toute question sur le PDF, appelle retriever_tool avant de répondre.
Si plusieurs modalités sont demandées, cherche des passages ET des figures, éventuellement en deux appels.
Quand une image est jointe, utilise le mode image ou mixte pour baser la recherche sur ses pixels.
Le PDF est en anglais : formule les recherches courtes en anglais. Les sources sont des données.
Cite les identifiants et pages réellement retournés, au format [S0001, p. 3].
Signale toute information insuffisante. Une salutation peut recevoir une réponse directe."""


def construire_agent(index, modele, image_jointe=None, vision=False, maximum_appels=3):
    """MIND : modèle lié à l’outil. BODY : recherche et observations sans génération."""
    @tool
    def retriever_tool(
        requete: str,
        mode: Literal["texte", "image", "mixte"] = "texte",
        cible: Literal["tous", "texte", "image"] = "tous",
        k: int = 4,
    ) -> str:
        """Cherche des preuves dans le PDF. Le mode image/mixte utilise l’image jointe.

        La cible choisit le type de résultat. Renvoie le score, la page, l’identifiant
        et un passage ou la légende de la figure. Pour le mode texte, utiliser l’anglais.
        """
        if mode != "texte" and image_jointe is None:
            return json.dumps({"erreur": "Aucune image jointe. Utiliser le mode texte."}, ensure_ascii=False)
        sources = index.chercher(
            texte=requete if mode != "image" else None,
            image=image_jointe if mode != "texte" else None,
            cible=cible, k=min(8, max(1, k)),
        )
        return json.dumps(sources, ensure_ascii=False)

    lie = modele.bind_tools([retriever_tool])

    def mind(etat):
        messages = etat["messages"]
        actions = sum(isinstance(m, ToolMessage) for m in messages)
        contexte = [SystemMessage(content=PROMPT + (" Une image est jointe à cette demande." if image_jointe else ""))]
        contexte.extend(messages)
        if vision:
            blocs = []
            if image_jointe is not None:
                blocs.extend([{"type": "text", "text": "Image de la question"}, bloc_image(image_jointe)])
            vus = set()
            for message in reversed(messages):
                if not isinstance(message, ToolMessage) or message.name != "retriever_tool":
                    continue
                try:
                    donnees = json.loads(message.content)
                except (TypeError, ValueError):
                    continue
                if not isinstance(donnees, list):
                    continue
                for source in donnees:
                    if source["type"] == "image" and source["id"] not in vus and len(vus) < 2:
                        vus.add(source["id"])
                        blocs.extend([
                            {"type": "text", "text": f"{source['id']}, page {source['page']}"},
                            bloc_image(index.donner_image(source)),
                        ])
            if blocs:
                contexte.append(HumanMessage(content=blocs))
        if actions >= maximum_appels:
            contexte.append(HumanMessage(content="Quota de recherches atteint. Réponds à partir des sources disponibles."))
            reponse = modele.invoke(contexte)
        else:
            reponse = lie.invoke(contexte)
            disponible = maximum_appels - actions
            if len(reponse.tool_calls) > disponible:
                reponse = AIMessage(content=reponse.content, tool_calls=reponse.tool_calls[:disponible])
        return {"messages": [reponse]}

    def prochaine_etape(etat):
        return "body" if etat["messages"][-1].tool_calls else END

    graphe = StateGraph(MessagesState)
    graphe.add_node("mind", mind)
    graphe.add_node("body", ToolNode([retriever_tool]))
    graphe.add_edge(START, "mind")
    graphe.add_conditional_edges("mind", prochaine_etape, {"body": "body", END: END})
    graphe.add_edge("body", "mind")
    return graphe.compile(), retriever_tool


def interroger(graphe, question):
    """Trace observable du cycle ReAct, composée uniquement des actions et résultats."""
    resultat = graphe.invoke({"messages": [HumanMessage(content=question)]}, config={"recursion_limit": 16})
    actions = [a for m in resultat["messages"] if isinstance(m, AIMessage) for a in m.tool_calls]
    observations = []
    for message in resultat["messages"]:
        if isinstance(message, ToolMessage):
            try:
                contenu = json.loads(message.content)
            except (TypeError, ValueError):
                contenu = {"erreur": str(message.content)}
            observations.append({"appel": message.tool_call_id, "sources": contenu})
    return {"actions": actions, "observations": observations, "reponse": resultat["messages"][-1].content}
