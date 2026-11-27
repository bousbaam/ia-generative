import base64
import binascii
import io
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer


MANUEL = [
    {
        "id": "PX200-T01", "page": 4, "type": "texte", "modele": "PX-200",
        "texte": "Voyant orange clignotant et pictogramme de feuille : vérifier le bac papier. Retirer le papier coincé sans forcer, remettre le bac et appuyer une fois sur Reprendre.",
    },
    {
        "id": "PX200-F01", "page": 4, "type": "figure", "modele": "PX-200",
        "texte": "Figure 1. Voyant orange clignotant avec une feuille : incident d’alimentation papier.",
        "image_ref": "img_px200_01",
    },
    {
        "id": "PX200-T02", "page": 6, "type": "texte", "modele": "PX-200",
        "texte": "Voyant rouge fixe et pictogramme de cartouche : éteindre l’imprimante et vérifier que la cartouche est correctement installée.",
    },
    {
        "id": "PX200-F02", "page": 6, "type": "figure", "modele": "PX-200",
        "texte": "Figure 2. Voyant rouge fixe avec une cartouche : vérifier la cartouche.",
        "image_ref": "img_px200_02",
    },
    {
        "id": "PX200-T03", "page": 8, "type": "texte", "modele": "PX-200",
        "texte": "Voyant bleu clignotant et pictogramme Wi-Fi : l’imprimante attend l’association réseau. Depuis le menu Réseau, sélectionner le Wi-Fi du domicile.",
    },
    {
        "id": "PX200-F03", "page": 8, "type": "figure", "modele": "PX-200",
        "texte": "Figure 3. Voyant bleu avec un symbole Wi-Fi : association au réseau en attente.",
        "image_ref": "img_px200_03",
    },
]


@lru_cache
def illustration(reference):
    """Image synthétique du manuel de démonstration, encodée en PNG."""
    couleurs = {"img_px200_01": "#f59e0b", "img_px200_02": "#dc2626", "img_px200_03": "#2563eb"}
    if reference not in couleurs:
        raise ValueError("Référence d’image inconnue.")
    image = Image.new("RGB", (300, 160), "#e8eef5")
    dessin = ImageDraw.Draw(image)
    dessin.rounded_rectangle((15, 18, 285, 142), radius=15, fill="#182437")
    dessin.ellipse((40, 45, 110, 115), fill=couleurs[reference])
    if reference == "img_px200_01":
        dessin.rectangle((165, 40, 215, 120), outline="white", width=5)
        dessin.line((176, 65, 202, 65), fill="white", width=4)
        dessin.line((176, 81, 202, 81), fill="white", width=4)
    elif reference == "img_px200_02":
        dessin.rounded_rectangle((155, 50, 226, 111), radius=5, outline="white", width=5)
        dessin.rectangle((171, 64, 208, 98), fill="white")
    else:
        for rayon in (20, 36, 52):
            dessin.arc((193-rayon, 100-rayon, 193+rayon, 100+rayon), 220, 320, fill="white", width=5)
        dessin.ellipse((189, 99, 197, 107), fill="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def descripteur_visuel(data):
    """Vecteur de couleurs et position sur un petit quadrillage RGB."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > 8_000_000:
                raise ValueError("L’image dépasse 8 mégapixels.")
            pixels = np.asarray(ImageOps.exif_transpose(img).convert("RGB").resize((12, 8)), dtype=float) / 255
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("L’image ne peut pas être lue.") from exc
    # La composante chromatique réduit l'influence des marges grises communes.
    pixels = pixels - pixels.mean(axis=-1, keepdims=True)
    return pixels.ravel()


@lru_cache
def index_manuel():
    vecteur = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    textes = vecteur.fit_transform([document["texte"] for document in MANUEL])
    images = {
        i: descripteur_visuel(illustration(doc["image_ref"]))
        for i, doc in enumerate(MANUEL) if doc["type"] == "figure"
    }
    return vecteur, textes, images


def rechercher(question, image_ref=None, image_base64=None):
    if image_ref and image_base64:
        raise ValueError("Choisir une référence ou un fichier image.")
    if not image_ref and not image_base64:
        reference = re.search(r"\bimg_px200_\d{2}\b", question, re.I)
        if reference:
            image_ref = reference.group().lower()
            if not any(doc.get("image_ref") == image_ref for doc in MANUEL):
                return []
    if re.search(r"\b[A-Z]{2,4}[- ]?\d{3,5}\b", question, re.I) and not re.search(r"\bPX[- ]?200\b", question, re.I):
        return []
    vecteur, textes, images = index_manuel()
    scores = (textes @ vecteur.transform([question]).T).toarray().ravel()
    photo = None
    if image_ref:
        photo = illustration(image_ref)
    if image_base64:
        if len(image_base64) > 3_000_000:
            raise ValueError("Le fichier image est trop volumineux.")
        try:
            photo = base64.b64decode(image_base64, validate=True)
        except binascii.Error as exc:
            raise ValueError("Encodage de l’image invalide.") from exc
    if photo:
        entree = descripteur_visuel(photo)
        similarites = {i: float(np.linalg.norm(entree-ref)) for i, ref in images.items()}
        meilleur = min(similarites, key=similarites.get)
        if similarites[meilleur] > 1.25:
            return []
        scores[meilleur] += 1.0
        for i, doc in enumerate(MANUEL):
            if doc["page"] == MANUEL[meilleur]["page"] and doc["type"] == "texte":
                scores[i] += 0.85
    if scores.max() < 0.15:
        return []
    classement = np.argsort(-scores, kind="stable")
    retenus = [i for i in classement if scores[i] >= 0.15][:2]
    return [dict(MANUEL[i], score=round(float(scores[i]), 3)) for i in retenus]


def synthese(sources):
    if not sources:
        return "Le manuel de démonstration ne permet pas de confirmer ce diagnostic."
    return "\n\n".join(f"{s['texte']} [{s['id']}, p. {s['page']}]" for s in sources)
