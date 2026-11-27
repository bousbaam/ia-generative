"""Registre fictif immuable, pour la démonstration SAV."""

import re
from datetime import date
from types import MappingProxyType


REGISTRE = MappingProxyType({
    "PX200-0042": ("PX-200", date(2027, 8, 14)),
    "PX200-0099": ("PX-200", date(2025, 6, 19)),
})


def consulter_garantie(numero_serie, au=None):
    numero = numero_serie.strip().upper()
    if not re.fullmatch(r"PX200-\d{4}", numero):
        return {"statut": "invalide"}
    fiche = REGISTRE.get(numero)
    if fiche is None:
        return {"statut": "introuvable"}
    modele, fin = fiche
    jour = au or date.today()
    return {
        "statut": "active" if jour <= fin else "expiree",
        "numero_serie": numero, "modele": modele,
        "fin": fin.isoformat(), "date_consultation": jour.isoformat(),
    }
