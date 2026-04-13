from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

output = Path(r"D:\IA_app\outputs\etude_de_cas_vehicule_autonome.docx")

title = "Etude de cas 1 - Vehicule autonome et accident mortel"
sections = [
    ("Introduction", [
        "L'intelligence artificielle appliquee aux vehicules autonomes represente un domaine prometteur mais sensible, car les decisions du systeme peuvent avoir un impact direct sur la vie humaine.",
        "Le cas etudie porte sur un vehicule autonome de niveau 4 implique dans un accident mortel lors d'un test en conditions reelles, avec detection tardive du pieton, freinage desactive et conducteur de securite inattentif.",
        "L'objectif de ce rapport est d'analyser le systeme sous les angles technique, ethique, reglementaire et industriel, puis de proposer des recommandations pour un deploiement responsable."
    ]),
    ("1. Presentation du systeme", [
        "Le vehicule autonome de niveau 4 est capable d'assurer lui-meme la conduite dans un domaine operationnel defini, avec un humain eventuellement present en supervision.",
        "Le pipeline repose sur des capteurs camera, LiDAR et radar, un modele de perception base sur un CNN, une fusion de capteurs, puis un systeme de planification, de decision et de controle.",
        "L'accident revele une defaillance systemique impliquant des composants techniques, des choix d'ingenierie, des arbitrages industriels et des insuffisances de gouvernance."
    ]),
    ("2. Analyse technique", [
        "La detection tardive du pieton indique une faiblesse du module de perception, soit par mauvaise qualite du signal, soit par manque de robustesse du modele en faible luminosite.",
        "La fusion de capteurs aurait du compenser les limites propres a chaque capteur. Son echec suggere une mauvaise ponderation des signaux ou une integration logicielle insuffisante.",
        "Le freinage desactive en mode test a retire une barriere de securite essentielle, alors qu'un systeme critique devrait au contraire renforcer ses protections.",
        "Le dataset, majoritairement urbain de jour, introduit un biais de couverture. Les situations nocturnes et pluvieuses impliquant des pietons etaient sous-representees.",
        "L'accuracy globale de 97 % ne suffit pas. Dans un contexte de securite, il faut privilegier le recall des pietons, les faux negatifs en conditions degradees et le temps de detection avant collision."
    ]),
    ("3. Analyse ethique", [
        "L'entreprise porte une responsabilite majeure, car elle choisit les conditions de test, les dispositifs de securite et le niveau de validation avant experimentation sur route ouverte.",
        "Les ingenieurs doivent documenter les limites du systeme, signaler les scenarios de risque et refuser des conditions de deploiement insuffisamment sures.",
        "Le conducteur de securite a contribue a l'accident, mais son inattention s'inscrit dans une organisation reposant sur une vigilance humaine passive peu realiste.",
        "Le test en conditions reelles parait difficilement acceptable sur le plan ethique, car le public a ete expose a un risque insuffisamment maitrise."
    ]),
    ("4. Analyse reglementaire", [
        "Au regard de l'AI Act, un vehicule autonome embarquant de l'IA pour percevoir l'environnement et prendre des decisions releve d'une logique de systeme a haut risque.",
        "Les obligations pertinentes comprennent la gestion des risques, la gouvernance des donnees, la documentation technique, les logs, la surveillance humaine et la robustesse.",
        "Le RGPD s'applique aussi lorsque les capteurs collectent des donnees permettant d'identifier directement ou indirectement des personnes, comme des visages, plaques ou trajectoires.",
        "La tracabilite des decisions est indispensable pour attribuer les responsabilites en cas d'accident."
    ]),
    ("5. Analyse industrielle", [
        "Le secteur du vehicule autonome subit une forte pression de mise sur le marche, ce qui peut encourager des tests trop precoces.",
        "Un accident mortel entraine un cout humain irreparable, mais aussi des couts economiques, juridiques et reputionnels tres lourds.",
        "Ce cas illustre une dette technique et ethique accumulee lorsqu'une entreprise reporte les investissements necessaires en validation, gouvernance et securite."
    ]),
    ("6. Recommandations pour un deploiement responsable", [
        "Il faut enrichir massivement le dataset avec des scenarios critiques : nuit, pluie, faible visibilite, pietons inattendus, vetements sombres et situations ambigues.",
        "Les criteres de validation doivent privilegier le recall des pietons, les faux negatifs, le temps de reaction, la distance d'arret et la performance sur les cas rares mais critiques.",
        "Le freinage d'urgence ne devrait jamais etre neutralise en environnement ouvert au public.",
        "La supervision humaine doit etre repensee avec des protocoles realistes, des alertes actives, une formation adaptee et une interface claire.",
        "Le deploiement doit etre progressif, de la simulation a des tests publics limites, chaque etape etant conditionnee par des preuves solides de surete."
    ]),
    ("Conclusion", [
        "L'accident resulte d'un enchainement de defaillances techniques, organisationnelles et ethiques, et non d'une seule erreur isolee.",
        "Un vehicule autonome ne devrait etre autorise en conditions reelles que s'il demontre de maniere robuste sa capacite a proteger les personnes les plus vulnerables dans le respect d'une IA responsable."
    ]),
    ("References", [
        "Support de cours IA Responsable, etude de cas 1, objectifs, pipeline ethique, principes et livrables.",
        "Reglement (UE) 2024/1689 sur l'intelligence artificielle (AI Act).",
        "Reglement (UE) 2016/679 (RGPD / GDPR)."
    ]),
]


def p(text, style=None):
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{props}<w:r><w:rPr><w:lang w:val="fr-FR"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


body = [p(title, "Title")]
for heading, paragraphs in sections:
    body.append(p(heading, "Heading1"))
    for text in paragraphs:
        body.append(p(text))

document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" mc:Ignorable="w14 wp14">
<w:body>{''.join(body)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''

styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:lang w:val="fr-FR"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
</w:styles>'''

core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>Etude de cas 1 - Vehicule autonome</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">2026-03-23T00:00:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2026-03-23T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''

app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application></Properties>'''

output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types)
    zf.writestr("_rels/.rels", root_rels)
    zf.writestr("word/document.xml", document)
    zf.writestr("word/_rels/document.xml.rels", doc_rels)
    zf.writestr("word/styles.xml", styles)
    zf.writestr("docProps/core.xml", core)
    zf.writestr("docProps/app.xml", app)

print(output)
