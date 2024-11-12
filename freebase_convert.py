from freebasetowiki.converter import EntityConverter

entity_converter = EntityConverter("https://query.wikidata.org/sparql")
id = entity_converter.get_wikidata_id("/m/0j_6gm")  # 'Q15978631'
print(id)