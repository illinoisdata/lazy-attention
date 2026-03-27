import json
import requests

blocks = [
    "<|user|>\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n",
    "- Title: Polish-Russian War (film)\nPolish-Russian War(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery \u017bu\u0142awski based on the novel Polish-Russian War under the white-red flag by Dorota Mas\u0142owska.\n",
    "- Title: Xawery \u017bu\u0142awski\nXawery \u017bu\u0142awski (born 22 December 1971 in Warsaw) is a Polish film director.In 1995 he graduated National Film School in \u0141\u00f3d\u017a.He is the son of actress Ma\u0142gorzata Braunek and director Andrzej \u017bu\u0142awski.His second feature \"Wojna polsko-ruska\" (2009), adapted from the controversial best-selling novel by Dorota Mas\u0142owska, won First Prize in the New Polish Films competition at the 9th Era New Horizons Film Festival in Wroc\u0142aw.In 2013, he stated he intends to direct a Polish novel \"Z\u0142y\" by Leopold Tyrmand.\u017bu\u0142awski and his wife Maria Strzelecka had 2 children together:son Kaj \u017bu\u0142awski (born 2002) and daughter Jagna \u017bu\u0142awska (born 2009).\n",
    "- Title: Viktor Yeliseyev\nViktor Petrovich Yeliseyev( born June 9, 1950) is a Russian general, orchestra conductor and music teacher.He is the director of the Ministry of the Interior Ensemble, one of the two Russian Red Army Choirs.\n- Title: Minamoto no Chikako\nShe was the mother of Prince Morinaga.\n- Title: Alice Washburn\nAlice Washburn( 1860- 1929) was an American stage and film actress.She worked at the Edison, Vitagraph and Kalem studios.Her final film Snow White was her only known feature film.She died of heart attack in November 1929.\n",
    "Please write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant).\nQuestion: Who is the mother of the director of film Polish-Russian War (Film)?\n<|assistant|>\n"
]

r = requests.post(
    url="http://127.0.0.1:9898/generate",
    data=json.dumps({"blocks": blocks}),
    headers={"Content-Type": "application/json"}
)

if not r.ok:
    print(f"Request failed: HTTP {r.status_code}")
    print("Response body:", r.text)
else:
    try:
        print(r.json()["generated"])
        print("Input IDs:", r.json()["input_ids"])
        print("Decoded Input:", r.json()["decoded_input"])
    except Exception:
        print("Failed to decode JSON response:")
        print(r.text)