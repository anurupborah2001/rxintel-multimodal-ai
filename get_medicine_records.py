try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass
import requests
import os
from typing import Optional, List, Dict, Any


def tavily_search(drug_name: str, max_results: int = 5) -> List[str]:
    """
    Query the Tavily Search API to retrieve relevant web links for a given drug.

    This function is typically used to gather external references
    (e.g., official documentation, trusted medical articles, or summaries)
    that can be cited or passed to an LLM for grounding.

    Args:
        drug_name (str): Name of the drug to search for.
        max_results (int): Maximum number of search results to return.

    Returns:
        List[str]: A list of formatted strings containing the title and URL
                    of each search result.
    """
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"}
    resp = requests.post(
        url, json={"query": drug_name, "num_results": max_results}, headers=headers
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        f"- {item.get('title','')} {item['url']}" for item in data.get("results", [])
    ]


# RXCUI = Concept Unique Identifier from the National Library of Medicine's (NLM) RxNorm system
def rxnav_lookup(drug_name: str) -> Dict[str, Any]:
    """
    Retrieve RxNorm metadata for a drug using the RxNav API.
    Get RxCUI and related ingredients/brands.

    This function resolves a drug name to its RxCUI and fetches:
    - Active ingredients (IN)
    - Brand names (BN)

    Args:
        drug_name (str): Drug name (brand or generic).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - rxcui (str | None)
            - ingredients (List[str])
            - brandNames (List[str])
    """
    encoded = requests.utils.quote(drug_name)
    base = "https://rxnav.nlm.nih.gov/REST"
    rxnorm = requests.get(f"{base}/rxcui.json?name={encoded}&search=2").json()
    rxcui = (rxnorm.get("idGroup", {}).get("rxnormId") or [None])[0]

    if not rxcui:
        return {"rxcui": None, "ingredients": [], "brandNames": []}

    ingredients = []
    ing = requests.get(f"{base}/rxcui/{rxcui}/related.json?tty=IN").json()
    for g in ing.get("relatedGroup", {}).get("conceptGroup", []):
        for c in g.get("conceptProperties", []):
            ingredients.append(c["name"])

    brand_names = []
    bn = requests.get(f"{base}/rxcui/{rxcui}/related.json?tty=BN").json()
    for g in bn.get("relatedGroup", {}).get("conceptGroup", []):
        for c in g.get("conceptProperties", []):
            brand_names.append(c["name"])

    return {"rxcui": rxcui, "ingredients": ingredients, "brandNames": brand_names}


def openfda_drug_label(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve structured FDA drug label information from OpenFDA.

    This function fetches official label sections such as indications,
    dosage, warnings, contraindications, and adverse reactions.

    Args:
        drug_name (str): Brand or generic drug name.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of drug label fields if found,
                                  otherwise None.
    """
    q = requests.utils.quote(
        f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"'
    )
    url = f"https://api.fda.gov/drug/label.json?search={q}&limit=1"
    try:
        data = requests.get(url).json()
        doc = data.get("results", [None])[0]
        if not doc:
            return None

        def pick(k):
            return "\n".join(doc[k]) if isinstance(doc.get(k), list) else doc.get(k)

        return {
            "indications": pick("indications_and_usage"),
            "dosage": pick("dosage_and_administration"),
            "warnings": pick("warnings") or pick("warnings_and_cautions"),
            "contraindications": pick("contraindications"),
            "adverse_reactions": pick("adverse_reactions"),
            "precautions": pick("precautions"),
            "patient_info": pick("information_for_patients")
            or pick("patient_medication_information"),
            "boxed_warning": pick("boxed_warning"),
        }
    except Exception:
        return None


def pubmed_summaries(drug_name: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Fetch publication summaries related to a drug from PubMed.

    Uses NCBI E-utilities:
    - esearch to retrieve PubMed IDs
    - esummary to fetch metadata for each article

    Args:
        drug_name (str): Drug name to search in PubMed.
        max_results (int): Maximum number of publications to return.

    Returns:
        List[Dict[str, str]]: A list of publication summaries containing
                              PMID, title, journal, and publication date.
    """
    esearch = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&retmax={max_results}&term={drug_name}&retmode=json"
    ).json()
    ids = esearch.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    id_str = ",".join(ids)
    esum = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={id_str}&retmode=json"
    ).json()
    result = esum.get("result", {})
    return [
        {
            "pmid": i,
            "title": result[i].get("title"),
            "journal": result[i].get("fulljournalname"),
            "pubdate": result[i].get("pubdate"),
        }
        for i in ids
        if i in result
    ]
