# chains/medicine_chain.py
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from schema import DrugInfo
from get_medicine_records import (
    tavily_search,
    rxnav_lookup,
    openfda_drug_label,
    pubmed_summaries,
)

# System-level instruction for the LLM
# Enforces medical safety, source prioritization, and strict JSON output
SYSTEM_PROMPT = """
You are a careful medical information assistant.
Summarize trustworthy drug information from provided tool outputs.
- Prefer FDA, NIH, NHS, PubMed.
- If a field is unknown, omit it.
- NEVER give personalized medical advice.
Output JSON matching this schema:
{schema}
"""

# User-level prompt that injects tool outputs into the model context
# Each section corresponds to a structured external data source
USER_PROMPT = """
Query drug name: {drug}

[TAVILY_TOP_LINKS]
{tavily}

[RXNAV]
{rxnav}

[OPEN_FDA_LABEL]
{openfda}

[PUBMED_SUMMARIES]
{pubmed}

Return ONLY JSON.
"""


def build_chain():
    """
    Construct and return the LLM inference chain.

    This chain combines:
    - A safety-focused system prompt
    - Structured medical data from multiple trusted sources
    - A strict JSON schema constraint

    The model is configured with low temperature to reduce hallucinations
    and improve determinism for medical summaries.

    Returns:
        LLMChain: A LangChain pipeline ready for execution.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT + USER_PROMPT,
        input_variables=[
            "drug",
            "tavily",
            "rxnav",
            "openfda",
            "pubmed",
            "schema",
        ],
    )

    return LLMChain(llm=llm, prompt=prompt)


def extract_json(text: str) -> dict:
    """
    Extract and parse JSON from LLM output.

    This function strips optional Markdown code fences (``` or ```json)
    that may be included by the model and safely parses the remaining JSON.

    Args:
        text (str): Raw LLM output.

    Returns:
        dict: Parsed JSON object.

    Raises:
        json.JSONDecodeError: If the cleaned output is not valid JSON.
    """
    # Remove markdown fences if present
    cleaned = re.sub(
        r"^```(?:json)?|```$",
        "",
        text.strip(),
        flags=re.MULTILINE,
    ).strip()

    return json.loads(cleaned)


def get_medicine_info(drug_name: str) -> DrugInfo:
    """
    Retrieve, normalize, and summarize trusted medical information
    for a given drug using a multimodal RAG pipeline.

    Data sources:
    - RxNav (RxNorm identifiers, ingredients, brand names)
    - OpenFDA (official drug label sections)
    - PubMed (recent scientific literature)
    - Tavily (trusted external references)

    The aggregated data is passed to an LLM constrained by a strict
    medical-safety prompt and validated against the DrugInfo schema.

    Args:
        drug_name (str): Drug name (brand or generic).

    Returns:
        DrugInfo: Structured, validated drug information object.

    Raises:
        ValueError: If the LLM returns invalid or non-conforming JSON.
    """
    # Fetch structured data from external medical sources
    rxnav = rxnav_lookup(drug_name)
    fda = openfda_drug_label(drug_name)
    pubs = pubmed_summaries(drug_name)
    tavily_links = tavily_search(drug_name)

    # Build LLM inference chain
    chain = build_chain()

    # Execute LLM with fully grounded medical context
    output = chain.run(
        drug=drug_name,
        tavily="\n".join(tavily_links),
        rxnav=json.dumps(rxnav, indent=2),
        openfda=json.dumps(fda or {}, indent=2),
        pubmed=json.dumps(pubs, indent=2),
        schema=DrugInfo.schema_json(indent=2),
    )

    try:
        # Parse and validate model output against Pydantic schema
        parsed = extract_json(output)
        drug_info = DrugInfo(**parsed)
        print(drug_info)
        return drug_info
    except Exception:
        # Fail fast if the model violates schema or formatting constraints
        raise ValueError("Invalid JSON output from LLM:\n" + output)
