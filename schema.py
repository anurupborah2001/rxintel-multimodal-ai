# schema.py
from pydantic import BaseModel
from typing import List, Optional


class DrugInfo(BaseModel):
    query: str
    normalized_name: Optional[str] = None
    rxcui: Optional[str] = None
    composition: Optional[List[str]] = None
    indications: Optional[str] = None
    dosage: Optional[str] = None
    instructions: Optional[str] = None
    warnings: Optional[str] = None
    contraindications: Optional[str] = None
    side_effects: Optional[str] = None
    precautions: Optional[str] = None
    suggestions: Optional[str] = None
    sources: Optional[List[str]] = None
