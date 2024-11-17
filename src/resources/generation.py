from typing import Dict, List, Union, Iterable, Optional
from pydantic import BaseModel

class Generation(BaseModel):
    text: str
    model: str
    best_of: Optional[int] = None
    echo: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[int] = None
    max_new_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str], None] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
