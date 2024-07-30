from typing import Dict, List

from pydantic import BaseModel, Field


class CityInfo(BaseModel):
    country: str = Field()
    population: int = Field()
    languages_spoken: List[str] = Field()


class ContextAndQuery(BaseModel):
    context: str = Field("", description="The context to extract information from")
    query: str = Field("", description="The query to answer")


class TwoCities(BaseModel):
    origin: str = Field()
    destination: str = Field()


class FlightRoute(BaseModel):
    airports: List[str] = Field()
    cost_of_flight: int = Field()


class Plan(BaseModel):
    steps: List[str] = Field()
    assumptions: List[str] = Field()


class NERInput(BaseModel):
    text: str = Field()
    entity_names: List[str] = Field()


class Entities(BaseModel):
    entities: Dict[str, str] = Field()


# Nested Model
class GoalAndPlan(BaseModel):
    goal: str = Field()
    plan: Plan = Field()


test_models: List[type[BaseModel]] = [
    CityInfo,
    ContextAndQuery,
    Entities,
    FlightRoute,
    GoalAndPlan,
    NERInput,
    TwoCities,
]
