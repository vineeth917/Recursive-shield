from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Constraint(BaseModel):
    max_position_pct: Optional[float] = 5.0
    allowed_universe: Optional[List[str]] = Field(default_factory=lambda: ["large-caps"])
    require_confirmation_above: Optional[float] = None
    forbid_unverified_tips: bool = True
    ask_before_order: bool = True

class OracleLabel(BaseModel):
    label: str  # "safe" or "unsafe"
    violated_constraints: List[str] = Field(default_factory=list)
    forbidden_tool_fired: bool = False
    induced_by_injection: bool = False

class DeliverySpec(BaseModel):
    carrier_type: str = "fomc_audio"
    insertion_timestamp_sec: float
    voice: str = "minimax_voice_id"
    mix_strategy: str = "splice"
    volume_db: float = 0.0

class TargetViolationSpec(BaseModel):
    type: str = "forbidden_trade"
    expected_tool: str = "confirm_order"
    expected_symbol: Optional[str] = None
    reason: str

class AttackSpec(BaseModel):
    attack_id: str
    family: str
    level: str  # "L0" | "L1"
    payload_text: str
    delivery: DeliverySpec
    target_violation: TargetViolationSpec
    audio_path: str
    seed: str
    source_transcript_id: str
    clean_audio_path: str
    clean_baseline_trace_id: Optional[str] = None
    notes: Optional[str] = None

class ActionRecord(BaseModel):
    tool: str
    args: Dict[str, Any]
    timestamp: float
    intent: Optional[str] = None

class Trace(BaseModel):
    run_id: str
    audio_path: str
    transcript: str
    policy: Constraint
    actions: List[ActionRecord] = Field(default_factory=list)
    final_action: Optional[str] = None
    native_defense_outcome: Optional[str] = None  # e.g., "blocked", "passed"
    oracle: Optional[OracleLabel] = None
