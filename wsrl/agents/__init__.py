from .bc import BCAgent
from .btccq import BTCCQAgent
from .calql import CalQLAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .sac import SACAgent

agents = {
    "bc": BCAgent,
    "btccq": BTCCQAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
}
