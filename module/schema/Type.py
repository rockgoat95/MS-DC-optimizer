from typing import Tuple
from pydantic import BaseModel
import numpy as np


class Ability(BaseModel):
    main_stat: int
    sub_stat: int
    damage: int
    boss_damage : int
    att_p : int
    defense_ignore : int
    critical_damage : int
    final_damage : float
    buff_indure_time : int
    maple_goddess_inc : int
    weapon_att : int

    def to_list(self):
        return [
            self.main_stat,
            self.sub_stat,
            self.damage,
            self.boss_damage,
            self.att_p,
            self.defense_ignore,
            self.critical_damage,
            self.final_damage,
            self.buff_indure_time,
            self.maple_goddess_inc,
            self.weapon_att
        ]
        
# a = Ability(main_stat = 1, sub_stat = 1, damage = 1, boss_damage = 1, att_p = 1, defense_ignore = 1, critical_damage= 1, final_damage = 1, buff_indure_time= 1,
#             maple_goddess_inc = 1, weapon_att = 1)
