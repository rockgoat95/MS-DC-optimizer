from pydantic import BaseModel


class Ability(BaseModel):
    main_stat: int
    sub_stat: int
    damage: int
    boss_damage: int
    att_p: int
    defense_ignore: float
    critical_damage: int
    final_damage: float
    buff_indure_time: int
    total_att: int
    maple_goddess2_inc: int
    weapon_puff_inc: int

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
            self.total_att,
            self.maple_goddess2_inc,
            self.weapon_puff_inc,
        ]

    def names(self):
        return [
            "main_stat",
            "sub_stat",
            "damage",
            "boss_damage",
            "att_p",
            "defense_ignore",
            "critical_damage",
            "final_damage",
            "buff_indure_time",
            "total_att",
            "maple_goddess2_inc",
            "weapon_puff_inc",
        ]
    
    def add(self, 
            main_stat = 0,
            sub_stat = 0,
            damage = 0,
            boss_damage =0,
            att_p = 0,
            defense_ignore = 0,
            critical_damage = 0,
            final_damage = 0,
            buff_indure_time = 0,
            total_att = 0,
            maple_goddess2_inc = 0,
            weapon_puff_inc = 0
            ):
        
        self.main_stat += main_stat
        self.sub_stat += sub_stat
        self.damage += damage
        self.boss_damage += boss_damage
        self.att_p += att_p
        self.defense_ignore += defense_ignore
        self.critical_damage += critical_damage
        self.final_damage += final_damage
        self.buff_indure_time += buff_indure_time
        self.total_att += total_att
        self.maple_goddess2_inc += maple_goddess2_inc
        self.weapon_puff_inc += weapon_puff_inc
