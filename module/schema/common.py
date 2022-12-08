from pydantic import BaseModel
from module import utils

# 능력치 저장 
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
            self.main_stat, # 주스탯 
            self.sub_stat, # 부스탯 (부스탯 두개인직업은 더해서 적용 )
            self.damage, # 데미지 
            self.boss_damage, # 보뎀
            self.att_p, # 공퍼 
            self.defense_ignore, # 방무 
            self.critical_damage, # 크뎀 
            self.final_damage, # 최종뎀 
            self.buff_indure_time, # 버프지속시간
            self.total_att, # 총 공격력 (스탯계산기 참고)
            self.maple_goddess2_inc, # 메용 2 주스탯 상승량 (이후 계산공식 도입으로 생략 )
            self.weapon_puff_inc, # 웨펖 주스탯 상승량 (이후 계산공식 도입으로 생략 )
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

    def add(
        self,
        main_stat=0,
        sub_stat=0,
        damage=0,
        boss_damage=0,
        att_p=0,
        defense_ignore=0,
        critical_damage=0,
        final_damage=0,
        buff_indure_time=0,
        total_att=0,
        maple_goddess2_inc=0,
        weapon_puff_inc=0,
    ):

        self.main_stat += main_stat
        self.sub_stat += sub_stat
        self.damage += damage
        self.boss_damage += boss_damage
        self.att_p += att_p
        self.defense_ignore = utils.defense_ignore_calculator(
            [self.defense_ignore, defense_ignore]
        ) # 방무 공식 적용 
        self.critical_damage += critical_damage
        self.final_damage = utils.final_damage_applier(self.final_damage, final_damage) # 최종뎀 공식 적용 
        self.buff_indure_time += buff_indure_time
        self.total_att += total_att
        self.maple_goddess2_inc += maple_goddess2_inc
        self.weapon_puff_inc += weapon_puff_inc
